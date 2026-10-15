"""Gated delta rule for speculative decoding without per-token state writes.

verify_no_store(): same arithmetic as vLLM's fused_sigmoid_gating_delta_rule_update_kernel on the spec-decode path,
but instead of storing the full recurrent state after every token it saves the per-token recurrence inputs
(L2-normalized k, v, decay g, beta) to a small buffer.
materialize(): after the rejection sampler has fixed num_accepted, replays tokens 0..num_accepted-1 from the
saved inputs, starting from the same stored start state, and stores the state once into the slot the existing
code reads next (state_indices[n, num_accepted - 1]). The fp32 operations and their order match the original
kernel, so the stored state is bit-identical to what the original would have written there.

This file holds the kernels and their bit-exactness test (run test() on a GPU). Wiring into the model runner is a
separate step.
"""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _verify_kernel(A_log, a, b, dt_bias, q, k, v, o, h0, cu_seqlens, state_indices, num_accepted_tokens,
                   save_k, save_v, save_g, save_beta, save_idx, save_prev, scale, beta_sp, threshold,
                   H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                   BV: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, stride_state: tl.constexpr,
                   stride_idx: tl.constexpr, USE_QK_L2NORM: tl.constexpr, STORE_PER_TOKEN: tl.constexpr = False):
    i_v, i_nh = tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    if i_v == 0 and i_hv == 0:  # keep the indices and start position this call used, for materialize
        o_s = tl.arange(0, BS)
        tl.store(save_idx + i_n * S + o_s, tl.load(state_indices + i_n * stride_idx + o_s, mask=o_s < S, other=0),
                 mask=o_s < S)
        tl.store(save_prev + i_n, tl.load(num_accepted_tokens + i_n))
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if T == 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    i_t0 = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    state_idx = tl.load(state_indices + i_n * stride_idx + i_t0).to(tl.int64)
    if state_idx <= 0:
        return
    p_h0 = h0 + state_idx * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    for i_t in range(0, T):
        t = bos + i_t
        b_q = tl.load(q + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(k + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(v + (t * HV + i_hv) * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(b + t * HV + i_hv).to(tl.float32)
        x = tl.load(a + t * HV + i_hv).to(tl.float32) + tl.load(dt_bias + i_hv).to(tl.float32)
        softplus_x = tl.where(beta_sp * x <= threshold, (1 / beta_sp) * tl.log(1 + tl.exp(beta_sp * x)), x)
        b_g = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
        b_beta = tl.sigmoid(b_b)
        if USE_QK_L2NORM:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # save the recurrence inputs (k once per key head, v per value-head block)
        slot = i_n * S + i_t
        if i_v == 0:
            tl.store(save_k + (slot * HV + i_hv) * K + o_k, b_k, mask=mask_k)
            tl.store(save_g + slot * HV + i_hv, b_g)
            tl.store(save_beta + slot * HV + i_hv, b_beta)
        tl.store(save_v + (slot * HV + i_hv) * V + o_v, b_v, mask=mask_v)
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(o + (t * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
        if STORE_PER_TOKEN:  # debug: behave like vLLM's kernel as well
            d_idx = tl.load(state_indices + i_n * stride_idx + i_t).to(tl.int64)
            if d_idx > 0:
                tl.store(h0 + d_idx * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :],
                         b_h.to(h0.dtype.element_ty), mask=mask_h)


@triton.jit
def _materialize_kernel(h0, state_indices, prev_accepted, new_accepted, save_k, save_v, save_g, save_beta,
                        HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                        S: tl.constexpr, stride_state: tl.constexpr, stride_idx: tl.constexpr):
    i_v, i_nh = tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    n_acc = tl.load(new_accepted + i_n).to(tl.int64)
    if n_acc <= 0:
        return
    src = tl.load(state_indices + i_n * stride_idx + tl.load(prev_accepted + i_n).to(tl.int64) - 1).to(tl.int64)
    dst = tl.load(state_indices + i_n * stride_idx + n_acc - 1).to(tl.int64)
    if src <= 0 or dst <= 0:
        return
    off = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(h0 + src * stride_state + off, mask=mask_h, other=0).to(tl.float32)
    for i_t in range(0, n_acc):
        slot = i_n * S + i_t
        b_k = tl.load(save_k + (slot * HV + i_hv) * K + o_k, mask=mask_k, other=0)
        b_v = tl.load(save_v + (slot * HV + i_hv) * V + o_v, mask=mask_v, other=0)
        b_g = tl.load(save_g + slot * HV + i_hv)
        b_beta = tl.load(save_beta + slot * HV + i_hv)
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
    tl.store(h0 + dst * stride_state + off, b_h.to(h0.dtype.element_ty), mask=mask_h)


@triton.jit
def _materialize_layers_kernel(h0, slots, idx, prev, new_accepted, save_k, save_v, save_g, save_beta,
                               HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                               BV: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, stride_state: tl.constexpr):
    """_materialize_kernel for many layers in one launch: program_id(0) picks the layer's slot in a ReplayStack."""
    l = tl.load(slots + tl.program_id(0)).to(tl.int64)
    i_v, i_nh = tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    n_acc = tl.load(new_accepted + i_n).to(tl.int64)
    if n_acc <= 0:
        return
    row = idx + (l * CAP + i_n) * S
    src = tl.load(row + tl.load(prev + l * CAP + i_n).to(tl.int64) - 1).to(tl.int64)
    dst = tl.load(row + n_acc - 1).to(tl.int64)
    if src <= 0 or dst <= 0:
        return
    off = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(h0 + src * stride_state + off, mask=mask_h, other=0).to(tl.float32)
    base = l * CAP * S + i_n * S
    for i_t in range(0, n_acc):
        slot = base + i_t
        b_k = tl.load(save_k + (slot * HV + i_hv) * K + o_k, mask=mask_k, other=0)
        b_v = tl.load(save_v + (slot * HV + i_hv) * V + o_v, mask=mask_v, other=0)
        b_g = tl.load(save_g + slot * HV + i_hv)
        b_beta = tl.load(save_beta + slot * HV + i_hv)
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
    tl.store(h0 + dst * stride_state + off, b_h.to(h0.dtype.element_ty), mask=mask_h)


class ReplayStack:
    """Save buffers for up to L layers in single tensors, so one launch can materialize every layer."""

    def __init__(self, L, max_reqs, S, HV, K, V, device):
        self.L, self.cap, self.S, self.HV, self.K, self.V = L, max_reqs, S, HV, K, V
        self.k = torch.empty(L, max_reqs * S, HV, K, device=device, dtype=torch.float32)
        self.v = torch.empty(L, max_reqs * S, HV, V, device=device, dtype=torch.float32)
        self.g = torch.empty(L, max_reqs * S, HV, device=device, dtype=torch.float32)
        self.beta = torch.empty(L, max_reqs * S, HV, device=device, dtype=torch.float32)
        self.idx = torch.zeros(L, max_reqs, S, device=device, dtype=torch.int32)
        self.prev = torch.ones(L, max_reqs, device=device, dtype=torch.int32)
        self.used = 0

    def take(self):
        assert self.used < self.L, "more GDN layers than the replay stack holds"
        b = ReplayBuffers.__new__(ReplayBuffers)
        i = self.used
        self.used += 1
        b.S, b.slot, b.start = self.S, i, None
        b.k, b.v, b.g, b.beta, b.idx, b.prev = (self.k[i], self.v[i], self.g[i], self.beta[i], self.idx[i],
                                                 self.prev[i])
        return b


def materialize_layers(state, stack, slots, n, new_accepted, BV=32, num_warps=4):
    """Materialize every layer in `slots` (int32 tensor of stack slots, all on this state tensor) for the first n
    requests, in one launch."""
    HV, V, K = state.shape[1], state.shape[2], state.shape[3]
    _materialize_layers_kernel[(slots.numel(), triton.cdiv(V, BV), n * HV)](
        state, slots, stack.idx, stack.prev, new_accepted, stack.k, stack.v, stack.g, stack.beta,
        HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BV, S=stack.S, CAP=stack.cap,
        stride_state=state.stride(0), num_warps=num_warps, num_stages=3)


class ReplayBuffers:
    """Per-layer save buffers for up to max_reqs requests of S tokens (fp32; about 0.8 MB per request per layer)."""

    def __init__(self, max_reqs, S, HV, K, V, device):
        self.S = S
        self.k = torch.empty(max_reqs * S, HV, K, device=device, dtype=torch.float32)
        self.v = torch.empty(max_reqs * S, HV, V, device=device, dtype=torch.float32)
        self.g = torch.empty(max_reqs * S, HV, device=device, dtype=torch.float32)
        self.beta = torch.empty(max_reqs * S, HV, device=device, dtype=torch.float32)
        # the state indices and start positions the verify call used (copied in the same stream, so CUDA graph
        # replays refresh them too)
        self.idx = torch.zeros(max_reqs, S, device=device, dtype=torch.int32)
        self.prev = torch.ones(max_reqs, device=device, dtype=torch.int32)
        self.start = None  # debug "check": start states saved before the verify call


def verify_no_store(A_log, a, b, dt_bias, q, k, v, state, cu_seqlens, state_indices, num_accepted_tokens, bufs,
                    scale=None, beta=1.0, threshold=20.0, use_qk_l2norm=True, store_per_token=False, BV=32,
                    num_warps=4):
    """q, k: [T, H, K]; v: [T, HV, V]; a, b: [T, HV]; state: [slots, HV, V, K]. Returns o: [T, HV, V]."""
    T, H, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    N = cu_seqlens.numel() - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), BV)
    o = torch.empty_like(v)
    scale = K ** -0.5 if scale is None else scale
    _verify_kernel[(1, triton.cdiv(V, BV), N * HV)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, q.contiguous(), k.contiguous(), v.contiguous(), o, state,
        cu_seqlens, state_indices, num_accepted_tokens, bufs.k, bufs.v, bufs.g, bufs.beta, bufs.idx, bufs.prev, scale,
        beta, threshold, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, S=bufs.S, BS=triton.next_power_of_2(bufs.S),
        stride_state=state.stride(0),
        stride_idx=state_indices.stride(0), USE_QK_L2NORM=use_qk_l2norm, STORE_PER_TOKEN=store_per_token,
        num_warps=num_warps, num_stages=3)
    return o


def materialize(state, state_indices, prev_accepted, new_accepted, bufs):
    N = new_accepted.numel()
    HV, V, K = state.shape[1], state.shape[2], state.shape[3]
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    _materialize_kernel[(1, triton.cdiv(V, BV), N * HV)](
        state, state_indices, prev_accepted, new_accepted, bufs.k, bufs.v, bufs.g, bufs.beta,
        HV=HV, K=K, V=V, BK=BK, BV=BV, S=bufs.S, stride_state=state.stride(0), stride_idx=state_indices.stride(0),
        num_warps=4, num_stages=3)


def test(N=2, T=11, H=16, HV=48, K=128, V=128, state_dtype=torch.bfloat16, seed=0):
    """Bit-exactness against vLLM's kernel: for every possible accepted count, the materialized state must equal the
    state vLLM's kernel stores at that position, and the outputs must match."""
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(N * T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(N * T, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(N * T, HV, V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=dev, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(HV, device=dev, dtype=torch.float32) * 0.5
    slots = 1 + N * T
    base = (torch.randn(slots, HV, V, K, device=dev) * 0.1).to(state_dtype)
    idx = torch.arange(1, slots, device=dev, dtype=torch.int32).view(N, T)
    prev = (torch.arange(N, device=dev, dtype=torch.int32) % T) + 1  # start state at column prev-1
    cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
    ref_state = base.clone()
    o_ref, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a.unsqueeze(0), b=b.unsqueeze(0), dt_bias=dt_bias, q=q.unsqueeze(0), k=k.unsqueeze(0),
        v=v.unsqueeze(0), initial_state=ref_state, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=idx, num_accepted_tokens=prev, use_qk_l2norm_in_kernel=True)
    bufs = ReplayBuffers(N, T, HV, K, V, dev)
    new_state = base.clone()
    o_new = verify_no_store(A_log, a, b, dt_bias, q, k, v, new_state, cu, idx, prev, bufs)
    ok_out = torch.equal(o_ref.squeeze(0), o_new)
    worst = 0
    for acc in range(1, T + 1):
        st = new_state.clone()
        materialize(st, idx, prev, torch.full((N,), acc, device=dev, dtype=torch.int32), bufs)
        for n in range(N):
            d = idx[n, acc - 1].item()
            if not torch.equal(st[d], ref_state[d]):
                worst = max(worst, (st[d].float() - ref_state[d].float()).abs().max().item() or 1e-30)
    # the same through a ReplayStack slot and the all-layer kernel
    stack = ReplayStack(3, N, T, HV, K, V, dev)
    stack.take()
    sb = stack.take()
    st2 = base.clone()
    o2 = verify_no_store(A_log, a, b, dt_bias, q, k, v, st2, cu, idx, prev, sb)
    ok_out = ok_out and torch.equal(o2, o_new)
    slots = torch.tensor([sb.slot], device=dev, dtype=torch.int32)
    for acc in range(1, T + 1):
        st = st2.clone()
        materialize_layers(st, stack, slots, N, torch.full((N,), acc, device=dev, dtype=torch.int32))
        for n in range(N):
            d = idx[n, acc - 1].item()
            if not torch.equal(st[d], ref_state[d]):
                worst = max(worst, (st[d].float() - ref_state[d].float()).abs().max().item() or 1e-30)
    print(f"outputs identical: {ok_out}; materialized states identical for all accepted counts: {worst == 0}"
          + ("" if worst == 0 else f" (max abs diff {worst})"))
    return ok_out and worst == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if test() and test(state_dtype=torch.float32) else 1)


# --------------------------------------------------------------------------------------------------------------
# Wiring (vLLM nightly, model runner v2, Qwen3.5/3.8 GDN layers, Triton spec-decode path).
# additional-config {"chisel_gdn_replay": true} turns it on. The spec-decode call of the GDN layers runs
# verify_no_store() (no per-token state stores; per-token inputs saved per layer), and after sampling, before
# vLLM's own mamba postprocess, materialize() writes each spec request's accepted state into column
# num_accepted - 1 of its spec state indices, where the next step and the align-mode copies read it.

_state = {"enabled": None, "cap": 16, "debug": "", "bufs": {}, "layers": {}, "step": None, "step_n": 0,
          "n_layers": 64, "stack": None, "groups": None}


def _enabled(vllm_config):
    """True/int from additional-config chisel_gdn_replay; an int sets the request capacity of the save buffers
    (default 16). Batches with more spec requests than that keep vLLM's own kernel."""
    extra = getattr(vllm_config, "additional_config", None) or {}
    val = extra.get("chisel_gdn_replay") if isinstance(extra, dict) else None
    if not val:
        return False, 16
    return True, (int(val) if not isinstance(val, bool) else 16)


def _check(st, n, new, log):
    """Debug: the verify stored every per-token state (vLLM's behaviour); replay each layer's accepted state from
    the saved start state into a scratch tensor and compare with what the verify stored at column new - 1."""
    bad, worst = 0, 0.0
    ar = torch.arange(1, n + 1, device=new.device, dtype=torch.int32)
    for state, bufs in st["layers"].values():
        scratch = torch.cat([bufs.start[:1], bufs.start[:n]])  # slot 0 is the null slot
        idx2 = ar.unsqueeze(1).expand(n, bufs.S).contiguous()
        materialize(scratch, idx2, torch.ones(n, device=new.device, dtype=torch.int32), new, bufs)
        dst = torch.gather(bufs.idx[:n], 1, (new.long() - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        ref = state.index_select(0, dst.long())
        d = (scratch[1:].float() - ref.float()).abs().amax(dim=(1, 2, 3))
        bad += int((d > 0).sum())
        worst = max(worst, float(d.max()))
    st["check_steps"] = st.get("check_steps", 0) + 1
    st["check_bad"] = st.get("check_bad", 0) + (bad > 0)
    if bad and st.get("check_logged", 0) < 30:
        st["check_logged"] = st.get("check_logged", 0) + 1
        log.warning("chisel gdn check: step %d n=%d new=%s prev=%s mismatching (layer, req) %d, max diff %g",
                    st["step_n"], n, new.tolist(), next(iter(st["layers"].values()))[1].prev[:n].tolist(), bad, worst)
    if st["check_steps"] % 500 == 0:
        log.warning("chisel gdn check: %d steps, %d with mismatches", st["check_steps"], st["check_bad"])


def install():
    from vllm.logger import init_logger
    log = init_logger("vllm.chisel_gdnreplay")
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_mod
        from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh
    except Exception:
        return
    if getattr(gdn_mod, "_chisel_replay_wrapped", False):
        return
    original = gdn_mod.fused_sigmoid_gating_delta_rule_update

    def wrapped(*args, **kwargs):
        if (not _state["enabled"] or kwargs.get("num_accepted_tokens") is None or args
                or kwargs["ssm_state_indices"].shape[0] > _state["cap"]):
            return original(*args, **kwargs)
        q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
        state = kwargs["initial_state"]
        idx = kwargs["ssm_state_indices"]
        cu = kwargs["cu_seqlens"]
        # Layer groups share one state tensor but use their own block indices, so key by the layer's own weights.
        key = kwargs["A_log"].data_ptr()
        bufs = _state["bufs"].get(key)
        S = idx.shape[1]
        HV, V, K = state.shape[1], state.shape[2], state.shape[3]
        if bufs is None:  # one fixed slot per layer in one stack, allocated once (CUDA graphs keep its address)
            if _state.get("stack") is None:
                _state["stack"] = ReplayStack(_state["n_layers"], _state["cap"], S, HV, K, V, state.device)
            bufs = _state["stack"].take()
            _state["bufs"][key] = bufs
        assert bufs.S == S, "spec-decode width changed"
        if _state["debug"] == "check":
            if bufs.start is None:
                bufs.start = torch.zeros(_state["cap"], HV, V, K, device=state.device, dtype=state.dtype)
            col = (kwargs["num_accepted_tokens"][: idx.shape[0]].long() - 1).clamp(min=0).unsqueeze(1)
            src = torch.gather(idx, 1, col).squeeze(1).long()
            torch.index_select(state, 0, src, out=bufs.start[: idx.shape[0]])
        o = verify_no_store(kwargs["A_log"], kwargs["a"], kwargs["b"], kwargs["dt_bias"], q.squeeze(0), k.squeeze(0),
                            v.squeeze(0), state, cu, idx, kwargs["num_accepted_tokens"], bufs,
                            scale=kwargs.get("scale"), use_qk_l2norm=kwargs.get("use_qk_l2norm_in_kernel", True),
                            store_per_token=_state["debug"] in ("store", "both", "check"),
                            BV=64)  # bit-identical to BV=32 at 4 warps (exact-replay-configs.log), faster
        _state["layers"][key] = (state, bufs)
        _state.setdefault("first_key", key)
        if _state["debug"] == "trace" and _state.get("trace_n", 0) < 40 and key == _state["first_key"]:
            _state["trace_n"] = _state.get("trace_n", 0) + 1
            log.warning("chisel gdn trace verify: N=%d acc=%s idx=%s cu=%s", idx.shape[0],
                        kwargs["num_accepted_tokens"][:4].tolist(), idx[:2, :4].tolist(), cu[:5].tolist())
        return o.unsqueeze(0), state

    gdn_mod.fused_sigmoid_gating_delta_rule_update = wrapped
    gdn_mod._chisel_replay_wrapped = True

    cls = mh.MambaHybridModelState
    orig_init, orig_prepare, orig_post = cls.__init__, cls.prepare_attn, cls.postprocess_state

    def __init__(self, vllm_config, *a, **kw):
        orig_init(self, vllm_config, *a, **kw)
        if _state["enabled"] is None:
            _state["enabled"], _state["cap"] = _enabled(vllm_config)
            extra = getattr(vllm_config, "additional_config", None) or {}
            _state["debug"] = str(extra.get("chisel_gdn_replay_debug", "")) if isinstance(extra, dict) else ""
            try:
                types = getattr(vllm_config.model_config.hf_text_config, "layer_types", None) or []
                _state["n_layers"] = sum(t == "linear_attention" for t in types) or 64
            except Exception:
                _state["n_layers"] = 64
            if _state["enabled"]:
                log.warning("chisel gdn replay: spec-decode GDN states are replayed, not stored per token")

    def prepare_attn(self, *a, **kw):
        md = orig_prepare(self, *a, **kw)
        if _state["enabled"]:
            step = None
            for m in (md or {}).values():
                idx = getattr(m, "spec_state_indices_tensor", None)
                if idx is not None and getattr(m, "num_spec_decodes", 0):
                    # same rule as the wrapper: only batches that ran the replay kernel
                    if idx.shape[0] <= _state["cap"]:
                        step = (m.num_spec_decodes, m.spec_sequence_masks, m.num_accepted_tokens, idx)
                    break
            _state["step"] = None if kw.get("for_capture") else step
            _state["step_n"] += 1
        return md

    def postprocess_state(self, idx_mapping, num_sampled, *a, **kw):
        step = _state["step"]
        _state["step"] = None
        if (_state["enabled"] and _state["debug"] != "store" and step is not None and not isinstance(num_sampled, int)
                and _state.get("layers")):
            n_spec, mask, prev, step_idx = step
            if mask is not None:
                sel = torch.nonzero(mask[: num_sampled.shape[0]], as_tuple=False).flatten()
                new = num_sampled.index_select(0, sel)[:n_spec].to(torch.int32)
            else:
                new = num_sampled[:n_spec].to(torch.int32)
            # step_idx, prev: this step's metadata buffers (all GDN layers share them; the next prepare_attn
            # overwrites them only after this runs)
            if _state["debug"] == "trace" and _state.get("trace_p", 0) < 40:
                _state["trace_p"] = _state.get("trace_p", 0) + 1
                log.warning("chisel gdn trace post: n_spec=%d prev=%s new=%s idx=%s sampled=%s mask=%s", n_spec,
                            prev[:4].tolist(), new[:4].tolist(), step_idx[:2, :4].tolist(), num_sampled[:4].tolist(),
                            None if mask is None else mask[:4].tolist())
            if _state["debug"] == "check":
                _check(_state, n_spec, new, log)
            else:
                groups = _state.get("groups")
                if groups is None or _state.get("groups_n") != len(_state["layers"]):
                    by = {}
                    for state, bufs in _state["layers"].values():
                        by.setdefault(state.data_ptr(), (state, []))[1].append(bufs.slot)
                    groups = [(st, torch.tensor(sl, device=st.device, dtype=torch.int32)) for st, sl in by.values()]
                    _state["groups"], _state["groups_n"] = groups, len(_state["layers"])
                for state, slots in groups:
                    materialize_layers(state, _state["stack"], slots, new.numel(), new)
        return orig_post(self, idx_mapping, num_sampled, *a, **kw)

    cls.__init__ = __init__
    cls.prepare_attn = prepare_attn
    cls.postprocess_state = postprocess_state

    # The memory-profiling run calls the GDN layers with temporary state tensors; forget those once the real KV
    # cache is bound, or materialize() would index a freed, smaller tensor with real block ids.
    try:
        from vllm.v1.worker.gpu import model_runner as mr
        orig_init_kv = mr.GPUModelRunner.initialize_kv_cache

        def initialize_kv_cache(self, *a, **kw):
            out = orig_init_kv(self, *a, **kw)
            _state["layers"].clear()
            _state["bufs"].clear()
            _state["stack"], _state["groups"] = None, None
            _state.pop("first_key", None)
            _state["step"] = None
            return out

        mr.GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    except Exception:
        pass
