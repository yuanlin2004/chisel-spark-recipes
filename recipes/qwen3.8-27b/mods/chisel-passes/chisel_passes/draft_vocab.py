"""Reduced-vocabulary output head for speculative draft steps.

The Qwen3.8 MTP drafter runs the full 248,320 x 5,120 BF16 output head (about 2.5 GB) once per draft
token, so a 3-token draft reads it three times per decode step on top of the verify step. Drafts only
propose tokens; the target model's full head decides what is kept. So the draft head can cover a
subset of the vocabulary: a draft token outside the subset is never proposed, which can only lower
acceptance, never change the output.

After vLLM's proposer loads (and possibly shares) the draft head, this replaces the draft model's
lm_head with a new head holding only the chosen rows, and sets draft_id_to_target_id so draft token
k maps back to target id k + d2t[k] (the convention vLLM's LocalArgmaxMixin and EAGLE-3 use).
compute_logits of the MTP model is wrapped to scatter subset logits into a full-vocabulary tensor
for code paths that read logits rather than the local argmax.

DFlash2 drafts through the target's head too (shared at load), once per draft block, and picks its
top-k candidates from it before the candidate selector scores them. For DFlash2 compute_candidates is
wrapped so the reduced head's row numbers map back to target ids before the selector (whose
codebooks are indexed by target id) sees them; vLLM 0.31 does not do this mapping (the bug
vllm#60365 fixes).
"""
import os

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.chisel_passes.draft_vocab")  # under "vllm" so vLLM's log handler prints it
_installed = False


def _load_ids(path):
    # A bare name (no "/") is a token list shipped with this package, e.g. "qwen3.8-27b-100k".
    if "/" not in path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", path + ".txt")
    with open(path) as fh:
        ids = sorted({int(x) for x in fh.read().split()})
    return ids


def _head_rows(full_head, vocab):
    """The head as a [vocab, hidden] tensor in its compute dtype: dequantizes an FP8 head (per-row scales), and
    undoes the [hidden, vocab] layout some quantized linear methods store after loading (the unsloth NVFP4
    checkpoint's head is FP8 on the nightly vLLM)."""
    w = full_head.weight
    if w.dim() == 2 and w.shape[0] != vocab and w.shape[1] == vocab:
        w = w.t()
    if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        scale = getattr(full_head, "weight_scale", None)
        out_dtype = torch.bfloat16
        w = w.to(torch.float32)
        if scale is not None:
            s = scale.to(torch.float32).reshape(-1)
            w = w * (s[:, None] if s.numel() == vocab else s)
        return w.to(out_dtype)
    return w


def _reduced_head(full_head, ids, vocab):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    w = _head_rows(full_head, vocab)
    idx = torch.tensor(ids, device=w.device, dtype=torch.long)
    rows = w.index_select(0, idx)
    head = ParallelLMHead(len(ids), rows.shape[1], params_dtype=rows.dtype)
    head = head.to(rows.device)
    with torch.no_grad():
        head.weight[: len(ids)].copy_(rows)
    return head


def _reduced_head_fp8(full_head, ids, vocab):
    """Keep the head's own FP8 rows and per-row scales (compressed-tensors W8A8 channel scheme): a shallow copy of
    the head module with the chosen output rows, so the same quant method and kernel run on fewer rows."""
    import copy
    w = full_head.weight  # [hidden, vocab] after loading: a transposed view of [vocab, hidden] rows
    assert w.dim() == 2 and w.shape[1] == vocab, "unexpected FP8 head layout"
    idx = torch.tensor(ids, device=w.device, dtype=torch.long)
    rows = w.t().index_select(0, idx).contiguous()  # [n, hidden]
    head = copy.copy(full_head)
    head._parameters = dict(full_head._parameters)
    head._buffers = dict(full_head._buffers)
    head._modules = dict(full_head._modules)
    weight = torch.nn.Parameter(rows.t(), requires_grad=False)
    weight.input_dim, weight.output_dim = 0, 1
    head.weight = weight
    s = full_head.weight_scale
    if s.dim() >= 1 and s.shape[0] == vocab:
        head.weight_scale = torch.nn.Parameter(s.index_select(0, idx).contiguous(), requires_grad=False)
    # the FP8 kernel object keeps the layer's logical output size, so the copy needs its own scheme and kernel
    scheme = getattr(full_head, "scheme", None)
    if scheme is not None:
        scheme = copy.copy(scheme)
        kern = getattr(scheme, "fp8_linear", None)
        if kern is not None:
            kern = copy.copy(kern)
            if hasattr(kern, "logical_output_size"):
                kern.logical_output_size = len(ids)
            scheme.fp8_linear = kern
        head.scheme = scheme
    for name in ("num_embeddings", "org_vocab_size", "num_embeddings_padded", "num_embeddings_per_partition",
                 "org_vocab_size_padded"):
        if hasattr(head, name):
            setattr(head, name, len(ids))
    return head


def _fp8_setting(vllm_config):
    extra = getattr(vllm_config, "additional_config", None) or {}
    return bool(extra.get("chisel_draft_vocab_fp8")) if isinstance(extra, dict) else False


def _setting(vllm_config):
    extra = getattr(vllm_config, "additional_config", None) or {}
    return (extra.get("chisel_draft_vocab") if isinstance(extra, dict) else None) or os.environ.get("CHISEL_DRAFT_VOCAB")


def _reduce(model, vllm_config):
    """Swap the draft model's lm_head for the chosen rows; no-op without a setting."""
    path = _setting(vllm_config)
    if not path:
        return
    ids = _load_ids(path)
    full = getattr(model, "lm_head", None)
    if full is None or not hasattr(full, "weight") or getattr(full, "tp_size", 1) != 1:
        logger.warning("chisel draft vocab: no single-GPU draft lm_head found; pass not applied")
        return
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or full.weight.shape[0])
    keep = [i for i in ids if i < vocab]
    if _fp8_setting(vllm_config) and full.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        new_head = _reduced_head_fp8(full, keep, vocab)
    else:
        new_head = _reduced_head(full, keep, vocab)
    del model.lm_head  # drops the reference only; a shared target head stays with the target
    model.lm_head = new_head
    dev = new_head.weight.device
    k = torch.arange(len(keep), device=dev, dtype=torch.long)
    model.draft_id_to_target_id = torch.tensor(keep, device=dev, dtype=torch.long) - k
    model._chisel_full_vocab = vocab
    model.logits_processor.org_vocab_size = len(keep)
    if getattr(model, "candidate_logits_processor", None) is not None:  # DFlash2's top-k path
        model.candidate_logits_processor.org_vocab_size = len(keep)
    cls = type(model)
    if not getattr(cls, "_chisel_wrapped", False):
        inner = cls.compute_logits

        def compute_logits(self_, hidden_states, *args, **kwargs):
            logits = inner(self_, hidden_states, *args, **kwargs)
            d2t = getattr(self_, "draft_id_to_target_id", None)
            if d2t is None or logits is None:
                return logits
            full_logits = logits.new_full((logits.shape[0], self_._chisel_full_vocab), float("-inf"))
            targets = torch.arange(logits.shape[1], device=logits.device) + d2t[: logits.shape[1]]
            full_logits[:, targets] = logits
            return full_logits

        cls.compute_logits = compute_logits
        if hasattr(cls, "compute_candidates"):
            inner_candidates = cls.compute_candidates

            def compute_candidates(self_, hidden_states, *args, **kwargs):
                ids, values = inner_candidates(self_, hidden_states, *args, **kwargs)
                d2t = getattr(self_, "draft_id_to_target_id", None)
                if d2t is None:
                    return ids, values
                return ids + d2t[ids], values

            cls.compute_candidates = compute_candidates
        cls._chisel_wrapped = True
    logger.warning("chisel draft vocab: draft head reduced to %d of %d tokens (%.2f GB to %.2f GB) from %s",
                len(keep), vocab, full.weight.numel() * full.weight.element_size() / 1e9,
                new_head.weight.numel() * new_head.weight.element_size() / 1e9, path)


def install():
    """Hook the drafter load paths in vLLM 0.31: the V1 proposer and the V2 model runner's MTP and DFlash speculators."""
    global _installed
    if _installed:
        return
    _installed = True
    from vllm.v1.spec_decode import llm_base_proposer as lbp

    original = lbp.SpecDecodeBaseProposer._maybe_share_lm_head

    def patched(self, target_language_model):
        original(self, target_language_model)
        _reduce(self.model, self.vllm_config)

    lbp.SpecDecodeBaseProposer._maybe_share_lm_head = patched
    try:
        from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp_spec
    except ImportError:
        return
    original_v2 = mtp_spec.MTPSpeculator.load_draft_model

    def load_draft_model(self, target_model, *args, **kwargs):
        model = original_v2(self, target_model, *args, **kwargs)
        _reduce(model, self.vllm_config)
        return model

    mtp_spec.MTPSpeculator.load_draft_model = load_draft_model
    try:
        from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_spec
    except ImportError:
        return
    original_dflash = dflash_spec.DFlashSpeculator.load_draft_model

    def load_dflash_draft_model(self, target_model, *args, **kwargs):
        model = original_dflash(self, target_model, *args, **kwargs)
        _reduce(model, self.vllm_config)
        return model

    dflash_spec.DFlashSpeculator.load_draft_model = load_dflash_draft_model
