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
"""
import os

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.chisel_draftvocab.draft_vocab")  # under "vllm" so vLLM's log handler prints it
_installed = False


def _load_ids(path):
    # A bare name (no "/") is a token list shipped with this package, e.g. "qwen3.8-27b".
    if "/" not in path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", path + ".txt")
    with open(path) as fh:
        ids = sorted({int(x) for x in fh.read().split()})
    return ids


def _reduced_head(full_head, ids):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    w = full_head.weight
    head = ParallelLMHead(len(ids), w.shape[1], params_dtype=w.dtype)
    head = head.to(w.device)
    idx = torch.tensor(ids, device=w.device, dtype=torch.long)
    with torch.no_grad():
        head.weight[: len(ids)].copy_(w.index_select(0, idx))
    return head


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
    vocab = full.weight.shape[0]
    keep = [i for i in ids if i < vocab]
    new_head = _reduced_head(full, keep)
    del model.lm_head  # drops the reference only; a shared target head stays with the target
    model.lm_head = new_head
    dev = new_head.weight.device
    k = torch.arange(len(keep), device=dev, dtype=torch.long)
    model.draft_id_to_target_id = torch.tensor(keep, device=dev, dtype=torch.long) - k
    model._chisel_full_vocab = vocab
    model.logits_processor.org_vocab_size = len(keep)
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
        cls._chisel_wrapped = True
    logger.warning("chisel draft vocab: draft head reduced to %d of %d tokens (%.2f GB to %.2f GB) from %s",
                len(keep), vocab, full.weight.numel() * full.weight.element_size() / 1e9,
                new_head.weight.numel() * new_head.weight.element_size() / 1e9, path)


def install():
    """Hook both drafter load paths in vLLM 0.31: the V1 proposer and the V2 model runner's MTP speculator."""
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
