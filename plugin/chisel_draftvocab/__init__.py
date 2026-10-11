"""Chisel passes for stock vLLM, applied through vLLM's general-plugin hook.

The hook runs in every vLLM process, and each pass reads its setting at model-load time from
--additional-config (vLLM hands that to the engine process; environment variables set on the
`vllm serve` command did not reach the engine core in vLLM 0.31). With no setting a pass does nothing,
so installing the plugin changes nothing by itself. Passes:

- additional-config {"chisel_draft_vocab": "<name or file of token ids>"} (or CHISEL_DRAFT_VOCAB): the
  speculative drafter's output head covers only these tokens (draft_vocab.py). The verify step keeps
  the full head, so outputs do not change; only the draft cost and possibly acceptance do.
"""


def register():
    import os
    from vllm.logger import init_logger
    from .draft_vocab import install
    install()
    init_logger("vllm.chisel_draftvocab").warning("chisel_draftvocab plugin loaded (pid %d)", os.getpid())
