"""Chisel's passes for speculative decoding in vLLM, loaded as a vLLM general plugin. Each pass is off unless the
recipe turns it on through --additional-config, so installing the plugin changes nothing by itself.

- {"chisel_gdn_replay": true or an int}: the linear-attention (GDN) layers of Qwen3.5/3.8 skip storing the full
  recurrent state after every draft token during verification; the accepted state is replayed and stored once.
  Bit-identical to vLLM's own state, so outputs do not change.
- {"chisel_draft_vocab": "<name or file of token ids>"}: the drafter's output head covers only these tokens, mapped
  back to the full vocabulary. The verify step keeps the full head. {"chisel_draft_vocab_fp8": true} keeps the
  reduced head in FP8 when the drafter's head is FP8.
"""


def register():
    import os
    from vllm.logger import init_logger
    from .draft_vocab import install as install_draft_vocab
    from .gdn_replay import install as install_gdn_replay
    install_draft_vocab()
    install_gdn_replay()
    init_logger("vllm.chisel_passes").warning("chisel_passes plugin loaded (pid %d)", os.getpid())
