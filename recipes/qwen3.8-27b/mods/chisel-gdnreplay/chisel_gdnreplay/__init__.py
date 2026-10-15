"""Chisel's GDN state replay for speculative decoding in vLLM, loaded as a vLLM general plugin.

With additional-config {"chisel_gdn_replay": true} (or an int, the request capacity of the save buffers), the
linear-attention (GDN) layers of Qwen3.5/3.8 skip storing the full recurrent state after every draft token during
verification. They save the small per-token inputs instead, and once the number of accepted tokens is known the
accepted state is replayed and stored once. The stored state is bit-identical to vLLM's own, so outputs do not
change. Without the setting the plugin does nothing.
"""


def register():
    import os
    from vllm.logger import init_logger
    from .gdn_replay import install
    install()
    init_logger("vllm.chisel_gdnreplay").warning("chisel_gdnreplay plugin loaded (pid %d)", os.getpid())
