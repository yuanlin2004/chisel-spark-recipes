# Environment

- Machine: NVIDIA DGX Spark (GB10, compute capability 12.1, 128 GB unified memory), driver 580.178.04, GPU clock capped at 2,000 MHz (`nvidia-smi -lgc 0,2000`).
- Container: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly@sha256:b77010699ec42ab88b3409f4dd101f2288dfd8895e2778e5bb737e03a20953f0` (vLLM 0.31.1rc1.dev380), the image the recipe pins. The benchmark and quality runs used a local build of that image with the same plugin code preinstalled; the recipe installs it at start from `recipes/qwen3.8-27b/mods/chisel-passes/` (the sparkrun check files were measured that way).
- Plugin: chisel-passes 0.1.0 (this repository, `recipes/qwen3.8-27b/mods/chisel-passes/`): GDN state replay with a capacity of 8 requests, and the drafter's output head reduced to 100,032 tokens (list `qwen3.8-27b-100k`), kept in FP8.
- Target weights: unsloth/Qwen3.8-27B-NVFP4, revision f0b7c9e7.
- Drafter: yuanlin2004/Qwen3.8-27B-Chisel-DFlash2-NVFP4-drafter (model.safetensors sha256 8d464c9f597c41a2e8917c585454239be14f254592f114675fd9d5d5dbc92f4c).
- Serving flags: server.cmd (local paths in place of the Hugging Face names); the same flags as `recipes/qwen3.8-27b/chisel-nvfp4-dflash2-replay.yaml`.
- Quality reference: Qwen/Qwen3.8-27B (BF16), revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, served with BF16 KV cache.
