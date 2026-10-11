# Environment

- Machine: NVIDIA DGX Spark (GB10, compute capability 12.1, 128 GB unified memory), driver 580.178.04, CUDA 13.0 driver API.
- benchmark/: GPU clock capped at 2,000 MHz (`nvidia-smi -lgc 0,2000`), held at 1,982 to 1,989 MHz under load (gpu.csv, gpu-check.json). Decode median 43.48 to 43.49 ms.
- benchmark-uncapped/: default clocks (`nvidia-smi -rgc`), 2,333 to 2,411 MHz under load, up to 90 W and 79 °C (gpu.csv, gpu-check.json). Decode median 43.61 to 43.67 ms, p90 58.0 ms, time to first token 312 to 327 ms.
- Container: `vllm/vllm-openai:v0.31.0`, digest in image-digest.json; vLLM 0.31.0 on its V2 model runner.
- Plugin: chisel-draftvocab 0.1.0 from this repository's `plugin/`, installed at container start (measured under its earlier package name chisel-vllm, which the logs show; the code is unchanged); server log confirms "chisel draft vocab: draft head reduced to 50858 of 248320 tokens (2.54 GB to 0.52 GB)" (server.log).
- Target weights: Qwen/Qwen3.8-27B-FP8, revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a. MTP head from the same checkpoint.
- Serving flags: exactly `recipes/qwen3.8-27b/fp8-mtp4-draftvocab.docker.sh` (same flags as the sparkrun recipe).
- Quality reference: Qwen/Qwen3.8-27B (BF16), revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, served with BF16 KV cache.
