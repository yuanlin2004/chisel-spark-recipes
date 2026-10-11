#!/usr/bin/env bash
# The same recipe without sparkrun: plain docker on one DGX Spark.
# Usage: [MODEL=<hf id or local dir>] [EXTRA_ARGS='<more vllm serve flags>'] bash recipes/qwen3.8-27b/fp8-mtp4-draftvocab.docker.sh [path to a local checkout of this repo]
# With a checkout path the plugin is installed from it; otherwise the release wheel below (the vLLM image has no git).
set -euo pipefail
IMAGE=vllm/vllm-openai:v0.31.0
REPO=${1:-}
PLUGIN=https://github.com/yuanlin2004/chisel-spark-recipes/releases/download/qwen3.8-27b-v1/chisel_draftvocab-0.1.0-py3-none-any.whl
MOUNT=()
if [[ -n "$REPO" ]]; then PLUGIN=/tmp/chisel-plugin; MOUNT=(-v "$(cd "$REPO" && pwd):/chisel:ro"); fi
# MODEL may be a Hugging Face id (downloaded into HF_HOME) or a local directory, mounted at the same path.
MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}
if [[ -d "$MODEL" ]]; then MOUNT+=(-v "$MODEL:$MODEL:ro"); fi
exec docker run --rm --name chisel-qwen38-27b --gpus all --ipc=host --network=host \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" "${MOUNT[@]}" \
  --entrypoint bash "$IMAGE" -c "
    { [ ! -d /chisel/plugin ] || cp -r /chisel/plugin /tmp/chisel-plugin; } && \
    pip install --no-deps '$PLUGIN' && \
    vllm serve $MODEL --served-model-name ${SERVED_NAME:-qwen3.8-27b} \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 131072 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.8 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
      --kv-cache-dtype fp8 --attention-backend flashinfer \
      --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 4}' \
      --additional-config '{\"chisel_draft_vocab\": \"qwen3.8-27b\"}' \
      --enable-prefix-caching ${EXTRA_ARGS:-}"
