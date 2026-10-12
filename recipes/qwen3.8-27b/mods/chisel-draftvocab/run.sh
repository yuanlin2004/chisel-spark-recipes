#!/usr/bin/env bash
# Installs Chisel's reduced-vocabulary draft head plugin (plugin/ in this repository)
# into the vLLM container before the serve command. vLLM loads it through its
# general-plugins entry point; the recipe turns it on with
# --additional-config '{"chisel_draft_vocab": "qwen3.8-27b"}'.
set -euo pipefail
WHEEL=https://github.com/yuanlin2004/chisel-spark-recipes/releases/download/qwen3.8-27b-v1/chisel_draftvocab-0.1.0-py3-none-any.whl
pip install --no-deps "$WHEEL"
