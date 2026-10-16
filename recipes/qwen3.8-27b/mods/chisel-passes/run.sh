#!/usr/bin/env bash
# Installs Chisel's passes plugin (this folder) into the vLLM container before the serve command.
# The recipe turns it on with --additional-config '{"chisel_gdn_replay": 8}'.
set -euo pipefail
cd "$(dirname "$0")"
pip install --no-deps --no-build-isolation .
