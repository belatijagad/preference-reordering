#!/usr/bin/env bash
set -euo pipefail

# First install PyTorch and the extension build requirements. FlashAttention
# then builds against that exact environment without isolation.
uv sync --no-dev

FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS="80;120" \
MAX_JOBS="${MAX_JOBS:-4}" \
uv sync --extra remote-cuda --no-dev
