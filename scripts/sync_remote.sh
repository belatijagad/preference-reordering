#!/usr/bin/env bash
set -euo pipefail

# First install PyTorch, then add the build tools that FlashAttention imports
# when uv builds it without isolation.
uv sync --no-dev
uv pip install "setuptools>=68" "wheel>=0.41"

FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS="80;120" \
MAX_JOBS="${MAX_JOBS:-4}" \
uv sync --extra remote-cuda --no-dev

# Do not leave a successful install unverified on the target GPU.
uv run --extra remote-cuda --no-sync python scripts/verify_flash_attention.py
