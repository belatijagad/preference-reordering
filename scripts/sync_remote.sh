#!/usr/bin/env bash
set -euo pipefail

# The remote extra uses the pinned CUDA 12.8 / PyTorch 2.7 / Python 3.10
# prebuilt FlashAttention wheel declared in pyproject.toml.
uv sync --extra remote-cuda --no-dev

# Do not leave a successful install unverified on the target GPU.
uv run --extra remote-cuda --no-sync python scripts/verify_flash_attention.py
