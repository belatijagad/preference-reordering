#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

config_file="${1:-configs/experiment.yaml}"
uv run --extra remote-cuda python -m scripts.run_experiment "${config_file}"
