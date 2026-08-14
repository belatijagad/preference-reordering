#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_root}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

config_file="${1:-configs/experiment.yaml}"
uv run --extra remote-cuda python -m scripts.run_experiment "${config_file}"
