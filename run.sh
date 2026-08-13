#!/usr/bin/env bash
set -euo pipefail

benchmark="${BENCHMARK:-custom}"
author_key="${AUTHOR_KEY:-0}"
output_dir="${OUTPUT_DIR:-outputs/${benchmark}-mistral-7b-instruct-ditto}"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing output directory: ${output_dir}" >&2
    echo "Set OUTPUT_DIR to a new path or move the existing directory." >&2
    exit 1
fi

ACCELERATE_LOG_LEVEL=info uv run accelerate launch \
    --config_file configs/single_gpu.yaml \
    scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
    --train_pkl="benchmarks/${benchmark}/processed/${benchmark}_train.pkl" \
    --train_author_key="${author_key}" \
    --output_dir="${output_dir}"

uv run python generate.py \
    --benchmark="${benchmark}" \
    --train_author_key="${author_key}" \
    --adapter-path="${output_dir}/ditto"
