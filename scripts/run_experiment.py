"""Run training and resumable generation from one experiment YAML."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.artifacts import ArtifactLayout, experiment_bundle_sha256


METHODS = {"zero_shot", "few_shot", "sft", "ditto"}
SPLITS = {"validation", "test"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return config


def require(config: dict[str, Any], name: str) -> Any:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"Missing required experiment setting: {name}")
    return value


def validate_choices(values: Any, allowed: set[str], name: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty YAML list.")
    choices = [str(value) for value in values]
    if len(choices) != len(set(choices)):
        raise ValueError(f"{name} must not contain duplicates.")
    invalid = set(choices) - allowed
    if invalid:
        raise ValueError(f"Unsupported {name}: {', '.join(sorted(invalid))}")
    return choices


def completed_run(run_file: Path) -> bool:
    if not run_file.is_file():
        return False
    with run_file.open(encoding="utf-8") as input_file:
        run = json.load(input_file)
    return isinstance(run, dict) and run.get("status") == "complete"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: run_experiment.py <experiment.yaml>")

    experiment_path = Path(sys.argv[1]).resolve()
    config = load_config(experiment_path)
    configured_model_path = Path(str(require(config, "model_config")))
    model_path = (
        configured_model_path
        if configured_model_path.is_absolute()
        else experiment_path.parent / configured_model_path
    ).resolve()
    model_config = load_config(model_path)
    source_config_sha256 = experiment_bundle_sha256(experiment_path, model_path)
    layout = ArtifactLayout.for_experiment(
        str(config.get("artifact_root", "outputs")),
        str(require(config, "experiment_name")),
        str(require(model_config, "model_name_or_path")),
        str(require(config, "benchmark")),
        int(require(config, "train_author_key")),
    )
    methods = validate_choices(config.get("generation_methods"), METHODS, "generation_methods")
    splits = validate_choices(config.get("generation_splits"), SPLITS, "generation_splits")
    num_return_sequences = int(config.get("generation_num_return_sequences", 3))
    if num_return_sequences <= 0:
        raise ValueError("generation_num_return_sequences must be greater than zero.")

    if layout.author_dir.exists() and not completed_run(layout.run_file):
        raise FileExistsError(f"Refusing to overwrite incomplete author run: {layout.author_dir}")

    if completed_run(layout.run_file):
        with layout.run_file.open(encoding="utf-8") as input_file:
            completed_metadata = json.load(input_file)
        if completed_metadata.get("source_config_sha256") != source_config_sha256:
            raise ValueError(
                f"Training settings differ from the completed run. Choose a new experiment_name: {experiment_path}"
            )
        print(f"Reusing completed training artifacts: {layout.author_dir}")
    else:
        training_command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            "configs/single_gpu.yaml",
            "scripts/run_ditto.py",
            str(model_path),
            f"--artifact_root={config.get('artifact_root', 'outputs')}",
            f"--dataset_root={config.get('dataset_root', 'benchmarks')}",
            f"--experiment_name={require(config, 'experiment_name')}",
            f"--benchmark={require(config, 'benchmark')}",
            f"--train_author_key={int(require(config, 'train_author_key'))}",
            f"--source_config_sha256={source_config_sha256}",
        ]
        for optional_field in ("train_instances", "train_pkl", "validation_pkl", "test_pkl"):
            if config.get(optional_field) is not None:
                training_command.append(f"--{optional_field}={config[optional_field]}")
        run(training_command)

    for split in splits:
        for method in methods:
            run(
                [
                    sys.executable,
                    "generate.py",
                    "--author-dir",
                    str(layout.author_dir),
                    "--method",
                    method,
                    "--split",
                    split,
                    "--num-return-sequences",
                    str(num_return_sequences),
                ]
            )


if __name__ == "__main__":
    main()
