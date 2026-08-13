"""Experiment artifact paths and metadata helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
OPERATIONAL_CONFIG_FIELDS = {
    "generation_methods",
    "generation_num_return_sequences",
    "generation_splits",
}


def model_slug(model_id: str) -> str:
    """Return the canonical output-directory name for a Hugging Face model ID."""

    model_name = model_id.rstrip("/").rsplit("/", maxsplit=1)[-1].lower()
    return re.sub(r"[^a-z0-9.]+", "-", model_name).strip("-")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_bundle_sha256(experiment_path: str | Path, model_path: str | Path) -> str:
    """Hash scientific settings across the experiment and referenced model YAMLs."""

    configs = []
    for path in (experiment_path, model_path):
        with Path(path).open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(f"Expected a YAML mapping in {path}.")
        configs.append(config)

    experiment_config, model_config = configs
    scientific_experiment = {
        key: value
        for key, value in experiment_config.items()
        if key not in OPERATIONAL_CONFIG_FIELDS and key != "model_config"
    }
    serialized = json.dumps(
        {"experiment": scientific_experiment, "model": model_config},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def package_versions(package_names: list[str]) -> dict[str, str | None]:
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


def git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary.replace(destination)


def write_yaml(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(value, output_file, sort_keys=False)
    temporary.replace(destination)


def ensure_shared_yaml(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    if not destination.exists():
        write_yaml(destination, value)
        return
    with destination.open(encoding="utf-8") as input_file:
        existing = yaml.safe_load(input_file)
    if existing != value:
        raise ValueError(f"Shared configuration at {destination} differs from this run.")


def ensure_shared_json(path: str | Path, value: dict[str, Any], identity_keys: tuple[str, ...]) -> None:
    destination = Path(path)
    if not destination.exists():
        write_json(destination, value)
        return
    existing = read_json(destination)
    mismatches = [key for key in identity_keys if existing.get(key) != value.get(key)]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"Shared metadata at {destination} conflicts on: {joined}.")


@dataclass(frozen=True)
class ArtifactLayout:
    model_dir: Path
    author_dir: Path

    @classmethod
    def for_experiment(
        cls,
        artifact_root: str | Path,
        experiment: str,
        model_id: str,
        benchmark: str,
        author_key: int,
    ) -> ArtifactLayout:
        if not experiment or Path(experiment).name != experiment:
            raise ValueError(f"experiment_name must be one directory name; received {experiment!r}.")
        if not benchmark or Path(benchmark).name != benchmark:
            raise ValueError(f"benchmark must be one directory name; received {benchmark!r}.")
        if not 0 <= author_key <= 999:
            raise ValueError(f"train_author_key must be between 0 and 999; received {author_key}.")
        model_dir = Path(artifact_root) / experiment / model_slug(model_id)
        author_dir = model_dir / benchmark / f"author-{author_key:03d}"
        return cls.for_training(model_dir, author_dir)

    @classmethod
    def for_training(cls, model_dir: str | Path, author_dir: str | Path) -> ArtifactLayout:
        layout = cls(Path(model_dir).resolve(), Path(author_dir).resolve())
        if re.fullmatch(r"author-\d{3}", layout.author_dir.name) is None:
            raise ValueError(f"Author directory must end in author-NNN; received {layout.author_dir}.")
        expected_model_dir = layout.author_dir.parent.parent
        if expected_model_dir != layout.model_dir:
            raise ValueError(
                f"Author directory must have the form <model>/<benchmark>/author-NNN; received {layout.author_dir}."
            )
        return layout

    @classmethod
    def from_author_dir(cls, author_dir: str | Path) -> ArtifactLayout:
        resolved_author_dir = Path(author_dir).resolve()
        return cls.for_training(resolved_author_dir.parent.parent, resolved_author_dir)

    @property
    def benchmark_dir(self) -> Path:
        return self.author_dir.parent

    @property
    def experiment_file(self) -> Path:
        return self.model_dir / "experiment.json"

    @property
    def config_file(self) -> Path:
        return self.model_dir / "config.yaml"

    @property
    def tokenizer_dir(self) -> Path:
        return self.model_dir / "tokenizer"

    @property
    def dataset_file(self) -> Path:
        return self.benchmark_dir / "dataset.json"

    @property
    def run_file(self) -> Path:
        return self.author_dir / "run.json"

    @property
    def adapters_dir(self) -> Path:
        return self.author_dir / "adapters"

    def adapter_dir(self, method: str) -> Path:
        return self.adapters_dir / method

    def metrics_file(self, method: str) -> Path:
        return self.author_dir / "metrics" / f"{method}.json"

    def checkpoint_dir(self, method: str) -> Path:
        return self.author_dir / "checkpoints" / method

    def generations_file(self, split: str, method: str) -> Path:
        return self.author_dir / "generations" / split / f"{method}.jsonl"
