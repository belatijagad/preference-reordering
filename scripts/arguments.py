"""Project argument definitions built on Transformers' public parser APIs."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerBase, TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Base model identifier or path."})
    base_model_revision: str | None = None
    model_revision: str = "main"
    tokenizer_name_or_path: str | None = None
    chat_template: str | None = None
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = True
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None


@dataclass
class DataArguments:
    preprocessing_num_workers: int | None = None
    truncation_side: str | None = None


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without replacing exported variables."""

    env_path = Path(path)
    if not env_path.is_file():
        return

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at {env_path}:{line_number}; expected KEY=VALUE.")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.isidentifier():
            raise ValueError(f"Invalid environment variable name at {env_path}:{line_number}: {key!r}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_yaml_and_cli(dataclass_types: tuple[type[Any], ...], argv: list[str] | None = None):
    """Parse one YAML config followed by `--field=value` overrides."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or not argv[0].endswith((".yaml", ".yml")):
        return HfArgumentParser(dataclass_types).parse_args_into_dataclasses(args=argv)

    with open(os.path.abspath(argv[0]), encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file) or {}
    if not isinstance(values, dict):
        raise ValueError("The experiment YAML must contain a mapping at its top level.")

    seen_overrides = set()
    for raw_argument in argv[1:]:
        if not raw_argument.startswith("--") or "=" not in raw_argument:
            raise ValueError(f"Expected YAML overrides in `--field=value` form, received: {raw_argument}")
        name, raw_value = raw_argument[2:].split("=", 1)
        name = name.replace("-", "_")
        if name in seen_overrides:
            raise ValueError(f"Duplicate argument provided: {name}")
        seen_overrides.add(name)
        values[name] = yaml.safe_load(raw_value)

    return HfArgumentParser(dataclass_types).parse_dict(values, allow_extra_keys=False)


def configure_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    chat_template: str | None = None,
) -> PreTrainedTokenizerBase:
    """Select a configured template or the tokenizer's native template."""

    resolved_template = chat_template if chat_template is not None else tokenizer.chat_template
    if isinstance(resolved_template, dict):
        if "default" in resolved_template:
            resolved_template = resolved_template["default"]
        elif len(resolved_template) == 1:
            resolved_template = next(iter(resolved_template.values()))
        else:
            available_templates = ", ".join(sorted(resolved_template))
            raise ValueError(
                "The selected tokenizer provides multiple named chat templates but no default. "
                f"Set `chat_template` explicitly. Available templates: {available_templates}."
            )
    if not isinstance(resolved_template, str) or not resolved_template.strip():
        raise ValueError(
            "The selected tokenizer does not provide a chat template. Choose an instruction-tuned tokenizer "
            "with a native template or set `chat_template` in the experiment configuration."
        )

    # Store the resolved template so saving the tokenizer also preserves the
    # exact formatting used by this experiment.
    tokenizer.chat_template = resolved_template
    return tokenizer


def get_tokenizer(model_args: ModelArguments, data_args: DataArguments):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    configure_chat_template(tokenizer, model_args.chat_template)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048
    return tokenizer


def is_openai_format(messages: Any) -> bool:
    return isinstance(messages, list) and all(
        isinstance(message, dict) and "role" in message and "content" in message for message in messages
    )


@dataclass
class DPOTrainingArguments(TrainingArguments):
    beta: float = 0.1
    max_prompt_length: int | None = None
    max_length: int | None = None
    loss_type: str = "sigmoid"
