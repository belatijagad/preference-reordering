#!/usr/bin/env python
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import logging
import pickle
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import transformers
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from transformers import AutoModelForCausalLM, set_seed
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer

from scripts.arguments import (
    DataArguments,
    DPOTrainingArguments,
    ModelArguments,
    get_tokenizer,
    is_openai_format,
    load_env_file,
    parse_yaml_and_cli,
)
from scripts.artifacts import (
    SCHEMA_VERSION,
    ArtifactLayout,
    ensure_shared_json,
    ensure_shared_yaml,
    git_metadata,
    model_slug,
    package_versions,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from scripts.ditto_trainer import DITTOTrainer
from scripts.verify_flash_attention import main as verify_flash_attention


class EarlyStoppingCallback(TrainerCallback):
    # any more training, and this overfits on train.
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def on_step_begin(self, args, state, control, **kwargs):

        if len(state.log_history) > 0:
            # get last log history
            last_loss = None

            for k in state.log_history[::-1]:
                if "loss" in k:
                    last_loss = k["loss"]
                    break

            if last_loss is not None and last_loss < self.threshold:
                control.should_training_stop = True


logger = logging.getLogger(__name__)


@dataclass
class DittoConfig(DPOTrainingArguments):
    output_dir: str | None = field(default=None)
    artifact_root: str = field(default="outputs")
    dataset_root: str = field(default="benchmarks")
    experiment_name: str = field(default="pilot-v1")
    benchmark: str | None = field(default=None)
    source_config_sha256: str | None = field(default=None)

    ditto_max_steps: int | None = field(
        default=30,
    )
    ditto_learning_rate: float | None = field(
        default=None,
    )

    ditto_lr_scheduler_type: str | None = field(
        default=None,
    )

    ditto_warmup_ratio: float | None = field(
        default=None,
    )
    ditto_per_device_train_batch_size: int | None = field(
        default=1,
    )
    ditto_gradient_accumulation_steps: int | None = field(
        default=8,
    )

    frac_expert: float | None = field(
        default=None,
    )
    frac_intermodel: float | None = field(
        default=None,
    )
    frac_replay: float | None = field(
        default=None,
    )
    rescale_batch: int | None = field(
        default=None,
    )

    resample_rate: int | None = field(default=10)
    bootstrap_count: int | None = field(default=10)
    generation_max_new_tokens: int | None = field(default=1024)
    generation_temperature: float | None = field(default=1.0)
    generation_batch_size: int | None = field(default=1)
    train_author_key: int | None = field(default=0)
    train_instances: int | None = field(default=None)
    train_pkl: str | None = field(default=None)
    validation_pkl: str | None = field(default=None)
    test_pkl: str | None = field(default=None)

    sft_stop_loss: float | None = field(
        default=1.25,
    )


def apply_chat_template(example, tokenizer, task: Literal["sft", "generation", "ditto"]):
    if task in ["sft", "generation"]:
        messages = example["chosen"]
        if not is_openai_format(messages):
            raise ValueError(f"Could not format example as dialogue for `{task}`; expected role/content messages.")

        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )

    elif task == "ditto":
        if not is_openai_format(example["chosen"]):
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
            )

        if "prompt" in example and is_openai_format(example["prompt"]):
            prompt_messages = example["prompt"]
            chosen_messages = example["chosen"]
        else:
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"][-1:]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        conversation_text = tokenizer.apply_chat_template(
            prompt_messages + chosen_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not conversation_text.startswith(prompt_text):
            raise ValueError(
                "The tokenizer's chat template does not render a full conversation with the generation prompt as a "
                "prefix. Set a compatible `chat_template` in the experiment configuration."
            )

        example["text_prompt"] = prompt_text
        example["text_chosen"] = conversation_text[len(prompt_text) :]
        if not example["text_chosen"]:
            raise ValueError("The tokenizer's chat template produced an empty assistant completion.")

    return example


def copy_adapter_weights(src_adapter_name, tgt_adapter_name, model):

    lora_modules = [module for module in model.modules() if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper))]

    with torch.no_grad():
        for model_module in lora_modules:
            if src_adapter_name in model_module.lora_A.keys():
                model_module.lora_A[tgt_adapter_name].load_state_dict(
                    model_module.lora_A[src_adapter_name].state_dict()
                )
                model_module.lora_B[tgt_adapter_name].load_state_dict(
                    model_module.lora_B[src_adapter_name].state_dict()
                )

            if src_adapter_name in model_module.lora_embedding_A.keys():
                model_module.lora_embedding_A[tgt_adapter_name].load_state_dict(
                    model_module.lora_embedding_A[src_adapter_name].state_dict()
                )
                model_module.lora_embedding_B[tgt_adapter_name].load_state_dict(
                    model_module.lora_embedding_B[src_adapter_name].state_dict()
                )


def initialize_artifacts(model_args, data_args, training_args) -> tuple[ArtifactLayout, dict]:
    """Create shared experiment metadata and an empty author run."""

    required_paths = {"train_pkl": training_args.train_pkl}
    for name, value in required_paths.items():
        if not value:
            raise ValueError(f"{name} must be configured.")
    if not training_args.benchmark:
        raise ValueError("benchmark must be configured.")
    if not training_args.source_config_sha256:
        raise ValueError("source_config_sha256 must be supplied by the experiment launcher.")

    layout = ArtifactLayout.for_experiment(
        training_args.artifact_root,
        training_args.experiment_name,
        model_args.model_name_or_path,
        training_args.benchmark,
        int(training_args.train_author_key),
    )
    if training_args.output_dir and Path(training_args.output_dir).resolve() != layout.author_dir:
        raise ValueError(f"output_dir conflicts with the path derived from the experiment YAML: {layout.author_dir}")
    expected_model_slug = model_slug(model_args.model_name_or_path)
    expected_author_dir = f"author-{int(training_args.train_author_key):03d}"
    if layout.model_dir.parent.name != training_args.experiment_name:
        raise ValueError(
            f"Model artifact directory must be nested under experiment {training_args.experiment_name!r}; "
            f"received {layout.model_dir}."
        )
    if layout.model_dir.name != expected_model_slug:
        raise ValueError(
            f"Model artifact directory must end in {expected_model_slug!r} for "
            f"{model_args.model_name_or_path!r}; received {layout.model_dir}."
        )
    if layout.benchmark_dir.name != training_args.benchmark:
        raise ValueError(
            f"Benchmark artifact directory must end in {training_args.benchmark!r}; received {layout.benchmark_dir}."
        )
    if layout.author_dir.name != expected_author_dir:
        raise ValueError(
            f"Author artifact directory must end in {expected_author_dir!r}; received {layout.author_dir}."
        )
    if layout.author_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing author run: {layout.author_dir}")

    run_specific_fields = {
        "artifact_root",
        "benchmark",
        "dataset_root",
        "experiment_name",
        "logging_dir",
        "output_dir",
        "run_name",
        "source_config_sha256",
        "test_pkl",
        "train_author_key",
        "train_instances",
        "train_pkl",
        "validation_pkl",
    }
    training_config = training_args.to_dict()
    for field_name in run_specific_fields:
        training_config.pop(field_name, None)
    shared_config = {
        "schema_version": SCHEMA_VERSION,
        "model": asdict(model_args),
        "data": asdict(data_args),
        "training": training_config,
    }
    ensure_shared_yaml(layout.config_file, shared_config)

    split_paths = {
        "train": training_args.train_pkl,
        "validation": training_args.validation_pkl,
        "test": training_args.test_pkl,
    }
    splits = {}
    for split, source in split_paths.items():
        if source:
            source_path = Path(source).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing {split} dataset: {source_path}")
            splits[split] = {"path": str(source_path), "sha256": sha256_file(source_path)}
    dataset_metadata = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": training_args.benchmark,
        "splits": splits,
    }
    ensure_shared_json(layout.dataset_file, dataset_metadata, ("benchmark", "splits"))

    experiment_metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment": training_args.experiment_name,
        "model_id": model_args.model_name_or_path,
        "model_revision": model_args.base_model_revision or "main",
        "tokenizer_id": model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        "tokenizer_revision": model_args.model_revision,
        "training_seed": training_args.seed,
        "config_sha256": sha256_file(layout.config_file),
        "git": git_metadata(),
        "packages": package_versions(
            ["accelerate", "datasets", "flash-attn", "peft", "torch", "transformers", "trl"]
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": ".".join(str(part) for part in torch.cuda.get_device_capability()),
            "cuda": torch.version.cuda,
        },
    }
    ensure_shared_json(
        layout.experiment_file,
        experiment_metadata,
        ("experiment", "model_id", "model_revision", "training_seed", "config_sha256"),
    )

    run_id = (
        f"{training_args.experiment_name}--{layout.model_dir.name}--{training_args.benchmark}"
        f"--author-{int(training_args.train_author_key):03d}"
    )
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "experiment": training_args.experiment_name,
        "benchmark": training_args.benchmark,
        "model_id": model_args.model_name_or_path,
        "source_config_sha256": training_args.source_config_sha256,
        "status": "running",
        "author_key": int(training_args.train_author_key),
        "started_at": utc_now(),
        "completed_at": None,
        "artifacts": {
            "sft_adapter": "adapters/sft",
            "ditto_adapter": "adapters/ditto",
            "sft_metrics": "metrics/sft.json",
            "ditto_metrics": "metrics/ditto.json",
            "generations": "generations",
        },
    }
    write_json(layout.run_file, run_metadata)
    return layout, run_metadata


def mark_incomplete_run_failed(layout: ArtifactLayout) -> None:
    """Mark ordinary Python failures while leaving completed runs unchanged."""

    if not layout.run_file.is_file():
        return
    run_metadata = read_json(layout.run_file)
    if run_metadata.get("status") == "running":
        run_metadata["status"] = "failed"
        run_metadata["completed_at"] = utc_now()
        write_json(layout.run_file, run_metadata)


def main():
    load_env_file()
    if len(sys.argv) < 2 or not sys.argv[1].endswith((".yaml", ".yml")):
        raise ValueError("run_ditto.py requires a model/training YAML as its first argument.")
    model_args, data_args, training_args = parse_yaml_and_cli((ModelArguments, DataArguments, DittoConfig))
    if not training_args.benchmark:
        raise ValueError("benchmark must be configured.")
    dataset_dir = Path(training_args.dataset_root) / training_args.benchmark / "processed"
    training_args.train_pkl = training_args.train_pkl or str(dataset_dir / f"{training_args.benchmark}_train.pkl")
    training_args.validation_pkl = training_args.validation_pkl or str(
        dataset_dir / f"{training_args.benchmark}_val.pkl"
    )
    training_args.test_pkl = training_args.test_pkl or str(dataset_dir / f"{training_args.benchmark}_test.pkl")
    if training_args.world_size != 1:
        raise ValueError("Artifact writes currently require a single-process Accelerate configuration.")
    if not model_args.use_peft:
        raise ValueError("DITTO experiments require `use_peft: true` for the SFT and DITTO adapters.")
    positive_integer_parameters = {
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "ditto_per_device_train_batch_size": training_args.ditto_per_device_train_batch_size,
        "ditto_gradient_accumulation_steps": training_args.ditto_gradient_accumulation_steps,
        "bootstrap_count": training_args.bootstrap_count,
        "resample_rate": training_args.resample_rate,
        "generation_max_new_tokens": training_args.generation_max_new_tokens,
        "generation_batch_size": training_args.generation_batch_size,
    }
    for parameter_name, value in positive_integer_parameters.items():
        if value is None or value <= 0:
            raise ValueError(f"{parameter_name} must be a positive integer.")
    if training_args.generation_temperature is None or training_args.generation_temperature <= 0:
        raise ValueError("generation_temperature must be greater than zero.")

    # Validate the exact BF16 forward/backward path before loading the dataset
    # or allocating the training model.
    verify_flash_attention()
    data_args.truncation_side = "left"  # Preserve assistant labels when sequences are truncated.
    layout, run_metadata = initialize_artifacts(model_args, data_args, training_args)
    atexit.register(mark_incomplete_run_failed, layout)
    training_args.output_dir = str(layout.checkpoint_dir("sft"))

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_dict = {}
    for dataset_key, path in [("train", training_args.train_pkl)]:
        prefs = {
            "prompt": [],
            "chosen": [],
        }

        with open(path, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        spec_dataset = data[int(training_args.train_author_key)]

        if training_args.train_instances:
            spec_dataset = spec_dataset[: int(training_args.train_instances)]

        run_metadata["training_examples"] = len(spec_dataset)
        run_metadata["training_indices"] = list(range(len(spec_dataset)))
        write_json(layout.run_file, run_metadata)

        for item in spec_dataset:
            prefs["prompt"].append(item["prompt"])

            prefs["chosen"].append(
                [{"content": item["prompt"], "role": "user"}, {"content": item["output"].strip(), "role": "assistant"}]
            )

        raw_dict[dataset_key] = Dataset.from_dict(prefs)

    raw_datasets = DatasetDict(raw_dict)
    if len(raw_datasets["train"]) == 0:
        raise ValueError("The training dataset is empty.")
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = get_tokenizer(model_args, data_args)
    if training_args.should_save and not layout.tokenizer_dir.exists():
        tokenizer.save_pretrained(layout.tokenizer_dir)

    #####################
    # Apply chat template
    #####################

    sft_raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "sft"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    sft_train_dataset = sft_raw_datasets["train"]

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "ditto"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns({"text_prompt": "prompt", "text_chosen": "chosen"})
        raw_datasets[split] = raw_datasets[split].add_column("example_id", list(range(len(raw_datasets[split]))))

    # Log up to two random samples from the training set.
    for index in random.sample(range(len(raw_datasets["train"])), min(2, len(raw_datasets["train"]))):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    if torch_dtype is not torch.bfloat16:
        raise ValueError("DITTO experiments require `torch_dtype: bfloat16`.")
    if not training_args.bf16 or training_args.fp16:
        raise ValueError("DITTO experiments require `bf16: true` and `fp16: false`.")
    if not model_args.use_flash_attention_2:
        raise ValueError("DITTO experiments require `use_flash_attention_2: true`.")

    model_kwargs = dict(
        revision=model_args.base_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    lora_config = LoraConfig(
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config, adapter_name="sft")
    model.set_adapter("sft")

    # we think it's useful, in our setting, to model the user too-
    # you could try uncommenting this

    # collator = DataCollatorForCompletionOnlyLM(
    #     # instruction_template=[733, 16289, 28793],
    #     response_template=[733, 28748, 16289, 28793],
    #     tokenizer=tokenizer, mlm=False
    # )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=sft_train_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        # data_collator=collator,
        packing=False,
        callbacks=[EarlyStoppingCallback(threshold=training_args.sft_stop_loss)],
    )

    # SFT Train
    sft_result = trainer.train()
    sft_metrics = dict(sft_result.metrics)
    sft_metrics["train_samples"] = len(sft_train_dataset)
    trainer.log_metrics("sft", sft_metrics)
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped_model.save_pretrained(
        layout.adapters_dir,
        selected_adapters=["sft"],
        is_main_process=trainer.accelerator.is_main_process,
    )
    if trainer.accelerator.is_main_process:
        write_json(layout.metrics_file("sft"), sft_metrics)

    #########################
    # Instantiate DPO trainer
    #########################

    training_args.learning_rate = training_args.ditto_learning_rate
    training_args.max_steps = training_args.ditto_max_steps
    training_args.lr_scheduler_type = training_args.ditto_lr_scheduler_type
    training_args.warmup_ratio = training_args.ditto_warmup_ratio
    training_args.per_device_train_batch_size = training_args.ditto_per_device_train_batch_size
    training_args.gradient_accumulation_steps = training_args.ditto_gradient_accumulation_steps
    training_args.output_dir = str(layout.checkpoint_dir("ditto"))

    model.add_adapter("ditto", lora_config)
    model.set_adapter("ditto")

    copy_adapter_weights("sft", "ditto", model)

    trainer = DITTOTrainer(
        model=model,
        ref_adapter_name="sft",  # keep the reference as the sft model.
        model_adapter_name="ditto",
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
        bootstrap_count=training_args.bootstrap_count,
        resample_rate=training_args.resample_rate,
        generation_max_new_tokens=training_args.generation_max_new_tokens,
        generation_temperature=training_args.generation_temperature,
        generation_batch_size=training_args.generation_batch_size,
    )

    ###############
    # Training loop
    ###############

    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("ditto", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    trainer.accelerator.wait_for_everyone()
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped_model.config.use_cache = True
    unwrapped_model.save_pretrained(
        layout.adapters_dir,
        selected_adapters=["ditto"],
        is_main_process=trainer.accelerator.is_main_process,
    )
    if trainer.accelerator.is_main_process:
        write_json(layout.metrics_file("ditto"), metrics)
        run_metadata = read_json(layout.run_file)
        run_metadata["status"] = "complete"
        run_metadata["completed_at"] = utc_now()
        write_json(layout.run_file, run_metadata)
        logger.info(f"Experiment artifacts saved to {layout.author_dir}")

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
