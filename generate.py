import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from scripts.arguments import configure_chat_template, load_env_file
from scripts.artifacts import SCHEMA_VERSION, ArtifactLayout, model_slug, read_json, sha256_file, write_json

METHODS = ("zero_shot", "few_shot", "sft", "ditto")
SPLITS = ("validation", "test")


def load_pickle_split(path: Path, author_key: int) -> list[dict[str, Any]]:
    with path.open("rb") as pickle_file:
        data = pickle.load(pickle_file)
    examples = data[author_key]
    if not isinstance(examples, list):
        raise ValueError(f"Expected author {author_key} in {path} to contain a list of examples.")
    return examples


def select_examples(path: Path, run: dict[str, Any], instances_per_author: int) -> list[dict[str, Any]]:
    with path.open("rb") as pickle_file:
        data = pickle.load(pickle_file)
    author_keys = [int(key) for key in run.get("selected_author_keys", [run["author_key"]])]
    examples = []
    for author_key in author_keys:
        if author_key not in data:
            raise KeyError(f"Author {author_key} is not present in {path}.")
        author_examples = data[author_key]
        if len(author_examples) < instances_per_author:
            raise ValueError(
                f"Author {author_key} has fewer than {instances_per_author} examples in {path}."
            )
        for index, example in enumerate(author_examples[:instances_per_author]):
            selected = dict(example)
            selected["author_key"] = author_key
            selected["source_index"] = index
            examples.append(selected)
    return examples


def few_shot_prompt(demonstrations: list[dict[str, Any]], prompt: str) -> str:
    sections = ["Below are a few writing samples."]
    for index, demonstration in enumerate(demonstrations, start=1):
        sections.extend(
            [
                f"### EXAMPLE {index}",
                str(demonstration["prompt"]),
                str(demonstration["output"]),
            ]
        )
    sections.extend(
        [
            "Respond to the following prompt in the same way as the writing samples:",
            prompt,
        ]
    )
    return "\n\n".join(sections)


def load_existing_generations(
    path: Path,
    run_id: str,
    method: str,
    split: str,
    temperature: float,
    max_new_tokens: int,
    generation_base_seed: int,
    num_return_sequences: int,
) -> set[str]:
    generation_ids = set()
    if not path.exists():
        return generation_ids

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from error
            expected = {"run_id": run_id, "method": method, "split": split}
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Existing generation row at {path}:{line_number} belongs to another run.")
            generation_settings = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "generation_base_seed": generation_base_seed,
                "num_return_sequences": num_return_sequences,
            }
            if any(row.get(key) != value for key, value in generation_settings.items()):
                raise ValueError(f"Generation settings at {path}:{line_number} differ from this invocation.")
            generation_id = row.get("generation_id")
            if not isinstance(generation_id, str) or generation_id in generation_ids:
                raise ValueError(f"Missing or duplicate generation_id at {path}:{line_number}.")
            generation_ids.add(generation_id)
    return generation_ids


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Generate resumable JSONL artifacts for one author run")
    parser.add_argument("--author-dir", required=True, help="Author artifact directory")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    parser.add_argument("--instances-per-author", type=int, required=True)
    parser.add_argument("--few-shot-instances-per-author", type=int, required=True)
    args = parser.parse_args()

    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    if args.temperature is not None and args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.num_return_sequences <= 0:
        parser.error("--num-return-sequences must be greater than zero")
    if args.instances_per_author <= 0:
        parser.error("--instances-per-author must be greater than zero")
    if args.few_shot_instances_per_author <= 0:
        parser.error("--few-shot-instances-per-author must be greater than zero")

    layout = ArtifactLayout.from_author_dir(args.author_dir)
    experiment = read_json(layout.experiment_file)
    dataset = read_json(layout.dataset_file)
    run = read_json(layout.run_file)
    if run.get("status") != "complete":
        raise ValueError(f"Author run is not complete: {layout.run_file}")
    expected_identity = {
        "experiment": layout.model_dir.parent.name,
        "benchmark": layout.benchmark_dir.name,
        "model_id": experiment.get("model_id"),
    }
    if any(run.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"Run metadata does not match its artifact directory: {layout.run_file}")
    if dataset.get("benchmark") != layout.benchmark_dir.name:
        raise ValueError(f"Dataset metadata does not match its artifact directory: {layout.dataset_file}")
    if layout.model_dir.name != model_slug(str(experiment.get("model_id", ""))):
        raise ValueError(f"Experiment metadata does not match its artifact directory: {layout.experiment_file}")
    expected_author_dir = f"author-{int(run['author_key']):03d}"
    if layout.author_dir.name != expected_author_dir:
        raise ValueError(f"Author metadata does not match its artifact directory: {layout.run_file}")
    config_sha256 = sha256_file(layout.config_file)
    if config_sha256 != experiment.get("config_sha256"):
        raise ValueError(f"Shared experiment configuration has changed: {layout.config_file}")

    with layout.config_file.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    training_config = config["training"]
    max_new_tokens = args.max_new_tokens or int(training_config["generation_max_new_tokens"])
    temperature = args.temperature or float(training_config["generation_temperature"])
    generation_seed = args.generation_seed if args.generation_seed is not None else int(experiment["training_seed"])

    split_metadata = dataset["splits"].get(args.split)
    if split_metadata is None:
        raise ValueError(f"The {args.split} split is not recorded in {layout.dataset_file}.")
    author_key = int(run["author_key"])
    split_path = Path(split_metadata["path"])
    if sha256_file(split_path) != split_metadata["sha256"]:
        raise ValueError(f"The recorded {args.split} dataset has changed: {split_path}")
    examples = select_examples(split_path, run, args.instances_per_author)
    if isinstance(run.get("selected_prompts"), dict) and not run["selected_prompts"].get(args.split):
        run["selected_prompts"][args.split] = examples[0]["prompt"]
        write_json(layout.run_file, run)

    demonstrations = None
    if args.method == "few_shot":
        train_metadata = dataset["splits"].get("train")
        if train_metadata is None:
            raise ValueError("Few-shot generation requires the training split in dataset.json.")
        train_path = Path(train_metadata["path"])
        if sha256_file(train_path) != train_metadata["sha256"]:
            raise ValueError(f"The recorded training dataset has changed: {train_path}")
        demonstrations = select_examples(
            train_path,
            run,
            args.few_shot_instances_per_author,
        )

    run_id = str(run["run_id"])
    output_path = layout.generations_file(args.split, args.method)
    existing_ids = load_existing_generations(
        output_path,
        run_id,
        args.method,
        args.split,
        temperature,
        max_new_tokens,
        generation_seed,
        args.num_return_sequences,
    )
    expected_ids = {
        f"{run_id}--{args.method}--{args.split}-{prompt_index:04d}--sample-{sample_id:02d}"
        for prompt_index in range(len(examples))
        for sample_id in range(args.num_return_sequences)
    }
    pending_ids = expected_ids - existing_ids
    unexpected_ids = existing_ids - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"Existing artifact has {len(unexpected_ids)} generation IDs outside this invocation: {output_path}"
        )
    if not pending_ids:
        print(f"Generation artifact is already complete: {output_path}")
        return

    model_id = str(experiment["model_id"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=experiment.get("model_revision"),
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    ).to("cuda")
    adapter = None
    if args.method in {"sft", "ditto"}:
        adapter = str(run["artifacts"][f"{args.method}_adapter"])
        adapter_path = layout.author_dir / adapter
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Missing {args.method} adapter: {adapter_path}")
        base_model = PeftModel.from_pretrained(base_model, adapter_path)
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(layout.tokenizer_dir)
    configure_chat_template(tokenizer)
    tokenizer.model_max_length = int(base_model.config.max_position_embeddings)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generator = pipeline("text-generation", model=base_model, device="cuda", tokenizer=tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        for prompt_index, example in enumerate(examples):
            prompt = str(example["prompt"])
            reference = str(example["output"])
            user_prompt = few_shot_prompt(demonstrations, prompt) if demonstrations is not None else prompt
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            for sample_id in range(args.num_return_sequences):
                generation_id = (
                    f"{run_id}--{args.method}--{args.split}-{prompt_index:04d}--sample-{sample_id:02d}"
                )
                if generation_id not in pending_ids:
                    continue

                sample_seed = generation_seed + prompt_index * 100_000 + sample_id
                set_seed(sample_seed)
                outputs = generator(
                    rendered_prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=1,
                    return_full_text=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generation = str(outputs[0]["generated_text"])
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "generation_id": generation_id,
                    "run_id": run_id,
                    "method": args.method,
                    "benchmark": dataset["benchmark"],
                    "split": args.split,
                    "author_key": int(example.get("author_key", author_key)),
                    "prompt_id": (
                        f"{dataset['benchmark']}:{args.split}:"
                        f"{int(example.get('author_key', author_key))}:{int(example.get('source_index', prompt_index)):04d}"
                    ),
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "reference": reference,
                    "sample_id": sample_id,
                    "training_seed": experiment["training_seed"],
                    "generation_base_seed": generation_seed,
                    "generation_seed": sample_seed,
                    "num_return_sequences": args.num_return_sequences,
                    "config_sha256": config_sha256,
                    "model_id": model_id,
                    "adapter": adapter,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "generated_tokens": len(tokenizer(generation, add_special_tokens=False)["input_ids"]),
                    "generation": generation,
                }
                output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                output_file.flush()

    print(f"Generation artifact updated: {output_path}")


if __name__ == "__main__":
    main()
