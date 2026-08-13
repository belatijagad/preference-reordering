import argparse
import pickle
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from scripts.arguments import configure_chat_template, load_env_file


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="Generate samples from a trained PEFT adapter")

    # Add arguments
    parser.add_argument("-b", "--benchmark", type=str, required=True, help="Name of benchmark dataset")
    parser.add_argument("-t", "--train_author_key", type=int, required=True, help="Author key in pkl file")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument("--model-id", help="Override the base model recorded in the adapter")
    parser.add_argument("--tokenizer-id", help="Override the tokenizer saved with the adapter or base model")
    parser.add_argument("--chat-template-file", help="Optional Jinja chat template for tokenizers without one")
    parser.add_argument("--adapter-path", required=True, help="Path to the trained adapter")
    parser.add_argument("--test-pkl", help="Path to the benchmark test pickle")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-return-sequences", type=int, default=10)

    # Execute the parse_args() method
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.num_return_sequences <= 0:
        parser.error("--num-return-sequences must be greater than zero")
    set_seed(args.seed)

    adapter_path = args.adapter_path
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    model_id = args.model_id or adapter_config.base_model_name_or_path
    if not model_id:
        parser.error("The adapter does not record a base model; provide --model-id explicitly")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    ).to("cuda")

    adapter_directory = Path(adapter_path)
    saved_tokenizer_directory = (
        adapter_directory if (adapter_directory / "tokenizer_config.json").is_file() else adapter_directory.parent
    )
    tokenizer_source = (
        args.tokenizer_id
        or (str(saved_tokenizer_directory) if (saved_tokenizer_directory / "tokenizer_config.json").is_file() else None)
        or model_id
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    chat_template = Path(args.chat_template_file).read_text(encoding="utf-8") if args.chat_template_file else None
    configure_chat_template(tokenizer, chat_template)

    base_model = PeftModel.from_pretrained(base_model, adapter_path)

    base_model.eval()

    generator = pipeline("text-generation", model=base_model, device="cuda", tokenizer=tokenizer)

    path = args.test_pkl or f"./benchmarks/{args.benchmark}/processed/{args.benchmark}_test.pkl"

    with open(path, "rb") as pickle_file:
        data = pickle.load(pickle_file)

    spec_dataset = data[args.train_author_key]

    tasks = []

    for item in spec_dataset:
        tasks.append([{"content": item["prompt"], "role": "user"}])

    for task in tasks:
        outs = generator(
            task,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=args.num_return_sequences,
            return_full_text=False,
        )

        for out in outs:
            print("SAMPLE: ")
            print(out["generated_text"])
            print()


if __name__ == "__main__":
    main()
