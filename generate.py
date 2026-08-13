import argparse
import pickle

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


def main():
    parser = argparse.ArgumentParser(description="GPT gen script")

    # Add arguments
    parser.add_argument("-b", "--benchmark", type=str, required=True, help="Name of benchmark dataset")
    parser.add_argument("-t", "--train_author_key", type=int, required=True, help="Author key in pkl file")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.2", help="Base model used for training")
    parser.add_argument("--adapter-path", help="Path to the trained DITTO adapter")
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

    model_id = args.model_id
    adapter_path = args.adapter_path or f"./outputs/{args.benchmark}-mistral-7b-instruct-ditto/ditto"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    base_model = PeftModel.from_pretrained(base_model, adapter_path)

    base_model.eval()

    generator = pipeline("text-generation", model=base_model, device="cuda", tokenizer=tokenizer)

    generator.tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

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
