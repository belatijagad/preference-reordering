# Show, Don't Tell

This repository is a fork of the [official DITTO repository](https://github.com/SALT-NLP/demonstrated-feedback) for [Show, Don't Tell: Aligning Language Models with Demonstrated Feedback](https://arxiv.org/abs/2406.00888).

This fork uses uv, unquantized BF16 training, FlashAttention 2, and supports A100 and Blackwell GPUs.

## Remote setup

Copy the environment file, add your Hugging Face token, and sync the remote environment:

```sh
cp .env.example .env
# Edit HF_TOKEN in .env
bash scripts/sync_remote.sh
```

## Run

Experiment, dataset, author, output, and generation settings are all in
[`configs/experiment.yaml`](configs/experiment.yaml). It references the separate
[`configs/ditto-mistral-7b-instruct.yaml`](configs/ditto-mistral-7b-instruct.yaml)
model and training configuration.
Dataset filenames and output paths are derived from the benchmark and model. Run
it with:

```sh
bash run.sh
```

To run another experiment, copy that YAML, edit it, and pass its path:

```sh
cp configs/experiment.yaml configs/my-experiment.yaml
bash run.sh configs/my-experiment.yaml
```

`run.sh` saves the shared configuration and tokenizer once under
`outputs/<experiment>/<model>/`. Each benchmark author gets separate SFT and
DITTO adapters, metrics, checkpoints, and resumable JSONL generations under:

```text
outputs/<experiment>/<model>/<benchmark>/author-NNN/
```

`.env` is only used for secrets such as `HF_TOKEN`. Rerunning the same command
reuses completed adapters and resumes any missing JSONL generations.

## Citation

```bibtex
@misc{shaikh2024show,
  title={Show, Don't Tell: Aligning Language Models with Demonstrated Feedback},
  author={Omar Shaikh and Michelle Lam and Joey Hejna and Yijia Shao and Michael Bernstein and Diyi Yang},
  year={2024},
  eprint={2406.00888},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
