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

Run the default custom benchmark for author `0`:

```sh
bash run.sh
```

Select another processed benchmark or author with environment variables:

```sh
BENCHMARK=ccat50 AUTHOR_KEY=3 bash run.sh
```

Experiment settings live in [`configs/`](configs/). `run.sh` launches training and generation with Mistral 7B Instruct.

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
