## Show, Don't Tell: Aligning Language Models with Demonstrated Feedback

This repository contains source code for the paper **Show, Don't Tell: Aligning Language Models with Demonstrated Feedback** by [Omar Shaikh](https://oshaikh.com/), [Michelle Lam](https://michelle123lam.github.io/), [Joey Hejna](http://joeyhejna.com/), [Yijia Shao](https://cs.stanford.edu/~shaoyj/), [Michael Bernstein](https://hci.stanford.edu/msb/), and [Diyi Yang](https://cs.stanford.edu/~diyiy/). Feel free to reach out to [Omar Shaikh](https://oshaikh.com/) with any questions!

[[Paper]](https://arxiv.org/abs/2406.00888)

### *Abstract* 

Language models are aligned to emulate the collective voice of many, resulting in outputs that align with no one in particular. Steering LLMs away from generic output is possible through supervised finetuning or RLHF, but requires prohibitively large datasets for new ad-hoc tasks. We argue that it is instead possible to align an LLM to a specific setting by leveraging a very small number ($<10$) of demonstrations as feedback. Our method, Demonstration ITerated Task Optimization (DITTO), directly aligns language model outputs to a user's demonstrated behaviors. Derived using ideas from online imitation learning, DITTO cheaply generates online comparison data by treating users' demonstrations as preferred over output from the LLM and its intermediate checkpoints. We evaluate DITTO's ability to learn fine-grained style and task alignment across domains such as news articles, emails, and blog posts. Additionally, we conduct a user study soliciting a range of demonstrations from participants ($N=16$). Across our benchmarks and user study, we find that win-rates for DITTO outperform few-shot prompting, supervised fine-tuning, and other self-play methods by an average of 19\% points. By using demonstrations as feedback directly, DITTO offers a novel method for effective customization of LLMs.

### *Instructions*

We build on the [Alignment Handbook](https://github.com/huggingface/alignment-handbook). Its source revision and all Python dependencies are declared in `pyproject.toml` and managed with [uv](https://docs.astral.sh/uv/).

Install uv, then create the environment on the Linux training machine:

```shell
uv sync --no-dev
```

This creates `.venv`, installs Python 3.10, installs the pinned Alignment Handbook commit, resolves the CUDA 12.8 build of PyTorch, and builds FlashAttention 2 with A100 (`sm_80`) and RTX 5090 / RTX PRO 6000 Blackwell (`sm_120`) code. Commit the generated `uv.lock` after resolving it on the remote Linux machine. To include linting, documentation, and test tools, use `uv sync` without `--no-dev`. To work with the data-preparation notebooks, add `--group notebooks`.

The remote host needs an R570-or-newer NVIDIA Linux driver and the full CUDA Toolkit 12.8 or newer, including `nvcc`. The PyTorch wheel bundles its CUDA runtime, but FlashAttention is deliberately compiled locally so that its extension matches PyTorch and contains `sm_120` kernels. A runtime-only CUDA installation is not sufficient. The build defaults to four parallel jobs to limit RAM usage; adjust `MAX_JOBS` in `pyproject.toml` if appropriate for the remote machine.

Training, online sampling, and generation all use unquantized BF16 model weights with FlashAttention 2. Quantized 4-bit and 8-bit loading is intentionally rejected.

PyTorch 2.7 added Blackwell support with CUDA 12.8, but FlashAttention 2's published support matrix still names Ampere, Ada, and Hopper. Building for `sm_120` solves the missing-kernel-image failure, but consumer Blackwell must still pass the exact BF16 forward/backward path used here. After syncing, run the repository smoke test on each remote GPU type before starting a full experiment:

```shell
uv run python scripts/verify_flash_attention.py
```

Run training and generation through uv:

```shell
uv run accelerate launch \
    --config_file configs/single_gpu.yaml \
    scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml
```

A sample shell script with training + generation is in `run.sh` (Mistral Instruct v0.2 7B). It defaults to the custom benchmark and author key 0. Select another processed benchmark or author with the `BENCHMARK` and `AUTHOR_KEY` environment variables. Note that you may need to change the config file for your specific hardware or dataset.

```shell 
bash run.sh
# or: BENCHMARK=ccat50 AUTHOR_KEY=3 bash run.sh
```

### Debugging

* `AttributeError: 'DittoConfig' object has no attribute 'packing'`: keep `trl==0.8.6` in `pyproject.toml`.
* `... CUDA capability sm_120 is not compatible ...` or `no kernel image is available`: the RTX PRO 6000 and RTX 5090 are Blackwell (`sm_120`) GPUs. The original PyTorch 2.1.2/CUDA 12.1 environment predates Blackwell. This project now uses PyTorch 2.7.1 from the CUDA 12.8 index and builds FlashAttention with an explicit `sm_120` target. An `sm_120` build is necessary but not sufficient; require `scripts/verify_flash_attention.py` to pass on the target machine.
* If FlashAttention reports that `CUDA_HOME` or `nvcc` is missing, install the full CUDA Toolkit 12.8+ on the remote host and make sure its `bin` directory is on `PATH`. The CUDA version displayed by `nvidia-smi` describes driver capability and does not prove that the compiler is installed.
* If the remote machine must reproduce the original A100 software stack exactly, use the repository revision before this migration. The CUDA 12.8 stack remains compatible with A100 hardware, but upgrading PyTorch and DeepSpeed may change numerical or performance characteristics.


### *How do I cite this work?* 

Feel free to use the following BibTeX entry.

**BibTeX:**

```tex
@misc{shaikh2024show,
      title={Show, Don't Tell: Aligning Language Models with Demonstrated Feedback}, 
      author={Omar Shaikh and Michelle Lam and Joey Hejna and Yijia Shao and Michael Bernstein and Diyi Yang},
      year={2024},
      eprint={2406.00888},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
