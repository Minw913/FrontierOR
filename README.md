<p align="center">
  <img src="figures/frontieror_logo.svg" alt="FrontierOR logo" height="100">
</p>

# FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization

<p align="center">
  <a href="https://frontieror.vercel.app/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Website-frontieror.vercel.app-000" alt="Website"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2605.25246"><img src="https://img.shields.io/badge/arXiv-2605.25246-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/SmartOR/FrontierOR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SmartOR%2FFrontierOR-FFD21E" alt="HuggingFace Dataset"></a>
</p>

<p align="center">
  <img src="figures/FrontierOR.png" alt="FrontierOR overview" width="100%">
</p>

> Overview of the FrontierOR benchmark. FrontierOR spans diverse problem domains, formulation types, and application fields. Each optimization problem involves 10² to 10⁷ decision variables and constraints (median ~10⁴), with Gurobi failing to reach optimality on **46%** of large-scale instances within a one-hour time budget. We construct the benchmark by collecting problems from leading OR journals, and ensure data quality through multi-round expert review.

---

## Introduction

Large language models (LLMs) are increasingly used for optimization modeling and solver-code generation, yet practical operations research (OR) problems often require a harder capability: designing *scalable algorithms* that exploit problem structure and outperform direct formulation-and-solve baselines. Existing benchmarks are limited to small or simplified examples far below real-world scale and complexity.

We introduce **FrontierOR**, among the first benchmarks to systematically evaluate LLM-based efficient algorithm design for realistic large-scale optimization problems. FrontierOR includes **180 tasks** derived from methodologically diverse papers published in top-tier OR venues, each shipped with:

- A natural-language **problem description**,
- A faithful **mathematical formulation**,
- A standardized suite of **large-scale instances**,
- An expert-verified **Gurobi reference baseline**,
- A standalone **feasibility checker**.

We currently evaluate seven LLMs backbones and three test-time evalution methods and results reveal that frontier models still struggle to move from executable formulations to *efficient* optimization algorithms: the strongest model outperforms Gurobi in only **31%** of cases on both solution quality and computational efficiency, and even strong coding agents with test-time evolution achieve only **50%** on selected hard tasks. FrontierOR thus establishes a practical platform for systematically testing whether future LLMs and agents can move beyond correct formulation toward feasible, high-quality, and *efficient* algorithms.

---

## Environment Setup

You can run FrontierOR in any of three execution backends. Both one-shot and self-evolve evaluation pipelines accept the backend via `--exec-mode` (e.g. `--exec-mode systemd --cpus 1`):

| Backend | What it does | When to use |
|---|---|---|
| `bare` | Runs each `code.py` subprocess directly in the host environment, no resource caps. | Local development, fastest startup, you trust the generated code. |
| `systemd` (default) | Wraps each subprocess in a `systemd-run --scope` unit with pinned CPUs (`AllowedCPUs`) and `MemoryMax`. | Multi-paper parallel runs on a Linux server — reproducible CPU/RAM caps, no Docker. |
| `docker` | Runs each subprocess in a Docker container with `--cpuset-cpus`, `--memory`, and `--network=none`. | Untrusted code, full isolation, or air-gapped reproducibility. Requires building the `frontier-or` image first (`docker build -t frontier-or .`). |

### Step 1 — Clone the repo and download the dataset

The code lives on GitHub; the benchmark data is hosted on HuggingFace at [`SmartOR/FrontierOR`](https://huggingface.co/datasets/SmartOR/FrontierOR).

```bash
# 1. Clone the code repo
git clone git@github.com:Minw913/FrontierOR.git
cd FrontierOR

# 2. Download the dataset into ./frontier-or/ (the path the eval scripts read from)
pip install -U "huggingface_hub[cli]"
huggingface-cli download SmartOR/FrontierOR --repo-type dataset --local-dir frontier-or
```

### Step 2 — Python environment

We recommend [`uv`](https://github.com/astral-sh/uv) for fast, reproducible installs:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step 3 — Gurobi license

During evaluation some LLM-generated solver programs require a valid `gurobipy` license. Place it at the path pointed to by `GRB_LICENSE_FILE` (the Dockerfile mounts it at `/opt/gurobi/gurobi.lic`).

### Step 4 — OpenRouter API key

LLM calls go through OpenRouter and the model registry is in `configs/oneshot.yaml`. `configs/api_keys.yaml` provides two scoped keys (one-shot generation and test-time self-evolution), allowing each workload to use a separate OpenRouter account or quota:

```yaml
OPENROUTER_API_KEY_ONESHOT: sk-or-...       
OPENROUTER_API_KEY_SELF_EVOLVE: sk-or-...   
```

You do **not** need this for the Quick Start below, which reuses pre-generated code in `samples/`.

---

## Quick Start

Run the following command to quickly conduct the one-shot evaluation, with results written to `eval/`. No API key is required, making this the fastest sanity check that the framework is set up correctly.

```bash
python -u one_shot_eval.py --paper_id bierwirth2017 liao2020 --reuse-code all --code-root samples/oneshot_code --exec-mode bare
```

---

## Run Evaluation

FrontierOR exposes two evaluation pipelines: **one-shot LLM generation**, and **test-time self-evolution**.

### One-shot LLM generation

Drives the full one-shot pipeline: prompt assembly → LLM code generation → tiny sanity check → large-instance evaluation.

```bash
python -u one_shot_eval.py \
    --models gpt-5.3-codex \
    --instances large_11 large_21 large_31 large_41 large_51 \
    --max_debug_retries 5 \
    --time_limit 3600 \
    --paper_workers 10 --model_workers 7 --instance_workers 50 \
    --exec-mode bare
```

Key flags:

- `--paper_id` — explicit paper IDs.
- `--models` — short names from `configs/oneshot.yaml` (`gpt-5.3-codex`, `claude-opus-4.6`, `gemini-3.1-pro-preview`, `deepseek-r1`, `grok-4.20-beta`, `qwen3-coder-plus`, `llama-4-maverick`), or `all`.
- `--max_debug_retries` — bounded debug loop when the LLM's program raises.
- `--paper_workers` / `--model_workers` / `--instance_workers` — three-level parallelism across the (paper × model × instance) grid.
- `--exec-mode` — `bare` / `systemd` / `docker`, paired with `--cpus` / `--memory`.
- `--reuse-code {none,incomplete,all}` — `incomplete` (default) skips (paper, instance) rows already in CSV; `all` re-runs against existing `code.py`; `none` always re-generates.

### Test-time Self-evolution

A single CLI wrapper drives all three self-evolving frameworks (`eoh`, `coral`, `openevolve`) on top of the same one-shot starting program. Defaults match the configurations reported in the paper — you usually only need to choose the framework and papers:

```bash
python -u test_time_self_evolution/run_eval_modes.py \
    --framework openevolve \
    --openevolve-iterations 30 \
    --primary-model google/gemini-3.1-pro-preview \
    --paper-workers 30 \
    --dev-set median \
    --test-instance-workers 4 \
    --exec-mode systemd \
    --cpus 1 --memory 640G \
    --run-id your_run_id
```

Switch frameworks via `--framework {eoh,coral,openevolve}`; framework-specific knobs (`--eoh-*`, `--coral-*`, `--openevolve-iterations`) override the defaults when needed. The stage1 (binary gate on `tiny`) → stage2 (dev set fitness) → test-set scoring pipeline is shared across all three frameworks for apples-to-apples comparison.

---

## Leaderboard

One-shot performance on **FrontierOR Full** (180 tasks) and **FrontierOR Hard** (50 tasks). Metrics: **Exec.** = execution rate, **Feas.** = large-instance feasibility, **Sol. q.** = solution-quality pass rate vs. Gurobi, **QTE** = joint quality-time-efficiency pass rate. **Bold** = best, _underline_ = second-best per column.

| Model | Exec. (Full) | Feas. (Full) | Sol. q. (Full) | QTE (Full) | Exec. (Hard) | Feas. (Hard) | Sol. q. (Hard) | QTE (Hard) |
|---|---|---|---|---|---|---|---|---|
| _**Frontier models**_ | | | | | | | | |
| Claude Opus 4.6      | _0.93_   | **0.62** | _0.48_   | **0.31** | 0.94     | _0.60_   | _0.44_   | **0.32** |
| GPT-5.3-Codex        | **0.98** | 0.60     | _0.48_   | _0.26_   | _0.98_   | 0.49     | 0.30     | 0.18     |
| Gemini 3.1 Pro       | _0.93_   | _0.61_   | **0.52** | 0.25     | **1.00** | **0.64** | **0.44** | _0.22_   |
| _**Cost-effective**_ | | | | | | | | |
| DeepSeek-R1          | 0.74     | 0.42     | 0.31     | 0.17     | 0.82     | 0.37     | 0.20     | 0.11     |
| Grok-4.20-beta       | 0.74     | 0.28     | 0.22     | 0.13     | 0.76     | 0.20     | 0.14     | 0.06     |
| Qwen3-Coder-Plus     | 0.60     | 0.26     | 0.20     | 0.09     | 0.52     | 0.21     | 0.12     | 0.07     |
| LLaMA-4-Maverick     | 0.47     | 0.18     | 0.13     | 0.06     | 0.52     | 0.13     | 0.07     | 0.02     |

Key takeaways:

1. **Frontier vs. cost-effective.** Frontier-tier feasibility clusters at 0.60–0.62 on Full and 0.49–0.64 on Hard; cost-effective models sit at 0.18–0.42 and 0.13–0.37 respectively — the gap is preserved at both scales.
2. **Execution is no longer the bottleneck.** GPT-5.3-Codex executes 98% of tasks but still scores only 0.49 feasibility on Hard; the difficulty has shifted from "compiles and runs" to "produces a valid, scalable algorithm".
3. **The Hard subset re-separates leaders.** On Full, the three frontier models are tightly bunched; on Hard the band widens — Claude Opus 4.6 retains the highest QTE (0.31 / 0.32), while GPT-5.3-Codex's Hard feasibility / QTE drop furthest.

For self-evolution results, continuous-metric variants, pair-wise comparisons, and per-paper case studies, see the [FrontierOR paper](https://arxiv.org/abs/2605.25246).

---

## Adding Support for New Models

FrontierOR routes all LLM calls through OpenRouter, so adding a model is a configuration-only change in most cases.

1. **Pick the OpenRouter route** (e.g. `anthropic/claude-opus-4.6`, `openai/gpt-5.3-codex`).
2. **Register a short name and route** in `configs/oneshot.yaml` — copy an existing block and edit the `route`, `short_name`, and any sampling parameters (temperature, max tokens, reasoning effort).
3. **(Optional) Tune the prompt** by editing the `build_prompt()` function in `one_shot_eval.py` if the model has unusual formatting requirements.
4. **Run** `python one_shot_eval.py --paper-id <ID> --models <short_name>` to verify the model's code is parsed correctly.

For self-evolution, the same short name flows through `--primary-model` / `--secondary-model` in `test_time_self_evolution/run_eval_modes.py`.

---

## Citation

If you use FrontierOR in your research, please cite:

```bibtex
@article{kong2026frontieror,
  title={FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization},
  author={Kong, Minwei and Jiang, Chonghe and Qu, Ao and Ouyang, Wenbin and Zeng, Zhaoming and Guo, Xiaotong and Li, Zhekai and Li, Junyi and Fan, Yi and Zheng, Xinshou and others},
  journal={arXiv preprint arXiv:2605.25246},
  year={2026}
}
```
