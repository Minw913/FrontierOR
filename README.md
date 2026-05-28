# FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization

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
- A standardized suite of **large-scale instances** (up to ~10⁷ vars/constraints),
- An expert-verified **Gurobi reference baseline**,
- A standalone **feasibility checker**.

We evaluate seven LLMs spanning frontier, cost-effective, and open-source tiers, in both **one-shot** and **test-time evolution** settings. Results reveal that frontier models still struggle to move from executable formulations to *efficient* optimization algorithms: the strongest one-shot model outperforms Gurobi in only **31%** of cases on both solution quality and computational efficiency, and even strong coding agents with test-time evolution achieve only **50%** on selected hard tasks. FrontierOR thus establishes a practical platform for systematically testing whether future LLMs and agents can move beyond correct formulation toward feasible, high-quality, and *efficient* algorithms.

---

## Environment Setup

You can run FrontierOR in any of three execution backends — pick based on how much isolation you want for the LLM-generated subprocess that solves each instance:

| Backend | What it does | When to use |
|---|---|---|
| `bare` (default) | Runs each `code.py` subprocess directly in the host environment, no resource caps. | Local development, fastest startup, you trust the generated code. |
| `systemd` | Wraps each subprocess in a `systemd-run --scope` unit with pinned CPUs (`AllowedCPUs`) and `MemoryMax`. | Multi-paper parallel runs on a Linux server — reproducible CPU/RAM caps, no Docker. |
| `docker` | Runs each subprocess in a Docker container with `--cpuset-cpus`, `--memory`, and `--network=none`. | Untrusted code, full isolation, or air-gapped reproducibility. Requires building the `frontier-or` image first (`docker build -t frontier-or .`). |

### Step 1 — Clone the repo (with Git LFS)

FrontierOR ships ~1k generated programs and instance bundles via Git LFS. Make sure LFS is installed before cloning:

```bash
git lfs install
git clone git@github.com:Minw913/frontier-or.git
cd frontier-or
git lfs pull
```

### Step 2 — Python environment

We recommend [`uv`](https://github.com/astral-sh/uv) for fast, reproducible installs (a `pip install -r requirements.txt` flow also works):

```bash
# uv (recommended)
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# or vanilla conda + pip
conda create -n FrontierOR python=3.13 -y && conda activate FrontierOR
pip install -r requirements.txt
```

### Step 3 — Gurobi license

The Gurobi baseline and feasibility checks require a working `gurobipy` license. Place it at the path pointed to by `GRB_LICENSE_FILE` (the Dockerfile mounts it at `/opt/gurobi/gurobi.lic`).

### Step 4 — OpenRouter API key (only for re-running generation)

LLM calls go through OpenRouter. `configs/oneshot.yaml` carries two scoped keys — one for one-shot generation and one for self-evolve — so you can route the two workloads through different OpenRouter accounts/quotas:

```yaml
OPENROUTER_API_KEY_ONESHOT: sk-or-...        # used by one_shot_eval.py
OPENROUTER_API_KEY_SELF_EVOLVE: sk-or-...    # used by self_evolving_frameworks/run_eval_modes.py
```

Alternatively, set a single env var that overrides both:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

You do **not** need this for the Quick Start below, which reuses pre-generated code in `samples/`.

---

## Quick Start

Five papers × seven models of pre-generated one-shot programs are shipped under `samples/oneshot_code/` for instant reproduction. The pipeline reads code from `eval/eval_papers/<paper>/<model>/code.py`, so first stage the shipped samples into that tree, then re-evaluate end-to-end (sanity-check on `tiny`, then large instances, with feasibility + solution-quality + QTE scoring) — no LLM API key needed:

```bash
mkdir -p eval && cp -r samples/oneshot_code eval/eval_papers

python -u one_shot_eval.py \
    --paper_id bierwirth2017 liao2020 pedersen2024 rahmaniani2018 walteros2020 \
    --models all \
    --reuse-code all \
    --instances tiny large_11 \
    --time_limit 600 \
    --paper_workers 5 --model_workers 7 --instance_workers 2 \
    --exec-mode bare
```

This reuses the staged `eval/eval_papers/<paper>/<model>/code.py` files, runs each program against the bundled instances, and writes results to per-model CSVs under `eval/`. Drop `--reuse-code all` to re-generate code from scratch (requires `OPENROUTER_API_KEY`).

---

## Evaluation

FrontierOR exposes three evaluation pipelines: the **Gurobi baseline**, **one-shot LLM generation**, and **test-time self-evolution**.

### Baseline — Gurobi reference solutions

Runs `gurobi_code.py` for each paper × instance, computes optimality gaps, and incrementally appends to a results CSV. Used to (re)build the reference solutions that LLM-generated programs are scored against.

```bash
python -u scripts/paper_reproduce/run_program_solutions.py \
    --paper-id gschwind2021 \
    --instances large_21 large_31 large_41 large_51 \
    --time_limit 3600 \
    --workers 5 \
    --backend systemd \
    --backend-cpus 1 \
    --backend-memory 640G \
    --force
```

Key flags:

- `--paper-id` — one or more paper IDs (folder names under `frontier-or/`). Omit to auto-discover.
- `--instances` — categorical names: `tiny`, `large_11`, `large_21`, `large_31`, `large_41`, `large_51`.
- `--time_limit` — per-run cap (seconds), forwarded to each subprocess.
- `--workers` — number of (paper, instance) cases to run in parallel.
- `--backend` — `bare` / `systemd` / `docker`, with companion `--backend-cpus` and `--backend-memory` caps.
- `--force` — overwrite any existing CSV row for a (paper, instance) pair.
- `--schema gurobi` writes the simplified `gurobi_solving_results.csv` (omit for the full schema, default).

### One-shot — generate + evaluate a program in a single LLM call

Drives the full one-shot pipeline: prompt assembly → LLM code generation → debug-retry loop → tiny sanity check → large-instance evaluation → per-model CSV row.

```bash
python -u one_shot_eval.py \
    --paper-tag A \
    --models all \
    --instances large_11 large_21 large_31 large_41 large_51 \
    --max_debug_retries 5 \
    --time_limit 3600 \
    --paper_workers 10 --model_workers 7 --instance_workers 50 \
    --exec-mode bare
```

Key flags:

- `--paper_id` / `--paper-tag` — explicit IDs vs. selecting by the `tag` column in `gurobi_results_all_new.csv` (the CSV is not shipped with the repo; supply your own at the repo root if you want to use `--paper-tag`).
- `--models` — short names from `configs/oneshot.yaml` (`gpt-5.3-codex`, `claude-opus-4.6`, `gemini-3.1-pro-preview`, `deepseek-r1`, `grok-4.20-beta`, `qwen3-coder-plus`, `llama-4-maverick`), or `all`.
- `--max_debug_retries` — bounded debug loop when the LLM's program raises (default 5).
- `--paper_workers` / `--model_workers` / `--instance_workers` — three-level parallelism across the (paper × model × instance) grid.
- `--exec-mode` — `bare` / `systemd` / `docker`, paired with `--cpus` / `--memory`.
- `--reuse-code {none,incomplete,all}` — `incomplete` (default) skips (paper, instance) rows already in CSV; `all` re-runs against existing `code.py`; `none` always re-generates.

### Self-evolve — test-time evolutionary frameworks

A single CLI wrapper drives all three self-evolving frameworks (`eoh`, `coral`, `openevolve`) on top of the same one-shot starting program. Defaults match the configurations reported in the paper — you usually only need to choose the framework and papers:

```bash
python -u self_evolving_frameworks/run_eval_modes.py \
    --framework eoh \
    --paper-id wangk2020 rostami2021 adulyasak2015 bertsimas2022 carvalho1999 \
               desaulniers2014 desaulniers2016 roberti2018 kobeaga2024 schwerdfeger2016 \
               archetti2007 watermeyer2020 pinnoi1997 furini2021 bard2002 \
               armbruster2012 nagy2015 pedersen2024 mehrotra1996 bront2009 \
    --primary-model gpt-5.3-codex \
    --paper-workers 10 \
    --eoh-pop-size 2 --eoh-n-pop 5 --eoh-operators e1 e2 m2 \
    --eoh-system-include-spec --eoh-enable-artifact \
    --run-id hardset_20_minwei_new
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
| _**Cost-effective / open-source models**_ | | | | | | | | |
| DeepSeek-R1          | 0.74     | 0.42     | 0.31     | 0.17     | 0.82     | 0.37     | 0.20     | 0.11     |
| Grok-4.20-beta       | 0.74     | 0.28     | 0.22     | 0.13     | 0.76     | 0.20     | 0.14     | 0.06     |
| Qwen3-Coder-Plus     | 0.60     | 0.26     | 0.20     | 0.09     | 0.52     | 0.21     | 0.12     | 0.07     |
| LLaMA-4-Maverick     | 0.47     | 0.18     | 0.13     | 0.06     | 0.52     | 0.13     | 0.07     | 0.02     |

Key takeaways:

1. **Frontier vs. cost-effective.** Frontier-tier feasibility clusters at 0.60–0.62 on Full and 0.49–0.64 on Hard; cost-effective models sit at 0.18–0.42 and 0.13–0.37 respectively — the gap is preserved at both scales.
2. **Execution is no longer the bottleneck.** GPT-5.3-Codex executes 98% of tasks but still scores only 0.49 feasibility on Hard; the difficulty has shifted from "compiles and runs" to "produces a valid, scalable algorithm".
3. **The Hard subset re-separates leaders.** On Full, the three frontier models are tightly bunched; on Hard the band widens — Claude Opus 4.6 retains the highest QTE (0.31 / 0.32), while GPT-5.3-Codex's Hard feasibility / QTE drop furthest.

For self-evolution results, continuous-metric variants, pair-wise comparisons, and per-paper case studies, see the [FrontierOR paper](FrontierOR__Benchmarking_LLMs__Capacity_for_Efficient_Algorithm_Design.pdf).

---

## Adding Support for New Models

FrontierOR routes all LLM calls through OpenRouter, so adding a model is a configuration-only change in most cases.

1. **Pick the OpenRouter route** (e.g. `anthropic/claude-opus-4.6`, `openai/gpt-5.3-codex`).
2. **Register a short name and route** in `configs/oneshot.yaml` — copy an existing block and edit the `route`, `short_name`, and any sampling parameters (temperature, max tokens, reasoning effort).
3. **(Optional) Tune the prompt** by editing the `build_prompt()` function in `one_shot_eval.py` if the model has unusual formatting requirements.
4. **Run** `python one_shot_eval.py --paper-id <ID> --models <short_name>` to verify the model's code is parsed correctly, then scale up via `--models all` or `--paper-tag`.

For self-evolution, the same short name flows through `--primary-model` / `--secondary-model` in `self_evolving_frameworks/run_eval_modes.py`.

<!-- --- -->

<!-- ## Citation

If you use FrontierOR in your research, please cite:

```bibtex
@inproceedings{frontieror2026,
  title     = {FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
``` -->
