"""
LLM Evaluation Pipeline

Assembles prompts from problem_description.txt + raw instance_1.json (as a
data-structure example), calls an LLM via OpenRouter, runs generated code
against each instance via --instance_path, compares with Gurobi optimal
solutions, and supports self-correction (multi-turn).

Usage:
  python one_shot_eval.py \
    --paper_id bodur2017 \
    --instances tiny large_1 \
    --max_correct_retries 0 \
    --max_debug_retries 5 \
    --time_limit 3600 \
    --paper_workers 4 --model_workers 2 \
    2>&1 | tee eval_log_2.txt
"""

import argparse
import atexit
import contextlib
import csv
import fcntl
import glob
import json
import math
import os
import re
import subprocess
import sys
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory containing the per-(paper, model) ``code.py`` files to READ from.
# Defaults to ``<repo>/eval/eval_papers``; overridden via ``--code-root`` (e.g.
# ``--code-root samples/oneshot_code`` to evaluate the shipped sample programs).
# Pipeline WRITES (logs, solutions, intermediate code) always go to
# ``eval/eval_papers/`` regardless, so the code-root tree stays untouched.
CODE_ROOT = None

# Results CSV resolution priority:
#   1. ``_RESULTS_CSV_OVERRIDE`` (set by --results_csv): wins for every model.
#   2. ``_results_csv_local.path`` (set by ``process_paper_model`` from the
#      per-model mapping when no override is given): scoped to a single
#      (paper, model) thread, so multiple models can run in parallel and each
#      writes to its own CSV.
#   3. ``_DEFAULT_RESULTS_CSV``: last-resort fallback when neither is set
#      (e.g., a model with no mapping entry).
_DEFAULT_RESULTS_CSV = os.path.join(ROOT_DIR, "eval", "eval_results.csv")
_RESULTS_CSV_OVERRIDE = None
_results_csv_local = threading.local()


def get_results_csv_path():
    if _RESULTS_CSV_OVERRIDE:
        return _RESULTS_CSV_OVERRIDE
    p = getattr(_results_csv_local, "path", None)
    if p:
        return p
    return _DEFAULT_RESULTS_CSV


# Per-paper instance fan-out — overridable via --instance_workers. Set from
# main() once argparse has run; _run_all_instances reads via getter so the
# value flows through all call sites without threading it through every
# function signature.
#
# Also seedable from the env var ``EFFICIENT_OR_INSTANCE_WORKERS`` at module
# import time. This lets parent processes (e.g. run_eval_modes.py) propagate
# the setting into framework subprocesses (OpenEvolve / CORAL) without each
# subprocess needing its own --instance-workers CLI plumbing.
_INSTANCE_WORKERS = 1
_env_iw = os.environ.get("EFFICIENT_OR_INSTANCE_WORKERS")
if _env_iw and _env_iw.strip().isdigit():
    _INSTANCE_WORKERS = max(1, int(_env_iw))


def get_instance_workers():
    return _INSTANCE_WORKERS


def set_instance_workers(n):
    global _INSTANCE_WORKERS
    _INSTANCE_WORKERS = max(1, int(n))


# Long-lived shared ThreadPoolExecutor. ``instance_workers`` semantics:
# the *global* concurrent-instance cap (across all papers/models in flight),
# not a per-paper fan-out. Lazily created on first use, sized once from
# ``_INSTANCE_WORKERS`` and never resized — the eval is started with a single
# CLI value, so a fixed-size pool is correct. Multiple ``_run_all_instances``
# calls (from concurrent ``process_paper_model`` threads) submit into the
# same pool, so a paper waiting on its own subset doesn't waste the slots
# other papers could fill.
_INSTANCE_POOL = None
_INSTANCE_POOL_LOCK = threading.Lock()


def _get_instance_pool():
    """Return the shared instance ThreadPoolExecutor, creating it on first
    use. Returns ``None`` when ``_INSTANCE_WORKERS <= 1`` so callers fall
    back to the sequential path."""
    global _INSTANCE_POOL
    if _INSTANCE_WORKERS <= 1:
        return None
    if _INSTANCE_POOL is not None:
        return _INSTANCE_POOL
    with _INSTANCE_POOL_LOCK:
        if _INSTANCE_POOL is None:
            _INSTANCE_POOL = ThreadPoolExecutor(
                max_workers=_INSTANCE_WORKERS,
                thread_name_prefix="instance-pool",
            )
    return _INSTANCE_POOL


def _shutdown_instance_pool():
    """Best-effort shutdown — main() registers this at exit so the pool's
    worker threads don't keep the interpreter alive on Ctrl-C."""
    global _INSTANCE_POOL
    pool = _INSTANCE_POOL
    if pool is not None:
        _INSTANCE_POOL = None
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python <3.9 lacks cancel_futures
            pool.shutdown(wait=False)


# Default paper sources when --paper_id is omitted:
#   * --paper-tag X selects every paper whose ``tag`` column equals X
#     (exact match) in gurobi_results_all_new.csv. Tags like "A,F" are excluded.
#   * Without --paper-tag, every paper directory under the data dir
#     (``frontier-or/`` by default) is used — full sweep.
_GUROBI_RESULTS_ALL = os.path.join(ROOT_DIR, "gurobi_results_all_new.csv")


# Per-model default results CSV. Used when --results_csv is omitted and a
# single model is selected: keeps each model's results in its own file so
# parallel runs don't fight for the same lock and rows stay grouped.
# Match by model short name (after stripping the "vendor/" prefix). Adding a
# new model? Drop a line here.
_MODEL_RESULTS_CSV = {
    "claude-opus-4.6":         "eval/eval_results_opus46.csv",
    "gemini-3.1-pro-preview":  "eval/eval_results_gemini31pro.csv",
    "gpt-5.3-codex":           "eval/eval_results_codex53.csv",
    "grok-4.20-beta":          "eval/eval_results_grok42beta.csv",
    "deepseek-r1":             "eval/eval_results_deepseekr1.csv",
    "qwen3-coder-plus":        "eval/eval_results_qwen3coder.csv",
    "llama-4-maverick":        "eval/eval_results_llama4maverick.csv",
}


def _default_results_csv_for_models(model_short_names):
    """Return the per-model default CSV path when exactly one mapped model is
    selected; otherwise None (caller should fall back to the global default)."""
    if not model_short_names or len(model_short_names) != 1:
        return None
    return _MODEL_RESULTS_CSV.get(model_short_names[0])


def _load_paper_ids_by_tag(tag):
    """Return sorted unique paper_ids whose ``tag`` column equals ``tag``
    (exact match) in gurobi_results_all_new.csv. Tags like ``A,F`` are excluded
    when ``tag='A'`` — supply the comma-form explicitly to match those."""
    if not os.path.exists(_GUROBI_RESULTS_ALL):
        raise FileNotFoundError(
            f"--paper-tag set but default source not found: {_GUROBI_RESULTS_ALL}"
        )
    out = set()
    with open(_GUROBI_RESULTS_ALL, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("tag") or "").strip() == tag:
                pid = (row.get("paper_id") or "").strip()
                if pid:
                    out.add(pid)
    return sorted(out)


def _load_all_paper_dirs():
    """Return sorted paper_ids by scanning the data dir for subdirectories.
    Used when both --paper_id and --paper-tag are omitted."""
    data_dir = get_data_dir()
    out = []
    for name in sorted(os.listdir(data_dir)):
        if name.startswith("__") or name.startswith("."):
            continue
        if os.path.isdir(os.path.join(data_dir, name)):
            out.append(name)
    return out


# Per-paper optimization direction ("min" or "max") from the metadata CSV.
# Used by compare_objectives and compute_aocc to avoid treating "LLM beats ref"
# as "LLM is further from ref".
_DIRECTION_META_PATH = os.path.join(ROOT_DIR, "results", "data_statistics", "paper_meta_info.csv")
_DIRECTIONS_CACHE = None


def _load_directions():
    """Load {paper_id: 'min'|'max'} from the direction registry CSV.

    Rows with a missing/malformed direction are deliberately *not* added to
    the dict, so that get_paper_direction() fails loud on them rather than
    silently guessing.
    """
    global _DIRECTIONS_CACHE
    if _DIRECTIONS_CACHE is not None:
        return _DIRECTIONS_CACHE
    out = {}
    if os.path.exists(_DIRECTION_META_PATH):
        with open(_DIRECTION_META_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = (row.get("paper_id") or "").strip()
                d = (row.get("direction") or "").strip().lower()
                if pid and d in ("min", "max"):
                    out[pid] = d
    _DIRECTIONS_CACHE = out
    return out


def get_paper_direction(paper_id):
    """Return 'min' or 'max' for a paper.

    The optimization direction is a correctness-critical fact about the
    problem, not a tunable knob: a wrong direction silently inverts every
    quality/QTE score, and for self-evolving frameworks it steers the search
    toward the *worst* solutions. There is therefore NO safe default --
    an unknown direction raises instead of guessing 'min'. Register the
    paper in ``results/data_statistics/paper_meta_info.csv`` before
    evaluating it.
    """
    d = _load_directions().get(paper_id)
    if d is None:
        if not os.path.exists(_DIRECTION_META_PATH):
            raise ValueError(
                f"Cannot resolve optimization direction for paper "
                f"{paper_id!r}: direction registry not found at "
                f"{_DIRECTION_META_PATH}."
            )
        raise ValueError(
            f"Cannot resolve optimization direction for paper {paper_id!r}: "
            f"it has no valid 'min'/'max' row in {_DIRECTION_META_PATH}. "
            f"Add one before evaluating -- a missing direction silently "
            f"inverts the paper's quality/QTE scores."
        )
    return d


def validate_paper_directions(paper_ids):
    """Preflight check: every paper in ``paper_ids`` has a registered,
    valid optimization direction.

    Raises ValueError listing *all* offenders at once, so a batch run fails
    upfront instead of part-way through (or, worse, silently). Call this
    before doing any evaluation work.
    """
    if not os.path.exists(_DIRECTION_META_PATH):
        raise ValueError(
            f"Direction registry not found at {_DIRECTION_META_PATH}; "
            f"cannot evaluate any paper without it."
        )
    directions = _load_directions()
    missing = sorted(set(paper_ids) - set(directions))
    if missing:
        raise ValueError(
            f"{len(missing)} paper(s) have no valid 'min'/'max' direction in "
            f"{_DIRECTION_META_PATH}: {missing}. Add them before evaluating "
            f"-- a missing direction silently inverts quality/QTE scores and "
            f"steers self-evolving search toward the worst solutions."
        )


def set_results_csv_path(path):
    """Called from main() when --results_csv is given. Sets the global
    override that wins over the per-model mapping. Accepts a relative path
    (resolved against CWD) or an absolute path."""
    global _RESULTS_CSV_OVERRIDE
    _RESULTS_CSV_OVERRIDE = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(_RESULTS_CSV_OVERRIDE), exist_ok=True)


def _resolve_csv_for_model(model_short):
    """Return the per-model CSV target. Used by ``process_paper_model`` to
    populate ``_results_csv_local.path`` when ``--results_csv`` was not
    explicitly given. Models without a mapping entry use the global default
    (``eval/eval_results.csv``)."""
    rel = _MODEL_RESULTS_CSV.get(model_short)
    if rel is None:
        return _DEFAULT_RESULTS_CSV
    target = os.path.abspath(os.path.join(ROOT_DIR, rel))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target


def configure_gurobi_license():
    """Set GRB_LICENSE_FILE from common local locations when not already set."""
    existing = os.environ.get("GRB_LICENSE_FILE")
    if existing:
        return existing
    candidates = [
        os.path.join(ROOT_DIR, "gurobi.lic"),
        os.path.expanduser("~/gurobi.lic"),
        "/home/chonghej/gurobi.lic",
        "/opt/gurobi/gurobi.lic",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            os.environ["GRB_LICENSE_FILE"] = candidate
            return candidate
    return None


def get_data_dir():
    """Return the root directory containing paper_id subdirectories."""
    override = os.environ.get("FRONTIER_OR_DATA_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(ROOT_DIR, "frontier-or")


TASK_SPECIFICATION = """\
## Your Task
Based on the problem description, instance data schema, and solution output schema above, write a complete Python program that solves this optimization problem as efficiently as possible, while aiming to achieve an objective value as close to optimal as you can.
The instance data schema (instance_schema.json) specifies the type, shape, and meaning of every field in the input JSON. Your program must be able to read and solve any instance file following this schema.
The solution output schema (solution_schema.json) specifies the type, key format, and meaning of every field your output JSON must contain. Your output MUST strictly follow this schema: include "objective_value" and all decision variable fields using the exact field names, key naming conventions, and nesting structure defined in the schema. 

### Implementation Requirements
1. Consider the tradeoff between solution quality against computational efficiency based on the nature of the problem. You can use any solution approach and any internal formulation you find effective.
2. The solution output schema describes the output format only. Your final output must project your solution onto the fields defined in solution_schema.json; any internal auxiliary variables should not appear in the output.
3. If you use a MIP/LP solver, use Gurobi (gurobipy), which is pre-installed in the execution environment. Do not use other solvers (CPLEX, SCIP, CBC, PuLP, OR-Tools CP-SAT, etc.).
4. Your program will run in a containerized environment restricted to a **single CPU core**.

### Time Limit Requirement
1. The program must enforce a maximum runtime via an `argparse` command-line argument `--time_limit` (type: int, seconds).
2. If the algorithm cannot find an optimal solution within this time limit, it must terminate and return the best feasible solution found so far.

### Convergence Logging
A utility module `solution_logger.py` is pre-installed alongside your program. Use it to record every incumbent (improved) solution found during the search. The required usage is:

```python
from solution_logger import SolutionLogger

# After parsing args, initialize the logger (use "minimize" or "maximize"):
logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

# Whenever a better feasible solution is found, call:
if logger:
    logger.log(objective_value)
```

Do NOT implement your own logging mechanism. Just call `logger.log(objective_value)` each time the algorithm finds a new best feasible solution.

### Output Format and Structure
1. The program MUST use `argparse` to define the following command-line arguments:
   - `--instance_path`: path to the JSON file containing the problem instance.
   - `--solution_path`: path where the final solution JSON file must be written.
   - `--time_limit`: maximum runtime in seconds (int).
   - `--log_path`: path to a JSONL file for logging intermediate solutions (optional).
2. The program MUST NOT hard-code or embed instance data. It must read all instance data from the file specified by `--instance_path`.
3. Output only the Python code, enclosed in a single ```python ... ``` block.
"""

def load_config():
    config_path = os.path.join(ROOT_DIR, "configs", "oneshot.yaml")
    keys_path = os.path.join(ROOT_DIR, "configs", "api_keys.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found. Create it with the 'models' list.")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    keys = {}
    if os.path.exists(keys_path):
        with open(keys_path, "r") as f:
            keys = yaml.safe_load(f) or {}
    api_key = os.environ.get("OPENROUTER_API_KEY") or keys.get("OPENROUTER_API_KEY_ONESHOT")
    if not api_key:
        print("ERROR: set env OPENROUTER_API_KEY or OPENROUTER_API_KEY_ONESHOT in configs/api_keys.yaml")
        sys.exit(1)
    config["OPENROUTER_API_KEY"] = api_key
    if not config.get("models"):
        if config.get("model"):
            config["models"] = [config["model"]]
        else:
            print("ERROR: 'models' list is empty in configs/oneshot.yaml")
            sys.exit(1)
    return config


def get_paper_dir(paper_id):
    return os.path.join(get_data_dir(), paper_id)


def get_eval_dir(paper_id):
    d = os.path.join(ROOT_DIR, "eval", "eval_papers", paper_id)
    os.makedirs(d, exist_ok=True)
    return d


def get_model_eval_dir(paper_id, model_name):
    d = os.path.join(ROOT_DIR, "eval", "eval_papers", paper_id, model_name)
    os.makedirs(d, exist_ok=True)
    return d


def get_model_code_dir(paper_id, model_name):
    """Per-(paper, model) directory to READ ``code.py`` from.

    Same as ``get_model_eval_dir`` unless ``--code-root`` overrides the base,
    in which case it points at the user-supplied tree (e.g. ``samples/oneshot_code``)
    while writes still target ``eval/eval_papers``.
    """
    base = CODE_ROOT if CODE_ROOT else os.path.join(ROOT_DIR, "eval", "eval_papers")
    return os.path.join(base, paper_id, model_name)


def read_problem_description(paper_id):
    """Read the canonical problem_description.txt for the paper."""
    paper_dir = get_paper_dir(paper_id)
    path = os.path.join(paper_dir, "problem_description.txt")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path, "r") as f:
        return f.read().strip()


def read_instance_template(paper_id):
    """Read instance_schema.json as the data schema specification."""
    path = os.path.join(get_paper_dir(paper_id), "instance_schema.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        content = f.read().strip()
    return f"# Instance Data Schema (instance_schema.json)\n\n{content}"



def read_solution_template(paper_id):
    """Read solution_schema.json as the output format specification."""
    path = os.path.join(get_paper_dir(paper_id), "solution_schema.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        content = f.read().strip()
    return f"# Solution Output Schema (solution_schema.json)\n\n{content}"



def build_prompt(problem_description, instance_template, solution_template):
    """Assemble full prompt: problem_description + instance schema + solution schema + task spec."""
    parts = [problem_description, instance_template, solution_template, TASK_SPECIFICATION]
    return "\n\n".join(parts)


def extract_python_code(text):
    """Extract the first ```python ... ``` code block from LLM response.
    Returns None if text is empty/None or no code block is found."""
    if not text:
        return None
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def load_model_pricing():
    """Load model pricing from model_pricing.json."""
    pricing_path = os.path.join(ROOT_DIR, "model_pricing.json")
    if not os.path.exists(pricing_path):
        return {}
    with open(pricing_path, "r") as f:
        data = json.load(f)
    return data.get("models", {})


MODEL_PRICING = load_model_pricing()


def _resolve_openrouter_api_key(config, model):
    return config["OPENROUTER_API_KEY"]


def call_openrouter(messages, config, model, temperature=None):
    """Call OpenRouter chat completions API.
    Returns (content, usage_dict) where usage_dict has prompt_tokens, completion_tokens.

    Raises ValueError if the response carries no usable text (neither 'content'
    nor 'reasoning'), so callers can catch and retry. Reasoning models
    (e.g. deepseek-r1) often return 'content': null with the real text in the
    'reasoning' field; we fall back to that automatically.

    NOTE: ``max_tokens`` is intentionally NOT set. Letting the provider choose
    its default avoids capping reasoning-model budgets too low (gemini
    breakage mode) while the ``content`` → ``reasoning`` fallback already
    handles providers that deliver nothing in ``content``."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {_resolve_openrouter_api_key(config, model)}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    if resp.status_code >= 400:
        print(f"  [call_openrouter] HTTP {resp.status_code} body: {resp.text[:800]}")
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if not content:
        reasoning = msg.get("reasoning")
        if reasoning:
            print(f"  [warning] {model} returned null/empty content; "
                  f"using 'reasoning' field ({len(reasoning)} chars) as fallback")
            content = reasoning
    if not content:
        finish_reason = data["choices"][0].get("finish_reason")
        raise ValueError(
            f"Empty response from {model} "
            f"(finish_reason={finish_reason!r}, no content or reasoning text)"
        )
    usage = data.get("usage", {})
    cached_tokens = usage.get("cached_tokens", 0) or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return content, {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": cached_tokens,
    }


def run_feasibility_check(paper_id, instance_path, solution_path, result_path):
    """Run the paper's feasibility_check.py on a solution file.
    Returns (feasible, reason, error). reason/error are set when feasible is None."""
    checker_path = os.path.join(get_paper_dir(paper_id), "feasibility_check.py")
    if not os.path.exists(checker_path):
        return None, "checker_unavailable", "Feasibility checker not found"
    if not os.path.exists(solution_path):
        return None, "checker_error", "Solution file not found for feasibility check"
    try:
        result = subprocess.run(
            [sys.executable, checker_path,
             "--instance_path", instance_path,
             "--solution_path", solution_path,
             "--result_path", result_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"    Feasibility check failed: {result.stderr[:200]}")
            return None, "checker_error", result.stderr.strip() or result.stdout.strip()
        with open(result_path, "r") as f:
            data = json.load(f)
        return data.get("feasible"), None, None
    except Exception as e:
        print(f"    Feasibility check error: {e}")
        return None, "checker_error", str(e)


sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "utils"))
from exec_backends import BACKENDS as EXEC_BACKENDS
from instance_paths import (
    DEFAULT_INSTANCES,
    instance_path as _instance_path,
    gurobi_solution_path as _gurobi_solution_path,
    parse_instances_arg,
)


def run_generated_code(code_path, solution_path, instance_path, time_limit,
                       log_path=None, exec_mode="bare", exec_cfg=None):
    """Run generated Python code via the selected execution backend.
    Returns (success, output/error, elapsed_seconds)."""
    backend = EXEC_BACKENDS[exec_mode]
    return backend(code_path, instance_path, solution_path, time_limit,
                   log_path=log_path, cfg=exec_cfg)


def parse_t_max(v):
    """Argparse type for ``--t_max``. Accepts a positive float or the literal
    string ``"gurobi"`` (sentinel: use each instance's Gurobi solve time as
    the AOCC horizon, resolved later at the per-instance call site)."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() == "gurobi":
        return "gurobi"
    return float(v)


def _resolve_t_max(t_max, paper_id, idx):
    """Translate the ``t_max`` argument to a concrete horizon for one instance.

    - ``None`` → None (caller falls back to elapsed wall time)
    - float    → that value (global override)
    - "gurobi" → per-instance Gurobi solve time from ``gurobi_results_*.csv``;
                 falls back to None if missing for this instance
    """
    if t_max is None or isinstance(t_max, (int, float)):
        return t_max
    if isinstance(t_max, str) and t_max.lower() == "gurobi":
        gd = load_gurobi_csv_data(paper_id)
        return (gd.get(idx) or {}).get("time")
    return None


def compute_aocc(log_path, gurobi_obj, time_limit, t_max=None, direction="min"):
    """
    Compute normalized AOCC from a convergence log file.

    Direction-aware: for ``direction="min"`` (the default), lower obj is better,
    so a log entry with ``obj <= gurobi_obj`` contributes gap=0 to the integral.
    For ``direction="max"`` the convention flips (higher obj is better).

    Args:
        log_path: Path to JSONL file with {"time": ..., "objective_value": ...} entries.
        gurobi_obj: Reference objective value (from Gurobi).
        time_limit: Actual time limit used for execution.
        t_max: Optional custom time horizon for AOCC computation.
        direction: "min" or "max"; optimization sense of the problem.

    Returns:
        aocc (float in [0,1]) or None if log is missing/empty.
    """
    if gurobi_obj is None:
        return None
    if not log_path or not os.path.exists(log_path):
        return None

    # Read log entries. We only need numeric ``time`` (seconds-since-start)
    # plus ``objective_value``. LLM-generated solvers occasionally name the
    # field differently or write ISO timestamps — accept the common variants
    # (``time`` / ``elapsed`` / ``elapsed_time`` / ``solve_time``) and silently
    # drop any entry that's still unparseable, instead of letting a single
    # malformed line crash the whole evaluation. Same for ``objective_value``
    # / ``obj`` / ``best_obj``.
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(d, dict):
                continue
            t = d.get("time")
            if t is None:
                t = d.get("elapsed", d.get("elapsed_time", d.get("solve_time")))
            try:
                t = float(t)
            except (TypeError, ValueError):
                continue
            obj = d.get("objective_value", d.get("obj", d.get("best_obj")))
            try:
                obj = float(obj)
            except (TypeError, ValueError):
                continue
            entries.append({"time": t, "objective_value": obj})
    if not entries:
        return 1.0  # no solution found (or log unparseable) → worst case

    # Direction-aware gap function: signed relative gap, clipped to [0, 1].
    # gap = 0 iff the incumbent matches or beats the reference.
    denom = max(abs(gurobi_obj), 1e-10)
    if direction == "max":
        def gap_fn(obj):
            return max(0.0, min(1.0, (gurobi_obj - obj) / denom))
    else:  # "min"
        def gap_fn(obj):
            return max(0.0, min(1.0, (obj - gurobi_obj) / denom))

    # Time horizon
    T = t_max if t_max is not None else time_limit

    # Build step function: [(time, gap), ...]
    # Sort by time and keep only monotonically improving entries
    steps = [(0.0, 1.0)]  # before first solution: 100% gap
    best_gap = 1.0
    for entry in sorted(entries, key=lambda e: e["time"]):
        t = entry["time"]
        if t > T:
            break
        g = gap_fn(entry["objective_value"])
        if g < best_gap - 1e-12:
            best_gap = g
            steps.append((t, g))
    steps.append((T, steps[-1][1]))  # extend to horizon

    # Integrate rectangles
    aocc = 0.0
    for i in range(len(steps) - 1):
        dt = steps[i + 1][0] - steps[i][0]
        aocc += steps[i][1] * dt

    return round(aocc / T, 6) if T > 0 else None


def compare_objectives(llm_solution_path, gurobi_solution_path, direction="min"):
    """
    Compare objective values. Returns (match, llm_obj, gurobi_obj, gap, error_msg).

    Direction-aware non-negative gap: 0 iff LLM matches or beats Gurobi,
    positive = how much worse the LLM is (fraction of |gurobi_obj|).
    For ``direction="min"``, gap = max(0, (llm - gurobi) / |gurobi|).
    For ``direction="max"``, gap = max(0, (gurobi - llm) / |gurobi|).

    match=True if gap <= GAP_TOLERANCE (10%).
    """
    GAP_TOLERANCE = 0.10  # 10%

    if not os.path.exists(llm_solution_path):
        return False, None, read_gurobi_obj(gurobi_solution_path), None, "LLM solution file not found"
    if not os.path.exists(gurobi_solution_path):
        # LLM file is present and not yet parsed; try to salvage its obj so
        # the caller still sees what the LLM produced.
        return False, read_gurobi_obj(llm_solution_path), None, None, "Gurobi solution file not found"

    try:
        with open(llm_solution_path, "r") as f:
            llm_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, None, read_gurobi_obj(gurobi_solution_path), None, f"Invalid solution JSON: {e}"
    try:
        with open(gurobi_solution_path, "r") as f:
            gurobi_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Gurobi side broke (e.g. LFS pointer stub) but llm_data is already
        # loaded — preserve the LLM's obj instead of discarding it.
        return False, _extract_obj_from_dict(llm_data), None, None, f"Invalid Gurobi solution JSON: {e}"

    if not isinstance(llm_data, dict):
        return False, None, read_gurobi_obj(gurobi_solution_path), None, "Invalid solution JSON: top-level value must be an object"
    if not isinstance(gurobi_data, dict):
        return False, _extract_obj_from_dict(llm_data), None, None, "Invalid Gurobi solution JSON: top-level value must be an object"

    llm_obj = llm_data.get("objective_value")
    gurobi_obj = gurobi_data.get("objective_value")

    if llm_obj is None:
        return False, None, gurobi_obj, None, "LLM solution missing 'objective_value'"
    if gurobi_obj is None:
        return False, llm_obj, None, None, "Gurobi solution missing 'objective_value'"

    try:
        llm_obj = float(llm_obj)
    except (ValueError, TypeError):
        return False, None, gurobi_obj, None, f"LLM solution has non-numeric 'objective_value': {llm_obj!r}"
    try:
        gurobi_obj = float(gurobi_obj)
    except (ValueError, TypeError):
        return False, llm_obj, None, None, f"Gurobi solution has non-numeric 'objective_value': {gurobi_obj!r}"

    abs_err = abs(llm_obj - gurobi_obj)
    if abs_err < 1e-6:
        return True, llm_obj, gurobi_obj, 0.0, None

    denom = max(abs(gurobi_obj), 1e-10)
    if direction == "max":
        signed = (gurobi_obj - llm_obj) / denom
    else:  # "min"
        signed = (llm_obj - gurobi_obj) / denom
    gap = max(0.0, signed)  # 0 iff LLM matches or beats gurobi

    if gap <= GAP_TOLERANCE:
        return True, llm_obj, gurobi_obj, gap, None

    return False, llm_obj, gurobi_obj, gap, f"Gap {gap:.2%} exceeds {GAP_TOLERANCE:.0%} tolerance"


# Sanity bounds on a candidate program's self-reported ``objective_value``.
# Programs occasionally try to game the scorer by returning sentinel values
# (``inf``, ``nan``, ``sys.float_info.max``) or absurdly inflated objectives
# that produce gigantic signed-gap "beats" against the Gurobi reference. We
# reject anything that is obviously not a real objective; the run treats the
# attempt as infeasible (fail_reason="invalid_obj") instead of crediting it.
_OBJ_ABS_CAP = 1.0e12         # any |obj| above this is implausible for OR problems
_OBJ_REL_CAP = 1.0e6          # |obj| can't legitimately be >1e6x the reference
_OBJ_REL_REF_FLOOR = 1.0e-3   # below this, |ref| is "near zero" — skip relative check


def obj_is_sane(obj, ref=None):
    """True iff ``obj`` is a plausible objective value; rejects sentinel /
    overflow / "obj is 1e6x the reference" exploits.

    Boundary tuning:
      - ``inf`` / ``nan`` / ``None``: rejected (trivial).
      - ``|obj| > 1e12``: rejected — no real OR objective is this large; this
        catches ``sys.float_info.max`` (1.8e308) and similar overflow sentinels.
      - ``|obj| > 1e6 * |ref|`` (when ``|ref|`` is non-negligible): rejected
        — closes the loophole where a program returns a "merely huge" obj
        (e.g. 1e11) that escapes the absolute cap but still produces a
        signed-gap of ~1e9 against a small reference.
    """
    if obj is None:
        return False
    try:
        x = float(obj)
    except (TypeError, ValueError):
        return False
    if math.isnan(x) or math.isinf(x):
        return False
    if abs(x) > _OBJ_ABS_CAP:
        return False
    if ref is not None:
        try:
            r = abs(float(ref))
        except (TypeError, ValueError):
            r = 0.0
        if r > _OBJ_REL_REF_FLOOR and abs(x) > _OBJ_REL_CAP * r:
            return False
    return True


def read_gurobi_obj(gurobi_solution_path):
    """Read objective_value from a solution JSON file. Returns float or None.
    Generic — works for both Gurobi reference and LLM solution files."""
    if not os.path.exists(gurobi_solution_path):
        return None
    try:
        with open(gurobi_solution_path, "r") as f:
            data = json.load(f)
        val = data.get("objective_value")
        return float(val) if val is not None else None
    except Exception:
        return None


def _extract_obj_from_dict(data):
    """Pull objective_value out of an already-parsed JSON dict. Returns float or None."""
    if not isinstance(data, dict):
        return None
    val = data.get("objective_value")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def get_model_short_name(model_str):
    """Extract a short filesystem-safe name from the model identifier."""
    # e.g. "google/gemini-2.5-pro-preview-05-06" -> "gemini-2.5-pro-preview-05-06"
    name = model_str.split("/")[-1]
    # sanitize for filenames
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name


def run_and_evaluate_instance(paper_id, model_name, idx, code_path,
                              time_limit, exec_mode, exec_cfg, t_max,
                              output_dir=None):
    """Run code on a single instance and evaluate the result.
    Returns (result_dict, error_summary_or_None)."""
    model_dir = output_dir or get_model_eval_dir(paper_id, model_name)
    os.makedirs(model_dir, exist_ok=True)
    paper_dir = get_paper_dir(paper_id)
    direction = get_paper_direction(paper_id)
    # `idx` here is a categorical instance name (e.g., "tiny", "large_1").
    # Model-output files are keyed by name: solution_tiny.json, log_large_1.jsonl, etc.
    solution_path = os.path.join(model_dir, f"solution_{idx}.json")
    log_path = os.path.join(model_dir, f"log_{idx}.jsonl")
    instance_path = _instance_path(paper_dir, idx)
    gurobi_solution_path = _gurobi_solution_path(paper_dir, idx)

    if not os.path.exists(instance_path):
        print(f"  Instance {idx}: SKIP — {instance_path} not found")
        return {
            "status": "fail", "fail_reason": "missing_instance", "retries": 0,
            "llm_obj": None,
            "gurobi_obj": read_gurobi_obj(gurobi_solution_path),
            "solve_time": None, "feasible": None, "gap": None,
            "aocc": None, "error": "Instance file not found",
        }, None  # not a code error, no point in self-correction

    if os.path.exists(solution_path):
        os.remove(solution_path)

    success, output, elapsed = run_generated_code(
        code_path, solution_path, instance_path, time_limit, log_path,
        exec_mode=exec_mode, exec_cfg=exec_cfg
    )

    if not success:
        print(f"  Instance {idx}: FAIL runtime_error ({elapsed}s)")
        res = {
            "status": "fail", "fail_reason": "runtime_error", "retries": 0,
            "llm_obj": None,
            "gurobi_obj": read_gurobi_obj(gurobi_solution_path),
            "solve_time": elapsed, "feasible": None, "gap": None,
            "aocc": 1.0, "error": output,
        }
        return res, f"Instance {idx}: execution failed:\n{output[:300]}"

    match, llm_obj, gurobi_obj, gap, err_msg = compare_objectives(
        solution_path, gurobi_solution_path, direction=direction
    )

    invalid_prefixes = (
        "LLM solution file not found",
        "Invalid solution JSON",
        "LLM solution missing",
        "LLM solution has non-numeric",
    )
    if err_msg and err_msg.startswith(invalid_prefixes):
        print(f"  Instance {idx}: FAIL invalid_solution ({err_msg[:120]})")
        res = {
            "status": "fail", "fail_reason": "invalid_solution", "retries": 0,
            "llm_obj": llm_obj, "gurobi_obj": gurobi_obj,
            "solve_time": elapsed, "feasible": None, "gap": gap,
            "aocc": 1.0, "error": err_msg,
        }
        return res, f"Instance {idx}: invalid solution output:\n{err_msg[:300]}"

    # Reject objectives that are obviously fabricated (inf / nan / overflow
    # sentinels / >1e6x the Gurobi reference). Without this, a program that
    # returns ``sys.float_info.max`` as its objective sails past feasibility
    # (the constraints are satisfied) and produces a gigantic synthetic "beat"
    # in the staged_qte signed_gap. Treat as invalid -- no score credit.
    if not obj_is_sane(llm_obj, gurobi_obj):
        err = f"Implausible objective reported by program: llm_obj={llm_obj!r}, gurobi_obj={gurobi_obj!r}"
        print(f"  Instance {idx}: FAIL invalid_obj ({err[:140]})")
        res = {
            "status": "fail", "fail_reason": "invalid_obj", "retries": 0,
            "llm_obj": llm_obj, "gurobi_obj": gurobi_obj,
            "solve_time": elapsed, "feasible": False, "gap": None,
            "aocc": 1.0, "error": err,
        }
        return res, f"Instance {idx}: {err}"

    feasi_result_path = os.path.join(model_dir, f"feasi_result_{idx}.json")
    feasible, checker_reason, checker_error = run_feasibility_check(
        paper_id, instance_path, solution_path, feasi_result_path
    )
    feasi_str = str(feasible) if feasible is not None else "N/A"

    aocc_t_max = _resolve_t_max(t_max, paper_id, idx)
    aocc = compute_aocc(log_path, gurobi_obj, elapsed, t_max=aocc_t_max, direction=direction)

    gap_str = f"{gap:.2%}" if gap is not None else "N/A"
    # Classification rule:
    #   feasible=False              -> fail, fail_reason=infeasible    (triggers correction)
    #   feasible=True + gap>10%     -> fail, fail_reason=gap_exceeds   (quality threshold)
    #   feasible=True + gap<=10%    -> pass
    # Phase 1 gate applies a separate gap<=10% check for tiny.
    FAIL_GAP_THRESHOLD = 0.10
    if feasible is False:
        status = "fail"
        fail_reason = "infeasible"
        feasi_detail = ""
        if os.path.exists(feasi_result_path):
            try:
                with open(feasi_result_path, "r") as ff:
                    feasi_data = json.load(ff)
                violations = feasi_data.get("violations", [])
                if violations:
                    feasi_detail = " Violations: " + "; ".join(
                        str(v)[:200] for v in violations[:5]
                    )
            except Exception:
                pass
        error_msg = "Solution is INFEASIBLE" + (feasi_detail if feasi_detail else "")
        error_summary = (
            f"Instance {idx}: INFEASIBLE.{feasi_detail} "
            f"(obj={llm_obj}, gap={gap_str})"
        )
        print(f"  Instance {idx}: FAIL infeasible (LLM={llm_obj}, gap={gap_str})")
    elif feasible is None:
        status = "fail"
        fail_reason = checker_reason or "checker_error"
        error_msg = checker_error or "Feasibility checker returned no result"
        error_summary = None
        print(f"  Instance {idx}: FAIL {fail_reason} (LLM={llm_obj}, gap={gap_str})")
    elif gap is not None and gap > FAIL_GAP_THRESHOLD:
        # feasible but gap blows past the quality threshold.
        status = "fail"
        fail_reason = "gap_exceeds"
        error_msg = (f"Gap {gap_str} exceeds {FAIL_GAP_THRESHOLD:.0%} threshold "
                     f"(obj={llm_obj}, gurobi={gurobi_obj})")
        error_summary = None
        print(f"  Instance {idx}: FAIL gap_exceeds (LLM={llm_obj}, gap={gap_str})")
    else:
        # feasible is True and gap within threshold (or unknown): treat as pass.
        status = "pass"
        fail_reason = None
        error_msg = err_msg  # informational (e.g., "gap exceeds 10% tolerance" note)
        error_summary = None
        gap_disp = f", gap={gap_str}"
        print(f"  Instance {idx}: PASS (LLM={llm_obj}{gap_disp}, time={elapsed}s, feasible={feasi_str})")

    res = {
        "status": status, "fail_reason": fail_reason, "retries": 0,
        "llm_obj": llm_obj, "gurobi_obj": gurobi_obj,
        "solve_time": elapsed, "feasible": feasible, "gap": gap,
        "aocc": aocc, "error": error_msg,
    }
    return res, error_summary


def _resolve_time_limit(time_limit, idx, default=300):
    """Resolve time_limit for a single instance.

    ``time_limit`` may be:
      - int   — same budget for every instance
      - dict  — per-instance override, falling back to ``default`` when ``idx`` missing
    """
    if isinstance(time_limit, dict):
        v = time_limit.get(idx)
        return int(v) if v is not None else int(default)
    return int(time_limit)


def _run_all_instances(paper_id, model_name, instance_indices, code_path,
                       time_limit, exec_mode, exec_cfg, t_max,
                       output_dir=None):
    """Run code against all instances, collect results and errors.

    ``time_limit`` may be an ``int`` (uniform) or a ``dict[str, int]`` for
    per-instance budgets (missing keys fall back to the int value; if no
    fallback is provided the default is 300s).
    """
    results = {}
    errors = []
    # Determine per-instance fallback budget: if a dict is given, use its own
    # int-valued default (not typical); otherwise the int is the default.
    default_budget = time_limit if isinstance(time_limit, int) else 300

    instance_indices = list(instance_indices)

    def _one(idx):
        per_tl = _resolve_time_limit(time_limit, idx, default=default_budget)
        return idx, run_and_evaluate_instance(
            paper_id, model_name, idx, code_path,
            per_tl, exec_mode, exec_cfg, t_max,
            output_dir=output_dir,
        )

    # Submit to the shared pool when ``instance_workers > 1``. The pool is
    # global, so other concurrent _run_all_instances calls (from other
    # paper/model threads) interleave their submissions; here we only wait
    # on our own futures. When ``instance_workers <= 1`` (or only one
    # instance to run), the per-thread overhead isn't worth it — run inline.
    pool = _get_instance_pool()
    if pool is None or len(instance_indices) <= 1:
        for idx in instance_indices:
            _, (res, error_summary) = _one(idx)
            results[idx] = res
            if error_summary:
                errors.append(error_summary)
        return results, errors

    futures = {pool.submit(_one, idx): idx for idx in instance_indices}
    for fut in as_completed(futures):
        idx, (res, error_summary) = fut.result()
        results[idx] = res
        if error_summary:
            errors.append(error_summary)
    return results, errors


def evaluate_candidate_code(paper_id, model_name, instance_indices, code_path,
                            output_dir, time_limit, exec_mode, exec_cfg,
                            t_max=None):
    """Run a candidate program on instances without writing eval_results.csv."""
    results, _ = _run_all_instances(
        paper_id, model_name, instance_indices, code_path,
        time_limit, exec_mode, exec_cfg, t_max,
        output_dir=output_dir,
    )
    return results


def _copy_results(results):
    """Create a shallow copy of result dicts keyed by instance index."""
    return {idx: dict(res) for idx, res in results.items()}


def _generate_initial_code(prompt, config, model, code_path, attempt0_path,
                           init_gen_max=6):
    """Generate code_v0 with an independent small retry budget (not counted
    against the shared self-correction budget). Returns (code_str, token_usage)
    or (None, token_usage) if all attempts failed to produce a code block.

    Transient failures (API error or missing code block) back off before the
    next attempt so a short provider flake doesn't nuke all 25 papers in a
    burst of 400s.
    """
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    messages = [{"role": "user", "content": prompt}]
    for init_attempt in range(init_gen_max):
        print(f"\n  [init-gen {init_attempt + 1}/{init_gen_max}] Calling LLM to generate code_v0...")
        transient_err = None
        try:
            assistant_reply, usage = call_openrouter(messages, config, model)
            token_usage["prompt_tokens"] += usage["prompt_tokens"]
            token_usage["completion_tokens"] += usage["completion_tokens"]
            token_usage["cached_tokens"] += usage.get("cached_tokens", 0)
        except Exception as e:
            transient_err = f"LLM API error: {e}"
        else:
            code = extract_python_code(assistant_reply)
            if code is None:
                transient_err = "No code block in response"
            else:
                with open(attempt0_path, "w") as f:
                    f.write(code)
                with open(code_path, "w") as f:
                    f.write(code)
                print(f"  [init-gen] code_v0 saved to {code_path}")
                return code, token_usage

        # Transient failure: back off (10, 20, 40, ... capped at 120s) with
        # jitter, unless this was the final attempt.
        print(f"  [init-gen {init_attempt + 1}] {transient_err}")
        if init_attempt < init_gen_max - 1:
            delay = min(120, 10 * (2 ** init_attempt)) + random.uniform(0, 5)
            print(f"  [init-gen] backing off {delay:.1f}s before retry...")
            time.sleep(delay)
    print(f"  [init-gen] exhausted {init_gen_max} attempts without a valid code block.")
    return None, token_usage


def generate_candidate_code(prompt, config, model, output_dir, candidate_id,
                            init_gen_max=3, temperature=None):
    """Generate one independent candidate program into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    code_path = os.path.join(output_dir, "code.py")
    attempt0_path = os.path.join(output_dir, "code_attempt0.py")
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(init_gen_max):
        try:
            assistant_reply, usage = call_openrouter(
                messages, config, model, temperature=temperature
            )
            token_usage["prompt_tokens"] += usage["prompt_tokens"]
            token_usage["completion_tokens"] += usage["completion_tokens"]
            token_usage["cached_tokens"] += usage.get("cached_tokens", 0)
        except Exception as e:
            last_error = f"LLM API error: {e}"
            print(f"  [candidate {candidate_id} gen {attempt + 1}/{init_gen_max}] {last_error}")
            continue

        code = extract_python_code(assistant_reply)
        if code is None:
            last_error = "No Python code block in response"
            print(f"  [candidate {candidate_id} gen {attempt + 1}/{init_gen_max}] {last_error}")
            continue

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        with open(attempt0_path, "w", encoding="utf-8") as f:
            f.write(code)
        return {
            "status": "ok",
            "code_path": code_path,
            "attempt0_path": attempt0_path,
            "usage": token_usage,
            "error": None,
        }

    return {
        "status": "fail",
        "code_path": None,
        "attempt0_path": None,
        "usage": token_usage,
        "error": locals().get("last_error", "LLM failed to produce a code block"),
    }


def _self_correct_once(prompt, cur_code, errors_str, config, model,
                       code_path, attempt_path):
    """One LLM self-correction call. Returns (new_code_str, token_usage) or
    (None, token_usage) if the LLM call failed or no code block was produced.
    Consumes one unit of the shared self-correction budget (accounted by the caller).
    """
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    feedback = (
        f"Your previous code produced the following issues:\n\n"
        f"{errors_str}\n\n"
        f"Please fix the code and return the corrected version "
        f"in a single ```python ... ``` block."
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"```python\n{cur_code}\n```"},
        {"role": "user", "content": feedback},
    ]
    try:
        assistant_reply, usage = call_openrouter(messages, config, model)
        token_usage["prompt_tokens"] += usage["prompt_tokens"]
        token_usage["completion_tokens"] += usage["completion_tokens"]
        token_usage["cached_tokens"] += usage.get("cached_tokens", 0)
    except Exception as e:
        print(f"  self-correction LLM API error: {e}")
        return None, token_usage
    new_code = extract_python_code(assistant_reply)
    if new_code is None:
        print(f"  self-correction: no code block in response")
        return None, token_usage
    with open(attempt_path, "w") as f:
        f.write(new_code)
    with open(code_path, "w") as f:
        f.write(new_code)
    return new_code, token_usage


def _is_timeout_error(res):
    """True iff the result is a runtime_error caused by --time_limit timeout
    (as opposed to a crash, exception, import error, or non-zero exit)."""
    if res.get("fail_reason") != "runtime_error":
        return False
    err = res.get("error") or ""
    return err.startswith("Execution timed out")


def _collect_debug_errors(results, indices):
    """Return error summaries for instances with a non-timeout runtime_error.
    These trigger the self-debug loop (separate --max_debug_retries budget)."""
    errors = []
    for idx in indices:
        r = results.get(idx, {})
        if r.get("fail_reason") != "runtime_error":
            continue
        if _is_timeout_error(r):
            continue
        err_text = r.get("error") or ""
        errors.append(f"Instance {idx}: {err_text[:500]}")
    return errors


def _collect_correction_errors(results, indices):
    """Return error summaries for instances that should trigger self-correction
    via the --max_correct_retries budget: infeasible, invalid_solution, or timeout
    runtime_error. Non-timeout runtime_errors are handled by the separate
    self-debug budget (see _collect_debug_errors). gap_exceeds is NOT a trigger."""
    errors = []
    for idx in indices:
        r = results.get(idx, {})
        fr = r.get("fail_reason")
        if fr == "runtime_error":
            if not _is_timeout_error(r):
                continue
        elif fr not in ("infeasible", "invalid_solution"):
            continue
        err_text = r.get("error") or ""
        errors.append(f"Instance {idx}: {err_text[:500]}")
    return errors


def _tiny_gate_passed(res):
    """Phase 1 gate criterion: feasible AND gap <= 10%."""
    if res.get("feasible") is not True:
        return False
    gap = res.get("gap")
    return gap is not None and gap <= 0.10


def _tiny_gate_correctable(res):
    """Whether the tiny gate failure can plausibly be fixed by changing solver code."""
    if res.get("fail_reason") in ("checker_unavailable", "checker_error", "missing_instance"):
        return False
    return True


def _tiny_gate_error_summary(tiny_idx, res):
    """Build an error summary to feed LLM when tiny fails the gate. Covers all
    three gate-failing reasons: infeasible / runtime_error / gap>10%."""
    fr = res.get("fail_reason")
    if fr == "runtime_error":
        return f"Instance {tiny_idx}: execution failed:\n{(res.get('error') or '')[:500]}"
    if fr == "invalid_solution":
        return f"Instance {tiny_idx}: invalid solution output:\n{(res.get('error') or '')[:500]}"
    if fr == "infeasible":
        return f"Instance {tiny_idx}: INFEASIBLE. {(res.get('error') or '')[:500]}"
    # gap>10% case
    gap = res.get("gap")
    gap_str = f"{gap:.2%}" if gap is not None else "N/A"
    return (
        f"Instance {tiny_idx}: gap={gap_str} exceeds 10% tolerance "
        f"(obj={res.get('llm_obj')}, gurobi={res.get('gurobi_obj')})"
    )



def _count_disk_attempts(model_dir):
    """Count historical LLM corrections from ``code_attempt*.py`` files in
    a model dir. Returns ``max(0, N - 1)`` where N is the file count —
    ``code_attempt0.py`` is the initial v0, every subsequent file is one
    correction (self-debug or self-correction)."""
    pattern = os.path.join(model_dir, "code_attempt*.py")
    n = len(glob.glob(pattern))
    return max(0, n - 1)


def _get_csv_done_instances(paper_id, model_name):
    """Return the set of instance names that already have a row in
    eval_results.csv for this (paper_id, model_name) pair. Used by
    ``--reuse-code incomplete`` to skip already-recorded pairs."""
    csv_path = get_results_csv_path()
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("paper_id") != paper_id or row.get("model") != model_name:
                continue
            inst = (row.get("instance") or "").strip()
            if inst:
                done.add(inst)
    return done


def _read_prev_result_rows(paper_id, model_name):
    """Read full current-run fields for each instance of a (paper, model)
    pair from eval_results.csv, shaped like the result dicts built by
    ``run_and_evaluate_instance``. Used under ``--reuse-code incomplete``
    to pre-populate ``results[idx]`` for instances already in CSV so Phase
    1 gate / print_summary can reason about them without re-running."""
    csv_path = get_results_csv_path()
    out = {}
    if not os.path.exists(csv_path):
        return out

    def _to_float(v):
        if v in (None, "", "None"):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    def _to_bool(v):
        if v == "True":
            return True
        if v == "False":
            return False
        return None

    def _to_int(v):
        if v in (None, "", "None"):
            return 0
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("paper_id") != paper_id or row.get("model") != model_name:
                continue
            inst = (row.get("instance") or "").strip()
            if not inst:
                continue
            debug_r = _to_int(row.get("debug_retries"))
            corr_r = _to_int(row.get("correction_retries"))
            out[inst] = {
                "status": row.get("status") or None,
                "fail_reason": row.get("fail_reason") or None,
                "error": row.get("error") or None,
                "llm_obj": _to_float(row.get("obj")),
                "gurobi_obj": None,
                "solve_time": _to_float(row.get("time")),
                "feasible": _to_bool(row.get("feasible")),
                "gap": _to_float(row.get("gap")),
                "aocc": _to_float(row.get("aocc")),
                "retries": debug_r + corr_r,
                "debug_retries": debug_r,
                "correction_retries": corr_r,
            }
    return out


def _read_prev_first_results(paper_id, model_name):
    """Recover historical ``first_*`` fields for each instance of a
    (paper, model) pair from eval_results.csv. Used under ``--reuse-code``
    to preserve the original v0 results in the CSV (the actual v0 code has
    been overwritten by subsequent self-corrections and is no longer
    runnable, so we cannot regenerate these values — they must be read
    from the CSV).

    Returns ``{instance_name: result_dict}`` where each dict shape matches
    what ``write_result_row`` expects from ``first_res``."""
    csv_path = get_results_csv_path()
    out = {}
    if not os.path.exists(csv_path):
        return out

    def _to_float(v):
        if v in (None, "", "None"):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    def _to_bool(v):
        if v == "True":
            return True
        if v == "False":
            return False
        return None

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("paper_id") != paper_id or row.get("model") != model_name:
                continue
            inst = (row.get("instance") or "").strip()
            if not inst:
                continue
            out[inst] = {
                "status": row.get("first_status") or None,
                "fail_reason": row.get("first_fail_reason") or None,
                "feasible": _to_bool(row.get("first_feasible")),
                "solve_time": _to_float(row.get("first_time")),
                "llm_obj": _to_float(row.get("first_obj")),
            }
    return out


def run_instances_with_existing_code(paper_id, instance_indices, model,
                                     time_limit, exec_mode="bare",
                                     exec_cfg=None, t_max=None,
                                     code_filename="code.py"):
    """
    Run existing generated code against specified instances without calling LLM.
    Returns (results, token_usage) where token_usage is always zero.
    """
    model_name = get_model_short_name(model)
    model_dir = get_model_eval_dir(paper_id, model_name)
    code_path = os.path.join(get_model_code_dir(paper_id, model_name), code_filename)

    if not os.path.exists(code_path):
        print(f"  ERROR: code not found at {code_path}")
        results = {}
        for idx in instance_indices:
            results[idx] = {
                "status": "fail", "retries": 0, "llm_obj": None,
                "gurobi_obj": None, "solve_time": None, "feasible": None,
                "aocc": None, "error": f"Code file not found: {code_path}",
            }
        return results, {"prompt_tokens": 0, "completion_tokens": 0}

    print(f"  Reusing code: {code_path}")
    results = {}
    for idx in instance_indices:
        res, _ = run_and_evaluate_instance(
            paper_id, model_name, idx, code_path,
            time_limit, exec_mode, exec_cfg, t_max
        )
        results[idx] = res

    return results, {"prompt_tokens": 0, "completion_tokens": 0}


def load_gurobi_csv_data(paper_id):
    """Load Gurobi baseline (objective, time) for a paper across all
    ``gurobi_results_*.csv`` files in ROOT_DIR (one per instance slot:
    ``tiny``, ``11``, ``31``, ...).

    Each CSV is tidy long format with columns:
        paper_id, instance, gurobi_feasibility_status, gurobi_solution,
        solution_status, gurobi_time, time_limit, failure_reason, failure_error
    ``instance`` values use the new categorical naming (``tiny``, ``large_1``,
    ``large_3``), matching eval's internal instance names.

    Returns dict keyed by instance name:
        {"tiny": {"solution": float|None, "time": float|None},
         "large_1": {...}, ...}
    ``N/A`` / ``time_out`` / empty values become ``None``."""
    csv_paths = sorted(glob.glob(os.path.join(ROOT_DIR, "gurobi_results_*.csv")))
    if not csv_paths:
        print(f"WARNING: no gurobi_results_*.csv files found under {ROOT_DIR}")
        return {}

    data = {}
    found_paper = False
    for csv_path in csv_paths:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("paper_id") != paper_id:
                    continue
                found_paper = True
                inst = (row.get("instance") or "").strip()
                if not inst:
                    continue
                sol_raw = (row.get("gurobi_solution") or "").strip()
                time_raw = (row.get("gurobi_time") or "").strip()
                try:
                    sol = float(sol_raw) if sol_raw not in ("", "N/A", "time_out") else None
                except (ValueError, TypeError):
                    sol = None
                try:
                    t = float(time_raw) if time_raw not in ("", "N/A", "time_out") else None
                except (ValueError, TypeError):
                    t = None
                data[inst] = {"solution": sol, "time": t}
    if not found_paper:
        print(f"WARNING: paper_id '{paper_id}' not found in any gurobi_results_*.csv")
    return data


def compute_gap(llm_obj, gurobi_obj, direction="min"):
    """Direction-aware signed gap vs the Gurobi reference objective.

    Convention: **negative gap always means LLM is better than Gurobi**.
      - ``direction="min"``: gap = (llm - gurobi) / |gurobi|
        (llm < gurobi → negative → LLM better at minimizing)
      - ``direction="max"``: gap = (gurobi - llm) / |gurobi|
        (llm > gurobi → negative → LLM better at maximizing)

    Returns None if either obj is missing, or if |gurobi| ≈ 0 and the two
    values don't numerically match (division ill-defined).
    """
    if llm_obj is None or gurobi_obj is None:
        return None
    if abs(gurobi_obj) < 1e-10:
        return 0.0 if abs(llm_obj) < 1e-10 else None
    if direction == "max":
        return (gurobi_obj - llm_obj) / abs(gurobi_obj)
    return (llm_obj - gurobi_obj) / abs(gurobi_obj)


_csv_lock = threading.Lock()


@contextlib.contextmanager
def _csv_file_lock(csv_path):
    """Cross-process + cross-thread exclusive lock for CSV read-modify-write.

    Uses fcntl.flock on a sibling ``.lock`` file plus the module-level
    threading.Lock. Either lock alone is insufficient: threading.Lock doesn't
    cross processes (multiple one_shot_eval.py invocations would race), and fcntl.flock
    only serializes file descriptors (multiple threads in one process could
    race). Both together make the read-modify-write atomic under any mix."""
    lock_path = csv_path + ".lock"
    # fcntl.flock requires a live fd held for the critical section.
    with _csv_lock:
        # Open separately per acquisition so a crash in the middle doesn't
        # leave a stale fd holding the lock.
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


RESULTS_CSV_COLUMNS = [
    "paper_id", "model", "instance",
    "status", "fail_reason", "error",
    "gap", "delta_time", "feasible",
    "debug_retries", "correction_retries",
    "obj", "time",
    "aocc",
    "first_status", "first_fail_reason",
    "first_gap", "first_feasible", "first_time", "first_obj",
]

API_COST_CSV_COLUMNS = [
    "paper_id", "model", "prompt_tokens", "completion_tokens", "cached_tokens", "api_cost",
]


def _fmt(v, decimals=6):
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, decimals)
    if isinstance(v, str) and ('\n' in v or '\r' in v):
        # Collapse embedded newlines so each CSV record is exactly one
        # physical line (grep-/diff-friendly). '|' marks original break points.
        return v.replace('\r\n', ' | ').replace('\n', ' | ').replace('\r', '')
    return v


def write_result_row(paper_id, model_name, instance_idx,
                     res, first_res, gurobi_csv_data):
    """
    Write (or overwrite) a single instance result row to eval/eval_results.csv.
    Thread-safe. Called after each instance finishes.
    """
    csv_path = get_results_csv_path()

    inst_data = gurobi_csv_data.get(instance_idx, {})
    gurobi_sol = inst_data.get("solution")
    gurobi_time = inst_data.get("time")

    # Direction from paper metadata; negative gap means LLM beats Gurobi
    # (consistently across min/max problems).
    direction = get_paper_direction(paper_id)
    llm_obj = res.get("llm_obj")
    gap = compute_gap(llm_obj, gurobi_sol, direction=direction)
    first_obj = first_res.get("llm_obj") if first_res else None
    first_gap = compute_gap(first_obj, gurobi_sol, direction=direction)

    solve_time = res.get("solve_time")
    delta_time = None
    if solve_time is not None and gurobi_time is not None:
        delta_time = solve_time - gurobi_time

    new_row = {
        "paper_id": paper_id,
        "model": model_name,
        "instance": instance_idx,
        "status": _fmt(res.get("status")),
        "fail_reason": _fmt(res.get("fail_reason")),
        "error": _fmt((res.get("error") or "")[:500]),
        "gap": _fmt(gap),
        "delta_time": _fmt(delta_time, 2),
        "feasible": _fmt(res.get("feasible")),
        "debug_retries": _fmt(res.get("debug_retries", 0)),
        "correction_retries": _fmt(res.get("correction_retries", 0)),
        "obj": _fmt(llm_obj),
        "time": _fmt(res.get("solve_time"), 2),
        "aocc": _fmt(res.get("aocc")),
        "first_status": _fmt(first_res.get("status") if first_res else None),
        "first_fail_reason": _fmt(first_res.get("fail_reason") if first_res else None),
        "first_gap": _fmt(first_gap),
        "first_feasible": _fmt(first_res.get("feasible") if first_res else None),
        "first_time": _fmt(first_res.get("solve_time") if first_res else None, 2),
        "first_obj": _fmt(first_obj),
    }

    new_key = (paper_id, model_name, instance_idx)

    with _csv_file_lock(csv_path):
        existing_rows = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["paper_id"], row["model"], row["instance"])
                    if key != (paper_id, model_name, str(instance_idx)):
                        existing_rows.append(row)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(new_row)

    print(f"    Result written to {csv_path}")


def write_api_cost_row(paper_id, model_name, model_id, token_usage):
    """
    Accumulate a (paper, model) row in eval/one_shot_api_cost.csv.

    Tokens and cost of the current invocation are ADDED to any existing row
    for the same (paper_id, model), so the row reflects total spend across
    all invocations. Thread-safe + cross-process safe via fcntl.flock.
    """
    csv_path = os.path.join(ROOT_DIR, "eval", "one_shot_api_cost.csv")

    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    cached_tokens = int(token_usage.get("cached_tokens", 0) or 0)

    new_key = (paper_id, model_name)

    def _to_int(v):
        try:
            return int(float(v)) if v not in (None, "", "None") else 0
        except (ValueError, TypeError):
            return 0

    with _csv_file_lock(csv_path):
        existing_rows = []
        prev_match = None
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["paper_id"], row["model"])
                    if key == new_key:
                        prev_match = row
                    else:
                        existing_rows.append(row)

        # Accumulate with the prior row's totals (if any) before recomputing cost.
        if prev_match:
            prompt_tokens += _to_int(prev_match.get("prompt_tokens"))
            completion_tokens += _to_int(prev_match.get("completion_tokens"))
            cached_tokens += _to_int(prev_match.get("cached_tokens"))

        pricing = MODEL_PRICING.get(model_id, {})
        # Cached tokens are billed at cache_read rate; remaining prompt tokens
        # at full input rate.
        non_cached_prompt = max(prompt_tokens - cached_tokens, 0)
        cache_read_price = pricing.get("cache_read", pricing.get("input", 0))
        api_cost = round(
            non_cached_prompt * pricing.get("input", 0) +
            cached_tokens * cache_read_price +
            completion_tokens * pricing.get("output", 0), 6
        )

        new_row = {
            "paper_id": paper_id,
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "api_cost": api_cost,
        }

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=API_COST_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(new_row)

    print(f"    API cost written to {csv_path}")



def print_summary(results):
    """Print a summary table of all instance results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Instance':<12} {'Status':<8} {'Retries':<9} {'LLM Obj':<18} {'Gurobi Obj':<18} {'Error'}")
    print("-" * 80)

    pass_count = 0
    fail_count = 0
    skip_count = 0
    error_details = []

    for idx, res in sorted(results.items()):
        status = res["status"]
        retries = res["retries"]
        llm_obj = f"{res['llm_obj']:.4f}" if res["llm_obj"] is not None else "N/A"
        gurobi_obj = f"{res['gurobi_obj']:.4f}" if res["gurobi_obj"] is not None else "N/A"
        error = res.get("error", "") or ""
        short_error = (error[:27] + "...") if len(error) > 30 else error

        print(f"  {idx:<10} {status:<8} {retries:<9} {llm_obj:<18} {gurobi_obj:<18} {short_error}")

        if error and len(error) > 30:
            error_details.append((idx, error))

        if status == "pass":
            pass_count += 1
        elif status == "fail":
            fail_count += 1
        else:
            skip_count += 1

    print("-" * 80)
    total = pass_count + fail_count + skip_count
    print(f"Total: {total}  |  Pass: {pass_count}  |  Fail: {fail_count}  |  Skip: {skip_count}")
    print("=" * 80)

    if error_details:
        print("\nERROR DETAILS")
        print("-" * 80)
        for idx, error in error_details:
            print(f"  Instance {idx}:\n{error}\n")


def process_paper_model(paper_id, config, model, instance_indices,
                        max_correct_retries, time_limit, prompt, gurobi_csv_data,
                        exec_mode="bare", exec_cfg=None, t_max=None,
                        skip_existing=False, reuse_code=False,
                        max_debug_retries=5):
    """Public entry. Routes the per-model results CSV via a thread-local
    so multiple models running concurrently each write to their own file
    (when ``--results_csv`` was not given as a global override). Delegates
    the real work to ``_process_paper_model_inner``."""
    model_name = get_model_short_name(model)
    prev_csv = getattr(_results_csv_local, "path", None)
    if not _RESULTS_CSV_OVERRIDE:
        _results_csv_local.path = _resolve_csv_for_model(model_name)
    try:
        return _process_paper_model_inner(
            paper_id, config, model, instance_indices,
            max_correct_retries, time_limit, prompt, gurobi_csv_data,
            exec_mode=exec_mode, exec_cfg=exec_cfg, t_max=t_max,
            skip_existing=skip_existing, reuse_code=reuse_code,
            max_debug_retries=max_debug_retries,
        )
    finally:
        if prev_csv is None:
            if hasattr(_results_csv_local, "path"):
                del _results_csv_local.path
        else:
            _results_csv_local.path = prev_csv


def _process_paper_model_inner(paper_id, config, model, instance_indices,
                               max_correct_retries, time_limit, prompt, gurobi_csv_data,
                               exec_mode="bare", exec_cfg=None, t_max=None,
                               skip_existing=False, reuse_code=False,
                               max_debug_retries=5):
    """The actual per-(paper, model) flow. Thread-safe; expects the caller
    (``process_paper_model``) to have configured the per-thread CSV target."""
    model_name = get_model_short_name(model)

    if skip_existing:
        code_path = os.path.join(get_model_eval_dir(paper_id, model_name), "code.py")
        if os.path.exists(code_path):
            print(f"\n  SKIP: {paper_id} / {model_name} — code already exists at {code_path}")
            return

    # Phase 1 gate uses ``instance_indices[0]`` as the "tiny-instance" gate
    # reference. If the user ran with ``--instances large_1 large_2 ...``
    # (no tiny), Phase 1 would accidentally gate on large_1 with strict
    # gap<=10% and cascade gate_fail to all the rest. To avoid that, auto-
    # inject tiny at position 0 when the CSV already has a tiny row for this
    # (paper, model). The injected tiny is treated as CSV-frozen regardless
    # of ``--reuse-code`` mode: Phase 1 reads its result from CSV, no re-run,
    # no re-flush.
    csv_done_all = _get_csv_done_instances(paper_id, model_name)
    tiny_auto_injected = False
    if "tiny" not in instance_indices and "tiny" in csv_done_all:
        print(f"  [gate] tiny row present in CSV; auto-injecting at position "
              f"0 of --instances so Phase 1 gates on tiny instead of "
              f"{instance_indices[0]!r}. Tiny is frozen (not re-run / not "
              f"re-flushed).")
        instance_indices = ["tiny"] + list(instance_indices)
        tiny_auto_injected = True

    # --reuse-code incomplete: do NOT drop CSV-done instances from
    # instance_indices — instead, pre-populate `results[idx]` from CSV so
    # Phase 1 gate can decide on the CSV value and we keep the canonical
    # [tiny, large, ...] ordering. CSV-done instances are then frozen:
    # skipped from every re-run path and from flush_results, so their CSV
    # rows stay untouched. When ``tiny_auto_injected`` is True we also mark
    # tiny as frozen regardless of reuse_code mode.
    csv_done = set()
    prev_rows = {}
    if reuse_code == "incomplete":
        csv_done = csv_done_all & set(instance_indices)
        if csv_done == set(instance_indices):
            print(f"\n  SKIP: {paper_id} / {model_name} — "
                  f"all {len(instance_indices)} instance(s) already in CSV "
                  f"(--reuse-code incomplete)")
            return
        if csv_done:
            prev_rows = _read_prev_result_rows(paper_id, model_name)
            print(f"\n  [reuse=incomplete] {paper_id}/{model_name}: "
                  f"frozen from CSV {sorted(csv_done)}, "
                  f"running {[i for i in instance_indices if i not in csv_done]}")
    elif tiny_auto_injected:
        # Non-incomplete mode still needs tiny frozen for gate purposes.
        csv_done = {"tiny"}
        prev_rows = _read_prev_result_rows(paper_id, model_name)

    print(f"\n{'='*80}")
    print(f"PAPER: {paper_id}  |  MODEL: {model} ({model_name})")
    print(f"{'='*80}\n")

    # Helper to flush results for a set of instances. Under --reuse-code
    # incomplete we never rewrite rows of CSV-done instances — those are
    # considered frozen.
    def flush_results(idxs, results, first_results):
        for idx in idxs:
            if idx in csv_done:
                continue
            res = results.get(idx, {})
            first = first_results.get(idx, {})
            write_result_row(paper_id, model_name, idx,
                             res, first, gurobi_csv_data)

    # === Unified flow: optional init-gen → v0 run on all instances
    # → Phase 0 debug (runtime_error via max_debug_retries)
    # → Phase 1 tiny gate (dispatch by error type) → Phase 2 dispatch loop.
    # --reuse-code skips init-gen and recovers prev (debug, correction) counts
    # from CSV / disk so budgets and CSV counters continue from where the
    # previous run stopped. ===
    model_dir = get_model_eval_dir(paper_id, model_name)
    if reuse_code:
        code_path = os.path.join(get_model_code_dir(paper_id, model_name), "code.py")
    else:
        code_path = os.path.join(model_dir, "code.py")
    attempt0_path = os.path.join(model_dir, "code_attempt0.py")

    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    def _add_usage(extra):
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            token_usage[k] += extra.get(k, 0)

    prev_debug = 0
    prev_correction = 0

    if reuse_code and not os.path.exists(code_path):
        # Nothing to reuse — fall back to fresh init-gen for this paper.
        # Lets a single run mix reuse + fresh papers without a second
        # process competing for the CSV lock.
        print(f"  [reuse] no code.py for {paper_id}/{model_name} — "
              f"falling back to fresh init-gen")
        reuse_code = False

    if reuse_code:
        # Per-run invariant: only one of {debug, correction} is non-zero in
        # any single run, so all prior attempts are attributed to whichever
        # budget is enabled in the current CLI (prefer debug if both > 0).
        historical = _count_disk_attempts(model_dir)
        if historical > 0:
            if max_debug_retries > 0:
                prev_debug = historical
            elif max_correct_retries > 0:
                prev_correction = historical
            print(f"  [reuse] counted {historical} prior attempt(s) on disk "
                  f"→ debug_retries={prev_debug}, "
                  f"correction_retries={prev_correction}")
        else:
            print(f"  [reuse] no prior code_attempt files — "
                  f"starting counters at 0.")
        code_v0 = "(existing code.py — init-gen skipped)"
    else:
        # --- Initial generation (code_v0) with its own 3-try budget ---
        code_v0, init_usage = _generate_initial_code(
            prompt, config, model, code_path, attempt0_path, init_gen_max=6
        )
        _add_usage(init_usage)

    tiny_idx = instance_indices[0]
    remaining_indices = instance_indices[1:]

    if code_v0 is None:
        # Initial generation failed to produce a valid code block after 3 tries.
        # Mark every instance as runtime_error-equivalent and bail.
        print(f"\n  *** Initial generation failed after 6 attempts. "
              f"Marking all instances as failed. ***\n")
        fail_record = {
            "status": "fail", "fail_reason": "runtime_error", "retries": 0,
            "llm_obj": None, "gurobi_obj": None,
            "solve_time": None, "feasible": None, "gap": None,
            "aocc": 1.0,
            "error": "LLM failed to produce a code block within 6 initial-generation attempts.",
        }
        results = {idx: dict(fail_record) for idx in instance_indices}
        first_results = {idx: dict(fail_record) for idx in instance_indices}
        flush_results(instance_indices, results, first_results)
        write_api_cost_row(paper_id, model_name, model, token_usage)
        print_summary(results)
        return

    # Instances we will actually run/touch this session. CSV-done instances
    # (incomplete mode) are excluded from every re-run path; their results
    # and first_results are pre-populated from CSV below.
    runnable_indices = [i for i in instance_indices if i not in csv_done]

    # Short-circuit: if tiny is CSV-frozen and its row already fails the gate,
    # the v0 run on remaining instances would just feed the inevitable cascade
    # gate_fail at Phase 1 — for large/multi-hour instances that's a 1h × N
    # wait for nothing. Cascade now and skip the v0 entirely. Only triggers
    # when (a) tiny is actually in csv_done, (b) prev_rows has its row, (c)
    # the row would not pass the 10% gate.
    if (runnable_indices
            and tiny_idx in csv_done
            and tiny_idx in prev_rows
            and not _tiny_gate_passed(prev_rows[tiny_idx])):
        tiny_prev = prev_rows[tiny_idx]
        print(f"\n  *** Pre-judge: tiny '{tiny_idx}' frozen from CSV "
              f"(status={tiny_prev.get('status')}, gap={tiny_prev.get('gap')}, "
              f"feasible={tiny_prev.get('feasible')}) fails Phase 1 gate. "
              f"Skipping v0 run; cascading gate_fail to {runnable_indices}. ***\n")
        historical_first = _read_prev_first_results(paper_id, model_name) if reuse_code else {}
        results = {}
        first_results = {}
        for idx in instance_indices:
            if idx in csv_done:
                results[idx] = dict(prev_rows.get(idx, {}))
                first_results[idx] = dict(historical_first.get(idx, prev_rows.get(idx, {})))
            else:
                cascade = {
                    "status": "gate_fail",
                    "fail_reason": tiny_prev.get("fail_reason"),
                    "retries": 0,
                    "debug_retries": 0,
                    "correction_retries": 0,
                    "llm_obj": None, "gurobi_obj": None,
                    "solve_time": None, "feasible": None, "gap": None,
                    "aocc": 1.0,
                    "error": f"Skipped: tiny-instance gate failed on instance {tiny_idx}",
                }
                results[idx] = dict(cascade)
                first_results[idx] = dict(cascade)
        flush_results(instance_indices, results, first_results)
        write_api_cost_row(paper_id, model_name, model, token_usage)
        print_summary(results)
        return

    # --- Run code on runnable instances to populate first_results ---
    print(f"\n  --- Running code_v0 on instances {runnable_indices} "
          f"for first_results ---")
    v0_results, _ = _run_all_instances(
        paper_id, model_name, runnable_indices, code_path,
        time_limit, exec_mode, exec_cfg, t_max
    )
    # Under --reuse-code the code on disk is NOT the historical v0 (it has
    # been overwritten by prior self-corrections), so v0_results reflects
    # the current post-correction state. Preserve the true historical
    # first_* fields by reading them back from the CSV; fall through to
    # v0_results for instances that have never been recorded.
    historical_first = _read_prev_first_results(paper_id, model_name) if reuse_code else {}
    first_results = {}
    results = {}
    for idx in instance_indices:
        if idx in csv_done:
            # Frozen: use CSV as both first_* (historical) and current.
            results[idx] = dict(prev_rows.get(idx, {}))
            first_results[idx] = dict(historical_first.get(idx, prev_rows.get(idx, {})))
        elif reuse_code and idx in historical_first:
            first_results[idx] = dict(historical_first[idx])
            results[idx] = dict(v0_results.get(idx, {}))
        else:
            first_results[idx] = dict(v0_results.get(idx, {}))
            results[idx] = dict(v0_results.get(idx, {}))

    # Two budgets, tracked independently but both visible as counters in CSV:
    #   retry_budget  (max_correct_retries): infeasible / invalid_solution /
    #                                        timeout runtime_error / gap>10%
    #   debug_budget  (max_debug_retries):   non-timeout runtime_error only
    # Under --reuse-code, prev counts recovered earlier are deducted from each
    # budget so the current run picks up where the previous one stopped.
    retry_budget = max(0, max_correct_retries - prev_correction)
    retries_used = prev_correction
    debug_budget = max(0, max_debug_retries - prev_debug)
    debug_used = prev_debug
    code_changed_since_v0 = False  # becomes True after first self-correction

    def _total_attempts():
        return debug_used + retries_used

    def _apply_retries(res):
        """Stamp combined + split retry counters onto a result dict."""
        res["retries"] = _total_attempts()
        res["debug_retries"] = debug_used
        res["correction_retries"] = retries_used
        return res

    def _rt_indices():
        """Instances currently in non-timeout runtime_error state. Excludes
        CSV-frozen (incomplete-mode) instances."""
        return [idx for idx in runnable_indices
                if results[idx].get("fail_reason") == "runtime_error"
                and not _is_timeout_error(results[idx])]

    # --- Phase 0: pre-gate self-debug for non-timeout runtime_error ---
    # Loops until no runtime_error remains OR debug budget is exhausted.
    # Each fix re-runs ONLY the currently-erroring instances (not passing ones).
    print(f"\n  --- Phase 0: Self-debug for runtime_error "
          f"(budget={debug_budget} remaining of {max_debug_retries}, "
          f"already used={debug_used}) ---")
    while debug_budget > 0:
        err_rt = _collect_debug_errors(results, runnable_indices)
        if not err_rt:
            break
        debug_budget -= 1
        debug_used += 1
        with open(code_path, "r") as f:
            cur_code = f.read()
        print(f"\n  [debug {debug_used}/{max_debug_retries}] "
              f"{len(err_rt)} instance(s) with runtime_error, calling LLM...")
        attempt_path = os.path.join(model_dir, f"code_attempt{_total_attempts()}.py")
        new_code, corr_usage = _self_correct_once(
            prompt, cur_code, "\n\n".join(err_rt), config, model,
            code_path, attempt_path
        )
        _add_usage(corr_usage)
        if new_code is None:
            continue
        code_changed_since_v0 = True
        rerun_subset = _rt_indices()
        subset_results, _ = _run_all_instances(
            paper_id, model_name, rerun_subset, code_path,
            time_limit, exec_mode, exec_cfg, t_max
        )
        for idx, r in subset_results.items():
            _apply_retries(r)
            results[idx] = r

    # --- Phase 1: tiny-instance gate (gap<=10%) ---
    # Dispatch: tiny has runtime_error (non-timeout) -> debug_budget;
    #           tiny has other correctable failure        -> retry_budget.
    # If tiny is CSV-frozen (incomplete mode), skip the correction loop
    # entirely — the CSV value is authoritative.
    print(f"\n  --- Phase 1: Tiny-instance gate (instance {tiny_idx}, gap<=10%) ---")
    if tiny_idx in csv_done:
        print(f"  [reuse=incomplete] tiny '{tiny_idx}' frozen from CSV "
              f"(status={results[tiny_idx].get('status')}, "
              f"gap={results[tiny_idx].get('gap')}). Skipping correction loop.")
    while tiny_idx not in csv_done:
        tiny_res = results[tiny_idx]
        if _tiny_gate_passed(tiny_res):
            break
        if not _tiny_gate_correctable(tiny_res):
            break
        is_rt_trigger = (tiny_res.get("fail_reason") == "runtime_error"
                         and not _is_timeout_error(tiny_res))
        if is_rt_trigger and debug_budget > 0:
            debug_budget -= 1
            debug_used += 1
            kind, count_str = "debug", f"{debug_used}/{max_debug_retries}"
        elif (not is_rt_trigger) and retry_budget > 0:
            retry_budget -= 1
            retries_used += 1
            kind, count_str = "phase1", f"{retries_used}/{max_correct_retries}"
        else:
            # Relevant budget exhausted — can't fix this kind of failure.
            break
        with open(code_path, "r") as f:
            cur_code = f.read()
        err_summary = _tiny_gate_error_summary(tiny_idx, tiny_res)
        print(f"\n  [{kind} {count_str}] tiny gate not passed, calling LLM...")
        attempt_path = os.path.join(model_dir, f"code_attempt{_total_attempts()}.py")
        new_code, corr_usage = _self_correct_once(
            prompt, cur_code, err_summary, config, model, code_path, attempt_path
        )
        _add_usage(corr_usage)
        if new_code is None:
            continue
        code_changed_since_v0 = True
        tr, _ = run_and_evaluate_instance(
            paper_id, model_name, tiny_idx, code_path,
            time_limit, exec_mode, exec_cfg, t_max
        )
        _apply_retries(tr)
        results[tiny_idx] = tr

    if not _tiny_gate_passed(results[tiny_idx]):
        # Gate failed: mark all remaining instances as gate_fail and exit.
        print(f"\n  *** Tiny-instance gate FAILED for instance {tiny_idx} "
              f"after {_total_attempts()} LLM correction(s) "
              f"(debug={debug_used}, phase1={retries_used}). "
              f"Skipping instances {remaining_indices}. ***\n")
        for idx in remaining_indices:
            if idx in csv_done:
                continue
            results[idx] = {
                "status": "gate_fail", "fail_reason": results[tiny_idx].get("fail_reason"),
                "retries": _total_attempts(),
                "debug_retries": debug_used,
                "correction_retries": retries_used,
                "llm_obj": None, "gurobi_obj": None,
                "solve_time": None, "feasible": None, "gap": None,
                "aocc": 1.0,
                "error": f"Skipped: tiny-instance gate failed on instance {tiny_idx}",
            }
        flush_results(instance_indices, results, first_results)
        write_api_cost_row(paper_id, model_name, model, token_usage)
        print_summary(results)
        return

    print(f"\n  --- Phase 1: gate PASSED "
          f"(tiny gap={results[tiny_idx].get('gap')}) ---")

    # Flush tiny's row as soon as its gate resolves: it is considered final
    # at this point. Phase 2 may re-run tiny only via retry_budget full-rerun
    # (regression check); if tiny is updated later, the final flush_results
    # call at the end rewrites the row via row-level dedup. (flush_results
    # itself no-ops for CSV-frozen instances in incomplete mode.)
    flush_results([tiny_idx], results, first_results)

    # --- Phase 2: remaining instances, relaxed correction trigger ---
    # If Phase 0/1 made corrections, cached v0 results for remaining instances
    # are stale — re-run remaining with current code. Otherwise v0 results are
    # still valid for `results` (they equal first_results). CSV-frozen
    # instances are never re-run.
    runnable_remaining = [i for i in remaining_indices if i not in csv_done]
    if runnable_remaining and code_changed_since_v0:
        print(f"\n  --- Phase 2: re-running remaining instances {runnable_remaining} "
              f"with current code ---")
        rem_results, _ = _run_all_instances(
            paper_id, model_name, runnable_remaining, code_path,
            time_limit, exec_mode, exec_cfg, t_max
        )
        for idx, r in rem_results.items():
            _apply_retries(r)
            results[idx] = r
    elif runnable_remaining:
        print(f"\n  --- Phase 2: v0 code passed gate; reusing v0 results for "
              f"instances {runnable_remaining} ---")
        # results already == first_results for those; nothing to rerun.

    # Phase 2 dispatch loop:
    #   non-timeout runtime_error  -> debug_budget, re-run only failing instances
    #   infeasible / invalid_solution / timeout runtime_error
    #                              -> retry_budget, re-run ALL (detect regressions)
    # Runtime_error is prioritized: if both kinds coexist, fix runtime_error first.
    while True:
        err_rt = _collect_debug_errors(results, runnable_indices)
        err_other = _collect_correction_errors(results, runnable_indices)
        if err_rt and debug_budget > 0:
            debug_budget -= 1
            debug_used += 1
            feedback = "\n\n".join(err_rt)
            kind, count_str = "debug", f"{debug_used}/{max_debug_retries}"
            rerun_subset = _rt_indices()
            err_count = len(err_rt)
        elif err_other and retry_budget > 0:
            retry_budget -= 1
            retries_used += 1
            feedback = "\n\n".join(err_other)
            kind, count_str = "phase2", f"{retries_used}/{max_correct_retries}"
            rerun_subset = runnable_indices
            err_count = len(err_other)
        else:
            break
        with open(code_path, "r") as f:
            cur_code = f.read()
        print(f"\n  [{kind} {count_str}] "
              f"{err_count} instance(s) triggered correction, calling LLM...")
        attempt_path = os.path.join(model_dir, f"code_attempt{_total_attempts()}.py")
        new_code, corr_usage = _self_correct_once(
            prompt, cur_code, feedback, config, model, code_path, attempt_path
        )
        _add_usage(corr_usage)
        if new_code is None:
            continue
        code_changed_since_v0 = True
        subset_results, _ = _run_all_instances(
            paper_id, model_name, rerun_subset, code_path,
            time_limit, exec_mode, exec_cfg, t_max
        )
        for idx, r in subset_results.items():
            _apply_retries(r)
            results[idx] = r

    remaining_rt = _collect_debug_errors(results, runnable_indices)
    remaining_other = _collect_correction_errors(results, runnable_indices)
    if remaining_rt:
        print(f"\n  {len(remaining_rt)} instance(s) still in runtime_error "
              f"after debug budget exhausted.")
    if remaining_other:
        print(f"\n  {len(remaining_other)} instance(s) still failing "
              f"(infeasible/invalid_solution/timeout) after retry budget exhausted.")
    if not remaining_rt and not remaining_other:
        print(f"\n  All correction-triggering failures resolved.")

    flush_results(instance_indices, results, first_results)
    write_api_cost_row(paper_id, model_name, model, token_usage)
    print_summary(results)


def _load_paper_context(paper_id):
    """Load per-paper context shared across all (paper, model) tasks.
    Returns ``(prompt, gurobi_csv_data)`` or ``None`` when essential schema
    files are missing — caller should treat ``None`` as "skip the paper"."""
    gurobi_csv_data = load_gurobi_csv_data(paper_id)
    problem_desc = read_problem_description(paper_id)
    instance_template = read_instance_template(paper_id)
    if instance_template is None:
        print(f"ERROR: instance_schema.json not found for {paper_id} — skipping paper")
        return None
    solution_template = read_solution_template(paper_id)
    if solution_template is None:
        print(f"ERROR: solution_schema.json not found for {paper_id} — skipping paper")
        return None
    prompt = build_prompt(problem_desc, instance_template, solution_template)
    return prompt, gurobi_csv_data


def process_paper(paper_id, config, models, instance_indices,
                   max_correct_retries, time_limit,
                   exec_mode="bare", exec_cfg=None, t_max=None,
                   model_workers=1, skip_existing=False, reuse_code=False,
                   max_debug_retries=5):
    """Process all models for a single paper. Models run in parallel when
    model_workers > 1. Kept for direct programmatic use; ``main()`` uses a
    flat ``(paper, model)`` task pool instead."""
    ctx = _load_paper_context(paper_id)
    if ctx is None:
        return
    prompt, gurobi_csv_data = ctx

    if model_workers <= 1:
        for model in models:
            process_paper_model(
                paper_id, config, model, instance_indices,
                max_correct_retries, time_limit, prompt, gurobi_csv_data,
                exec_mode=exec_mode, exec_cfg=exec_cfg, t_max=t_max,
                skip_existing=skip_existing, reuse_code=reuse_code,
                max_debug_retries=max_debug_retries
            )
    else:
        with ThreadPoolExecutor(max_workers=model_workers) as pool:
            futures = {
                pool.submit(
                    process_paper_model,
                    paper_id, config, model, instance_indices,
                    max_correct_retries, time_limit, prompt, gurobi_csv_data,
                    exec_mode=exec_mode, exec_cfg=exec_cfg, t_max=t_max,
                    skip_existing=skip_existing, reuse_code=reuse_code,
                    max_debug_retries=max_debug_retries
                ): model
                for model in models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] paper_id '{paper_id}', model '{model}': {e}",
                          file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    parser.add_argument("--paper_id", nargs="+", default=None,
                        help="Paper ID(s) to evaluate (e.g., belvaux2000 wang2018). "
                             "If omitted, --paper-tag drives the selection; if both "
                             "are omitted, every paper directory under the data dir "
                             "is used (full sweep).")
    parser.add_argument("--paper-tag", type=str, default=None,
                        help="Restrict to paper IDs with this tag in "
                             "gurobi_results_all_new.csv (exact match, e.g. 'A'). Only "
                             "consulted when --paper_id is omitted. Multi-tag rows "
                             "like 'A,F' do not match 'A' — pass the literal "
                             "comma-form to match those.")
    parser.add_argument("--max_correct_retries", type=int, default=0,
                        help="Max self-correction retries for infeasible / "
                             "invalid_solution / timeout / gap over 5%% (default: 0).")
    parser.add_argument("--max_debug_retries", type=int, default=5,
                        help="Max self-debug retries for non-timeout runtime errors "
                             "(separate budget from --max_correct_retries; default: 5)")
    parser.add_argument("--time_limit", type=int, default=300, help="Time limit per code execution (seconds)")
    parser.add_argument("--model_workers", type=int, default=1,
                        help="Number of models to evaluate in parallel within each paper (default: 1).")
    parser.add_argument("--paper_workers", type=int, default=1,
                        help="Number of papers to process in parallel (default: 1). "
                             "Combines with --model_workers: total in-flight (paper, model) pairs "
                             "= paper_workers * model_workers.")
    parser.add_argument("--instance_workers", type=int, default=1,
                        help="Global cap on concurrently-running instance solvers "
                             "(default: 1). All paper/model threads share a single "
                             "ThreadPoolExecutor of this size, so when one instance "
                             "finishes the freed slot picks up the next queued instance "
                             "from any paper — a paper waiting on its own subset doesn't "
                             "leave the slot idle. Each instance is its own systemd-run "
                             "job; size --memory for ~``instance_workers`` concurrent jobs.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models from config to run (short names, e.g. gpt-4o gemini-3-pro-preview). "
                             "Pass 'all' (i.e. --models all) as an alias for every configured model. "
                             "If omitted, all models in config are used.")
    parser.add_argument("--exec-mode", type=str, default="systemd",
                        choices=["bare", "systemd", "docker"],
                        help="Execution backend: bare (no CPU limits — debug only!), "
                             "systemd (default; cgroup + taskset pinning to --cpus cores), "
                             "docker (full container isolation).")
    parser.add_argument("--cpus", type=int, default=1,
                        help="CPU cores for systemd/docker execution (default: 1).")
    parser.add_argument("--memory", type=str, default="640G",
                        help="Memory limit for systemd/docker execution (default: 640G).")
    parser.add_argument("--t_max", type=parse_t_max, default=None,
                        help="Custom time horizon for AOCC computation. "
                             "Accepts a positive float (seconds, global) or the "
                             "literal 'gurobi' to use each instance's own Gurobi "
                             "solve time (from gurobi_results_*.csv) as horizon. "
                             "If omitted, uses --time_limit.")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="Categorical instance names to run (e.g., --instances tiny large_1). "
                             f"Default: {' '.join(DEFAULT_INSTANCES)}.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (paper, model) pairs that already have generated code.")
    parser.add_argument("--reuse-code", nargs="?", const="all",
                        default="incomplete",
                        choices=["none", "all", "incomplete"],
                        help="Reuse existing code.py on disk (fallback to fresh "
                             "init-gen when missing). "
                             "none: always fresh init-gen, do NOT skip CSV-done. "
                             "all: run every --instances entry (overwriting CSV rows). "
                             "incomplete (DEFAULT): skip (paper, instance) pairs "
                             "already recorded in the results CSV, only run the rest. "
                             "Bare --reuse-code is equivalent to 'all'.")
    parser.add_argument("--code-root", type=str, default=None,
                        help="Directory to read code.py from, layout "
                             "<code-root>/<paper>/<model>/code.py "
                             "(default: eval/eval_papers). Quick Start uses "
                             "'samples/oneshot_code' to evaluate the shipped "
                             "pre-generated programs directly. Pipeline writes "
                             "still go to eval/eval_papers regardless.")
    parser.add_argument("--results_csv", type=str, default=None,
                        help="Path to the results CSV (default: "
                             "eval/eval_results.csv). Use a per-run file to "
                             "isolate outputs from parallel one_shot_eval.py runs.")
    args = parser.parse_args()
    if args.code_root:
        global CODE_ROOT
        CODE_ROOT = os.path.abspath(os.path.expanduser(args.code_root))
        if not os.path.isdir(CODE_ROOT):
            print(f"ERROR: --code-root {CODE_ROOT!r} is not a directory")
            sys.exit(1)
    # Downstream uses truthy checks on reuse_code; normalize the "none" label.
    if args.reuse_code == "none":
        args.reuse_code = None
    # "--models all" is an alias for "every configured model" (== omitted).
    if args.models and [m.lower() for m in args.models] == ["all"]:
        args.models = None
    if not args.paper_id:
        if args.paper_tag:
            args.paper_id = _load_paper_ids_by_tag(args.paper_tag)
            if not args.paper_id:
                print(f"ERROR: --paper-tag {args.paper_tag!r} matched no rows "
                      f"in {_GUROBI_RESULTS_ALL}", file=sys.stderr)
                sys.exit(1)
            print(f"[paper_id] omitted; defaulting to {len(args.paper_id)} papers "
                  f"with tag={args.paper_tag!r} from gurobi_results_all_new.csv")
        else:
            args.paper_id = _load_all_paper_dirs()
            if not args.paper_id:
                print(f"ERROR: --paper_id and --paper-tag both omitted, but "
                      f"data dir {get_data_dir()} has no paper subdirs",
                      file=sys.stderr)
                sys.exit(1)
            print(f"[paper_id] omitted; defaulting to {len(args.paper_id)} papers "
                  f"by scanning {get_data_dir()}")

    # Preflight: every paper must have a registered optimization direction.
    # A missing direction silently inverts quality/QTE scores -- fail upfront.
    try:
        validate_paper_directions(args.paper_id)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.results_csv:
        set_results_csv_path(args.results_csv)
        print(f"Results CSV: {get_results_csv_path()} (--results_csv override)")
    else:
        # Per-model routing: each ``process_paper_model`` invocation sets a
        # thread-local target from ``_MODEL_RESULTS_CSV``. Show what each
        # selected model will write to, so the user sees the routing up front.
        if args.models:
            print(f"Results CSV: per-model auto-routing")
            for m in args.models:
                target = _MODEL_RESULTS_CSV.get(m, _DEFAULT_RESULTS_CSV)
                print(f"  {m:25s} -> {target}")
        else:
            print(f"Results CSV: per-model auto-routing for all configured models")
    gurobi_license = configure_gurobi_license()

    # Parse instance names (categorical: "tiny", "large_1", ...)
    try:
        instance_indices = parse_instances_arg(args.instances or DEFAULT_INSTANCES)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    config = load_config()
    all_models = config["models"]

    if args.models:
        models = [m for m in all_models if get_model_short_name(m) in args.models]
        unknown = set(args.models) - {get_model_short_name(m) for m in all_models}
        if unknown:
            print(f"WARNING: unknown model(s) {unknown}, available: {[get_model_short_name(m) for m in all_models]}")
        if not models:
            print("ERROR: no matching models found. Exiting.")
            sys.exit(1)
    else:
        models = all_models
    print(f"Models: {models}")
    print(f"Papers: {args.paper_id}")
    print(f"Instances: {instance_indices}")
    print(f"Data dir: {get_data_dir()}")
    print(f"GRB_LICENSE_FILE: {gurobi_license or os.environ.get('GRB_LICENSE_FILE') or '<not set>'}")
    exec_mode = args.exec_mode
    exec_cfg = {"cpus": args.cpus, "memory": args.memory}
    t_max = args.t_max
    reuse_code = args.reuse_code
    set_instance_workers(args.instance_workers)
    atexit.register(_shutdown_instance_pool)
    print(f"Max correct retries: {args.max_correct_retries}, Max debug retries: {args.max_debug_retries}, "
          f"Time limit: {args.time_limit}s, "
          f"Paper workers: {args.paper_workers}, Model workers: {args.model_workers}, "
          f"Instance workers: {args.instance_workers}")
    print(f"Exec: {exec_mode} (mem={args.memory}), T_max: {t_max or 'time_limit'}")
    if reuse_code == "all":
        print("Mode: REUSE-CODE=all (reuse code.py on disk; "
              "fresh init-gen where missing; run all --instances)")
    elif reuse_code == "incomplete":
        print("Mode: REUSE-CODE=incomplete (reuse code.py on disk; "
              "fresh init-gen where missing; skip (paper, instance) pairs "
              "already in eval_results.csv)")
    print()

    # Flat (paper, model) task pool. Total worker count =
    # paper_workers × model_workers. Each task is independent; per-model
    # CSV routing happens inside process_paper_model via thread-local.
    total_workers = max(1, args.paper_workers * args.model_workers)

    # Pre-build per-paper contexts (prompt + gurobi data) once each, so
    # the 7 model tasks of one paper share one prompt build instead of
    # rebuilding per model. Papers missing required schema files are
    # skipped here with a warning, and never enter the task list.
    paper_contexts = {}
    skipped_papers = []
    for paper_id in args.paper_id:
        ctx = _load_paper_context(paper_id)
        if ctx is None:
            skipped_papers.append(paper_id)
            continue
        paper_contexts[paper_id] = ctx

    if skipped_papers:
        print(f"[WARN] {len(skipped_papers)} paper(s) skipped due to missing schema: "
              f"{skipped_papers[:5]}{'...' if len(skipped_papers)>5 else ''}")

    tasks = [(p, m) for p in paper_contexts for m in models]
    print(f"[task-pool] dispatching {len(tasks)} (paper, model) tasks "
          f"across {total_workers} worker(s) "
          f"(paper_workers × model_workers = {args.paper_workers} × {args.model_workers})")

    def _run_one(paper_id, model):
        prompt, gurobi_csv_data = paper_contexts[paper_id]
        process_paper_model(
            paper_id, config, model, instance_indices,
            args.max_correct_retries, args.time_limit, prompt, gurobi_csv_data,
            exec_mode=exec_mode, exec_cfg=exec_cfg, t_max=t_max,
            skip_existing=args.skip_existing, reuse_code=reuse_code,
            max_debug_retries=args.max_debug_retries,
        )

    if total_workers <= 1:
        for p, m in tasks:
            try:
                _run_one(p, m)
            except Exception as e:
                print(f"[ERROR] paper={p} model={m}: {e}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=total_workers) as pool:
            futures = {pool.submit(_run_one, p, m): (p, m) for p, m in tasks}
            for future in as_completed(futures):
                p, m = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] paper={p} model={m}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
