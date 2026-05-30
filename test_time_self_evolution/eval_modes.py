"""Framework-agnostic helpers for multi-mode evaluation runners.

OpenEvolve-specific orchestration lives in
:mod:`test_time_self_evolution.openevolve.runner`.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import shutil
from typing import Dict, Iterable, List, Optional

import one_shot_eval as eval_core


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODE_API_COST_COLUMNS = [
    "run_id",
    "mode",
    "paper_id",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "api_cost",
    "note",
]


def discover_papers(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    return sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    )


def _safe_float(value, default=math.inf):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def abs_gap(result: Dict) -> float:
    return abs(_safe_float(result.get("gap")))


def combined_score(result: Dict) -> float:
    """Return OpenEvolve-compatible score in [0, 1] for a single eval result."""
    if result.get("feasible") is not True:
        return 0.0
    gap = abs_gap(result)
    if math.isinf(gap):
        return 0.0
    quality_score = max(0.0, 1.0 - min(gap, 1.0))
    solve_time = _safe_float(result.get("solve_time"), default=math.inf)
    runtime_score = 0.0 if math.isinf(solve_time) else 1.0 / (1.0 + max(solve_time, 0.0))
    return round(0.95 * quality_score + 0.05 * runtime_score, 6)


def selection_key(candidate: Dict, selection_instance: str):
    result = candidate.get("results", {}).get(selection_instance, {})
    feasible_rank = 0 if result.get("feasible") is True else 1
    gap_rank = abs_gap(result)
    time_rank = _safe_float(result.get("solve_time"))
    candidate_rank = candidate.get("candidate_id", math.inf)
    return feasible_rank, gap_rank, time_rank, candidate_rank


def select_best_candidate(candidates: Iterable[Dict], selection_instance: str) -> Dict:
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidates available for selection")
    return min(candidates, key=lambda candidate: selection_key(candidate, selection_instance))


def merge_usage(total: Dict, usage: Optional[Dict]):
    usage = usage or {}
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        total[key] = total.get(key, 0) + usage.get(key, 0)
    return total


def compute_api_cost(model_id: str, usage: Dict) -> float:
    pricing = eval_core.MODEL_PRICING.get(model_id, {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    non_cached_prompt = max(prompt_tokens - cached_tokens, 0)
    cache_read_price = pricing.get("cache_read", pricing.get("input", 0))
    return round(
        non_cached_prompt * pricing.get("input", 0)
        + cached_tokens * cache_read_price
        + completion_tokens * pricing.get("output", 0),
        6,
    )


def write_api_cost_row(run_id: str, mode: str, paper_id: str, model_id: str,
                       model_name: str, usage: Dict, note: str = ""):
    csv_path = os.path.join(ROOT_DIR, "eval", "self_evolve_api_cost.csv")
    append_csv_row(csv_path, MODE_API_COST_COLUMNS, {
        "run_id": run_id,
        "mode": mode,
        "paper_id": paper_id,
        "model": model_name,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("cached_tokens", 0),
        "api_cost": compute_api_cost(model_id, usage),
        "note": note,
    })


def mode_run_dir(run_id: str, mode: str, paper_id: str, model_name: str) -> str:
    """Per-run output directory: ``eval/<mode>/<run_id>/<paper_id>/<model_name>/``.

    ``mode`` is one of ``one_shot`` / ``best_of_k`` (per-mode runners) or
    ``openevolve`` / ``eoh`` / ``coral`` (per-framework self-evolve runners) —
    the framework name itself is used so paths self-identify which framework
    wrote them.
    """
    return os.path.join(ROOT_DIR, "eval", mode, run_id, paper_id, model_name)


def append_csv_row(csv_path: str, fieldnames: List[str], row: Dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _fmt(value, decimals=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, decimals)
    return value


def copy_selected_code(code_path: str, destination_dir: str) -> str:
    os.makedirs(destination_dir, exist_ok=True)
    dest = os.path.join(destination_dir, "selected_code.py")
    shutil.copyfile(code_path, dest)
    return dest


# --- Self-evolve unified per-instance results CSV (one row per (paper, instance)).
# Same shape as one_shot_eval.RESULTS_CSV_COLUMNS but tailored to self-evolve:
# drop debug/correction retries (no self-correction), drop most first_* columns
# (we only keep first_status to track "did evolve actually move past seed"),
# and add iteration_found + generation to identify which iter / lineage depth
# produced the final best program.
#
# Two files per framework:
#   eval/eval_dev_results_<framework>.csv   ← dev set rows (15 cols, w/ first_status)
#   eval/eval_test_results_<framework>.csv  ← test set rows (14 cols, no first_status)

SELF_EVOLVE_DEV_RESULTS_COLUMNS = [
    "run_id",
    "paper_id", "model", "instance",
    "iteration_found", "generation",
    "status", "fail_reason", "error",
    "gap", "delta_time", "feasible",
    "obj", "time", "aocc",
    "if_beat_gurobi",
    # staged_qte stage2 score + decomposition (see scoring/staged_qte.py).
    # Populated for runs using the staged_qte scorer (default).
    # ``score_staged`` ∈ [0, 2+] (>2 ⇔ LLM beats Gurobi).
    "score_staged", "stage_id",
    "quality_part", "speed_part", "signed_gap", "beat_amount",
    "first_status",
]

SELF_EVOLVE_TEST_RESULTS_COLUMNS = [
    "run_id",
    "paper_id", "model", "instance",
    "iteration_found", "generation",
    "status", "fail_reason", "error",
    "gap", "delta_time", "feasible",
    "obj", "time", "aocc",
    "if_beat_gurobi",
    "score_staged", "stage_id",
    "quality_part", "speed_part", "signed_gap", "beat_amount",
]


def _self_evolve_csv_path(framework: str, kind: str) -> str:
    """``kind`` ∈ {'dev', 'test'}; framework e.g. 'openevolve' / 'eoh' / 'coral'."""
    return os.path.join(ROOT_DIR, "eval", f"eval_{kind}_results_{framework}.csv")


def _load_oneshot_first_status(paper_id: str, model_short: str, instance: str) -> Optional[str]:
    """Read v0's status (= one-shot's ``first_status``) for (paper, model,
    instance) from any ``eval/eval_results_*.csv`` file.

    Used when self-evolve reused one-shot's ``code_attempt0.py`` as its seed
    (see ``runner._try_reuse_oneshot_seed``): the seed IS the same code that
    one-shot evaluated as v0, so one-shot's already-computed ``first_status``
    is the authoritative source for our ``first_status`` column.

    The one-shot CSV filename has its own historical convention
    (``eval_results_codex53.csv``, ``eval_results_deepseekr1.csv``, ...) that
    isn't always derivable from ``model_short``, so we glob all
    ``eval_results_*.csv`` files and filter by the ``model`` column instead.
    Returns ``None`` if no matching row exists.
    """
    pattern = os.path.join(ROOT_DIR, "eval", "eval_results_*.csv")
    for csv_path in glob.glob(pattern):
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                # Skip files without first_status column (defensive — would
                # mean it's not a one-shot CSV)
                if not reader.fieldnames or "first_status" not in reader.fieldnames:
                    continue
                for row in reader:
                    if (row.get("paper_id") == paper_id and
                            row.get("model") == model_short and
                            row.get("instance") == str(instance)):
                        val = row.get("first_status")
                        return val if val else None
        except Exception:
            continue
    return None


def _build_self_evolve_row(
    paper_id, model_name, instance,
    result, gurobi_obj, gurobi_time,
    iteration_found, generation,
    seed_status=None,
    include_first_status=True,
    run_id=None,
):
    """Build one self-evolve CSV row dict. ``result`` has the same shape that
    ``one_shot_eval.run_and_evaluate_instance`` produces (status / fail_reason /
    error / feasible / gap / solve_time / aocc / llm_obj)."""
    direction = eval_core.get_paper_direction(paper_id)
    llm_obj = (result or {}).get("llm_obj")
    gap = eval_core.compute_gap(llm_obj, gurobi_obj, direction=direction)
    solve_time = (result or {}).get("solve_time")
    delta_time = (
        solve_time - gurobi_time
        if (solve_time is not None and gurobi_time is not None)
        else None
    )
    feasible = (result or {}).get("feasible")
    # if_beat_gurobi: best program is feasible AND quality-matched-or-better
    # (signed gap < 1e-4) AND strictly faster than Gurobi. ``gap`` is signed
    # via compute_gap (negative = LLM better), so gap < 1e-4 captures both
    # "matched" (gap ≈ 0) and "beat" (gap < 0). Returns False (not None) when
    # any prerequisite is missing — keeps the column boolean-typed for
    # downstream group-by/aggregations.
    if_beat_gurobi = (
        feasible is True
        and gap is not None and gap < 1e-4
        and delta_time is not None and delta_time < 0
    )

    row = {
        "run_id": _fmt(run_id),
        "paper_id": paper_id,
        "model": model_name,
        "instance": instance,
        "iteration_found": "" if iteration_found is None else int(iteration_found),
        "generation": "" if generation is None else int(generation),
        "status": _fmt((result or {}).get("status")),
        "fail_reason": _fmt((result or {}).get("fail_reason")),
        "error": _fmt(((result or {}).get("error") or "")[:500]),
        "gap": _fmt(gap),
        "delta_time": _fmt(delta_time, 2),
        "feasible": _fmt(feasible),
        "obj": _fmt(llm_obj),
        "time": _fmt(solve_time, 2),
        "aocc": _fmt((result or {}).get("aocc")),
        "if_beat_gurobi": if_beat_gurobi,
        # staged_qte score + stage decomposition (see scoring/staged_qte.py).
        # ``score_staged`` ∈ [0, 0.8] for stage 1, [0.8, 2+] for stage 2 (>2
        # when LLM beats Gurobi). Blank if scorer is `aocc`.
        "score_staged": _fmt((result or {}).get("score")),
        "stage_id": _fmt((result or {}).get("stage_id")),
        "quality_part": _fmt((result or {}).get("quality_part")),
        "speed_part": _fmt((result or {}).get("speed_part")),
        "signed_gap": _fmt((result or {}).get("signed_gap")),
        "beat_amount": _fmt((result or {}).get("beat_amount")),
    }
    if include_first_status:
        row["first_status"] = _fmt(seed_status)
    return row


def _write_self_evolve_csv_with_dedup(csv_path, columns, new_row):
    """Replace any existing row matching (paper_id, model, instance);
    append otherwise. Uses fcntl.flock for cross-process safety."""
    new_key = (
        new_row["paper_id"],
        new_row["model"],
        str(new_row["instance"]),
    )
    with eval_core._csv_file_lock(csv_path):
        existing = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["paper_id"], row["model"], row["instance"])
                    if key != new_key:
                        existing.append(row)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(new_row)


def write_self_evolve_results(
    *,
    paper_id: str,
    model_name: str,
    framework: str,
    dev_instances: List[str],
    dev_results: Dict[str, Dict],
    dev_seed_results: Optional[Dict[str, Dict]],
    test_instances: List[str],
    test_results: Dict[str, Dict],
    iteration_found: Optional[int] = None,
    generation: Optional[int] = None,
    run_id: Optional[str] = None,
):
    """Write per-instance rows to dev / test CSV files for one paper.

    Args:
        paper_id, model_name: identifiers (model_name is e.g. 'deepseek-r1')
        framework: 'openevolve' / 'eoh' / 'coral' — used in CSV filename
        dev_instances: dev-set instance names (typically 1, e.g. ['large_21'])
        dev_results: best program's per-dev-instance result dict
        dev_seed_results: seed's per-dev-instance result dict (sets first_status);
                          pass {} or None to leave first_status blank
        test_instances: test-set instance names (typically 4)
        test_results: best program's per-test-instance result dict
        iteration_found: iter at which best was found (None → blank)
        generation: best program's lineage depth (None → blank)
    """
    dev_path = _self_evolve_csv_path(framework, "dev")
    test_path = _self_evolve_csv_path(framework, "test")
    gurobi_data = eval_core.load_gurobi_csv_data(paper_id)

    seed_dev = dev_seed_results or {}
    model_short = eval_core.get_model_short_name(model_name)
    for inst in dev_instances:
        gd = gurobi_data.get(inst) or {}
        # Prefer one-shot's stored first_status when available — it's the
        # authoritative v0 evaluation, and self-evolve's seed (when reused via
        # _try_reuse_oneshot_seed) IS that same code. Falls back to the
        # reconstructed seed status from the OpenEvolve checkpoint.
        seed_status = _load_oneshot_first_status(paper_id, model_short, inst)
        if seed_status is None:
            seed_status = (seed_dev.get(inst) or {}).get("status")
        row = _build_self_evolve_row(
            paper_id, model_name, inst,
            dev_results.get(inst) or {},
            gd.get("solution"), gd.get("time"),
            iteration_found, generation,
            seed_status=seed_status,
            include_first_status=True,
            run_id=run_id,
        )
        _write_self_evolve_csv_with_dedup(dev_path, SELF_EVOLVE_DEV_RESULTS_COLUMNS, row)

    for inst in test_instances:
        gd = gurobi_data.get(inst) or {}
        row = _build_self_evolve_row(
            paper_id, model_name, inst,
            test_results.get(inst) or {},
            gd.get("solution"), gd.get("time"),
            iteration_found, generation,
            seed_status=None,
            include_first_status=False,
            run_id=run_id,
        )
        _write_self_evolve_csv_with_dedup(test_path, SELF_EVOLVE_TEST_RESULTS_COLUMNS, row)


def _resolve_test_time_limits(
    paper_id: str,
    test_instances: List[str],
    test_time_limit: int,
    test_time_policy: str,
    test_time_buffer: int,
) -> Dict[str, int]:
    """Per-instance time budget for the post-evolve test eval.
    Returns ``{inst: seconds}``.
    """
    from test_time_self_evolution.scoring.building_blocks import lookup_gurobi_time

    final_tl: Dict[str, int] = {}
    for inst in test_instances:
        tau_g = lookup_gurobi_time(paper_id, inst)
        if test_time_policy == "gurobi_time" and tau_g is not None:
            final_tl[inst] = min(int(tau_g), test_time_limit)
        elif test_time_policy == "gurobi_time_plus_buffer" and tau_g is not None:
            final_tl[inst] = min(int(tau_g) + test_time_buffer, test_time_limit)
        else:
            final_tl[inst] = test_time_limit
    return final_tl


def evaluate_best_on_test_set(
    paper_id: str,
    model_name: str,
    best_program_path: str,
    test_instances: List[str],
    test_time_limit: int,
    test_time_policy: str,
    test_time_buffer: int,
    output_dir: str,
    exec_mode: str,
    exec_cfg: Dict,
    t_max,
    *,
    max_workers: int = 4,
) -> Dict[str, Dict]:
    """Evaluate the final best program on test instances IN PARALLEL.

    Each test instance runs in its own thread, calling
    ``eval_core.evaluate_candidate_code([single_inst], ...)``. Per-instance
    files (``solution_<inst>.json`` / ``log_<inst>.jsonl`` / ``feasi_result_<inst>.json``)
    don't collide because the filenames are keyed by instance name.

    Use ``max_workers=1`` to disable parallelism (e.g., on machines with
    limited CPU or limited Gurobi license tokens).
    """
    if not test_instances:
        return {}

    final_tl = _resolve_test_time_limits(
        paper_id, test_instances, test_time_limit, test_time_policy, test_time_buffer,
    )

    n_workers = max(1, min(max_workers, len(test_instances)))
    if n_workers <= 1 or len(test_instances) <= 1:
        return eval_core.evaluate_candidate_code(
            paper_id, model_name, test_instances, best_program_path,
            output_dir, final_tl, exec_mode, exec_cfg, t_max,
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval_one(inst: str) -> Dict[str, Dict]:
        return eval_core.evaluate_candidate_code(
            paper_id, model_name, [inst], best_program_path,
            output_dir, final_tl[inst], exec_mode, exec_cfg, t_max,
        )

    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_eval_one, inst): inst for inst in test_instances}
        for fut in as_completed(futures):
            results.update(fut.result())
    return results


def run_one_shot(
    run_id: str,
    paper_id: str,
    model: str,
    prompt: str,
    config: Dict,
    instances: List[str],
    time_limit: int,
    exec_mode: str,
    exec_cfg: Dict,
    t_max,
):
    model_name = eval_core.get_model_short_name(model)
    base_dir = mode_run_dir(run_id, "one_shot", paper_id, model_name)
    candidate_dir = os.path.join(base_dir, "candidate_0")
    generated = eval_core.generate_candidate_code(
        prompt, config, model, candidate_dir, candidate_id=0
    )
    token_usage = generated.get("usage", {})
    if generated["status"] != "ok":
        results = {
            instance: {
                "status": "fail",
                "fail_reason": "generation_error",
                "feasible": None,
                "gap": None,
                "solve_time": None,
                "llm_obj": None,
                "gurobi_obj": None,
                "aocc": None,
                "error": generated.get("error"),
                "retries": 0,
            }
            for instance in instances
        }
        code_path = ""
    else:
        code_path = generated["code_path"]
        results = eval_core.evaluate_candidate_code(
            paper_id, model_name, instances, code_path, candidate_dir,
            time_limit, exec_mode, exec_cfg, t_max,
        )

    selected_code = copy_selected_code(code_path, base_dir) if code_path else ""
    write_api_cost_row(run_id, "one_shot", paper_id, model, model_name, token_usage)
    return {"candidate_id": 0, "results": results, "code_path": selected_code or code_path}


def run_best_of_k(
    run_id: str,
    paper_id: str,
    model: str,
    prompt: str,
    config: Dict,
    instances: List[str],
    k: int,
    selection_instance: str,
    time_limit: int,
    exec_mode: str,
    exec_cfg: Dict,
    t_max,
):
    model_name = eval_core.get_model_short_name(model)
    base_dir = mode_run_dir(run_id, "best_of_k", paper_id, model_name)
    candidates = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    for candidate_id in range(k):
        candidate_dir = os.path.join(base_dir, f"candidate_{candidate_id}")
        generated = eval_core.generate_candidate_code(
            prompt, config, model, candidate_dir,
            candidate_id=candidate_id, temperature=0.8,
        )
        merge_usage(token_usage, generated.get("usage", {}))
        code_path = generated.get("code_path")
        if generated["status"] == "ok":
            results = eval_core.evaluate_candidate_code(
                paper_id, model_name, [selection_instance], code_path, candidate_dir,
                time_limit, exec_mode, exec_cfg, t_max,
            )
        else:
            results = {
                selection_instance: {
                    "status": "fail",
                    "fail_reason": "generation_error",
                    "feasible": None,
                    "gap": None,
                    "solve_time": None,
                    "llm_obj": None,
                    "gurobi_obj": None,
                    "aocc": None,
                    "error": generated.get("error"),
                    "retries": 0,
                }
            }
            code_path = ""

        candidate = {
            "candidate_id": candidate_id,
            "results": results,
            "code_path": code_path,
            "usage": generated.get("usage", {}),
        }
        candidates.append(candidate)

    selected = select_best_candidate(candidates, selection_instance)
    selected_candidate_id = selected["candidate_id"]
    selected_code_path = selected.get("code_path") or ""

    if selected_code_path:
        selected_dir = os.path.join(base_dir, "selected")
        final_results = eval_core.evaluate_candidate_code(
            paper_id, model_name, instances, selected_code_path, selected_dir,
            time_limit, exec_mode, exec_cfg, t_max,
        )
        selected_code = copy_selected_code(selected_code_path, base_dir)
    else:
        final_results = {
            instance: {
                "status": "fail",
                "fail_reason": "generation_error",
                "feasible": None,
                "gap": None,
                "solve_time": None,
                "llm_obj": None,
                "gurobi_obj": None,
                "aocc": None,
                "error": "No generated candidate code was available",
                "retries": 0,
            }
            for instance in instances
        }
        selected_code = ""

    write_api_cost_row(run_id, "best_of_k", paper_id, model, model_name, token_usage)
    return {
        "candidate_id": selected_candidate_id,
        "results": final_results,
        "code_path": selected_code or selected_code_path,
    }


