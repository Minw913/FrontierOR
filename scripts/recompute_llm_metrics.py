#!/usr/bin/env python3
"""Recompute stored LLM CSV metrics after Gurobi baselines change.

This script does not rerun generated code or checkers. It reads existing
LLM outputs under eval/eval_papers/ and refreshes CSV fields that depend on
the Gurobi reference objective/runtime:

  gap, delta_time, aocc, first_gap

It also refreshes pass/fail classification for rows whose stored feasibility
is True and whose status is therefore governed by the quality gap.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import one_shot_eval  # noqa: E402


GAP_FAIL_THRESHOLD = 0.10


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _fmt(value, decimals=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return str(round(value, decimals))
    return str(value)


def _candidate_results_csvs(explicit):
    if explicit:
        return [Path(p).expanduser().resolve() for p in explicit]
    eval_dir = ROOT_DIR / "eval"
    paths = sorted(eval_dir.glob("eval_results*.csv"))
    return [p for p in paths if p.is_file()]


def _log_path(paper_id, model, instance):
    return ROOT_DIR / "eval" / "eval_papers" / paper_id / model / f"log_{instance}.jsonl"


def _recompute_row(row, baseline, *, t_max_mode):
    instance = row.get("instance", "")
    ref = baseline.get(instance) or {}
    gurobi_obj = ref.get("solution")
    gurobi_time = ref.get("time")

    direction = one_shot_eval.get_paper_direction(row["paper_id"])
    llm_obj = _to_float(row.get("obj"))
    first_obj = _to_float(row.get("first_obj"))
    solve_time = _to_float(row.get("time"))

    gap = one_shot_eval.compute_gap(llm_obj, gurobi_obj, direction=direction)
    first_gap = one_shot_eval.compute_gap(first_obj, gurobi_obj, direction=direction)
    delta_time = (
        solve_time - gurobi_time
        if solve_time is not None and gurobi_time is not None
        else None
    )

    t_max = gurobi_time if t_max_mode == "gurobi" else None
    aocc = one_shot_eval.compute_aocc(
        str(_log_path(row["paper_id"], row["model"], instance)),
        gurobi_obj,
        solve_time,
        t_max=t_max,
        direction=direction,
    ) if solve_time is not None else None

    out = dict(row)
    out["gap"] = _fmt(gap)
    out["first_gap"] = _fmt(first_gap)
    out["delta_time"] = _fmt(delta_time, decimals=2)
    out["aocc"] = _fmt(aocc)

    feasible = _to_bool(row.get("feasible"))
    if feasible is True:
        if gap is not None and gap > GAP_FAIL_THRESHOLD:
            out["status"] = "fail"
            out["fail_reason"] = "gap_exceeds"
            out["error"] = (
                f"Gap {gap:.2%} exceeds {GAP_FAIL_THRESHOLD:.0%} threshold "
                f"(obj={llm_obj}, gurobi={gurobi_obj})"
            )
        else:
            out["status"] = "pass"
            out["fail_reason"] = ""
            if (row.get("fail_reason") or "") == "gap_exceeds":
                out["error"] = ""

    return out


def recompute_csv(csv_path, task_ids, *, t_max_mode, dry_run):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return 0
        rows = list(reader)

    baseline_cache = {}
    changed = 0
    new_rows = []
    for row in rows:
        paper_id = row.get("paper_id")
        if paper_id not in task_ids:
            new_rows.append(row)
            continue
        if paper_id not in baseline_cache:
            baseline_cache[paper_id] = one_shot_eval.load_gurobi_csv_data(paper_id)
        new_row = _recompute_row(row, baseline_cache[paper_id], t_max_mode=t_max_mode)
        if new_row != row:
            changed += 1
        new_rows.append(new_row)

    if changed and not dry_run:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(new_rows)
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Recompute existing LLM result CSV metrics for selected tasks."
    )
    parser.add_argument("task_ids", nargs="+", help="Paper/task IDs to refresh.")
    parser.add_argument(
        "--results-csv",
        action="append",
        default=None,
        help="CSV to update. Can be passed multiple times. Default: eval/eval_results*.csv.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="FrontierOR data root. Defaults to FRONTIER_OR_DATA_DIR or one_shot_eval.py default.",
    )
    parser.add_argument(
        "--t-max",
        choices=["gurobi", "elapsed"],
        default="gurobi",
        help="AOCC horizon. Default uses each Gurobi solution JSON runtime.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["FRONTIER_OR_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())

    task_ids = set(args.task_ids)
    csvs = _candidate_results_csvs(args.results_csv)
    if not csvs:
        raise SystemExit("No eval_results*.csv files found.")

    total = 0
    for csv_path in csvs:
        changed = recompute_csv(
            csv_path,
            task_ids,
            t_max_mode=args.t_max,
            dry_run=args.dry_run,
        )
        total += changed
        action = "would update" if args.dry_run else "updated"
        print(f"{action} {changed} row(s): {csv_path}")
    print(f"total rows {'would update' if args.dry_run else 'updated'}: {total}")


if __name__ == "__main__":
    main()
