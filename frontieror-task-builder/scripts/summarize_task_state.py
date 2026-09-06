#!/usr/bin/env python3
"""Summarize a FrontierOR task directory and recommend the next build step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INSTANCE_NAMES = ["tiny"] + [f"large_{i}" for i in range(1, 6)]


def exists(task_dir: Path, rel: str) -> bool:
    return (task_dir / rel).is_file()


def slot_file(name: str, suffix: str, ext: str) -> str:
    if name == "tiny":
        return f"tiny_{suffix}.{ext}"
    if name.startswith("large_"):
        return f"large_{suffix}_{name.split('_', 1)[1]}.{ext}"
    raise ValueError(f"invalid instance name: {name}")


def summarize(task_dir: Path) -> dict:
    required = [
        "problem_description.txt",
        "original_formulation.tex",
        "instance_schema.json",
        "solution_schema.json",
        "feasibility_check.py",
        "generate_multiple_instances.py",
        "gurobi_code.py",
    ]
    instance_files = [f"instance/{slot_file(name, 'instance', 'json')}" for name in INSTANCE_NAMES]
    solution_files = [f"gurobi_solution/{slot_file(name, 'solution', 'json')}" for name in INSTANCE_NAMES]
    feasi_files = [f"gurobi_feasi_result/{slot_file(name, 'feasi_result', 'json')}" for name in INSTANCE_NAMES]

    groups = {
        "core": required,
        "instances": instance_files,
        "gurobi_solutions": solution_files,
        "gurobi_feasibility_results": feasi_files,
    }

    found = []
    missing = []
    for files in groups.values():
        for rel in files:
            (found if exists(task_dir, rel) else missing).append(rel)

    if not exists(task_dir, "problem_description.txt") and not exists(task_dir, "original_formulation.tex"):
        next_step = "choose_flow_and_create_confirmed_description_formulation_pair"
    elif not exists(task_dir, "problem_description.txt"):
        next_step = "generate_or_confirm_problem_description"
    elif not exists(task_dir, "original_formulation.tex"):
        next_step = "generate_or_confirm_original_formulation"
    elif not exists(task_dir, "instance_schema.json") or not exists(task_dir, "solution_schema.json"):
        next_step = "generate_data_specification"
    elif not exists(task_dir, "feasibility_check.py"):
        next_step = "generate_feasibility_checker"
    elif any(not exists(task_dir, rel) for rel in instance_files):
        next_step = "generate_multiple_instances"
    elif not exists(task_dir, "gurobi_code.py"):
        next_step = "generate_gurobi_code"
    elif any(not exists(task_dir, rel) for rel in solution_files):
        next_step = "run_gurobi_reference_solver"
    elif any(not exists(task_dir, rel) for rel in feasi_files):
        next_step = "run_feasibility_checker"
    else:
        next_step = "validate_complete_task"

    return {
        "task_dir": str(task_dir),
        "found": found,
        "missing": missing,
        "recommended_next_step": next_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    report = summarize(args.task_dir.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"task_dir: {report['task_dir']}")
    print(f"recommended_next_step: {report['recommended_next_step']}")
    print("\nfound:")
    for rel in report["found"]:
        print(f"  - {rel}")
    print("\nmissing:")
    for rel in report["missing"]:
        print(f"  - {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
