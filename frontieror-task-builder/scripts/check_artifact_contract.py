#!/usr/bin/env python3
"""Static contract checks for a FrontierOR task directory."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


INSTANCE_NAMES = ["tiny"] + [f"large_{i}" for i in range(1, 6)]
CORE_FILES = [
    "problem_description.txt",
    "original_formulation.tex",
    "instance_schema.json",
    "solution_schema.json",
    "feasibility_check.py",
    "generate_multiple_instances.py",
    "gurobi_code.py",
]


def slot_file(name: str, suffix: str, ext: str) -> str:
    if name == "tiny":
        return f"tiny_{suffix}.{ext}"
    if name.startswith("large_"):
        return f"large_{suffix}_{name.split('_', 1)[1]}.{ext}"
    raise ValueError(f"invalid instance name: {name}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def script_defines_argparse_flags(path: Path, flags: list[str]) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, flags
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False, flags
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
    missing = [flag for flag in flags if flag not in found]
    return not missing, missing


def check(task_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in CORE_FILES:
        if not (task_dir / rel).is_file():
            errors.append(f"missing required file: {rel}")

    for name in INSTANCE_NAMES:
        rel = f"instance/{slot_file(name, 'instance', 'json')}"
        if not (task_dir / rel).is_file():
            errors.append(f"missing canonical instance: {rel}")

    for rel in ("instance_schema.json", "solution_schema.json"):
        path = task_dir / rel
        if path.is_file():
            try:
                data = load_json(path)
                if not isinstance(data, dict):
                    errors.append(f"{rel} must be a JSON object")
            except Exception as exc:
                errors.append(f"{rel} is not valid JSON: {exc}")

    ok, missing = script_defines_argparse_flags(
        task_dir / "gurobi_code.py",
        ["--instance_path", "--solution_path", "--time_limit"],
    )
    if not ok:
        errors.append(f"gurobi_code.py missing argparse flags: {', '.join(missing)}")

    ok, missing = script_defines_argparse_flags(
        task_dir / "feasibility_check.py",
        ["--instance_path", "--solution_path", "--result_path"],
    )
    if not ok:
        errors.append(f"feasibility_check.py missing argparse flags: {', '.join(missing)}")

    tiny_solution = task_dir / "gurobi_solution/tiny_solution.json"
    if tiny_solution.is_file():
        try:
            solution = load_json(tiny_solution)
            if not isinstance(solution, dict):
                errors.append("gurobi_solution/tiny_solution.json must be a JSON object")
            elif not isinstance(solution.get("objective_value"), (int, float)):
                errors.append("gurobi_solution/tiny_solution.json missing numeric objective_value")
        except Exception as exc:
            errors.append(f"gurobi_solution/tiny_solution.json is not valid JSON: {exc}")
    else:
        warnings.append("tiny Gurobi solution not present; run solver validation later")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    errors, warnings = check(args.task_dir.resolve())
    if args.json:
        print(json.dumps({"errors": errors, "warnings": warnings}, indent=2))
    else:
        for warning in warnings:
            print(f"WARN: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        if not errors:
            print("OK: artifact contract checks passed")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
