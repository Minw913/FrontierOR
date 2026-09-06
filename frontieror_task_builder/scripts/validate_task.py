#!/usr/bin/env python3
"""Run lightweight FrontierOR task validation.

This script intentionally performs deterministic local checks only. It validates
JSON syntax, static contracts, optional jsonschema conformance, and existing
feasibility result files. Running Gurobi or regenerating solutions remains a
caller-controlled step because it may require licenses and long runtimes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INSTANCE_NAMES = ["tiny"] + [f"large_{i}" for i in range(1, 6)]


def slot_file(name: str, suffix: str, ext: str) -> str:
    if name == "tiny":
        return f"tiny_{suffix}.{ext}"
    if name.startswith("large_"):
        return f"large_{suffix}_{name.split('_', 1)[1]}.{ext}"
    raise ValueError(f"invalid instance name: {name}")


def import_contract_checker():
    path = SCRIPT_DIR / "check_artifact_contract.py"
    spec = importlib.util.spec_from_file_location("check_artifact_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_jsonschema_validate(schema_path: Path, data_path: Path, errors: list[str], warnings: list[str]):
    try:
        import jsonschema  # type: ignore
    except Exception:
        warnings.append("jsonschema is not installed; skipped JSON Schema validation")
        return
    try:
        schema = load_json(schema_path)
        data = load_json(data_path)
        jsonschema.validate(instance=data, schema=schema)
    except Exception as exc:
        errors.append(f"{data_path.relative_to(schema_path.parent)} failed schema validation: {exc}")


def validate(task_dir: Path, check_schemas: bool) -> dict:
    contract = import_contract_checker()
    errors, warnings = contract.check(task_dir)

    for name in INSTANCE_NAMES:
        inst = task_dir / f"instance/{slot_file(name, 'instance', 'json')}"
        if inst.is_file():
            try:
                load_json(inst)
            except Exception as exc:
                errors.append(f"{inst.relative_to(task_dir)} is not valid JSON: {exc}")
            if check_schemas and (task_dir / "instance_schema.json").is_file():
                maybe_jsonschema_validate(task_dir / "instance_schema.json", inst, errors, warnings)

        sol = task_dir / f"gurobi_solution/{slot_file(name, 'solution', 'json')}"
        if sol.is_file():
            try:
                data = load_json(sol)
                if not isinstance(data, dict) or not isinstance(data.get("objective_value"), (int, float)):
                    errors.append(f"{sol.relative_to(task_dir)} missing numeric objective_value")
            except Exception as exc:
                errors.append(f"{sol.relative_to(task_dir)} is not valid JSON: {exc}")
            if check_schemas and (task_dir / "solution_schema.json").is_file():
                maybe_jsonschema_validate(task_dir / "solution_schema.json", sol, errors, warnings)

        feasi = task_dir / f"gurobi_feasi_result/{slot_file(name, 'feasi_result', 'json')}"
        if feasi.is_file():
            try:
                data = load_json(feasi)
                if data.get("feasible") is not True:
                    errors.append(f"{feasi.relative_to(task_dir)} does not mark feasible=true")
                for key in ("violated_constraints", "violations", "violation_magnitudes"):
                    if key not in data:
                        errors.append(f"{feasi.relative_to(task_dir)} missing {key}")
            except Exception as exc:
                errors.append(f"{feasi.relative_to(task_dir)} is not valid JSON: {exc}")

    return {
        "task_dir": str(task_dir),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--schema", action="store_true", help="Run jsonschema validation when jsonschema is installed.")
    parser.add_argument("--report-path", type=Path, help="Optional path for validation report JSON.")
    args = parser.parse_args()

    report = validate(args.task_dir.resolve(), check_schemas=args.schema)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report_path:
        args.report_path.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
