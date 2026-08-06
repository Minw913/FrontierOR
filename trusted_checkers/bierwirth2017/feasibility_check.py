#!/usr/bin/env python3
"""Strict FrontierOR wrapper for the Bierwirth JSPTWT checker.

The dataset checker accepts two historical output layouts, but it assumes
several non-schema fields exist and does not reject all omitted operations.
This wrapper validates a complete schedule, derives all redundant fields from
the instance, independently verifies the objective, and then delegates the
constraint checks to the original paper checker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Any


ABS_TOL = 1e-6
REL_TOL = 1e-9


class InvalidSolution(ValueError):
    pass


def _load_object(path: str, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise InvalidSolution(f"{label} must be a JSON object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise InvalidSolution(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise InvalidSolution(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise InvalidSolution(f"{label} must be a finite number")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidSolution(f"{label} must be an integer")
    return value


def _close(left: float, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    tolerance = max(ABS_TOL, REL_TOL * max(abs(left), abs(right)))
    return abs(left - right) <= tolerance


def _completion_time(start_time: float, processing_time: float, label: str) -> float:
    completion_time = start_time + processing_time
    if not math.isfinite(completion_time):
        raise InvalidSolution(f"{label} produces a non-finite completion time")
    return completion_time


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidSolution(f"{label} must be an array")
    return value


def _instance_jobs(instance: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    jobs = _require_list(instance.get("jobs"), "instance.jobs")
    by_id: dict[int, dict[str, Any]] = {}
    for position, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise InvalidSolution(f"instance.jobs[{position}] must be an object")
        job_id = _integer(job.get("job_id"), f"instance.jobs[{position}].job_id")
        if job_id in by_id:
            raise InvalidSolution(f"instance contains duplicate job_id {job_id}")
        operations = _require_list(
            job.get("operations"),
            f"instance.jobs[{position}].operations",
        )
        if not operations:
            raise InvalidSolution(f"instance job {job_id} has no operations")
        by_id[job_id] = job
    if len(by_id) != instance.get("num_jobs"):
        raise InvalidSolution("instance num_jobs does not match jobs")
    return jobs, by_id


def _expected_operation(
    jobs_by_id: dict[int, dict[str, Any]],
    job_id: int,
    operation_index: int,
) -> tuple[int, float]:
    try:
        operations = jobs_by_id[job_id]["operations"]
    except KeyError as exc:
        raise InvalidSolution(f"unknown job {job_id}") from exc
    if operation_index < 0 or operation_index >= len(operations):
        raise InvalidSolution(
            f"unknown operation index {operation_index} for job {job_id}"
        )
    operation = operations[operation_index]
    if not isinstance(operation, dict):
        raise InvalidSolution(
            f"instance operation ({job_id}, {operation_index}) must be an object"
        )
    machine = _integer(
        operation.get("machine"),
        f"instance operation ({job_id}, {operation_index}).machine",
    )
    processing_time = _finite_number(
        operation.get("processing_time"),
        f"instance operation ({job_id}, {operation_index}).processing_time",
    )
    return machine, processing_time


def _validate_optional_derived(
    operation: dict[str, Any],
    *,
    processing_time: float,
    completion_time: float,
    label: str,
) -> None:
    if "processing_time" in operation:
        reported = _finite_number(
            operation["processing_time"],
            f"{label}.processing_time",
        )
        if not _close(reported, processing_time):
            raise InvalidSolution(f"{label}.processing_time does not match the instance")
    for field in ("end_time", "completion_time"):
        if field in operation:
            reported = _finite_number(operation[field], f"{label}.{field}")
            if not _close(reported, completion_time):
                raise InvalidSolution(f"{label}.{field} does not match start_time + processing_time")


def _starts_from_schedule(
    solution: dict[str, Any],
    jobs: list[dict[str, Any]],
    jobs_by_id: dict[int, dict[str, Any]],
) -> dict[tuple[int, int], float]:
    schedule = _require_list(solution.get("schedule"), "solution.schedule")
    expected_job_ids = set(jobs_by_id)
    seen_job_ids: set[int] = set()
    starts: dict[tuple[int, int], float] = {}

    for row_index, row in enumerate(schedule):
        if not isinstance(row, dict):
            raise InvalidSolution(f"solution.schedule[{row_index}] must be an object")
        job_id = _integer(
            row.get("job_id"),
            f"solution.schedule[{row_index}].job_id",
        )
        if job_id not in expected_job_ids:
            raise InvalidSolution(f"solution contains unknown job {job_id}")
        if job_id in seen_job_ids:
            raise InvalidSolution(f"solution contains duplicate job {job_id}")
        seen_job_ids.add(job_id)

        operations = _require_list(
            row.get("operations"),
            f"solution schedule for job {job_id}.operations",
        )
        expected_operations = jobs_by_id[job_id]["operations"]
        if len(operations) != len(expected_operations):
            raise InvalidSolution(
                f"job {job_id} must contain exactly {len(expected_operations)} operations"
            )

        for operation_index, operation in enumerate(operations):
            label = f"solution operation ({job_id}, {operation_index})"
            if not isinstance(operation, dict):
                raise InvalidSolution(f"{label} must be an object")
            expected_machine, processing_time = _expected_operation(
                jobs_by_id,
                job_id,
                operation_index,
            )
            machine = _integer(operation.get("machine"), f"{label}.machine")
            if machine != expected_machine:
                raise InvalidSolution(f"{label}.machine does not match the instance")
            start_time = _finite_number(operation.get("start_time"), f"{label}.start_time")
            completion_time = _completion_time(start_time, processing_time, label)
            _validate_optional_derived(
                operation,
                processing_time=processing_time,
                completion_time=completion_time,
                label=label,
            )
            starts[(job_id, operation_index)] = start_time

        last_index = len(expected_operations) - 1
        _, last_processing_time = _expected_operation(jobs_by_id, job_id, last_index)
        completion_time = _completion_time(
            starts[(job_id, last_index)],
            last_processing_time,
            f"solution job {job_id}",
        )
        due_date = _finite_number(jobs_by_id[job_id].get("due_date"), f"job {job_id}.due_date")
        tardiness = max(0.0, completion_time - due_date)
        reported_completion = _finite_number(
            row.get("completion_time"),
            f"solution schedule for job {job_id}.completion_time",
        )
        reported_tardiness = _finite_number(
            row.get("tardiness"),
            f"solution schedule for job {job_id}.tardiness",
        )
        if not _close(reported_completion, completion_time):
            raise InvalidSolution(f"job {job_id} completion_time is inconsistent")
        if not _close(reported_tardiness, tardiness):
            raise InvalidSolution(f"job {job_id} tardiness is inconsistent")

    if seen_job_ids != expected_job_ids:
        missing = sorted(expected_job_ids - seen_job_ids)
        raise InvalidSolution(f"solution is missing jobs: {missing}")
    return starts


def _starts_from_machine_schedules(
    solution: dict[str, Any],
    jobs: list[dict[str, Any]],
    jobs_by_id: dict[int, dict[str, Any]],
    num_machines: int,
) -> dict[tuple[int, int], float]:
    machine_rows = _require_list(
        solution.get("machine_schedules"),
        "solution.machine_schedules",
    )
    expected_machines = set(range(num_machines))
    seen_machines: set[int] = set()
    starts: dict[tuple[int, int], float] = {}

    for row_index, row in enumerate(machine_rows):
        if not isinstance(row, dict):
            raise InvalidSolution(
                f"solution.machine_schedules[{row_index}] must be an object"
            )
        raw_machine = row.get("machine", row.get("machine_id"))
        machine = _integer(
            raw_machine,
            f"solution.machine_schedules[{row_index}].machine",
        )
        if machine not in expected_machines:
            raise InvalidSolution(f"solution contains unknown machine {machine}")
        if machine in seen_machines:
            raise InvalidSolution(f"solution contains duplicate machine {machine}")
        seen_machines.add(machine)
        operations = _require_list(
            row.get("operations"),
            f"solution machine {machine}.operations",
        )
        for operation_position, operation in enumerate(operations):
            label = f"solution machine {machine} operation {operation_position}"
            if not isinstance(operation, dict):
                raise InvalidSolution(f"{label} must be an object")
            job_id = _integer(operation.get("job"), f"{label}.job")
            operation_index = _integer(
                operation.get("operation_index"),
                f"{label}.operation_index",
            )
            key = (job_id, operation_index)
            if key in starts:
                raise InvalidSolution(f"solution contains duplicate operation {key}")
            expected_machine, processing_time = _expected_operation(
                jobs_by_id,
                job_id,
                operation_index,
            )
            if machine != expected_machine:
                raise InvalidSolution(f"operation {key} is assigned to the wrong machine")
            start_time = _finite_number(operation.get("start_time"), f"{label}.start_time")
            completion_time = _completion_time(start_time, processing_time, label)
            _validate_optional_derived(
                operation,
                processing_time=processing_time,
                completion_time=completion_time,
                label=label,
            )
            starts[key] = start_time

    if seen_machines != expected_machines:
        missing = sorted(expected_machines - seen_machines)
        raise InvalidSolution(f"solution is missing machine schedules: {missing}")

    expected_operations = {
        (job["job_id"], operation_index)
        for job in jobs
        for operation_index in range(len(job["operations"]))
    }
    if set(starts) != expected_operations:
        missing = sorted(expected_operations - set(starts))
        extra = sorted(set(starts) - expected_operations)
        raise InvalidSolution(
            f"solution operation set is incomplete (missing={missing}, extra={extra})"
        )

    completions = _require_list(
        solution.get("job_completions"),
        "solution.job_completions",
    )
    expected_job_ids = set(jobs_by_id)
    seen_job_ids: set[int] = set()
    for position, completion in enumerate(completions):
        if not isinstance(completion, dict):
            raise InvalidSolution(f"solution.job_completions[{position}] must be an object")
        job_id = _integer(
            completion.get("job"),
            f"solution.job_completions[{position}].job",
        )
        if job_id not in expected_job_ids or job_id in seen_job_ids:
            raise InvalidSolution(f"invalid or duplicate job completion for job {job_id}")
        seen_job_ids.add(job_id)
        last_index = len(jobs_by_id[job_id]["operations"]) - 1
        _, processing_time = _expected_operation(jobs_by_id, job_id, last_index)
        actual_completion = _completion_time(
            starts[(job_id, last_index)],
            processing_time,
            f"solution job completion {job_id}",
        )
        due_date = _finite_number(jobs_by_id[job_id].get("due_date"), f"job {job_id}.due_date")
        actual_tardiness = max(0.0, actual_completion - due_date)
        reported_completion = _finite_number(
            completion.get("completion_time"),
            f"solution job completion {job_id}.completion_time",
        )
        reported_tardiness = _finite_number(
            completion.get("tardiness"),
            f"solution job completion {job_id}.tardiness",
        )
        if not _close(reported_completion, actual_completion):
            raise InvalidSolution(f"job {job_id} completion_time is inconsistent")
        if not _close(reported_tardiness, actual_tardiness):
            raise InvalidSolution(f"job {job_id} tardiness is inconsistent")

    if seen_job_ids != expected_job_ids:
        missing = sorted(expected_job_ids - seen_job_ids)
        raise InvalidSolution(f"solution is missing job completions: {missing}")
    return starts


def _normalize_solution(
    instance: dict[str, Any],
    solution: dict[str, Any],
) -> dict[str, Any]:
    jobs, jobs_by_id = _instance_jobs(instance)
    has_schedule = "schedule" in solution
    has_machine_schedules = "machine_schedules" in solution
    if has_schedule == has_machine_schedules:
        raise InvalidSolution(
            "solution must contain exactly one of schedule or machine_schedules"
        )

    if has_schedule:
        starts = _starts_from_schedule(solution, jobs, jobs_by_id)
    else:
        num_machines = _integer(instance.get("num_machines"), "instance.num_machines")
        starts = _starts_from_machine_schedules(
            solution,
            jobs,
            jobs_by_id,
            num_machines,
        )

    reported_objective = _finite_number(
        solution.get("objective_value"),
        "solution.objective_value",
    )
    true_objective = 0.0
    normalized_schedule: list[dict[str, Any]] = []
    for job in jobs:
        job_id = job["job_id"]
        normalized_operations: list[dict[str, Any]] = []
        for operation_index, operation in enumerate(job["operations"]):
            machine, processing_time = _expected_operation(
                jobs_by_id,
                job_id,
                operation_index,
            )
            start_time = starts[(job_id, operation_index)]
            normalized_operations.append(
                {
                    "machine": machine,
                    "start_time": start_time,
                    "processing_time": processing_time,
                    "end_time": _completion_time(
                        start_time,
                        processing_time,
                        f"solution operation ({job_id}, {operation_index})",
                    ),
                }
            )
        completion_time = normalized_operations[-1]["end_time"]
        due_date = _finite_number(job.get("due_date"), f"job {job_id}.due_date")
        weight = _finite_number(job.get("weight"), f"job {job_id}.weight")
        tardiness = max(0.0, completion_time - due_date)
        true_objective += weight * tardiness
        if not math.isfinite(true_objective):
            raise InvalidSolution("solution produces a non-finite objective")
        normalized_schedule.append(
            {
                "job_id": job_id,
                "completion_time": completion_time,
                "tardiness": tardiness,
                "operations": normalized_operations,
            }
        )

    if not _close(reported_objective, true_objective):
        raise InvalidSolution(
            "objective_value does not match total weighted tardiness "
            f"(reported={reported_objective}, recomputed={true_objective})"
        )
    return {
        "objective_value": true_objective,
        "schedule": normalized_schedule,
    }


def _dataset_checker(instance_path: str) -> str:
    paper_dir = os.path.dirname(os.path.dirname(os.path.abspath(instance_path)))
    return os.path.join(paper_dir, "feasibility_check.py")


def _run_dataset_checker(
    checker_path: str,
    instance_path: str,
    normalized_solution: dict[str, Any],
) -> dict[str, Any]:
    if not os.path.isfile(checker_path):
        raise RuntimeError(f"dataset checker not found: {checker_path}")
    with tempfile.TemporaryDirectory(prefix="frontieror_bierwirth_checker_") as tmp:
        solution_path = os.path.join(tmp, "solution.json")
        result_path = os.path.join(tmp, "result.json")
        with open(solution_path, "w", encoding="utf-8") as handle:
            json.dump(normalized_solution, handle)
        completed = subprocess.run(
            [
                sys.executable,
                checker_path,
                "--instance_path",
                instance_path,
                "--solution_path",
                solution_path,
                "--result_path",
                result_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise RuntimeError(f"dataset checker failed: {detail}")
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)
    if not isinstance(result, dict) or not isinstance(result.get("feasible"), bool):
        raise RuntimeError("dataset checker returned an invalid result")
    return result


def _failure_result(message: str) -> dict[str, Any]:
    return {
        "feasible": False,
        "violated_constraints": ["solution_contract"],
        "violations": [message],
        "violation_magnitudes": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    try:
        instance = _load_object(args.instance_path, "instance")
        solution = _load_object(args.solution_path, "solution")
        normalized = _normalize_solution(instance, solution)
        result = _run_dataset_checker(
            _dataset_checker(args.instance_path),
            args.instance_path,
            normalized,
        )
    except (InvalidSolution, OSError, json.JSONDecodeError) as exc:
        result = _failure_result(str(exc))
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        result = _failure_result(f"trusted checker failure: {exc}")

    with open(args.result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
