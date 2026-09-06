"""Trace and verdict helpers for official FrontierOR submissions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: os.PathLike[str] | str, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: os.PathLike[str] | str, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_trace(
    *,
    submission_id: str,
    paper_id: str,
    instance: str,
    stage: str,
    result: Dict[str, Any],
    gurobi_obj: float | None,
    gurobi_time: float | None,
    gap: float | None,
    score: float,
    score_debug: Dict[str, Any],
    exec_backend: str,
    exec_cfg: Dict[str, Any],
    code_sha256: str,
    time_limit: int,
    scorer: str,
    timestamp: str,
) -> Dict[str, Any]:
    return {
        "submission_id": submission_id,
        "paper_id": paper_id,
        "instance": instance,
        "stage": stage,
        "status": result.get("status"),
        "fail_reason": result.get("fail_reason"),
        "error": result.get("error"),
        "feasible": result.get("feasible"),
        "objective": result.get("llm_obj"),
        "reference_objective": gurobi_obj,
        "reference_time": gurobi_time,
        "gap": gap,
        "solve_time": result.get("solve_time"),
        "time_limit": time_limit,
        "scorer": scorer,
        "score": score,
        "score_debug": score_debug,
        "aocc": result.get("aocc"),
        "exec_backend": exec_backend,
        "exec_cfg": dict(exec_cfg),
        "code_sha256": code_sha256,
        "timestamp": timestamp,
    }


def build_verdict(
    *,
    submission_id: str,
    valid: bool,
    aggregate_score: float,
    traces: List[Dict[str, Any]],
    scorer: str,
    timestamp: str,
) -> Dict[str, Any]:
    total = len(traces)
    feasible_count = sum(1 for trace in traces if trace.get("feasible") is True)
    passed_count = sum(1 for trace in traces if trace.get("status") == "pass")
    return {
        "submission_id": submission_id,
        "valid": bool(valid),
        "aggregate_score": round(float(aggregate_score), 6),
        "scorer": scorer,
        "total_instances": total,
        "passed_count": passed_count,
        "feasible_count": feasible_count,
        "timestamp": timestamp,
    }


def build_public_leaderboard_row(
    *,
    bundle_metadata: Dict[str, Any],
    verdict: Dict[str, Any],
    code_sha256: str,
) -> Dict[str, Any]:
    return {
        "submission_id": verdict["submission_id"],
        "paper_id": bundle_metadata["paper_id"],
        "author": bundle_metadata["author"],
        "model_or_agent": bundle_metadata["model_or_agent"],
        "framework": bundle_metadata["framework"],
        "track": bundle_metadata["track"],
        "code_sha256": code_sha256,
        "aggregate_score": verdict["aggregate_score"],
        "scorer": verdict["scorer"],
        "total_instances": verdict["total_instances"],
        "passed_count": verdict["passed_count"],
        "feasible_count": verdict["feasible_count"],
        "verified_at": verdict["timestamp"],
    }
