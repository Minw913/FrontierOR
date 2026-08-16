"""Trusted FrontierOR worker for the Tide-eval Executor bridge."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
IDENTITY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class WorkerRequestError(ValueError):
    """Raised when an episode does not satisfy the FrontierOR contract."""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _attempt_trace(run_dir: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[tuple[float, Path, dict[str, Any]]] = []
    for path in run_dir.glob("**/public/attempts/*.json"):
        attempt = _read_json(path)
        if attempt is None or attempt.get("status") == "pending":
            continue
        timestamp = _parse_timestamp(attempt.get("timestamp"))
        records.append((timestamp or path.stat().st_mtime, path, attempt))
    records.sort(key=lambda item: (item[0], item[1].name))
    if not records:
        return [], 0
    started = records[0][0]
    trace = []
    for timestamp, _, attempt in records:
        score = attempt.get("score")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            continue
        trace.append(
            {
                "t": max(0.0, timestamp - started),
                "score": float(score),
                "data": {
                    "commit_hash": attempt.get("commit_hash"),
                    "agent_id": attempt.get("agent_id"),
                    "status": attempt.get("status"),
                },
            }
        )
    return trace, len(records)


def _openevolve_trace(run_dir: Path) -> list[dict[str, Any]]:
    records: list[tuple[float, Path, dict[str, Any]]] = []
    for path in run_dir.glob("**/checkpoints/*/best_program_info.json"):
        info = _read_json(path)
        if info is None:
            continue
        timestamp = _parse_timestamp(info.get("timestamp"))
        records.append((timestamp or path.stat().st_mtime, path, info))
    records.sort(key=lambda item: (item[0], item[1].as_posix()))
    if not records:
        return []
    started = records[0][0]
    trace = []
    for timestamp, _, info in records:
        metrics = info.get("metrics")
        score = metrics.get("combined_score") if isinstance(metrics, dict) else None
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            continue
        trace.append(
            {
                "t": max(0.0, timestamp - started),
                "score": float(score),
                "data": {
                    "iteration": info.get("iteration"),
                    "generation": info.get("generation"),
                    "program_id": info.get("id"),
                },
            }
        )
    return trace


def _eoh_trace(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    pattern = "**/results/pops_best/population_generation_*.json"
    for path in run_dir.glob(pattern):
        value = _read_json(path)
        objective = value.get("objective") if value else None
        if (
            not isinstance(objective, int | float)
            or isinstance(objective, bool)
            or not math.isfinite(float(objective))
        ):
            continue
        try:
            generation = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        records.append((generation, path, float(objective)))
    records.sort(key=lambda item: (item[0], item[1].as_posix()))
    return [
        {
            "t": float(generation),
            "score": -objective,
            "data": {"generation": generation},
        }
        for generation, _, objective in records
    ]


def _final_reward(
    *,
    root: Path,
    framework: str,
    run_id: str,
    paper_id: str,
    model_name: str,
    expected_instances: list[str],
) -> tuple[dict[str, float], list[str]]:
    csv_path = root / "eval" / f"eval_test_results_{framework}.csv"
    rows: dict[str, dict[str, str]] = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("run_id") == run_id
                    and row.get("paper_id") == paper_id
                    and row.get("model") == model_name
                ):
                    rows[str(row.get("instance"))] = row
    missing = [instance for instance in expected_instances if instance not in rows]
    scores = []
    for instance in expected_instances:
        raw = rows.get(instance, {}).get("score_staged")
        try:
            score = float(raw) if raw not in (None, "") else 0.0
        except ValueError:
            score = 0.0
        scores.append(score if math.isfinite(score) else 0.0)
    reward = sum(scores) / len(expected_instances) if expected_instances else 0.0
    return {"reward": reward}, missing


def _result_reward(
    results: Any, expected_instances: list[str]
) -> tuple[dict[str, float], list[str]]:
    result_map = results if isinstance(results, dict) else {}
    missing = [instance for instance in expected_instances if instance not in result_map]
    scores = []
    for instance in expected_instances:
        result = result_map.get(instance)
        raw = result.get("score") if isinstance(result, dict) else None
        try:
            score = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score if math.isfinite(score) else 0.0)
    reward = sum(scores) / len(expected_instances) if expected_instances else 0.0
    return {"reward": reward}, missing


def _artifact_sha256(run_dir: Path) -> str | None:
    code = run_dir / "selected_code.py"
    if not code.is_file():
        return None
    digest = hashlib.sha256()
    with code.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_episode(request: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(request, dict) or request.get("schema_version") != SCHEMA_VERSION:
        raise WorkerRequestError("unsupported or missing process schema_version")
    episode = request.get("episode")
    if not isinstance(episode, dict):
        raise WorkerRequestError("episode must be an object")
    agent = episode.get("agent")
    if not isinstance(agent, dict):
        raise WorkerRequestError("episode.agent must be an object")
    argv = agent.get("frontieror_argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise WorkerRequestError("agent.frontieror_argv must be a string array")
    return episode, argv


def _validate_identity(value: str, label: str) -> None:
    if IDENTITY_PATTERN.fullmatch(value) is None:
        raise WorkerRequestError(f"invalid {label}: {value!r}")


def execute_episode(request: Any, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    episode, argv = _validate_episode(request)
    agent = episode["agent"]
    entrypoint = agent.get("entrypoint")
    if entrypoint not in {"agent", "runner"}:
        raise WorkerRequestError("entrypoint must be 'agent' or 'runner'")

    from frontieror.infra.cli import _agent_parser
    from test_time_self_evolution import run_eval_modes

    if entrypoint == "agent":
        parsed = _agent_parser().parse_args(argv)
        framework = "coral"
        command = [sys.executable, "-m", "frontieror.infra", "agent", *argv]
    else:
        parsed = run_eval_modes.parse_args(argv)
        framework = parsed.framework
        if framework not in {"openevolve", "eoh"}:
            raise WorkerRequestError(
                "runner episodes support only OpenEvolve and EoH"
            )
        if parsed.modes != ["self_evolve"]:
            raise WorkerRequestError("runner episodes support only self_evolve mode")
        if not parsed.anti_hack or parsed.exec_mode != "docker":
            raise WorkerRequestError(
                "runner episodes require anti-hack Docker execution"
            )
        if parsed.stage2_scorer != "staged_qte":
            raise WorkerRequestError("runner episodes require staged_qte")
        command = [
            sys.executable,
            "-m",
            "test_time_self_evolution.run_eval_modes",
            *argv,
        ]

    if agent.get("name") != framework:
        raise WorkerRequestError(
            "episode agent name does not match the executed framework"
        )

    paper_ids = list(parsed.paper_ids or [])
    if len(paper_ids) != 1:
        raise WorkerRequestError("one Tide episode must contain exactly one paper")
    paper_id = paper_ids[0]
    _validate_identity(paper_id, "paper id")
    if episode.get("task") != f"frontieror/{paper_id}":
        raise WorkerRequestError("episode task does not match --paper-id")
    if not parsed.run_id:
        raise WorkerRequestError("Tide episodes require an explicit --run-id")
    _validate_identity(parsed.run_id, "run id")
    expected_instances = list(parsed.test_instances or [])
    if not expected_instances:
        raise WorkerRequestError("Tide episodes require a non-empty final test set")

    from one_shot_eval import get_model_short_name

    model_name = get_model_short_name(parsed.primary_model)
    run_dir = root / "eval" / framework / parsed.run_id / paper_id / model_name
    worker_dir = root / "eval" / "tide_eval_workers" / parsed.run_id / paper_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_path = worker_dir / f"{framework}.log"
    raw_result_path = worker_dir / "runner_episode_result.json"
    raw_result_path.unlink(missing_ok=True)
    runner_env = os.environ.copy()
    runner_env["FRONTIER_OR_EPISODE_RESULT_PATH"] = str(raw_result_path)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=root,
            env=runner_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    raw_result = _read_json(raw_result_path)
    result_payload = raw_result.get("result") if raw_result else None
    final_results = (
        result_payload.get("results") if isinstance(result_payload, dict) else None
    )
    rewards, missing = _result_reward(final_results, expected_instances)
    csv_rewards, csv_missing = _final_reward(
        root=root,
        framework=framework,
        run_id=parsed.run_id,
        paper_id=paper_id,
        model_name=model_name,
        expected_instances=expected_instances,
    )
    if framework == "coral":
        trace, submissions = _attempt_trace(run_dir)
    elif framework == "openevolve":
        trace = _openevolve_trace(run_dir)
        submissions = len(trace)
    else:
        trace = _eoh_trace(run_dir)
        submissions = len(trace)

    errors = []
    if proc.returncode != 0:
        errors.append(f"FrontierOR runner exited {proc.returncode}; see {log_path}")
    if missing:
        errors.append("runner result is missing final instances: " + ", ".join(missing))
    if raw_result is None:
        errors.append("FrontierOR runner did not write its episode result envelope")
    elif (
        raw_result.get("status") != "ok"
        or raw_result.get("run_id") != parsed.run_id
        or raw_result.get("paper_id") != paper_id
        or raw_result.get("framework") != framework
    ):
        errors.append(
            "invalid FrontierOR runner result envelope: "
            + str(raw_result.get("error") or raw_result.get("status"))
        )
    if not missing and (
        csv_missing or abs(csv_rewards["reward"] - rewards["reward"]) > 1e-6
    ):
        errors.append("runner result does not match the final reporting CSV")
    candidate_id = (
        result_payload.get("candidate_id") if isinstance(result_payload, dict) else None
    )
    if candidate_id in {"coral_seed_fail", "coral_no_attempt"}:
        errors.append(f"framework produced no eligible final artifact: {candidate_id}")
    if not run_dir.is_dir():
        errors.append(f"run artifact directory was not created: {run_dir}")

    artifact_dir = run_dir if run_dir.is_dir() else worker_dir
    result = {
        "rewards": rewards,
        "uri": str(artifact_dir.resolve()),
        "trace": trace,
        "usage": {"n_submissions": float(submissions)},
        "error": "; ".join(errors) or None,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FrontierOR",
        "orchestrator": "tide-eval",
        "evaluator_version": "1",
        "run_id": parsed.run_id,
        "paper_id": paper_id,
        "framework": framework,
        "model": model_name,
        "expected_final_instances": len(expected_instances),
        "missing_final_instances": len(missing),
        "artifact_sha256": _artifact_sha256(run_dir),
        "runner_log": str(log_path.resolve()),
        "result": result,
    }
    manifest_path = artifact_dir / "tide_episode_result.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, manifest_path)
    return {"schema_version": SCHEMA_VERSION, "result": result}


def main() -> int:
    try:
        request = json.load(sys.stdin)
        response = execute_episode(request)
    except (Exception, SystemExit) as exc:
        response = {
            "schema_version": SCHEMA_VERSION,
            "result": {
                "rewards": {},
                "trace": [],
                "usage": {},
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
