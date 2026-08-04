#!/usr/bin/env python3
"""Official local submission verifier for FrontierOR final-code bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import one_shot_eval as eval_core
from frontieror.infra.checkers import (
    feasibility_checker_path,
    validate_objective_checker,
)
from frontieror.infra.paths import REPO_ROOT
from frontieror.infra.policy import (
    DEFAULT_AGENT_DOCKER_IMAGE,
    validate_anti_hack_runtime,
)
from frontieror.infra.submission.bundle import (
    SubmissionBundleError,
    load_submission_bundle,
)
from frontieror.infra.submission.trace import (
    build_public_leaderboard_row,
    build_trace,
    build_verdict,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from frontieror.infra.execution import resolve_docker_image, validate_docker_wls
from scripts.utils.instance_paths import (
    gurobi_solution_path,
    instance_path,
    parse_instances_arg,
)
from test_time_self_evolution.scoring import ScoreContext, get_scorer


ROOT_DIR = REPO_ROOT


def _sha256_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT_DIR), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _build_verifier_provenance(
    *,
    paper_id: str,
    final_instances: List[str],
    scorer: str,
    docker_image_ref: str,
    docker_image_id: str,
) -> Dict[str, Any]:
    paper_dir = Path(eval_core.get_paper_dir(paper_id))
    checker = Path(
        feasibility_checker_path(
            paper_dir=str(paper_dir),
            paper_id=paper_id,
        )
    )
    scorer_source = ROOT_DIR / "test_time_self_evolution" / "scoring" / f"{scorer}.py"
    artifacts = []
    for instance in final_instances:
        instance_file = Path(instance_path(str(paper_dir), instance))
        reference_file = Path(gurobi_solution_path(str(paper_dir), instance))
        artifacts.append(
            {
                "instance": instance,
                "instance_sha256": _sha256_if_file(instance_file),
                "reference_sha256": _sha256_if_file(reference_file),
            }
        )
    dirty_output = _git_output("status", "--porcelain", "--untracked-files=no")
    return {
        "infra_commit": _git_output("rev-parse", "HEAD"),
        "infra_dirty": None if dirty_output is None else bool(dirty_output),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "available_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "docker_image_ref": docker_image_ref,
        "docker_image_id": docker_image_id,
        "checker_path": str(checker.relative_to(ROOT_DIR)) if checker.is_relative_to(ROOT_DIR) else str(checker),
        "checker_sha256": _sha256_if_file(checker),
        "scorer": scorer,
        "scorer_sha256": _sha256_if_file(scorer_source),
        "hidden_artifacts": artifacts,
    }


def _reference_row(
    *,
    paper_dir: str,
    instance: str,
    csv_row: Dict[str, Any],
) -> Dict[str, float | None]:
    row: Dict[str, float | None] = {
        "solution": csv_row.get("solution"),
        "time": csv_row.get("time"),
    }
    if row["solution"] is not None and row["time"] is not None:
        return row
    reference_path = Path(gurobi_solution_path(paper_dir, instance))
    try:
        with reference_path.open(encoding="utf-8") as handle:
            reference = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot load reference solution for {instance}: {exc}") from exc
    if not isinstance(reference, dict):
        raise ValueError(f"reference solution for {instance} must be a JSON object")
    if row["solution"] is None:
        try:
            objective = float(reference["objective_value"])
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"reference solution for {instance} has no numeric objective_value"
            ) from exc
        if not math.isfinite(objective):
            raise ValueError(
                f"reference solution for {instance} has a non-finite objective_value"
            )
        row["solution"] = objective
    if row["time"] is None:
        for field in ("wall_time", "solve_time", "runtime"):
            raw_time = reference.get(field)
            if raw_time is None:
                continue
            try:
                reference_time = float(raw_time)
            except (OverflowError, TypeError, ValueError):
                continue
            if math.isfinite(reference_time) and reference_time > 0:
                row["time"] = reference_time
                break
    return row


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a FrontierOR official final-code submission bundle.",
    )
    parser.add_argument("submission_dir", help="Directory containing submission.json and code.py")
    parser.add_argument("--paper-id", required=True, help="Paper ID to evaluate")
    parser.add_argument(
        "--final-instances",
        nargs="*",
        default=[],
        help="Held-out final instances used for official scoring",
    )
    parser.add_argument("--output-root", default="eval/submissions")
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory", default="640G")
    parser.add_argument("--docker-image", default=DEFAULT_AGENT_DOCKER_IMAGE)
    parser.add_argument(
        "--wls-egress",
        choices=["auto", "off", "required"],
        default="off",
        help=(
            "Gurobi WLS policy. Official third-party verification defaults to "
            "'off'; controlled internal runs may opt into 'auto' or 'required'."
        ),
    )
    parser.add_argument("--scorer", default="staged_qte", choices=["staged_qte", "aocc"])
    parser.add_argument("--stage-boundary", type=float, default=0.01)
    parser.add_argument("--t-max", type=float, default=None)
    return parser.parse_args(argv)


def _score_result(
    *,
    paper_id: str,
    instance: str,
    result: Dict[str, Any],
    gurobi_row: Dict[str, Any],
    time_limit: int,
    scorer_name: str,
    stage_boundary: float,
) -> tuple[float, Dict[str, Any], float | None]:
    direction = eval_core.get_paper_direction(paper_id)
    gurobi_obj = gurobi_row.get("solution")
    gurobi_time = gurobi_row.get("time")
    gap = eval_core.compute_gap(result.get("llm_obj"), gurobi_obj, direction=direction)

    kwargs = {"stage_boundary": stage_boundary} if scorer_name == "staged_qte" else {}
    scorer = get_scorer(scorer_name, **kwargs)
    score, debug = scorer.score_instance(
        result,
        ScoreContext(
            time_limit=time_limit,
            gurobi_time=gurobi_time,
            gurobi_obj=gurobi_obj,
            direction=direction,
            paper_id=paper_id,
            instance=instance,
        ),
    )
    return float(score), debug, gap


def _aggregate_scores(scores: List[float], scorer_name: str, stage_boundary: float) -> float:
    kwargs = {"stage_boundary": stage_boundary} if scorer_name == "staged_qte" else {}
    scorer = get_scorer(scorer_name, **kwargs)
    return round(float(scorer.aggregate(scores)), 6)


def run_submission(args: argparse.Namespace) -> int:
    if not args.final_instances:
        print("ERROR: --final-instances must contain at least one instance", file=sys.stderr)
        return 2
    try:
        final_instances = parse_instances_arg(args.final_instances)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        validate_anti_hack_runtime(
            enabled=True,
            exec_mode="docker",
            final_test_instances=final_instances,
            scorer=args.scorer,
        )
        bundle = load_submission_bundle(
            args.submission_dir,
            expected_paper_id=args.paper_id,
        )
        for instance in final_instances:
            validate_objective_checker(
                paper_dir=eval_core.get_paper_dir(args.paper_id),
                instance=instance,
            )
        docker_image_id = resolve_docker_image(args.docker_image)
    except (SubmissionBundleError, RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    exec_cfg = {
        "cpus": args.cpus,
        "memory": args.memory,
        "anti_hack": True,
        "docker_image": docker_image_id,
        "docker_image_ref": args.docker_image,
        "wls_egress": args.wls_egress,
    }
    try:
        validate_docker_wls(exec_cfg)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / bundle.submission_id
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        print(
            f"ERROR: evaluation directory already exists: {run_dir}",
            file=sys.stderr,
        )
        return 2
    private_dir = run_dir / "private"
    public_dir = run_dir / "public"
    artifacts_dir = private_dir / "artifacts"
    for path in (artifacts_dir, public_dir):
        path.mkdir(parents=True, exist_ok=True)

    staged_code = private_dir / "code.py"
    staged_code.write_bytes(bundle.code_bytes)
    staged_code.chmod(0o600)

    provenance = _build_verifier_provenance(
        paper_id=args.paper_id,
        final_instances=final_instances,
        scorer=args.scorer,
        docker_image_ref=args.docker_image,
        docker_image_id=docker_image_id,
    )
    results = eval_core.evaluate_candidate_code(
        args.paper_id,
        bundle.metadata["model_or_agent"],
        final_instances,
        str(staged_code),
        str(artifacts_dir),
        args.time_limit,
        "docker",
        exec_cfg,
        args.t_max,
    )

    gurobi_data = eval_core.load_gurobi_csv_data(args.paper_id, quiet=True)
    timestamp = utc_now_iso()
    traces = []
    scores = []
    paper_dir = eval_core.get_paper_dir(args.paper_id)
    for instance in final_instances:
        result = dict(results.get(instance) or {})
        if not result:
            result = {
                "status": "missing",
                "fail_reason": "missing_result",
                "error": "Evaluator did not return a result for this instance",
                "feasible": None,
            }
        gurobi_row = _reference_row(
            paper_dir=paper_dir,
            instance=instance,
            csv_row=dict(gurobi_data.get(instance) or {}),
        )
        score, debug, gap = _score_result(
            paper_id=args.paper_id,
            instance=instance,
            result=result,
            gurobi_row=gurobi_row,
            time_limit=args.time_limit,
            scorer_name=args.scorer,
            stage_boundary=args.stage_boundary,
        )
        scores.append(score)
        traces.append(
            build_trace(
                submission_id=bundle.submission_id,
                paper_id=args.paper_id,
                instance=instance,
                stage="final",
                result=result,
                gurobi_obj=gurobi_row.get("solution"),
                gurobi_time=gurobi_row.get("time"),
                gap=gap,
                score=score,
                score_debug=debug,
                exec_backend="docker",
                exec_cfg=exec_cfg,
                code_sha256=bundle.code_sha256,
                time_limit=args.time_limit,
                scorer=args.scorer,
                timestamp=timestamp,
            )
        )

    aggregate_score = _aggregate_scores(scores, args.scorer, args.stage_boundary)
    valid = all(trace.get("status") == "pass" and trace.get("feasible") is True for trace in traces)
    verdict = build_verdict(
        submission_id=bundle.submission_id,
        valid=valid,
        aggregate_score=aggregate_score,
        traces=traces,
        scorer=args.scorer,
        timestamp=timestamp,
    )
    public_row = build_public_leaderboard_row(
        bundle_metadata=bundle.metadata,
        verdict=verdict,
        code_sha256=bundle.code_sha256,
    )
    public_row["infra_commit"] = provenance["infra_commit"]
    public_row["docker_image_id"] = provenance["docker_image_id"]

    manifest = dict(bundle.manifest)
    manifest.update({
        "final_instance_count": len(final_instances),
        "exec_backend": "docker",
        "exec_cfg": exec_cfg,
        "scorer": args.scorer,
        "time_limit": args.time_limit,
        "verified_at": timestamp,
        "verifier_provenance": provenance,
    })
    write_json(private_dir / "submission_manifest.json", manifest)
    write_jsonl(private_dir / "traces.jsonl", traces)
    write_json(private_dir / "verdict.json", verdict)
    write_json(public_dir / "leaderboard_row.json", public_row)

    print(f"Submission ID: {bundle.submission_id}")
    print(f"Verdict: {'valid' if verdict['valid'] else 'invalid'}")
    print(f"Aggregate score: {verdict['aggregate_score']:.6f}")
    print(f"Artifacts: {run_dir}")
    return 0 if verdict["valid"] else 1


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_submission(args)


if __name__ == "__main__":
    raise SystemExit(main())
