"""Canonical CLI for trusted FrontierOR evaluation infrastructure."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from .contracts import public_scoring_contract, visibility_contract
from .policy import hardened_agent_argv


def _agent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m frontieror.infra agent",
        description=(
            "Run a CORAL agent system with the FrontierOR trusted evaluation "
            "profile. Docker isolation, brokered dev evaluation, staged_qte, "
            "credential-isolating model access, and hidden final grading are "
            "mandatory."
        ),
    )
    parser.add_argument("--paper-id", nargs="+", dest="paper_ids")
    parser.add_argument("--primary-model", default=None)
    parser.add_argument("--secondary-model", default=None)
    parser.add_argument("--stage1-instances", nargs="+")
    parser.add_argument("--dev-set", nargs="+", dest="stage2_instances")
    parser.add_argument("--test-set", nargs="+", dest="test_instances")
    parser.add_argument("--stage1-time-limit", type=int)
    parser.add_argument("--stage2-time-limit", type=int)
    parser.add_argument("--test-time-limit", type=int)
    parser.add_argument("--stage1-gap-threshold", type=float)
    parser.add_argument("--stage2-stage-boundary", type=float)
    parser.add_argument(
        "--stage2-time-policy",
        choices=("uniform", "gurobi_time", "gurobi_time_plus_buffer"),
    )
    parser.add_argument("--stage2-time-buffer", type=int)
    parser.add_argument(
        "--test-time-policy",
        choices=("uniform", "gurobi_time", "gurobi_time_plus_buffer"),
    )
    parser.add_argument("--test-time-buffer", type=int)
    parser.add_argument("--coral-attempts", type=int)
    parser.add_argument("--coral-max-seconds")
    parser.add_argument("--coral-attempts-budget-multiplier", type=float)
    parser.add_argument("--coral-agent-count", type=int)
    parser.add_argument("--coral-agent-model")
    parser.add_argument("--coral-max-steps", type=int)
    parser.add_argument("--coral-max-turns", type=int)
    parser.add_argument("--coral-heartbeat-reflect-every", type=int)
    parser.add_argument("--coral-heartbeat-pivot-every", type=int)
    parser.add_argument("--coral-heartbeat-consolidate-every", type=int)
    parser.add_argument("--paper-workers", type=int)
    parser.add_argument("--dev-instance-workers", type=int)
    parser.add_argument("--test-instance-workers", type=int)
    parser.add_argument("--wls-egress", choices=("auto", "off", "required"))
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory")
    parser.add_argument("--t_max")
    parser.add_argument("--run-id")
    return parser


def _run_agent(argv: Sequence[str]) -> int:
    # Parse once with the narrow public surface before forwarding to the
    # upstream runner. Unknown or unsafe legacy flags fail here.
    _agent_parser().parse_args(list(argv))
    try:
        forwarded = hardened_agent_argv(argv)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    os.environ["FRONTIER_OR_EVALUATION_PROFILE"] = "agent"
    from test_time_self_evolution import run_eval_modes

    return int(run_eval_modes.main(forwarded) or 0)


def _show_contract() -> int:
    print(
        json.dumps(
            {
                "scoring": public_scoring_contract(),
                "visibility": visibility_contract(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_submission(argv: Sequence[str]) -> int:
    from frontieror.infra.submission.cli import main as submission_main

    return int(submission_main(list(argv)) or 0)


def _run_security_check(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m frontieror.infra security-check",
        description="Run black-box attacks against the candidate Docker boundary.",
    )
    parser.add_argument("--candidate-image", default="frontieror-candidate:1")
    args = parser.parse_args(list(argv))
    from .security_check import run_security_checks

    try:
        report = run_security_checks(args.candidate_image)
    except (OSError, RuntimeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "passed": False,
            "error": str(exc),
            "probes": [],
        }
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m frontieror.infra",
        description="Trusted FrontierOR agent and submission evaluation.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("agent", "submission", "contract", "security-check"),
    )
    if not raw or raw[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    args = parser.parse_args(raw[:1])
    remainder = raw[1:]
    if args.command == "agent":
        return _run_agent(remainder)
    if args.command == "submission":
        return _run_submission(remainder)
    if args.command == "security-check":
        return _run_security_check(remainder)
    if remainder:
        parser.error("contract does not accept additional arguments")
    return _show_contract()
