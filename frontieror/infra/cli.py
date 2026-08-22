"""Canonical CLI for trusted FrontierOR evaluation infrastructure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from .contracts import public_scoring_contract, visibility_contract
from .policy import hardened_agent_argv


TIDE_EVALUATOR_VERSION = "1"


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


def _replace_option(
    argv: Sequence[str], option: str, values: Sequence[str]
) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        output.append(token)
        index += 1
    return [*output, option, *values]


def _run_tide_eval(argv: Sequence[str]) -> int:
    raw = list(argv)
    if "--" not in raw:
        if any(token in {"-h", "--help"} for token in raw):
            raw = [*raw, "--"]
        else:
            raise SystemExit(
                "ERROR: tide-eval options and FrontierOR options must be separated by --"
            )
    divider = raw.index("--")
    own, forwarded = raw[:divider], raw[divider + 1 :]
    parser = argparse.ArgumentParser(
        prog="python -m frontieror.infra tide-eval",
        description=(
            "Run hardened CORAL, OpenEvolve, or EoH episodes under "
            "Tide-eval orchestration."
        ),
    )
    parser.add_argument(
        "--framework",
        choices=("coral", "openevolve", "eoh"),
        required=True,
    )
    parser.add_argument(
        "--tide-python",
        default=os.environ.get("TIDE_EVAL_PYTHON"),
        help="Python executable from an environment containing tide-eval.",
    )
    parser.add_argument("--lab", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--budget-hours", type=float)
    options = parser.parse_args(own)
    if not options.tide_python:
        parser.error("--tide-python or TIDE_EVAL_PYTHON is required")
    if options.concurrency < 1:
        parser.error("--concurrency must be positive")
    if options.budget_hours is not None and options.budget_hours <= 0:
        parser.error("--budget-hours must be positive")

    if options.framework == "coral":
        parsed = _agent_parser().parse_args(forwarded)
        entrypoint = "agent"
    else:
        from test_time_self_evolution import run_eval_modes

        forwarded = _replace_option(
            forwarded, "--framework", [options.framework]
        )
        forwarded = _replace_option(forwarded, "--modes", ["self_evolve"])
        forwarded = _replace_option(forwarded, "--exec-mode", ["docker"])
        forwarded = _replace_option(
            forwarded, "--stage2-scorer", ["staged_qte"]
        )
        if "--anti-hack" not in forwarded:
            forwarded.append("--anti-hack")
        parsed = run_eval_modes.parse_args(forwarded)
        entrypoint = "runner"
        if parsed.resume:
            parser.error("Tide-eval owns resume; do not pass FrontierOR --resume")
    if not parsed.test_instances:
        parser.error("Tide episodes require an explicit FrontierOR --test-set")

    paper_ids = list(parsed.paper_ids or [])
    if not paper_ids:
        parser.error("FrontierOR --paper-id is required")
    run_id = parsed.run_id or f"tide-eval-{time.strftime('%Y%m%d-%H%M%S')}"
    identity_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    if identity_pattern.fullmatch(run_id) is None:
        parser.error(
            "FrontierOR --run-id must be 1-128 ASCII letters, digits, '.', '_', "
            "or '-', and must start with a letter or digit"
        )
    invalid_papers = [
        paper_id
        for paper_id in paper_ids
        if identity_pattern.fullmatch(paper_id) is None
    ]
    if invalid_papers:
        parser.error(f"invalid FrontierOR paper id: {invalid_papers[0]!r}")
    primary_model = parsed.primary_model or "gpt-5.3-codex"
    budget = {
        key: value
        for key, value in {
            "time_h": options.budget_hours,
            "max_submissions": (
                parsed.coral_attempts if options.framework == "coral" else None
            ),
        }.items()
        if value is not None
    }

    calls = []
    for paper_id in paper_ids:
        episode_argv = _replace_option(forwarded, "--paper-id", [paper_id])
        episode_argv = _replace_option(episode_argv, "--run-id", [run_id])
        episode_argv = _replace_option(
            episode_argv, "--primary-model", [primary_model]
        )
        episode_argv = _replace_option(episode_argv, "--paper-workers", ["1"])
        config_digest = hashlib.sha256(
            json.dumps(
                {
                    "evaluator_version": TIDE_EVALUATOR_VERSION,
                    "argv": episode_argv,
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
        calls.append(
            {
                "task": f"frontieror/{paper_id}",
                "agent": {
                    "name": options.framework,
                    "entrypoint": entrypoint,
                    "frontieror_argv": episode_argv,
                },
                "tags": {
                    "benchmark": "FrontierOR",
                    "framework": options.framework,
                    "model": primary_model,
                    "paper_id": paper_id,
                    "run_id": run_id,
                    "evaluator_version": TIDE_EVALUATOR_VERSION,
                },
                "budget": budget or None,
                "key": (
                    f"frontieror:{run_id}:{paper_id}:{options.framework}:"
                    f"{config_digest}"
                ),
            }
        )

    tide_python = Path(os.path.abspath(os.path.expanduser(options.tide_python)))
    if not tide_python.is_file():
        parser.error(f"Tide-eval Python does not exist: {tide_python}")
    tide_checkout = next(
        (
            parent
            for parent in tide_python.parents
            if (parent / "pyproject.toml").is_file()
            and (parent / "tide" / "__init__.py").is_file()
        ),
        None,
    )
    repo_root = Path(__file__).resolve().parents[2]
    request = {
        "schema_version": 1,
        "repo_root": str(repo_root),
        "lab": str(Path(options.lab).expanduser().resolve()),
        "concurrency": options.concurrency,
        "worker_command": [
            sys.executable,
            "-m",
            "frontieror.infra.tide_eval_worker",
        ],
        "calls": calls,
    }
    request_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="frontieror-tide-eval-",
            suffix=".json",
            delete=False,
        ) as handle:
            json.dump(request, handle)
            request_path = handle.name
        driver = Path(__file__).with_name("tide_eval_driver.py")
        driver_env = os.environ.copy()
        if tide_checkout is not None:
            existing_pythonpath = driver_env.get("PYTHONPATH")
            driver_env["PYTHONPATH"] = os.pathsep.join(
                value
                for value in (str(tide_checkout), existing_pythonpath)
                if value
            )
        proc = subprocess.run(
            [str(tide_python), str(driver), request_path],
            cwd=tide_checkout or repo_root,
            env=driver_env,
            check=False,
        )
        return proc.returncode
    finally:
        if request_path is not None:
            Path(request_path).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m frontieror.infra",
        description="Trusted FrontierOR agent and submission evaluation.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("agent", "submission", "contract", "security-check", "tide-eval"),
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
    if args.command == "tide-eval":
        return _run_tide_eval(remainder)
    if remainder:
        parser.error("contract does not accept additional arguments")
    return _show_contract()
