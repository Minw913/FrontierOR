"""Black-box release checks for the untrusted candidate boundary."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from frontieror.infra import execution
from frontieror.infra.checkers import run_checker_isolated
from frontieror.infra.files import read_regular_file


SCHEMA_VERSION = 1
PROBE_TIMEOUT_SECONDS = 5
MAX_PROBE_RESULT_BYTES = 64 * 1024


def _write(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")
    os.chmod(path, 0o600)


def _run_probe(
    root: Path,
    *,
    name: str,
    program: str,
    image_digest: str,
    time_limit: float,
) -> tuple[bool, str, float, Path]:
    probe_dir = root / name
    probe_dir.mkdir(mode=0o700)
    code_path = probe_dir / "code.py"
    instance_path = probe_dir / "instance.json"
    solution_path = probe_dir / "solution.json"
    log_path = probe_dir / "log.jsonl"
    _write(code_path, program)
    _write(instance_path, "{}\n")
    _write(solution_path, "")
    _write(log_path, "")
    success, output, elapsed = execution.run_docker(
        os.fspath(code_path),
        os.fspath(instance_path),
        os.fspath(solution_path),
        time_limit,
        log_path=os.fspath(log_path),
        cfg={
            "anti_hack": True,
            "docker_image": image_digest,
            "wls_egress": "off",
            "cpus": 1,
            "memory": "1G",
        },
    )
    return success, output, elapsed, solution_path


def run_security_checks(image_ref: str) -> dict[str, Any]:
    """Execute malicious probes through the same API used by official grading."""
    image_digest = execution.resolve_docker_image(image_ref)
    marker = "frontieror-" + secrets.token_hex(24)
    old_marker = os.environ.get("FRONTIER_OR_BLACKBOX_SECRET")
    os.environ["FRONTIER_OR_BLACKBOX_SECRET"] = marker
    probes: list[dict[str, Any]] = []

    try:
        with tempfile.TemporaryDirectory(prefix="frontieror-security-check-") as tmp:
            root = Path(tmp)
            host_secret_path = root / "host-only-canary.txt"
            _write(host_secret_path, marker)
            boundary_program = f'''\
import argparse
import json
import os
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--instance_path", required=True)
parser.add_argument("--solution_path", required=True)
parser.add_argument("--time_limit", required=True)
parser.add_argument("--log_path")
args = parser.parse_args()

observed = {{}}
try:
    with open({os.fspath(host_secret_path)!r}, encoding="utf-8") as handle:
        observed["host_file"] = handle.read()
except OSError:
    observed["host_file"] = "blocked"

observed["host_env"] = os.environ.get("FRONTIER_OR_BLACKBOX_SECRET", "blocked")
try:
    with open("/frontieror-blackbox-root-write", "w", encoding="utf-8") as handle:
        handle.write("unexpected")
    observed["root_write"] = "allowed"
except OSError:
    observed["root_write"] = "blocked"

try:
    connection = socket.create_connection(("example.com", 443), timeout=1)
    connection.close()
    observed["network"] = "allowed"
except OSError:
    observed["network"] = "blocked"

observed["objective_value"] = 0
with open(args.solution_path, "w", encoding="utf-8") as handle:
    json.dump(observed, handle)
'''
            success, output, _elapsed, solution_path = _run_probe(
                root,
                name="boundary",
                program=boundary_program,
                image_digest=image_digest,
                time_limit=PROBE_TIMEOUT_SECONDS,
            )
            boundary_passed = False
            detail = "candidate probe failed to execute"
            if success:
                try:
                    observed = json.loads(
                        read_regular_file(
                            solution_path,
                            max_bytes=MAX_PROBE_RESULT_BYTES,
                            label="security probe result",
                            require_single_link=True,
                        ).decode("utf-8")
                    )
                    boundary_passed = all(
                        observed.get(key) == "blocked"
                        for key in ("host_file", "host_env", "root_write", "network")
                    )
                    detail = (
                        "host filesystem, environment, root filesystem, and network blocked"
                        if boundary_passed
                        else "one or more isolation controls were bypassed"
                    )
                except (OSError, ValueError):
                    detail = "candidate probe returned an invalid solution"
            probes.append(
                {"id": "isolation_boundary", "passed": boundary_passed, "detail": detail}
            )

            checker_paper = root / "checker-paper"
            checker_instance = checker_paper / "instance" / "tiny_instance.json"
            checker_solution = root / "checker-solution.json"
            checker_result = root / "checker-result.json"
            checker_program = f'''\
import argparse
import json
import os
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--instance_path", required=True)
parser.add_argument("--solution_path", required=True)
parser.add_argument("--result_path", required=True)
args = parser.parse_args()

observed = {{}}
try:
    with open({os.fspath(host_secret_path)!r}, encoding="utf-8") as handle:
        observed["host_file"] = handle.read()
except OSError:
    observed["host_file"] = "blocked"
observed["host_env"] = os.environ.get("FRONTIER_OR_BLACKBOX_SECRET", "blocked")
try:
    with open("/frontieror-checker-root-write", "w", encoding="utf-8") as handle:
        handle.write("unexpected")
    observed["root_write"] = "allowed"
except OSError:
    observed["root_write"] = "blocked"
try:
    connection = socket.create_connection(("example.com", 443), timeout=1)
    connection.close()
    observed["network"] = "allowed"
except OSError:
    observed["network"] = "blocked"
observed["feasible"] = True
with open(args.result_path, "w", encoding="utf-8") as handle:
    json.dump(observed, handle)
'''
            checker_path = root / "checker.py"
            _write(checker_path, checker_program)
            checker_instance.parent.mkdir(parents=True, mode=0o700)
            _write(checker_instance, "{}\n")
            _write(checker_solution, '{"objective_value": 0}\n')
            checker_success, _checker_output, _checker_elapsed = (
                run_checker_isolated(
                    checker_path=os.fspath(checker_path),
                    paper_dir=os.fspath(checker_paper),
                    instance_file=os.fspath(checker_instance),
                    solution_file=os.fspath(checker_solution),
                    result_file=os.fspath(checker_result),
                    cfg={
                        "anti_hack": True,
                        "docker_image": image_digest,
                        "wls_egress": "off",
                    },
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
            )
            checker_passed = False
            checker_detail = "checker probe failed to execute"
            if checker_success:
                try:
                    checker_observed = json.loads(
                        read_regular_file(
                            checker_result,
                            max_bytes=MAX_PROBE_RESULT_BYTES,
                            label="checker security probe result",
                            require_single_link=True,
                        ).decode("utf-8")
                    )
                    checker_passed = all(
                        checker_observed.get(key) == "blocked"
                        for key in ("host_file", "host_env", "root_write", "network")
                    )
                    checker_detail = (
                        "checker host filesystem, environment, root, and network blocked"
                        if checker_passed
                        else "one or more checker isolation controls were bypassed"
                    )
                except (OSError, ValueError):
                    checker_detail = "checker probe returned an invalid result"
            probes.append(
                {
                    "id": "checker_boundary",
                    "passed": checker_passed,
                    "detail": checker_detail,
                }
            )

            timeout_program = '''\
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--instance_path", required=True)
parser.add_argument("--solution_path", required=True)
parser.add_argument("--time_limit", required=True)
parser.add_argument("--log_path")
parser.parse_args()
time.sleep(60)
'''
            success, output, elapsed, _ = _run_probe(
                root,
                name="timeout",
                program=timeout_program,
                image_digest=image_digest,
                time_limit=0.5,
            )
            timeout_passed = (
                not success
                and "timed out" in output.lower()
                and elapsed < 5
            )
            probes.append(
                {
                    "id": "wall_clock_timeout",
                    "passed": timeout_passed,
                    "detail": (
                        "candidate was terminated at the trusted deadline"
                        if timeout_passed
                        else "candidate exceeded or escaped the trusted deadline"
                    ),
                }
            )

            stdout_program = '''\
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--instance_path", required=True)
parser.add_argument("--solution_path", required=True)
parser.add_argument("--time_limit", required=True)
parser.add_argument("--log_path")
args = parser.parse_args()
sys.stdout.write("x" * (2 * 1024 * 1024))
with open(args.solution_path, "w", encoding="utf-8") as handle:
    json.dump({"objective_value": 0}, handle)
'''
            success, output, _elapsed, _ = _run_probe(
                root,
                name="stdout",
                program=stdout_program,
                image_digest=image_digest,
                time_limit=PROBE_TIMEOUT_SECONDS,
            )
            stdout_passed = (
                success
                and "output truncated" in output
                and len(output.encode("utf-8")) < 1_100_000
            )
            probes.append(
                {
                    "id": "bounded_output",
                    "passed": stdout_passed,
                    "detail": (
                        "candidate output was drained and bounded"
                        if stdout_passed
                        else "candidate output was not safely bounded"
                    ),
                }
            )
    finally:
        if old_marker is None:
            os.environ.pop("FRONTIER_OR_BLACKBOX_SECRET", None)
        else:
            os.environ["FRONTIER_OR_BLACKBOX_SECRET"] = old_marker

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_image": {"ref": image_ref, "digest": image_digest},
        "passed": all(probe["passed"] for probe in probes),
        "probes": probes,
    }
