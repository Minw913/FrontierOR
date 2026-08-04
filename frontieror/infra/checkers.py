"""Resolve and preflight trusted feasibility checkers."""

from __future__ import annotations

import functools
import json
import math
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

from frontieror.infra.files import read_regular_file
from frontieror.infra.paths import TRUSTED_CHECKER_ROOT
from scripts.utils.instance_paths import gurobi_solution_path, instance_path


MAX_CHECKER_RESULT_BYTES = 16 * 1024 * 1024
MAX_CHECKER_INPUT_BYTES = 256 * 1024 * 1024
DEFAULT_CHECKER_MEMORY = "8G"


def feasibility_checker_path(*, paper_dir: str, paper_id: str | None = None) -> str:
    """Prefer a versioned trusted overlay, falling back to the dataset checker."""
    resolved_paper_id = paper_id or os.path.basename(os.path.normpath(paper_dir))
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
        resolved_paper_id,
    ) is None:
        raise ValueError(f"invalid paper id: {resolved_paper_id!r}")

    overlay = os.path.join(
        os.fspath(TRUSTED_CHECKER_ROOT),
        resolved_paper_id,
        "feasibility_check.py",
    )
    if os.path.isfile(overlay):
        return overlay
    return os.path.join(os.path.abspath(paper_dir), "feasibility_check.py")


def build_isolated_checker_cmd(
    *,
    checker_path: str,
    paper_dir: str,
    instance_file: str,
    solution_file: str,
    result_file: str,
    cfg: dict,
) -> list[str]:
    """Build the no-network Docker command used for untrusted checker input."""
    from frontieror.infra import execution

    paper_root = Path(paper_dir).resolve()
    instance_path = Path(instance_file).resolve()
    try:
        relative_instance = instance_path.relative_to(paper_root)
    except ValueError as exc:
        raise ValueError("checker instance must be inside its trusted paper directory") from exc

    cpus = int(cfg.get("checker_cpus", 1))
    if cpus < 1:
        raise ValueError("checker_cpus must be positive")
    memory = str(cfg.get("checker_memory", DEFAULT_CHECKER_MEMORY))
    image = str(cfg.get("docker_image", execution.DEFAULT_DOCKER_IMAGE))
    restricted_network = cfg.get("_restricted_network")
    restricted_proxy = cfg.get("_restricted_proxy")
    if bool(restricted_network) != bool(restricted_proxy):
        raise ValueError(
            "restricted checker network and proxy must be configured together"
        )

    core_set = execution._allocate_cores(cpus)
    container_name = f"frontieror-candidate-checker-{uuid.uuid4().hex}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--log-driver=none",
        f"--user={os.getuid()}:{os.getgid()}",
        f"--cpus={cpus}",
        f"--memory={memory}",
        f"--memory-swap={memory}",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--stop-timeout",
        "1",
        "--ulimit",
        f"fsize={MAX_CHECKER_RESULT_BYTES}:{MAX_CHECKER_RESULT_BYTES}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g",
    ]
    if restricted_network:
        command.extend(["--network", str(restricted_network)])
    else:
        command.append("--network=none")
    if core_set:
        command.append(f"--cpuset-cpus={core_set}")

    command.extend(
        [
            "--mount",
            f"type=bind,src={paper_root},dst=/workspace/paper,readonly",
            "--mount",
            (
                f"type=bind,src={Path(checker_path).resolve()},"
                "dst=/workspace/checker.py,readonly"
            ),
            "--mount",
            (
                f"type=bind,src={Path(solution_file).resolve()},"
                "dst=/workspace/solution.json,readonly"
            ),
            "--mount",
            (
                f"type=bind,src={Path(result_file).resolve()},"
                "dst=/workspace/result.json"
            ),
        ]
    )

    license_path = str(
        cfg.get("gurobi_lic", os.environ.get("GRB_LICENSE_FILE", ""))
    )
    if license_path and Path(license_path).is_file():
        command.extend(
            [
                "--mount",
                (
                    f"type=bind,src={Path(license_path).resolve()},"
                    "dst=/opt/gurobi/gurobi.lic,readonly"
                ),
                "-e",
                "GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic",
            ]
        )
    command.extend(["-e", "PYTHONDONTWRITEBYTECODE=1"])
    if restricted_proxy:
        command.extend(
            [
                "-e",
                f"HTTPS_PROXY={restricted_proxy}",
                "-e",
                f"HTTP_PROXY={restricted_proxy}",
                "-e",
                "NO_PROXY=localhost,127.0.0.1",
            ]
        )
    command.extend(
        [
            image,
            "python",
            "/workspace/checker.py",
            "--instance_path",
            f"/workspace/paper/{relative_instance.as_posix()}",
            "--solution_path",
            "/workspace/solution.json",
            "--result_path",
            "/workspace/result.json",
        ]
    )
    return command


def run_checker_isolated(
    *,
    checker_path: str,
    paper_dir: str,
    instance_file: str,
    solution_file: str,
    result_file: str,
    cfg: dict,
    timeout: int = 60,
) -> tuple[bool, str, float]:
    """Evaluate untrusted solution data in a separate trusted-checker sandbox."""
    from frontieror.infra import execution
    from frontieror.infra.files import sha256_regular_file

    sha256_regular_file(
        solution_file,
        max_bytes=MAX_CHECKER_INPUT_BYTES,
        label="candidate solution for checker",
    )
    result_path = Path(result_file)
    if os.path.lexists(result_path):
        raise ValueError("checker result path must not already exist")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        result_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    os.close(descriptor)

    effective_cfg = dict(cfg)
    configured_license = str(
        effective_cfg.get("gurobi_lic", os.environ.get("GRB_LICENSE_FILE", ""))
    )
    if configured_license:
        effective_cfg["gurobi_lic"] = execution._canonical_license_path(
            configured_license
        )
    try:
        with execution._wls_execution_slot(effective_cfg):
            with execution._restricted_wls_egress(effective_cfg) as egress:
                redactions: tuple[str, ...] = ()
                if egress is not None:
                    effective_cfg["_restricted_network"] = egress["network"]
                    effective_cfg["_restricted_proxy"] = egress["proxy_url"]
                    redactions = egress["redactions"]
                command = build_isolated_checker_cmd(
                    checker_path=checker_path,
                    paper_dir=paper_dir,
                    instance_file=instance_file,
                    solution_file=solution_file,
                    result_file=result_file,
                    cfg=effective_cfg,
                )
                success, output, elapsed = execution._exec(command, timeout)
                return success, execution._redact_values(output, redactions), elapsed
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"Isolated checker setup failed: {exc}", 0.0


def _run_checker(
    checker_path: str,
    instance_file: str,
    solution_file: str,
    result_file: str,
) -> dict:
    from frontieror.infra.execution import _exec

    success, output, _ = _exec(
        [
            sys.executable,
            checker_path,
            "--instance_path",
            instance_file,
            "--solution_path",
            solution_file,
            "--result_path",
            result_file,
        ],
        60,
    )
    if not success:
        raise ValueError(
            "feasibility checker failed its security preflight: "
            + output.strip()[:500]
        )
    try:
        payload = json.loads(
            read_regular_file(
                result_file,
                max_bytes=MAX_CHECKER_RESULT_BYTES,
                label="feasibility checker preflight result",
            ).decode("utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(
            "feasibility checker did not produce a valid preflight result"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("feasible"), bool):
        raise ValueError(
            "feasibility checker preflight result must contain a boolean feasible field"
        )
    return payload


@functools.lru_cache(maxsize=1024)
def _validate_objective_checker_cached(paper_dir: str, instance: str) -> None:
    checker = feasibility_checker_path(paper_dir=paper_dir)
    instance_file = instance_path(paper_dir, instance)
    reference_file = gurobi_solution_path(paper_dir, instance)
    for label, path in (
        ("feasibility checker", checker),
        ("instance", instance_file),
        ("reference solution", reference_file),
    ):
        if not Path(path).is_file():
            raise ValueError(f"agent preflight is missing {label}: {path}")

    try:
        reference = json.loads(
            read_regular_file(
                reference_file,
                max_bytes=256 * 1024 * 1024,
                label="reference solution",
                require_single_link=False,
            ).decode("utf-8")
        )
        reference_obj = float(reference["objective_value"])
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(
            "reference solution must be valid JSON with a finite objective_value"
        ) from exc
    if not isinstance(reference, dict) or not math.isfinite(reference_obj):
        raise ValueError(
            "reference solution must be valid JSON with a finite objective_value"
        )

    tamper_delta = max(1.0, abs(reference_obj) * 0.02)
    with tempfile.TemporaryDirectory(prefix="frontieror_checker_preflight_") as tmp:
        baseline = _run_checker(
            checker,
            instance_file,
            reference_file,
            os.path.join(tmp, "reference_result.json"),
        )
        if baseline["feasible"] is not True:
            raise ValueError("feasibility checker rejects the trusted reference solution")

        for label, forged_obj in (
            ("higher", reference_obj + tamper_delta),
            ("lower", reference_obj - tamper_delta),
        ):
            tampered = dict(reference)
            tampered["objective_value"] = forged_obj
            solution_file = os.path.join(tmp, f"tampered_{label}_solution.json")
            result_file = os.path.join(tmp, f"tampered_{label}_result.json")
            Path(solution_file).write_text(json.dumps(tampered), encoding="utf-8")
            if _run_checker(checker, instance_file, solution_file, result_file)["feasible"]:
                raise ValueError(
                    "feasibility checker does not reject a forged objective_value"
                )


def validate_objective_checker(*, paper_dir: str, instance: str) -> None:
    """Require the checker to bind reported objective to decision variables."""
    _validate_objective_checker_cached(os.path.realpath(paper_dir), str(instance))
