"""
Execution backends for running LLM-generated code with resource limits.

Three modes:
  - "bare":        Direct subprocess, no resource limits (default, for debugging)
  - "systemd":     systemd-run with CPU/memory cgroups (lightweight, Linux only)
  - "docker":      Docker container with resource limits (fully isolated, reproducible)

All backends share the same interface:
    (success, output, elapsed) = run(code_path, instance_path, solution_path,
                                     time_limit, log_path, cfg)
"""

import contextlib
import fcntl
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from frontieror.infra.contracts import CANDIDATE_SHUTDOWN_RESERVE_SECONDS

# Default resource limits
DEFAULT_CPUS = 1          # number of CPU cores
DEFAULT_MEMORY = "32G"    # memory limit (uppercase for systemd compatibility)
DEFAULT_DOCKER_IMAGE = "frontier-or"
DEFAULT_CAPTURE_BYTES = 1024 * 1024
DEFAULT_OUTPUT_FILE_BYTES = 256 * 1024 * 1024
MAX_LICENSE_FILE_BYTES = 64 * 1024
WLS_REQUIRED_FIELDS = frozenset({"WLSACCESSID", "WLSSECRET", "LICENSEID"})
WLS_EGRESS_MODES = frozenset({"auto", "off", "required"})
WLS_TOKEN_HOST = "token.gurobi.com"
DEFAULT_WLS_CONCURRENCY = 0
MAX_WLS_CONCURRENCY = 64


def resolve_docker_image(image: str) -> str:
    """Return Docker's immutable image ID for a local image reference."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect Docker image {image!r}: {exc}") from exc
    image_id = result.stdout.strip()
    if result.returncode != 0 or not image_id.startswith("sha256:"):
        detail = (result.stderr or result.stdout).strip()[:500]
        raise RuntimeError(
            f"cannot resolve Docker image {image!r} to an immutable ID: {detail}"
        )
    return image_id


def _read_license_fields(path: str) -> dict[str, str]:
    """Read a small regular license file without following symlinks."""
    if not path:
        return {}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        return {}
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > MAX_LICENSE_FILE_BYTES
        ):
            return {}
        payload = os.read(fd, MAX_LICENSE_FILE_BYTES + 1)
    finally:
        os.close(fd)
    if len(payload) > MAX_LICENSE_FILE_BYTES:
        return {}
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {}
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        key = key.strip().upper()
        if separator and key and value.strip():
            fields[key] = value.strip()
    return fields


def _wls_license_fields(path: str) -> dict[str, str]:
    fields = _read_license_fields(path)
    if WLS_REQUIRED_FIELDS.issubset(fields):
        return {key: fields[key] for key in WLS_REQUIRED_FIELDS}
    return {}


def _canonical_license_path(path: str) -> str:
    """Resolve a trusted host-side license symlink before Docker mounts it."""
    if not path:
        return ""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


@contextlib.contextmanager
def _wls_execution_slot(cfg: dict):
    """Optionally limit WLS-backed candidates across host processes.

    Zero leaves WLS execution fully parallel. A positive value enables that many
    cross-process slots and covers token acquisition plus candidate lifetime.
    """
    mode = str(cfg.get("wls_egress", "auto")).strip().lower()
    license_path = str(cfg.get("gurobi_lic", ""))
    if mode == "off" or not _wls_license_fields(license_path):
        yield None
        return

    try:
        concurrency = int(
            cfg.get(
                "wls_concurrency",
                os.environ.get(
                    "FRONTIER_OR_WLS_CONCURRENCY",
                    DEFAULT_WLS_CONCURRENCY,
                ),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("wls_concurrency must be an integer") from exc
    if concurrency < 0 or concurrency > MAX_WLS_CONCURRENCY:
        raise ValueError(
            f"wls_concurrency must be between 0 and {MAX_WLS_CONCURRENCY}"
        )
    if concurrency == 0:
        yield None
        return

    runtime_root = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime_root:
        runtime_root = os.path.join(
            tempfile.gettempdir(),
            f"frontieror-wls-{os.getuid()}",
        )
    lock_dir = os.path.join(runtime_root, "frontieror-wls-slots")
    os.makedirs(lock_dir, mode=0o700, exist_ok=True)

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    while True:
        for index in range(concurrency):
            path = os.path.join(lock_dir, f"slot-{index}.lock")
            fd = os.open(path, flags, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            try:
                yield index
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            return
        time.sleep(0.1)


def _docker_control(args: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Docker control command failed: {exc}") from exc


def _require_docker_success(result: subprocess.CompletedProcess, action: str) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout).strip()[:500]
    raise RuntimeError(f"cannot {action}: {detail}")


def _wait_for_proxy(container_name: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    probe = (
        "import socket; "
        "s=socket.create_connection(('127.0.0.1',3128),1); s.close()"
    )
    while time.monotonic() < deadline:
        result = _docker_control(
            ["exec", container_name, "python", "-c", probe],
            timeout=3,
        )
        if result.returncode == 0:
            return
        time.sleep(0.1)
    raise RuntimeError("restricted WLS proxy did not become ready")


@contextlib.contextmanager
def _restricted_wls_egress(cfg: dict):
    """Create a per-candidate internal network and exact-host CONNECT proxy."""
    mode = str(cfg.get("wls_egress", "auto")).strip().lower()
    if mode not in WLS_EGRESS_MODES:
        raise ValueError(
            f"wls_egress must be one of {sorted(WLS_EGRESS_MODES)}, got {mode!r}"
        )
    license_path = str(
        cfg.get("gurobi_lic", os.environ.get("GRB_LICENSE_FILE", ""))
    )
    fields = _wls_license_fields(license_path)
    if mode == "required" and not fields:
        raise RuntimeError(
            "restricted WLS egress was required, but GRB_LICENSE_FILE does not "
            "contain WLSACCESSID, WLSSECRET, and LICENSEID"
        )
    if mode == "off" or not fields:
        yield None
        return

    identity = uuid.uuid4().hex
    network_name = f"frontieror-wls-{identity}"
    proxy_name = f"frontieror-wls-proxy-{identity}"
    image = str(cfg.get("docker_image", DEFAULT_DOCKER_IMAGE))
    network_created = False
    proxy_created = False
    try:
        result = _docker_control(["network", "create", "--internal", network_name])
        _require_docker_success(result, "create restricted WLS network")
        network_created = True
        result = _docker_control(
            [
                "run",
                "-d",
                "--name",
                proxy_name,
                "--log-driver=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--cpus",
                "0.25",
                "--memory",
                "128m",
                "--memory-swap",
                "128m",
                "--ulimit",
                "nofile=128:128",
                "--stop-timeout",
                "1",
                "--user",
                "65534:65534",
                "--network",
                "bridge",
                image,
                "python",
                "/opt/bench/restricted_egress_proxy.py",
                "--allow-host",
                WLS_TOKEN_HOST,
            ]
        )
        _require_docker_success(result, "start restricted WLS proxy")
        proxy_created = True
        result = _docker_control(
            [
                "network",
                "connect",
                "--alias",
                "frontieror-wls-egress",
                network_name,
                proxy_name,
            ]
        )
        _require_docker_success(result, "attach restricted WLS proxy")
        _wait_for_proxy(proxy_name)
        yield {
            "network": network_name,
            "proxy_url": "http://frontieror-wls-egress:3128",
            "redactions": tuple(fields.values()),
        }
    finally:
        if proxy_created:
            with contextlib.suppress(RuntimeError):
                _docker_control(["rm", "-f", proxy_name], timeout=10)
        if network_created:
            for _ in range(3):
                try:
                    result = _docker_control(
                        ["network", "rm", network_name], timeout=10
                    )
                except RuntimeError:
                    continue
                if result.returncode == 0:
                    break
                time.sleep(0.1)


def _redact_values(text: str, values: tuple[str, ...]) -> str:
    redacted = text
    for value in sorted((item for item in values if item), key=len, reverse=True):
        redacted = redacted.replace(value, "<redacted-wls-credential>")
    return redacted


def validate_docker_wls(cfg: dict, timeout: int = 30) -> None:
    """Fail fast when restricted WLS was requested but cannot initialize."""
    mode = str(cfg.get("wls_egress", "auto")).strip().lower()
    if mode not in WLS_EGRESS_MODES:
        raise ValueError(
            f"wls_egress must be one of {sorted(WLS_EGRESS_MODES)}, got {mode!r}"
        )
    if mode == "off":
        return
    configured_license = str(
        cfg.get("gurobi_lic", os.environ.get("GRB_LICENSE_FILE", ""))
    )
    license_path = _canonical_license_path(configured_license)
    fields = _wls_license_fields(license_path)
    if not fields:
        if mode == "required":
            raise RuntimeError(
                "restricted WLS egress is required, but no complete WLS license "
                "was found"
            )
        return

    probe_program = """\
import argparse
import gurobipy as gp

p = argparse.ArgumentParser()
p.add_argument("--instance_path", required=True)
p.add_argument("--solution_path", required=True)
p.add_argument("--time_limit", required=True)
p.add_argument("--log_path")
p.parse_args()
env = gp.Env()
env.close()
print("FRONTIEROR_WLS_READY")
"""
    with tempfile.TemporaryDirectory(prefix="frontieror-wls-preflight-") as root:
        code_path = os.path.join(root, "code.py")
        instance_path = os.path.join(root, "instance.json")
        solution_path = os.path.join(root, "solution.json")
        log_path = os.path.join(root, "log.jsonl")
        for path, payload in (
            (code_path, probe_program.encode("utf-8")),
            (instance_path, b"{}"),
            (solution_path, b""),
            (log_path, b""),
        ):
            fd = os.open(
                path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
        probe_cfg = dict(cfg)
        probe_cfg["gurobi_lic"] = license_path
        probe_cfg["wls_egress"] = "required"
        success, output, _ = run_docker(
            code_path,
            instance_path,
            solution_path,
            timeout,
            log_path=log_path,
            cfg=probe_cfg,
        )
    if not success or "FRONTIEROR_WLS_READY" not in output:
        detail = output.strip()[:1000]
        raise RuntimeError(f"Gurobi WLS preflight failed: {detail}")


@contextlib.contextmanager
def _instance_sandbox(instance_path):
    """Isolate the candidate program from the ground-truth tree.

    The candidate receives ``--instance_path`` and routinely derives
    ``paper_dir = dirname(dirname(instance_path))`` to reach sibling
    directories. Against the real benchmark tree that lets a program read or
    overwrite ``<paper>/gurobi_solution/<inst>.json`` (the Gurobi reference the
    evaluator compares against) or walk up to the repo-root
    ``gurobi_solving_results*.csv`` -- a reference-leak exploit that fakes
    ``gap≈0``.

    We copy ONLY the instance JSON into a throwaway ``/tmp`` tree that mirrors
    the ``<root>/instance/<file>`` layout, so the program's derived
    ``paper_dir`` is the sandbox root -- which has no ``gurobi_solution/`` and
    is not inside the repo, so walking up never finds the results CSV either.
    The instance basename is preserved (programs parse the ``large_instance_N``
    suffix). The trusted evaluator keeps using the real paths for the
    feasibility check and gap computation; only the program's view is sandboxed.

    docker already achieves this via volume mounts, so only the bare/systemd
    backends route through here.

    Yields the sandboxed instance path; the temp tree is removed on exit.
    """
    real = os.path.abspath(instance_path)
    tmp_root = tempfile.mkdtemp(prefix="eob_sbx_")
    try:
        inst_dir = os.path.join(tmp_root, "instance")
        os.makedirs(inst_dir, exist_ok=True)
        sandboxed = os.path.join(inst_dir, os.path.basename(real))
        if os.path.exists(real):
            shutil.copy2(real, sandboxed)
        yield sandboxed, tmp_root
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _build_args(code_path, instance_path, solution_path, time_limit, log_path):
    """Build the common argparse arguments for the generated code."""
    args = [
        "--instance_path", instance_path,
        "--solution_path", solution_path,
        "--time_limit", str(time_limit),
    ]
    if log_path:
        args.extend(["--log_path", log_path])
    return args


def build_bare_cmd(code_path, instance_path, solution_path, time_limit,
                   log_path=None, cfg=None):
    """Build a ``python code.py ...`` command, optionally pinned to N cores
    via ``taskset -c`` (util-linux, no systemd required). No CPU quota or
    memory cap — use systemd / docker backend for those."""
    _ensure_logger(code_path)
    cfg = cfg or {}
    cpus = cfg.get("cpus", DEFAULT_CPUS)
    core_set = _allocate_cores(cpus)
    inner = [sys.executable, code_path] + _build_args(
        code_path, instance_path, solution_path, time_limit, log_path
    )
    if core_set:
        inner = ["taskset", "-c", core_set] + inner
    return inner


def run_bare(code_path, instance_path, solution_path, time_limit,
             log_path=None, cfg=None):
    """Run directly via subprocess. No resource limits.

    Routes through the instance sandbox so the program cannot reach the
    ground-truth ``gurobi_solution/`` or ``gurobi_solving_results*.csv`` via
    path derivation or cwd-relative access (see ``_instance_sandbox``).
    """
    if (cfg or {}).get("anti_hack"):
        return False, "anti-hack mode requires the docker execution backend", 0.0
    code_path = os.path.abspath(code_path)
    solution_path = os.path.abspath(solution_path)
    log_path = os.path.abspath(log_path) if log_path else log_path
    with _instance_sandbox(instance_path) as (sb_instance, sb_root):
        cmd = build_bare_cmd(code_path, sb_instance, solution_path,
                             time_limit, log_path, cfg)
        return _exec(cmd, time_limit, cwd=sb_root)


_core_counter = 0
_core_lock = threading.Lock()


def _allocate_cores(n):
    """Allocate n cores within the host CPU range. Returns a comma-separated CPU list."""
    if n <= 0:
        raise ValueError("cpus must be positive")
    try:
        available = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        total = os.cpu_count()
        available = list(range(total)) if total else []
    if not available:
        return None
    global _core_counter
    with _core_lock:
        start = _core_counter % len(available)
        _core_counter += n
    cores = [available[(start + offset) % len(available)] for offset in range(n)]
    return ",".join(str(c) for c in cores)


def build_systemd_cmd(code_path, instance_path, solution_path, time_limit,
                      log_path=None, cfg=None):
    """Build a systemd-run scope + taskset command enforcing 1 pinned core,
    a memory cap, and network isolation.

    Layers (each is independent so a missing delegation still leaves the others):
      - ``systemd-run --scope --user -p MemoryMax=<mem>`` — hard memory cap via
        cgroup ``memory.max`` (memory controller is delegated to user slices by
        default on modern systemd).
      - ``-p IPAddressDeny=any`` — no network (eBPF egress filter, Linux ≥ 4.19).
      - ``-p AllowedCPUs=<core>`` — cpuset pinning (only if cpuset controller is
        delegated to user slice; otherwise silently ignored).
      - ``taskset -c <core>`` — userspace CPU pinning via ``sched_setaffinity``.
        Works without any cgroup delegation; this is the guaranteed pin.
    """
    cfg = cfg or {}
    cpus = cfg.get("cpus", DEFAULT_CPUS)
    memory = cfg.get("memory", DEFAULT_MEMORY)

    _ensure_logger(code_path)
    core_set = _allocate_cores(cpus)
    properties = [
        "systemd-run", "--scope", "--user", "-q",
        "-p", f"CPUQuota={cpus * 100}%",
        "-p", f"MemoryMax={memory}",
        "-p", "IPAddressDeny=any",
    ]
    if core_set:
        properties += ["-p", f"AllowedCPUs={core_set}"]
    inner = [sys.executable, code_path] + _build_args(
        code_path, instance_path, solution_path, time_limit, log_path
    )
    if core_set:
        inner = ["taskset", "-c", core_set] + inner
    return properties + inner


def run_systemd(code_path, instance_path, solution_path, time_limit,
                log_path=None, cfg=None):
    """Run via systemd-run with cgroup resource limits and pinned cores.

    Routes through the instance sandbox (see ``_instance_sandbox``):
    systemd-run --scope enforces cpu/memory/network but NOT filesystem
    isolation, so without this the program could still read/overwrite the
    ground-truth ``gurobi_solution/`` and ``gurobi_solving_results*.csv``.
    """
    if (cfg or {}).get("anti_hack"):
        return False, "anti-hack mode requires the docker execution backend", 0.0
    code_path = os.path.abspath(code_path)
    solution_path = os.path.abspath(solution_path)
    log_path = os.path.abspath(log_path) if log_path else log_path
    with _instance_sandbox(instance_path) as (sb_instance, sb_root):
        cmd = build_systemd_cmd(code_path, sb_instance, solution_path,
                                time_limit, log_path, cfg)
        return _exec(cmd, time_limit, cwd=sb_root)


def build_docker_cmd(code_path, instance_path, solution_path, time_limit,
                     log_path=None, cfg=None):
    """Build the ``docker run`` command for an isolated single-core run.

    Enforces:
      - ``--cpuset-cpus=<core>`` (pinned single core, round-robin across workers)
      - ``--cpus=<n>`` (hard CPU quota, matches cpuset size)
      - ``--memory=<m>`` (hard RAM cap)
      - no network, or an internal network whose only egress is the WLS proxy
    Mounts: code.py (ro), instance (ro), output directory (rw), Gurobi license (ro).
    """
    cfg = cfg or {}
    cpus = cfg.get("cpus", DEFAULT_CPUS)
    memory = cfg.get("memory", DEFAULT_MEMORY)
    image = cfg.get("docker_image", DEFAULT_DOCKER_IMAGE)
    gurobi_lic = cfg.get("gurobi_lic", os.environ.get("GRB_LICENSE_FILE", ""))
    anti_hack = bool(cfg.get("anti_hack"))
    candidate_time_limit = time_limit
    if anti_hack:
        # The trusted host deadline includes Docker startup and output
        # promotion. Give untrusted code a slightly smaller integer budget so
        # a solver that honors its CLI limit can serialize before that hard
        # deadline without creating a compute grace period.
        candidate_time_limit = max(
            1,
            int(float(time_limit)) - CANDIDATE_SHUTDOWN_RESERVE_SECONDS,
        )
    restricted_network = cfg.get("_restricted_network")
    restricted_proxy = cfg.get("_restricted_proxy")
    if bool(restricted_network) != bool(restricted_proxy):
        raise ValueError(
            "restricted candidate network and proxy must be configured together"
        )
    core_set = _allocate_cores(cpus)
    container_name = f"frontieror-candidate-{uuid.uuid4().hex}"

    code_path = os.path.abspath(code_path)

    c_code = "/workspace/code.py"
    c_instance = "/workspace/instance.json"
    volumes = [
        "-v", f"{code_path}:{c_code}:ro",
        "-v", f"{os.path.abspath(instance_path)}:{c_instance}:ro",
    ]
    sol_dir = os.path.dirname(os.path.abspath(solution_path))
    c_solution = f"/workspace/output/{os.path.basename(solution_path)}"
    c_log = None
    if anti_hack:
        volumes += [
            "--mount",
            f"type=bind,src={os.path.abspath(solution_path)},dst={c_solution}",
        ]
        if log_path:
            c_log = f"/workspace/output/{os.path.basename(log_path)}"
            volumes += [
                "--mount",
                f"type=bind,src={os.path.abspath(log_path)},dst={c_log}",
            ]
    else:
        volumes += ["-v", f"{sol_dir}:/workspace/output"]
        if log_path:
            log_dir = os.path.dirname(os.path.abspath(log_path))
            if log_dir != sol_dir:
                volumes += ["-v", f"{log_dir}:/workspace/logs"]
                c_log = f"/workspace/logs/{os.path.basename(log_path)}"
            else:
                c_log = f"/workspace/output/{os.path.basename(log_path)}"

    if gurobi_lic and os.path.exists(gurobi_lic):
        volumes += ["-v", f"{gurobi_lic}:/opt/gurobi/gurobi.lic:ro"]

    docker_flags = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--log-driver=none",
        f"--user={os.getuid()}:{os.getgid()}",
        f"--cpus={cpus}",
        f"--memory={memory}",
        f"--memory-swap={memory}",
    ]
    if restricted_network:
        docker_flags += ["--network", str(restricted_network)]
    else:
        docker_flags += ["--network=none"]
    if anti_hack:
        docker_flags += [
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "512",
            "--stop-timeout", "1",
            "--ulimit",
            "fsize={0}:{0}".format(
                int(cfg.get("max_output_file_bytes", DEFAULT_OUTPUT_FILE_BYTES))
            ),
            "--tmpfs",
            (
                f"/workspace/output:rw,nosuid,nodev,size=64m,"
                f"uid={os.getuid()},gid={os.getgid()},mode=0700"
            ),
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g",
        ]
    if core_set:
        docker_flags += [f"--cpuset-cpus={core_set}"]

    env_flags = [
        "-e", "PYTHONPATH=/opt/bench",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]
    if gurobi_lic and os.path.exists(gurobi_lic):
        env_flags += ["-e", "GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic"]
    if restricted_proxy:
        env_flags += [
            "-e", f"HTTPS_PROXY={restricted_proxy}",
            "-e", f"HTTP_PROXY={restricted_proxy}",
            "-e", "NO_PROXY=localhost,127.0.0.1",
        ]

    cmd = docker_flags + volumes + env_flags + [
        image,
        "python", c_code,
        "--instance_path", c_instance,
        "--solution_path", c_solution,
        "--time_limit", str(candidate_time_limit),
    ]
    if c_log:
        cmd += ["--log_path", c_log]
    return cmd


def run_docker(code_path, instance_path, solution_path, time_limit,
               log_path=None, cfg=None):
    """Run inside a Docker container with resource limits (pinned 1 core by default)."""
    effective_cfg = dict(cfg or {})
    configured_license = str(
        effective_cfg.get(
            "gurobi_lic",
            os.environ.get("GRB_LICENSE_FILE", ""),
        )
    )
    if configured_license:
        effective_cfg["gurobi_lic"] = _canonical_license_path(configured_license)
    mode = str(effective_cfg.get("wls_egress", "auto")).strip().lower()
    if (
        mode == "off"
        and _wls_license_fields(str(effective_cfg.get("gurobi_lic", "")))
    ):
        # Official third-party runs default to off. Do not expose a long-lived
        # WLS key to candidate code when it cannot and should not use WLS.
        effective_cfg["gurobi_lic"] = ""
    try:
        with _wls_execution_slot(effective_cfg):
            with _restricted_wls_egress(effective_cfg) as egress:
                redactions: tuple[str, ...] = ()
                if egress is not None:
                    effective_cfg["_restricted_network"] = egress["network"]
                    effective_cfg["_restricted_proxy"] = egress["proxy_url"]
                    redactions = egress["redactions"]
                cmd = build_docker_cmd(
                    code_path,
                    instance_path,
                    solution_path,
                    time_limit,
                    log_path,
                    effective_cfg,
                )
                success, output, elapsed = _exec(cmd, time_limit)
                return success, _redact_values(output, redactions), elapsed
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"Restricted WLS setup failed: {exc}", 0.0


def _ensure_logger(code_path):
    """Copy solution_logger.py next to the generated code if not already there."""
    code_dir = os.path.dirname(os.path.abspath(code_path))
    dest = os.path.join(code_dir, "solution_logger.py")
    if not os.path.exists(dest):
        src = Path(__file__).resolve().parents[2] / "scripts" / "utils" / "solution_logger.py"
        if src.exists():
            shutil.copy2(src, dest)


def _exec(cmd, time_limit, cwd=None):
    """Execute a command with timeout. Returns (success, output, elapsed).

    ``cwd`` (when set) runs the subprocess from that working directory; the
    bare/systemd backends point it at the instance sandbox so a program doing
    ``open("gurobi_solving_results.csv")`` or globbing the cwd finds nothing.

    Uses Popen + ``start_new_session=True`` so the spawned process is the
    leader of a new process group (its pgid = its pid). On timeout we call
    ``os.killpg(pgid, SIGKILL)`` to kill **the entire process group** rather
    than just the immediate child.

    Why this matters: ``subprocess.run(timeout=...)`` only sends SIGKILL to
    the direct child. With ``systemd-run --scope``, the actual python
    script is a grandchild that runs inside the scope's cgroup. When
    systemd-run dies, the python grandchild gets reparented to init and
    keeps running — bypassing the timeout entirely.

    Killing the process group guarantees taskset + python all die together.
    The scope cgroup auto-cleans once empty.
    """
    deadline = time.monotonic() + time_limit
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
            cwd=cwd,
        )
    except (OSError, ValueError) as e:
        return False, f"Failed to launch subprocess: {e}", 0.0

    captures = {"stdout": bytearray(), "stderr": bytearray()}
    totals = {"stdout": 0, "stderr": 0}

    def _drain(name, stream):
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                totals[name] += len(chunk)
                remaining = DEFAULT_CAPTURE_BYTES - len(captures[name])
                if remaining > 0:
                    captures[name].extend(chunk[:remaining])
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=_drain, args=("stdout", proc.stdout), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_drain, args=("stderr", proc.stderr), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=max(0.1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        # Hard kill the whole process group (systemd-run + taskset + python script)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        _force_remove_docker_container(cmd)
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    _force_remove_docker_container(cmd)
    elapsed = round(time.monotonic() - start, 2)
    if timed_out:
        return False, f"Execution timed out after {time_limit} seconds", elapsed

    def _render(name):
        rendered = captures[name].decode("utf-8", errors="replace").strip()
        omitted = totals[name] - len(captures[name])
        if omitted > 0:
            suffix = f"\n[output truncated; {omitted} bytes omitted]"
            rendered += suffix
        return rendered

    out = _render("stdout")
    err = _render("stderr")
    if proc.returncode != 0:
        error_msg = err or out
        return False, f"Process exited with code {proc.returncode}:\n{error_msg}", elapsed
    return True, out, elapsed


def _force_remove_docker_container(cmd) -> None:
    """Remove a named `docker run` container after its client is killed."""
    if not cmd or len(cmd) < 4 or cmd[0:2] != ["docker", "run"]:
        return
    name = None
    for index, arg in enumerate(cmd):
        if arg == "--name" and index + 1 < len(cmd):
            name = cmd[index + 1]
            break
        if isinstance(arg, str) and arg.startswith("--name="):
            name = arg.split("=", 1)[1]
            break
    if not name or not str(name).startswith("frontieror-candidate-"):
        return
    try:
        subprocess.run(
            ["docker", "rm", "-f", str(name)],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


# Registry
BACKENDS = {
    "bare": run_bare,
    "systemd": run_systemd,
    "docker": run_docker,
}

BUILDERS = {
    "bare": build_bare_cmd,
    "systemd": build_systemd_cmd,
    "docker": build_docker_cmd,
}
