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
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

# Default resource limits
DEFAULT_CPUS = 1          # number of CPU cores
DEFAULT_MEMORY = "32G"    # memory limit (uppercase for systemd compatibility)
DEFAULT_DOCKER_IMAGE = "frontier-or"


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
    total = os.cpu_count()
    if not total:
        return None
    global _core_counter
    with _core_lock:
        start = _core_counter % total
        _core_counter += n
    cores = [(start + offset) % total for offset in range(n)]
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
      - ``--network=none`` (no network access)
    Mounts: paper code dir (ro), instance (ro), solution dir (rw), Gurobi license (ro).
    """
    cfg = cfg or {}
    cpus = cfg.get("cpus", DEFAULT_CPUS)
    memory = cfg.get("memory", DEFAULT_MEMORY)
    image = cfg.get("docker_image", DEFAULT_DOCKER_IMAGE)
    gurobi_lic = cfg.get("gurobi_lic", os.environ.get("GRB_LICENSE_FILE", ""))
    core_set = _allocate_cores(cpus)

    _ensure_logger(code_path)
    code_dir = os.path.dirname(os.path.abspath(code_path))

    c_instance = "/workspace/instance.json"
    volumes = [
        "-v", f"{code_dir}:/workspace/codedir:ro",
        "-v", f"{os.path.abspath(instance_path)}:{c_instance}:ro",
    ]
    sol_dir = os.path.dirname(os.path.abspath(solution_path))
    volumes += ["-v", f"{sol_dir}:/workspace/output"]
    c_solution = f"/workspace/output/{os.path.basename(solution_path)}"
    c_log = None
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
        f"--cpus={cpus}",
        f"--memory={memory}",
        "--network=none",
    ]
    if core_set:
        docker_flags += [f"--cpuset-cpus={core_set}"]

    cmd = docker_flags + volumes + [
        "-e", "PYTHONPATH=/workspace/codedir:/opt/bench",
        image,
        "python", f"/workspace/codedir/{os.path.basename(code_path)}",
        "--instance_path", c_instance,
        "--solution_path", c_solution,
        "--time_limit", str(time_limit),
    ]
    if c_log:
        cmd += ["--log_path", c_log]
    return cmd


def run_docker(code_path, instance_path, solution_path, time_limit,
               log_path=None, cfg=None):
    """Run inside a Docker container with resource limits (pinned 1 core by default)."""
    cmd = build_docker_cmd(code_path, instance_path, solution_path,
                           time_limit, log_path, cfg)
    return _exec(cmd, time_limit)


def _ensure_logger(code_path):
    """Copy solution_logger.py next to the generated code if not already there."""
    code_dir = os.path.dirname(os.path.abspath(code_path))
    dest = os.path.join(code_dir, "solution_logger.py")
    if not os.path.exists(dest):
        src = os.path.join(os.path.dirname(__file__), "solution_logger.py")
        if os.path.exists(src):
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
    grace_seconds = 30  # buffer over time_limit for cleanup tasks
    deadline = time.time() + time_limit + grace_seconds
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            cwd=cwd,
        )
    except (OSError, ValueError) as e:
        return False, f"Failed to launch subprocess: {e}", 0.0
    try:
        out, err = proc.communicate(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        # Hard kill the whole process group (systemd-run + taskset + python script)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        elapsed = round(time.time() - start, 2)
        return False, f"Execution timed out after {time_limit} seconds", elapsed
    elapsed = round(time.time() - start, 2)
    if proc.returncode != 0:
        error_msg = (err or "").strip() or (out or "").strip()
        return False, f"Process exited with code {proc.returncode}:\n{error_msg}", elapsed
    return True, (out or "").strip(), elapsed


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
