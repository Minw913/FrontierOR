"""CORAL-specific orchestration for frontier-or self evolution.

CORAL optimizes by running coding agents over a seed repository.  This adapter
materializes a CORAL task whose hidden grader calls the benchmark evaluator,
starts a bounded CORAL run, extracts the best committed ``code.py``, and writes
the same CSV outputs as the other eval modes.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

import one_shot_eval as eval_core
from test_time_self_evolution import eval_modes
from frontieror.infra.checkers import validate_objective_checker
from frontieror.infra.files import SecureFileError, copy_regular_file
from frontieror.infra.policy import AgentModePolicy, validate_anti_hack_runtime
from frontieror.infra.visibility import (
    materialize_public_paper_view,
    validate_instance_content_split,
)
from frontieror.infra.agent.broker import drain_eval_requests
from frontieror.infra.agent.runtime import MAX_SECURE_AGENTS


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EXTERNAL_CORAL_DIR = os.path.join(ROOT_DIR, "external", "coral")
AGENT_WORKSPACE_MAX_BYTES = 512 * 1024 * 1024
AGENT_WORKSPACE_MAX_FILES = 20_000
MAX_SELECTED_CODE_BYTES = 4 * 1024 * 1024
_GATEWAY_PORT_BASE = 40_000
_GATEWAY_PORT_COUNT = 20_000
_GATEWAY_PORT_LOCK = threading.Lock()
_RESERVED_GATEWAY_PORTS: set[int] = set()


@dataclass
class CoralTask:
    task_name: str
    task_dir: str
    seed_dir: str
    config_path: str
    run_dir: str
    coral_dir: str
    repo_dir: str
    log_path: str
    agent_ids: tuple[str, ...]


def prepare_coral_env(base_env: Optional[Dict[str, str]], config: Dict) -> Dict[str, str]:
    """Prepare environment for CORAL without writing secrets into task files."""
    env = dict(base_env or os.environ)
    key = env.get("OPENROUTER_API_KEY") or config.get("OPENROUTER_API_KEY")
    if key:
        env["OPENROUTER_API_KEY"] = key
    if os.environ.get("GRB_LICENSE_FILE"):
        env["GRB_LICENSE_FILE"] = os.environ["GRB_LICENSE_FILE"]

    pythonpath = [ROOT_DIR]
    if os.path.isdir(EXTERNAL_CORAL_DIR):
        pythonpath.insert(0, EXTERNAL_CORAL_DIR)
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _trusted_git_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "core.hooksPath"
    env["GIT_CONFIG_VALUE_0"] = os.devnull
    return env


def _write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _reserve_gateway_port(identity: str) -> int:
    """Select a distinct loopback port for one host-owned CORAL gateway."""
    start = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
    with _GATEWAY_PORT_LOCK:
        for offset in range(_GATEWAY_PORT_COUNT):
            port = _GATEWAY_PORT_BASE + ((start + offset) % _GATEWAY_PORT_COUNT)
            if port in _RESERVED_GATEWAY_PORTS:
                continue
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            _RESERVED_GATEWAY_PORTS.add(port)
            return port
    raise RuntimeError("no loopback port is available for the CORAL model gateway")


def _try_reuse_oneshot_seed(paper_id: str, model_name: str, seed_dir: str) -> Optional[str]:
    """Try to copy ``eval/eval_papers/<paper>/<model_short>/code_attempt0.py``
    into ``<seed_dir>/code.py`` and return its path. Returns None if the
    one-shot artifact doesn't exist (caller should fall back to live LLM
    generation via ``eval_core.generate_candidate_code``).

    Saves 1 LLM call per paper when a one-shot run with the same model has
    already populated eval_papers/. Also makes CORAL/OpenEvolve start from
    the same seed (apples-to-apples comparison).
    """
    seed_model_name = (
        os.environ.get("FRONTIER_OR_CORAL_SEED_MODEL", "").strip() or model_name
    )
    short = eval_core.get_model_short_name(seed_model_name)
    src = os.path.join(
        ROOT_DIR, "eval", "eval_papers", paper_id, short, "code_attempt0.py",
    )
    if not os.path.lexists(src):
        return None
    os.makedirs(seed_dir, exist_ok=True)
    dst = os.path.join(seed_dir, "code.py")
    try:
        copied = copy_regular_file(
            src,
            dst,
            max_bytes=MAX_SELECTED_CODE_BYTES,
            label="one-shot seed code.py",
        )
    except SecureFileError as exc:
        print(f"[warn] rejected one-shot seed {paper_id}/{short}: {exc}")
        return None
    if copied == 0:
        os.unlink(dst)
        return None
    provenance = os.path.join(seed_dir, "_seed_source.txt")
    with open(provenance, "w", encoding="utf-8") as f:
        f.write(
            f"reused from one-shot v0: "
            f"eval/eval_papers/{paper_id}/{short}/code_attempt0.py\n"
            f"seed_model: {seed_model_name}\n"
            f"agent_model: {model_name}\n"
        )
    print(
        f"[reuse-oneshot:coral] {paper_id}/{short}: copied one-shot v0 "
        f"for agent model {model_name} → {dst}"
    )
    return dst


def _seed_readme(paper_id: str, prompt: str) -> str:
    return f"""# Efficient-OR Task: {paper_id}

{prompt}
"""


def _grader_code(root_dir: str) -> str:
    del root_dir
    return "from frontieror.infra.agent.grader import Grader\n"


def _write_gateway_config(path: str, model_alias: str, model_id: str):
    config = {
        "model_list": [
            {
                "model_name": model_alias,
                "litellm_params": {
                    "model": f"openrouter/{model_id}",
                    "api_key": "os.environ/OPENROUTER_API_KEY",
                    "api_base": "https://openrouter.ai/api/v1",
                },
            }
        ],
        "litellm_settings": {"drop_params": True},
    }
    _write_text(path, yaml.safe_dump(config, sort_keys=False))


def _coral_model_for_runtime(primary_model: str, agent_runtime: str, agent_model: Optional[str]) -> str:
    if agent_model:
        return agent_model
    if agent_runtime == "codex":
        return eval_core.get_model_short_name(primary_model)
    return primary_model


def validate_local_codex_model(model: str, timeout: int = 60) -> None:
    """Fail fast when the active Codex login cannot use the requested model."""
    cmd = [
        "codex",
        "exec",
        "Reply exactly FRONTIEROR_MODEL_READY and do not use tools.",
        "--model",
        model,
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "--disable",
        "apps",
        "--disable",
        "plugins",
        "--json",
    ]
    with tempfile.TemporaryDirectory(prefix="frontieror-model-preflight-") as root:
        try:
            result = subprocess.run(
                cmd,
                cwd=root,
                input="",
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Codex model preflight could not run for {model!r}: {exc}"
            ) from exc
    if result.returncode == 0 and "FRONTIEROR_MODEL_READY" in result.stdout:
        return

    details: list[str] = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") in {"error", "turn.failed"}:
            message = event.get("message")
            if message is None and isinstance(event.get("error"), dict):
                message = event["error"].get("message")
            if message:
                details.append(str(message))
    detail = details[-1] if details else (result.stderr or result.stdout).strip()
    raise RuntimeError(
        f"Codex model {model!r} is unavailable with the active local auth: "
        f"{detail[:1000]}"
    )


def validate_openrouter_model(
    model: str,
    api_key: str,
    timeout: int = 60,
) -> None:
    """Verify the exact proxy model and credential before starting paper workers."""
    if not api_key:
        raise RuntimeError("OpenRouter model preflight requires an API key")
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply exactly FRONTIEROR_MODEL_READY.",
                }
            ],
            "max_tokens": 16,
        }
    ).encode()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FrontierOR-Infra/model-preflight",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(1024 * 1024)
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read(1024 * 1024)
        status = exc.code
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"OpenRouter model preflight could not run for {model!r}: {exc}"
        ) from exc

    try:
        result = json.loads(body)
    except (TypeError, ValueError):
        result = {}
    if status == 200 and result.get("id"):
        return
    error = result.get("error")
    if isinstance(error, dict):
        detail = error.get("message") or json.dumps(error, sort_keys=True)
    else:
        detail = error or body.decode("utf-8", errors="replace")
    raise RuntimeError(
        f"OpenRouter model {model!r} is unavailable with the configured "
        f"credential (HTTP {status}): {str(detail)[:1000]}"
    )


def write_coral_task(
    *,
    base_dir: str,
    paper_id: str,
    prompt: str,
    model_name: str,
    primary_model: str,
    stage1_instances: List[str],
    stage2_instances: List[str],
    stage1_time_limit: int,
    stage2_time_limit: int,
    stage1_gap_threshold: float,
    exec_mode: str,
    exec_cfg: Dict,
    t_max,
    stage2_scorer: str,
    agent_runtime: str,
    agent_count: int,
    agent_model: Optional[str],
    max_turns: int,
    max_steps: Optional[int] = None,
    gateway_enabled: bool,
    openrouter_api_key: Optional[str] = None,
    stage2_stage_boundary: float = 0.01,
    stage2_time_policy: str = "uniform",
    stage2_time_buffer: int = 0,
    heartbeat_reflect_every: int = 0,
    heartbeat_pivot_every: int = 5,
    heartbeat_consolidate_every: int = 0,
    anti_hack: bool = False,
) -> CoralTask:
    task_dir = os.path.join(base_dir, "coral_task")
    seed_dir = os.path.join(task_dir, "public_seed" if anti_hack else "seed")
    eval_dir = os.path.join(task_dir, "eval")
    run_dir = os.path.join(base_dir, "coral_run")
    results_dir = os.path.join(base_dir, "coral_results")
    config_path = os.path.join(task_dir, "task.yaml")
    log_path = os.path.join(base_dir, "coral_start.log")

    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # seed/code.py is populated by the caller (run_self_evolve) — either by
    # reusing one-shot's code_attempt0.py or by live LLM generation. We don't
    # write a hardcoded stub here because the bench evaluates code.py as an
    # argparse script (subprocess), and any contract-violating stub would just
    # cost the agent 1 wasted attempt.
    _write_text(os.path.join(seed_dir, "README.md"), _seed_readme(paper_id, prompt))
    if anti_hack:
        paper_dir = eval_core.get_paper_dir(paper_id)
        materialize_public_paper_view(
            paper_dir=paper_dir,
            public_root=seed_dir,
            paper_id=paper_id,
            instances=list(
                dict.fromkeys(list(stage1_instances) + list(stage2_instances))
            ),
            stage_boundary=stage2_stage_boundary,
        )
    _write_text(os.path.join(eval_dir, "grader.py"), _grader_code(ROOT_DIR))

    coral_model = _coral_model_for_runtime(primary_model, agent_runtime, agent_model)
    task_name = f"efficient_or_{paper_id}_{model_name}"
    # heartbeat: each action only included when its `*_every` > 0.
    # - reflect (interval): pause every N evals, write a note. High cost.
    # - pivot (plateau): only fires after N non-improving evals. Low cost.
    # - consolidate (interval, global): merge cross-agent notes; needs >1 agent.
    heartbeat: List[Dict] = []
    if heartbeat_reflect_every > 0:
        heartbeat.append({
            "name": "reflect",
            "every": int(heartbeat_reflect_every),
            "trigger": "interval",
        })
    if heartbeat_pivot_every > 0:
        heartbeat.append({
            "name": "pivot",
            "every": int(heartbeat_pivot_every),
            "trigger": "plateau",
        })
    if heartbeat_consolidate_every > 0:
        heartbeat.append({
            "name": "consolidate",
            "every": int(heartbeat_consolidate_every),
            "trigger": "interval",
            "is_global": True,
        })

    agents = {
        "runtime": agent_runtime,
        "count": int(agent_count),
        "model": coral_model,
        # Preserve the experiment's native CORAL runtime setting. The hardened
        # wrapper records this value but does not reinterpret Codex tool events
        # as turns or terminate the agent after N shell commands.
        "max_turns": int(max_steps or max_turns) if anti_hack else int(max_turns),
        # research: keep off — agent web search would find the source paper
        # and leak benchmark answers. Reproducibility/cost also worse.
        "research": False,
        "heartbeat": heartbeat,
    }
    if anti_hack and agent_runtime == "codex":
        agents["runtime_options"] = {"web_search": "disabled"}
    if gateway_enabled:
        gateway_config_path = os.path.join(task_dir, "litellm_config.yaml")
        _write_gateway_config(gateway_config_path, coral_model, primary_model)
        agents["gateway"] = {
            "enabled": True,
            "port": _reserve_gateway_port(os.path.abspath(base_dir)),
            "config": gateway_config_path,
            "api_key": "",
        }

    if anti_hack:
        tips = (
            "Use only the files in this repository and the feedback returned by "
            "the grader. Hidden grader inputs and private reference data are not "
            "part of the task workspace."
        )
    else:
        tips = (
            "**Forbidden (answer leakage)** — never read these; the grader "
            "uses them internally and reading them constitutes cheating that "
            "voids the run:\n"
            "- `frontier-or/<paper>/gurobi_solution/` (ground-truth solutions)\n"
            "- any file matching `large_solution_*.json`\n"
            "- `.coral/private/` (hidden grader code)"
        )

    task_config = {
        "task": {
            "name": task_name,
            "description": (
                "Optimize code.py to maximize the score on the optimization "
                "problem described in README.md."
            ),
            # Tips are auto-rendered as a ## Tips section in CORAL.md (upstream
            # template feature, see external/coral/coral/template/coral_md.py).
            # Keep this minimal — only project-specific guardrails that the
            # upstream CORAL.md template, the rendered seed/README.md, and the
            # reused seed/code.py do not already convey.
            "tips": tips,
        },
        "grader": {
            "timeout": int(stage1_time_limit + stage2_time_limit + 120),
            "direction": "maximize",
            "args": {
                "paper_id": paper_id,
                "model_name": model_name,
                "base_dir": base_dir,
                "stage1_instances": stage1_instances,
                "stage2_instances": stage2_instances,
                "stage1_time_limit": stage1_time_limit,
                "stage2_time_limit": stage2_time_limit,
                "stage1_gap_threshold": stage1_gap_threshold,
                "stage2_scorer": stage2_scorer,
                "stage2_stage_boundary": stage2_stage_boundary,
                "stage2_time_policy": stage2_time_policy,
                "stage2_time_buffer": stage2_time_buffer,
                "exec_mode": exec_mode,
                "exec_cfg": exec_cfg or {},
                "t_max": t_max,
                "anti_hack": bool(anti_hack),
                "agent_native_max_steps": int(max_steps or max_turns),
            },
        },
        "agents": agents,
        "workspace": {
            "results_dir": results_dir,
            "repo_path": seed_dir,
            "run_dir": run_dir,
        },
        "run": {
            "session": "local",
            "verbose": False,
            "ui": False,
        },
    }
    _write_text(config_path, yaml.safe_dump(task_config, sort_keys=False))

    return CoralTask(
        task_name=task_name,
        task_dir=task_dir,
        seed_dir=seed_dir,
        config_path=config_path,
        run_dir=run_dir,
        coral_dir=os.path.join(run_dir, ".coral"),
        repo_dir=os.path.join(run_dir, "repo"),
        log_path=log_path,
        agent_ids=tuple(f"agent-{index}" for index in range(1, agent_count + 1)),
    )


def _coral_cli() -> List[str]:
    return [sys.executable, "-m", "test_time_self_evolution.coral.coral_cli_wrapper"]


def read_attempts(coral_dir: str) -> List[Dict]:
    attempts_dir = os.path.join(coral_dir, "public", "attempts")
    if not os.path.isdir(attempts_dir):
        return []
    attempts = []
    for name in sorted(os.listdir(attempts_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(attempts_dir, name), encoding="utf-8") as f:
                attempts.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return attempts


def read_best_attempt(coral_dir: str) -> Optional[Dict]:
    scored = [
        attempt
        for attempt in read_attempts(coral_dir)
        if attempt.get("status") != "pending" and attempt.get("score") is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda attempt: float(attempt.get("score") or 0.0))


def _stop_coral(task: CoralTask, env: Dict[str, str]):
    if not os.path.isdir(task.coral_dir):
        return
    subprocess.run(
        _coral_cli() + ["stop", "--task", task.task_name, "--run", os.path.basename(task.run_dir)],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _workspace_usage(path: str) -> tuple[int, int]:
    """Return no-follow byte/file counts for the untrusted per-run workspace."""
    total_bytes = 0
    entries = 0
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            children = os.scandir(current)
        except OSError:
            continue
        with children:
            for child in children:
                entries += 1
                try:
                    if child.is_dir(follow_symlinks=False):
                        pending.append(child.path)
                    elif child.is_file(follow_symlinks=False):
                        total_bytes += child.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
                if (
                    entries > AGENT_WORKSPACE_MAX_FILES
                    or total_bytes > AGENT_WORKSPACE_MAX_BYTES
                ):
                    return total_bytes, entries
    return total_bytes, entries


def run_coral_until_done(task: CoralTask, env: Dict[str, str], attempts: int, max_seconds: int,
                         resume: bool = False):
    os.makedirs(os.path.dirname(task.log_path), exist_ok=True)
    if resume:
        # CORAL's native resume CLI continues a prior run by --task / --run
        # path. Requires the previous coral_dir / attempts to still exist.
        if not os.path.isdir(task.coral_dir):
            raise RuntimeError(
                f"Cannot resume CORAL at {task.coral_dir}: dir does not exist. "
                f"The prior run may not have started successfully."
            )
        cmd = _coral_cli() + [
            "resume",
            "--task", task.task_name,
            "--run", os.path.basename(task.run_dir),
            "run.session=local",
        ]
        log_mode = "a"  # append to existing log
        print(f"[resume:coral] resuming task={task.task_name} run={os.path.basename(task.run_dir)}")
    else:
        cmd = _coral_cli() + ["start", "-c", task.config_path, "run.session=local"]
        log_mode = "w"
    with open(task.log_path, log_mode, encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    deadline = time.monotonic() + max_seconds
    stop_reason = "wall_clock_exhausted"
    try:
        while time.monotonic() < deadline:
            if env.get("FRONTIER_OR_ANTI_HACK") == "1":
                drain_eval_requests(
                    task,
                    max_attempts=attempts,
                    allowed_agent_ids=task.agent_ids,
                )
                workspace_bytes, workspace_files = _workspace_usage(task.run_dir)
                if (
                    workspace_bytes > AGENT_WORKSPACE_MAX_BYTES
                    or workspace_files > AGENT_WORKSPACE_MAX_FILES
                ):
                    stop_reason = "workspace_quota_exhausted"
                    break
            finalized = [
                attempt for attempt in read_attempts(task.coral_dir)
                if attempt.get("status") != "pending"
            ]
            if len(finalized) >= attempts:
                stop_reason = "attempt_budget_complete"
                break
            if proc.poll() is not None:
                stop_reason = "coral_process_exit"
                break
            time.sleep(1)
    finally:
        if proc.poll() is None:
            try:
                _stop_coral(task, env)
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=10)
        if env.get("FRONTIER_OR_ANTI_HACK") == "1":
            from frontieror.infra.agent.runtime import (
                cleanup_secure_runtime,
                finalize_secure_runtime_audit,
            )
            try:
                finalize_secure_runtime_audit(task.run_dir)
            finally:
                cleanup_secure_runtime(task.run_dir)
    return {"stop_reason": stop_reason, "supervisor": None}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_final_audit(
    task: CoralTask,
    *,
    code_path: str,
    run_status: Dict,
    selected_attempt: Dict,
) -> None:
    audit_dir = os.path.join(task.coral_dir, "private", "audit")
    os.makedirs(audit_dir, exist_ok=True)
    payload = {
        "schema_version": 1,
        "stop_reason": run_status.get("stop_reason"),
        "supervisor": run_status.get("supervisor"),
        "artifact_sha256": _sha256_file(code_path),
        "artifact_name": os.path.basename(code_path),
        "selection": {
            "source": "best_broker_scored_attempt",
            "commit_hash": selected_attempt.get("commit_hash"),
            "agent_id": selected_attempt.get("agent_id"),
            "score": selected_attempt.get("score"),
            "status": selected_attempt.get("status"),
        },
    }
    _write_text(
        os.path.join(audit_dir, "final_artifact.json"),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def extract_attempt_code(task: CoralTask, attempt: Dict, destination_dir: str) -> str:
    commit_hash = attempt["commit_hash"]
    os.makedirs(destination_dir, exist_ok=True)
    dest = os.path.join(destination_dir, f"coral_{commit_hash[:12]}_code.py")
    result = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            task.repo_dir,
            "show",
            f"{commit_hash}:code.py",
        ],
        capture_output=True,
        text=False,
        check=True,
        timeout=10,
        env=_trusted_git_env(),
    )
    if len(result.stdout) > MAX_SELECTED_CODE_BYTES:
        raise ValueError("selected code.py exceeds the submission size limit")
    try:
        code = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("selected code.py must be UTF-8 text") from exc
    _write_text(dest, code)
    return dest


def _results_from_metadata(metadata: Dict, instances: List[str]) -> Dict[str, Dict]:
    """Reconstruct per-instance result dicts from grader sidecar metadata.

    The keys returned must match what ``eval_modes._build_self_evolve_row``
    reads. In addition to the basic feasibility/gap fields, we forward the
    staged_qte breakdown (``score``/``stage_id``/``quality_part``/...) so the
    dev/test CSVs are populated with the same level of detail as openevolve.
    """
    results: Dict[str, Dict] = {}
    for inst in instances:
        prefix = f"inst_{inst}"
        feasible_raw = metadata.get(f"{prefix}_feasible")
        feasible = True if feasible_raw == 1.0 else (False if feasible_raw == 0.0 else None)
        results[inst] = {
            "status": "pass" if feasible is True else ("fail" if feasible is False else "missing"),
            "fail_reason": None if feasible is True else "coral_metadata",
            "feasible": feasible,
            "gap": metadata.get(f"{prefix}_gap"),
            "llm_obj": metadata.get(f"{prefix}_obj"),
            "gurobi_obj": metadata.get(f"{prefix}_gurobi_obj"),
            "solve_time": metadata.get(f"{prefix}_time"),
            "aocc": metadata.get(f"{prefix}_aocc"),
            # staged_qte breakdown — read by _build_self_evolve_row to fill
            # score_staged / stage_id / quality_part / speed_part / signed_gap
            # / beat_amount columns. Without these the dev CSV has 6 blanks.
            "score": metadata.get(f"{prefix}_score"),
            "stage_id": metadata.get(f"{prefix}_stage_id"),
            "quality_part": metadata.get(f"{prefix}_quality_part"),
            "speed_part": metadata.get(f"{prefix}_speed_part"),
            "signed_gap": metadata.get(f"{prefix}_signed_gap"),
            "beat_amount": metadata.get(f"{prefix}_beat_amount"),
            "error": None,
            "retries": 0,
        }
    return results


def _failure_results(instances: List[str], error: str) -> Dict[str, Dict]:
    return {
        inst: {
            "status": "fail",
            "fail_reason": "coral_no_attempt",
            "feasible": None,
            "gap": None,
            "solve_time": None,
            "llm_obj": None,
            "gurobi_obj": None,
            "aocc": None,
            "error": error,
            "retries": 0,
        }
        for inst in instances
    }


def run_self_evolve(
    *,
    run_id: str,
    paper_id: str,
    primary_model: str,
    prompt: str,
    config: Dict,
    stage1_instances: List[str],
    stage2_instances: List[str],
    test_instances: List[str],
    stage1_time_limit: int,
    stage2_time_limit: int,
    test_time_limit: int,
    stage1_gap_threshold: float,
    exec_mode: str,
    exec_cfg: Dict,
    t_max,
    stage2_scorer: str = "staged_qte",
    stage2_stage_boundary: float = 0.01,
    attempts: int = 1,
    max_seconds: int = 900,
    agent_runtime: str = "codex",
    agent_count: int = 1,
    agent_model: Optional[str] = None,
    max_turns: int = 20,
    max_steps: Optional[int] = None,
    gateway_enabled: bool = False,
    stage2_time_policy: str = "uniform",
    stage2_time_buffer: int = 0,
    test_time_policy: str = "uniform",
    test_time_buffer: int = 0,
    test_instance_workers: int = 4,
    heartbeat_reflect_every: int = 0,
    heartbeat_pivot_every: int = 5,
    heartbeat_consolidate_every: int = 0,
    secondary_model: Optional[str] = None,
    resume: bool = False,
    anti_hack: bool = False,
    agent_isolation: str = "docker",
    model_access: str = "local-auth",
    agent_image: str = "frontieror-coral-agent:0.1",
):
    validate_anti_hack_runtime(
        enabled=anti_hack or bool((exec_cfg or {}).get("anti_hack")),
        exec_mode=exec_mode,
        final_test_instances=test_instances,
        scorer=stage2_scorer,
    )
    del secondary_model
    if anti_hack and agent_runtime != "codex":
        raise ValueError("anti-hack CORAL currently requires --coral-agent-runtime codex")
    if anti_hack and agent_isolation != "docker":
        raise ValueError("anti-hack CORAL requires --coral-agent-isolation docker")
    if anti_hack and not 1 <= agent_count <= MAX_SECURE_AGENTS:
        raise ValueError(
            "anti-hack CORAL requires between 1 and "
            f"{MAX_SECURE_AGENTS} agents"
        )
    if anti_hack and model_access not in {"local-auth", "proxy"}:
        raise ValueError(f"unsupported anti-hack CORAL model access: {model_access!r}")
    if anti_hack and gateway_enabled:
        raise ValueError(
            "anti-hack CORAL uses its credential-isolating platform model proxy; "
            "the legacy --coral-gateway path is not allowed"
        )
    if max_steps is not None and max_steps < 1:
        raise ValueError("--coral-max-steps must be positive")
    if anti_hack:
        policy = AgentModePolicy()
        policy.validate_split(
            stage1_instances=stage1_instances,
            dev_instances=stage2_instances,
            final_instances=test_instances,
        )
        validate_instance_content_split(
            paper_dir=eval_core.get_paper_dir(paper_id),
            dev_instances=list(stage1_instances) + list(stage2_instances),
            final_instances=test_instances,
        )
    if anti_hack or bool((exec_cfg or {}).get("anti_hack")):
        for instance in test_instances:
            validate_objective_checker(
                paper_dir=eval_core.get_paper_dir(paper_id),
                instance=instance,
            )
    primary_name = eval_core.get_model_short_name(primary_model)
    # CSV rows are disambiguated across frameworks by the `framework` column,
    # so the short name needs no framework prefix.
    model_name = primary_name
    base_dir = eval_modes.mode_run_dir(run_id, "coral", paper_id, model_name)
    selection_instance = stage1_instances[0] if stage1_instances else (
        (test_instances or stage2_instances or ["tiny"])[0]
    )
    final_instances = list(test_instances)
    reporting_instances = final_instances or list(stage2_instances)

    task = write_coral_task(
        base_dir=base_dir,
        paper_id=paper_id,
        prompt=prompt,
        model_name=model_name,
        primary_model=primary_model,
        stage1_instances=stage1_instances,
        stage2_instances=stage2_instances,
        stage1_time_limit=stage1_time_limit,
        stage2_time_limit=stage2_time_limit,
        stage1_gap_threshold=stage1_gap_threshold,
        exec_mode=exec_mode,
        exec_cfg=exec_cfg,
        t_max=t_max,
        stage2_scorer=stage2_scorer,
        stage2_stage_boundary=stage2_stage_boundary,
        agent_runtime=agent_runtime,
        agent_count=agent_count,
        agent_model=agent_model,
        max_turns=max_turns,
        max_steps=max_steps,
        gateway_enabled=gateway_enabled,
        openrouter_api_key=config.get("OPENROUTER_API_KEY"),
        stage2_time_policy=stage2_time_policy,
        stage2_time_buffer=stage2_time_buffer,
        heartbeat_reflect_every=heartbeat_reflect_every,
        heartbeat_pivot_every=heartbeat_pivot_every,
        heartbeat_consolidate_every=heartbeat_consolidate_every,
        anti_hack=anti_hack,
    )

    # Seed code.py priority: resume (keep existing) > one-shot reuse > live LLM generation.
    seed_dir = task.seed_dir
    seed_code_path = os.path.join(seed_dir, "code.py")
    seed_token_usage: Dict = {}
    if resume and os.path.exists(seed_code_path) and os.path.getsize(seed_code_path) > 0:
        print(f"[resume:coral] keeping existing seed at {seed_code_path}")
    elif (os.environ.get("EFFICIENT_OR_REUSE_SEED_IF_EXISTS") == "1"
          and os.path.exists(seed_code_path) and os.path.getsize(seed_code_path) > 0):
        print(f"[reuse-seed] using existing seed at {seed_code_path} (skip LLM generation)")
    else:
        reused = _try_reuse_oneshot_seed(paper_id, primary_model, seed_dir)
        if reused is None:
            generated = eval_core.generate_candidate_code(
                prompt, config, primary_model, seed_dir,
                candidate_id="seed", temperature=0.4,
            )
            if generated["status"] != "ok":
                fallback_instances = final_instances or stage1_instances or [selection_instance]
                results = {
                    inst: {
                        "status": "fail",
                        "fail_reason": "generation_error",
                        "feasible": None,
                        "gap": None,
                        "solve_time": None,
                        "llm_obj": None,
                        "gurobi_obj": None,
                        "aocc": None,
                        "error": generated.get("error", "seed generation failed"),
                        "retries": 0,
                    }
                    for inst in fallback_instances
                }
                eval_modes.write_api_cost_row(
                    run_id, "self_evolve", paper_id, primary_model, model_name,
                    generated.get("usage", {}),
                    note=f"CORAL seed generation failed: {generated.get('error', '?')}",
                )
                return {"candidate_id": "coral_seed_fail", "results": results, "code_path": ""}
            seed_token_usage = generated.get("usage", {}) or {}

    env = prepare_coral_env(os.environ, config)
    if anti_hack:
        env["FRONTIER_OR_ANTI_HACK"] = "1"
        env["FRONTIER_OR_CORAL_MODEL_ACCESS"] = model_access
        env["FRONTIER_OR_CORAL_UPSTREAM_MODEL"] = primary_model
        env["FRONTIER_OR_CORAL_AGENT_IMAGE"] = agent_image
        env["FRONTIER_OR_CORAL_AGENT_COUNT"] = str(agent_count)
        env["FRONTIER_OR_CORAL_EVAL_WAIT_SECONDS"] = str(
            stage1_time_limit + stage2_time_limit + 300
        )
        # Agent-written repository config/hooks must never execute in the host
        # manager or grader during `git worktree add`.
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "core.hooksPath"
        env["GIT_CONFIG_VALUE_0"] = os.devnull
    run_status = run_coral_until_done(task, env, attempts=attempts, max_seconds=max_seconds, resume=resume)
    best = read_best_attempt(task.coral_dir)
    if not best:
        error = "CORAL produced no broker-scored submission"
        dev_results = _failure_results(list(stage2_instances), error)
        test_results = _failure_results(list(final_instances), error)
        eval_modes.write_self_evolve_results(
            paper_id=paper_id,
            model_name=model_name,
            framework="coral",
            dev_instances=list(stage2_instances),
            dev_results=dev_results,
            dev_seed_results={},
            test_instances=list(final_instances),
            test_results=test_results,
            iteration_found=None,
            generation=0,
            run_id=run_id,
        )
        eval_modes.write_api_cost_row(
            run_id, "self_evolve", paper_id, primary_model, model_name,
            seed_token_usage,
            note=(
                "CORAL produced no broker-scored submission; no fallback artifact "
                f"was final-graded; see {task.log_path}"
            ),
        )
        return {
            "candidate_id": "coral_no_attempt",
            "results": test_results or dev_results,
            "code_path": "",
            **run_status,
        }

    candidate_id = f"coral_best:{best['commit_hash']}"
    extracted = extract_attempt_code(task, best, os.path.join(base_dir, "selected"))

    # Read per-instance metadata sidecar that the grader wrote during eval.
    # CORAL's Attempt JSON drops Score.metadata, so we keep our own keyed by
    # commit_hash under <base_dir>/coral_eval/attempt_metadata/<hash>.json.
    sidecar_path = os.path.join(
        base_dir, "coral_eval", "attempt_metadata", f"{best['commit_hash']}.json",
    )
    best_metadata: Dict = {}
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                best_metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] failed to read sidecar {sidecar_path}: {e}")

    if final_instances:
        final_results = eval_modes.evaluate_best_on_test_set(
            paper_id, model_name, extracted, final_instances,
            test_time_limit, test_time_policy, test_time_buffer,
            os.path.join(base_dir, "final_eval"),
            exec_mode, exec_cfg, t_max,
            max_workers=test_instance_workers,
        )
    else:
        final_results = _results_from_metadata(best_metadata, reporting_instances)
    selected_code = eval_modes.copy_selected_code(extracted, base_dir)
    if anti_hack:
        _write_final_audit(
            task,
            code_path=selected_code,
            run_status=run_status,
            selected_attempt=best,
        )

    # CORAL attempts are independent (no parent → generation always 0).
    # iteration_found = 1-indexed position of best in sorted attempts list.
    all_attempts = read_attempts(task.coral_dir)
    iteration_found = None
    for idx, a in enumerate(all_attempts, 1):
        if a.get("commit_hash") == best.get("commit_hash"):
            iteration_found = idx
            break
    dev_results_for_csv = _results_from_metadata(
        best_metadata, list(stage2_instances),
    )
    test_results_for_csv = final_results if final_instances else {}
    eval_modes.write_self_evolve_results(
        paper_id=paper_id,
        model_name=model_name,
        framework="coral",
        dev_instances=list(stage2_instances),
        dev_results=dev_results_for_csv,
        dev_seed_results={},   # CORAL has no seed concept (each attempt is independent)
        test_instances=list(test_instances),
        test_results=test_results_for_csv,
        iteration_found=iteration_found,
        generation=0,
        run_id=run_id,
    )

    eval_modes.write_api_cost_row(
        run_id, "self_evolve", paper_id, primary_model, model_name,
        seed_token_usage,
        note=("seed cost only; CORAL agent usage is tracked in CORAL logs "
              f"under {task.coral_dir}"),
    )
    return {
        "candidate_id": candidate_id,
        "results": final_results,
        "code_path": selected_code,
        "verification_tier": (
            f"isolated-shared-container-{model_access}" if anti_hack else "legacy"
        ),
        **run_status,
    }
