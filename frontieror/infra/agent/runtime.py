"""Docker-isolated Codex runtime used by FrontierOR anti-hack runs."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from frontieror.infra.agent.model_proxy import agent_token
from frontieror.infra.agent.protocol import AgentHandle, write_agent_log_entry

logger = logging.getLogger(__name__)
DEFAULT_IMAGE = "frontieror-coral-agent:0.1"
DEFAULT_MODEL_PROXY_IMAGE = "frontieror-coral-model-proxy:0.1"
AGENT_MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_AGENT_LOG_BYTES = 64 * 1024 * 1024
MAX_SECURE_AGENTS = 8
AGENT_CPUS = 2
AGENT_MEMORY_GIB = 4
AGENT_PIDS = 256
MODEL_GATEWAY_LISTEN_PORT = 8080
ACTION_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "tool_call",
    "web_search",
}
_CONTAINER_LOCK = threading.Lock()
_PREPARED_CONTAINERS: set[str] = set()
_PREPARED_MODEL_PROXIES: set[str] = set()
_MODEL_PROXY_MASTERS: dict[str, str] = {}


def _run_identity(run_dir: Path) -> str:
    return hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:12]


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], text=True, capture_output=True, check=check)


def _ensure_egress(image: str, run_dir: Path) -> tuple[str, str]:
    identity = _run_identity(run_dir)
    network = f"frontieror-agent-{identity}"
    proxy = f"frontieror-egress-{identity}"
    if _docker("network", "inspect", network, check=False).returncode != 0:
        _docker("network", "create", "--internal", network)
    if _docker("inspect", proxy, check=False).returncode != 0:
        _docker(
            "run", "-d", "--name", proxy,
            "--log-driver=none",
            "--read-only", "--cap-drop=ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", "128m", "--memory-swap", "128m",
            "--network", "bridge", image,
            "python3", "/opt/frontieror/secure_egress_proxy.py",
        )
        _docker("network", "connect", "--alias", "frontieror-egress", network, proxy)
    return network, proxy


def _ensure_model_proxy(
    image: str,
    run_dir: Path,
    network: str,
    upstream_api_key: str,
    upstream_model: str,
    agent_count: int,
) -> tuple[str, str]:
    """Create a credential-owning proxy that exposes one fixed model."""
    if not upstream_api_key:
        raise RuntimeError("secure CORAL proxy mode requires OPENROUTER_API_KEY")
    if (
        not upstream_model
        or len(upstream_model) > 256
        or any(character.isspace() for character in upstream_model)
    ):
        raise RuntimeError("secure CORAL proxy mode received an invalid upstream model")
    identity = _run_identity(run_dir)
    proxy_name = f"frontieror-model-proxy-{identity}"
    audit_dir = run_dir / ".coral" / "private" / "audit" / "model_proxy"
    audit_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(audit_dir, 0o700)
    with _CONTAINER_LOCK:
        master_token = _MODEL_PROXY_MASTERS.setdefault(
            identity,
            secrets.token_urlsafe(32),
        )
        if proxy_name in _PREPARED_MODEL_PROXIES:
            running = _docker(
                "inspect",
                "--format",
                "{{.State.Running}}",
                proxy_name,
                check=False,
            )
            if running.returncode == 0 and running.stdout.strip() == "true":
                return proxy_name, master_token
            _PREPARED_MODEL_PROXIES.discard(proxy_name)

        _docker("rm", "-f", proxy_name, check=False)
        uid, gid = os.getuid(), os.getgid()
        result = _docker(
            "run",
            "-d",
            "--rm",
            "--init",
            "--name",
            proxy_name,
            "--label",
            f"frontieror.model-proxy={identity}",
            "--log-driver=none",
            f"--user={uid}:{gid}",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--cpus",
            "1",
            "--memory",
            "512m",
            "--memory-swap",
            "512m",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
            "--network",
            "bridge",
            "--env",
            f"OPENROUTER_API_KEY={upstream_api_key}",
            "--env",
            f"FRONTIER_OR_PROXY_MASTER_TOKEN={master_token}",
            "--env",
            f"FRONTIER_OR_ALLOWED_MODEL={upstream_model}",
            "--env",
            f"FRONTIER_OR_AGENT_COUNT={agent_count}",
            "--mount",
            f"type=bind,src={audit_dir},dst=/frontieror/model-audit",
            image,
            "python3",
            "/opt/frontieror/secure_model_proxy.py",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "failed to start the secure model proxy: "
                f"{result.stderr.strip()[:500]}"
            )
        connected = _docker(
            "network",
            "connect",
            "--alias",
            "frontieror-model-gateway",
            network,
            proxy_name,
            check=False,
        )
        if connected.returncode != 0:
            _docker("rm", "-f", proxy_name, check=False)
            raise RuntimeError(
                "failed to attach the secure model proxy: "
                f"{connected.stderr.strip()[:500]}"
            )
        health = None
        for _ in range(300):
            health = _docker(
                "exec",
                proxy_name,
                "python3",
                "-c",
                (
                    "import urllib.request;"
                    "urllib.request.urlopen("
                    "'http://127.0.0.1:8080/healthz',timeout=1).read()"
                ),
                check=False,
            )
            if health.returncode == 0:
                break
            time.sleep(0.1)
        if health is None or health.returncode != 0:
            _docker("rm", "-f", proxy_name, check=False)
            _MODEL_PROXY_MASTERS.pop(identity, None)
            raise RuntimeError("secure model proxy failed its health check")
        _PREPARED_MODEL_PROXIES.add(proxy_name)
        return proxy_name, master_token


def cleanup_secure_runtime(run_dir: str | Path) -> None:
    identity = _run_identity(Path(run_dir))
    container = f"frontieror-agent-system-{identity}"
    proxy = f"frontieror-egress-{identity}"
    model_proxy = f"frontieror-model-proxy-{identity}"
    network = f"frontieror-agent-{identity}"
    _docker("rm", "-f", container, check=False)
    _docker("rm", "-f", proxy, check=False)
    _docker("rm", "-f", model_proxy, check=False)
    _docker("network", "rm", network, check=False)
    with _CONTAINER_LOCK:
        _PREPARED_CONTAINERS.discard(container)
        _PREPARED_MODEL_PROXIES.discard(model_proxy)
        _MODEL_PROXY_MASTERS.pop(identity, None)


def _image_digest(image: str) -> str:
    result = _docker("image", "inspect", image, "--format", "{{.Id}}", check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _configured_agent_count() -> int:
    raw = os.environ.get("FRONTIER_OR_CORAL_AGENT_COUNT", "1")
    try:
        count = int(raw)
    except ValueError as exc:
        raise RuntimeError("FRONTIER_OR_CORAL_AGENT_COUNT must be an integer") from exc
    if count < 1 or count > MAX_SECURE_AGENTS:
        raise RuntimeError(
            f"hardened CORAL supports between 1 and {MAX_SECURE_AGENTS} agents"
        )
    return count


def _agent_ids(agent_count: int) -> tuple[str, ...]:
    return tuple(f"agent-{index}" for index in range(1, agent_count + 1))


def _validate_agent_id(agent_id: str, agent_count: int) -> None:
    if agent_id not in _agent_ids(agent_count):
        raise RuntimeError(f"unregistered hardened CORAL agent id: {agent_id!r}")


@contextlib.contextmanager
def _audit_lock(coral_dir: Path):
    audit_dir = coral_dir / "private" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    lock_path = audit_dir / ".agent_activity.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _completed_actions(log_dir: Path, agent_id: str) -> int:
    completed = 0
    for path in sorted(log_dir.glob(f"{agent_id}*.log")):
        try:
            if path.stat().st_size > MAX_AGENT_LOG_BYTES:
                return MAX_AGENT_LOG_BYTES
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    item = event.get("item") or {}
                    if (
                        event.get("type") == "item.completed"
                        and item.get("type") in ACTION_TYPES
                    ):
                        completed += 1
        except OSError:
            return MAX_AGENT_LOG_BYTES
    return completed


def _record_runtime_start(
    coral_dir: Path,
    log_dir: Path,
    agent_id: str,
    native_max_steps: int,
    agent_count: int,
) -> dict[str, Any]:
    """Record lifecycle and tool activity without enforcing a tool-action cap."""
    path = coral_dir / "private" / "audit" / "agent_activity.json"
    registered = _agent_ids(agent_count)
    _validate_agent_id(agent_id, agent_count)
    with _audit_lock(coral_dir):
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            prior = {}
        except (AttributeError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("agent activity audit is unreadable") from exc

        prior_agents = prior.get("agents", {}) if isinstance(prior, dict) else {}
        if not isinstance(prior_agents, dict):
            raise RuntimeError("agent activity audit has an invalid agents object")
        if prior:
            if int(prior.get("configured_agent_count", agent_count)) != agent_count:
                raise RuntimeError("agent count does not match the activity audit")
            if int(prior.get("native_max_steps", native_max_steps)) != native_max_steps:
                raise RuntimeError("native max steps changed during the run")

        agents: dict[str, dict[str, int]] = {}
        for registered_id in registered:
            previous = prior_agents.get(registered_id, {})
            if not isinstance(previous, dict):
                raise RuntimeError("agent activity audit contains an invalid agent record")
            starts = int(previous.get("runtime_starts", 0))
            if registered_id == agent_id:
                starts += 1
            agents[registered_id] = {
                "native_max_steps": native_max_steps,
                "observed_actions": _completed_actions(log_dir, registered_id),
                "runtime_starts": starts,
                "container_starts": starts,
            }

        payload: dict[str, Any] = {
            "schema_version": 3,
            "configured_agent_count": agent_count,
            "native_max_steps": native_max_steps,
            "tool_action_limit_enforced": False,
            "lifecycle_owner": "coral",
            "agents": agents,
            "totals": {
                "observed_actions": sum(
                    record["observed_actions"] for record in agents.values()
                ),
                "runtime_starts": sum(
                    record["runtime_starts"] for record in agents.values()
                ),
            },
        }
        _write_private_json(path, payload)
    return payload


def finalize_secure_runtime_audit(run_dir: str | Path) -> None:
    """Reconcile observational activity after the CORAL manager has stopped."""
    run_dir = Path(run_dir)
    coral_dir = run_dir / ".coral"
    activity_path = coral_dir / "private" / "audit" / "agent_activity.json"
    with _audit_lock(coral_dir):
        try:
            activity = json.loads(activity_path.read_text(encoding="utf-8"))
            agent_count = int(activity["configured_agent_count"])
            native_max_steps = int(activity["native_max_steps"])
            prior_agents = activity["agents"]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot finalize agent activity audit") from exc
        if not isinstance(prior_agents, dict):
            raise RuntimeError("cannot finalize invalid agent activity audit")

        agents: dict[str, dict[str, int]] = {}
        for agent_id in _agent_ids(agent_count):
            previous = prior_agents.get(agent_id, {})
            starts = int(previous.get("runtime_starts", 0))
            agents[agent_id] = {
                "native_max_steps": native_max_steps,
                "observed_actions": _completed_actions(
                    coral_dir / "public" / "logs",
                    agent_id,
                ),
                "runtime_starts": starts,
                "container_starts": starts,
            }
        final_activity = {
            "schema_version": 3,
            "configured_agent_count": agent_count,
            "native_max_steps": native_max_steps,
            "tool_action_limit_enforced": False,
            "lifecycle_owner": "coral",
            "agents": agents,
            "totals": {
                "observed_actions": sum(
                    record["observed_actions"] for record in agents.values()
                ),
                "runtime_starts": sum(
                    record["runtime_starts"] for record in agents.values()
                ),
            },
        }
        _write_private_json(activity_path, final_activity)

    policy_path = coral_dir / "private" / "audit" / "agent_runtime.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot finalize agent runtime audit") from exc
    policy.setdefault("policy", {})["observed_activity"] = final_activity
    _write_private_json(policy_path, policy)


def _ensure_shared_agent_container(
    *,
    run_dir: Path,
    coral_dir: Path,
    repo_dir: Path,
    image_digest: str,
    auth_path: Path | None,
    network: str,
    agent_count: int,
) -> str:
    """Create one hardened outer container shared by the submitted agent system."""
    identity = _run_identity(run_dir)
    container_name = f"frontieror-agent-system-{identity}"
    with _CONTAINER_LOCK:
        if container_name in _PREPARED_CONTAINERS:
            running = _docker(
                "inspect",
                "--format",
                "{{.State.Running}}",
                container_name,
                check=False,
            )
            if running.returncode == 0 and running.stdout.strip() == "true":
                return container_name
            _PREPARED_CONTAINERS.discard(container_name)

        # A prior host process may have died before cleanup. Never reuse an
        # unverified container with stale processes or mounts.
        _docker("rm", "-f", container_name, check=False)

        agents_dir = run_dir / "agents"
        request_dir = coral_dir / "private" / "eval_requests" / "inbox"
        public_dir = coral_dir / "public"
        attempts_dir = public_dir / "attempts"
        notes_dir = public_dir / "notes"
        git_config = repo_dir / ".git" / "config"
        git_hooks = repo_dir / ".git" / "hooks"
        for path in (agents_dir, request_dir, attempts_dir, notes_dir):
            path.mkdir(parents=True, exist_ok=True)
        if not git_config.is_file() or not git_hooks.is_dir():
            raise RuntimeError(
                "isolated CORAL runtime requires a non-bare run repository"
            )

        uid, gid = os.getuid(), os.getgid()
        cpus = AGENT_CPUS * agent_count
        memory_gib = AGENT_MEMORY_GIB * agent_count
        pids = AGENT_PIDS * agent_count
        tmp_gib = max(1, agent_count)
        codex_home_mib = 512 * agent_count
        proxy_url = "http://frontieror-egress:3128"
        run_args = [
            "run",
            "-d",
            "--rm",
            "--init",
            "--name",
            container_name,
            "--label",
            f"frontieror.agent-system={identity}",
            "--log-driver=none",
            f"--user={uid}:{gid}",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(pids),
            "--cpus",
            str(cpus),
            "--memory",
            f"{memory_gib}g",
            "--memory-swap",
            f"{memory_gib}g",
            "--ulimit",
            f"fsize={AGENT_MAX_FILE_BYTES}:{AGENT_MAX_FILE_BYTES}",
            "--ulimit",
            "nofile=2048:2048",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={tmp_gib}g,mode=1777",
            "--tmpfs",
            (
                f"/codex-home:rw,nosuid,nodev,size={codex_home_mib}m,"
                f"uid={uid},gid={gid},mode=0700"
            ),
            "--tmpfs",
            f"{public_dir}:rw,nosuid,nodev,size=16m",
            "--tmpfs",
            (
                f"{git_hooks}:rw,nosuid,nodev,noexec,size=1m,"
                f"uid={uid},gid={gid},mode=0700"
            ),
            "--network",
            network,
            "-e",
            f"HTTPS_PROXY={proxy_url}",
            "-e",
            f"HTTP_PROXY={proxy_url}",
            "-e",
            "NO_PROXY=localhost,127.0.0.1,frontieror-model-gateway",
            "-e",
            "GIT_CONFIG_NOSYSTEM=1",
            "-e",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "-e",
            "GIT_NO_REPLACE_OBJECTS=1",
            "-e",
            "GIT_CONFIG_COUNT=1",
            "-e",
            "GIT_CONFIG_KEY_0=core.hooksPath",
            "-e",
            "GIT_CONFIG_VALUE_0=/dev/null",
            "--mount",
            f"type=bind,src={agents_dir},dst={agents_dir}",
            "--mount",
            f"type=bind,src={repo_dir},dst={repo_dir}",
            "--mount",
            f"type=bind,src={git_config},dst={git_config},readonly",
            "--mount",
            f"type=bind,src={attempts_dir},dst={attempts_dir},readonly",
            "--mount",
            f"type=bind,src={notes_dir},dst={notes_dir}",
            "--mount",
            f"type=bind,src={request_dir},dst=/frontieror/requests",
        ]
        if auth_path is not None:
            run_args.extend(
                [
                    "--mount",
                    (
                        f"type=bind,src={auth_path.resolve()},"
                        "dst=/frontieror/codex-auth.json,readonly"
                    ),
                ]
            )
        run_args.extend(
            [
            image_digest,
            "python3",
            "-c",
            "import time; time.sleep(31536000)",
            ]
        )
        result = _docker(
            *run_args,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "failed to start shared hardened CORAL container: "
                f"{result.stderr.strip()[:500]}"
            )
        _PREPARED_CONTAINERS.add(container_name)
        return container_name


def _record_runtime_policy(
    *,
    coral_dir: Path,
    container_name: str,
    image: str,
    image_digest: str,
    proxy_name: str,
    model_proxy_name: str | None,
    model_proxy_image: str | None,
    model_proxy_image_digest: str | None,
    model_access: str,
    model: str,
    agent_id: str,
    agent_count: int,
    native_max_steps: int,
    activity: dict[str, Any],
) -> None:
    path = coral_dir / "private" / "audit" / "agent_runtime.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("agent runtime audit is unreadable") from exc
    if payload and (
        payload.get("image_digest") != image_digest
        or int(payload.get("agent_count", agent_count)) != agent_count
        or payload.get("verification_tier")
        != f"isolated-shared-container-{model_access}"
        or payload.get("model_proxy_image_digest") != model_proxy_image_digest
    ):
        raise RuntimeError("shared agent runtime policy changed during the run")

    agents = payload.get("agents", {})
    if not isinstance(agents, dict):
        raise RuntimeError("agent runtime audit has an invalid agents object")
    agents[agent_id] = {
        "model": model,
        "native_max_steps": native_max_steps,
    }
    payload = {
        "schema_version": 2,
        "verification_tier": f"isolated-shared-container-{model_access}",
        "container_scope": "submitted-agent-system",
        "container_name": container_name,
        "agent_count": agent_count,
        "registered_agent_ids": list(_agent_ids(agent_count)),
        "agents": agents,
        # Retained for compatibility with single-agent audit readers.
        "model": model,
        "image": image,
        "image_digest": image_digest,
        "model_proxy_image": model_proxy_image,
        "model_proxy_image_digest": model_proxy_image_digest,
        "network": {
            "mode": "internal",
            "egress_proxy": proxy_name,
            "model_proxy": model_proxy_name,
            "allowlist": (
                ["platform-owned-fixed-model-proxy"]
                if model_access == "proxy"
                else ["*.openai.com:443", "*.chatgpt.com:443"]
            ),
        },
        "resources": {
            "scope": "shared",
            "cpus": AGENT_CPUS * agent_count,
            "memory": f"{AGENT_MEMORY_GIB * agent_count}g",
            "pids_limit": AGENT_PIDS * agent_count,
            "max_file_bytes": AGENT_MAX_FILE_BYTES,
            "nofile": 2048,
        },
        "mounts": {
            "agent_worktrees": "read-write shared system workspace",
            "git_metadata": "read-write shared system workspace",
            "git_config": "read-only",
            "git_hooks": "empty tmpfs",
            "trusted_attempts": "read-only",
            "shared_notes": "read-write",
            "other_public_coral": "empty tmpfs",
            "eval_request_inbox": "write-only-by-contract",
            "codex_auth": (
                "not mounted; ephemeral gateway token only"
                if model_access == "proxy"
                else "read-only shared credential"
            ),
        },
        "policy": {
            "sandbox": "outer_shared_docker",
            "codex_inner_sandbox": "danger-full-access",
            "approval": "never",
            "web_search": "disabled",
            "apps_and_plugins": "disabled",
            "codex_subagents": "disabled",
            "host_git_hooks": "disabled-by-environment",
            "git_replace_refs": "disabled",
            "lifecycle_owner": "coral",
            "native_max_steps": native_max_steps,
            "tool_action_limit_enforced": False,
            "termination": "trusted attempt budget and wall-clock deadline",
            "observed_activity": activity,
        },
    }
    _write_private_json(path, payload)


class SecureCodexRuntime:
    """Codex runtime whose shell is confined by an outer Docker boundary."""

    @property
    def instruction_filename(self) -> str:
        return "AGENTS.md"

    @property
    def shared_dir_name(self) -> str:
        return ".codex"

    def extract_session_id(self, log_path: Path) -> str | None:
        try:
            with log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                handle.seek(max(0, end - 1024 * 1024))
                lines = handle.read(1024 * 1024).decode(
                    "utf-8", errors="replace"
                ).splitlines()
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("thread_id"):
                    return event["thread_id"]
                if event.get("session_id"):
                    return event["session_id"]
        except OSError:
            pass
        return None

    def start(
        self,
        worktree_path: Path,
        coral_md_path: Path,
        model: str = "gpt-5.4",
        runtime_options: dict[str, Any] | None = None,
        max_turns: int = 20,
        log_dir: Path | None = None,
        verbose: bool = False,
        resume_session_id: str | None = None,
        prompt: str | None = None,
        prompt_source: str | None = None,
        task_name: str | None = None,
        task_description: str | None = None,
        gateway_url: str | None = None,
        gateway_api_key: str | None = None,
    ) -> AgentHandle:
        del coral_md_path, runtime_options, verbose
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        worktree_path = worktree_path.resolve()
        run_dir = worktree_path.parent.parent.resolve()
        coral_dir = run_dir / ".coral"
        repo_dir = run_dir / "repo"
        agent_id = (worktree_path / ".coral_agent_id").read_text().strip()
        agent_count = _configured_agent_count()
        _validate_agent_id(agent_id, agent_count)
        image = os.environ.get("FRONTIER_OR_CORAL_AGENT_IMAGE", DEFAULT_IMAGE)
        model_access = os.environ.get("FRONTIER_OR_CORAL_MODEL_ACCESS", "local-auth")
        if model_access not in {"local-auth", "proxy"}:
            raise RuntimeError(f"unsupported secure model access mode: {model_access!r}")
        auth_path: Path | None = None
        if model_access == "local-auth":
            auth_path = (
                Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
                / "auth.json"
            )
            if not auth_path.is_file():
                raise RuntimeError(f"Codex auth file not found: {auth_path}")
            if gateway_url or gateway_api_key:
                raise RuntimeError("local-auth mode must not receive gateway credentials")
        elif gateway_url or gateway_api_key:
            raise RuntimeError(
                "hardened proxy mode uses the platform model proxy, not the "
                "legacy CORAL gateway"
            )
        if _docker("image", "inspect", image, check=False).returncode != 0:
            raise RuntimeError(
                f"Secure CORAL image {image!r} is not built. Run: "
                "docker build -f frontieror/infra/docker/agent.Dockerfile "
                f"-t {image} ."
            )
        image_digest = _image_digest(image)
        if not image_digest.startswith("sha256:"):
            raise RuntimeError(
                f"Secure CORAL image {image!r} has no immutable local image ID"
            )
        model_proxy_image: str | None = None
        model_proxy_image_digest: str | None = None
        if model_access == "proxy":
            model_proxy_image = os.environ.get(
                "FRONTIER_OR_CORAL_MODEL_PROXY_IMAGE",
                DEFAULT_MODEL_PROXY_IMAGE,
            )
            if (
                _docker("image", "inspect", model_proxy_image, check=False).returncode
                != 0
            ):
                raise RuntimeError(
                    f"Secure CORAL model proxy image {model_proxy_image!r} is not "
                    "built. Run: docker build -f "
                    "frontieror/infra/docker/model-proxy.Dockerfile "
                    f"-t {model_proxy_image} ."
                )
            model_proxy_image_digest = _image_digest(model_proxy_image)
            if not model_proxy_image_digest.startswith("sha256:"):
                raise RuntimeError(
                    f"Secure CORAL model proxy image {model_proxy_image!r} has no "
                    "immutable local image ID"
                )

        network, proxy_name = _ensure_egress(image_digest, run_dir)
        model_proxy_name: str | None = None
        internal_gateway_url: str | None = None
        proxy_agent_token: str | None = None
        if model_access == "proxy":
            assert model_proxy_image_digest is not None
            upstream_model = os.environ.get("FRONTIER_OR_CORAL_UPSTREAM_MODEL", "")
            model_proxy_name, master_token = _ensure_model_proxy(
                model_proxy_image_digest,
                run_dir,
                network,
                os.environ.get("OPENROUTER_API_KEY", ""),
                upstream_model,
                agent_count,
            )
            internal_gateway_url = (
                f"http://frontieror-model-gateway:{MODEL_GATEWAY_LISTEN_PORT}"
            )
            proxy_agent_token = agent_token(master_token, agent_id)
        request_dir = coral_dir / "private" / "eval_requests" / "inbox"
        request_dir.mkdir(parents=True, exist_ok=True)
        attempts_dir = coral_dir / "public" / "attempts"
        public_dir = coral_dir / "public"
        for path in (attempts_dir, public_dir):
            path.mkdir(parents=True, exist_ok=True)
        container_name = _ensure_shared_agent_container(
            run_dir=run_dir,
            coral_dir=coral_dir,
            repo_dir=repo_dir,
            image_digest=image_digest,
            auth_path=auth_path,
            network=network,
            agent_count=agent_count,
        )

        if log_dir is None:
            log_dir = public_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_idx = len(list(log_dir.glob(f"{agent_id}*.log")))
        log_path = log_dir / f"{agent_id}.{log_idx}.log"
        log_file = open(log_path, "w", buffering=1, encoding="utf-8")
        prompt = prompt or ("Session resumed. Continue where you left off." if resume_session_id else "Begin.")
        write_agent_log_entry(
            log_file,
            prompt=prompt,
            source=prompt_source or ("restart" if resume_session_id else "start"),
            agent_id=agent_id,
            session_id=resume_session_id,
            task_name=task_name,
            task_description=task_description,
        )
        activity = _record_runtime_start(
            coral_dir,
            log_dir,
            agent_id,
            max_turns,
            agent_count,
        )

        uid, gid = os.getuid(), os.getgid()
        cmd = [
            "docker",
            "exec",
            "--user",
            f"{uid}:{gid}",
            "--workdir",
            str(worktree_path),
            "--env",
            f"HOME=/tmp/{agent_id}",
            "--env",
            f"CODEX_HOME=/codex-home/{agent_id}",
            "--env",
            f"FRONTIER_OR_AGENT_ID={agent_id}",
            "--env",
            f"FRONTIER_OR_CORAL_AGENT_COUNT={agent_count}",
            "--env",
            "FRONTIER_OR_EVAL_REQUEST_DIR=/frontieror/requests",
            "--env",
            f"FRONTIER_OR_ATTEMPTS_DIR={attempts_dir}",
            "--env",
            (
                "FRONTIER_OR_EVAL_WAIT_SECONDS="
                + os.environ.get("FRONTIER_OR_CORAL_EVAL_WAIT_SECONDS", "7200")
            ),
            "--env",
            f"FRONTIER_OR_MODEL_ACCESS={model_access}",
        ]
        if model_access == "proxy":
            assert proxy_agent_token is not None
            cmd.extend(["--env", f"OPENAI_API_KEY={proxy_agent_token}"])
        cmd.extend(
            [
            container_name,
            "python3", "/opt/frontieror/secure_codex_entrypoint.py",
            "--model", model,
            "--prompt", prompt,
            "--max-steps", str(max_turns),
            ]
        )
        if internal_gateway_url is not None:
            cmd.extend(["--gateway-url", internal_gateway_url])
        if resume_session_id:
            cmd.extend(["--resume-session-id", resume_session_id])

        _record_runtime_policy(
            coral_dir=coral_dir,
            container_name=container_name,
            image=image,
            image_digest=image_digest,
            proxy_name=proxy_name,
            model_proxy_name=model_proxy_name,
            model_proxy_image=model_proxy_image,
            model_proxy_image_digest=model_proxy_image_digest,
            model_access=model_access,
            model=model,
            agent_id=agent_id,
            agent_count=agent_count,
            native_max_steps=max_turns,
            activity=activity,
        )
        logger.info(
            "Starting isolated Codex agent %s in shared container %s",
            agent_id,
            container_name,
        )
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        return AgentHandle(
            agent_id=agent_id,
            process=process,
            worktree_path=worktree_path,
            log_path=log_path,
            session_id=resume_session_id,
            _log_file=log_file,
        )
