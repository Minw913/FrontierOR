"""Run Codex with immutable policy inside the outer agent container.

CORAL owns the agent lifecycle. The configured ``max_steps`` value is passed
through for provenance only; this wrapper must not reinterpret it as a count of
Codex tool events. Attempt and wall-clock limits are enforced by the trusted
CORAL host process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


ACTION_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "tool_call",
    "web_search",
}


def build_codex_command(
    *,
    model: str,
    prompt: str,
    resume_session_id: str | None,
    model_access: str,
    gateway_url: str | None,
) -> list[str]:
    cmd = [
        "codex",
        "exec",
        "--model",
        model,
        # The Docker container is the mandatory sandbox. Running Codex's bwrap
        # sandbox inside it either requires extra container privileges or is
        # incompatible with current permission profiles.
        "--sandbox",
        "danger-full-access",
        "--ignore-user-config",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "--disable",
        "apps",
        "--disable",
        "plugins",
        "--disable",
        "remote_plugin",
        "--disable",
        "plugin_sharing",
        "--disable",
        "browser_use",
        "--disable",
        "browser_use_external",
        "--disable",
        "browser_use_full_cdp_access",
        "--disable",
        "computer_use",
        "--disable",
        "image_generation",
        "--disable",
        "multi_agent",
        "--disable",
        "auth_elicitation",
        "--disable",
        "tool_call_mcp_elicitation",
        "--disable",
        "skill_mcp_dependency_install",
        "--json",
    ]
    if model_access == "proxy":
        assert gateway_url
        base_url = gateway_url.rstrip("/") + "/v1"
        cmd.extend(
            [
                "-c",
                'model_provider="frontieror_gateway"',
                "-c",
                'model_providers.frontieror_gateway.name="FrontierOR Model Gateway"',
                "-c",
                f'model_providers.frontieror_gateway.base_url="{base_url}"',
                "-c",
                'model_providers.frontieror_gateway.wire_api="responses"',
                "-c",
                'model_providers.frontieror_gateway.env_key="OPENAI_API_KEY"',
            ]
        )
    if resume_session_id:
        cmd.extend(["resume", resume_session_id, prompt])
    else:
        cmd.append(prompt)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--resume-session-id")
    parser.add_argument("--gateway-url")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    agent_id = os.environ.get("FRONTIER_OR_AGENT_ID", "")
    agent_count = int(os.environ.get("FRONTIER_OR_CORAL_AGENT_COUNT", "1"))
    code_home = Path(os.environ["CODEX_HOME"])
    model_access = os.environ.get("FRONTIER_OR_MODEL_ACCESS", "local-auth")
    host_auth = Path("/frontieror/codex-auth.json")
    if not agent_id:
        raise RuntimeError("shared agent container is missing its identity")
    if model_access not in {"local-auth", "proxy"}:
        raise RuntimeError(f"unsupported model access mode: {model_access!r}")
    if model_access == "local-auth" and not host_auth.is_file():
        raise RuntimeError("shared agent container is missing its Codex auth")
    if model_access == "proxy" and (
        not args.gateway_url or not os.environ.get("OPENAI_API_KEY")
    ):
        raise RuntimeError("proxy model access requires a gateway URL and ephemeral token")
    Path(os.environ.get("HOME", "/tmp")).mkdir(parents=True, exist_ok=True)
    code_home.mkdir(parents=True, exist_ok=True)
    os.chmod(code_home, 0o700)
    auth_link = code_home / "auth.json"
    if auth_link.is_symlink() or auth_link.exists():
        auth_link.unlink()
    if model_access == "local-auth":
        auth_link.symlink_to(host_auth)

    cmd = build_codex_command(
        model=args.model,
        prompt=args.prompt,
        resume_session_id=args.resume_session_id,
        model_access=model_access,
        gateway_url=args.gateway_url,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=os.environ.copy(),
    )
    observed_actions = 0
    stop_reason = "agent_exit"
    requested_signal: int | None = None
    shutdown = threading.Event()

    def _forward_signal(signum: int, _frame) -> None:
        nonlocal requested_signal
        requested_signal = signum
        shutdown.set()
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") in ACTION_TYPES:
                observed_actions += 1
    finally:
        try:
            return_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return_code = proc.wait(timeout=5)
    if requested_signal == signal.SIGINT:
        stop_reason = "host_interrupt"
    elif requested_signal == signal.SIGTERM:
        stop_reason = "host_terminate"
    elif return_code != 0:
        stop_reason = "agent_error"
    print(json.dumps({
        "type": "frontieror.runtime_exit",
        "agent_id": agent_id,
        "agent_count": agent_count,
        "model_access": model_access,
        "observed_actions": observed_actions,
        "configured_native_max_steps": args.max_steps,
        "action_limit_enforced": False,
        "stop_reason": stop_reason,
        "codex_returncode": return_code,
    }), flush=True)
    return 0 if requested_signal is not None else return_code


if __name__ == "__main__":
    raise SystemExit(main())
