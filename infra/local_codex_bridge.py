"""Loopback OpenAI-compatible bridge backed by the local Codex CLI.

The bridge is for trusted model-in-framework runners. Codex receives prompt
text in an empty read-only workspace. Any item outside the explicit
text/reasoning allowlist fails the request, so tool output cannot contribute a
candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable


MAX_REQUEST_BYTES = 32 * 1024 * 1024
SAFE_ITEM_TYPES = {"agent_message", "reasoning", "todo_list"}
DISABLED_FEATURES = (
    "apps",
    "plugins",
    "remote_plugin",
    "plugin_sharing",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "image_generation",
    "multi_agent",
    "auth_elicitation",
    "tool_call_mcp_elicitation",
    "skill_mcp_dependency_install",
    "shell_tool",
    "unified_exec",
    "code_mode",
    "code_mode_host",
    "view_image",
    "skill_search",
    "tool_suggest",
    "goals",
)


def normalize_codex_model(model: str) -> str:
    return str(model).strip().removeprefix("openai/")


def render_messages(messages: Iterable[dict[str, Any]]) -> str:
    sections = [
        "You are a text-only model backend for an optimization evolution framework.",
        "Do not use shell, file, web, MCP, or other tools. Return only the answer requested by the conversation.",
    ]
    for message in messages:
        role = str(message.get("role", "user")).strip().upper() or "USER"
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {
                    "text",
                    "input_text",
                    "output_text",
                }:
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)
        sections.append(f"<{role}>\n{content}\n</{role}>")
    sections.append("Respond to the conversation above without using tools.")
    return "\n\n".join(sections)


def build_codex_command(model: str, output_path: Path) -> list[str]:
    command = [
        "codex",
        "exec",
        "-",
        "--model",
        normalize_codex_model(model),
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
    ]
    for feature in DISABLED_FEATURES:
        command.extend(("--disable", feature))
    command.extend(("--output-last-message", os.fspath(output_path), "--json"))
    return command


def parse_codex_events(stdout: str) -> tuple[set[str], dict[str, int]]:
    forbidden_events: set[str] = set()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            forbidden_events.add("malformed_event")
            continue
        event_type = str(event.get("type", ""))
        if event_type == "error" or event_type.endswith(".failed"):
            forbidden_events.add(event_type or "error")
        item = event.get("item") or {}
        item_type = item.get("type")
        if (
            event.get("type") in {"item.started", "item.completed"}
            and item_type
            and item_type not in SAFE_ITEM_TYPES
        ):
            forbidden_events.add(str(item_type))
        event_usage = event.get("usage")
        if isinstance(event_usage, dict):
            usage["prompt_tokens"] = int(
                event_usage.get("input_tokens", event_usage.get("prompt_tokens", 0))
                or 0
            )
            usage["completion_tokens"] = int(
                event_usage.get(
                    "output_tokens", event_usage.get("completion_tokens", 0)
                )
                or 0
            )
            usage["cached_tokens"] = int(
                event_usage.get(
                    "cached_input_tokens", event_usage.get("cached_tokens", 0)
                )
                or 0
            )
    return forbidden_events, usage


class LocalCodexRunner:
    def __init__(
        self,
        *,
        allowed_models: Iterable[str],
        timeout: int,
        max_concurrency: int,
        audit_path: Path,
    ) -> None:
        self.allowed_models = {normalize_codex_model(model) for model in allowed_models}
        self.timeout = timeout
        self.semaphore = threading.BoundedSemaphore(max_concurrency)
        self.audit_path = audit_path
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_lock = threading.Lock()

    def _audit(self, payload: dict[str, Any]) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        with self._audit_lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def generate(
        self,
        *,
        requested_model: str,
        messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, int], str]:
        model = normalize_codex_model(requested_model)
        if model not in self.allowed_models:
            raise ValueError(f"model {requested_model!r} is not allowed by this bridge")
        prompt = render_messages(messages)
        request_id = f"local-codex-{uuid.uuid4().hex}"
        started = time.monotonic()
        returncode = -1
        forbidden_events: set[str] = set()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        error: str | None = None
        response = ""
        try:
            with self.semaphore, tempfile.TemporaryDirectory(
                prefix="frontieror-local-codex-"
            ) as directory:
                output_path = Path(directory) / "response.txt"
                process = subprocess.run(
                    build_codex_command(model, output_path),
                    cwd=directory,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                    env=os.environ.copy(),
                )
                returncode = process.returncode
                forbidden_events, usage = parse_codex_events(process.stdout)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"codex exec exited with code {process.returncode}: "
                        f"{(process.stderr or process.stdout)[-1000:]}"
                    )
                if forbidden_events:
                    raise RuntimeError(
                        "local Codex attempted forbidden tools: "
                        + ", ".join(sorted(forbidden_events))
                    )
                try:
                    response = output_path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise RuntimeError("codex exec produced no final response") from exc
                if not response.strip():
                    raise RuntimeError("codex exec produced an empty final response")
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
            error = str(exc)
            raise
        finally:
            self._audit(
                {
                    "request_id": request_id,
                    "model": model,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_bytes": len(prompt.encode()),
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "response_bytes": len(response.encode()),
                    "tool_events": sorted(forbidden_events),
                    "returncode": returncode,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "usage": usage,
                    "error": error,
                }
            )
        return response, usage, request_id


class _BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    runner: LocalCodexRunner
    api_token: str

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model"}
                        for model in sorted(self.runner.allowed_models)
                    ],
                },
            )
        else:
            self._write_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": {"message": "not found"}})
            return
        if self.headers.get("Authorization") != f"Bearer {self.api_token}":
            self._write_json(401, {"error": {"message": "invalid bridge token"}})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._write_json(413, {"error": {"message": "invalid request size"}})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            model = str(payload["model"])
            messages = payload["messages"]
            if not isinstance(messages, list) or not all(
                isinstance(message, dict) for message in messages
            ):
                raise ValueError("messages must be a list of objects")
            response, usage, request_id = self.runner.generate(
                requested_model=model,
                messages=messages,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._write_json(400, {"error": {"message": str(exc)}})
            return
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self._write_json(502, {"error": {"message": str(exc)}})
            return
        self._write_json(
            200,
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": normalize_codex_model(model),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )


class LocalCodexBridge(AbstractContextManager["LocalCodexBridge"]):
    def __init__(
        self,
        *,
        allowed_models: Iterable[str],
        audit_path: str | Path,
        timeout: int = 900,
        max_concurrency: int = 1,
    ) -> None:
        if timeout < 1 or max_concurrency < 1:
            raise ValueError("timeout and max_concurrency must be positive")
        token = f"sk-frontieror-local-{uuid.uuid4().hex}"
        runner = LocalCodexRunner(
            allowed_models=allowed_models,
            timeout=timeout,
            max_concurrency=max_concurrency,
            audit_path=Path(audit_path),
        )
        handler = type(
            "ConfiguredLocalCodexHandler",
            (_BridgeHandler,),
            {"runner": runner, "api_token": token},
        )
        self.api_token = token
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def api_base(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "LocalCodexBridge":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
