"""Small runtime protocol independent of the optional CORAL installation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any


@dataclass
class AgentHandle:
    agent_id: str
    process: subprocess.Popen | None
    worktree_path: Path
    log_path: Path
    session_id: str | None = None
    _log_file: object | None = None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _close(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    def stop(self) -> None:
        if self.process is not None and self.alive:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                terminate = getattr(self.process, "terminate", None)
                if terminate is not None:
                    terminate()
            try:
                self.process.wait(timeout=10)
            except (AttributeError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    kill = getattr(self.process, "kill", None)
                    if kill is not None:
                        kill()
        self._close()

    def interrupt(self) -> str | None:
        if self.process is not None and self.alive:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                send_signal = getattr(self.process, "send_signal", None)
                if send_signal is not None:
                    send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=15)
            except (AttributeError, subprocess.TimeoutExpired):
                self.stop()
        self._close()
        return self.session_id


def write_agent_log_entry(
    log_file: IO[str],
    *,
    prompt: str,
    source: str,
    agent_id: str,
    session_id: str | None = None,
    task_name: str | None = None,
    task_description: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "type": "coral",
        "subtype": "prompt",
        "source": source,
        "agent_id": agent_id,
        "prompt": prompt,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    for key, value in (
        ("session_id", session_id),
        ("task_name", task_name),
        ("task_description", task_description),
    ):
        if value:
            payload[key] = value
    log_file.write(json.dumps(payload, sort_keys=True) + "\n")
    log_file.flush()
