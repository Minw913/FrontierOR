"""Host lifecycle handle that signals the container, not just docker exec."""

from __future__ import annotations

import json
import signal
import subprocess
from dataclasses import dataclass

from trusted_eval_infra.agent.protocol import AgentHandle


@dataclass
class ContainerAgentHandle(AgentHandle):
    container_name: str = ""
    invocation: str = ""
    _cleaned: bool = False

    def _signal_container(self, signum: int) -> None:
        result = subprocess.run(
            ["docker", "exec", self.container_name, "python3",
             "/opt/frontieror/secure_process_control.py", self.invocation, str(signum)],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            # Missing containers are already clean. Any other failure must
            # stop the manager from silently creating another leaked tree.
            state = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", self.container_name],
                capture_output=True, text=True, timeout=10,
            )
            if state.returncode == 0 and state.stdout.strip() == "true":
                raise RuntimeError("could not signal the in-container agent invocation")

    def _cleanup(self) -> None:
        if not self._cleaned:
            self._signal_container(signal.SIGKILL)
            self._cleaned = True

    @property
    def alive(self) -> bool:
        live = self.process is not None and self.process.poll() is None
        if not live:
            self._cleanup()
        return live

    def _finish(self, signum: int) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                self._signal_container(signum)
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._cleanup()
                    self.process.wait(timeout=10)
        finally:
            self._cleanup()
            self._close()

    def stop(self) -> None:
        self._finish(signal.SIGTERM)

    def interrupt(self) -> str | None:
        self._finish(signal.SIGINT)
        # Codex's thread ID is the resumable session ID, including first start.
        with self.log_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "thread.started":
                    self.session_id = event.get("thread_id") or self.session_id
        return self.session_id
