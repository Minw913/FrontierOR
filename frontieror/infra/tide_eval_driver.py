"""Tide-eval driver for trusted FrontierOR process episodes.

This module runs in the Tide-eval Python environment.  It implements Tide's
small Executor protocol locally so FrontierOR does not depend on a fork of
Tide-eval or on the unrelated ``gauthierpiarrette/tide`` fleet package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class WorkerProtocolError(RuntimeError):
    """Raised when the trusted FrontierOR worker violates the wire contract."""


class FrontierORExecutor:
    """Execute one Tide episode through the trusted FrontierOR worker."""

    def __init__(
        self,
        worker_command: list[str],
        *,
        cwd: str,
        timeout_grace_sec: float = 120.0,
    ) -> None:
        if not worker_command:
            raise ValueError("worker_command must not be empty")
        if timeout_grace_sec < 0:
            raise ValueError("timeout_grace_sec must be non-negative")
        self.worker_command = list(worker_command)
        self.cwd = cwd
        self.timeout_grace_sec = timeout_grace_sec

    async def execute(self, spec):
        from tide.types import EpisodeResult, TracePoint

        request = {
            "schema_version": SCHEMA_VERSION,
            "episode": {
                "task": spec.task,
                "agent": spec.agent,
                "overrides": spec.overrides,
            },
        }
        process = await asyncio.create_subprocess_exec(
            *self.worker_command,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timeout = spec.agent.get("override_timeout_sec")
        if timeout is not None:
            timeout = float(timeout) + self.timeout_grace_sec
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(request).encode()),
                timeout=timeout,
            )
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
            return EpisodeResult(
                rewards={},
                error="trusted FrontierOR worker exceeded the episode deadline",
            )

        try:
            result = _decode_worker_response(stdout)
        except WorkerProtocolError as exc:
            detail = stderr.decode(errors="replace")[-1000:]
            return EpisodeResult(
                rewards={},
                error=f"{exc}; worker stderr: {detail}",
            )
        if process.returncode != 0:
            detail = stderr.decode(errors="replace")[-1000:]
            result["error"] = (
                f"trusted FrontierOR worker exited {process.returncode}: {detail}"
            )
        return EpisodeResult(
            rewards=result["rewards"],
            uri=result.get("uri"),
            trace=tuple(
                TracePoint(
                    t=point["t"],
                    score=point["score"],
                    data=point.get("data", {}),
                )
                for point in result["trace"]
            ),
            error=result.get("error"),
            usage=result["usage"],
        )


def _numeric_mapping(value: Any, label: str) -> dict[str, float | int]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and isinstance(number, int | float)
        and not isinstance(number, bool)
        and math.isfinite(float(number))
        for key, number in value.items()
    ):
        raise WorkerProtocolError(f"worker {label} must be a numeric object")
    return value


def _decode_worker_response(stdout: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("worker returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise WorkerProtocolError("worker returned an unsupported schema version")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise WorkerProtocolError("worker result must be an object")
    result["rewards"] = _numeric_mapping(result.get("rewards"), "rewards")
    result["usage"] = _numeric_mapping(result.get("usage", {}), "usage")
    trace = result.get("trace", [])
    if not isinstance(trace, list):
        raise WorkerProtocolError("worker trace must be an array")
    for point in trace:
        if (
            not isinstance(point, dict)
            or not isinstance(point.get("t"), int | float)
            or isinstance(point.get("t"), bool)
            or not math.isfinite(float(point["t"]))
            or not isinstance(point.get("score"), int | float)
            or isinstance(point.get("score"), bool)
            or not math.isfinite(float(point["score"]))
            or not isinstance(point.get("data", {}), dict)
        ):
            raise WorkerProtocolError("worker trace contains an invalid point")
    if result.get("uri") is not None and not isinstance(result["uri"], str):
        raise WorkerProtocolError("worker uri must be a string or null")
    if result.get("error") is not None and not isinstance(result["error"], str):
        raise WorkerProtocolError("worker error must be a string or null")
    return result


def _load_request(path: Path) -> dict[str, Any]:
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid Tide-eval request: {exc}") from exc
    if not isinstance(request, dict) or request.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("invalid Tide-eval request schema")
    if not isinstance(request.get("worker_command"), list):
        raise SystemExit("Tide-eval request has no worker command")
    if not isinstance(request.get("calls"), list):
        raise SystemExit("Tide-eval request has no episode calls")
    return request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args(argv)
    request = _load_request(Path(args.request))

    from tide import Lab

    executor = FrontierORExecutor(
        request["worker_command"],
        cwd=request["repo_root"],
        timeout_grace_sec=float(request.get("timeout_grace_sec", 120)),
    )
    lab = Lab(
        request["lab"],
        executor=executor,
        concurrency=int(request.get("concurrency", 1)),
    )
    rows = asyncio.run(lab.run_many(request["calls"]))
    failures = 0
    for row in rows:
        error = row.tags.get("error")
        if error or "reward" not in row.rewards:
            failures += 1
        print(
            json.dumps(
                {
                    "key": row.key,
                    "task": row.task,
                    "rewards": row.rewards,
                    "error": error,
                    "uri": row.uri,
                },
                sort_keys=True,
            )
        )
    print(f"Tide lab: {Path(request['lab']).resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
