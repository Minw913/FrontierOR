"""Agent-side `coral` shim for brokered evaluation requests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


def _git(workdir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", f"safe.directory={workdir}", *args],
        cwd=workdir,
        text=True,
        capture_output=True,
        check=check,
    )


def _commit(workdir: Path, message: str) -> str:
    _git(workdir, "add", "-A")
    changed = _git(workdir, "diff", "--cached", "--quiet", check=False)
    if changed.returncode == 0:
        return _git(workdir, "rev-parse", "HEAD").stdout.strip()
    _git(
        workdir,
        "-c", "user.name=FrontierOR Agent",
        "-c", "user.email=agent@frontieror.invalid",
        "commit", "-m", message,
    )
    return _git(workdir, "rev-parse", "HEAD").stdout.strip()


def _submit(message: str, wait: bool) -> int:
    workdir = Path.cwd().resolve()
    inbox = Path(os.environ["FRONTIER_OR_EVAL_REQUEST_DIR"])
    attempts = Path(os.environ["FRONTIER_OR_ATTEMPTS_DIR"])
    agent_id = os.environ["FRONTIER_OR_AGENT_ID"]
    commit_hash = _commit(workdir, message)
    nonce = uuid.uuid4().hex
    request = {
        "schema_version": 1,
        "nonce": nonce,
        "agent_id": agent_id,
        "commit_hash": commit_hash,
        "message": message[:200],
    }
    inbox.mkdir(parents=True, exist_ok=True)
    tmp = inbox / f".{nonce}.tmp"
    target = inbox / f"{nonce}.json"
    tmp.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    print(f"Evaluation requested for {commit_hash[:12]}.")
    if not wait:
        return 0

    timeout = float(os.environ.get("FRONTIER_OR_EVAL_WAIT_SECONDS", "3600"))
    deadline = time.monotonic() + timeout
    result_path = attempts / f"{commit_hash}.json"
    while time.monotonic() < deadline:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(0.2)
            continue
        if result.get("status") != "pending":
            print(json.dumps({
                "commit_hash": commit_hash,
                "status": result.get("status"),
                "score": result.get("score"),
                "feedback": result.get("feedback", ""),
            }, indent=2))
            return 0
        time.sleep(0.2)
    print("Timed out waiting for the trusted grader.", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="coral")
    sub = parser.add_subparsers(dest="command", required=True)
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("-m", "--message", required=True)
    eval_parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if args.command == "eval":
        return _submit(args.message, not args.no_wait)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
