"""Agent-side `coral` shim for brokered evaluation requests."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    pending = [row for row in _public_attempts()
               if row.get("agent_id") == agent_id and row.get("status") == "pending"]
    if pending:
        commit_hash = pending[0]["commit_hash"]
        print(f"Evaluation {commit_hash[:12]} is pending; no new candidate was submitted.", flush=True)
        return _wait_for_result(attempts, commit_hash, wait=wait)
    commit_hash = _commit(workdir, message)
    if (attempts / f"{commit_hash}.json").is_file():
        # An unchanged candidate already has an authoritative record. Reading
        # it must not create another request, including at the attempt limit.
        return _wait_for_result(attempts, commit_hash, wait=wait)
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
    print(f"Evaluation requested for {commit_hash[:12]}.", flush=True)
    return _wait_for_result(attempts, commit_hash, wait=wait, nonce=nonce)


def _wait_for_result(attempts: Path, commit_hash: str, *, wait: bool, nonce: str | None = None) -> int:
    timeout = float(os.environ.get("FRONTIER_OR_EVAL_WAIT_SECONDS", "3600"))
    deadline = time.monotonic() + timeout
    response_dir = os.environ.get("FRONTIER_OR_EVAL_RESPONSE_DIR")
    while True:
        if nonce and response_dir:
            try:
                response = json.loads((Path(response_dir) / f"{nonce}.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                response = None
            if response and not response.get("accepted"):
                existing = response.get("waiting_for_commit")
                if isinstance(existing, str) and re.fullmatch(r"[0-9a-f]{40}", existing):
                    commit_hash, nonce = existing, None
                    print(f"Another evaluation is pending ({existing[:12]}); this candidate was not submitted.", flush=True)
                elif response.get("reason") != "commit was already submitted":
                    print(json.dumps({"status": "request_rejected", "reason": response.get("reason"),
                                      "message": "Submission was not accepted; this is not a candidate score."}), flush=True)
                    return 2
        result_path = attempts / f"{commit_hash}.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            result = None
        if result and result.get("status") != "pending":
            print(json.dumps({
                "commit_hash": commit_hash,
                "status": result.get("status"),
                "score": result.get("score"),
                "feedback": result.get("feedback", ""),
            }, indent=2))
            return 0
        if not wait or time.monotonic() >= deadline:
            print(json.dumps({
                "commit_hash": commit_hash, "status": "pending", "score": None,
                "message": "No score is available yet. Queue or polling time is not evaluation failure. Wait before submitting again.",
            }), flush=True)
            return 0
        time.sleep(0.2)


def main() -> int:
    parser = argparse.ArgumentParser(prog="coral")
    sub = parser.add_subparsers(dest="command", required=True)
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("-m", "--message", required=True)
    eval_parser.add_argument("--no-wait", action="store_true")
    log_parser = sub.add_parser("log")
    log_parser.add_argument("-n", "--count", type=int, default=20)
    log_parser.add_argument("--recent", action="store_true")
    log_parser.add_argument("--agent")
    log_parser.add_argument("--search")
    for command in ("show", "checkout"):
        sub.add_parser(command).add_argument("hash")
    args = parser.parse_args()
    if args.command == "eval":
        return _submit(args.message, not args.no_wait)
    if args.command == "log":
        rows = _public_attempts()
        if args.agent:
            rows = [r for r in rows if r.get("agent_id") == args.agent]
        if args.search:
            rows = [r for r in rows if args.search.lower() in json.dumps(r).lower()]
        if args.recent:
            rows.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        else:
            rows.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
        print(json.dumps(rows[:max(0, min(args.count, 200))], indent=2))
        return 0
    if args.command in {"show", "checkout"}:
        if not re.fullmatch(r"[0-9a-f]{4,40}", args.hash):
            parser.error("expected a commit hash or unambiguous prefix")
        matches = [r for r in _public_attempts() if r["commit_hash"].startswith(args.hash)]
        if len(matches) != 1:
            parser.error("attempt hash is missing or ambiguous")
        row = matches[0]
        if args.command == "show":
            print(json.dumps(row, indent=2))
        else:
            workdir = Path.cwd().resolve()
            # Keep the agent branch and ancestry: restore only the candidate,
            # preserving notes and public task files. Save any local work first.
            _commit(workdir, "Checkpoint before restoring evaluated candidate")
            _git(workdir, "restore", "--source", row["commit_hash"], "--", "code.py")
            print(f"Restored code.py from {row['commit_hash']}; submit with coral eval.")
        return 0
    return 2


def _public_attempts() -> list[dict]:
    rows = []
    allowed = {"commit_hash", "agent_id", "title", "score", "status", "timestamp", "parent_hash", "feedback"}
    for path in Path(os.environ["FRONTIER_OR_ATTEMPTS_DIR"]).glob("*.json"):
        if not re.fullmatch(r"[0-9a-f]{40}", path.stem) or path.is_symlink():
            continue
        try:
            if path.stat().st_size > 1024 * 1024:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("commit_hash") != path.stem:
                continue
            rows.append({k: v for k, v in data.items() if k in allowed})
        except (OSError, ValueError, AttributeError):
            continue
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
