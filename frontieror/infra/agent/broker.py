"""Trusted host-side broker for untrusted CORAL evaluation requests."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from frontieror.infra.files import SecureFileError, read_regular_file


MAX_REQUEST_BYTES = 64 * 1024
MAX_CODE_BYTES = 4 * 1024 * 1024
MAX_TREE_FILE_BYTES = 8 * 1024 * 1024
MAX_TREE_BYTES = 64 * 1024 * 1024
MAX_TREE_FILES = 512
MAX_REQUESTS_PER_DRAIN = 64
GIT_TIMEOUT_SECONDS = 10

def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(repo),
                *args,
            ],
            text=True,
            capture_output=True,
            env=env,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            args=["git", "-C", str(repo), *args],
            returncode=124,
            stdout="",
            stderr=str(exc),
        )


def _validate_tree(repo: Path, commit_hash: str) -> tuple[bool, str]:
    tree = _git(repo, "ls-tree", "-r", "-l", "-z", commit_hash)
    if tree.returncode != 0:
        return False, "cannot inspect submitted commit tree"

    file_count = 0
    total_bytes = 0
    code_size = None
    for raw_entry in tree.stdout.split("\0"):
        if not raw_entry:
            continue
        try:
            header, path = raw_entry.split("\t", 1)
            mode, object_type, _object_id, size_text = header.split()
            size = int(size_text)
        except (ValueError, TypeError):
            return False, "submitted commit has an invalid tree entry"
        if mode not in {"100644", "100755"} or object_type != "blob":
            return False, "submitted commit contains a non-regular file"
        if size < 0 or size > MAX_TREE_FILE_BYTES:
            return False, "submitted commit contains an oversized file"
        file_count += 1
        total_bytes += size
        if file_count > MAX_TREE_FILES or total_bytes > MAX_TREE_BYTES:
            return False, "submitted commit exceeds the repository size limit"
        if path == "code.py":
            code_size = size

    if code_size is None:
        return False, "commit does not contain a regular code.py"
    if code_size > MAX_CODE_BYTES:
        return False, "code.py exceeds the submission size limit"
    code = _git(repo, "show", f"{commit_hash}:code.py")
    if code.returncode != 0:
        return False, "cannot read code.py from submitted commit"
    if len(code.stdout.encode("utf-8")) > MAX_CODE_BYTES:
        return False, "code.py exceeds the submission size limit"
    return True, "accepted"


def _validate_request(
    repo: Path,
    request: dict,
    *,
    allowed_agent_ids: frozenset[str],
    request_name: str | None = None,
) -> tuple[bool, str, str | None]:
    required = {"schema_version", "nonce", "agent_id", "commit_hash", "message"}
    if set(request) != required or request.get("schema_version") != 1:
        return False, "invalid request schema", None
    commit_hash = request.get("commit_hash", "")
    agent_id = request.get("agent_id", "")
    nonce = request.get("nonce", "")
    message = request.get("message")
    if not isinstance(commit_hash, str) or re.fullmatch(r"[0-9a-f]{40}", commit_hash) is None:
        return False, "invalid commit hash", None
    if agent_id not in allowed_agent_ids:
        return False, "invalid agent id", None
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32,64}", nonce) is None:
        return False, "invalid request nonce", None
    if request_name is not None and request_name != f"{nonce}.json":
        return False, "request filename does not match nonce", None
    if not isinstance(message, str) or not message.strip() or len(message) > 200:
        return False, "invalid evaluation message", None
    branch = f"refs/heads/coral/{agent_id}"
    if _git(repo, "merge-base", "--is-ancestor", commit_hash, branch).returncode != 0:
        return False, "commit is not on the submitting agent branch", None
    valid_tree, tree_reason = _validate_tree(repo, commit_hash)
    if not valid_tree:
        return False, tree_reason, None
    parent = _git(repo, "rev-parse", f"{commit_hash}^")
    candidate_parent = parent.stdout.strip()
    parent_hash = (
        candidate_parent
        if parent.returncode == 0
        and re.fullmatch(r"[0-9a-f]{40}", candidate_parent) is not None
        else None
    )
    return True, "accepted", parent_hash


def _read_request(path: Path) -> tuple[dict, str | None]:
    try:
        raw = read_regular_file(
            path,
            max_bytes=MAX_REQUEST_BYTES,
            label="evaluation request",
        )
        request = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return {}, "evaluation request must be UTF-8"
    except json.JSONDecodeError:
        return {}, "evaluation request is not valid JSON"
    except SecureFileError as exc:
        return {}, str(exc)
    if not isinstance(request, dict):
        return {}, "evaluation request must contain a JSON object"
    return request, None


def drain_eval_requests(
    task,
    max_attempts: int,
    *,
    allowed_agent_ids: tuple[str, ...] | list[str] | None = None,
) -> list[dict]:
    coral_dir = Path(task.coral_dir)
    if allowed_agent_ids is None:
        allowed_agent_ids = list(getattr(task, "agent_ids", ("agent-1",)))
    registered_agents = frozenset(str(agent_id) for agent_id in allowed_agent_ids)
    if not registered_agents:
        raise ValueError("evaluation broker requires at least one registered agent")
    if any(re.fullmatch(r"agent-[1-9][0-9]*", agent_id) is None for agent_id in registered_agents):
        raise ValueError("evaluation broker received an invalid registered agent id")
    inbox = coral_dir / "private" / "eval_requests" / "inbox"
    archive = coral_dir / "private" / "eval_requests" / "archive"
    attempts_dir = coral_dir / "public" / "attempts"
    audit_path = coral_dir / "private" / "audit" / "eval_requests.jsonl"
    inbox.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = len(list(attempts_dir.glob("*.json")))
    decisions = []
    for path in sorted(inbox.glob("*.json"))[:MAX_REQUESTS_PER_DRAIN]:
        request, read_error = _read_request(path)
        if read_error is not None:
            valid, reason, parent_hash = False, read_error, None
        elif accepted_count >= max_attempts:
            valid, reason, parent_hash = False, "evaluation budget exhausted", None
        else:
            valid, reason, parent_hash = _validate_request(
                Path(task.repo_dir),
                request,
                allowed_agent_ids=registered_agents,
                request_name=path.name,
            )
        commit_hash = request.get("commit_hash") if isinstance(request, dict) else None
        if valid and (attempts_dir / f"{commit_hash}.json").exists():
            valid, reason = False, "commit was already submitted"
        if valid and commit_hash:
            attempt = {
                "commit_hash": commit_hash,
                "agent_id": request["agent_id"],
                "title": str(request["message"])[:200],
                "score": None,
                "status": "pending",
                "parent_hash": parent_hash,
                "timestamp": datetime.now(UTC).isoformat(),
                "feedback": "",
                "metadata": {"submission_channel": "trusted_request_broker"},
            }
            _atomic_json(attempts_dir / f"{commit_hash}.json", attempt)
            accepted_count += 1
        decision = {
            "timestamp": datetime.now(UTC).isoformat(),
            "accepted": valid,
            "reason": reason,
            "request": request,
        }
        with audit_path.open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(decision, sort_keys=True) + "\n")
        _atomic_json(archive / path.name, decision)
        path.unlink(missing_ok=True)
        decisions.append(decision)
    return decisions
