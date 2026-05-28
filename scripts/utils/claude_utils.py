"""Utility to run claude with token usage tracking."""

import json
import os
import subprocess
import sys
import threading

USAGE_LOG_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "token_usage.jsonl",
)

_write_lock = threading.Lock()


def run_claude_tracked(prompt: str, label: str = ""):
    """
    Run ``claude -p --output-format json`` and track token usage.

    * Prints the text result to stdout.
    * Appends a JSON-line with usage info to ``token_usage.jsonl``.
    * Raises ``subprocess.CalledProcessError`` on non-zero exit.
    * Raises ``FileNotFoundError`` if the ``claude`` binary is missing.
    """
    result = subprocess.run(
        [
            "claude", "-p", "-", "--dangerously-skip-permissions",
            "--output-format", "json",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, "claude")

    usage_entry: dict = {"label": label}
    try:
        data = json.loads(result.stdout)
        text = data.get("result", "")
        if text:
            print(text)
        for key in (
            "cost_usd", "total_cost_usd",
            "duration_ms", "duration_api_ms",
            "num_turns",
            "input_tokens", "output_tokens",
        ):
            if key in data:
                usage_entry[key] = data[key]
        # Some claude versions nest usage under "usage"
        if isinstance(data.get("usage"), dict):
            usage_entry.update(data["usage"])
    except json.JSONDecodeError:
        print(result.stdout)

    with _write_lock:
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(usage_entry) + "\n")


def run_claude_capture(prompt: str, label: str = "") -> str:
    """Like ``run_claude_tracked`` but RETURNS the agent text instead of
    printing it. Thread-safe — does not touch ``sys.stdout``, so it can be
    called concurrently without losing output to other threads' prints.
    """
    result = subprocess.run(
        [
            "claude", "-p", "-", "--dangerously-skip-permissions",
            "--output-format", "json",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "claude",
                                             output=result.stdout,
                                             stderr=result.stderr)
    usage_entry: dict = {"label": label}
    text = ""
    try:
        data = json.loads(result.stdout)
        text = data.get("result", "") or ""
        for key in (
            "cost_usd", "total_cost_usd",
            "duration_ms", "duration_api_ms",
            "num_turns",
            "input_tokens", "output_tokens",
        ):
            if key in data:
                usage_entry[key] = data[key]
        if isinstance(data.get("usage"), dict):
            usage_entry.update(data["usage"])
    except json.JSONDecodeError:
        text = result.stdout
    with _write_lock:
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(usage_entry) + "\n")
    return text
