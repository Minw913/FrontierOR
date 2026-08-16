import json
import subprocess
from pathlib import Path

import pytest
import yaml

from frontieror.infra import local_codex_bridge
from test_time_self_evolution.eoh import runner as eoh_runner
from test_time_self_evolution.openevolve import preflight
from test_time_self_evolution.openevolve import runner as openevolve_runner


def test_local_codex_command_is_text_only_and_ephemeral(tmp_path):
    command = local_codex_bridge.build_codex_command(
        "openai/gpt-5.4", tmp_path / "answer.txt"
    )

    assert command[:3] == ["codex", "exec", "-"]
    assert command[command.index("--model") + 1] == "gpt-5.4"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command
    for feature in ("apps", "plugins", "multi_agent", "shell_tool", "unified_exec"):
        assert feature in command


def test_local_codex_event_parser_rejects_unknown_item_type():
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "future_host_capability"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 12,
                        "cached_input_tokens": 3,
                        "output_tokens": 7,
                    },
                }
            ),
        ]
    )

    events, usage = local_codex_bridge.parse_codex_events(stdout)

    assert events == {"future_host_capability"}
    assert usage == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
    }


def test_local_codex_event_parser_rejects_malformed_or_failed_events():
    events, _usage = local_codex_bridge.parse_codex_events(
        'not-json\n{"type":"turn.failed"}\n'
    )

    assert events == {"malformed_event", "turn.failed"}


def test_local_codex_runner_fails_closed_on_tool_event(tmp_path, monkeypatch):
    def fake_run(command, **_kwargs):
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("must be discarded", encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"type": "item.completed", "item": {"type": "file_change"}}
            ),
            stderr="",
        )

    monkeypatch.setattr(local_codex_bridge.subprocess, "run", fake_run)
    audit_path = tmp_path / "audit.jsonl"
    runner = local_codex_bridge.LocalCodexRunner(
        allowed_models=["gpt-5.4"],
        timeout=10,
        max_concurrency=1,
        audit_path=audit_path,
    )

    with pytest.raises(RuntimeError, match="forbidden tools: file_change"):
        runner.generate(
            requested_model="openai/gpt-5.4",
            messages=[{"role": "user", "content": "answer"}],
        )

    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["tool_events"] == ["file_change"]
    assert record["response_bytes"] == 0


def test_openevolve_config_accepts_loopback_model_endpoint(tmp_path):
    destination = tmp_path / "openevolve.yaml"
    openevolve_runner.write_openevolve_config(
        str(destination),
        "gpt-5.4",
        api_base="http://127.0.0.1:12345/v1",
    )

    config = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert config["llm"]["api_base"] == "http://127.0.0.1:12345/v1"


def test_eoh_runtime_endpoint_overrides_yaml_default():
    endpoint = eoh_runner.resolve_eoh_api_endpoint(
        {"MODEL_API_BASE": "http://127.0.0.1:43123/v1"},
        {"api_endpoint": "openrouter.ai"},
    )

    assert endpoint == "http://127.0.0.1:43123/v1"


def test_eoh_rejects_plain_http_non_loopback_endpoint():
    with pytest.raises(ValueError, match="loopback"):
        eoh_runner._api_connection("http://example.com/v1")


def test_model_preflight_rejects_non_loopback_local_bridge():
    issues = preflight._check_openrouter_key(
        {
            "OPENROUTER_API_KEY": "local-token",
            "MODEL_API_BASE": "http://example.com/v1",
        },
        "gpt-5.4",
    )

    assert issues == [
        "model_api_base: local Codex bridge must use an HTTP loopback URL"
    ]
