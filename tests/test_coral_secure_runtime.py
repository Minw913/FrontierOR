import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from test_time_self_evolution.coral.eval_broker import drain_eval_requests
from test_time_self_evolution.coral.secure_egress_proxy import _allowed
from test_time_self_evolution.coral.secure_instructions import generate_secure_coral_md


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "code.py").write_text("print('ok')\n", encoding="utf-8")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-m", "seed")
    _git(repo, "checkout", "-b", "coral/agent-1")
    (repo / "code.py").write_text("print('agent')\n", encoding="utf-8")
    _git(repo, "commit", "-am", "agent")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_secure_instructions_use_native_coral_lifecycle_and_no_research():
    config = SimpleNamespace(grader=SimpleNamespace(args={"agent_max_steps": 20}))
    rendered = generate_secure_coral_md(config, "agent-1")
    assert "completed tool actions" not in rendered
    assert "native attempt" in rendered
    assert "wall-clock budget" in rendered
    assert "coral eval" in rendered
    assert "Network access" in rendered
    assert "web search" not in rendered.lower()
    assert "deep-research" not in rendered


def test_secure_multi_agent_instructions_use_shared_notes():
    config = SimpleNamespace(grader=SimpleNamespace(args={"agent_max_steps": 10}))
    rendered = generate_secure_coral_md(
        config,
        "agent-2",
        single_agent=False,
    )
    assert "submitted multi-agent system" in rendered
    assert "`.codex/notes/`" in rendered
    assert "Codex-spawned subagents remain disabled" in rendered


def test_egress_allowlist_only_accepts_model_service_domains():
    assert _allowed("api.openai.com")
    assert _allowed("chatgpt.com")
    assert _allowed("auth.openai.com")
    assert not _allowed("openai.com.evil.example")
    assert not _allowed("github.com")
    assert not _allowed("127.0.0.1")


def test_eval_broker_accepts_agent_branch_and_writes_trusted_pending_attempt(tmp_path):
    repo, commit_hash = _repo(tmp_path)
    coral_dir = tmp_path / ".coral"
    inbox = coral_dir / "private" / "eval_requests" / "inbox"
    inbox.mkdir(parents=True)
    request = {
        "schema_version": 1,
        "nonce": "a" * 32,
        "agent_id": "agent-1",
        "commit_hash": commit_hash,
        "message": "improve solver",
    }
    (inbox / f"{request['nonce']}.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    task = SimpleNamespace(coral_dir=str(coral_dir), repo_dir=str(repo))

    decisions = drain_eval_requests(task, max_attempts=1)

    assert decisions[0]["accepted"] is True
    attempt = json.loads(
        (coral_dir / "public" / "attempts" / f"{commit_hash}.json").read_text()
    )
    assert attempt["status"] == "pending"
    assert attempt["score"] is None
    assert attempt["metadata"]["submission_channel"] == "trusted_request_broker"
    assert not list(inbox.glob("*.json"))


def test_eval_broker_accepts_registered_second_agent_and_rejects_unregistered(tmp_path):
    repo, _ = _repo(tmp_path)
    _git(repo, "checkout", "-b", "coral/agent-2")
    (repo / "code.py").write_text("print('agent-2')\n", encoding="utf-8")
    _git(repo, "commit", "-am", "agent 2")
    commit_hash = _git(repo, "rev-parse", "HEAD")
    coral_dir = tmp_path / ".coral"
    inbox = coral_dir / "private" / "eval_requests" / "inbox"
    inbox.mkdir(parents=True)

    accepted = {
        "schema_version": 1,
        "nonce": "2" * 32,
        "agent_id": "agent-2",
        "commit_hash": commit_hash,
        "message": "second agent candidate",
    }
    (inbox / f"{accepted['nonce']}.json").write_text(
        json.dumps(accepted),
        encoding="utf-8",
    )
    task = SimpleNamespace(
        coral_dir=str(coral_dir),
        repo_dir=str(repo),
        agent_ids=("agent-1", "agent-2"),
    )
    decisions = drain_eval_requests(task, max_attempts=2)
    assert decisions[0]["accepted"] is True

    rejected = dict(accepted)
    rejected["nonce"] = "3" * 32
    rejected["agent_id"] = "agent-3"
    (inbox / f"{rejected['nonce']}.json").write_text(
        json.dumps(rejected),
        encoding="utf-8",
    )
    decisions = drain_eval_requests(task, max_attempts=2)
    assert decisions[0]["accepted"] is False
    assert decisions[0]["reason"] == "invalid agent id"


def test_eval_broker_rejects_commit_from_another_branch(tmp_path):
    repo, _ = _repo(tmp_path)
    _git(repo, "checkout", "-b", "untrusted")
    (repo / "code.py").write_text("print('forged')\n", encoding="utf-8")
    _git(repo, "commit", "-am", "forged")
    forged = _git(repo, "rev-parse", "HEAD")
    coral_dir = tmp_path / ".coral"
    inbox = coral_dir / "private" / "eval_requests" / "inbox"
    inbox.mkdir(parents=True)
    request = {
        "schema_version": 1,
        "nonce": "b" * 32,
        "agent_id": "agent-1",
        "commit_hash": forged,
        "message": "forged",
    }
    (inbox / f"{request['nonce']}.json").write_text(
        json.dumps(request), encoding="utf-8"
    )

    decisions = drain_eval_requests(
        SimpleNamespace(coral_dir=str(coral_dir), repo_dir=str(repo)), max_attempts=1
    )

    assert decisions[0]["accepted"] is False
    assert "not on the submitting agent branch" in decisions[0]["reason"]
    assert not (coral_dir / "public" / "attempts" / f"{forged}.json").exists()


def test_eval_broker_rejects_symlink_request_without_reading_target(tmp_path):
    repo, commit_hash = _repo(tmp_path)
    coral_dir = tmp_path / ".coral"
    inbox = coral_dir / "private" / "eval_requests" / "inbox"
    inbox.mkdir(parents=True)
    nonce = "c" * 32
    secret = tmp_path / "host-secret.json"
    secret.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "nonce": nonce,
                "agent_id": "agent-1",
                "commit_hash": commit_hash,
                "message": "steal",
            }
        ),
        encoding="utf-8",
    )
    (inbox / f"{nonce}.json").symlink_to(secret)

    decisions = drain_eval_requests(
        SimpleNamespace(coral_dir=str(coral_dir), repo_dir=str(repo)),
        max_attempts=1,
    )

    assert decisions[0]["accepted"] is False
    assert "symlink" in decisions[0]["reason"]
    assert decisions[0]["request"] == {}


def test_eval_broker_rejects_symlink_code_in_commit(tmp_path):
    repo, _ = _repo(tmp_path)
    (repo / "code.py").unlink()
    (repo / "code.py").symlink_to("/etc/passwd")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-m", "symlink")
    commit_hash = _git(repo, "rev-parse", "HEAD")
    coral_dir = tmp_path / ".coral"
    inbox = coral_dir / "private" / "eval_requests" / "inbox"
    inbox.mkdir(parents=True)
    nonce = "d" * 32
    request = {
        "schema_version": 1,
        "nonce": nonce,
        "agent_id": "agent-1",
        "commit_hash": commit_hash,
        "message": "symlink",
    }
    (inbox / f"{nonce}.json").write_text(json.dumps(request), encoding="utf-8")

    decisions = drain_eval_requests(
        SimpleNamespace(coral_dir=str(coral_dir), repo_dir=str(repo)),
        max_attempts=1,
    )

    assert decisions[0]["accepted"] is False
    assert "non-regular" in decisions[0]["reason"]


def test_secure_runtime_docker_command_has_outer_boundary(tmp_path, monkeypatch):
    from test_time_self_evolution.coral import secure_runtime

    run_dir = tmp_path / "run"
    worktree = run_dir / "agents" / "agent-1"
    second_worktree = run_dir / "agents" / "agent-2"
    repo = run_dir / "repo"
    attempts = run_dir / ".coral" / "public" / "attempts"
    worktree.mkdir(parents=True)
    second_worktree.mkdir(parents=True)
    repo.mkdir(parents=True)
    (repo / ".git" / "hooks").mkdir(parents=True)
    (repo / ".git" / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    attempts.mkdir(parents=True)
    (worktree / ".coral_agent_id").write_text("agent-1", encoding="utf-8")
    (second_worktree / ".coral_agent_id").write_text("agent-2", encoding="utf-8")
    auth_home = tmp_path / "codex"
    auth_home.mkdir()
    (auth_home / "auth.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(auth_home))
    monkeypatch.setenv("FRONTIER_OR_CORAL_AGENT_IMAGE", "test-image@sha256:abc")
    monkeypatch.setenv("FRONTIER_OR_CORAL_AGENT_COUNT", "2")

    docker_calls = []
    shared_running = False
    def fake_docker(*args, check=True):
        nonlocal shared_running
        docker_calls.append(args)
        if args[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(args, 0, "sha256:abc\n", "")
        if args and args[0] == "run" and any(
            str(value).startswith("frontieror-agent-system-") for value in args
        ):
            shared_running = True
        if args and args[0] == "inspect" and any(
            str(value).startswith("frontieror-agent-system-") for value in args
        ):
            return subprocess.CompletedProcess(
                args,
                0 if shared_running else 1,
                "true\n" if shared_running else "",
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    exec_commands = []

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        exec_commands.append(cmd)
        return FakeProcess()

    monkeypatch.setattr(secure_runtime, "_docker", fake_docker)
    monkeypatch.setattr(secure_runtime.subprocess, "Popen", fake_popen)
    runtime = secure_runtime.SecureCodexRuntime()
    handle = runtime.start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        model="gpt-5.4",
        max_turns=20,
        log_dir=run_dir / ".coral" / "public" / "logs",
    )
    handle._log_file.close()

    second_agent = runtime.start(
        worktree_path=second_worktree,
        coral_md_path=second_worktree / "AGENTS.md",
        model="gpt-5.4",
        max_turns=20,
        log_dir=run_dir / ".coral" / "public" / "logs",
    )
    second_agent._log_file.close()

    container_runs = [
        call for call in docker_calls
        if call and call[0] == "run"
        and any(str(value).startswith("frontieror-agent-system-") for value in call)
    ]
    assert len(container_runs) == 1
    outer = container_runs[0]
    outer_flat = " ".join(outer)
    assert "--read-only" in outer
    assert "--log-driver=none" in outer
    assert "--cap-drop=ALL" in outer
    assert "no-new-privileges" in outer
    assert "--network" in outer
    assert "--ulimit" in outer
    assert "GIT_NO_REPLACE_OBJECTS=1" in outer
    assert "dst=" + str(attempts) + ",readonly" in outer_flat
    assert "dst=/frontieror/codex-auth.json,readonly" in outer_flat
    assert "dst=" + str(repo / ".git" / "config") + ",readonly" in outer_flat
    assert str(repo / ".git" / "hooks") + ":rw,nosuid,nodev,noexec" in outer_flat
    assert "sha256:abc" in outer

    assert len(exec_commands) == 2
    assert all(command[:2] == ["docker", "exec"] for command in exec_commands)
    assert exec_commands[0][exec_commands[0].index("--workdir") + 1] == str(worktree)
    assert exec_commands[1][exec_commands[1].index("--workdir") + 1] == str(second_worktree)
    assert "--max-steps 20" in " ".join(exec_commands[0])
    assert "--max-steps 20" in " ".join(exec_commands[1])
    all_exec = " ".join(" ".join(command) for command in exec_commands)
    assert "--dangerously-bypass-approvals-and-sandbox" not in all_exec

    with handle.log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({
            "type": "item.completed",
            "item": {"type": "command_execution"},
        }) + "\n")
    second = runtime.start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        model="gpt-5.4",
        max_turns=20,
        log_dir=run_dir / ".coral" / "public" / "logs",
    )
    second._log_file.close()
    assert len(container_runs) == 1
    assert "--max-steps 20" in " ".join(exec_commands[-1])
    activity = json.loads(
        (
            run_dir
            / ".coral"
            / "private"
            / "audit"
            / "agent_activity.json"
        ).read_text()
    )
    assert activity["configured_agent_count"] == 2
    assert activity["native_max_steps"] == 20
    assert activity["tool_action_limit_enforced"] is False
    assert activity["agents"]["agent-1"]["observed_actions"] == 1
    assert activity["agents"]["agent-1"]["runtime_starts"] == 2
    assert activity["agents"]["agent-2"]["runtime_starts"] == 1
    secure_runtime.finalize_secure_runtime_audit(run_dir)
    policy = json.loads(
        (run_dir / ".coral" / "private" / "audit" / "agent_runtime.json").read_text()
    )
    assert policy["model"] == "gpt-5.4"
    assert policy["verification_tier"] == "isolated-shared-container-local-auth"
    assert set(policy["agents"]) == {"agent-1", "agent-2"}
    observed = policy["policy"]["observed_activity"]
    assert observed["tool_action_limit_enforced"] is False
    assert observed["totals"]["observed_actions"] == 1
    assert observed["totals"]["runtime_starts"] == 3


def test_secure_model_proxy_owns_upstream_key_and_exposes_internal_alias(
    tmp_path,
    monkeypatch,
):
    from test_time_self_evolution.coral import secure_runtime

    calls = []

    def fake_docker(*args, check=True):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(secure_runtime, "_docker", fake_docker)
    proxy_name, master_token = secure_runtime._ensure_model_proxy(
        "sha256:test-image",
        tmp_path / "run",
        "frontieror-agent-test",
        "sk-upstream-test",
        "openai/gpt-5.3-codex",
        2,
    )

    run = next(call for call in calls if call and call[0] == "run")
    connected = next(
        call for call in calls if call[:2] == ("network", "connect")
    )
    flat = " ".join(run)
    assert "--network bridge" in flat
    assert "host.docker.internal" not in flat
    assert "/opt/frontieror/secure_model_proxy.py" in flat
    assert "FRONTIER_OR_ALLOWED_MODEL=openai/gpt-5.3-codex" in flat
    assert "OPENROUTER_API_KEY=sk-upstream-test" in flat
    assert connected[-2:] == ("frontieror-agent-test", proxy_name)
    assert "frontieror-model-gateway" in connected
    assert master_token
    assert "sk-upstream-test" not in secure_runtime.agent_token(
        master_token,
        "agent-1",
    )


def test_secure_model_proxy_uses_distinct_ephemeral_agent_tokens():
    from test_time_self_evolution.coral.secure_model_proxy import agent_token

    first = agent_token("master", "agent-1")
    second = agent_token("master", "agent-2")
    assert first.startswith("sk-frontieror-agent-1-")
    assert second.startswith("sk-frontieror-agent-2-")
    assert first != second
    source = (
        Path(__file__).resolve().parents[1]
        / "frontieror"
        / "infra"
        / "agent"
        / "model_proxy.py"
    ).read_text(encoding="utf-8")
    assert 'request_payload["model"] = self.allowed_model' in source
    assert "http.client.HTTPSConnection(" in source
    assert '"/api/v1/responses"' in source
    assert "litellm" not in source


def test_secure_runtime_proxy_mounts_no_host_auth_or_upstream_key(
    tmp_path,
    monkeypatch,
):
    from test_time_self_evolution.coral import secure_runtime

    run_dir = tmp_path / "run"
    worktree = run_dir / "agents" / "agent-1"
    worktree.mkdir(parents=True)
    (worktree / ".coral_agent_id").write_text("agent-1", encoding="utf-8")
    monkeypatch.setenv("FRONTIER_OR_CORAL_MODEL_ACCESS", "proxy")
    monkeypatch.setenv("FRONTIER_OR_CORAL_AGENT_COUNT", "2")
    monkeypatch.setenv("FRONTIER_OR_CORAL_AGENT_IMAGE", "test-image")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-upstream-secret")
    monkeypatch.setenv(
        "FRONTIER_OR_CORAL_UPSTREAM_MODEL",
        "openai/gpt-5.3-codex",
    )

    monkeypatch.setattr(
        secure_runtime,
        "_docker",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(secure_runtime, "_image_digest", lambda _image: "sha256:abc")
    monkeypatch.setattr(
        secure_runtime,
        "_ensure_egress",
        lambda *_args: ("internal-network", "egress-proxy"),
    )
    monkeypatch.setattr(
        secure_runtime,
        "_ensure_model_proxy",
        lambda *_args: ("model-proxy", "proxy-master"),
    )
    shared = {}

    def fake_shared(**kwargs):
        shared.update(kwargs)
        return "agent-system"

    monkeypatch.setattr(
        secure_runtime,
        "_ensure_shared_agent_container",
        fake_shared,
    )
    commands = []

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(secure_runtime.subprocess, "Popen", fake_popen)
    handle = secure_runtime.SecureCodexRuntime().start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        model="gpt-5.3-codex",
        max_turns=10,
    )
    handle._log_file.close()

    assert shared["auth_path"] is None
    command = " ".join(commands[0])
    assert "sk-upstream-secret" not in command
    assert secure_runtime.agent_token("proxy-master", "agent-1") in command
    assert "--gateway-url http://frontieror-model-gateway:8080" in command
    policy = json.loads(
        (
            run_dir
            / ".coral"
            / "private"
            / "audit"
            / "agent_runtime.json"
        ).read_text()
    )
    assert policy["verification_tier"] == "isolated-shared-container-proxy"
    assert policy["mounts"]["codex_auth"].startswith("not mounted")


def test_secure_entrypoint_disables_external_tool_surfaces():
    from test_time_self_evolution.coral.secure_codex_entrypoint import (
        build_codex_command,
    )

    source = (
        Path(__file__).resolve().parents[1]
        / "frontieror" / "infra" / "agent" / "codex_entrypoint.py"
    ).read_text(encoding="utf-8")
    command = build_codex_command(
        model="gpt-5.3-codex",
        prompt="begin",
        resume_session_id=None,
        model_access="proxy",
        gateway_url="http://model-gateway:8080",
    )
    pairs = set(zip(command, command[1:]))

    assert "dangerously-bypass" not in source
    assert 'web_search="disabled"' in source
    assert ("--disable", "apps") in pairs
    assert ("--disable", "plugins") in pairs
    assert ("--sandbox", "danger-full-access") in pairs
    assert "use_legacy_landlock" not in source
    assert "step_budget_exhausted" not in source
    assert '"action_limit_enforced": False' in source
    assert 'model_provider="frontieror_gateway"' in source
    assert 'env_key="OPENAI_API_KEY"' in source


def test_secure_entrypoint_places_parent_options_before_resume():
    from test_time_self_evolution.coral.secure_codex_entrypoint import (
        build_codex_command,
    )

    command = build_codex_command(
        model="gpt-5.3-codex",
        prompt="continue",
        resume_session_id="session-id",
        model_access="proxy",
        gateway_url="http://model-gateway:8080",
    )

    resume_index = command.index("resume")
    assert command.index("--sandbox") < resume_index
    assert command.index("--ignore-user-config") < resume_index
    assert command[-3:] == ["resume", "session-id", "continue"]
