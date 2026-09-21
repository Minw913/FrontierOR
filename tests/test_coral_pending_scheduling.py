import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trusted_eval_infra.agent import submit
from trusted_eval_infra.agent.broker import drain_eval_requests
from trusted_eval_infra.agent.scheduling import install_pending_guards


def write_attempt(root, commit="a" * 40, agent="agent-1", status="pending", score=None):
    path = root / "public" / "attempts" / f"{commit}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(commit_hash=commit, agent_id=agent, status=status, score=score,
                timestamp=commit, feedback="", title="candidate")
    path.write_text(json.dumps(data))
    return data


def manager(root):
    class Manager:
        def _build_score_prompt(self, attempt, count):
            return str(attempt["score"]) if attempt["score"] is not None else "FAILED"

        def _restart_agent(self, idx, prompt=None, prompt_source=None):
            self.calls.append(("restart", idx, prompt))
            return "restarted"

        def _interrupt_and_resume(self, idx, prompt, prompt_source=None):
            self.calls.append(("interrupt", idx, prompt_source))
            return "interrupted"

        def _get_eval_count(self):
            return 1

        def _get_heartbeat_runner(self, agent_id):
            return SimpleNamespace(check=lambda **_: [SimpleNamespace(name="reflect", prompt="Reflect on the scored result.")])

        def monitor_loop(self, check_interval=5):
            return self._get_seen_attempts()

        def _get_seen_attempts(self):
            return {p.name for p in (root / "public/attempts").glob("*.json")}

        def _filter_scored(self, names):
            return {name for name in names if json.loads((root / "public/attempts" / name).read_text())["status"] != "pending"}

    install_pending_guards(Manager)
    install_pending_guards(Manager)  # CLI initialization must be idempotent.
    value = Manager()
    value.paths = SimpleNamespace(coral_dir=root)
    value.handles = [SimpleNamespace(agent_id="agent-1", alive=False), SimpleNamespace(agent_id="agent-2", alive=False)]
    value.calls = []
    value._agent_eval_counts = {}
    value._agent_best_scores = {}
    value._agent_evals_since_improvement = {}
    value.config = SimpleNamespace(grader=SimpleNamespace(direction="maximize"))
    return value


def test_initial_pending_is_rechecked_and_history_counters_restored(tmp_path):
    value = manager(tmp_path)
    write_attempt(tmp_path)
    write_attempt(tmp_path, commit="b" * 40, agent="agent-2", status="improved", score=0.25)
    assert value.monitor_loop() == {"b" * 40 + ".json"}
    assert value._agent_eval_counts == {"agent-1": 0, "agent-2": 1}
    assert value._agent_best_scores == {"agent-2": 0.25}
    write_attempt(tmp_path, status="improved", score=0.5)
    assert value._filter_scored({"a" * 40 + ".json"}) == {"a" * 40 + ".json"}


def test_simultaneous_scores_are_not_silently_coalesced(tmp_path):
    value = manager(tmp_path)
    for commit in ("a", "b", "c"):
        write_attempt(tmp_path, commit=commit * 40, status="improved", score=1.0)
    remaining = value._get_seen_attempts()
    delivered = set()
    for _ in range(3):
        finalized = value._filter_scored(remaining)
        assert len(finalized) == 1
        assert not finalized & delivered
        remaining -= finalized
        delivered |= finalized
    assert len(delivered) == 3 and not remaining


def test_dead_grader_does_not_leave_run_silently_waiting(tmp_path):
    value = manager(tmp_path)
    value._grader_proc = SimpleNamespace(is_alive=lambda: False)
    with pytest.raises(RuntimeError, match="grading daemon exited"):
        value._get_seen_attempts()


def test_heartbeat_for_exited_agent_is_delivered_on_restart(tmp_path):
    value = manager(tmp_path)
    write_attempt(tmp_path, status="improved", score=0.5)
    assert value._get_heartbeat_runner("agent-1").check(local_eval_count=3) == []
    assert value._restart_agent(0) == "restarted"
    assert "Reflect on the scored result." in value.calls[0][2]
    assert not value._frontieror_deferred_heartbeats


def test_stage2_infeasibility_feedback_does_not_expose_reference_data():
    from trusted_eval_infra.agent.grader import public_stage2_feedback
    text = public_stage2_feedback({"stage2_total": 1, "stage2_feasible_count": 0,
                                   "private_reference": "SECRET"})
    assert "infeasible" in text and "SECRET" not in text
    assert public_stage2_feedback({"stage2_total": 1, "stage2_feasible_count": 1}) == ""


def test_pending_is_not_failed_and_does_not_restart(tmp_path):
    value = manager(tmp_path)
    pending = write_attempt(tmp_path)
    prompt = value._build_score_prompt(pending, 0)
    assert "PENDING" in prompt and "FAILED" not in prompt
    for _ in range(20):
        assert value._restart_agent(0, prompt="FAILED") is value.handles[0]
        assert value._interrupt_and_resume(0, "stalled", prompt_source="timeout") is value.handles[0]
    assert not value.calls
    write_attempt(tmp_path, status="improved", score=0.5)
    assert value._restart_agent(0, prompt="FAILED") == "restarted"
    assert value.calls == [("restart", 0, "0.5")]


def test_restart_uses_own_result_and_preserves_heartbeat(tmp_path):
    value = manager(tmp_path)
    write_attempt(tmp_path, status="improved", score=0.25)
    write_attempt(tmp_path, commit="b" * 40, agent="agent-2")
    assert value._restart_agent(0, prompt="other agent FAILED") == "restarted"
    assert value.calls[-1] == ("restart", 0, "0.25")
    assert value._restart_agent(1) is value.handles[1]
    for source in ("heartbeat:reflect", "heartbeat:pivot", "heartbeat:consolidate"):
        assert value._interrupt_and_resume(0, "feedback", prompt_source=source) == "interrupted"


@pytest.mark.parametrize("status,score", [("crashed", None), ("timeout", None), ("baseline", 0.0)])
def test_terminal_failures_are_still_delivered(tmp_path, status, score):
    value = manager(tmp_path)
    write_attempt(tmp_path, status=status, score=score)
    assert value._restart_agent(0) == "restarted"
    assert value.calls[-1][-1] == (str(score) if score is not None else "FAILED")


def test_submit_reuses_pending_without_committing_or_enqueuing(tmp_path, monkeypatch, capsys):
    write_attempt(tmp_path)
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("FRONTIER_OR_EVAL_REQUEST_DIR", str(inbox))
    monkeypatch.setenv("FRONTIER_OR_ATTEMPTS_DIR", str(tmp_path / "public/attempts"))
    monkeypatch.setenv("FRONTIER_OR_AGENT_ID", "agent-1")
    monkeypatch.setenv("FRONTIER_OR_EVAL_WAIT_SECONDS", "0")
    monkeypatch.setattr(submit, "_commit", lambda *_: pytest.fail("must not commit while pending"))
    assert submit._submit("another", wait=True) == 0
    assert not inbox.exists()
    text = capsys.readouterr().out
    assert '"status": "pending"' in text and "no new candidate" in text


def test_poll_timeout_is_pending_not_evaluation_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRONTIER_OR_EVAL_WAIT_SECONDS", "0")
    assert submit._wait_for_result(tmp_path, "a" * 40, wait=True) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pending" and result["score"] is None


def test_unchanged_scored_candidate_does_not_enqueue_again(tmp_path, monkeypatch, capsys):
    write_attempt(tmp_path, status="improved", score=0.5)
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("FRONTIER_OR_EVAL_REQUEST_DIR", str(inbox))
    monkeypatch.setenv("FRONTIER_OR_ATTEMPTS_DIR", str(tmp_path / "public/attempts"))
    monkeypatch.setenv("FRONTIER_OR_AGENT_ID", "agent-1")
    monkeypatch.setattr(submit, "_commit", lambda *_: "a" * 40)
    assert submit._submit("same candidate", wait=True) == 0
    assert not inbox.exists()
    assert json.loads(capsys.readouterr().out)["score"] == 0.5


def test_admission_response_waits_on_existing_attempt(tmp_path, monkeypatch, capsys):
    responses = tmp_path / "responses"
    responses.mkdir()
    (responses / ("c" * 32 + ".json")).write_text(json.dumps({
        "accepted": False, "waiting_for_commit": "a" * 40,
        "reason": "agent already has a pending evaluation",
    }))
    monkeypatch.setenv("FRONTIER_OR_EVAL_RESPONSE_DIR", str(responses))
    monkeypatch.setenv("FRONTIER_OR_EVAL_WAIT_SECONDS", "0")
    assert submit._wait_for_result(tmp_path, "b" * 40, wait=True, nonce="c" * 32) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["commit_hash"] == "a" * 40 and result["status"] == "pending"


def test_broker_enforces_one_pending_per_agent_without_using_budget(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (repo / "code.py").write_text("pass\n")
    git("add", ".")
    git("commit", "-qm", "seed")
    git("checkout", "-qb", "coral/agent-1")
    commits = []
    for index in range(3):
        (repo / "code.py").write_text(f"print({index})\n")
        git("commit", "-qam", "candidate")
        commits.append(git("rev-parse", "HEAD"))
    root = tmp_path / ".coral"
    inbox = root / "private/eval_requests/inbox"
    inbox.mkdir(parents=True)
    def request(index, nonce):
        (inbox / f"{nonce}.json").write_text(json.dumps({
            "schema_version": 1, "nonce": nonce, "agent_id": "agent-1",
            "commit_hash": commits[index], "message": "candidate",
        }))
    task = SimpleNamespace(coral_dir=root, repo_dir=repo, agent_ids=("agent-1",))
    request(0, "a" * 32)
    request(1, "b" * 32)
    decisions = drain_eval_requests(task, max_attempts=2)
    assert [d["accepted"] for d in decisions] == [True, False]
    assert len(list((root / "public/attempts").glob("*.json"))) == 1
    response = json.loads((root / "public/request_status" / ("b" * 32 + ".json")).read_text())
    assert response["waiting_for_commit"] == commits[0]
    write_attempt(root, commit=commits[0], status="improved", score=0.1)
    request(1, "c" * 32)
    assert drain_eval_requests(task, max_attempts=2)[0]["accepted"]
    request(2, "d" * 32)
    assert drain_eval_requests(task, max_attempts=2)[0]["reason"] == "evaluation budget exhausted"
    assert len(list((root / "public/attempts").glob("*.json"))) == 2


def test_timeout_covers_all_instances_and_queue_without_cooldown(monkeypatch):
    from test_time_self_evolution.coral.runner import coral_evaluation_timeout
    monkeypatch.setenv("FRONTIER_OR_WLS_CONCURRENCY", "2")
    monkeypatch.setenv("FRONTIER_OR_WLS_QUEUE_TIMEOUT_SECONDS", "7200")
    monkeypatch.delenv("FRONTIER_OR_WLS_RELEASE_DELAY_SECONDS", raising=False)
    result = coral_evaluation_timeout(["tiny"], ["large_1", "large_2"], 300, 3600, {})
    assert result == 300 + 7200 + 3 * (7200 + 180) + 120
    assert coral_evaluation_timeout(["tiny"], ["large_1"], 300, 3600, {"wls_egress": "off"}) == 4380


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_license_queue_timeouts_fail_fast(monkeypatch, value):
    from trusted_eval_infra.execution import wls_queue_timeout_seconds
    monkeypatch.setenv("FRONTIER_OR_WLS_QUEUE_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="finite and positive"):
        wls_queue_timeout_seconds({})
