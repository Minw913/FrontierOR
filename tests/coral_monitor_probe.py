"""Exercise the real pinned CORAL monitor without calling a model or solver.

Run from an installed checkout: PYTHONPATH=external/coral:. python tests/coral_monitor_probe.py
"""
import json
import signal
import tempfile
from pathlib import Path
from types import SimpleNamespace

from coral.agent.manager import AgentManager
from trusted_eval_infra.agent.scheduling import install_pending_guards


def main():
    class ProbeManager(AgentManager):
        def _get_heartbeat_runner(self, agent_id):
            def check(**kwargs):
                return [SimpleNamespace(name="reflect", prompt="Reflect on the three results.")] if kwargs["local_eval_count"] == 3 else []
            return SimpleNamespace(check=check)
    install_pending_guards(ProbeManager)
    with tempfile.TemporaryDirectory(prefix="coral-monitor-probe-") as tmp:
        root = Path(tmp)
        attempts = root / "public/attempts"
        attempts.mkdir(parents=True)
        log = root / "agent.log"
        log.write_text("")
        pending = dict(commit_hash="a" * 40, agent_id="agent-1", status="pending",
                       score=None, feedback="", title="first", timestamp="1")
        path = attempts / ("a" * 40 + ".json")
        path.write_text(json.dumps(pending))
        instance = ProbeManager.__new__(ProbeManager)
        instance.paths = SimpleNamespace(coral_dir=root)
        instance.config = SimpleNamespace(grader=SimpleNamespace(direction="maximize"),
                                          agents=SimpleNamespace(timeout=1))
        instance._restart_counts = {}
        instance._agent_eval_counts = {}
        instance._agent_best_scores = {}
        instance._agent_evals_since_improvement = {}
        instance._running = True
        instance._stopping = False
        instance.verbose = False
        instance.runtime = SimpleNamespace(extract_session_id=lambda _: "saved-session")
        dead = SimpleNamespace(agent_id="agent-1", process=None, log_path=log,
                               alive=False, stop=lambda: None)
        instance.handles = [dead]
        starts = []
        def start(agent_id, **kwargs):
            starts.append((agent_id, kwargs))
            return SimpleNamespace(agent_id=agent_id, process=None, log_path=log, alive=True,
                                   stop=lambda: None, interrupt=lambda: "saved-session")
        instance._setup_and_start_agent = start
        instance._write_agent_pids = lambda: None
        class Ticks:
            count = 0
            def wait(self, timeout):
                self.count += 1
                if self.count <= 3:
                    assert not starts, "model restarted before any score"
                if self.count == 3:
                    path.write_text(json.dumps(dict(pending, status="improved", score=0.5)))
                if self.count == 4:
                    assert len(starts) == 1
                    assert "0.5000000000" in starts[0][1]["prompt"]
                    assert starts[0][1]["resume_session_id"] == "saved-session"
                    assert instance._agent_eval_counts["agent-1"] == 1
                    # Complete two other attempts between monitor ticks. Both
                    # must be counted, although upstream handles one per tick.
                    for letter in ("b", "c"):
                        data = dict(pending, commit_hash=letter * 40, timestamp=letter,
                                    status="baseline", score=0.4)
                        (attempts / (letter * 40 + ".json")).write_text(json.dumps(data))
                    instance.handles[0].alive = False
                if self.count >= 5 and instance._agent_eval_counts["agent-1"] == 3:
                    assert len(starts) == 2
                    assert starts[-1][1]["prompt_source"] == "heartbeat:reflect"
                    assert "Reflect on the three results." in starts[-1][1]["prompt"]
                    return True
                assert self.count < 8, "monitor failed to deliver all completed results"
                return False
        instance._stop_event = Ticks()
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            instance.monitor_loop(check_interval=0)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        print(json.dumps({"pending_restart_count": 0, "post_score_restart_count": 1,
                          "deferred_heartbeat_resume_count": 1, "resumed_session": "saved-session",
                          "counted_scores": 3, "success": True}))


if __name__ == "__main__":
    main()
