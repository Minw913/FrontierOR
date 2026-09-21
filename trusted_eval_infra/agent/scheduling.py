"""Pending-aware lifecycle guards for the pinned upstream CORAL manager.

Install in the host CLI adapter, not by modifying the external CORAL checkout.
The grader and native heartbeat counters still own terminal-result delivery.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

from trusted_eval_infra.files import read_regular_file

logger = logging.getLogger(__name__)


def agent_attempts(coral_dir: Path, agent_id: str) -> list[dict]:
    rows = []
    for path in (coral_dir / "public" / "attempts").glob("*.json"):
        # The directory is host-owned; agents can only read it.
        value = json.loads(read_regular_file(
            path, max_bytes=1024 * 1024, label="public evaluation attempt"
        ))
        if value.get("agent_id") == agent_id:
            rows.append(value)
    return sorted(rows, key=lambda value: str(value.get("timestamp", "")))


def pending_prompt(attempt: dict) -> str:
    commit = str(attempt.get("commit_hash", "unknown"))[:12]
    return (
        f"Evaluation PENDING (commit {commit}). No score is available yet. "
        "This is not a failed evaluation. Wait for the trusted grader; "
        "do not submit another candidate or infer solution quality from queue time."
    )


def install_pending_guards(manager_class) -> None:
    """Patch only restart/timeout edges, preserving the upstream monitor loop."""
    if manager_class.__dict__.get("_frontieror_pending_guards", False):
        return
    original_prompt = manager_class._build_score_prompt
    original_restart = manager_class._restart_agent
    original_interrupt = manager_class._interrupt_and_resume
    original_monitor = manager_class.monitor_loop
    original_seen = manager_class._get_seen_attempts
    original_filter = manager_class._filter_scored
    original_runner = manager_class._get_heartbeat_runner

    def rows(manager, idx):
        if manager.paths is None:
            return []
        return agent_attempts(Path(manager.paths.coral_dir), manager.handles[idx].agent_id)

    def waiting(manager, idx, attempts):
        pending = [a for a in attempts if a.get("status") == "pending"]
        if not pending:
            return False
        agent_id = manager.handles[idx].agent_id
        noted = getattr(manager, "_frontieror_waiting", {})
        commit = pending[0]["commit_hash"]
        if noted.get(agent_id) != commit:
            logger.info("%s waiting for evaluation %s; model restart deferred", agent_id, commit[:12])
            noted[agent_id] = commit
            manager._frontieror_waiting = noted
        return True

    @functools.wraps(original_prompt)
    def build_score_prompt(self, attempt, eval_count):
        if attempt.get("status") == "pending":
            return pending_prompt(attempt)
        return original_prompt(self, attempt, eval_count)

    @functools.wraps(original_restart)
    def restart(self, idx, prompt=None, prompt_source=None):
        attempts = rows(self, idx)
        if waiting(self, idx, attempts):
            # Do not consume another model invocation while its score is absent.
            # The upstream loop rechecks this dead handle after grading advances.
            return self.handles[idx]
        getattr(self, "_frontieror_waiting", {}).pop(self.handles[idx].agent_id, None)
        # Upstream's latest attempt is global and can belong to another agent.
        # Resume with only this agent's latest terminal result.
        prompt = self._build_score_prompt(attempts[-1], self._get_eval_count()) if attempts else None
        deferred = getattr(self, "_frontieror_deferred_heartbeats", {}).get(self.handles[idx].agent_id, [])
        if deferred:
            prompt = "\n\n".join([prompt or "Evaluation completed.", *[text for _, text in deferred]])
            prompt_source = "heartbeat:" + ", ".join(dict.fromkeys(name for name, _ in deferred))
        handle = original_restart(self, idx, prompt=prompt, prompt_source=prompt_source)
        getattr(self, "_frontieror_deferred_heartbeats", {}).pop(self.handles[idx].agent_id, None)
        return handle

    @functools.wraps(original_interrupt)
    def interrupt(self, idx, prompt, prompt_source=None):
        if prompt_source == "timeout" and waiting(self, idx, rows(self, idx)):
            # A quiet `coral eval` poll is not a stalled model invocation.
            return self.handles[idx]
        return original_interrupt(self, idx, prompt, prompt_source=prompt_source)

    @functools.wraps(original_seen)
    def seen_attempts(self):
        grader = getattr(self, "_grader_proc", None)
        if grader is not None and not grader.is_alive():
            raise RuntimeError("Trusted grading daemon exited; stop the run instead of waiting or generating more candidates")
        names = original_seen(self)
        if getattr(self, "_frontieror_initial_scan", False):
            self._frontieror_initial_scan = False
            # Pending attempts present before monitor startup must stay on the
            # recheck list. Restore counters for already completed history.
            completed = original_filter(self, names)
            for handle in self.handles:
                history = [a for a in agent_attempts(Path(self.paths.coral_dir), handle.agent_id)
                           if f"{a['commit_hash']}.json" in completed]
                self._agent_eval_counts[handle.agent_id] = len(history)
                best, plateau = None, 0
                minimize = self.config.grader.direction == "minimize"
                for attempt in history:
                    score = attempt.get("score")
                    if score is not None and (best is None or (score < best if minimize else score > best)):
                        best, plateau = score, 0
                    else:
                        plateau += 1
                if best is not None:
                    self._agent_best_scores[handle.agent_id] = best
                self._agent_evals_since_improvement[handle.agent_id] = plateau
            return completed
        return names

    @functools.wraps(original_filter)
    def filter_scored(self, names):
        completed = original_filter(self, names)
        if not completed:
            return set()
        # Upstream processes only one result but marks the whole set as seen.
        # Drain one at a time so simultaneous completions cannot lose feedback
        # or skip per-agent heartbeat/plateau counter increments.
        directory = Path(self.paths.coral_dir) / "public" / "attempts"
        return {min(completed, key=lambda name: (directory.joinpath(name).stat().st_mtime_ns, name))}

    @functools.wraps(original_monitor)
    def monitor(self, check_interval=5):
        self._frontieror_initial_scan = True
        return original_monitor(self, check_interval=check_interval)

    @functools.wraps(original_runner)
    def heartbeat_runner(self, agent_id):
        runner = original_runner(self, agent_id)
        manager = self

        class PendingAwareHeartbeat:
            def check(self, **kwargs):
                actions = runner.check(**kwargs)
                if actions and not any(h.agent_id == agent_id and h.alive for h in manager.handles):
                    # Upstream drops actions when the receiving agent is dead.
                    # Attach them to its next restart instead of starting a
                    # model only to interrupt it immediately for reflection.
                    deferred = getattr(manager, "_frontieror_deferred_heartbeats", {})
                    deferred.setdefault(agent_id, []).extend((a.name, a.prompt) for a in actions if a.prompt)
                    manager._frontieror_deferred_heartbeats = deferred
                    return []
                return actions

        return PendingAwareHeartbeat()

    manager_class._build_score_prompt = build_score_prompt
    manager_class._restart_agent = restart
    manager_class._interrupt_and_resume = interrupt
    manager_class._get_seen_attempts = seen_attempts
    manager_class._filter_scored = filter_scored
    manager_class.monitor_loop = monitor
    manager_class._get_heartbeat_runner = heartbeat_runner
    manager_class._frontieror_pending_guards = True
