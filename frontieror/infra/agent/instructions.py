"""Minimal instructions for an untrusted benchmark agent."""

from __future__ import annotations


def generate_secure_coral_md(config, agent_id: str, **kwargs) -> str:
    single_agent = bool(kwargs.get("single_agent", True))
    collaboration = ""
    if not single_agent:
        collaboration = """
## Collaboration

- You are one member of a submitted multi-agent system. Other registered
  agents are collaborators, not separate benchmark submissions.
- Exchange useful findings through `.codex/notes/`. Keep evaluation attempts
  read-only and do not modify another agent's branch or worktree.
- Evaluation, resource, and wall-clock limits apply to the complete submitted
  system. Codex-spawned subagents remain disabled; CORAL controls the registered
  top-level agents.
- Only broker-scored submissions are eligible as the final artifact. Submit
  useful candidates from your own branch so both agents can contribute.
"""
    return f"""# FrontierOR Agent Task

You are `{agent_id}`. Improve `code.py` for the optimization task in `README.md`.

## Boundary

- Use only files visible in this workspace and feedback returned by `coral eval`.
- Network access, web research, host paths, hidden tests, grader code, reference
  solutions, and other submissions are outside the task boundary.
- Do not probe mounts, credentials, processes, environment internals, or the
  scoring service. A high score is valid only when produced by `code.py`.
- Treat attempt JSON and logs as read-only outputs. Never create or edit them.
- The Agent shell is intentionally smaller than the candidate evaluator and may
  omit candidate-only dependencies such as `gurobipy` and its license. A local
  import failure does not describe the evaluator environment; use `coral eval`
  for authoritative execution and scoring.
{collaboration}

## Workflow

1. Read `README.md`, the public schemas/instances, and the current `code.py`.
2. Make a focused solver improvement.
3. Submit it with `coral eval -m "short description"` and use only the returned
   feedback to decide the next change.
4. Submit a viable candidate early, then iterate within CORAL's native attempt
   and wall-clock budget.
5. Keep `code.py` runnable under the documented CLI contract.
"""
