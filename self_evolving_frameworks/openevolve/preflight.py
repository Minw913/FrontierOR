"""Pre-flight environment checks for self_evolve runs.

Catches statically-detectable env errors before any LLM call so a 50-iter
run doesn't silently waste compute on score=0. Used by:
  - ``runner.run_self_evolve`` — full pre-flight at startup
  - ``evaluator`` — mid-run Gurobi license re-probe
    (``verify_gurobi_license_now``) for strategy C abort logic

Skip flags (env vars):
  - ``EFFICIENT_OR_SKIP_NETWORK_PREFLIGHT=1`` — skip OpenRouter HTTP probe
  - ``EFFICIENT_OR_SKIP_GUROBI_PREFLIGHT=1``  — skip Gurobi license probe
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import one_shot_eval as eval_core


def _check_paper_files(paper_dir: str, all_instances: List[str]) -> List[str]:
    """Prompt files + checker + per-instance JSON + per-instance Gurobi baseline.

    Covers env-error fail_reasons that ``one_shot_eval._tiny_gate_correctable``
    treats as unrecoverable (``missing_instance``, ``checker_unavailable``)
    plus the silent-failure cases where Stage1/Stage2 score=0 because no
    Gurobi baseline exists to compute ``gap`` against.
    """
    from scripts.utils.instance_paths import (
        gurobi_solution_path as _gurobi_solution_path,
        instance_path as _instance_path,
    )

    issues: List[str] = []
    for fname in ("problem_description.txt", "instance_schema.json", "solution_schema.json"):
        p = os.path.join(paper_dir, fname)
        if not os.path.exists(p):
            issues.append(f"prompt   : missing {p}")
    checker_path = os.path.join(paper_dir, "feasibility_check.py")
    if not os.path.exists(checker_path):
        issues.append(f"checker  : missing {checker_path}")
    seen = set()
    for inst in all_instances:
        if not inst or inst in seen:
            continue
        seen.add(inst)
        ip = _instance_path(paper_dir, inst)
        if not os.path.exists(ip):
            issues.append(f"instance : missing {ip}")
        bp = _gurobi_solution_path(paper_dir, inst)
        if not os.path.exists(bp):
            issues.append(f"baseline : missing {bp}")
            continue
        try:
            with open(bp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            issues.append(f"baseline : {bp} unreadable ({type(e).__name__}: {str(e)[:80]})")
            continue
        if data.get("objective_value") is None:
            issues.append(f"baseline : {bp} has no 'objective_value' field")
    return issues


def _check_gurobi_time_required(
    paper_id: str,
    stage2_instances: List[str],
    test_instances: List[str],
    stage2_time_policy: str,
    test_time_policy: str,
    stage2_scorer: str,
) -> List[str]:
    """Verify τ_g is available for every instance that will need it.

    staged_qte scorer's Stage 2 speed bonus needs τ_g (otherwise speed
    contribution falls back to 0). aocc backup doesn't need τ_g. The
    gurobi_time / gurobi_time_plus_buffer time policies additionally need
    it for sizing per-instance budgets.
    """
    from self_evolving_frameworks.scoring.building_blocks import lookup_gurobi_time

    issues: List[str] = []
    needs: Dict[str, List[str]] = {}  # inst -> list of reason labels
    if stage2_scorer == "staged_qte":
        for inst in stage2_instances:
            needs.setdefault(inst, []).append("scorer=staged_qte (Stage 2 speed bonus)")
    if stage2_time_policy in ("gurobi_time", "gurobi_time_plus_buffer"):
        for inst in stage2_instances:
            needs.setdefault(inst, []).append(f"--stage2-time-policy={stage2_time_policy}")
    if test_time_policy in ("gurobi_time", "gurobi_time_plus_buffer"):
        for inst in test_instances:
            needs.setdefault(inst, []).append(f"--test-time-policy={test_time_policy}")
    for inst, reasons in needs.items():
        if lookup_gurobi_time(paper_id, inst) is None:
            issues.append(
                f"τ_g      : '{inst}' has no Gurobi time "
                f"(checked gurobi_solving_results.csv + gurobi_solution_log/) "
                f"— required by " + ", ".join(sorted(set(reasons)))
            )
    return issues


def _check_openrouter_key(config: Dict, primary_model: str) -> List[str]:
    """Verify an API key is configured and (best-effort) authorizes against
    OpenRouter's /api/v1/key endpoint. Set ``EFFICIENT_OR_SKIP_NETWORK_PREFLIGHT=1``
    to skip the HTTP probe (offline mode). Network/transient errors warn-only;
    only HTTP 401/403 (definitive auth rejection) blocks the run.
    """
    issues: List[str] = []
    key = config.get("OPENROUTER_API_KEY") if isinstance(config, dict) else None
    if not key:
        issues.append(f"openrouter_key: not configured for model={primary_model!r}")
        return issues
    if os.environ.get("EFFICIENT_OR_SKIP_NETWORK_PREFLIGHT") == "1":
        return issues
    try:
        import http.client
        conn = http.client.HTTPSConnection("openrouter.ai", timeout=10)
        conn.request("GET", "/api/v1/key", headers={"Authorization": f"Bearer {key}"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status in (401, 403):
            issues.append(
                f"openrouter_key: HTTP {resp.status} {resp.reason} — key invalid or expired"
            )
        elif resp.status != 200:
            print(f"[preflight] WARNING: OpenRouter probe returned HTTP {resp.status} "
                  f"{resp.reason}; continuing (likely transient).")
    except Exception as e:
        print(f"[preflight] WARNING: OpenRouter probe failed "
              f"({type(e).__name__}: {str(e)[:80]}); continuing (network may be down). "
              f"Set EFFICIENT_OR_SKIP_NETWORK_PREFLIGHT=1 to silence.")
    return issues


def _check_gurobi_license() -> List[str]:
    """Spawn a probe subprocess that constructs an empty Gurobi model.

    Catches expired licenses, missing GRB_LICENSE_FILE, and unreachable
    Compute-Server / token-server setups before LLM iter 0. Set
    ``EFFICIENT_OR_SKIP_GUROBI_PREFLIGHT=1`` to skip (e.g. running a
    LLM-only experiment).
    """
    if os.environ.get("EFFICIENT_OR_SKIP_GUROBI_PREFLIGHT") == "1":
        return []
    probe = (
        "import sys\n"
        "try:\n"
        "    import gurobipy as gp\n"
        "    m = gp.Model('preflight')\n"
        "    m.setParam('OutputFlag', 0)\n"
        "    m.dispose()\n"
        "except Exception as e:\n"
        "    print(f'GUROBI_ERROR:{type(e).__name__}:{e}', file=sys.stderr)\n"
        "    sys.exit(2)\n"
    )
    issues: List[str] = []
    try:
        r = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            issues.append("gurobi_license: probe failed:\n      " + "\n      ".join(tail))
    except subprocess.TimeoutExpired:
        issues.append("gurobi_license: probe timed out (20s) — license server unreachable?")
    except Exception as e:
        issues.append(
            f"gurobi_license: probe error ({type(e).__name__}: {str(e)[:80]})"
        )
    return issues


def verify_gurobi_license_now() -> Optional[str]:
    """Public mid-run re-probe of the Gurobi license.

    Returns ``None`` if the license is currently usable; returns a single
    short error string if not. Used by ``evaluator`` to
    distinguish transient license-shaped stderr (LLM-side misuse, hiccup)
    from a genuinely-broken license that warrants aborting the run.
    """
    issues = _check_gurobi_license()
    return issues[0] if issues else None


def preflight_environment_check(
    paper_id: str,
    stage1_instances: List[str],
    stage2_instances: List[str],
    test_instances: List[str],
    config: Dict,
    primary_model: str,
    stage2_time_policy: str,
    test_time_policy: str,
    stage2_scorer: str,
) -> None:
    """All-or-nothing environment validation for self_evolve.

    Aborts with a single ``RuntimeError`` listing every issue at once
    (not first-failure-wins) so the user can fix everything in one pass.
    """
    paper_dir = eval_core.get_paper_dir(paper_id)
    all_instances = list(stage1_instances) + list(stage2_instances) + list(test_instances)

    issues: List[str] = []
    issues += _check_paper_files(paper_dir, all_instances)
    issues += _check_gurobi_time_required(
        paper_id, stage2_instances, test_instances,
        stage2_time_policy, test_time_policy, stage2_scorer,
    )
    issues += _check_openrouter_key(config, primary_model)
    issues += _check_gurobi_license()

    if issues:
        raise RuntimeError(
            f"ENV_ERROR: cannot start self_evolve for paper '{paper_id}'. "
            f"{len(issues)} issue(s) found:\n  - " + "\n  - ".join(issues)
        )
