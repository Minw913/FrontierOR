"""OpenEvolve evaluator adapter for frontier-or programs.

Exposes three entry points:
  - ``evaluate_stage1`` — binary pass/fail on quick (``tiny``) instances.
      Score = 1.0 iff every stage1 instance is feasible and |gap| <= threshold.
  - ``evaluate_stage2`` — main fitness signal on ``large_dev`` instances.
      Score = mean(1 - AOCC) across instances; infeasible contributes 0.
  - ``evaluate`` — direct (non-cascade) fallback using the legacy env vars.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys


ROOT_DIR = os.environ.get(
    "EFFICIENT_OR_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import one_shot_eval as eval_core  # noqa: E402
from test_time_self_evolution import eval_modes  # noqa: E402

# EvaluationResult lets us return artifacts alongside metrics. Imported
# lazily so the evaluator still works if openevolve is not installed
# (e.g. local debugging via ``python evaluator.py``).
try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
    _HAS_EVALUATION_RESULT = True
except ImportError:
    EvaluationResult = None  # type: ignore
    _HAS_EVALUATION_RESULT = False


# --- Artifact constants. Tweak here, not at the call sites. -----------------
TINY_TIME_LIMIT_S = 300
STAGE1_GAP_PCT = 10
STDERR_TAIL_CHARS = 1500
VIOLATION_PER_LINE_CHARS = 200
MAX_DISTINCT_CONSTRAINTS = 8

_VIOLATION_PREFIX_RE = re.compile(r"^Constraint \((\d+)\):")


# --- Artifact builders ------------------------------------------------------

def _classify_failure(result):
    """Map a per-instance result dict to one of the 6 artifact categories.

    Returns one of:
      "pass" | "infeasible" | "gap_exceeds" | "runtime_crash" |
      "runtime_timeout" | "malformed"

    ``checker_unavailable`` and ``missing_instance`` are env errors that
    pre-flight should have caught before iter 0; if they reach here we
    still return ``"malformed"`` so the LLM gets *some* signal rather
    than a silent score=0.
    """
    if (result or {}).get("status") == "pass":
        return "pass"
    fr = (result or {}).get("fail_reason")
    if fr == "infeasible":
        return "infeasible"
    if fr == "gap_exceeds":
        return "gap_exceeds"
    if fr in ("invalid_solution", "checker_error"):
        return "malformed"
    if fr == "runtime_error":
        err = (result or {}).get("error") or ""
        if err.startswith("Execution timed out"):
            return "runtime_timeout"
        return "runtime_crash"
    # Unknown / env-level — fall back to malformed.
    return "malformed"


def _load_feasi_result(output_dir, inst):
    """Read the per-paper feasibility checker JSON for an instance.
    Returns ``{}`` on any error so callers can downgrade gracefully."""
    path = os.path.join(output_dir, f"feasi_result_{inst}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _dedupe_violations(feasi_data):
    """Group violations by constraint id (parsed from the ``Constraint (N):``
    prefix) and pick one representative per group, preserving the per-group
    count. Returns ``(top_groups, n_unique_constraints, n_violations_total)``.

    ``top_groups`` is a list of ``(cid, count, first_text)`` ordered by
    count descending and capped at MAX_DISTINCT_CONSTRAINTS.
    Violations without the standard prefix fall back to the whole text as
    the dedup key (so we don't crash, just don't dedup that branch).
    """
    violations = (feasi_data or {}).get("violations") or []
    if not violations:
        return [], 0, 0
    groups = {}
    for v in violations:
        m = _VIOLATION_PREFIX_RE.match(v)
        cid = int(m.group(1)) if m else v[:VIOLATION_PER_LINE_CHARS]
        text = v[:VIOLATION_PER_LINE_CHARS]
        if cid not in groups:
            groups[cid] = [1, text]
        else:
            groups[cid][0] += 1
    sorted_groups = sorted(groups.items(), key=lambda kv: -kv[1][0])
    bounded = [(cid, c, t) for cid, (c, t) in sorted_groups[:MAX_DISTINCT_CONSTRAINTS]]
    return bounded, len(groups), len(violations)


def _build_failure_breakdown(kind, result, output_dir, inst, time_limit, paper_id):
    """Build the ``failure_breakdown`` text for one (kind, instance) pair.

    Used by both stage 1 (single instance, kind from `_classify_failure`)
    and stage 2 (per-failed-instance block, indented by caller).
    """
    if kind == "infeasible":
        feasi = _load_feasi_result(output_dir, inst)
        groups, n_g, n_v = _dedupe_violations(feasi)
        if not groups:
            return "(checker reported infeasible but no detailed violations available)"
        header = (
            f"Constraint violations reported by the feasibility checker "
            f"(showing one example per distinct constraint, "
            f"{n_g} constraints, {n_v} total violations):"
        )
        lines = [header]
        for cid, count, text in groups:
            cid_label = f"Constraint ({cid})" if isinstance(cid, int) else "raw"
            lines.append(f"  - [{cid_label}, {count} total]  {text}")
        if n_g > MAX_DISTINCT_CONSTRAINTS:
            lines.append(
                f"  ... and {n_g - MAX_DISTINCT_CONSTRAINTS} more constraint "
                f"IDs not shown"
            )
        return "\n".join(lines)

    if kind == "gap_exceeds":
        gap = result.get("gap")
        gap_pct = (gap * 100) if isinstance(gap, (int, float)) else 0.0
        llm_obj = result.get("llm_obj")
        gurobi_obj = result.get("gurobi_obj")
        try:
            direction = eval_core.get_paper_direction(paper_id) if paper_id else "?"
        except Exception:
            direction = "?"
        llm_str = f"{llm_obj:.6g}" if isinstance(llm_obj, (int, float)) else "n/a"
        gur_str = f"{gurobi_obj:.6g}" if isinstance(gurobi_obj, (int, float)) else "n/a"
        return (
            f"Required:  |gap| <= {STAGE1_GAP_PCT}%\n"
            f"Observed:  gap = {gap_pct:.1f}%\n"
            f"  your objective:    {llm_str}\n"
            f"  Gurobi reference:  {gur_str}\n"
            f"  direction:         {direction}"
        )

    if kind == "runtime_crash":
        err = (result.get("error") or "")
        tail = err[-STDERR_TAIL_CHARS:]
        return f"Last {STDERR_TAIL_CHARS} characters of stderr:\n{tail}"

    if kind == "runtime_timeout":
        elapsed = result.get("solve_time")
        try:
            elapsed_s = float(elapsed)
        except (TypeError, ValueError):
            elapsed_s = float(time_limit)
        return f"Time limit: {time_limit}s    Elapsed: {elapsed_s:.1f}s (terminated)"

    if kind == "malformed":
        err = (result.get("error") or "")
        tail = err[-STDERR_TAIL_CHARS:] if err else "(no error message)"
        return f"Loader / checker error:\n{tail}"

    return ""


# Stage 1 summary templates
_STAGE1_SUMMARY = {
    "infeasible":      "Stage 1 (tiny) FAILED: solution infeasible — violates one or more constraints.",
    "gap_exceeds":     "Stage 1 (tiny) FAILED: solution feasible but objective too far from optimum (gap > {gap_pct}%).",
    "runtime_crash":   "Stage 1 (tiny) FAILED: candidate program crashed.",
    "runtime_timeout": "Stage 1 (tiny) FAILED: candidate exceeded the {tiny_time_limit}s time limit on the smallest instance — algorithm scales poorly.",
    "malformed":       "Stage 1 (tiny) FAILED: solution output is malformed (parse error or checker crash). Make sure the JSON strictly matches solution_schema.json.",
    "pass":            "Stage 1 (tiny) PASSED. gap={gap:.3f}, solve_time={solve_time:.1f}s.",
}


def _build_stage1_artifacts(result, output_dir, paper_id, inst="tiny"):
    """Build stage-1 artifact dict. Returns dict with 'summary' and (when
    applicable) 'failure_breakdown'."""
    kind = _classify_failure(result)
    if kind == "pass":
        gap = result.get("gap") or 0.0
        solve_time = result.get("solve_time") or 0.0
        return {
            "summary": _STAGE1_SUMMARY["pass"].format(
                gap=float(gap), solve_time=float(solve_time)
            )
        }
    summary = _STAGE1_SUMMARY[kind].format(
        gap_pct=STAGE1_GAP_PCT,
        tiny_time_limit=TINY_TIME_LIMIT_S,
    )
    breakdown = _build_failure_breakdown(
        kind, result, output_dir, inst, TINY_TIME_LIMIT_S, paper_id,
    )
    artifacts = {"summary": summary}
    if breakdown:
        artifacts["failure_breakdown"] = breakdown
    return artifacts


# Stage 2 score_summary templates
_STAGE2_SUMMARY_STAGED_QTE = (
    "combined_score = {combined_score:.3f} (mean across {n_dev} dev "
    "{instance_word})\n"
    "\n"
    "Two-stage piecewise score (higher is better):\n"
    "  Stage 1 (gap > 0.2):  score = max(0, 1 - signed_gap)                          ∈ [0, 0.8]\n"
    "  Stage 2 (gap ≤ 0.2):  score = (1 - signed_gap) + (τ_g - t_solve) / τ_g        ∈ [0.8, 2+]\n"
    "\n"
    "Components (averaged across dev instances):\n"
    "  mean_signed_gap = {mean_signed_gap:.3f}  "
    "(sign · (obj_llm - obj_gurobi) / |obj_gurobi|;  <0 ⇒ beat Gurobi)\n"
    "  mean_speed      = {mean_speed:.3f}  "
    "((τ_g - t_solve) / τ_g;  >0 ⇒ faster than Gurobi)"
)

_STAGE2_SUMMARY_AOCC = (
    "combined_score = {combined_score:.3f} (mean across {n_dev} dev "
    "{instance_word})\n"
    "\n"
    "Pure 1 − AOCC anytime score (higher is better, range [0, 1]):\n"
    "  score = 1 − (1/T)·∫_0^T gap(t) dt\n"
    "\n"
    "Component:\n"
    "  mean_aocc = {mean_aocc:.3f}  (smaller = faster convergence; 0 = reached optimum at t=0)"
)


def _build_stage2_one_line(kind, result, output_dir, inst, time_limit):
    """One-line summary of a single failed dev instance, for the
    failure_breakdown header of each (i)-numbered block."""
    if kind == "infeasible":
        feasi = _load_feasi_result(output_dir, inst)
        _, n_g, n_v = _dedupe_violations(feasi)
        return (
            f"solution infeasible ({n_g} distinct constraints, "
            f"{n_v} total violations)."
        )
    if kind == "runtime_crash":
        return "candidate program crashed."
    if kind == "runtime_timeout":
        return f"candidate timed out at {time_limit}s without producing any incumbent."
    if kind == "malformed":
        return "solution output is malformed (parse error or checker crash)."
    return f"failure ({kind})."


def _build_stage2_artifacts(scorer_name, results, instances, per_instance_tl,
                            metrics_out, output_dir, paper_id):
    """Build stage-2 artifact dict. Returns dict with 'score_summary' and
    (when at least one instance failed) 'failure_breakdown'.

    Instances are referred to anonymously as ``(1)``, ``(2)``, ... using a
    run-local ordinal — never by their real name, to avoid LLMs hard-coding
    instance-specific branches.
    """
    n_dev = len(instances)
    instance_word = "instance" if n_dev == 1 else "instances"

    if scorer_name == "staged_qte":
        score_summary = _STAGE2_SUMMARY_STAGED_QTE.format(
            combined_score=metrics_out.get("combined_score", 0.0),
            n_dev=n_dev,
            instance_word=instance_word,
            mean_speed=metrics_out.get("stage2_mean_speed_part", 0.0),
            mean_signed_gap=metrics_out.get("stage2_mean_signed_gap", 1.0),
        )
    elif scorer_name == "aocc":
        score_summary = _STAGE2_SUMMARY_AOCC.format(
            combined_score=metrics_out.get("combined_score", 0.0),
            n_dev=n_dev,
            instance_word=instance_word,
            mean_aocc=metrics_out.get("stage2_mean_aocc", 1.0),
        )
    else:
        # Unknown scorer — emit a minimal summary so artifact is non-empty.
        score_summary = (
            f"combined_score = {metrics_out.get('combined_score', 0.0):.3f} "
            f"(mean across {n_dev} dev {instance_word}, scorer={scorer_name})"
        )

    artifacts = {"score_summary": score_summary}

    # Failure breakdown — anonymized per-instance blocks
    failed = []
    for inst in instances:
        r = results.get(inst) or {}
        if r.get("feasible") is True:
            continue
        kind = _classify_failure(r)
        if kind == "pass":
            continue
        failed.append((inst, r, kind))

    if failed:
        n_failed = len(failed)
        lines = [f"{n_failed} of {n_dev} dev instances failed:", ""]
        for ordinal, (inst, r, kind) in enumerate(failed, 1):
            tl = per_instance_tl.get(inst, 0)
            one_line = _build_stage2_one_line(kind, r, output_dir, inst, tl)
            detail = _build_failure_breakdown(
                kind, r, output_dir, inst, tl, paper_id,
            )
            lines.append(f"({ordinal}) {one_line}")
            for d_line in detail.split("\n"):
                lines.append(f"    {d_line}")
            lines.append("")
        artifacts["failure_breakdown"] = "\n".join(lines).rstrip()

    return artifacts


def _maybe_wrap_with_artifacts(metrics, artifacts):
    """Return ``EvaluationResult(metrics, artifacts)`` if openevolve is
    importable AND artifacts is non-empty; otherwise return the bare
    metrics dict (preserves backward compatibility)."""
    if _HAS_EVALUATION_RESULT and artifacts:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return metrics


# Substrings whose presence in a candidate's stderr suggests a Gurobi license
# issue rather than an LLM-code bug. Triggers a license re-probe. Conservative
# — false positives only cost one ~1s probe; false negatives lose the abort
# signal so we err wide.
GUROBI_LICENSE_STDERR_PATTERNS = (
    "Gurobi license",
    "No Gurobi license found",
    "Failed to retrieve a token",
    "Restricted license",
    "License Manager Error",
    "Unable to use Gurobi license",
    "GRB_ERROR_NO_LICENSE",
    "Gurobi error 10009",
    "Token server",
    "no available license tokens",
    "license file does not exist",
)


def _stderr_has_gurobi_license_marker(err_text):
    if not err_text:
        return False
    return any(pat in err_text for pat in GUROBI_LICENSE_STDERR_PATTERNS)


def _abort_if_license_died(results, stage_label):
    """Scan per-instance stderr; if a Gurobi-license-shaped error is present,
    re-probe the license. If the probe still passes, treat as transient (let
    candidate score 0 normally). If the probe now fails, abort the OpenEvolve
    run by raising SystemExit(2) — this bypasses the upstream
    ``except Exception`` in ``openevolve/evaluator.py`` so the run halts
    instead of silently retrying for 50 iterations.
    """
    triggered_inst = None
    triggered_err = ""
    for inst, r in (results or {}).items():
        rd = r or {}
        if rd.get("fail_reason") != "runtime_error":
            continue
        err = rd.get("error") or ""
        if _stderr_has_gurobi_license_marker(err):
            triggered_inst = inst
            triggered_err = err
            break
    if triggered_inst is None:
        return

    from test_time_self_evolution.openevolve.preflight import verify_gurobi_license_now
    probe_msg = verify_gurobi_license_now()
    if probe_msg is None:
        # License re-probe passed → was a transient hiccup or LLM misuse.
        # Don't abort; the candidate already gets score=0 via runtime_error.
        return

    msg = (
        f"\n[ENV_ERROR] Gurobi license died mid-{stage_label} "
        f"(detected on candidate instance '{triggered_inst}').\n"
        f"Stderr tail from triggering candidate:\n"
        f"  ...{(triggered_err.strip())[-400:]}\n\n"
        f"License re-probe (just now): {probe_msg}\n\n"
        f"Aborting OpenEvolve run to avoid wasting compute on a broken "
        f"license. Restart after fixing the Gurobi license / token server.\n"
    )
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(2)


def _split_instances(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _common_env():
    paper_id = os.environ["EFFICIENT_OR_PAPER_ID"]
    model_name = os.environ.get("EFFICIENT_OR_MODEL_NAME", "openevolve")
    exec_mode = os.environ.get("EFFICIENT_OR_EXEC_MODE", "bare")
    base_output = os.environ.get(
        "EFFICIENT_OR_OUTPUT_DIR",
        os.path.join(ROOT_DIR, "eval", "openevolve_tmp"),
    )
    t_max_raw = os.environ.get("EFFICIENT_OR_T_MAX", "")
    if not t_max_raw:
        t_max = None
    elif t_max_raw.strip().lower() == "gurobi":
        t_max = "gurobi"  # resolved per-instance in one_shot_eval._resolve_t_max
    else:
        t_max = float(t_max_raw)
    exec_cfg = {
        "cpus": int(os.environ.get("EFFICIENT_OR_EXEC_CPUS", "1")),
        "memory": os.environ.get("EFFICIENT_OR_EXEC_MEMORY", "32G"),
    }
    return paper_id, model_name, exec_mode, base_output, t_max, exec_cfg


def evaluate_stage1(program_path: str):
    """Binary gate: every stage1 instance must be feasible and within gap threshold."""
    paper_id, model_name, exec_mode, base_output, t_max, exec_cfg = _common_env()
    instances = _split_instances(os.environ.get("EFFICIENT_OR_STAGE1_INSTANCES", "tiny"))
    time_limit = int(os.environ.get("EFFICIENT_OR_STAGE1_TIME_LIMIT", "100"))
    gap_threshold = float(os.environ.get("EFFICIENT_OR_STAGE1_GAP_THRESHOLD", "0.10"))

    if not instances:
        return {"combined_score": 0.0, "stage1_error": "no_stage1_instances"}

    output_dir = os.path.join(base_output, "stage1")
    os.makedirs(output_dir, exist_ok=True)
    results = eval_core.evaluate_candidate_code(
        paper_id, model_name, instances, program_path, output_dir,
        time_limit, exec_mode, exec_cfg, t_max,
    )
    _abort_if_license_died(results, "stage1")

    passed = 0
    feasible_count = 0
    worst_gap = 0.0
    for inst in instances:
        r = results.get(inst) or {}
        feasible = r.get("feasible") is True
        gap = r.get("gap")
        gap_val = abs(float(gap)) if gap is not None else math.inf
        if feasible:
            feasible_count += 1
        if feasible and gap_val <= gap_threshold:
            passed += 1
        if not math.isinf(gap_val) and gap_val > worst_gap:
            worst_gap = gap_val

    score = 1.0 if passed == len(instances) else 0.0
    metrics = {
        "combined_score": score,
        "stage1_passed": float(passed),
        "stage1_total": float(len(instances)),
        "stage1_feasible_count": float(feasible_count),
        "stage1_worst_gap": float(worst_gap) if worst_gap > 0 else 0.0,
        "stage1_gap_threshold": gap_threshold,
    }
    # Build artifact for the *first* stage1 instance (typically "tiny").
    # Multi-instance stage1 is supported but rare; we summarize the first.
    artifacts = _build_stage1_artifacts(
        results.get(instances[0]) or {}, output_dir, paper_id, inst=instances[0],
    )
    return _maybe_wrap_with_artifacts(metrics, artifacts)


def evaluate_stage2(program_path: str):
    """Stage 2 scoring. Dispatches to a pluggable Scorer selected via env var.

    ``EFFICIENT_OR_STAGE2_SCORER`` ∈ {"staged_qte", "aocc"}
    (default: staged_qte). See test_time_self_evolution/scoring/ for details.
    """
    # Imported lazily so unit tests that mock eval_core don't need the scoring deps.
    from test_time_self_evolution.scoring import DEFAULT_SCORER, get_scorer
    from test_time_self_evolution.scoring.base import ScoreContext
    from test_time_self_evolution.scoring.building_blocks import lookup_gurobi_time

    paper_id, model_name, exec_mode, base_output, t_max, exec_cfg = _common_env()
    instances = _split_instances(os.environ.get("EFFICIENT_OR_STAGE2_INSTANCES", ""))
    time_limit_cap = int(os.environ.get("EFFICIENT_OR_STAGE2_TIME_LIMIT", "3600"))
    time_policy = os.environ.get("EFFICIENT_OR_STAGE2_TIME_POLICY", "uniform")
    time_buffer = int(os.environ.get("EFFICIENT_OR_STAGE2_TIME_BUFFER", "0"))
    scorer_name = os.environ.get("EFFICIENT_OR_STAGE2_SCORER", DEFAULT_SCORER)
    # staged_qte's Stage1/Stage2 gap split is CLI-tunable (--stage2-stage-boundary).
    # Only staged_qte takes it; aocc has no such knob, so pass it conditionally.
    _stage_boundary_raw = os.environ.get("EFFICIENT_OR_STAGE2_STAGE_BOUNDARY", "")
    scorer_kwargs = {}
    if scorer_name == "staged_qte" and _stage_boundary_raw:
        scorer_kwargs["stage_boundary"] = float(_stage_boundary_raw)

    if not instances:
        return {"combined_score": 0.0, "stage2_error": "no_stage2_instances"}

    # Pre-compute τ_g for each instance (needed by both the scorer and, when
    # the time policy is gurobi-based, by the per-instance time budget).
    tau_g_map = {inst: lookup_gurobi_time(paper_id, inst) for inst in instances}

    # Build the per-instance time budget according to the policy.
    #   "uniform"                → every instance uses ``time_limit_cap``
    #   "gurobi_time"            → T_i = min(τ_g_i, time_limit_cap); fallback time_limit_cap
    #   "gurobi_time_plus_buffer"→ T_i = min(τ_g_i + buffer, time_limit_cap); fallback cap
    per_instance_tl: "dict[str, int]" = {}
    for inst in instances:
        tau_g = tau_g_map.get(inst)
        if time_policy == "gurobi_time" and tau_g is not None:
            per_instance_tl[inst] = min(int(tau_g), time_limit_cap)
        elif time_policy == "gurobi_time_plus_buffer" and tau_g is not None:
            per_instance_tl[inst] = min(int(tau_g) + time_buffer, time_limit_cap)
        else:
            per_instance_tl[inst] = time_limit_cap

    output_dir = os.path.join(base_output, "stage2")
    os.makedirs(output_dir, exist_ok=True)
    results = eval_core.evaluate_candidate_code(
        paper_id, model_name, instances, program_path, output_dir,
        per_instance_tl, exec_mode, exec_cfg, t_max,
    )
    _abort_if_license_died(results, "stage2")

    # Direction matters: "LLM beats Gurobi" is direction-dependent. Pulled from
    # paper_meta_info.csv via eval_core.get_paper_direction.
    direction = eval_core.get_paper_direction(paper_id)

    scorer = get_scorer(scorer_name, **scorer_kwargs)

    per_scores: list = []
    per_debug: list = []
    for inst in instances:
        r = results.get(inst) or {}
        log_path = os.path.join(output_dir, f"log_{inst}.jsonl")
        ctx = ScoreContext(
            time_limit=per_instance_tl[inst],
            gurobi_time=tau_g_map.get(inst),
            gurobi_obj=r.get("gurobi_obj"),
            direction=direction,
            log_path=log_path,
            paper_id=paper_id,
            instance=inst,
        )
        s, dbg = scorer.score_instance(r, ctx)
        per_scores.append(s)
        per_debug.append((inst, s, dbg))

    combined = scorer.aggregate(per_scores)

    # Common aggregate fields across scorers.
    feasible_count = sum(
        1 for _, _, d in per_debug if d.get("reason") not in {"infeasible", "missing_gurobi_baseline", "empty_log", "no_incumbent_within_tau_g"}
    )
    out = {
        "combined_score": round(combined, 6),
        "stage2_scorer": scorer_name,
        "stage2_total": float(len(instances)),
        "stage2_feasible_count": float(feasible_count),
    }

    # Scorer-specific aggregate metrics, surfaced for logging / ablation.
    if scorer_name == "staged_qte":
        scored = [d for _, _, d in per_debug if "stage_id" in d and d["stage_id"] != 0]
        n = max(len(scored), 1)
        out["stage2_mean_quality_part"] = (
            round(sum(d.get("quality_part", 0.0) for d in scored) / n, 6) if scored else 0.0
        )
        out["stage2_mean_speed_part"] = (
            round(sum(d.get("speed_part", 0.0) for d in scored) / n, 6) if scored else 0.0
        )
        out["stage2_mean_signed_gap"] = (
            round(sum(d.get("signed_gap", 1.0) for d in scored) / n, 6) if scored else 1.0
        )
        # Stage breakdown counts (how many instances landed in stage 1 vs 2)
        out["stage2_n_stage1"] = float(sum(1 for d in scored if d.get("stage_id") == 1))
        out["stage2_n_stage2"] = float(sum(1 for d in scored if d.get("stage_id") == 2))
        # Beat-Gurobi metrics
        beat_list = [d for d in scored if d.get("beat_gurobi")]
        out["stage2_any_beat_gurobi"] = 1.0 if beat_list else 0.0
        out["stage2_beat_gurobi_count"] = float(len(beat_list))
        out["stage2_mean_beat_amount"] = (
            round(sum(d.get("beat_amount", 0.0) for d in scored) / n, 6) if scored else 0.0
        )
        out["stage2_match_count"] = float(sum(1 for d in scored if d.get("matched")))
    elif scorer_name == "aocc":
        aoccs = [d.get("aocc", 1.0) for _, _, d in per_debug if "aocc" in d]
        out["stage2_mean_aocc"] = round(sum(aoccs) / len(aoccs), 6) if aoccs else 1.0

    # Per-instance flattened fields. These let downstream consumers
    # (run_self_evolve) reconstruct per-instance result rows from the best
    # program's checkpoint JSON without re-running the final evaluation.
    for inst, s, dbg in per_debug:
        r = results.get(inst) or {}
        p = f"inst_{inst}"
        # feasibility as 1.0/0.0/-1.0 (True/False/None)
        feasible = r.get("feasible")
        out[f"{p}_feasible"] = (
            1.0 if feasible is True else (0.0 if feasible is False else -1.0)
        )
        # Core result fields
        gap = r.get("gap")
        out[f"{p}_gap"] = float(gap) if gap is not None else 1.0
        obj = r.get("llm_obj")
        out[f"{p}_obj"] = float(obj) if obj is not None else 0.0
        gref = r.get("gurobi_obj")
        out[f"{p}_gurobi_obj"] = float(gref) if gref is not None else 0.0
        tsolve_val = r.get("solve_time")
        out[f"{p}_time"] = float(tsolve_val) if tsolve_val is not None else 0.0
        aocc_val = r.get("aocc")
        out[f"{p}_aocc"] = float(aocc_val) if aocc_val is not None else 1.0
        # Per-instance score that went into aggregate
        out[f"{p}_score"] = round(float(s), 6) if s is not None else 0.0
        # staged_qte-specific debug fields
        if scorer_name == "staged_qte":
            out[f"{p}_stage_id"] = float(dbg.get("stage_id", 0))
            out[f"{p}_quality_part"] = float(dbg.get("quality_part", 0.0))
            out[f"{p}_speed_part"] = float(dbg.get("speed_part", 0.0))
            out[f"{p}_signed_gap"] = float(dbg.get("signed_gap", 1.0))
            out[f"{p}_beat_amount"] = float(dbg.get("beat_amount", 0.0))
            out[f"{p}_beat_gurobi"] = 1.0 if dbg.get("beat_gurobi") else 0.0
            out[f"{p}_matched"] = 1.0 if dbg.get("matched") else 0.0

    artifacts = _build_stage2_artifacts(
        scorer_name, results, instances, per_instance_tl, out, output_dir, paper_id,
    )
    return _maybe_wrap_with_artifacts(out, artifacts)


def evaluate(program_path: str):
    """Direct (non-cascade) fallback. Uses legacy env vars."""
    paper_id, model_name, exec_mode, base_output, t_max, exec_cfg = _common_env()
    selection_instance = os.environ.get("EFFICIENT_OR_SELECTION_INSTANCE", "tiny")
    instances = _split_instances(os.environ.get("EFFICIENT_OR_INSTANCES", selection_instance))
    time_limit = int(os.environ.get("EFFICIENT_OR_TIME_LIMIT", "300"))

    output_dir = base_output
    os.makedirs(output_dir, exist_ok=True)
    results = eval_core.evaluate_candidate_code(
        paper_id, model_name, instances, program_path, output_dir,
        time_limit, exec_mode, exec_cfg, t_max,
    )
    result = results.get(selection_instance) or next(iter(results.values()))
    score = eval_modes.combined_score(result)
    gap = result.get("gap")
    solve_time = result.get("solve_time")
    return {
        "combined_score": score,
        "feasible": 1.0 if result.get("feasible") is True else 0.0,
        "gap": float(gap) if gap is not None else 1.0,
        "runtime": float(solve_time) if solve_time is not None else time_limit,
    }


def evaluate_program(program_path: str):
    return evaluate(program_path)


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: evaluator.py <program_path> [stage1|stage2|direct]", file=sys.stderr)
        return 2
    stage = sys.argv[2] if len(sys.argv) == 3 else "direct"
    fn = {
        "stage1": evaluate_stage1,
        "stage2": evaluate_stage2,
        "direct": evaluate,
    }.get(stage, evaluate)
    print(json.dumps(fn(sys.argv[1]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
