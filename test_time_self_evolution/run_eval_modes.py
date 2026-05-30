#!/usr/bin/env python
"""Run one-shot, best-of-K, and self-evolving evaluation modes."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import traceback

import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import one_shot_eval as eval_core  # noqa: E402
from test_time_self_evolution import eval_modes  # noqa: E402
from test_time_self_evolution.coral import runner as coral_runner  # noqa: E402
from test_time_self_evolution.eoh import runner as eoh_runner  # noqa: E402
from test_time_self_evolution.openevolve import runner as openevolve_runner  # noqa: E402
from scripts.utils.instance_paths import (  # noqa: E402
    DEFAULT_INSTANCES,
    SELF_EVOLVE_STAGE1_INSTANCES,
    SELF_EVOLVE_STAGE2_INSTANCES,
    SELF_EVOLVE_TEST_INSTANCES,
    parse_instances_arg,
)


def load_mode_config():
    config_path = os.path.join(ROOT_DIR, "configs", "oneshot.yaml")
    keys_path = os.path.join(ROOT_DIR, "configs", "api_keys.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    keys = {}
    if os.path.exists(keys_path):
        with open(keys_path, encoding="utf-8") as f:
            keys = yaml.safe_load(f) or {}
    api_key = os.environ.get("OPENROUTER_API_KEY") or keys.get("OPENROUTER_API_KEY_SELF_EVOLVE")
    if not api_key:
        raise SystemExit("ERROR: set env OPENROUTER_API_KEY or OPENROUTER_API_KEY_SELF_EVOLVE in configs/api_keys.yaml")
    config["OPENROUTER_API_KEY"] = api_key
    if not config.get("models"):
        if config.get("model"):
            config["models"] = [config["model"]]
        else:
            raise SystemExit("ERROR: configs/oneshot.yaml must define models or model")
    return config


def select_models(config, model_short_names):
    all_models = config["models"]
    if not model_short_names:
        return all_models
    by_short = {eval_core.get_model_short_name(model): model for model in all_models}
    unknown = sorted(set(model_short_names) - set(by_short))
    if unknown:
        print(f"WARNING: unknown model(s) {unknown}, available: {sorted(by_short)}")
    models = [by_short[name] for name in model_short_names if name in by_short]
    if not models:
        raise SystemExit("ERROR: no matching models selected")
    return models


def resolve_model_id(config, ref):
    """Accept either a short name (gpt-5.3-codex) or full id (openai/gpt-5.3-codex)."""
    if ref is None:
        return None
    all_models = config["models"]
    by_short = {eval_core.get_model_short_name(m): m for m in all_models}
    if ref in by_short:
        return by_short[ref]
    if ref not in all_models:
        print(f"WARNING: '{ref}' not listed in configs/oneshot.yaml models; using as-is.")
    return ref


def build_prompt(paper_id):
    problem_desc = eval_core.read_problem_description(paper_id)
    instance_template = eval_core.read_instance_template(paper_id)
    solution_template = eval_core.read_solution_template(paper_id)
    if instance_template is None:
        raise RuntimeError(f"instance_template.json not found for {paper_id}")
    if solution_template is None:
        raise RuntimeError(f"solution_template.json not found for {paper_id}")
    return eval_core.build_prompt(problem_desc, instance_template, solution_template)


def run_per_model_modes(args, run_id, config, paper_id, model, prompt):
    """Run modes that iterate over --models (one_shot, best_of_k)."""
    common = {
        "run_id": run_id,
        "paper_id": paper_id,
        "model": model,
        "prompt": prompt,
        "config": config,
        "instances": args.instances,
        "time_limit": args.time_limit,
        "exec_mode": args.exec_mode,
        "exec_cfg": {"cpus": args.cpus, "memory": args.memory},
        "t_max": args.t_max,
    }
    results = {}
    if "one_shot" in args.modes:
        results["one_shot"] = eval_modes.run_one_shot(**common)
    if "best_of_k" in args.modes:
        results["best_of_k"] = eval_modes.run_best_of_k(
            **common,
            k=args.k,
            selection_instance=args.selection_instance,
        )
    return results


_DEV_SET_SENTINEL_MAX = "__SENTINEL_MAX_TAU_G__"
_DEV_SET_SENTINEL_MEDIAN = "__SENTINEL_MEDIAN_TAU_G__"
_DEV_SET_SENTINEL_KEYWORDS_MAX = {"max", "max_tau_g", "max-tau-g", "auto"}
_DEV_SET_SENTINEL_KEYWORDS_MEDIAN = {"median", "median_tau_g", "median-tau-g"}


def _normalize_dev_set_arg(raw):
    """Convert user-provided ``--dev-set`` value into either an internal
    sentinel string or a validated list of explicit instance names.

    Accepted forms:
      - ``None``                  → median-τ_g sentinel (omit-flag default)
      - ``["max"|"auto"|...]``    → max-τ_g sentinel
      - ``["median"|"median_tau_g"]`` → median-τ_g sentinel
      - ``["large_11", ...]``     → explicit list (validated by parse_instances_arg)
    """
    if raw is None:
        return [_DEV_SET_SENTINEL_MEDIAN]
    if not raw:
        raise SystemExit(
            "ERROR: --dev-set cannot be explicitly empty. "
            "Omit the flag for max-τ_g auto-pick, pass 'median' for median-τ_g "
            "auto-pick, or pass one or more instance names."
        )
    if len(raw) == 1:
        token = str(raw[0]).strip().lower()
        if token in _DEV_SET_SENTINEL_KEYWORDS_MAX:
            return [_DEV_SET_SENTINEL_MAX]
        if token in _DEV_SET_SENTINEL_KEYWORDS_MEDIAN:
            return [_DEV_SET_SENTINEL_MEDIAN]
    # Otherwise — explicit instance names; validate via parse_instances_arg
    return parse_instances_arg(raw)


def _resolve_dev_set_for_paper(paper_id: str, dev_set_arg):
    """Resolve ``--dev-set`` to concrete instance list for this paper.

    ``dev_set_arg`` is the output of ``_normalize_dev_set_arg`` — either a
    one-element sentinel list or a validated explicit instance list.
    """
    if dev_set_arg == [_DEV_SET_SENTINEL_MAX]:
        from test_time_self_evolution.scoring.building_blocks import pick_max_tau_g_instance
        picked = pick_max_tau_g_instance(paper_id)
        if not picked:
            raise SystemExit(
                f"ERROR: cannot auto-pick --dev-set for paper '{paper_id}' "
                f"(no large_* instances on disk, or no Gurobi τ_g recorded for any). "
                f"Pass --dev-set explicitly."
            )
        print(f"  [dev-set auto] {paper_id} → {picked} (max τ_g)")
        return [picked]
    if dev_set_arg == [_DEV_SET_SENTINEL_MEDIAN]:
        from test_time_self_evolution.scoring.building_blocks import pick_median_tau_g_instance
        picked = pick_median_tau_g_instance(paper_id)
        if not picked:
            raise SystemExit(
                f"ERROR: cannot auto-pick --dev-set (median) for paper '{paper_id}' "
                f"(no large_* instances on disk, or no Gurobi τ_g recorded for any). "
                f"Pass --dev-set explicitly."
            )
        print(f"  [dev-set auto] {paper_id} → {picked} (median τ_g)")
        return [picked]
    # Explicit list
    return list(dev_set_arg)


def _resolve_test_set_for_paper(paper_id: str, test_set_arg, dev_set_resolved):
    """Compute final test instances per paper.

    When ``--test-set`` is the **default** (``SELF_EVOLVE_TEST_INSTANCES``),
    auto-compute test = (all ``large_*`` instances on disk for this paper)
    minus dev pick. This avoids the dev/test overlap that arises when
    ``--dev-set median`` happens to pick an instance also listed in the
    default 4-instance test set (e.g. ``large_31`` often has median τ_g and
    is also in ``[large_21, large_31, large_41, large_51]``).

    When the user explicitly passes ``--test-set``, trust them and use as-is
    — preserves backwards-compat escape hatch.

    Affects all 3 frameworks (OpenEvolve / EoH / CORAL) since the
    dispatcher (``run_self_evolve_mode``) is shared.

    Edge cases:
      - paper directory missing on disk → fall back to declared default
      - no ``large_instance_*.json`` files → return empty (runner handles
        empty test_instances by reading from cache instead of re-evaluating)
      - dev pick not in any large_* (e.g. ``--dev-set tiny``) → test = all
        large_* (no exclusion needed)
    """
    is_default = (list(test_set_arg) == list(SELF_EVOLVE_TEST_INSTANCES))
    if not is_default:
        return list(test_set_arg)

    instance_dir = os.path.join(
        ROOT_DIR, "frontier-or", paper_id, "instance",
    )
    if not os.path.isdir(instance_dir):
        # Paper data missing — preserve original behavior
        return list(test_set_arg)

    pattern = os.path.join(instance_dir, "large_instance_*.json")
    inst_ids = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"large_instance_(\d+)\.json$", path)
        if m:
            inst_ids.append(f"large_{m.group(1)}")
    if not inst_ids:
        return []

    dev_set_clean = set(dev_set_resolved)
    test_set = [i for i in inst_ids if i not in dev_set_clean]
    print(f"  [test-set auto] {paper_id} → {test_set} "
          f"(all {len(inst_ids)} large_* on disk - dev {sorted(dev_set_clean)})")
    return test_set


def _coral_max_seconds_per_paper(
    paper_id: str,
    dev_instances,
    stage1_time_limit: int,
    stage2_time_limit: int,
    stage2_time_policy: str,
    stage2_time_buffer: int,
    *,
    attempts: int = 1,
    agent_thinking_budget: int = 300,
    framework_overhead: int = 60,
    multiplier: float = 1.0,
) -> int:
    """Auto-derive per-paper CORAL wall-clock budget.

    Components per attempt:
      - stage1_time_limit (tiny gate cap)
      - sum(min(τ_g_i, stage2_time_limit)) over dev instances under gurobi_time policy
      - agent_thinking_budget (LLM turns between attempts ≈ 300s for 3-5 turns)
      - framework_overhead (post-commit hook + JSON IO + git ≈ 60s)

    Multiplied by ``attempts`` × ``multiplier`` for multi-attempt runs.
    """
    from test_time_self_evolution.scoring.building_blocks import lookup_gurobi_time

    stage2_actual = 0
    for inst in dev_instances:
        tau_g = lookup_gurobi_time(paper_id, inst)
        if stage2_time_policy == "gurobi_time" and tau_g is not None:
            stage2_actual += min(int(tau_g), stage2_time_limit)
        elif stage2_time_policy == "gurobi_time_plus_buffer" and tau_g is not None:
            stage2_actual += min(int(tau_g) + stage2_time_buffer, stage2_time_limit)
        else:
            stage2_actual += stage2_time_limit

    per_attempt = stage1_time_limit + stage2_actual + agent_thinking_budget + framework_overhead
    return int(per_attempt * attempts * multiplier)


def _resolve_coral_max_seconds(args, paper_id: str, dev_set):
    """Resolve --coral-max-seconds (int or 'auto') to a concrete int per paper."""
    raw = str(args.coral_max_seconds).strip().lower()
    if raw == "auto":
        seconds = _coral_max_seconds_per_paper(
            paper_id, dev_set,
            args.stage1_time_limit, args.stage2_time_limit,
            args.stage2_time_policy, args.stage2_time_buffer,
            attempts=args.coral_attempts,
            multiplier=args.coral_attempts_budget_multiplier,
        )
        print(f"  [coral max_seconds auto] {paper_id} → {seconds}s "
              f"(stage1={args.stage1_time_limit} + stage2_actual + agent_thinking 300 "
              f"+ overhead 60, × attempts={args.coral_attempts} × {args.coral_attempts_budget_multiplier})")
        return seconds
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(
            f"ERROR: --coral-max-seconds must be an integer or 'auto', got {args.coral_max_seconds!r}"
        )


def run_self_evolve_mode(args, run_id, config, paper_id, prompt, primary_model, secondary_model):
    dev_set = _resolve_dev_set_for_paper(paper_id, args.stage2_instances)
    test_set = _resolve_test_set_for_paper(paper_id, args.test_instances, dev_set)
    common = {
        "run_id": run_id,
        "paper_id": paper_id,
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "prompt": prompt,
        "config": config,
        "stage1_instances": args.stage1_instances,
        "stage2_instances": dev_set,
        "test_instances": test_set,
        "stage1_time_limit": args.stage1_time_limit,
        "stage2_time_limit": args.stage2_time_limit,
        "test_time_limit": args.test_time_limit,
        "stage1_gap_threshold": args.stage1_gap_threshold,
        "exec_mode": args.exec_mode,
        "exec_cfg": {"cpus": args.cpus, "memory": args.memory},
        "t_max": args.t_max,
        "stage2_scorer": args.stage2_scorer,
        "stage2_stage_boundary": args.stage2_stage_boundary,
        "stage2_time_policy": args.stage2_time_policy,
        "stage2_time_buffer": args.stage2_time_buffer,
        "test_time_policy": args.test_time_policy,
        "test_time_buffer": args.test_time_buffer,
        "test_instance_workers": args.test_instance_workers,
    }
    if args.framework == "eoh":
        return eoh_runner.run_self_evolve(
            **common,
            pop_size=args.eoh_pop_size,
            n_pop=args.eoh_n_pop,
            workers=args.eoh_workers,
            timeout=args.eoh_timeout,
            operators=args.eoh_operators,
            resume=args.resume,
            enable_artifact=args.eoh_enable_artifact,
            system_include_spec=args.eoh_system_include_spec,
        )
    if args.framework == "coral":
        return coral_runner.run_self_evolve(
            **common,
            attempts=args.coral_attempts,
            max_seconds=_resolve_coral_max_seconds(args, paper_id, dev_set),
            agent_runtime=args.coral_agent_runtime,
            agent_count=args.coral_agent_count,
            agent_model=args.coral_agent_model,
            max_turns=args.coral_max_turns,
            gateway_enabled=args.coral_gateway,
            heartbeat_reflect_every=args.coral_heartbeat_reflect_every,
            heartbeat_pivot_every=args.coral_heartbeat_pivot_every,
            heartbeat_consolidate_every=args.coral_heartbeat_consolidate_every,
            resume=args.resume,
        )
    return openevolve_runner.run_self_evolve(
        **common,
        iterations=args.openevolve_iterations,
        resume=args.resume,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["self_evolve"],
                        choices=["one_shot", "best_of_k", "self_evolve"],
                        help="Evaluation mode(s) to run. Default: self_evolve only. "
                             "Pass multiple to run several modes in one invocation.")
    parser.add_argument("--framework", default="openevolve", choices=["openevolve", "eoh", "coral"],
                        help="Framework backend for --modes self_evolve. Default: openevolve.")
    parser.add_argument("--paper-id", dest="paper_ids", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model short names from configs/oneshot.yaml, e.g. gpt-5.3-codex. "
                             "Iterates one_shot/best_of_k over each. Ignored by self_evolve.")
    parser.add_argument("--primary-model", default="google/gemini-3.1-pro-preview",
                        help="self_evolve primary model (full OpenRouter id). "
                             "Default: google/gemini-3.1-pro-preview. Override with e.g. "
                             "openai/gpt-5.3-codex / deepseek/deepseek-r1 / "
                             "anthropic/claude-opus-4.6.")
    parser.add_argument("--secondary-model", default=None,
                        help="self_evolve secondary model. Defaults to --primary-model.")
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES,
                        help="Instances for one_shot/best_of_k. Ignored by self_evolve.")
    parser.add_argument("--selection-instance", default="tiny",
                        help="best_of_k selection instance; must be in --instances.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--openevolve-iterations", type=int, default=30)
    parser.add_argument("--eoh-pop-size", type=int, default=4,
                        help="EoH population size for --framework eoh. Default: 4.")
    parser.add_argument("--eoh-n-pop", type=int, default=4,
                        help="EoH number of populations/generations. Default: 4.")
    parser.add_argument("--eoh-workers", type=int, default=None,
                        help="EoH offspring-level parallel workers (joblib n_jobs). "
                             "Default: matches --eoh-pop-size (so each operator's pop_size "
                             "offspring all run concurrently). Set explicitly to a smaller "
                             "value if you hit OpenRouter rate limits or want fewer "
                             "concurrent CPUs.")
    parser.add_argument("--eoh-timeout", type=int, default=None,
                        help="EoH per-candidate evaluation timeout. Default: stage2 time limit + 60s.")
    parser.add_argument("--eoh-operators", nargs="+", default=None,
                        choices=["e1", "e2", "m1", "m2", "m3"],
                        help="EoH evolution operators (subset of e1/e2/m1/m2/m3). "
                             "Default: all 5. Examples: '--eoh-operators e1 e2 m1' to "
                             "drop m2/m3 (saves ~40%% evals); '--eoh-operators e2 m1' for "
                             "minimal exploit-only setting. Affects total LLM evals via "
                             "the formula pop_size + len(operators) × pop_size × n_pop.")
    parser.add_argument("--eoh-enable-artifact", action="store_true",
                        help="EoH: feed OpenEvolve evaluator artifacts "
                             "(failure_breakdown / score_summary text) back into "
                             "the next-generation prompts (e1/e2/m1/m2/m3). "
                             "Default: off — keep EoH's native 'no LLM-feedback' "
                             "design as the baseline.")
    parser.add_argument("--eoh-system-include-spec", action="store_true",
                        help="EoH e/m operators: include the full benchmark "
                             "spec (paper problem_description + instance/solution "
                             "schemas + TASK_SPECIFICATION) in the system message. "
                             "Default off — system message contains only the CLI "
                             "CONTRACT preamble (matches OpenEvolve, which doesn't "
                             "send the per-paper spec to the LLM but relies on "
                             "parent code carrying that context implicitly). Turn "
                             "on if you want EoH e/m operators to always re-see "
                             "the paper spec at the system level. i1 is unaffected.")
    parser.add_argument("--coral-attempts", type=int, default=1,
                        help="CORAL: stop after this many finalized attempts. Default: 1.")
    parser.add_argument("--coral-max-seconds", default="900",
                        help="CORAL wall-clock cap (seconds) before stopping agents. "
                             "Integer (default 900), or 'auto' to derive per-paper from "
                             "stage1+stage2_actual(τ_g)+agent_thinking(300)+overhead(60), "
                             "× --coral-attempts × --coral-attempts-budget-multiplier. "
                             "'auto' avoids 'coral_no_attempt' on high-τ_g papers without "
                             "inflating fast papers' budgets.")
    parser.add_argument("--coral-attempts-budget-multiplier", type=float, default=1.0,
                        help="Safety multiplier applied to '--coral-max-seconds auto' "
                             "calculation. Default 1.0 (just-fit). Bump to 1.3-1.5 if you "
                             "see frequent wall-clock interrupts; lower to 0.8 if you want "
                             "tighter budgets and accept more truncations.")
    parser.add_argument("--coral-agent-runtime", default="codex",
                        choices=["codex", "claude_code", "opencode", "kiro"],
                        help="CORAL agent runtime. Default: codex.")
    parser.add_argument("--coral-agent-count", type=int, default=1,
                        help="CORAL agent count. Default: 1.")
    parser.add_argument("--coral-agent-model", default=None,
                        help="CORAL agent model override. Defaults to the primary model short name for codex.")
    parser.add_argument("--coral-max-turns", type=int, default=20,
                        help="CORAL max turns per agent process. Default: 20.")
    parser.add_argument("--coral-gateway", action="store_true",
                        help="CORAL: route agent traffic through its LiteLLM gateway using OPENROUTER_API_KEY.")
    parser.add_argument("--coral-heartbeat-reflect-every", type=int, default=0,
                        help="CORAL: trigger 'reflect' heartbeat every N agent evals (interval). "
                             "Pauses agent and resumes with a 'review your recent work' prompt. "
                             "0 = off (default). Each trigger ≈ doubles per-attempt LLM cost.")
    parser.add_argument("--coral-heartbeat-pivot-every", type=int, default=5,
                        help="CORAL: trigger 'pivot' heartbeat after N consecutive non-improving "
                             "agent evals (plateau). Pauses agent and resumes with a 'change "
                             "direction' prompt. 0 = off. Default 5 — only fires when stuck, "
                             "so cost is negligible on healthy runs.")
    parser.add_argument("--coral-heartbeat-consolidate-every", type=int, default=0,
                        help="CORAL: trigger 'consolidate' heartbeat every N GLOBAL evals "
                             "(across all agents). Useful only with --coral-agent-count > 1 to "
                             "merge cross-agent notes. 0 = off (default).")
    parser.add_argument("--time_limit", type=int, default=300,
                        help="time limit for one_shot/best_of_k. self_evolve uses --stage*-time-limit.")
    # self_evolve stage presets (override to customize per run).
    parser.add_argument("--stage1-instances", nargs="+", default=SELF_EVOLVE_STAGE1_INSTANCES,
                        help="self_evolve stage1 instances (binary gate). Preset: tiny.")
    parser.add_argument("--dev-set", "--dev_set", "--stage2-instances",
                        dest="stage2_instances",
                        nargs="+", default=SELF_EVOLVE_STAGE2_INSTANCES,
                        help="self_evolve dev set: instances used inside the evolve loop as "
                             "the fitness signal (stage2). Three modes: "
                             "(1) omit flag → auto-pick the large_* instance with the MEDIAN "
                             "Gurobi τ_g per paper (representative, avoids the worst-case slowest); "
                             "(2) '--dev-set max' (or 'max_tau_g' / 'auto') → auto-pick the LARGEST "
                             "τ_g instance per paper (hardest available); "
                             "(3) '--dev-set large_11' (or multiple names) → explicit override. "
                             "Aliases for the flag: --dev_set, --stage2-instances.")
    parser.add_argument("--test-set", "--test_set", "--test-instances",
                        dest="test_instances",
                        nargs="+", default=SELF_EVOLVE_TEST_INSTANCES,
                        help="self_evolve held-out test set: instances used only after the "
                             "evolve loop, on the final best program. "
                             "Preset: large_21 large_31 large_41 large_51. "
                             "Empty → skip the post-evolve eval and reconstruct per-instance "
                             "rows for the dev results CSV from OpenEvolve's checkpoint. "
                             "Aliases: --test_set, --test-instances.")
    parser.add_argument("--stage1-time-limit", type=int, default=300,
                        help="Per-instance time limit (s) for stage1 (tiny gate). "
                             "Preset: 300.")
    parser.add_argument("--stage2-time-limit", type=int, default=3600,
                        help="Per-instance time limit (s) for stage2. Preset: 3600.")
    parser.add_argument("--test-time-limit", type=int, default=3600,
                        help="Per-instance time limit (s) for final test. Preset: 3600.")
    parser.add_argument("--stage1-gap-threshold", type=float, default=0.10,
                        help="Stage1 passes iff feasible and |gap| <= this. Preset: 0.10 (10%%), "
                             "matching one_shot_eval.py's tiny gate.")
    parser.add_argument("--stage2-scorer", default="staged_qte",
                        choices=["staged_qte", "aocc"],
                        help="self_evolve Stage2 scoring scheme. "
                             "'staged_qte' (default): two-stage QTE parameterized by "
                             "--stage2-stage-boundary (call it b). Stage 1 (gap>b) "
                             "score=max(0, 1-signed_gap) ∈ [0, 1-b]; Stage 2 (gap≤b) "
                             "score=(1-signed_gap)+max(0, (τ_g-t_solve)/τ_g) ∈ "
                             "[1-b, 2+|beat|]. Anchors (independent of b): "
                             "1.0 = parity with Gurobi, 2.0 = match+instant, "
                             ">2.0 = beat Gurobi. "
                             "'aocc' (backup): pure 1-AOCC anytime baseline ∈ [0, 1].")
    parser.add_argument("--stage2-stage-boundary", type=float, default=0.01,
                        help="staged_qte scorer's Stage1/Stage2 gap split. A candidate "
                             "with signed gap > this is in Stage 1 (score ∈ [0, 1-boundary], "
                             "quality only); gap ≤ this enters Stage 2 (score ≥ 1-boundary, "
                             "quality + speed bonus). Preset: 0.01 (1%%). Raise to relax "
                             "the quality bar. Ignored when --stage2-scorer=aocc.")
    parser.add_argument("--stage2-time-policy", default="gurobi_time",
                        choices=["uniform", "gurobi_time", "gurobi_time_plus_buffer"],
                        help="How to set per-instance stage2 time_limit. "
                             "'uniform' uses --stage2-time-limit for all; "
                             "'gurobi_time' uses τ_g per instance (capped at --stage2-time-limit); "
                             "'gurobi_time_plus_buffer' uses τ_g+buffer capped at --stage2-time-limit. "
                             "Default: gurobi_time (matches dev-set auto-pick by max τ_g; "
                             "candidates eval up to τ_g per instance, capped at --stage2-time-limit).")
    parser.add_argument("--stage2-time-buffer", type=int, default=0,
                        help="Buffer seconds added to τ_g when policy is gurobi_time_plus_buffer. "
                             "Default: 300.")
    parser.add_argument("--test-time-policy", default="uniform",
                        choices=["uniform", "gurobi_time", "gurobi_time_plus_buffer"],
                        help="Same as --stage2-time-policy but for the post-evolve final "
                             "test evaluation. Default: uniform — give each test instance "
                             "the full --test-time-limit budget regardless of τ_g, so the "
                             "best program gets a fair shot to demonstrate quality without "
                             "being cut short by Gurobi's possibly-too-tight τ_g.")
    parser.add_argument("--test-time-buffer", type=int, default=0,
                        help="Buffer seconds added to τ_g for final test when policy is "
                             "gurobi_time_plus_buffer. Default: 0 (matches the runner "
                             "function default; pass a positive value to widen the budget).")
    parser.add_argument("--test-instance-workers", type=int, default=None,
                        help="ThreadPool size for the post-evolve test eval — fans out "
                             "test instances in parallel via "
                             "eval_modes.evaluate_best_on_test_set. Default: matches "
                             "--test-instances size (so all test instances run concurrently). "
                             "Lower this if Gurobi license cap is tight (peak license usage "
                             "= paper-workers × test-instance-workers).")
    parser.add_argument("--paper-workers", type=int, default=6,
                        help="Number of papers to run in parallel (default 6, conservative). "
                             "Each paper is otherwise independent; CSV writes are protected by "
                             "fcntl.flock. Combines multiplicatively with OpenEvolve's "
                             "parallel_evaluations and --test-instance-workers, so size "
                             "carefully relative to total CPUs and Gurobi license tokens "
                             "(worst case: paper_workers × (parallel_evaluations + "
                             "test_instance_workers) concurrent subprocesses). Bump to 100 "
                             "for full-throttle 186-paper batch runs on a 112-CPU server with "
                             "100-token academic license.")
    parser.add_argument("--dev-instance-workers", type=int, default=None,
                        help="Per-candidate dev-set instance fan-out (ThreadPool size). "
                             "Default: matches --dev-set size (so all dev instances "
                             "evaluate concurrently for each candidate). Sentinel dev-set "
                             "(max/median auto-pick) resolves to 1 instance per paper, so "
                             "default workers = 1 in that case. Set explicitly to lower "
                             "concurrency if memory-bound or to widen if you've expanded "
                             "--dev-set. Multiplies with --paper-workers and per-framework "
                             "offspring/iteration parallelism.")
    parser.add_argument("--exec-mode", choices=sorted(eval_core.EXEC_BACKENDS.keys()), default="systemd",
                        help="Execution backend. Default 'systemd' pins each candidate to "
                             "--cpus cores (via taskset + cgroup), preventing multi-thread "
                             "fitness cheats. Use 'bare' only for debugging; it has no CPU limits.")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory", default="640G",
                        help="cgroup memory hard cap per candidate subprocess (systemd "
                             "exec mode). Default 640G — sized for 112-CPU server with ~1TB "
                             "RAM. Lower this if your machine has less memory; the cap is "
                             "a defensive ceiling against LLM-generated code that allocates "
                             "unbounded arrays, not the typical solver footprint.")
    from one_shot_eval import parse_t_max as _parse_t_max
    parser.add_argument("--t_max", type=_parse_t_max, default=None,
                        help="AOCC horizon. Float (seconds) or 'gurobi' for "
                             "per-instance Gurobi solve time.")
    parser.add_argument("--run-id", default=None,
                        help="Run id (also used as output dir key). Required when --resume is set.")
    parser.add_argument("--resume", action="store_true",
                        help="self_evolve: skip seed generation and resume OpenEvolve from the "
                             "latest checkpoint under eval/modes/<run-id>/self_evolve/.../openevolve_run/checkpoints/.")
    return parser.parse_args()


def main():
    args = parse_args()
    # --eoh-workers: when omitted, default to pop_size so each operator's
    # pop_size offspring all run concurrently.
    if args.eoh_workers is None:
        args.eoh_workers = max(1, args.eoh_pop_size)
    # --dev-instance-workers: when omitted, match --dev-set size. Sentinel
    # dev-sets (max/median auto-pick) resolve to a 1-element list before this
    # point, so default workers = 1 in that case.
    if args.dev_instance_workers is None:
        # ``args.stage2_instances`` here may still be a sentinel list (e.g.
        # ``[__SENTINEL_MEDIAN_TAU_G__]``) — len() is 1 for sentinels, which
        # is correct (per-paper resolution always yields 1 instance).
        args.dev_instance_workers = max(1, len(args.stage2_instances or []))
    # --test-instance-workers: when omitted, match --test-instances size.
    if args.test_instance_workers is None:
        args.test_instance_workers = max(1, len(args.test_instances or []))
    if args.resume and not args.run_id:
        raise SystemExit("ERROR: --resume requires --run-id (must match the prior run's id).")
    if args.resume and "self_evolve" not in args.modes:
        raise SystemExit("ERROR: --resume only applies to --modes self_evolve.")
    per_model_modes = [m for m in args.modes if m in ("one_shot", "best_of_k")]

    if per_model_modes:
        args.instances = parse_instances_arg(args.instances)
        if args.selection_instance not in args.instances:
            raise SystemExit("--selection-instance must be included in --instances")

    if "self_evolve" in args.modes:
        args.stage1_instances = parse_instances_arg(args.stage1_instances)
        args.test_instances = parse_instances_arg(args.test_instances)
        # --dev-set: 3 modes (omit → max-τ_g auto, 'median' → median-τ_g auto,
        # 'large_11 ...' → explicit). Normalize to either a sentinel string or
        # a validated instance list; per-paper resolution happens later in
        # run_self_evolve_mode → _resolve_dev_set_for_paper.
        args.stage2_instances = _normalize_dev_set_arg(args.stage2_instances)
        if not args.stage1_instances:
            raise SystemExit("ERROR: self_evolve requires at least one --stage1-instances.")

    # Dev-set instance fan-out: set BEFORE any framework subprocess spawns so
    # OpenEvolve/CORAL inherit the env var via prepare_*_env's
    # ``dict(os.environ)``; also call set_instance_workers directly so EOH
    # (which evaluates candidates in this same process) picks it up.
    if args.dev_instance_workers > 1:
        os.environ["EFFICIENT_OR_INSTANCE_WORKERS"] = str(args.dev_instance_workers)
        eval_core.set_instance_workers(args.dev_instance_workers)

    gurobi_license = eval_core.configure_gurobi_license()
    config = load_mode_config()
    models = select_models(config, args.models) if per_model_modes else []
    data_dir = eval_core.get_data_dir()
    paper_ids = args.paper_ids or eval_modes.discover_papers(data_dir)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")

    # Preflight: every paper must have a registered optimization direction.
    # A missing direction silently inverts quality/QTE scores and steers
    # self-evolving search toward the worst solutions -- fail upfront.
    eval_core.validate_paper_directions(paper_ids)

    primary_model = secondary_model = None
    if "self_evolve" in args.modes:
        fallback = models[0] if models else None
        primary_model = resolve_model_id(config, args.primary_model) or fallback
        if primary_model is None:
            raise SystemExit("ERROR: self_evolve requires --primary-model (or at least one --models entry).")
        secondary_model = resolve_model_id(config, args.secondary_model) or primary_model

    print(f"Run ID: {run_id}")
    print(f"Modes: {args.modes}")
    print(f"Papers: {len(paper_ids)}")
    if per_model_modes:
        print(f"Per-model modes: {per_model_modes}")
        print(f"Models: {[eval_core.get_model_short_name(model) for model in models]}")
        print(f"Instances: {args.instances}")
        print(f"Selection instance: {args.selection_instance}")
    if "self_evolve" in args.modes:
        print(f"self_evolve framework: {args.framework}")
        print(f"self_evolve primary:   {primary_model}")
        print(f"self_evolve secondary: {secondary_model}")
        print(f"  stage1: {args.stage1_instances}  t={args.stage1_time_limit}s  gap<={args.stage1_gap_threshold}")
        if args.stage2_instances == [_DEV_SET_SENTINEL_MAX]:
            dev_label = "<auto: max-τ_g per paper>"
        elif args.stage2_instances == [_DEV_SET_SENTINEL_MEDIAN]:
            dev_label = "<auto: median-τ_g per paper>"
        else:
            dev_label = args.stage2_instances
        print(f"  dev:  {dev_label}  t<={args.stage2_time_limit}s  policy={args.stage2_time_policy}"
              + (f"(+{args.stage2_time_buffer}s)" if args.stage2_time_policy == "gurobi_time_plus_buffer" else "")
              + f"  scorer={args.stage2_scorer}")
        print(f"  test: {args.test_instances}  t<={args.test_time_limit}s  policy={args.test_time_policy}"
              + (f"(+{args.test_time_buffer}s)" if args.test_time_policy == "gurobi_time_plus_buffer" else ""))
        if args.framework == "coral":
            mx = args.coral_max_seconds
            if str(mx).strip().lower() == "auto":
                mx = f"auto(×{args.coral_attempts_budget_multiplier})"
            print(f"  coral: attempts={args.coral_attempts} max_seconds={mx} "
                  f"runtime={args.coral_agent_runtime} agents={args.coral_agent_count} "
                  f"gateway={args.coral_gateway}")
            hb_parts = []
            if args.coral_heartbeat_reflect_every > 0:
                hb_parts.append(f"reflect/{args.coral_heartbeat_reflect_every}")
            if args.coral_heartbeat_pivot_every > 0:
                hb_parts.append(f"pivot/{args.coral_heartbeat_pivot_every}")
            if args.coral_heartbeat_consolidate_every > 0:
                hb_parts.append(f"consolidate/{args.coral_heartbeat_consolidate_every}")
            print(f"  coral heartbeat: {', '.join(hb_parts) if hb_parts else 'off'}")
    print(f"Data dir: {data_dir}")
    print(f"GRB_LICENSE_FILE: {gurobi_license or os.environ.get('GRB_LICENSE_FILE') or '<not set>'}")

    def _process_paper(paper_id: str) -> tuple:
        """Run all selected modes for one paper. Used both serially and as a
        ThreadPoolExecutor task when --paper-workers > 1. Catches exceptions so
        one paper's failure doesn't kill the whole run."""
        try:
            prompt = build_prompt(paper_id)
            if per_model_modes:
                for model in models:
                    print(f"\n=== {paper_id} | {eval_core.get_model_short_name(model)} | {per_model_modes} ===")
                    run_per_model_modes(args, run_id, config, paper_id, model, prompt)
            if "self_evolve" in args.modes:
                label = eval_core.get_model_short_name(primary_model)
                if primary_model != secondary_model:
                    label = f"{label} / {eval_core.get_model_short_name(secondary_model)}"
                print(f"\n=== {paper_id} | {label} | self_evolve:{args.framework} ===")
                run_self_evolve_mode(args, run_id, config, paper_id, prompt, primary_model, secondary_model)
            return paper_id, "ok", None
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n!!! [paper={paper_id}] FAILED: {type(e).__name__}: {e}\n{tb}",
                  file=sys.stderr, flush=True)
            return paper_id, "failed", f"{type(e).__name__}: {e}"

    n_paper_workers = max(1, args.paper_workers)
    if n_paper_workers <= 1 or len(paper_ids) <= 1:
        statuses = [_process_paper(pid) for pid in paper_ids]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"\n[paper-pool] running {len(paper_ids)} papers with {n_paper_workers} workers")
        statuses = []
        with ThreadPoolExecutor(max_workers=n_paper_workers) as ex:
            futs = {ex.submit(_process_paper, pid): pid for pid in paper_ids}
            for fut in as_completed(futs):
                statuses.append(fut.result())

    failed = [(pid, err) for pid, st, err in statuses if st != "ok"]
    if failed:
        print(f"\n[paper-pool] {len(failed)}/{len(paper_ids)} papers FAILED:", file=sys.stderr)
        for pid, err in failed:
            print(f"  - {pid}: {err}", file=sys.stderr)

    print(f"\nDev results:  {os.path.join(ROOT_DIR, 'eval', 'eval_dev_results_openevolve.csv')}")
    print(f"Test results: {os.path.join(ROOT_DIR, 'eval', 'eval_test_results_openevolve.csv')}")
    print(f"API cost:     {os.path.join(ROOT_DIR, 'eval', 'self_evolve_api_cost.csv')}")


if __name__ == "__main__":
    main()
