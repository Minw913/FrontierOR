"""OpenEvolve-specific orchestration.

Functions here are the OpenEvolve runner / glue: spawning the OpenEvolve CLI,
preparing its env, writing its YAML config, locating its best program, and
the top-level :func:`run_self_evolve` orchestrator. Framework-agnostic
helpers live in :mod:`self_evolving_frameworks.eval_modes`.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

import yaml

import one_shot_eval as eval_core

from self_evolving_frameworks import eval_modes
from self_evolving_frameworks.openevolve.preflight import preflight_environment_check


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OPENEVOLVE_BASE_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "openevolve.yaml")


def prepare_openevolve_env(base_env: Optional[Dict[str, str]], config: Dict) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    key = env.get("OPENROUTER_API_KEY") or config.get("OPENROUTER_API_KEY")
    if key:
        env["OPENROUTER_API_KEY"] = key
        env["OPENAI_API_KEY"] = key
    if os.environ.get("GRB_LICENSE_FILE"):
        env["GRB_LICENSE_FILE"] = os.environ["GRB_LICENSE_FILE"]
    # Force the subprocess to import openevolve from external/openevolve/ (the
    # vendored, patched copy) rather than whatever happens to be first on the
    # global sys.path. Without this, a stray clone elsewhere in ~/Code wins
    # and any patches under external/ are silently ignored.
    local_oe = os.path.join(ROOT_DIR, "external", "openevolve")
    if os.path.isdir(local_oe):
        prev_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = local_oe + (os.pathsep + prev_pp if prev_pp else "")
    return env


def openevolve_runner_path() -> str:
    local_runner = os.path.join(ROOT_DIR, "external", "openevolve", "openevolve-run.py")
    if os.path.exists(local_runner):
        return local_runner
    return "openevolve-run.py"


def find_best_openevolve_program(run_dir: str, fallback_path: str) -> str:
    final_best = os.path.join(run_dir, "best", "best_program.py")
    if os.path.exists(final_best):
        return final_best
    candidates = []
    for dirpath, _, filenames in os.walk(run_dir):
        for filename in filenames:
            lower = filename.lower()
            if lower.endswith(".py") and ("best" in lower or "program" in lower):
                candidates.append(os.path.join(dirpath, filename))
    if not candidates:
        return fallback_path
    candidates.sort(key=lambda path: (0 if "best" in os.path.basename(path).lower() else 1, path))
    return candidates[0]


def write_openevolve_config(config_path: str, primary_model: str, secondary_model: Optional[str] = None):
    """Load configs/openevolve.yaml as base and inject primary/secondary model fields.

    Also resolves any relative ``prompt.template_dir`` against the SOURCE
    config's parent (configs/) so that when this copy is written into the
    per-paper output dir, openevolve's loader (which resolves template_dir
    relative to the config it loaded) still finds the right directory.
    """
    if secondary_model is None:
        secondary_model = primary_model
    if not os.path.exists(OPENEVOLVE_BASE_CONFIG_PATH):
        raise FileNotFoundError(
            f"OpenEvolve base config not found at {OPENEVOLVE_BASE_CONFIG_PATH}"
        )
    with open(OPENEVOLVE_BASE_CONFIG_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    llm = base.setdefault("llm", {})
    llm["primary_model"] = primary_model
    llm["secondary_model"] = secondary_model
    prompt_block = base.get("prompt") or {}
    tpl_dir = prompt_block.get("template_dir")
    if tpl_dir and not os.path.isabs(tpl_dir):
        src_dir = os.path.dirname(OPENEVOLVE_BASE_CONFIG_PATH)
        prompt_block["template_dir"] = os.path.abspath(os.path.join(src_dir, tpl_dir))
        base["prompt"] = prompt_block
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, sort_keys=False)


def run_openevolve(
    initial_program: str,
    evaluator_path: str,
    config_path: str,
    output_dir: str,
    iterations: int,
    env: Dict[str, str],
    resume_from: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        openevolve_runner_path(),
        initial_program,
        evaluator_path,
        "--config",
        config_path,
        "--iterations",
        str(iterations),
        "--output",
        output_dir,
    ]
    if resume_from:
        cmd += ["--checkpoint", resume_from]
    subprocess.run(cmd, check=True, cwd=ROOT_DIR, env=env)
    return find_best_openevolve_program(output_dir, initial_program)


def latest_checkpoint_dir(run_dir: str) -> Optional[str]:
    """Return the highest-numbered checkpoint inside {run_dir}/checkpoints/, or None."""
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None
    entries = []
    for name in os.listdir(ckpt_root):
        if name.startswith("checkpoint_"):
            try:
                n = int(name.rsplit("_", 1)[1])
            except ValueError:
                continue
            entries.append((n, os.path.join(ckpt_root, name)))
    if not entries:
        return None
    return max(entries)[1]


def read_latest_best_info(run_dir: str) -> Optional[Dict]:
    """Find the highest-numbered checkpoint that has a best_program_info.json
    and return its parsed content. None if no such checkpoint exists yet.
    """
    import json as _json
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None
    nums = []
    for name in os.listdir(ckpt_root):
        if name.startswith("checkpoint_"):
            try:
                nums.append(int(name.rsplit("_", 1)[1]))
            except ValueError:
                continue
    for n in sorted(nums, reverse=True):
        info_path = os.path.join(ckpt_root, f"checkpoint_{n}", "best_program_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    return _json.load(f)
            except Exception:
                continue
    return None


def augment_results_with_staged_qte(
    results: Dict[str, Dict],
    paper_id: str,
    output_dir: str,
    stage_boundary: float = 0.01,
) -> Dict[str, Dict]:
    """Run :class:`StagedQteScorer` on each instance's convergence log under
    ``output_dir`` and merge the score + stage decomposition into ``results``.

    Used by the post-evolve test-set eval path, where results come from
    :func:`evaluate_best_on_test_set` (one-shot style) and lack the
    flattened ``inst_<name>_*`` staged_qte fields that stage2 writes natively.

    ``stage_boundary`` must match the boundary used during evolution so the
    test-set rescore is on the same scale as the dev-set fitness.

    Modifies ``results`` in place and returns it. Logs follow the
    ``log_<inst>.jsonl`` convention (one_shot_eval._run_one_instance:708).
    """
    from self_evolving_frameworks.scoring import get_scorer
    from self_evolving_frameworks.scoring.base import ScoreContext
    from self_evolving_frameworks.scoring.building_blocks import lookup_gurobi_time

    if not results:
        return results
    scorer = get_scorer("staged_qte", stage_boundary=stage_boundary)
    direction = eval_core.get_paper_direction(paper_id)

    for inst, r in results.items():
        if r is None:
            continue
        log_path = os.path.join(output_dir, f"log_{inst}.jsonl")
        gurobi_obj = r.get("gurobi_obj")
        gurobi_time = lookup_gurobi_time(paper_id, inst)
        time_limit = int(r.get("solve_time") or 0) or int(gurobi_time or 0)
        ctx = ScoreContext(
            time_limit=time_limit,
            gurobi_time=gurobi_time,
            gurobi_obj=gurobi_obj,
            direction=direction,
            log_path=log_path,
            paper_id=paper_id,
            instance=inst,
        )
        score, dbg = scorer.score_instance(r, ctx)
        r["score"] = round(float(score), 6)
        r["stage_id"] = float(dbg.get("stage_id", 0))
        r["quality_part"] = float(dbg.get("quality_part", 0.0))
        r["speed_part"] = float(dbg.get("speed_part", 0.0))
        r["signed_gap"] = float(dbg.get("signed_gap", 1.0))
        r["beat_amount"] = float(dbg.get("beat_amount", 0.0))
        r["beat_gurobi_flag"] = 1.0 if dbg.get("beat_gurobi") else 0.0
        r["matched_flag"] = 1.0 if dbg.get("matched") else 0.0
    return results


def reconstruct_results_from_metrics(metrics: Dict, instances: List[str]) -> Dict[str, Dict]:
    """Rebuild per-instance result dicts from the flattened ``inst_<name>_*``
    fields produced by :func:`evaluator.evaluate_stage2`.

    Returns a dict keyed by instance name, each value shaped like the result
    dicts normally produced by ``eval_core.evaluate_candidate_code`` (so the
    unified self-evolve CSV writer in ``eval_modes.write_self_evolve_results``
    can consume them). Instances with no matching metrics get a placeholder
    entry.
    """
    reconstructed: Dict[str, Dict] = {}
    for inst in instances:
        p = f"inst_{inst}"
        f_val = metrics.get(f"{p}_feasible")
        if f_val == 1.0:
            feasible, status, fail_reason = True, "pass", None
        elif f_val == 0.0:
            feasible, status, fail_reason = False, "fail", "infeasible"
        else:
            feasible, status, fail_reason = None, "missing", "no_metric_in_checkpoint"

        reconstructed[inst] = {
            "status": status,
            "fail_reason": fail_reason,
            "retries": 0,
            "feasible": feasible,
            "gap": metrics.get(f"{p}_gap"),
            "llm_obj": metrics.get(f"{p}_obj"),
            "gurobi_obj": metrics.get(f"{p}_gurobi_obj"),
            "solve_time": metrics.get(f"{p}_time"),
            "aocc": metrics.get(f"{p}_aocc"),
            "error": None,
            # Surface per-instance score + staged_qte fields when stage2
            # used staged_qte scorer (each evaluator wrote inst_<name>_* keys;
            # see evaluator.py:_build_stage2_*).
            "score": metrics.get(f"{p}_score"),
            "stage_id": metrics.get(f"{p}_stage_id"),
            "quality_part": metrics.get(f"{p}_quality_part"),
            "speed_part": metrics.get(f"{p}_speed_part"),
            "signed_gap": metrics.get(f"{p}_signed_gap"),
            "beat_amount": metrics.get(f"{p}_beat_amount"),
            "beat_gurobi_flag": metrics.get(f"{p}_beat_gurobi"),
            "matched_flag": metrics.get(f"{p}_matched"),
        }
    return reconstructed


def _try_reuse_oneshot_seed(paper_id: str, model_name: str, seed_dir: str) -> Optional[str]:
    """Try to copy ``eval/eval_papers/<paper>/<model_short>/code_attempt0.py``
    into ``seed_dir/code.py`` and return its path. Returns None if the one-shot
    artifact doesn't exist (caller should fall back to live LLM generation).

    The path encodes the model_short, so reuse is **never** cross-model — if
    self-evolve runs with deepseek-r1, only deepseek-r1's v0 can be reused.

    Saves 1 LLM call per paper (~$0.005-0.05 + 30-60s wall-time) when a one-shot
    run with the same model has already populated eval_papers/.
    """
    short = eval_core.get_model_short_name(model_name)
    src = os.path.join(
        ROOT_DIR, "eval", "eval_papers", paper_id, short, "code_attempt0.py",
    )
    if not os.path.exists(src) or os.path.getsize(src) == 0:
        return None
    os.makedirs(seed_dir, exist_ok=True)
    dst = os.path.join(seed_dir, "code.py")
    shutil.copyfile(src, dst)
    # Mark provenance for debugging — the seed wasn't generated by this run's LLM.
    provenance = os.path.join(seed_dir, "_seed_source.txt")
    with open(provenance, "w", encoding="utf-8") as f:
        f.write(f"reused from one-shot v0: {src}\n")
    print(f"[reuse-oneshot] {paper_id}/{short}: copied one-shot v0 → {dst}")
    return dst


def _read_seed_dev_results(oe_run_dir: str, dev_instances: List[str]) -> Dict[str, Dict]:
    """Reconstruct the seed program's per-dev-instance results from the latest
    OpenEvolve checkpoint, for the ``first_status`` column of the unified CSV.

    The seed is identified as the program with ``generation == 0`` (or
    ``parent_id is None``) in ``checkpoints/checkpoint_<latest>/programs/``.
    Its metrics dict has ``inst_<name>_*`` flattened fields written by the
    stage 2 evaluator. Returns ``{}`` if seed can't be located (e.g. evicted
    before any checkpoint, which shouldn't happen at checkpoint_interval=1).
    """
    ckpt = latest_checkpoint_dir(oe_run_dir)
    if not ckpt:
        return {}
    seed_meta = None
    for fp in glob.glob(os.path.join(ckpt, "programs", "*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("generation") == 0 or d.get("parent_id") is None:
            seed_meta = d
            break
    if seed_meta is None or "metrics" not in seed_meta:
        return {}
    return reconstruct_results_from_metrics(seed_meta["metrics"], list(dev_instances))


def run_self_evolve(
    run_id: str,
    paper_id: str,
    primary_model: str,
    prompt: str,
    config: Dict,
    stage1_instances: List[str],
    stage2_instances: List[str],
    test_instances: List[str],
    stage1_time_limit: int,
    stage2_time_limit: int,
    test_time_limit: int,
    stage1_gap_threshold: float,
    iterations: int,
    exec_mode: str,
    exec_cfg: Dict,
    t_max,
    secondary_model: Optional[str] = None,
    resume: bool = False,
    stage2_scorer: str = "staged_qte",
    stage2_stage_boundary: float = 0.01,
    stage2_time_policy: str = "uniform",
    stage2_time_buffer: int = 0,
    test_time_policy: str = "uniform",
    test_time_buffer: int = 0,
    test_instance_workers: int = 4,
):
    preflight_environment_check(
        paper_id=paper_id,
        stage1_instances=stage1_instances,
        stage2_instances=stage2_instances,
        test_instances=test_instances,
        config=config,
        primary_model=primary_model,
        stage2_time_policy=stage2_time_policy,
        test_time_policy=test_time_policy,
        stage2_scorer=stage2_scorer,
    )
    if secondary_model is None:
        secondary_model = primary_model
    primary_name = eval_core.get_model_short_name(primary_model)
    secondary_name = eval_core.get_model_short_name(secondary_model)
    model_name = primary_name if primary_name == secondary_name else f"{primary_name}__{secondary_name}"
    base_dir = eval_modes.mode_run_dir(run_id, "openevolve", paper_id, model_name)

    # ``test_instances`` now governs the final evaluation exclusively:
    #   - empty (default) → no post-evolve re-run; best-program metrics are
    #                       reconstructed from OpenEvolve's last checkpoint
    #                       (requires the stage2 evaluator to have flattened
    #                       ``inst_<name>_*`` fields into its return dict)
    #   - non-empty       → run a clean final eval on exactly those instances
    final_instances: List[str] = list(test_instances)
    # The set of instances whose per-instance rows we will emit to the CSVs;
    # always the stage2 set if no explicit test set is given.
    reporting_instances: List[str] = final_instances or list(stage2_instances)
    selection_instance = stage1_instances[0] if stage1_instances else (
        reporting_instances[0] if reporting_instances else "tiny"
    )

    seed_dir = os.path.join(base_dir, "seed")
    seed_code_path = os.path.join(seed_dir, "code.py")
    resume_checkpoint = None
    resume_remaining_iters = iterations  # how many more iters to run; defaults to full budget
    if resume:
        if not os.path.exists(seed_code_path):
            raise RuntimeError(
                f"Cannot resume run_id={run_id}: seed code not found at {seed_code_path}"
            )
        resume_checkpoint = latest_checkpoint_dir(os.path.join(base_dir, "openevolve_run"))
        if resume_checkpoint is None:
            raise RuntimeError(
                f"Cannot resume run_id={run_id}: no checkpoints found under "
                f"{os.path.join(base_dir, 'openevolve_run', 'checkpoints')}"
            )
        # OpenEvolve's --iterations is a delta from the checkpoint, so passing
        # the full --iterations target naively makes resume *overshoot*
        # (target=30, checkpoint=30 -> runs 30 *more*, ending at 60). Convert
        # to "remaining budget = target - already_done", clamped at 0.
        try:
            already_done = int(os.path.basename(resume_checkpoint).rsplit("_", 1)[1])
        except (ValueError, IndexError):
            already_done = 0
        resume_remaining_iters = max(0, iterations - already_done)
        print(f"[resume] reusing seed at {seed_code_path}")
        print(f"[resume] continuing from checkpoint {resume_checkpoint} "
              f"(already_done={already_done}, target={iterations}, "
              f"remaining={resume_remaining_iters})")
        generated = {
            "status": "ok",
            "code_path": seed_code_path,
            "usage": {},
        }
    else:
        reuse_flag = os.environ.get("EFFICIENT_OR_REUSE_SEED_IF_EXISTS") == "1"
        if reuse_flag and os.path.exists(seed_code_path) and os.path.getsize(seed_code_path) > 0:
            # Same run was started before — keep the previously-generated seed.
            print(f"[reuse-seed] using existing seed at {seed_code_path} (skip LLM generation)")
            generated = {"status": "ok", "code_path": seed_code_path, "usage": {}}
        else:
            # Try to reuse one-shot's code_attempt0.py for (paper, model). This
            # saves 1 LLM call per paper since self-evolve's seed prompt is
            # identical to one-shot's v0 prompt (both via eval_core.build_prompt).
            # Falls back to live generation when one-shot data is unavailable.
            reused = _try_reuse_oneshot_seed(paper_id, primary_model, seed_dir)
            if reused is not None:
                generated = {"status": "ok", "code_path": reused, "usage": {}}
            else:
                generated = eval_core.generate_candidate_code(
                    prompt, config, primary_model, seed_dir, candidate_id="seed", temperature=0.4
                )
    token_usage = generated.get("usage", {})
    if generated["status"] != "ok":
        fallback_instances = final_instances or stage1_instances or [selection_instance]
        results = {
            instance: {
                "status": "fail",
                "fail_reason": "generation_error",
                "feasible": None,
                "gap": None,
                "solve_time": None,
                "llm_obj": None,
                "gurobi_obj": None,
                "aocc": None,
                "error": generated.get("error"),
                "retries": 0,
            }
            for instance in fallback_instances
        }
        eval_modes.write_api_cost_row(run_id, "self_evolve", paper_id, primary_model, model_name, token_usage)
        return {"candidate_id": "seed_generation_failed", "results": results, "code_path": ""}

    env = prepare_openevolve_env(os.environ, config)
    env.update({
        "EFFICIENT_OR_ROOT": ROOT_DIR,
        "EFFICIENT_OR_PAPER_ID": paper_id,
        "EFFICIENT_OR_MODEL_NAME": model_name,
        "EFFICIENT_OR_STAGE1_INSTANCES": ",".join(stage1_instances),
        "EFFICIENT_OR_STAGE1_TIME_LIMIT": str(stage1_time_limit),
        "EFFICIENT_OR_STAGE1_GAP_THRESHOLD": str(stage1_gap_threshold),
        "EFFICIENT_OR_STAGE2_INSTANCES": ",".join(stage2_instances),
        "EFFICIENT_OR_STAGE2_TIME_LIMIT": str(stage2_time_limit),
        "EFFICIENT_OR_STAGE2_TIME_POLICY": stage2_time_policy,
        "EFFICIENT_OR_STAGE2_TIME_BUFFER": str(stage2_time_buffer),
        "EFFICIENT_OR_STAGE2_SCORER": stage2_scorer,
        "EFFICIENT_OR_STAGE2_STAGE_BOUNDARY": str(stage2_stage_boundary),
        # Legacy env vars preserved so the non-cascade ``evaluate()`` path still works.
        "EFFICIENT_OR_SELECTION_INSTANCE": selection_instance,
        "EFFICIENT_OR_INSTANCES": ",".join(stage1_instances) or selection_instance,
        "EFFICIENT_OR_TIME_LIMIT": str(stage1_time_limit),
        "EFFICIENT_OR_EXEC_MODE": exec_mode,
        "EFFICIENT_OR_T_MAX": "" if t_max is None else str(t_max),
        "EFFICIENT_OR_OUTPUT_DIR": os.path.join(base_dir, "openevolve_eval"),
    })
    for key, value in (exec_cfg or {}).items():
        env[f"EFFICIENT_OR_EXEC_{key.upper()}"] = str(value)

    oe_run_dir = os.path.join(base_dir, "openevolve_run")
    oe_config_path = os.path.join(base_dir, "openevolve_config.yaml")
    write_openevolve_config(oe_config_path, primary_model, secondary_model)
    evaluator_path = os.path.join(ROOT_DIR, "self_evolving_frameworks", "openevolve", "evaluator.py")
    if resume and resume_remaining_iters == 0:
        # Already at or past the target budget — skip evolution entirely.
        # Just resolve the best program from the existing checkpoints.
        print(f"[resume] checkpoint already meets target {iterations} iters; "
              f"skipping evolution, going straight to test phase")
        best_code_path = find_best_openevolve_program(oe_run_dir, generated["code_path"])
    else:
        iters_to_run = resume_remaining_iters if resume else iterations
        best_code_path = run_openevolve(
            generated["code_path"], evaluator_path, oe_config_path,
            oe_run_dir, iters_to_run, env,
            resume_from=resume_checkpoint,
        )

    final_dir = os.path.join(base_dir, "selected")

    # Read best program metadata once — needed for both the legacy fallback
    # below AND for the unified self-evolve CSV (iteration_found / generation).
    best_info = read_latest_best_info(oe_run_dir)
    iteration_found = best_info.get("iteration") if best_info else None
    generation = best_info.get("generation") if best_info else None

    if final_instances:
        # Explicit final eval on the user-supplied test_instances.
        # Per-instance time-policy + thread-pool fan-out are handled by the
        # shared helper (see :func:`eval_modes.evaluate_best_on_test_set`).
        final_results = eval_modes.evaluate_best_on_test_set(
            paper_id, model_name, best_code_path, final_instances,
            test_time_limit, test_time_policy, test_time_buffer,
            final_dir, exec_mode, exec_cfg, t_max,
            max_workers=test_instance_workers,
        )
        # Score test logs with staged_qte so test CSV gets the same stage
        # decomposition columns as dev. final_dir holds log_<inst>.jsonl.
        final_results = augment_results_with_staged_qte(
            final_results, paper_id, final_dir, stage_boundary=stage2_stage_boundary)
    else:
        # No explicit test set → reconstruct results from OpenEvolve's
        # latest checkpoint (relies on the evaluator flattening per-instance
        # fields as ``inst_<name>_*``). Avoids a redundant re-evaluation
        # when the user's test_instances would equal stage2.
        if best_info and best_info.get("metrics"):
            final_results = reconstruct_results_from_metrics(
                best_info["metrics"], reporting_instances
            )
            print(f"[final] reconstructed {len(final_results)} per-instance rows "
                  f"from checkpoint (best_id={best_info.get('id','?')[:8]}, "
                  f"iteration={best_info.get('iteration','?')})")
        else:
            final_results = {}
            print("[final] no best_program_info.json found; dev results CSV rows will be empty")
    selected_code = eval_modes.copy_selected_code(best_code_path, base_dir)

    # Schema matches one_shot_eval RESULTS_CSV_COLUMNS (minus retries & most
    # first_*; plus iteration_found + generation).
    if best_info and best_info.get("metrics"):
        dev_results_for_csv = reconstruct_results_from_metrics(
            best_info["metrics"], list(stage2_instances),
        )
    else:
        dev_results_for_csv = {}
    seed_dev_results = _read_seed_dev_results(oe_run_dir, list(stage2_instances))
    test_results_for_csv = final_results if final_instances else {}
    eval_modes.write_self_evolve_results(
        paper_id=paper_id,
        model_name=model_name,
        framework="openevolve",
        dev_instances=list(stage2_instances),
        dev_results=dev_results_for_csv,
        dev_seed_results=seed_dev_results,
        test_instances=list(test_instances),
        test_results=test_results_for_csv,
        iteration_found=iteration_found,
        generation=generation,
        run_id=run_id,
    )

    eval_modes.write_api_cost_row(
        run_id, "self_evolve", paper_id, primary_model, model_name, token_usage,
        note=(
            f"Seed generation only (primary={primary_name}, secondary={secondary_name}); "
            "OpenEvolve internal LLM usage is tracked by OpenEvolve logs."
        ),
    )
    return {
        "candidate_id": "openevolve_best",
        "results": final_results,
        "code_path": selected_code,
    }
