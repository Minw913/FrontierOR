"""
Path resolvers for the post-migration paper folder layout.

Layout (per docs/self_evolve_agent_eval_plan.md, applied 2026-04):

    {paper_dir}/
    ├── instance/                  tiny_instance.json,  large_instance_1.json
    ├── gurobi_solution/           tiny_solution.json,  large_solution_1.json
    ├── gurobi_solution_log/       tiny_log.jsonl,      large_log_1.jsonl
    ├── gurobi_feasi_result/       tiny_feasi_result.json, large_feasi_result_1.json
    ├── efficient_solution/        tiny_solution.json,  large_solution_1.json
    ├── efficient_feasi_result/    tiny_feasi_result.json, large_feasi_result_1.json
    ├── instance_2..9.json         (legacy, untouched)
    ├── gurobi_solution_2..9.json  (legacy, untouched)
    └── ...

Instance names are categorical: "tiny", "large_1", "large_2", ...
The "tiny" slot is the smoke-test instance.
"large_N" slots are main-test instances; "large_1" is the smallest of the larges.

Use ``DEFAULT_INSTANCES`` for the canonical eval set.
"""
from __future__ import annotations
import os
from typing import Iterable, List, Optional

# Canonical eval set. The on-disk dataset ships large_11..large_51; the default
# below stays at "tiny" only so a no-flag run does a fast smoke test. Callers
# that want the full large sweep pass --instances explicitly.
DEFAULT_INSTANCES: List[str] = ["tiny"]

# --- self_evolve instance presets ------------------------------------------
# Stage1:  quick feasibility/ballpark gate. Binary score (pass iff |gap| <= threshold).
# Dev set: stage2 fitness signal — drives the evolve loop's combined_score.
# Test:    post-evolve held-out evaluation on the final best program (not used
#          inside the evolve loop). Empty → reuse dev metrics for reporting
#          (avoids a redundant final re-run when test would equal dev).
#
# SELF_EVOLVE_STAGE2_INSTANCES is None by default to mean "auto-pick the
# instance with the median Gurobi τ_g per paper" (representative, avoids the
# worst-case slowest). The resolution happens per paper in
# run_self_evolve_mode(); see
# test_time_self_evolution.scoring.building_blocks.pick_median_tau_g_instance.
# Pass '--dev-set max' (or 'max_tau_g' / 'auto') to opt into the largest-τ_g pick.
SELF_EVOLVE_STAGE1_INSTANCES: List[str] = ["tiny"]
SELF_EVOLVE_STAGE2_INSTANCES: Optional[List[str]] = None                              # CLI: --dev-set (None = auto-pick median-τ_g)
SELF_EVOLVE_TEST_INSTANCES: List[str] = ["large_21", "large_31", "large_41", "large_51"]  # CLI: --test-set


def is_valid_instance_name(name: str) -> bool:
    if name == "tiny":
        return True
    if name.startswith("large_"):
        suffix = name.split("_", 1)[1]
        return suffix.isdigit() and int(suffix) >= 1
    return False


def _split_name(name: str) -> tuple[str, str]:
    """Return (slot, n_suffix) where slot ∈ {'tiny','large'}, n_suffix is '' or '_N'."""
    if name == "tiny":
        return "tiny", ""
    if name.startswith("large_"):
        n = name.split("_", 1)[1]
        return "large", f"_{n}"
    raise ValueError(f"Unknown instance name: {name!r}")


def _basename(name: str, suffix: str, ext: str) -> str:
    """Build the per-instance filename, e.g. ('tiny','solution','json') -> 'tiny_solution.json'."""
    slot, n = _split_name(name)
    return f"{slot}_{suffix}{n}.{ext}"


def instance_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "instance", _basename(name, "instance", "json"))


def gurobi_solution_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "gurobi_solution", _basename(name, "solution", "json"))


def gurobi_log_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "gurobi_solution_log", _basename(name, "log", "jsonl"))


def gurobi_feasi_result_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "gurobi_feasi_result", _basename(name, "feasi_result", "json"))


def efficient_solution_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "efficient_solution", _basename(name, "solution", "json"))


def efficient_log_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "efficient_solution_log", _basename(name, "log", "jsonl"))


def efficient_feasi_result_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "efficient_feasi_result", _basename(name, "feasi_result", "json"))


def evolved_efficient_solution_path(paper_dir: str, name: str) -> str:
    return os.path.join(paper_dir, "evolved_efficient_solution", _basename(name, "solution", "json"))


def parse_instances_arg(values: Iterable[str]) -> List[str]:
    """Validate and normalize a list of instance names from CLI."""
    out = []
    for v in values:
        if not is_valid_instance_name(v):
            raise ValueError(
                f"Invalid instance name {v!r}. Must be 'tiny' or 'large_N' (N>=1)."
            )
        out.append(v)
    return out


# --- Survey-script helper for the legacy 1..10 iteration model ---
# Some diagnostic scripts (scripts/check_results/survey_*.py) still iterate
# `for i in range(1, 11)`. They use this resolver to find the instance file
# regardless of whether it's in the new subdir (idx 1, 10) or at the paper
# folder root (idx 2..9).

_LEGACY_IDX_TO_NAME = {1: "tiny", 10: "large_1"}


def instance_path_for_legacy_idx(paper_dir: str, idx: int) -> str:
    """Resolve `instance_<idx>.json`-style path for either layout.

    - idx in {1, 10}  → new subdir (instance/tiny_instance.json or instance/large_instance_1.json)
    - idx in {2..9}   → legacy paper_dir root (instance_<idx>.json)
    """
    if idx in _LEGACY_IDX_TO_NAME:
        return instance_path(paper_dir, _LEGACY_IDX_TO_NAME[idx])
    return os.path.join(paper_dir, f"instance_{idx}.json")
