"""Direction-aware building blocks shared across scoring schemes.

All functions here are pure (no side effects beyond filesystem reads).
Importable without pulling in one_shot_eval.py or any OpenEvolve internals.
"""

from __future__ import annotations

import csv
import json
import os
from typing import List, Optional, Tuple

# Root of the repo. Used to locate gurobi_results_*.csv files and the
# per-paper gurobi_solution_log directories.
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ------------------------------ gap + beat ------------------------------

NEAR_ZERO_REF = 1e-3


def _scaled_denom(ref: float, obj: float) -> float:
    # When |ref| is a reasonable scale, use it (classic relative-gap behavior).
    # When |ref| is near-zero, the relative formula blows up; fall back to
    # max(|obj|, NEAR_ZERO_REF) so the result stays bounded and scale-aware.
    if abs(ref) >= NEAR_ZERO_REF:
        return abs(ref)
    return max(abs(obj), NEAR_ZERO_REF)


def signed_relative_gap(obj: Optional[float], ref: Optional[float],
                        direction: str = "min", eps: float = 1e-10) -> Optional[float]:
    """Non-negative direction-aware relative gap.

    Returns 0.0 iff obj matches or beats ref (direction-aware).
    Returns a positive value iff obj is worse than ref.

    For min problems: gap = max(0, (obj - ref) / D)
    For max problems: gap = max(0, (ref - obj) / D)

    D = |ref| when |ref| >= NEAR_ZERO_REF, else max(|obj|, NEAR_ZERO_REF).
    The fallback keeps the value bounded when ref is tiny.

    Returns None iff either argument is None. ``eps`` kept for API compatibility.
    """
    if obj is None or ref is None:
        return None
    denom = max(_scaled_denom(ref, obj), eps)
    if direction == "max":
        raw = (ref - obj) / denom
    else:
        raw = (obj - ref) / denom
    return max(0.0, raw)


def signed_beat(obj: Optional[float], ref: Optional[float],
                direction: str = "min", eps: float = 1e-10) -> float:
    """Non-negative amount by which obj strictly beats ref. 0.0 iff tie-or-worse.

    For min: beat > 0 iff obj < ref  →  beat = (ref - obj) / D
    For max: beat > 0 iff obj > ref  →  beat = (obj - ref) / D

    D uses the same near-zero guard as signed_relative_gap.
    """
    if obj is None or ref is None:
        return 0.0
    denom = max(_scaled_denom(ref, obj), eps)
    if direction == "max":
        raw = (obj - ref) / denom
    else:
        raw = (ref - obj) / denom
    return max(0.0, raw)


# ------------------------------ log parsing ------------------------------

def read_log_entries(log_path: Optional[str]) -> List[Tuple[float, float]]:
    """Read (time, objective_value) entries from a convergence JSONL log.

    Entries are sorted by time ascending. Malformed lines are silently skipped.
    Returns [] if the log doesn't exist or is empty.
    """
    if not log_path or not os.path.exists(log_path):
        return []
    entries: List[Tuple[float, float]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                t = float(d.get("time"))
                o = float(d.get("objective_value"))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            entries.append((t, o))
    entries.sort(key=lambda e: e[0])
    return entries


def best_obj_in_log(log_entries: List[Tuple[float, float]],
                    up_to: float, direction: str = "min") -> Optional[float]:
    """Best obj in log entries whose time <= up_to. None if no entries in range.

    For direction="min", returns smallest obj; for "max", largest.
    """
    filtered = [o for (t, o) in log_entries if t <= up_to]
    if not filtered:
        return None
    return max(filtered) if direction == "max" else min(filtered)


def first_time_reaching_ref(log_entries: List[Tuple[float, float]],
                            ref: float,
                            eps: float,
                            direction: str = "min") -> Optional[float]:
    """Earliest time at which best-so-far reaches gap <= eps vs ref.

    Walks entries in time order, tracks best-so-far, returns first time when the
    running best satisfies signed_relative_gap <= eps. Returns None if never.
    """
    best_so_far = None
    for (t, o) in log_entries:
        if best_so_far is None or (o < best_so_far if direction == "min" else o > best_so_far):
            best_so_far = o
        g = signed_relative_gap(best_so_far, ref, direction)
        if g is not None and g <= eps:
            return t
    return None


# ------------------------------ gurobi_time lookup ------------------------------

def gurobi_log_path_for(paper_id: str, instance: str) -> Optional[str]:
    """Path to Gurobi's convergence log for (paper, instance)."""
    try:
        # Local import to avoid pulling instance_paths into scoring module at import time.
        import sys
        utils_dir = os.path.join(ROOT_DIR, "scripts", "utils")
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        from instance_paths import gurobi_log_path  # noqa: E402
    except Exception:
        return None
    paper_dir = os.path.join(ROOT_DIR, "frontier-or", paper_id)
    if not os.path.isdir(paper_dir):
        return None
    try:
        return gurobi_log_path(paper_dir, instance)
    except Exception:
        return None


def _data_root() -> str:
    """Inline minimal copy of one_shot_eval.get_data_dir() — kept here so
    building_blocks stays free of the one_shot_eval import (per the module
    docstring)."""
    override = os.environ.get("FRONTIER_OR_DATA_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(ROOT_DIR, "frontier-or")


def list_large_instances(paper_id: str) -> List[str]:
    """Return canonical names of all ``large_*`` instances present on disk
    for a paper, sorted by integer suffix.

    Scans ``<paper_dir>/instance/large_instance_*.json``. Used by the
    auto-pick logic for ``--dev-set`` (see ``pick_max_tau_g_instance``).
    """
    import glob
    paper_dir = os.path.join(_data_root(), paper_id)
    pattern = os.path.join(paper_dir, "instance", "large_instance_*.json")
    out: List[Tuple[int, str]] = []
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        # base = "large_instance_<N>.json"
        suffix = base[len("large_instance_"):-len(".json")]
        if suffix.isdigit():
            out.append((int(suffix), f"large_{suffix}"))
    out.sort()
    return [name for _, name in out]


def pick_max_tau_g_instance(paper_id: str) -> Optional[str]:
    """Pick the dev-set instance with the largest Gurobi solve time τ_g.

    Used as the default for ``--dev-set`` so each paper gets its hardest
    available instance as the evolve-loop fitness signal. Returns ``None``
    iff no large instance has a recorded τ_g (caller should then fail with
    a clear "pass --dev-set explicitly" message).
    """
    timed: List[Tuple[float, str]] = []
    for inst in list_large_instances(paper_id):
        tau_g = lookup_gurobi_time(paper_id, inst)
        if tau_g is not None:
            timed.append((float(tau_g), inst))
    if not timed:
        return None
    timed.sort(reverse=True)
    return timed[0][1]


def pick_median_tau_g_instance(paper_id: str) -> Optional[str]:
    """Pick the dev-set instance with the median Gurobi solve time τ_g.

    Alternative to ``pick_max_tau_g_instance`` for users who want a more
    representative dev instance (avoids the worst-case bottleneck of always
    picking the slowest one). For odd-count, returns the strict middle; for
    even-count, returns the lower of the two middle τ_g values (favoring
    faster wall-time over the upper-half one). Returns ``None`` iff no
    large instance has a recorded τ_g.
    """
    timed: List[Tuple[float, str]] = []
    for inst in list_large_instances(paper_id):
        tau_g = lookup_gurobi_time(paper_id, inst)
        if tau_g is not None:
            timed.append((float(tau_g), inst))
    if not timed:
        return None
    timed.sort()  # ascending by τ_g
    n = len(timed)
    # Lower median for even n (favors faster wall-time); strict middle for odd
    return timed[(n - 1) // 2][1]


def lookup_gurobi_time(paper_id: str, instance: str) -> Optional[float]:
    """Look up Gurobi's solve_time (τ_g) for (paper, instance).

    Fallback chain:
      1. ``gurobi_results_<suffix>.csv`` per-instance file in repo root
         (suffix: "tiny" / "11" / "21" / "31" / "41" / "51"). Schema:
         ``paper_id, instance, gurobi_feasibility_status, gurobi_solution,
         solution_status, gurobi_time, time_limit, failure_reason, failure_error``.
         The ``gurobi_time`` field is the authoritative τ_g (Gurobi's wall to
         optimal, or its time_limit when it timed out at incumbent).
      2. ``gurobi_solution_log/<inst>_log.jsonl`` LAST-entry time —
         **approximate**: this only records incumbent-improvement events, so
         the last entry's time is "time of final improvement", not the full
         Gurobi run wall. Use only when the CSV is missing.

    Returns None iff none of the sources have data.
    """
    suffix = _instance_suffix(instance)
    if suffix is not None:
        csv_path = os.path.join(ROOT_DIR, f"gurobi_results_{suffix}.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("paper_id") == paper_id and row.get("instance") == instance:
                            raw = row.get("gurobi_time")
                            if raw:
                                try:
                                    return float(raw)
                                except ValueError:
                                    pass
            except Exception:
                pass

    log_path = gurobi_log_path_for(paper_id, instance)
    if log_path:
        entries = read_log_entries(log_path)
        if entries:
            return entries[-1][0]

    return None


def _instance_suffix(instance: str) -> Optional[str]:
    """Map instance canonical name → CSV-file suffix.

    ``"tiny"`` → ``"tiny"``;  ``"large_21"`` → ``"21"``. Returns None for
    names that don't match either pattern (caller falls through to log).
    """
    if instance == "tiny":
        return "tiny"
    if instance.startswith("large_"):
        rest = instance[len("large_"):]
        if rest.isdigit():
            return rest
    return None
