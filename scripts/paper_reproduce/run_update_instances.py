"""
Update/add/replace data instances for specified papers via Claude Code Agent.

This script sends per-paper update instructions to Claude, which creates
an `update_instances.py` script that reuses the existing `generate_instance()`
logic from `generate_multiple_instances.py`.

Modes:
  --mode replace    Replace infeasible instances (change seed, keep same scale)
  --mode add        Add new instances (interpolate/extrapolate from existing
                    INSTANCE_CONFIGS entries)
  --mode custom     Read --instructions-file JSON for free-form per-paper text
  --mode scale-up   Rotate the canonical 5 large instances (current j=1 → j=2,
                    marked [FAILED]) and generate fresh j=1 row at a LARGER
                    scale. Ceiling = the `l` tier of `(s, m, l)` thresholds
                    in the `[STRICT]` block — the largest realistic scale
                    surveyed for the paper. Sub-A papers (below ceiling) →
                    target = ceiling; Sub-B papers (already AT ceiling) →
                    target = ceiling × override-factor (default 1.5).
  --mode scale-down Mirror of scale-up for E-class papers that time-out
                    without finding any feasible solution. Floor = the `m`
                    tier of `(s, m, l)` thresholds — the lower bound of the
                    "large" band (i.e., upper bound of "medium"). If current
                    scale > m: target = floor; else target = current /
                    floor-divisor (default 1.5). Same rotate / sync /
                    smoke-test machinery as scale-up.
  --mode tiny-scale-down
                    Scale-DOWN the single `tiny_instance.json` file for
                    papers whose tiny solve time currently exceeds 100s
                    (target = tiny solve <100s). Initial target = the `s`
                    tier of `(s, m, l)` thresholds — upper bound of the
                    "small" band. On each smoke-test failure (gurobi >100s)
                    the target is divided by --tiny-scale-down-divisor
                    (default 1.5) and the agent rewrites tiny again. KEEP
                    semantics are INVERTED relative to large modes: a fast
                    solve passes (we want easy tiny), a slow solve triggers
                    further scale-down.

Usage examples:
    # Replace infeasible instance_2 for borndorfer2007
    python scripts/paper_reproduce/run_update_instances.py --mode replace \\
        --paper-id borndorfer2007 --targets '{"borndorfer2007": [2]}'

    # Add 2 larger instances beyond instance_10 for kong2021
    python scripts/paper_reproduce/run_update_instances.py --mode add \\
        --paper-id kong2021 --add-direction up --add-count 2

    # Use a custom instruction file for fine-grained control
    python scripts/paper_reproduce/run_update_instances.py \\
        --paper-id borndorfer2007 --instructions-file update_plan.json

    # Scale up C-class papers: rotate old j=1 → j=2 [FAILED], write new j=1
    python scripts/paper_reproduce/run_update_instances.py --mode scale-up \\
        --paper-id wangk2020 archetti2007 --workers 4 \\
        --scale-up-override-factor 1.5
"""

import argparse
import glob
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- Robust subprocess helpers (smoke-test child cleanup) -------------------
# subprocess.run(timeout=...) only SIGKILLs the immediate child. Gurobi can
# spawn helper threads / license-token children that survive, and if the
# orchestrator itself is killed the smoke workers are orphaned and continue
# writing /tmp/_smoke_*.json. The helpers below
#   1. put each child in its OWN session (new pgid) so we can signal the
#      whole subtree atomically;
#   2. set PR_SET_PDEATHSIG=SIGKILL on Linux so the kernel kills the child
#      the moment the orchestrator dies;
#   3. on timeout escalate SIGTERM → grace wait → SIGKILL over the pgid
#      instead of relying on Popen.kill() which only hits the leader.

_PR_SET_PDEATHSIG = 1


def _smoke_preexec():
    """Run in the child between fork() and execvp(): become a session leader
    (own pgid) and ask the kernel to SIGKILL us if the parent ever dies."""
    try:
        os.setsid()
    except OSError:
        pass
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        except Exception:
            pass


def _kill_proc_tree(proc, grace=5.0):
    """SIGTERM the child's pgid, give it `grace` seconds, then SIGKILL the
    pgid. Tolerates races where the process or group has already exited."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = proc.pid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=grace if sig == signal.SIGTERM else 2.0)
            return
        except subprocess.TimeoutExpired:
            continue


def _run_smoke_subprocess(cmd, timeout):
    """Popen-based replacement for subprocess.run(..., timeout=...).

    Returns (returncode, stdout, stderr, timed_out). On timeout we escalate
    SIGTERM → SIGKILL across the child's whole process group so any helper
    process Gurobi may have spawned dies with it.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        preexec_fn=_smoke_preexec,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err, False
    except subprocess.TimeoutExpired:
        _kill_proc_tree(proc, grace=5.0)
        try:
            out, err = proc.communicate(timeout=2.0)
        except Exception:
            out, err = "", ""
        return -signal.SIGKILL, out, err, True

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.utils.claude_utils import run_claude_capture, run_claude_tracked

# Judge-helper imports (lazy: only used when --judge-before-bump is set)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from patch_applier import apply_patches as _apply_patches
except ImportError:
    _apply_patches = None  # judge mode unavailable

PAPER_DATA_DIR = os.path.join(BASE_DIR, "data", "paper_data")
BENCH_DIR = os.path.join(BASE_DIR, "frontier-or")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "paper_reproduce", "prompt_update_instances.txt")

# Hardcoded paper_id list. If non-empty, this takes highest priority.
HARDCODED_PAPER_IDS: list[str] = []

# Hardcoded exclusion list. These IDs are always removed.
HARDCODED_PAPER_EXCLUDE_IDS: list[str] = []


def discover_paper_ids() -> list[str]:
    """Auto-discover paper_ids by scanning data/paper_data/*/*.pdf."""
    ids = set()
    for pdf in glob.glob(os.path.join(PAPER_DATA_DIR, "*", "*.pdf")):
        ids.add(os.path.basename(os.path.dirname(pdf)))
    return sorted(ids)


def max_existing_instance(paper_dir: str) -> int:
    """Return the largest x such that instance_x.json exists, or 0."""
    import re
    max_x = 0
    for path in glob.glob(os.path.join(paper_dir, "instance_*.json")):
        m = re.search(r"instance_(\d+)\.json$", os.path.basename(path))
        if m:
            max_x = max(max_x, int(m.group(1)))
    return max_x


def build_replace_instructions(paper_id: str, instance_ids: list[int]) -> str:
    """Build update instructions for replacing infeasible instances."""
    lines = [
        f"### Goal: Replace infeasible instance(s) for `{paper_id}`\n",
        "The following instance(s) were found to be infeasible (Gurobi returned INFEASIBLE or "
        "the feasibility check failed). Replace each one by generating a new instance with "
        "the **same scale and similar parameters** but a **different seed**.\n",
    ]
    for iid in instance_ids:
        lines.append(
            f"- **instance_{iid}**: Replace by outputting to `instance_{iid}.json`. "
            f"Use the same config as the original `INSTANCE_CONFIGS` entry for instance_{iid}, "
            f"but set `base_seed = {iid} * 100000 + 500000` to get a different random stream."
        )
    lines.append(
        "\nKeep the same scale label (small/medium/large) and approximate parameter values. "
        "The goal is only to find a feasible instance at the same difficulty level."
    )
    return "\n".join(lines)


def build_add_instructions(paper_id: str, direction: str, count: int,
                           paper_dir: str) -> str:
    """Build update instructions for adding new instances."""
    max_inst = max_existing_instance(paper_dir)
    lines = [
        f"### Goal: Add {count} new instance(s) for `{paper_id}`\n",
    ]
    if direction == "up":
        start_id = max_inst + 1
        lines.append(
            f"Scale **up** beyond the current largest instance (instance_{max_inst}). "
            f"Increase the primary size parameter(s) proportionally to create larger, "
            f"more challenging instances.\n"
        )
        lines.append("Output files:")
        for i in range(count):
            lines.append(f"- `instance_{start_id + i}.json`")
    elif direction == "down":
        # New small instances get IDs after existing max (appended to end)
        start_id = max_inst + 1
        lines.append(
            f"Scale **down** below the current smallest instance (instance_1). "
            f"Decrease the primary size parameter(s) to create smaller, easier instances.\n"
        )
        lines.append("Output files:")
        for i in range(count):
            lines.append(f"- `instance_{start_id + i}.json`")
    elif direction == "mid":
        start_id = max_inst + 1
        lines.append(
            "Create instance(s) that **interpolate** between existing instances to fill "
            "gaps in the scale distribution. Examine the existing INSTANCE_CONFIGS to find "
            "the largest jump in problem size between consecutive instances, and place new "
            "instances in that gap.\n"
        )
        lines.append("Output files:")
        for i in range(count):
            lines.append(f"- `instance_{start_id + i}.json`")
    else:
        lines.append(f"Direction: {direction}\n")

    return "\n".join(lines)


_LARGE_INST_RE = re.compile(r"^large_instance_(\d)(\d)\.json$")


def rotate_instance_files(instance_dir, scope_indices=None):
    """Rotate `large_instance_{ij}.json` → `large_instance_{i(j+1)}.json` so
    that the new j=1 slot becomes vacant. Returns list of (old, new) moves.

    Operates from the highest existing j down to 1 to avoid clobbering.
    No-op if instance_dir doesn't exist or contains no matching files.

    If ``scope_indices`` is given (e.g. [1, 5]), only those `i` values in
    1..5 are rotated; the other instances (and their audit copies) are
    left untouched. Used by --fix-by-condition + scale-down so we don't
    demote already-solved instances.

    Per the demoted-instance retention rule: this rotation creates audit
    copies (j>=2). Audit copies SHOULD ONLY live under
    ``data/paper_data/<pid>/instance/``. Do NOT call this on the
    ``frontier-or/`` tree — use ``clear_bench_j1_files`` instead.
    """
    if not os.path.isdir(instance_dir):
        return []
    scope = set(scope_indices) if scope_indices else set(range(1, 6))
    files_by_ij = {}
    for f in os.listdir(instance_dir):
        m = _LARGE_INST_RE.match(f)
        if m:
            files_by_ij[(int(m.group(1)), int(m.group(2)))] = os.path.join(instance_dir, f)
    if not files_by_ij:
        return []
    in_scope = {(i, j): p for (i, j), p in files_by_ij.items() if i in scope}
    if not in_scope:
        return []
    max_j = max(j for _, j in in_scope.keys())
    if max_j >= 9:
        # No room to rotate further (j is single digit in the naming scheme)
        raise RuntimeError(
            f"Cannot rotate: {instance_dir} already has j={max_j}; naming "
            "scheme supports only j ∈ 1..9.")
    moves = []
    for j in range(max_j, 0, -1):
        for i in sorted(scope):
            src = in_scope.get((i, j))
            if not src:
                continue
            dst = os.path.join(instance_dir, f"large_instance_{i}{j+1}.json")
            os.rename(src, dst)
            moves.append((src, dst))
    return moves


def clear_bench_j1_files(instance_dir, scope_indices=None):
    """Delete j=1 files in the BENCH dir without creating any audit
    copies. The agent will write fresh j=1 in the workspace;
    sync_new_j1_to_bench will copy them in afterwards.

    If ``scope_indices`` is given, only those `i` values are cleared; the
    other j=1 files (which already have valid solutions) are left alone.
    Stray j>=2 audit copies are also only wiped within scope.

    Used in place of rotate_instance_files for ``frontier-or/``
    so the bench dir never accumulates demoted (j>=2) files.

    Returns list of removed file paths.
    """
    removed = []
    if not os.path.isdir(instance_dir):
        return removed
    scope = set(scope_indices) if scope_indices else set(range(1, 6))
    for i in sorted(scope):
        f = os.path.join(instance_dir, f"large_instance_{i}1.json")
        if os.path.isfile(f):
            os.remove(f)
            removed.append(f)
    # Defensive: also wipe any stray j>=2 audit copies for scoped i's.
    for f in os.listdir(instance_dir):
        m = _LARGE_INST_RE.match(f)
        if m and int(m.group(2)) >= 2 and int(m.group(1)) in scope:
            os.remove(os.path.join(instance_dir, f))
    return removed


def sync_new_j1_to_bench(paper_id, scope_indices=None):
    """After a scale-up/down agent finishes, copy the freshly-written
    `data/paper_data/<pid>/instance/large_instance_{11..51}.json` into
    `frontier-or/<pid>/instance/`. Returns list of synced files
    (or empty if nothing to sync).

    If ``scope_indices`` is given, only sync those `i` values; other
    instances (which were never touched and still hold the original
    baseline) remain in bench unchanged.
    """
    import shutil
    src_dir = os.path.join(PAPER_DATA_DIR, paper_id, "instance")
    dst_dir = os.path.join(BENCH_DIR, paper_id, "instance")
    if not os.path.isdir(src_dir) or not os.path.isdir(dst_dir):
        return []
    scope = set(scope_indices) if scope_indices else set(range(1, 6))
    synced = []
    for i in sorted(scope):
        src = os.path.join(src_dir, f"large_instance_{i}1.json")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, f"large_instance_{i}1.json"))
            synced.append(f"large_instance_{i}1.json")
    return synced


def run_gurobi_smoke_test(paper_id, time_limit=100, j_index=1, indices=None):
    """Run gurobi_code.py on the large_instance_{i}{j_index}.json files in
    parallel with the given per-instance time_limit.

    By default smokes all 5 (i ∈ 1..5). If ``indices`` is given (e.g.
    [1, 5]), only those i's are smoked — used by --fix-by-condition +
    scale-down so we don't waste 100s budget re-verifying instances we
    never touched.

    Returns: list of dicts (one per smoked i, in the iteration order):
      {"i": int, "wall_time": float|None, "status": str|None, "obj": Any|None}
    """
    bench_paper = os.path.join(BENCH_DIR, paper_id)
    code = os.path.join(bench_paper, "gurobi_code.py")

    def run_one(i):
        inst = os.path.join(bench_paper, "instance",
                            f"large_instance_{i}{j_index}.json")
        sol = os.path.join("/tmp", f"_smoke_{paper_id}_{i}{j_index}.json")
        if not os.path.isfile(inst) or not os.path.isfile(code):
            return {"i": i, "wall_time": None, "status": "missing", "obj": None}
        # Clear any stale stub from a previous (possibly leaked) run so we
        # can't be tricked into reading a different session's output.
        try:
            os.unlink(sol)
        except FileNotFoundError:
            pass
        t0 = time.time()
        try:
            _, _, _, timed_out = _run_smoke_subprocess(
                ["python", code,
                 "--instance_path", inst, "--solution_path", sol,
                 "--time_limit", str(time_limit)],
                timeout=time_limit + 30,
            )
            if timed_out:
                raise subprocess.TimeoutExpired(cmd="python", timeout=time_limit + 30)
            elapsed = time.time() - t0
            if os.path.isfile(sol):
                try:
                    d = json.load(open(sol))
                    # Many gurobi_code.py write status as int (Gurobi status
                    # code, e.g. 2=OPTIMAL); normalise to lowercase string and
                    # map known integer codes to canonical names so the
                    # bad_status set in all_solved_optimal_fast works.
                    raw = d.get("status")
                    if raw in (None, ""):
                        raw = d.get("solution_status") or ""
                    GUROBI_STATUS_MAP = {
                        2: "optimal", 3: "infeasible",
                        4: "inf_or_unbd", 5: "unbounded",
                        6: "cutoff", 7: "iteration_limit",
                        8: "node_limit", 9: "time_limit",
                        10: "solution_limit", 11: "interrupted",
                        12: "numeric", 13: "suboptimal",
                        14: "inprogress", 15: "user_obj_limit",
                    }
                    if isinstance(raw, bool):
                        status = "True" if raw else "False"
                    elif isinstance(raw, int):
                        status = GUROBI_STATUS_MAP.get(raw, str(raw))
                    else:
                        status = str(raw).lower()
                    return {"i": i, "wall_time": elapsed, "status": status,
                            "obj": d.get("objective_value")}
                except Exception:
                    return {"i": i, "wall_time": elapsed, "status": "unparsable",
                            "obj": None}
            return {"i": i, "wall_time": elapsed, "status": "no_output", "obj": None}
        except subprocess.TimeoutExpired:
            return {"i": i, "wall_time": time_limit + 30, "status": "killed",
                    "obj": None}
        except Exception as e:
            return {"i": i, "wall_time": None, "status": f"err:{e}", "obj": None}

    smoke_idxs = sorted(set(indices)) if indices else list(range(1, 6))
    with ThreadPoolExecutor(max_workers=min(5, len(smoke_idxs))) as ex:
        return list(ex.map(run_one, smoke_idxs))


def all_solved_optimal_fast(results, time_limit):
    """All 5 finished comfortably under time_limit with a usable solution.
    "Solved fast" = wall_time strictly under budget AND a numeric
    objective_value was recorded (status field absent is OK — some
    gurobi_code.py variants don't emit it but still write obj).
    Statuses that indicate Gurobi hit the time limit (incumbent / time_limit)
    are treated as "hard enough" (NOT solved-fast).
    """

    def _solved_fast(r):
        if r["wall_time"] is None or r["wall_time"] >= time_limit - 1.0:
            return False
        # Definite-failure / no-solution states. Substring match on
        # "incumbent" / "time_limit" was too aggressive (e.g. matched
        # custom logger label "incumbent_updated" which is a fast
        # solve) — wall_time check above already catches genuine
        # time-limit cases, so we only flag explicit failure tokens
        # here using exact membership rather than substring.
        bad_status = {
            "killed", "no_output", "missing", "unparsable",
            "incumbent", "time_limit", "time-limit", "timelimit",
        }
        if r["status"] and r["status"] in bad_status:
            return False
        # Otherwise require a numeric objective_value to be present (so a
        # crash that happens to produce no output is not mistaken for fast).
        try:
            float(r["obj"])
        except (TypeError, ValueError):
            return False
        return True

    return all(_solved_fast(r) for r in results)


def judge_patch_succeeded(results, time_limit):
    """Stricter post-judge success check than ``all_solved_optimal_fast``.

    Used after the judge agent's ``manual`` / ``compare_tiny`` patches and
    a re-smoke. Returns True iff EVERY instance in ``results``:

    * finished strictly under ``time_limit`` (no killed / timeout grace),
    * has a numeric ``obj``, AND
    * status is NOT one of the unambiguous failure tokens
      (``killed``, ``no_output``, ``missing``, ``unparsable``, ``infeasible``,
      ``time_limit``).

    Differs from ``all_solved_optimal_fast`` in that ``status="incumbent"``
    (TLE-with-incumbent) WOULD count here as a real failure too, since
    we re-smoked at 100 s — finishing < 99 s with incumbent already implies
    status will be ``optimal`` or numeric. The point is to be honest about
    "patches did NOT actually fix the model" so the pipeline doesn't fake
    a SMOKE-KEEP after a degenerate ``no_output`` re-smoke. Only the
    instances actually passed in (i.e. the scoped subset under
    ``--fix-by-condition``) are evaluated.
    """
    REAL_FAILURES = {
        "killed", "no_output", "missing", "unparsable",
        "infeasible", "time_limit", "time-limit", "timelimit",
    }
    if not results:
        return False
    for r in results:
        if r["wall_time"] is None or r["wall_time"] >= time_limit - 1.0:
            return False
        if r["status"] and r["status"] in REAL_FAILURES:
            return False
        if r["obj"] is None:
            return False
        try:
            float(r["obj"])
        except (TypeError, ValueError):
            return False
    return True


def judge_patches_overfit(results, time_limit, trivial_threshold=10.0):
    """Did the judge's patches OVER-SHRINK the model to triviality?

    Returns True if every instance solved OPT in less than
    ``trivial_threshold`` seconds. That signals the patches reduced
    dimensions so aggressively that the model lands as tag C (trivially
    solvable) which then needs scale-up to undo. Use this to force REDO
    with a less-aggressive patch on the next iteration.

    NOT triggered when even ONE instance takes meaningful time (≥
    threshold), because patch effectiveness is per-instance: agent may
    have legitimately fixed the hardest of the scoped set, leaving
    others naturally fast.
    """
    if not results:
        return False
    for r in results:
        if r.get("wall_time") is None:
            return False
        if r["wall_time"] >= trivial_threshold:
            return False
    return True


def judge_patch_broke_model(results, time_limit):
    """Did the judge's patches actively BREAK the model (vs just leave it
    unsolved within the smoke budget)?

    Returns True ONLY if some instance shows an unambiguous regression:
    * `infeasible` — patches over-tightened a constraint to LP-INFEAS.
    * `no_output` / `missing` / `unparsable` — IO corruption.
    * `killed` with wall_time well below the smoke budget — gurobi died
      mid-run (OOM / SIGSEGV / other crash). Distinguished from "killed
      because the smoke wrapper hit its grace timeout (~time_limit + 30s)",
      which means the model just ran the full budget without finishing —
      that is NOT a regression for OOM-prone papers; it indicates the
      patches at least let gurobi run without crashing, and a 1h gurobi
      rerun is what really verifies the fix.

    Returns False otherwise — including TLE_no_incumbent and TLE_with_incumbent.
    Use together with ``judge_patch_succeeded``: if neither (succeeded nor
    broke), the patches are a "soft KEEP" — smoke didn't prove a clean fix
    but didn't introduce a new failure either, so persist them and let the
    1h gurobi rerun decide.
    """
    # Substring match — wrappers may emit verbose status strings like
    # "infeasible or no solution found", so exact-set membership misses
    # them. Any status containing one of these tokens counts as broken.
    BROKEN_TOKENS = ("infeasible", "no_output", "missing", "unparsable",
                     "no solution", "no_solution")
    if not results:
        return False
    for r in results:
        s = (r.get("status") or "").lower()
        if any(tok in s for tok in BROKEN_TOKENS):
            return True
        # OOM / crash detection: killed AND wall_time finished well under
        # the smoke budget (grace is +30s on top of time_limit). True TLE
        # sits at wall_time >= time_limit - 1; an OOM would be far below.
        if s == "killed" and r["wall_time"] is not None and \
           r["wall_time"] < time_limit - 5.0:
            return True
    return False


def revert_scale_up_bench_only(paper_id, scope_indices=None):
    """Partial revert for the SMOKE-GAVEUP path:

    - Restore `frontier-or/<pid>/instance/` to baseline (delete
      the latest synced j=1, move j=2 back to j=1) so eval/gurobi pipelines
      continue to read the proven-good original instances.
    - LEAVE the workspace `data/paper_data/<pid>/instance/` AND
      `scale_diversity_parameters.txt` untouched. Workspace retains the
      last iteration's scaled-up files at j=1 (and the original at j=2),
      which a future run can either inspect or use as a starting point
      for further scale-up without re-doing the iterations already paid for.

    If ``scope_indices`` is given, only the scoped i's are reverted.
    """
    bench_inst_dir = os.path.join(BENCH_DIR, paper_id, "instance")
    ws_inst_dir = os.path.join(PAPER_DATA_DIR, paper_id, "instance")
    if not os.path.isdir(bench_inst_dir):
        return
    scope = sorted(set(scope_indices) if scope_indices else set(range(1, 6)))
    # Delete the latest synced j=1 in bench (scoped only).
    for i in scope:
        f = os.path.join(bench_inst_dir, f"large_instance_{i}1.json")
        if os.path.exists(f):
            os.remove(f)
    # Restore baseline by copying the workspace's audit (j=2) into bench's
    # j=1 slot. Bench never holds the j=2 copy itself; the demoted/audit
    # files live only under ``data/paper_data/``.
    for i in scope:
        baseline = os.path.join(ws_inst_dir, f"large_instance_{i}2.json")
        target = os.path.join(bench_inst_dir, f"large_instance_{i}1.json")
        if os.path.exists(baseline):
            shutil.copy2(baseline, target)


def revert_scale_up_rotation(paper_id, spec_backup, scope_indices=None):
    """Undo the most recent rotation. Workspace is rotated back (j=2 → j=1
    audit restore); bench mirrors the workspace's restored j=1 via copy
    (bench never holds j>=2 audit copies).

    Also restores ``scale_diversity_parameters.txt`` from ``spec_backup``.

    If ``scope_indices`` is given, only the scoped i's are rolled back;
    other instances (which were never touched in this iteration) stay put.
    """
    scope = sorted(set(scope_indices) if scope_indices else set(range(1, 6)))
    # Workspace: delete the latest agent-written j=1, move audit j=2 → j=1
    ws_inst_dir = os.path.join(PAPER_DATA_DIR, paper_id, "instance")
    if os.path.isdir(ws_inst_dir):
        for i in scope:
            f = os.path.join(ws_inst_dir, f"large_instance_{i}1.json")
            if os.path.exists(f):
                os.remove(f)
        for i in scope:
            old = os.path.join(ws_inst_dir, f"large_instance_{i}2.json")
            new = os.path.join(ws_inst_dir, f"large_instance_{i}1.json")
            if os.path.exists(old):
                os.rename(old, new)
    # Bench: delete the synced j=1, then mirror the workspace's restored
    # j=1 (which is the original baseline after the rename above).
    bench_inst_dir = os.path.join(BENCH_DIR, paper_id, "instance")
    if os.path.isdir(bench_inst_dir):
        for i in scope:
            f = os.path.join(bench_inst_dir, f"large_instance_{i}1.json")
            if os.path.exists(f):
                os.remove(f)
        for i in scope:
            src = os.path.join(ws_inst_dir, f"large_instance_{i}1.json")
            dst = os.path.join(bench_inst_dir, f"large_instance_{i}1.json")
            if os.path.exists(src):
                shutil.copy2(src, dst)
    # Restore spec text
    spec_path = os.path.join(PAPER_DATA_DIR, paper_id, "scale_diversity_parameters.txt")
    if spec_backup is not None and os.path.isdir(os.path.dirname(spec_path)):
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(spec_backup)


def is_paper_at_cap(spec_text):
    """Heuristic: scale_diversity_parameters.txt contains the explicit phrase
    "AT the cap exactly" when the paper's j=1 row already saturates [STRICT]."""
    return "AT the cap exactly" in spec_text


# ============================================================================
# Tiny-scale-down helpers
# ============================================================================
# These mirror the large rotate/sync/smoke helpers above, but operate on a
# single `tiny_instance.json` file. KEEP semantics are INVERTED for tiny:
# we want the tiny instance to solve FAST (<100s), so a fast solve = success.

TINY_AUDIT_NAME = "tiny_instance_2.json"


def rotate_tiny_instance_file(instance_dir):
    """Rotate `tiny_instance.json` → `tiny_instance_2.json` (audit retention).

    No-op if `tiny_instance.json` doesn't exist. Overwrites any prior
    `tiny_instance_2.json` (the audit slot is single-deep — earlier audits
    are dropped to keep the loop simple).

    Per the demoted-instance retention rule: this rotation creates an audit
    copy. Audit copies SHOULD ONLY live under
    ``data/paper_data/<pid>/instance/``. Do NOT call this on the
    ``frontier-or/`` tree — use ``clear_bench_tiny`` instead.
    """
    if not os.path.isdir(instance_dir):
        return []
    src = os.path.join(instance_dir, "tiny_instance.json")
    dst = os.path.join(instance_dir, TINY_AUDIT_NAME)
    if not os.path.isfile(src):
        return []
    if os.path.isfile(dst):
        os.remove(dst)
    os.rename(src, dst)
    return [(src, dst)]


def clear_bench_tiny(instance_dir):
    """Delete `tiny_instance.json` in the BENCH dir without creating an audit
    copy. The agent will write fresh tiny in the workspace; sync_tiny_to_bench
    will copy it in afterwards.

    Used in place of rotate_tiny_instance_file for ``frontier-or/``
    so the bench dir never accumulates demoted (tiny_instance_2.json) audit
    files.

    Returns list of removed file paths.
    """
    removed = []
    if not os.path.isdir(instance_dir):
        return removed
    f = os.path.join(instance_dir, "tiny_instance.json")
    if os.path.isfile(f):
        os.remove(f)
        removed.append(f)
    # Defensive: also wipe any stray tiny audit files from older runs.
    audit = os.path.join(instance_dir, TINY_AUDIT_NAME)
    if os.path.isfile(audit):
        os.remove(audit)
    return removed


def sync_tiny_to_bench(paper_id):
    """After a tiny-scale-down agent finishes, copy
    `data/paper_data/<pid>/instance/tiny_instance.json` into
    `frontier-or/<pid>/instance/`. Returns list of synced files
    (or empty if nothing to sync).
    """
    src = os.path.join(PAPER_DATA_DIR, paper_id, "instance", "tiny_instance.json")
    dst_dir = os.path.join(BENCH_DIR, paper_id, "instance")
    if not os.path.isfile(src) or not os.path.isdir(dst_dir):
        return []
    shutil.copy2(src, os.path.join(dst_dir, "tiny_instance.json"))
    return ["tiny_instance.json"]


def run_gurobi_smoke_test_tiny(paper_id, time_limit=100):
    """Run gurobi_code.py on `tiny_instance.json` once with the given
    per-instance `time_limit`.

    Returns: dict with keys
        {"wall_time": float|None, "status": str|None, "obj": Any|None}
    """
    bench_paper = os.path.join(BENCH_DIR, paper_id)
    code = os.path.join(bench_paper, "gurobi_code.py")
    inst = os.path.join(bench_paper, "instance", "tiny_instance.json")
    sol = os.path.join("/tmp", f"_smoke_tiny_{paper_id}.json")
    if not os.path.isfile(inst) or not os.path.isfile(code):
        return {"wall_time": None, "status": "missing", "obj": None}
    # Drop any leaked stub from a prior session before we launch.
    try:
        os.unlink(sol)
    except FileNotFoundError:
        pass
    t0 = time.time()
    try:
        _, _, _, timed_out = _run_smoke_subprocess(
            ["python", code, "--instance_path", inst,
             "--solution_path", sol, "--time_limit", str(time_limit)],
            timeout=time_limit + 60,
        )
        if timed_out:
            raise subprocess.TimeoutExpired(cmd="python", timeout=time_limit + 60)
        elapsed = time.time() - t0
        if os.path.isfile(sol):
            try:
                with open(sol) as f:
                    data = json.load(f)
                # Normalise status: many gurobi_code.py write integer
                # codes (e.g. 2=OPTIMAL, 9=TIME_LIMIT). Map to canonical
                # lowercase strings so bad_status comparisons work.
                raw = data.get("status")
                if raw in (None, ""):
                    raw = data.get("solution_status") or "unknown"
                GUROBI_STATUS_MAP = {
                    2: "optimal", 3: "infeasible",
                    4: "inf_or_unbd", 5: "unbounded",
                    6: "cutoff", 7: "iteration_limit",
                    8: "node_limit", 9: "time_limit",
                    10: "solution_limit", 11: "interrupted",
                    12: "numeric", 13: "suboptimal",
                    14: "inprogress", 15: "user_obj_limit",
                }
                if isinstance(raw, bool):
                    status = "True" if raw else "False"
                elif isinstance(raw, int):
                    status = GUROBI_STATUS_MAP.get(raw, str(raw))
                else:
                    status = str(raw).lower()
                obj = data.get("objective_value", data.get("obj"))
                return {"wall_time": elapsed, "status": status, "obj": obj}
            except Exception:
                pass
        return {"wall_time": elapsed, "status": "no_output", "obj": None}
    except subprocess.TimeoutExpired:
        return {"wall_time": time.time() - t0, "status": "killed", "obj": None}


def tiny_solved_fast(result, time_limit):
    """True if the tiny instance was cleanly solved within the budget.

    Conditions:
      - wall_time < time_limit - 1.0 (left some slack for overhead)
      - status not in {time_limit, killed, no_output, missing, unparsable, ...}
      - objective_value is numeric

    Used by tiny-scale-down: True → KEEP (target hardness reached);
    False → REDO with smaller scale.
    """
    if result["wall_time"] is None or result["wall_time"] >= time_limit - 1.0:
        return False
    # Exact-match membership (substring match was too aggressive — e.g.
    # custom logger labels like "incumbent_updated" are valid fast solves).
    bad_status = {
        "incumbent", "time_limit", "time-limit", "timelimit",
        "killed", "no_output", "missing", "unparsable",
    }
    if result["status"] and result["status"] in bad_status:
        return False
    try:
        float(result["obj"])
    except (TypeError, ValueError):
        return False
    return True


def revert_tiny_bench_only(paper_id):
    """Partial revert for the tiny SMOKE-GAVEUP path:

    - Restore `frontier-or/<pid>/instance/tiny_instance.json` to
      the demoted baseline (delete the latest synced tiny, move
      tiny_instance_2.json back to tiny_instance.json).
    - LEAVE the workspace `data/paper_data/<pid>/instance/tiny_instance.json`
      AND `scale_diversity_parameters.txt` untouched. Workspace retains the
      last iteration's scaled-down tiny so a future re-run can use it as a
      starting point.
    """
    bench_inst_dir = os.path.join(BENCH_DIR, paper_id, "instance")
    ws_inst_dir = os.path.join(PAPER_DATA_DIR, paper_id, "instance")
    if not os.path.isdir(bench_inst_dir):
        return
    cur = os.path.join(bench_inst_dir, "tiny_instance.json")
    if os.path.isfile(cur):
        os.remove(cur)
    # Restore baseline by copying the workspace's audit (tiny_instance_2.json)
    # into bench's tiny_instance.json. Bench never holds the audit copy itself.
    baseline = os.path.join(ws_inst_dir, TINY_AUDIT_NAME)
    if os.path.isfile(baseline):
        shutil.copy2(baseline, cur)


def revert_tiny_rotation(paper_id, spec_backup):
    """Undo the most recent tiny rotation. Workspace is rotated back
    (tiny_instance_2.json → tiny_instance.json audit restore); bench mirrors
    the workspace's restored tiny via copy (bench never holds audit copies).

    Also restores ``scale_diversity_parameters.txt`` from ``spec_backup``.
    """
    # Workspace: delete the latest agent-written tiny, restore from audit
    ws_inst_dir = os.path.join(PAPER_DATA_DIR, paper_id, "instance")
    if os.path.isdir(ws_inst_dir):
        cur = os.path.join(ws_inst_dir, "tiny_instance.json")
        bak = os.path.join(ws_inst_dir, TINY_AUDIT_NAME)
        if os.path.isfile(cur):
            os.remove(cur)
        if os.path.isfile(bak):
            os.rename(bak, cur)
    # Bench: delete synced tiny, mirror workspace's restored tiny.
    bench_inst_dir = os.path.join(BENCH_DIR, paper_id, "instance")
    if os.path.isdir(bench_inst_dir):
        cur_b = os.path.join(bench_inst_dir, "tiny_instance.json")
        if os.path.isfile(cur_b):
            os.remove(cur_b)
        src = os.path.join(ws_inst_dir, "tiny_instance.json")
        if os.path.isfile(src):
            shutil.copy2(src, cur_b)
    spec_path = os.path.join(PAPER_DATA_DIR, paper_id, "scale_diversity_parameters.txt")
    if spec_backup is not None and os.path.isdir(os.path.dirname(spec_path)):
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(spec_backup)


def build_tiny_scale_down_instructions(paper_id, multiplier, divisor):
    """Build the tiny-scale-down instruction text for one paper.

    `multiplier` semantics:
      - 1.0  : initial target = the `s` tier of `(s, m, l)` thresholds
               (upper bound of the "small" band).
      - <1.0 : iterative — target = s tier × multiplier (further scale-down
               below s when prior smoke-test still solved >100s).

    Returns None if the spec is missing.
    """
    paper_dir = os.path.join(PAPER_DATA_DIR, paper_id)
    spec_path = os.path.join(paper_dir, "scale_diversity_parameters.txt")
    if not os.path.isfile(spec_path):
        return None
    if multiplier == 1.0:
        sub_label = "Initial — saturate at s tier"
        target_rule = (
            "target = the `s` tier of the `(s, m, l)` threshold triple "
            "from the `## [STRICT]` block (upper bound of the 'small' band; "
            "round each axis to a sensible integer / native granularity)"
        )
    else:
        sub_label = f"Iterative scale-down × {multiplier:.4f}"
        target_rule = (
            f"target = `s` tier × {multiplier:.4f} (smoke-test signalled "
            f"the prior tiny was still too hard to solve in 100s; further "
            f"scale-down by 1/{divisor} per iteration applied to the s tier)"
        )
    return f"""### Goal: Tiny scale-DOWN for `{paper_id}` — produce a SMALLER `instance/tiny_instance.json` ({sub_label})

The previous `tiny_instance.json` has ALREADY been rotated by the wrapper:
  tiny_instance.json  →  tiny_instance_2.json
The `tiny_instance_2.json` file is now considered [FAILED] (Gurobi solve
exceeded the 100s smoke-test budget on the previous tiny scale).

Your task: write `update_instances.py` that produces a fresh single
`instance/tiny_instance.json` with **scaled-DOWN** parameters. The tiny
instance is a smoke-test instance (1 file, no diversity index — just one
representative for sanity-check / quick eval).

### Tiny scale-DOWN policy for this paper
- Classification: {sub_label}
- Target scale rule: {target_rule}

### Steps
1. Read `scale_diversity_parameters.txt` to extract:
   - The existing `## Tiny scale parameters` block if present (the OLD
     tiny scale params). If not present, infer from the demoted
     `instance/tiny_instance_2.json` directly (load it as JSON, read the
     primary scale fields).
   - The `[STRICT]` block, ESPECIALLY the line listing the
     `(s, m, l)` triple — the `s` value is the small tier (the UPPER
     bound for tiny scale-down per the rule above).
2. Compute the new tiny target scale parameters per the rule above. If an
   axis is integer-valued, round to the nearest sensible integer, but
   never below 1.
3. Write `update_instances.py` that:
   - Imports `gen_module.generate_instance(...)` via importlib (per the
     boilerplate elsewhere in this prompt).
   - Defines `UPDATE_CONFIGS` with EXACTLY ONE entry — diversity index
     does NOT apply to tiny (just one tiny file).
   - `output_filename = "instance/tiny_instance.json"`.
   - Use `base_seed = 42` (or another simple deterministic seed). Tiny
     uses 1 seed only since there's no diversity dimension.
   - Runs feasibility checks per the boilerplate.
4. RUN `update_instances.py` and confirm `tiny_instance.json` is FEASIBLE.
5. UPDATE `scale_diversity_parameters.txt`:
   - INSERT or REPLACE a `## Tiny scale parameters` block recording the
     NEW (smaller) tiny scale params. If the section did not exist
     before, add it just before `## [STRICT] Real-world largest-scale cap`.
   - If a prior `## Tiny scale parameters` existed, copy its body into
     a new `## [FAILED] Tiny scale parameters (demoted, Gurobi >100s in
     smoke test)` block placed immediately after the new `## Tiny scale
     parameters` section.
   - Append a one-line note in `## Notes` recording the tiny scale-down
     (date + iteration target rule) for audit.

### Constraints
- DO NOT alter `tiny_instance_2.json` (the demoted tiny, retained for audit).
- DO NOT touch `large_instance_*.json` or any large file.
- The new `tiny_instance.json` MUST share the same JSON top-level schema as
  the demoted `tiny_instance_2.json`.
"""


def build_scale_down_instructions(paper_id, multiplier, floor_divisor,
                                  skip_below_floor=False,
                                  scope_indices=None):
    """Build a scale-DOWN instruction text for one paper.

    `multiplier` semantics for scale-down:
      - 1.0  : target = m tier (the "large" lower bound from the (s, m, l)
               threshold triple in the [STRICT] block); for papers already
               at or below the m tier, target = current_scale / floor_divisor.
      - > 1.0: target = m tier × multiplier (i.e., slightly LESS aggressive
               scale-down than the floor; used when smoke-test says iter 1
               was too aggressive — instances became trivial).

    If ``scope_indices`` is given (e.g. [1, 5] from --fix-by-condition),
    only those `i` values should be regenerated; the other instances are
    already solved successfully and MUST NOT be touched. The generated
    `UPDATE_CONFIGS` will then have only `len(scope_indices)` entries.

    Returns None if the spec is missing.
    """
    paper_dir = os.path.join(PAPER_DATA_DIR, paper_id)
    spec_path = os.path.join(paper_dir, "scale_diversity_parameters.txt")
    if not os.path.isfile(spec_path):
        return None
    scope_set = sorted(set(scope_indices)) if scope_indices else None
    if multiplier == 1.0:
        target_rule = (
            f"target = m tier of the `(s, m, l)` threshold triple from the "
            f"`## [STRICT]` block (the lower bound of the 'large' band). "
            f"If the paper's CURRENT primary-axis scale is already ≤ m tier, "
            f"set target = current_scale / {floor_divisor} (further scale-down "
            f"below the m floor when the paper is already at or below m)"
        )
    else:
        target_rule = (
            f"target = m tier × {multiplier} (slightly LESS aggressive than "
            f"floor; smoke-test signalled iter 1 was too aggressive — bumping "
            f"the target up by {multiplier}× the m tier)"
        )
    if scope_set is None:
        scope_clause_top = ""
        scope_clause_step = ""
        rotated_list = "{11..51}"
        rotated_to = "{12..52}"
        instance_count_phrase = "5 entries (i ∈ 1..5)"
        feasi_phrase = "all 5 new files"
    else:
        scope_str = ", ".join(str(i) for i in scope_set)
        n_scope = len(scope_set)
        scope_clause_top = (
            f"\n\n**SCOPE — partial regeneration**: ONLY these instance ids "
            f"are being scaled down: **i ∈ {{{scope_str}}}**. The other "
            f"large_instance_*1.json files have already been solved "
            f"successfully (have a Gurobi incumbent) and **MUST NOT be "
            f"regenerated, modified, or referenced** in `UPDATE_CONFIGS`. "
            f"They will remain on disk in their current state."
        )
        scope_clause_step = (
            f"   - **Only emit {n_scope} entries** for `UPDATE_CONFIGS` — one "
            f"per i in {{{scope_str}}}. DO NOT include entries for the "
            f"other i values.\n"
        )
        rotated_list = "{" + ", ".join(f"{i}1" for i in scope_set) + "}"
        rotated_to = "{" + ", ".join(f"{i}2" for i in scope_set) + "}"
        instance_count_phrase = (
            f"{n_scope} entries (i ∈ {{{scope_str}}})")
        feasi_phrase = f"the {n_scope} new file(s) in scope"
    return f"""### Goal: Scale-DOWN for `{paper_id}` — produce a SMALLER `large_instance_*1.json`

The previous large instance(s) at j=1 were 1-hour Gurobi time-outs that
returned NO feasible solution (output: N/A). The wrapper has already rotated
them:
  large_instance_{rotated_list}.json  →  large_instance_{rotated_to}.json
The rotated files are now considered [FAILED] (Gurobi could not finish).{scope_clause_top}

Your task: write `update_instances.py` that produces a fresh j=1 row at
`instance/large_instance_*1.json` (within scope) with **scaled-DOWN**
parameters but the **same diversity choices** as the demoted j=1 row.

### Scale-DOWN policy for this paper
- Target scale rule: {target_rule}

### Steps
1. Read `scale_diversity_parameters.txt` to extract:
   - The current j=1 scale params (under `## Scale parameters` — these are
     the OLD too-large values, just copied to `## [FAILED] j=2 ...`).
   - The `[STRICT]` block, ESPECIALLY the line `(s, m, l) = (S, M, L)` — that M
     is the m tier (the floor for scale-down per the rule above).
   - The `[LARGE_SCALE_RANGE]` reasoning section if present.
   - The 5 diversity structures listed under `## 5 diversity structures` —
     PRESERVE these diversity choices for the new j=1 row.
2. Compute the new target scale parameters per the rule above. If the axis is
   integer-valued, round to the nearest sensible integer.
3. Write `update_instances.py` that:
   - Imports `gen_module.generate_instance(...)` via importlib.
   - Defines `UPDATE_CONFIGS` with {instance_count_phrase}, all using the NEW
     (smaller) scale params, with `output_filename =
     "instance/large_instance_{{i}}1.json"`.
{scope_clause_step}   - PRESERVES per-i diversity choice from the demoted j=1 row; only scale
     changes.
   - Uses `base_seed = i * 100000 + 200` to avoid collisions with demoted seeds.
   - Runs feasibility checks per the boilerplate.
4. After running `update_instances.py` and confirming {feasi_phrase} are
   FEASIBLE, UPDATE `scale_diversity_parameters.txt`:
   - INSERT a new `## [FAILED] j=2 scale parameters
     (demoted from previous j=1, Gurobi 1h time-out with no feasible)`
     block, copying the OLD scale param values into it.
   - INSERT a new `## [FAILED] j=2 — 5 diversity structures
     (demoted from previous j=1)` block with the 5 sub-headings renumbered
     `### large_instance_{{i1}}` → `### large_instance_{{i2}}`.
   - REPLACE `## Scale parameters` with the NEW (smaller) values.
   - REPLACE `## 5 diversity structures` to describe the new j=1 row (same
     diversity choices, scale prose updated).
   - Append a one-line note in `## Notes` recording the scale-down (date +
     classification + target rule) for audit.

### Constraints
- DO NOT alter the rotated `large_instance_*2.json` files (the demoted files).
- DO NOT touch `tiny_instance.json`.
- DO NOT modify any `large_instance_*1.json` whose `i` is OUTSIDE the scope
  declared above (those instances already have valid Gurobi incumbents).
- The new file(s) MUST share the same JSON top-level schema as the demoted
  files.
"""


def build_scale_up_instructions(paper_id, multiplier, skip_at_ceiling=False):
    """Construct the Claude instruction text for one paper's scale-up.

    The "ceiling" is the `l` tier of the `(s, m, l)` threshold triple in the
    `[STRICT]` block — the largest realistic scale surveyed for this paper.

    `multiplier` semantics:
      - 1.0  : target = ceiling exactly (saturate-at-ceiling; Sub-A default)
      - > 1.0: target = ceiling × multiplier (override; Sub-B default 1.5)

    Returns None if the paper should be skipped (spec missing, or
    --scale-up-skip-at-ceiling/--scale-up-skip-at-cap and paper is at ceiling).
    """
    paper_dir = os.path.join(PAPER_DATA_DIR, paper_id)
    spec_path = os.path.join(paper_dir, "scale_diversity_parameters.txt")
    if not os.path.isfile(spec_path):
        return None
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_text = f.read()
    at_ceiling = is_paper_at_cap(spec_text)
    if at_ceiling and skip_at_ceiling:
        return None
    if multiplier == 1.0:
        sub_label = ("Sub-A (below ceiling, saturate-at-ceiling)" if not at_ceiling
                     else "Sub-B (AT ceiling, no override requested)")
        target_rule = (
            "target = `[STRICT]` ceiling exactly on every axis "
            "(the `l` tier value; saturate at ceiling)"
        )
    else:
        sub_label = (f"Sub-B (AT ceiling, applying override × {multiplier})"
                     if at_ceiling
                     else f"Sub-A scaled-up × {multiplier}")
        target_rule = (
            f"target = `[STRICT]` ceiling × {multiplier} (the `l` tier × "
            f"{multiplier}; round each axis to a sensible integer / native "
            f"granularity); explicit user override of the ceiling"
        )
    return f"""### Goal: Scale-up for `{paper_id}` — produce new larger `large_instance_{{11..51}}.json` ({sub_label})

The previous 5 large instances at j=1 have ALREADY been rotated by the wrapper:
  large_instance_{{11..51}}.json  →  large_instance_{{12..52}}.json
The {{12..52}} files are now considered [FAILED] (Gurobi solved them in <100s;
too easy for benchmarking).

Your task: write `update_instances.py` that produces a fresh j=1 row at
`instance/large_instance_{{11..51}}.json` with **scaled-up** parameters but the
**same 5 diversity choices** as the demoted j=1 row.

### Scale-up policy for this paper
- Classification: {sub_label}
- Target scale rule: {target_rule}

### Steps
1. Read `scale_diversity_parameters.txt` to extract:
   - The current j=1 scale params (under `## Scale parameters`)
   - The `[STRICT]` real-world cap
   - The `[LARGE_SCALE_RANGE]` reasoning section (companion params, range)
   - The 5 diversity structures listed under `## 5 diversity structures`
2. Compute the new target scale parameters per the rule above. If a parameter
   axis is integer-valued, round to the nearest sensible integer.
3. Write `update_instances.py` that:
   - Imports `gen_module.generate_instance(...)` via importlib (per the
     boilerplate elsewhere in this prompt).
   - Defines `UPDATE_CONFIGS` with 5 entries, one per diversity index `i ∈ 1..5`,
     all using the NEW target scale params, with `output_filename =
     "instance/large_instance_{{i}}1.json"` (j=1 slot, now vacant).
   - PRESERVES the diversity choice for each `i` from the demoted j=1 row
     (only the scale changes; the per-i diversity dimension value stays).
   - Uses `base_seed = i * 100000 + 100` so the new files don't collide with the
     demoted seeds.
   - Runs feasibility checks per the boilerplate (retry up to MAX_SEED_ATTEMPTS).
4. After running `update_instances.py` and confirming all 5 new files are
   written and FEASIBLE, UPDATE `scale_diversity_parameters.txt`:
   - INSERT a new section header `## [FAILED] j=2 scale parameters
     (demoted from previous j=1, too easy for Gurobi)` immediately after the
     existing `## Scale parameters` block, copying the OLD scale param values
     into it.
   - INSERT a new section `## [FAILED] j=2 — 5 diversity structures
     (demoted from previous j=1)` containing the 5 sub-headings renumbered
     from `### large_instance_{{i1}}` to `### large_instance_{{i2}}` and the
     same body text.
   - REPLACE the existing `## Scale parameters` block in place with the NEW
     target scale param values.
   - REPLACE the existing `## 5 diversity structures` block to describe the
     new j=1 row: same 5 diversity choices, but each entry's "scale" prose
     should reflect the new target params.
   - Append a one-line note in `## Notes` that scale-up was applied (date + the
     classification + target rule), so the rotation history is auditable.

### Constraints
- DO NOT alter `large_instance_{{12..52}}.json` (the demoted files).
- DO NOT touch `tiny_instance.json`.
- The 5 NEW files MUST share the same JSON top-level schema as the demoted
  files (so checker / eval pipelines keep working unchanged).
"""


# ===================================================================
# Judge-agent helpers (used when --judge-before-bump is set).
# ===================================================================

_JUDGE_GUROBI_HEAD_LINES = 60
_JUDGE_GUROBI_BODY_LINES = 200
_JUDGE_INSTANCE_BYTES = 16_000
_JUDGE_SMOKE_TAIL_LINES = 80


def _judge_extract_smoke_summary(log: str) -> str:
    interesting = re.compile(
        r"(infeasible|Best objective|Time limit|Optimal|Objective|"
        r"Found heuristic|presolve|Root relaxation|Status:)",
        re.IGNORECASE)
    lines = log.splitlines()
    matched = [ln for ln in lines if interesting.search(ln)]
    tail = lines[-_JUDGE_SMOKE_TAIL_LINES:]
    seen = set()
    keep = []
    for ln in matched + tail:
        if ln not in seen:
            seen.add(ln)
            keep.append(ln)
    return "\n".join(keep[-_JUDGE_SMOKE_TAIL_LINES:])


def _judge_extract_gurobi_code(paper_id: str) -> str:
    code_path = os.path.join(BENCH_DIR, paper_id, "gurobi_code.py")
    if not os.path.isfile(code_path):
        return f"(gurobi_code.py not found for {paper_id})"
    with open(code_path) as f:
        lines = f.read().splitlines()
    head = "\n".join(lines[:_JUDGE_GUROBI_HEAD_LINES])
    body_start = next((i for i, ln in enumerate(lines)
                       if "addConstr" in ln), -1)
    if body_start < 0:
        body = ""
    else:
        body_start = max(0, body_start - 30)
        body_end = min(len(lines), body_start + _JUDGE_GUROBI_BODY_LINES)
        body = "\n".join(lines[body_start:body_end])
    return f"{head}\n# ... (skip imports/utilities) ...\n\n{body}"


def _judge_extract_instance_excerpt(paper_id: str, instance_id: str) -> str:
    inst_path = os.path.join(BENCH_DIR, paper_id, "instance",
                              f"large_instance_{instance_id}.json")
    if not os.path.isfile(inst_path):
        return f"(no large_instance_{instance_id}.json)"
    with open(inst_path) as f:
        raw = f.read()
    if len(raw) <= _JUDGE_INSTANCE_BYTES:
        return raw
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:_JUDGE_INSTANCE_BYTES]
    schema = {}
    for k, v in d.items():
        if isinstance(v, list) and v:
            schema[k] = {"_type": f"list[len={len(v)}]",
                         "_sample": v[0] if not isinstance(v[0], list)
                                    else v[0][:3]}
        elif isinstance(v, dict):
            sub_keys = list(v.keys())[:5]
            schema[k] = {"_type": "dict", "_keys": sub_keys,
                         "_sample": {k2: v[k2] for k2 in sub_keys}}
        else:
            schema[k] = v
    return json.dumps(schema, indent=2, default=str)[:_JUDGE_INSTANCE_BYTES]


def _judge_problem_summary(paper_id: str) -> str:
    pd_path = os.path.join(BENCH_DIR, paper_id, "problem_description.txt")
    if not os.path.isfile(pd_path):
        return f"(no problem_description for {paper_id})"
    with open(pd_path) as f:
        text = f.read()
    paras = re.split(r"\n\s*\n", text)
    return "\n\n".join(paras[:2])[:1500]


def _load_tiny_large_compare(paper_id: str, csv_path: str):
    """Look up the row for `paper_id` in tiny_large_parameters.csv. Returns
    a dict {tiny_gurobi_time, tiny_parameters, large_instance_parameters,
    difference} or None if csv is missing or paper has no row."""
    if not csv_path or not os.path.isfile(csv_path):
        return None
    import csv as _csv
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("paper_id") == paper_id:
                return {
                    "tiny_gurobi_time": (row.get("tiny_gurobi_time")
                                         or "").strip(),
                    "tiny_parameters": (row.get("tiny_parameters")
                                        or "").strip(),
                    "large_instance_parameters": (
                        row.get("large_instance_parameters") or "").strip(),
                    "difference": (row.get("difference") or "").strip(),
                }
    return None


def _format_tiny_large_block(compare: dict) -> str:
    """Render a comparison dict as a markdown block to inject into the
    judge prompt."""
    return (
        "## Tiny ↔ Large comparison (from tiny_large_parameters.csv)\n\n"
        "The tiny instance for THIS paper solves quickly to optimality. The "
        "large instance shares the same paper / same gurobi_code but is "
        "currently TLE without an incumbent. Treat tiny as a known-feasible "
        "lower-bound on dimension structure. The pure dimension axes that "
        "the paper's generator parameterizes are listed below; you may pick "
        "a midpoint between them to produce a model that is harder than "
        "tiny but solvable in <1h.\n\n"
        f"- **tiny gurobi_time**: {compare['tiny_gurobi_time']}s "
        "(confirms tiny is solvable on the same code)\n"
        f"- **tiny scale parameters**: `{compare['tiny_parameters']}`\n"
        f"- **current large parameters**: "
        f"`{compare['large_instance_parameters']}`\n"
        f"- **per-axis ratio (large / tiny)**: {compare['difference']}\n\n"
        "If you choose `decision=\"compare_tiny\"`, propose `multiply` "
        "patches whose factors move each large dimension axis toward tiny "
        "by some informed amount (e.g. geometric midpoint factor "
        "`1 / sqrt(ratio)`, or the arithmetic mean target divided by the "
        "current value). Specify the patches in the same JSON schema as "
        "`manual` — `manual_overrides.patches[]` with `op:multiply, "
        "factor:<f>, path:<json_path>`. Do not over-shrink (factor below "
        "0.1 is forbidden); aim for a meaningful reduction (factor in "
        "[0.4, 0.85] is the typical sweet spot for high-ratio axes).\n"
    )


def _judge_parse_decision(raw: str, debug_label: str = "") -> dict:
    """Parse the agent's JSON decision. Tries (1) ```json fenced block with
    balanced braces, (2) any balanced top-level {...}, (3) ValueError. On
    failure, saves the raw output to /tmp for debugging.
    """
    def _find_balanced(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    # 1) ```json fence with balanced braces
    fence = re.search(r"```json\s*(\{)", raw)
    if fence:
        candidate = _find_balanced(raw, fence.start(1))
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # 2) Any balanced top-level {...}
    first_brace = raw.find('{')
    if first_brace >= 0:
        candidate = _find_balanced(raw, first_brace)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Save raw for debug
    if debug_label:
        debug_path = os.path.join("/tmp", f"_judge_raw_{debug_label}.txt")
        try:
            with open(debug_path, "w") as f:
                f.write(raw)
            print(f"  [JUDGE-DEBUG] saved raw agent output to {debug_path}",
                  file=sys.stderr)
        except OSError:
            pass
    raise ValueError("no JSON object found in agent output")


def call_judge_agent(paper_id: str, instance_id: str, smoke_log: str,
                      time_limit: int, prompt_path: str,
                      force_manual: bool = False,
                      tiny_large_csv: str | None = None,
                      prefer_compare_tiny: bool = False) -> dict:
    """Build the judge prompt, invoke claude, return parsed decision dict.

    Returns dict with keys: decision (str), reason (str),
    manual_overrides (dict|None).

    When ``force_manual`` is True an override block is appended to the
    prompt instructing the agent that ``"scale"`` is unavailable — only
    ``"manual"`` or ``"giveup"`` are allowed.

    When ``prefer_compare_tiny`` is True (and the Tiny↔Large block is
    actually injected — i.e. ``tiny_large_csv`` resolves to a real row)
    an override block is appended that flips the default priority:
    ``compare_tiny`` becomes the preferred choice over ``manual`` whenever
    the tiny anchor is usable.
    """
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"judge prompt not found: {prompt_path}")
    with open(prompt_path) as f:
        template = f.read()
    prompt = (template
              .replace("{paper_id}", paper_id)
              .replace("{problem_summary}", _judge_problem_summary(paper_id))
              .replace("{time_limit}", str(time_limit))
              .replace("{smoke_output}",
                       _judge_extract_smoke_summary(smoke_log))
              .replace("{gurobi_code_excerpt}",
                       _judge_extract_gurobi_code(paper_id))
              .replace("{instance_excerpt}",
                       _judge_extract_instance_excerpt(paper_id, instance_id)))
    if force_manual:
        prompt += (
            "\n\n## FORCED MANUAL MODE — operator override\n"
            "The blind multiplier scale-down path has been **disabled** for "
            "this run because it has empirically failed to produce incumbent "
            "solutions for this paper class. Your `decision` MUST be "
            "`\"manual\"`, `\"compare_tiny\"`, or `\"giveup\"`. `\"scale\"` "
            "is NOT a valid choice.\n\n"
            "If you would otherwise have chosen `\"scale\"`, commit to a "
            "manual diagnosis instead: pick the most-likely-binding "
            "constraint from the diagnostic checklist (resource cap, "
            "initial state, fleet capacity, edge coverage, time window) "
            "and emit a targeted `set` / `multiply` / `clamp_max_per_item` "
            "patch on it. Use a 1.05–1.15× multiplier on the suspected "
            "binding capacity if you have no quantitative evidence — better "
            "an imperfect manual patch than a no-op scale-down. Use "
            "`\"giveup\"` ONLY when the model is provably structurally "
            "broken and no parameter override could repair it.\n"
        )
    # Append the Tiny↔Large comparison block when available — gives the
    # agent quantitative grounding to choose a `compare_tiny` decision.
    tiny_block_injected = False
    if tiny_large_csv:
        compare = _load_tiny_large_compare(paper_id, tiny_large_csv)
        if compare:
            prompt += "\n\n" + _format_tiny_large_block(compare)
            tiny_block_injected = True
    # `--prefer-compare-tiny` only takes effect when we actually have
    # tiny-anchor numbers to feed the agent. Without injected ratios, the
    # priority flip would leave the agent picking compare_tiny with no
    # quantitative basis — that's worse than the default order.
    if prefer_compare_tiny and tiny_block_injected:
        prompt += (
            "\n\n## PREFER COMPARE_TINY — operator override\n"
            "Empirically `manual` single-binding patches have not solved "
            "this paper class within the smoke budget. **Default to "
            "`\"compare_tiny\"`** when ANY axis ratio in the Tiny↔Large "
            "block is ≥ 2.0 (relax the usual ≥5× threshold). Compute the "
            "geometric-midpoint factors `1/sqrt(ratio)` for those axes and "
            "emit them as `multiply` patches.\n\n"
            "Pick `\"manual\"` ONLY when one of these is true:\n"
            "  • Step-0 classification is INFEAS_PROVEN AND the diagnostic "
            "checklist surfaces an UNAMBIGUOUS single-binding constraint "
            "(e.g. mu = sum_min/cap > 1, or `initial_power > "
            "shutdown_ramp_limit` for a clear majority of units). Generic "
            "1.10× guesses at a 'most-likely binding' DO NOT qualify under "
            "this override.\n"
            "  • All axis ratios in the comparison block are < 2.0 (tiny "
            "and large are already comparable on every dimension).\n\n"
            "Pick `\"giveup\"` only when both compare_tiny and manual are "
            "structurally impossible (e.g. tiny is itself unsolvable, or "
            "the model is provably broken).\n"
        )
    # Use thread-safe capture (does NOT redirect sys.stdout — that would
    # be a concurrency bug under ThreadPoolExecutor since other workers'
    # print() calls would land in the captured buffer).
    raw = run_claude_capture(prompt, label=f"judge:{paper_id}:{instance_id}")
    return _judge_parse_decision(raw,
                                  debug_label=f"{paper_id}_{instance_id}")


def apply_judge_patches(paper_id: str, decision: dict,
                          scope_indices=None) -> int:
    """Apply manual_overrides.patches from a decision to large j=1 instance
    files in BENCH_DIR. Returns total patches × instances applied.

    If ``scope_indices`` is given (e.g. [2, 4]), only those instance ids in
    1..5 are patched — used by --fix-by-condition so we don't mutate
    instances that already solved cleanly. Default (None) → all 5.

    Aborts (raises) if any patch fails on any instance.
    """
    if _apply_patches is None:
        raise RuntimeError("patch_applier not importable; cannot apply manual"
                           " overrides")
    overrides = decision.get("manual_overrides") or {}
    patches = overrides.get("patches", [])
    if not patches:
        return 0
    inst_dir = os.path.join(BENCH_DIR, paper_id, "instance")
    indices = list(scope_indices) if scope_indices else list(range(1, 6))
    # Two-pass: dry-run on all targets first so a sanity failure on
    # instance i=3 doesn't leave i=1, i=2 already mutated on disk.
    targets = []
    for i in indices:
        fp = os.path.join(inst_dir, f"large_instance_{i}1.json")
        if not os.path.isfile(fp):
            continue
        with open(fp) as f:
            inst = json.load(f)
        targets.append((fp, inst))
    if not targets:
        return 0
    # Pass 1 — dry-run; let any ValueError raise (caller catches).
    for fp, inst in targets:
        _apply_patches(inst, patches, dry_run=True)
    # Pass 2 — real apply (re-load to keep on-disk and in-memory aligned).
    total = 0
    for fp, _ in targets:
        with open(fp) as f:
            inst = json.load(f)
        results, _ = _apply_patches(inst, patches)
        with open(fp, "w") as f:
            json.dump(inst, f)
        total += sum(n for _, n in results)
    return total


# ---------------------------------------------------------------------------
# --fix-by-condition: per-instance targeting based on Gurobi result CSV.
#
# Picks instances from each paper that match BOTH:
#   • gurobi_time >= 3600s OR string contains 'time_out'
#   • gurobi_solution is non-numeric (N/A / 'time_out' / exit_code dict /
#     missing) — i.e. no incumbent was found
# This is the canonical "TLE-no-incumbent" pattern and is exactly what
# scale-down / manual-patch fixes target.
# ---------------------------------------------------------------------------

def _parse_csv_string_list(s):
    """Parse a CSV cell holding a JSON list-of-strings (e.g.
    '["0.11", "time_out", ...]'). Returns [] on parse failure / empty."""
    if not s or not s.strip():
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _is_failing_per_condition(time_str, sol_str):
    """Return True iff (time>=3600 OR 'time_out') AND (sol N/A OR 'time_out'
    OR non-numeric/exit_code) — i.e. instance hit time limit without
    producing a usable incumbent."""
    t = (time_str or "").strip()
    s = (sol_str or "").strip()
    # Time signal: explicit "time_out" string OR numeric >= 3600
    is_timeout = "time_out" in t.lower()
    if not is_timeout:
        try:
            if float(t) >= 3600:
                is_timeout = True
        except (ValueError, TypeError):
            pass
    if not is_timeout:
        return False
    # Solution signal: non-numeric (no incumbent)
    if s == "" or s.upper() == "N/A":
        return True
    if "time_out" in s.lower():
        return True
    if s.startswith("{"):  # exit_code wrapper from a crashed run
        return True
    try:
        float(s)
        return False  # numeric → has incumbent → don't pick
    except (ValueError, TypeError):
        return True  # non-numeric weirdness → no incumbent


def compute_fix_by_condition_targets(paper_ids, csv_path):
    """Read the merged results CSV; return {pid: [instance_idx, ...]} for
    papers that have at least one failing instance. instance_idx is 1..5
    (matching the canonical large_instance_<i>1.json naming; index 0 in
    the CSV's per-scale list is the tiny instance and is skipped).

    Raises FileNotFoundError if csv_path is missing.
    """
    import csv as _csv  # local to avoid polluting module top
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"results CSV not found: {csv_path}")
    paper_set = set(paper_ids)
    targets = {}
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            pid = row.get("paper_id")
            if not pid or pid not in paper_set:
                continue
            times = _parse_csv_string_list(row.get("gurobi_time", ""))
            sols = _parse_csv_string_list(row.get("gurobi_solution", ""))
            n = min(len(times), len(sols))
            failing = []
            # Per merge_and_classify: index 0 is tiny, 1..5 are large_1..5
            for i in range(1, min(n, 6)):
                if _is_failing_per_condition(times[i], sols[i]):
                    failing.append(i)
            if failing:
                targets[pid] = failing
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Update/add/replace data instances via Claude Code Agent."
    )
    parser.add_argument(
        "--paper-id", nargs="+", dest="paper_ids", default=None,
        help="Paper IDs to process.",
    )
    parser.add_argument(
        "--exclude-paper-id", nargs="+", dest="exclude_paper_ids", default=[],
        help="Paper IDs to exclude.",
    )
    parser.add_argument(
        "--paper_tag", "--paper-tag", dest="paper_tag", type=str, default=None,
        help="Filter paper_ids by tag (e.g. 'E') — uses the tag column of "
             "gurobi_results_all_new.csv at the repo root. Combined with "
             "--paper-id, the intersection is used; combined with neither "
             "--paper-id nor a hardcoded list, all matching papers are "
             "selected.",
    )
    parser.add_argument(
        "--mode",
        choices=["replace", "add", "custom", "scale-up", "scale-down",
                 "tiny-scale-down"],
        default="custom",
        help="Update mode: 'replace' for infeasible instances, 'add' for new "
             "instances, 'custom' for instructions-file, 'scale-up' to rotate "
             "current j=1 → j=2 [FAILED] and generate fresh larger j=1 row, "
             "'scale-down' to do the same but with target = m tier (or "
             "current/floor_divisor if already ≤ m), 'tiny-scale-down' to "
             "scale DOWN the single tiny_instance.json with target = s tier "
             "(then divided by --tiny-scale-down-divisor on each REDO).",
    )
    # Scale-up mode options
    parser.add_argument(
        "--scale-up-override-factor", type=float, default=1.5,
        help="For papers already at the [STRICT] ceiling (l-tier) — i.e. "
             "Sub-B — multiplier to apply when computing the scale-up target "
             "(default: 1.5).",
    )
    parser.add_argument(
        "--scale-up-skip-at-ceiling", "--scale-up-skip-at-cap",
        action="store_true", dest="scale_up_skip_at_ceiling",
        help="In scale-up mode, skip papers that are already AT the ceiling "
             "(l-tier) instead of applying the override factor. The "
             "--scale-up-skip-at-cap alias is kept for backward compatibility.",
    )
    # Scale-down mode options
    parser.add_argument(
        "--scale-down-floor-divisor", type=float, default=1.5,
        help="In scale-down mode, divisor applied to current scale when the "
             "paper is already at or below the m tier (default: 1.5).",
    )
    # Tiny-scale-down mode options
    parser.add_argument(
        "--tiny-scale-down-divisor", type=float, default=1.5,
        help="In tiny-scale-down mode, divisor applied on each REDO when the "
             "smoke-test still showed gurobi >100s on the tiny (default: 1.5). "
             "Iter 1 target = s tier; iter k target = s tier / divisor^(k-1).",
    )
    parser.add_argument(
        "--no-sync", action="store_true",
        help="In scale-up mode, do NOT auto-sync the new j=1 files from "
             "data/paper_data/<pid>/instance/ to "
             "frontier-or/<pid>/instance/. Default is to sync after "
             "each paper's agent finishes.",
    )
    # Smoke-test loop options
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="In scale-up mode, run a Gurobi smoke test (5 instances in "
             "parallel, time_limit=100s) on each freshly-generated j=1 row. "
             "If all 5 solve to optimal under the budget, REVERT the rotation, "
             "bump the multiplier by --smoke-test-multiplier-growth, and "
             "regenerate. Repeat up to --smoke-test-max-iterations.",
    )
    parser.add_argument(
        "--smoke-test-time-limit", type=int, default=100,
        help="Per-instance Gurobi time_limit during the smoke test (default: 100s).",
    )
    parser.add_argument(
        "--smoke-test-multiplier-growth", type=float, default=1.5,
        help="Factor to multiply the scale-up multiplier on each retry "
             "(default: 1.5).",
    )
    parser.add_argument(
        "--smoke-test-max-iterations", type=int, default=5,
        help="Max iterations of generate-and-test (default: 5).",
    )
    # Replace mode options
    parser.add_argument(
        "--targets", type=str, default=None,
        help='JSON dict mapping paper_id to list of instance IDs to replace. '
             'E.g. \'{"borndorfer2007": [2], "contreras2011": [6]}\'',
    )
    # Add mode options
    parser.add_argument(
        "--add-direction", choices=["up", "down", "mid"], default="up",
        help="Direction to add instances (default: up).",
    )
    parser.add_argument(
        "--add-count", type=int, default=1,
        help="Number of instances to add per paper (default: 1).",
    )
    # Custom mode
    parser.add_argument(
        "--instructions-file", type=str, default=None,
        help="Path to JSON file with per-paper instructions: "
             '{paper_id: "instruction text", ...}',
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1).",
    )
    # Judge agent (smoke-fail diagnostic before bumping multiplier)
    parser.add_argument(
        "--judge-before-bump", action="store_true",
        help="In smoke-test mode (scale-up/scale-down), call a judge agent on "
             "each smoke failure to decide between (a) continuing with the "
             "blind multiplier bump, (b) applying a targeted parameter "
             "override on the existing instance via patch_applier, or "
             "(c) giving up. Without this flag, smoke failures always bump "
             "the multiplier.",
    )
    parser.add_argument(
        "--judge-prompt-path", type=str,
        default=os.path.join(BASE_DIR, "prompts", "paper_reproduce",
                              "prompt_judge_scale_decision.txt"),
        help="Judge prompt template path (used in scale-down / replace).",
    )
    parser.add_argument(
        "--judge-prompt-path-scale-up", type=str,
        default=os.path.join(BASE_DIR, "prompts", "paper_reproduce",
                              "prompt_judge_scale_up_decision.txt"),
        help="Judge prompt template path used in scale-up mode (the "
             '"too easy / 5/5 fast" diagnostic).',
    )
    parser.add_argument(
        "--force-manual", action="store_true",
        help="Implies --judge-before-bump. Strip the 'scale' option from the "
             "judge prompt so the agent must choose 'manual' (apply a "
             "targeted patch) or 'giveup' on every smoke failure. Use this "
             "when the blind multiplier bump is empirically not improving "
             "incumbents for the paper class. If the agent disobeys and "
             "still returns 'scale', it is downgraded to 'giveup' rather "
             "than falling through to the multiplier bump.",
    )
    parser.add_argument(
        "--enable-compare-tiny", "--enable_compare_tiny",
        dest="enable_compare_tiny", action="store_true",
        help="In scale-down/replace judge invocations, append a Tiny ↔ Large "
             "comparison block (read from --tiny-large-csv) so the agent can "
             "pick a 4th decision `compare_tiny` — i.e. propose multiply "
             "patches that move large dimensions toward tiny by a geometric "
             "midpoint. Use this when blind scale-down has cycled without "
             "producing a solvable instance and you want the agent to anchor "
             "on the known-feasible tiny dimensions.",
    )
    parser.add_argument(
        "--tiny-large-csv", "--tiny_large_csv", dest="tiny_large_csv",
        type=str,
        default=os.path.join(BASE_DIR, "tiny_large_parameters.csv"),
        help="Path to the tiny_large_parameters.csv used by "
             "--enable-compare-tiny (default: <repo>/tiny_large_parameters.csv).",
    )
    parser.add_argument(
        "--prefer-compare-tiny", "--prefer_compare_tiny",
        dest="prefer_compare_tiny", action="store_true",
        help="Implies --enable-compare-tiny. Flip the judge priority so "
             "`compare_tiny` (geometric-midpoint shrink anchored on tiny) "
             "becomes the DEFAULT decision whenever any axis ratio in the "
             "Tiny↔Large block is ≥ 2.0; `manual` is reserved for "
             "INFEAS_PROVEN cases with an unambiguous single-binding "
             "constraint. Use this when blind scale-down + manual "
             "single-binding patches have already failed and you want the "
             "agent to commit to a tiny-anchored shrink instead of guessing "
             "at a binding constraint with a generic 1.10× bump. Has no "
             "effect for papers without a tiny_large_parameters.csv row.",
    )
    parser.add_argument(
        "--fix-by-condition", "--fix_by_condition", dest="fix_by_condition",
        action="store_true",
        help="Per-instance targeting: read gurobi_results_all_new.csv at repo "
             "root; for each candidate paper (from --paper-id and/or "
             "--paper_tag) keep only those whose at least one large "
             "instance matches the condition (gurobi_time>=3600s OR "
             "'time_out') AND (gurobi_solution is N/A OR 'time_out' OR "
             "non-numeric — i.e. no incumbent). Only the matching instance "
             "ids are processed. With --mode replace, args.targets is "
             "auto-populated. With --mode scale-up/scale-down + judge, the "
             "judge's manual patches are applied ONLY to the failing "
             "instances (working ones are left untouched).",
    )
    parser.add_argument(
        "--fix-by-condition-csv", "--fix_by_condition_csv",
        dest="fix_by_condition_csv", type=str,
        default=os.path.join(BASE_DIR, "gurobi_results_all_new.csv"),
        help="CSV path used by --fix-by-condition. Default: "
             "gurobi_results_all_new.csv at repo root.",
    )
    args = parser.parse_args()

    # --force-manual implies --judge-before-bump (the override is meaningless
    # without the judge in the loop).
    if args.force_manual and not args.judge_before_bump:
        args.judge_before_bump = True
    # --prefer-compare-tiny implies --enable-compare-tiny (the priority flip
    # is meaningless without the Tiny↔Large block being injected).
    if args.prefer_compare_tiny and not args.enable_compare_tiny:
        args.enable_compare_tiny = True

    if HARDCODED_PAPER_IDS:
        paper_ids = list(HARDCODED_PAPER_IDS)
    elif args.paper_ids:
        paper_ids = args.paper_ids
    else:
        paper_ids = discover_paper_ids()

    if args.paper_tag:
        results_csv = os.path.join(BASE_DIR, "gurobi_results_all_new.csv")
        if not os.path.isfile(results_csv):
            print(f"ERROR: --paper_tag requires {results_csv} (not found).",
                  file=sys.stderr)
            sys.exit(1)
        import csv as _csv
        tagged_set = set()
        with open(results_csv) as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            if not header or len(header) < 2 or header[1] != "tag":
                print(f"ERROR: {results_csv} missing expected 'tag' column "
                      f"at index 1 (got header: {header}).", file=sys.stderr)
                sys.exit(1)
            for row in reader:
                if len(row) >= 2 and row[1].strip() == args.paper_tag:
                    tagged_set.add(row[0].strip())
        if not tagged_set:
            print(f"ERROR: no papers match --paper_tag {args.paper_tag!r} in "
                  f"{results_csv}.", file=sys.stderr)
            sys.exit(1)
        n_before = len(paper_ids)
        paper_ids = [p for p in paper_ids if p in tagged_set]
        print(f"--paper_tag {args.paper_tag}: filtered "
              f"{n_before} → {len(paper_ids)} paper(s) "
              f"(matching tag set has {len(tagged_set)} entries)",
              file=sys.stderr)

    if HARDCODED_PAPER_EXCLUDE_IDS:
        hard_exclude = {e.lower() for e in HARDCODED_PAPER_EXCLUDE_IDS}
        paper_ids = [p for p in paper_ids if p.lower() not in hard_exclude]
    if args.exclude_paper_ids:
        exclude_set = {e.lower() for e in args.exclude_paper_ids}
        paper_ids = [p for p in paper_ids if p.lower() not in exclude_set]

    if not paper_ids:
        print("ERROR: No paper_ids to process.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # --fix-by-condition: restrict paper_ids to those that have at least
    # one instance matching the TLE-no-incumbent condition, and remember
    # the per-paper failing-instance index list for downstream use
    # (args.targets in replace mode; apply_judge_patches scope in
    # scale-up/down + judge mode).
    # ------------------------------------------------------------------
    args.fix_by_condition_targets = None
    if args.fix_by_condition:
        try:
            cond_targets = compute_fix_by_condition_targets(
                paper_ids, args.fix_by_condition_csv)
        except FileNotFoundError as e:
            print(f"ERROR: --fix-by-condition: {e}", file=sys.stderr)
            sys.exit(1)
        if not cond_targets:
            print("[FIX-BY-CONDITION] no candidate papers had any instance "
                  "matching the failure condition (time>=3600s/'time_out' AND "
                  "sol N/A/'time_out'/non-numeric). Nothing to do.",
                  file=sys.stderr)
            sys.exit(0)
        # If the user ALSO passed --targets explicitly, intersect.
        if args.targets:
            try:
                user_targets = json.loads(args.targets)
            except json.JSONDecodeError as e:
                print(f"ERROR: --targets is not valid JSON: {e}",
                      file=sys.stderr)
                sys.exit(1)
            for pid in list(cond_targets.keys()):
                u = set(user_targets.get(pid, []))
                if u:
                    inter = sorted(u & set(cond_targets[pid]))
                    if inter:
                        cond_targets[pid] = inter
                    else:
                        del cond_targets[pid]
        # Restrict paper_ids to those with surviving failing instances.
        paper_ids = [p for p in paper_ids if p in cond_targets]
        if not paper_ids:
            print("[FIX-BY-CONDITION] all candidate papers were filtered "
                  "out after intersecting with --targets. Nothing to do.",
                  file=sys.stderr)
            sys.exit(0)
        print(f"[FIX-BY-CONDITION] {len(paper_ids)} paper(s) selected:")
        for pid in paper_ids:
            print(f"  - {pid}: failing instances {cond_targets[pid]}")
        # In replace mode, auto-populate args.targets so the existing
        # build_replace_instructions path picks them up unchanged.
        if args.mode == "replace":
            args.targets = json.dumps(cond_targets)
        # Stash the map for scale-up/down + judge invocations to scope
        # apply_judge_patches.
        args.fix_by_condition_targets = cond_targets

    instructions_map: dict[str, str] = {}

    if args.mode == "replace":
        if not args.targets:
            print("ERROR: --targets required for replace mode (or pass "
                  "--fix-by-condition to auto-derive them).",
                  file=sys.stderr)
            sys.exit(1)
        targets = json.loads(args.targets)
        for pid in paper_ids:
            if pid in targets:
                instructions_map[pid] = build_replace_instructions(pid, targets[pid])
    elif args.mode == "add":
        for pid in paper_ids:
            paper_dir = os.path.join(PAPER_DATA_DIR, pid)
            if os.path.isdir(paper_dir):
                instructions_map[pid] = build_add_instructions(
                    pid, args.add_direction, args.add_count, paper_dir)
    elif args.mode == "custom":
        if not args.instructions_file:
            print("ERROR: --instructions-file required for custom mode.", file=sys.stderr)
            sys.exit(1)
        with open(args.instructions_file, "r", encoding="utf-8") as f:
            instructions_map = json.load(f)
        # Filter to requested paper_ids
        instructions_map = {k: v for k, v in instructions_map.items() if k in paper_ids}
    elif args.mode in ("scale-up", "scale-down", "tiny-scale-down"):
        skipped_at_cap = []
        rotated = {}
        cond_map_outer = getattr(args, "fix_by_condition_targets", None) or {}
        for pid in paper_ids:
            scope_outer = (cond_map_outer.get(pid)
                           if args.mode == "scale-down"
                           and cond_map_outer.get(pid)
                           else None)
            if args.mode == "scale-up":
                instr = build_scale_up_instructions(
                    pid, args.scale_up_override_factor, args.scale_up_skip_at_ceiling)
            elif args.mode == "scale-down":
                instr = build_scale_down_instructions(
                    pid, 1.0, args.scale_down_floor_divisor,
                    scope_indices=scope_outer)
            else:  # tiny-scale-down
                instr = build_tiny_scale_down_instructions(
                    pid, 1.0, args.tiny_scale_down_divisor)
            if instr is None:
                paper_dir = os.path.join(PAPER_DATA_DIR, pid)
                spec_path = os.path.join(paper_dir, "scale_diversity_parameters.txt")
                if not os.path.isfile(spec_path):
                    print(f"  [SKIP] {pid}: missing scale_diversity_parameters.txt",
                          file=sys.stderr)
                else:
                    skipped_at_cap.append(pid)
                    print(f"  [SKIP] {pid}: skipped per --scale-up-skip-at-cap",
                          file=sys.stderr)
                continue
            instructions_map[pid] = instr
            # Pre-step: rotate files NOW only for the one-shot path. Smoke-test
            # mode runs its own iterative rotate/revert loop and would
            # double-rotate if we did it here.
            if not args.smoke_test:
                try:
                    if args.mode == "tiny-scale-down":
                        ws_moves = rotate_tiny_instance_file(
                            os.path.join(PAPER_DATA_DIR, pid, "instance"))
                        bench_removes = clear_bench_tiny(
                            os.path.join(BENCH_DIR, pid, "instance"))
                    else:
                        ws_moves = rotate_instance_files(
                            os.path.join(PAPER_DATA_DIR, pid, "instance"),
                            scope_indices=scope_outer)
                        bench_removes = clear_bench_j1_files(
                            os.path.join(BENCH_DIR, pid, "instance"),
                            scope_indices=scope_outer)
                    rotated[pid] = (len(ws_moves), len(bench_removes))
                    print(f"  [ROTATE] {pid}: workspace +{len(ws_moves)} moves, "
                          f"bench cleared {len(bench_removes)} file(s)", file=sys.stderr)
                except RuntimeError as e:
                    print(f"  [ABORT] {pid}: rotation failed — {e}", file=sys.stderr)
                    instructions_map.pop(pid, None)
                    continue
        if skipped_at_cap:
            print(f"\n{args.mode}: skipped {len(skipped_at_cap)} paper(s)",
                  file=sys.stderr)
        if rotated:
            print(f"{args.mode}: rotated files for {len(rotated)} paper(s)",
                  file=sys.stderr)

    if not instructions_map:
        print("ERROR: No update instructions generated for any paper.", file=sys.stderr)
        sys.exit(1)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    def process_paper(paper_id: str, update_instructions: str):
        paper_dir = os.path.join(PAPER_DATA_DIR, paper_id)
        if not os.path.isdir(paper_dir):
            print(f"[WARN] Skipping '{paper_id}': directory not found.", file=sys.stderr)
            return

        print(f"[START] paper_id: {paper_id}")
        prompt_body = (prompt_template
                       .replace("{paper_id}", paper_id)
                       .replace("{update_instructions}", update_instructions))
        full_prompt = (
            f"Please read the files in the directory at the absolute path below:\n"
            f"{paper_dir}\n\n"
            f"{prompt_body}\n\n"
            f"All outputs must be saved inside the absolute path below:\n"
            f"{paper_dir}\n"
        )
        try:
            run_claude_tracked(full_prompt, label=f"update_instances:{paper_id}")
            print(f"[DONE]  paper_id: {paper_id}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[WARN]  claude failed for '{paper_id}': {e}", file=sys.stderr)
            return

        # Post-step: sync freshly-written file(s) to the benchmark dir.
        # scale-up / scale-down → 5 large j=1 files;
        # tiny-scale-down       → single tiny_instance.json.
        if args.mode in ("scale-up", "scale-down") and not args.no_sync:
            cond_map_proc = getattr(args, "fix_by_condition_targets", None) or {}
            scope_proc = (cond_map_proc.get(paper_id)
                          if args.mode == "scale-down"
                          and cond_map_proc.get(paper_id)
                          else None)
            synced = sync_new_j1_to_bench(paper_id, scope_indices=scope_proc)
            if synced:
                print(f"[SYNC]  {paper_id}: {len(synced)} file(s) → "
                      f"{os.path.join(BENCH_DIR, paper_id, 'instance')}")
            else:
                print(f"[SYNC-MISS] {paper_id}: no new j=1 files found in "
                      f"{os.path.join(PAPER_DATA_DIR, paper_id, 'instance')}",
                      file=sys.stderr)
        elif args.mode == "tiny-scale-down" and not args.no_sync:
            synced = sync_tiny_to_bench(paper_id)
            if synced:
                print(f"[SYNC]  {paper_id}: tiny_instance.json → "
                      f"{os.path.join(BENCH_DIR, paper_id, 'instance')}")
            else:
                print(f"[SYNC-MISS] {paper_id}: no tiny_instance.json found in "
                      f"{os.path.join(PAPER_DATA_DIR, paper_id, 'instance')}",
                      file=sys.stderr)

    # ----- Smoke-test branch (scale-up / scale-down / tiny-scale-down) -----
    if args.smoke_test and args.mode in ("scale-up", "scale-down",
                                          "tiny-scale-down"):
        if args.no_sync:
            print("WARN: --smoke-test implies sync (smoke test reads from bench dir); "
                  "ignoring --no-sync.", file=sys.stderr)
        smoke_targets = list(instructions_map.keys())

        def smoke_worker(pid):
            paper_dir = os.path.join(PAPER_DATA_DIR, pid)
            spec_path = os.path.join(paper_dir, "scale_diversity_parameters.txt")
            with open(spec_path, "r", encoding="utf-8") as f:
                spec_baseline = f.read()
            if args.mode == "scale-up":
                at_cap = is_paper_at_cap(spec_baseline)
                multiplier = args.scale_up_override_factor if at_cap else 1.0
                print(f"[SMOKE-START] {pid} mode=scale-up "
                      f"(initial multiplier={multiplier}, at_cap={at_cap})")
            elif args.mode == "scale-down":
                multiplier = 1.0
                print(f"[SMOKE-START] {pid} mode=scale-down "
                      f"(initial multiplier={multiplier}, "
                      f"floor_divisor={args.scale_down_floor_divisor})")
            else:  # tiny-scale-down
                multiplier = 1.0
                print(f"[SMOKE-START] {pid} mode=tiny-scale-down "
                      f"(initial multiplier={multiplier}, "
                      f"divisor={args.tiny_scale_down_divisor})")

            # --fix-by-condition + scale-down: pull the per-paper failing
            # instance scope; pipeline only rotates / regenerates / smokes
            # those, leaving instances that already have a valid incumbent
            # untouched. None on other modes / when fix-by-condition is off.
            cond_map = getattr(args, "fix_by_condition_targets", None) or {}
            scope = (cond_map.get(pid)
                     if args.mode == "scale-down" and cond_map.get(pid)
                     else None)
            if scope:
                print(f"[SMOKE-SCOPE] {pid}: fix-by-condition restricting "
                      f"to instances {scope} (others left untouched)")

            for it in range(1, args.smoke_test_max_iterations + 1):
                # Build instruction with current multiplier
                if args.mode == "scale-up":
                    instr = build_scale_up_instructions(
                        pid, multiplier, args.scale_up_skip_at_ceiling)
                elif args.mode == "scale-down":
                    instr = build_scale_down_instructions(
                        pid, multiplier, args.scale_down_floor_divisor,
                        scope_indices=scope)
                else:  # tiny-scale-down
                    instr = build_tiny_scale_down_instructions(
                        pid, multiplier, args.tiny_scale_down_divisor)
                if instr is None:
                    print(f"[SMOKE-SKIP]  {pid}: build_*_instructions "
                          f"returned None (skipped)")
                    return
                # Rotate workspace (creates audit j>=2); clear bench (no
                # audit copies kept under frontier-or/).
                try:
                    if args.mode == "tiny-scale-down":
                        rotate_tiny_instance_file(
                            os.path.join(paper_dir, "instance"))
                        clear_bench_tiny(
                            os.path.join(BENCH_DIR, pid, "instance"))
                    else:
                        rotate_instance_files(
                            os.path.join(paper_dir, "instance"),
                            scope_indices=scope)
                        clear_bench_j1_files(
                            os.path.join(BENCH_DIR, pid, "instance"),
                            scope_indices=scope)
                except RuntimeError as e:
                    print(f"[SMOKE-ABORT] {pid} iter {it}: rotation failed — {e}",
                          file=sys.stderr)
                    return
                print(f"[SMOKE-AGENT] {pid} iter {it}: multiplier={multiplier}")
                prompt_body = (prompt_template
                               .replace("{paper_id}", pid)
                               .replace("{update_instructions}", instr))
                full_prompt = (
                    f"Please read the files in the directory at the absolute path below:\n"
                    f"{paper_dir}\n\n"
                    f"{prompt_body}\n\n"
                    f"All outputs must be saved inside the absolute path below:\n"
                    f"{paper_dir}\n"
                )
                try:
                    run_claude_tracked(
                        full_prompt,
                        label=f"update_instances:{pid}:smoke{it}")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    # The rotation (j=1 → j=2, bench j=1 cleared) already ran
                    # BEFORE the agent call. If the agent fails (e.g. claude
                    # CLI returns exit 1 on API rate-limit), returning here
                    # without reverting leaves the bench dir EMPTY. Restore
                    # the rotation so bench holds the pre-rotation baseline.
                    print(f"[SMOKE-FAIL] {pid} iter {it}: claude failed — {e}; "
                          f"reverting rotation to restore bench baseline.",
                          file=sys.stderr)
                    if args.mode == "tiny-scale-down":
                        revert_tiny_rotation(pid, spec_baseline)
                    else:
                        revert_scale_up_rotation(pid, spec_baseline,
                                                 scope_indices=scope)
                    return
                # Sync new file(s) to bench so gurobi reads the latest
                if args.mode == "tiny-scale-down":
                    synced = sync_tiny_to_bench(pid)
                    if not synced:
                        print(f"[SMOKE-SYNCMISS] {pid} iter {it}: no tiny "
                              f"to sync; agent may have failed to write output; "
                              f"reverting rotation to restore bench baseline.",
                              file=sys.stderr)
                        revert_tiny_rotation(pid, spec_baseline)
                        return
                else:
                    synced = sync_new_j1_to_bench(pid, scope_indices=scope)
                    if not synced:
                        print(f"[SMOKE-SYNCMISS] {pid} iter {it}: no new j=1 "
                              f"files to sync; agent may have failed to write "
                              f"outputs; reverting rotation to restore bench "
                              f"baseline.", file=sys.stderr)
                        revert_scale_up_rotation(pid, spec_baseline,
                                                 scope_indices=scope)
                        return
                # Gurobi smoke test (large j=1, scoped if --fix-by-condition;
                # else all 5 — or 1 tiny)
                if args.mode == "tiny-scale-down":
                    result = run_gurobi_smoke_test_tiny(
                        pid, time_limit=args.smoke_test_time_limit)
                    wall = result["wall_time"]
                    wall_s = (f"{wall:.1f}s" if wall is not None else "-")
                    print(f"[SMOKE-TEST]  {pid} iter {it}: "
                          f"status={result['status']}/{wall_s}")
                else:
                    results = run_gurobi_smoke_test(
                        pid, time_limit=args.smoke_test_time_limit,
                        j_index=1, indices=scope)
                    summary = ", ".join(
                        f"i={r['i']}:{r['status']}/{r['wall_time']:.1f}s"
                        if r['wall_time'] is not None
                        else f"i={r['i']}:{r['status']}/-"
                        for r in results
                    )
                    print(f"[SMOKE-TEST]  {pid} iter {it}: {summary}")
                # Decide KEEP / REDO / GAVEUP. Large modes: KEEP if not
                # all_solved_fast (some hard enough). Tiny: KEEP if solved_fast.
                if args.mode == "tiny-scale-down":
                    is_keep = tiny_solved_fast(result, args.smoke_test_time_limit)
                else:
                    is_keep = not all_solved_optimal_fast(
                        results, args.smoke_test_time_limit)

                # ---- Judge-agent intervention (large modes only) ----
                # Two trigger conditions, depending on direction:
                #   • scale-down / replace: any instance has no solution
                #     (INFEAS / 0-incumbent / killed) — judge decides
                #     whether targeted patch can repair it, or the
                #     multiplier bump should proceed, or it's unfixable.
                #   • scale-up: 5/5 solved too fast (is_keep is already
                #     False, REDO is about to fire) — judge decides
                #     whether a targeted patch (e.g. tighten a slack
                #     cap) is more effective than blind multiplier
                #     × growth, or the model is at structural ceiling.
                # Skipped for tiny mode (tuning-loop, no INFEAS axis).
                if (args.judge_before_bump
                        and args.mode != "tiny-scale-down"):
                    judge_target = None  # (instance_idx, smoke_log, prompt_path)
                    no_sol = [r for r in results
                              if r.get("obj") is None
                              or r.get("status") in (
                                  "infeasible", "no_solution", "missing",
                                  "killed", "unparsable")]
                    if args.mode == "scale-up":
                        if not is_keep:  # 5/5 fast → about to REDO blindly
                            first_i = results[0]["i"]
                            log_path = os.path.join(
                                "/tmp", f"_smoke_{pid}_{first_i}1.log")
                            smoke_log = ""
                            if os.path.isfile(log_path):
                                with open(log_path) as f:
                                    smoke_log = f.read()
                            else:
                                smoke_log = json.dumps(results)
                            print(f"[JUDGE]      {pid} iter {it}: "
                                  f"5/5 solved < {args.smoke_test_time_limit}s "
                                  f"(too easy) — invoking scale-up judge")
                            judge_target = (
                                first_i, smoke_log,
                                args.judge_prompt_path_scale_up)
                    else:  # scale-down / replace
                        if no_sol:
                            first_i = no_sol[0]["i"]
                            log_path = os.path.join(
                                "/tmp", f"_smoke_{pid}_{first_i}1.log")
                            smoke_log = ""
                            if os.path.isfile(log_path):
                                with open(log_path) as f:
                                    smoke_log = f.read()
                            else:
                                smoke_log = json.dumps(no_sol[0])
                            print(f"[JUDGE]      {pid} iter {it}: "
                                  f"{len(no_sol)}/{len(results)} instance(s) "
                                  f"have no solution — invoking judge agent")
                            judge_target = (
                                first_i, smoke_log, args.judge_prompt_path)

                    if judge_target is not None:
                        first_i, smoke_log, prompt_path = judge_target
                        try:
                            decision = call_judge_agent(
                                pid, f"{first_i}1", smoke_log,
                                args.smoke_test_time_limit,
                                prompt_path,
                                force_manual=args.force_manual,
                                tiny_large_csv=(
                                    args.tiny_large_csv
                                    if args.enable_compare_tiny else None),
                                prefer_compare_tiny=args.prefer_compare_tiny)
                        except Exception as e:
                            # Judge call itself failed (claude API error,
                            # parse error, etc.). Initial smoke had no_sol
                            # so without the judge's intervention there's
                            # no fix path. Revert + exit honestly to avoid
                            # the outer-`is_keep` false-KEEP bug.
                            print(f"[JUDGE-FAIL] {pid} iter {it}: "
                                  f"judge invocation failed — {e}; "
                                  f"reverting bench to baseline.",
                                  file=sys.stderr)
                            revert_scale_up_bench_only(
                                pid, scope_indices=scope)
                            return
                        else:
                            decision_kind = decision.get("decision")
                            # Under --force-manual, "scale" is not allowed.
                            # Downgrade rogue "scale" verdicts to "giveup"
                            # so we don't silently slide back into the blind
                            # multiplier path the operator disabled.
                            if (args.force_manual
                                    and decision_kind == "scale"):
                                print(f"[JUDGE-OVERRIDE] {pid} iter {it}: "
                                      f"agent returned 'scale' under "
                                      f"--force-manual; downgrading to "
                                      f"'giveup'", file=sys.stderr)
                                decision_kind = "giveup"
                            print(f"[JUDGE]      {pid} iter {it}: "
                                  f"decision={decision_kind!r} "
                                  f"reason={decision.get('reason', '')[:120]}")
                            if decision_kind == "giveup":
                                print(f"[JUDGE-GAVEUP] {pid} iter {it}: "
                                      f"agent declined to act. Reverting "
                                      f"bench to baseline.", file=sys.stderr)
                                revert_scale_up_bench_only(
                                    pid, scope_indices=scope)
                                return
                            # `manual` and `compare_tiny` share the same
                            # patch-apply + re-smoke flow; only the log
                            # label differs (so we can tell which decision
                            # the agent took from logs).
                            if decision_kind in ("manual", "compare_tiny"):
                                label = ("JUDGE-PATCH" if decision_kind ==
                                         "manual" else "JUDGE-COMPARE-TINY")
                                try:
                                    # If --fix-by-condition was set, only
                                    # patch the failing instances for this
                                    # paper. Otherwise patch all 5.
                                    cond_map = getattr(
                                        args, "fix_by_condition_targets",
                                        None) or {}
                                    scope_idx = cond_map.get(pid)
                                    n = apply_judge_patches(
                                        pid, decision, scope_indices=scope_idx)
                                    scope_label = (f"{len(scope_idx)} scoped "
                                                   f"instances {scope_idx}"
                                                   if scope_idx else
                                                   "5 instances")
                                    print(f"[{label}] {pid} iter {it}: "
                                          f"applied {n} patch-changes across "
                                          f"{scope_label} — re-running smoke")
                                except Exception as e:
                                    # Patch sanity / wildcard mismatch
                                    # rejected the agent's plan. The
                                    # initial smoke had no_sol, so without
                                    # the patches we still don't have a
                                    # usable scoped instance. Don't slide
                                    # into the outer-`is_keep` false-KEEP
                                    # bug — revert and exit honestly so
                                    # bench reads the original baseline.
                                    print(f"[{label}-FAIL] {pid} iter "
                                          f"{it}: {e}; reverting bench "
                                          f"to baseline (judge could not "
                                          f"act on agent's regen).",
                                          file=sys.stderr)
                                    revert_scale_up_bench_only(
                                        pid, scope_indices=scope)
                                    return
                                else:
                                    # Re-smoke after patch (scoped if --fix-by-condition)
                                    results = run_gurobi_smoke_test(
                                        pid,
                                        time_limit=args.smoke_test_time_limit,
                                        j_index=1, indices=scope)
                                    summary = ", ".join(
                                        f"i={r['i']}:{r['status']}/"
                                        f"{r['wall_time']:.1f}s"
                                        if r['wall_time'] is not None
                                        else f"i={r['i']}:{r['status']}/-"
                                        for r in results)
                                    print(f"[JUDGE-RESMOKE] {pid} iter {it}: "
                                          f"{summary}")
                                    # Three-way assessment:
                                    #   1. SUCCEEDED — every scoped instance
                                    #      produced a usable incumbent in <
                                    #      smoke budget → strong KEEP.
                                    #   2. BROKE_MODEL — patches introduced
                                    #      a regression (INFEAS, no_output,
                                    #      OOM/crash with wall_time well
                                    #      below budget) → REVERT.
                                    #   3. else — patches ran the full smoke
                                    #      budget without crashing or going
                                    #      INFEAS, just didn't OPT in the
                                    #      tight 100s smoke → SOFT KEEP. The
                                    #      1h gurobi rerun will judge the
                                    #      real impact; smoke is not a fail
                                    #      gate for OOM-prone / hard models.
                                    if judge_patch_succeeded(
                                            results,
                                            args.smoke_test_time_limit):
                                        # Guard against over-shrink: if every
                                        # instance OPTed in <10s the agent
                                        # likely over-truncated, landing
                                        # tag C ("trivially solvable"). Revert
                                        # and REDO with less aggression.
                                        if judge_patches_overfit(
                                                results,
                                                args.smoke_test_time_limit,
                                                trivial_threshold=10.0):
                                            more_iters = (it <
                                                args.smoke_test_max_iterations)
                                            tail = ("falling through to next "
                                                    "iter with less "
                                                    "aggressive patch."
                                                    if more_iters else
                                                    "no more iters; outer "
                                                    "GAVEUP will fire.")
                                            print(f"[JUDGE-OVERFIT] {pid} "
                                                  f"iter {it}: {decision_kind}"
                                                  f" patches OVER-shrunk "
                                                  f"model (5/5 OPT in <10s; "
                                                  f"would land tag C, not "
                                                  f"the intended B/A); "
                                                  f"reverting bench and "
                                                  f"forcing REDO. {tail}",
                                                  file=sys.stderr)
                                            revert_scale_up_bench_only(
                                                pid, scope_indices=scope)
                                            is_keep = False
                                            # Fall through to outer
                                            # is_keep=False → REDO branch.
                                        else:
                                            print(f"[SMOKE-KEEP]  {pid} iter "
                                                  f"{it}: judge "
                                                  f"{decision_kind} "
                                                  f"successfully fixed "
                                                  f"scoped instances within "
                                                  f"budget.")
                                            return
                                    elif judge_patch_broke_model(
                                            results,
                                            args.smoke_test_time_limit):
                                        # Revert this iter's broken state but
                                        # do NOT hard-return — fall through
                                        # to the outer loop so the remaining
                                        # iterations (REDO bump + new agent
                                        # regen) get a chance. If this is
                                        # the last iter, the outer GAVEUP
                                        # handler naturally fires.
                                        more_iters = (it <
                                            args.smoke_test_max_iterations)
                                        tail = ("falling through to next "
                                                "iter REDO."
                                                if more_iters else
                                                "no more iters; outer GAVEUP "
                                                "will fire.")
                                        print(f"[JUDGE-BROKE-MODEL] {pid} "
                                              f"iter {it}: {decision_kind} "
                                              f"patches introduced a "
                                              f"regression (INFEAS / OOM / "
                                              f"crash) on scoped instances "
                                              f"({summary}); reverting to "
                                              f"baseline. {tail}",
                                              file=sys.stderr)
                                        revert_scale_up_bench_only(
                                            pid, scope_indices=scope)
                                        # Force REDO at the bottom of the
                                        # loop body: outer is_keep was set
                                        # by initial smoke result — must
                                        # become False so REDO/GAVEUP fires.
                                        is_keep = False
                                        # Do NOT return — let outer loop
                                        # process is_keep=False branch.
                                    else:
                                        print(f"[SMOKE-KEEP-SOFT] {pid} "
                                              f"iter {it}: {decision_kind} "
                                              f"patches applied; smoke ran "
                                              f"full budget without "
                                              f"regressing ({summary}). "
                                              f"Persisting; user 1h gurobi "
                                              f"rerun will verify real "
                                              f"impact.")
                                        return
                            # decision_kind == "scale" → fall through to the
                            # original is_keep so the outer loop's natural
                            # KEEP/REDO logic decides. Do NOT force is_keep
                            # here — scale-down + 5/5 TLE → KEEP (cannot
                            # reduce further), scale-up + 5/5 fast → REDO
                            # (bump up).

                if is_keep:
                    if args.mode == "tiny-scale-down":
                        print(f"[SMOKE-KEEP]  {pid} iter {it}: tiny solved fast "
                              f"— keeping multiplier={multiplier}, "
                              f"target reached.")
                    else:
                        print(f"[SMOKE-KEEP]  {pid} iter {it}: hard enough — "
                              f"keeping multiplier={multiplier}, target reached.")
                    return
                # REDO path: revert and bump multiplier in the appropriate
                # direction (×growth for scale-up/down, /divisor for tiny).
                if it < args.smoke_test_max_iterations:
                    if args.mode == "tiny-scale-down":
                        new_mult = multiplier / args.tiny_scale_down_divisor
                        print(f"[SMOKE-REDO]  {pid} iter {it}: tiny still "
                              f">{args.smoke_test_time_limit - 1}s; reverting "
                              f"and reducing multiplier from {multiplier:.4f} "
                              f"to {new_mult:.4f}")
                        revert_tiny_rotation(pid, spec_baseline)
                        multiplier = new_mult
                    else:
                        new_mult = multiplier * args.smoke_test_multiplier_growth
                        print(f"[SMOKE-REDO]  {pid} iter {it}: all 5 optimal "
                              f"under {args.smoke_test_time_limit}s; reverting "
                              f"and bumping multiplier to {new_mult}")
                        revert_scale_up_rotation(pid, spec_baseline,
                                                  scope_indices=scope)
                        multiplier = new_mult
                else:
                    if args.mode == "tiny-scale-down":
                        print(f"[SMOKE-GAVEUP] {pid}: exceeded "
                              f"{args.smoke_test_max_iterations} iterations; "
                              f"tiny still slow at multiplier={multiplier:.4f}. "
                              f"Bench dir reverted to baseline (eval reads "
                              f"original tiny); workspace + spec RETAINED at "
                              f"this iter's tiny for future continuation.",
                              file=sys.stderr)
                        revert_tiny_bench_only(pid)
                    else:
                        print(f"[SMOKE-GAVEUP] {pid}: exceeded "
                              f"{args.smoke_test_max_iterations} iterations; "
                              f"final multiplier was {multiplier}. Bench dir "
                              f"reverted to baseline (eval reads originals); "
                              f"workspace + spec RETAINED at this iter's scale "
                              f"for future continuation.", file=sys.stderr)
                        revert_scale_up_bench_only(pid, scope_indices=scope)
                    return

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(smoke_worker, pid): pid for pid in smoke_targets}
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] {pid}: {e}", file=sys.stderr)
        return

    items = list(instructions_map.items())
    print(f"Processing {len(items)} paper(s) with {args.workers} worker(s).")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_paper, pid, instr): pid
            for pid, instr in items
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] paper_id '{paper_id}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
