"""
Unified pipeline runner for paper reproduction.

Loop orders (--order):
    paper  (default)  For each paper, run all selected scripts before moving
                      on to the next paper.  Supports paper-level parallelism
                      via --workers.
    script            For each script, run it across all papers before moving
                      on to the next script.

Scripts (by index):
    0  scripts/paper_reproduce/run_generate_instances.py   Generate test instances
    1  scripts/paper_reproduce/run_extract_model_algo.py   Extract model & algorithm from papers
    2  scripts/paper_reproduce/run_generate_programs.py    Generate gurobi_code.py & efficient_algorithm.py
    3  scripts/paper_reproduce/run_program_solutions.py    Run programs and compute optimality gaps
    4  scripts/paper_reproduce/run_feasibility_check.py    Generate & run feasibility checks via Claude

Paper selection priority (highest to lowest):
    1) HARDCODED_PAPER_IDS       (if non-empty, overrides --paper-id)
    2) --paper-id CLI arguments  (if provided)
    3) Neither is passed         → paper-first: auto-discover from data/paper_data
                                 → script-first: sub-scripts fall back to their own logic
    Exclusions (HARDCODED_PAPER_EXCLUDE_IDS + --exclude-paper-id) are always applied.

Usage examples:
    # Paper-first (default): run all scripts per paper
    python run_pipeline.py

    # Script-first: run each script across all papers
    python run_pipeline.py --order script

    # Run only scripts 0-2 for specific papers, 3 papers in parallel:
    python run_pipeline.py --run_scripts 0-2 --paper-id mingozzi1999 amaldi2013 --workers 3

    # Resume from a specific paper (paper-first only):
    python run_pipeline.py --continue-from adulyasak2015

    # All arguments combined:
    python run_pipeline.py --order paper --run_scripts 0-4 --paper-id mingozzi1999 \
                           --exclude-paper-id roberti2021 \
                           --num_instances 10 --workers 2 \
                           --instances tiny large_1 --time_limit 3600
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
USAGE_LOG = os.path.join(BASE_DIR, "token_usage.jsonl")
PAPER_DATA_DIR = os.path.join(BASE_DIR, "data", "paper_data")

# Hardcoded paper_id controls. Edit to select / exclude papers without
# changing CLI args. Leave as [] to fall back to CLI / auto-discovery.
HARDCODED_PAPER_IDS: list[str] = []

HARDCODED_PAPER_EXCLUDE_IDS: list[str] = []

SCRIPTS = [
    os.path.join(BASE_DIR, "scripts", "paper_reproduce", "run_generate_instances.py"),
    os.path.join(BASE_DIR, "scripts", "paper_reproduce", "run_extract_model_algo.py"),
    os.path.join(BASE_DIR, "scripts", "paper_reproduce", "run_generate_programs.py"),
    os.path.join(BASE_DIR, "scripts", "paper_reproduce", "run_program_solutions.py"),
    os.path.join(BASE_DIR, "scripts", "paper_reproduce", "run_feasibility_check.py"),
]

SCRIPT_NAMES = [
    "run_generate_instances.py",
    "run_extract_model_algo.py",
    "run_generate_programs.py",
    "run_program_solutions.py",
    "run_feasibility_check.py",
]

SCRIPT_ACCEPTS = {
    0: {"paper_id", "exclude_paper_id", "num_instances", "workers"},
    1: {"paper_id", "exclude_paper_id", "workers"},
    2: {"paper_id", "exclude_paper_id", "num_instances", "workers"},
    3: {"paper_id", "exclude_paper_id", "instances", "time_limit", "rerun_null", "solution_workers"},
    4: {"paper_id", "exclude_paper_id", "workers"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_run_scripts(value: str) -> tuple[int, int]:
    """Parse --run_scripts value like '0', '0-3', '1-2'. Returns (start, end) inclusive."""
    parts = value.split("-")
    if len(parts) == 1:
        idx = int(parts[0])
        return idx, idx
    elif len(parts) == 2:
        return int(parts[0]), int(parts[1])
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid format '{value}'. Expected N or N-M (e.g. '0', '0-3', '1-2')."
        )


def discover_paper_ids() -> list[str]:
    """Auto-discover paper IDs from data/paper_data directories."""
    if not os.path.isdir(PAPER_DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(PAPER_DATA_DIR)
        if os.path.isdir(os.path.join(PAPER_DATA_DIR, d))
    )


# Thread-safe print lock (used in paper-first parallel mode)
_print_lock = threading.Lock()


def _log(msg: str, *, file=sys.stdout):
    with _print_lock:
        print(msg, flush=True, file=file)


def print_token_summary(tag: str = "run_pipeline"):
    """Read token_usage.jsonl and print an aggregated summary."""
    if not os.path.isfile(USAGE_LOG):
        return

    entries = []
    with open(USAGE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        print(f"\n[{tag}] No token usage data recorded.")
        return

    total_cost = sum(e.get("cost_usd") or 0 for e in entries)
    total_input = sum(e.get("input_tokens") or 0 for e in entries)
    total_output = sum(e.get("output_tokens") or 0 for e in entries)
    total_duration = sum(e.get("duration_ms") or 0 for e in entries)

    print(f"\n{'='*60}")
    print(f"[{tag}] Token Usage Summary")
    print(f"{'='*60}")

    by_script = defaultdict(list)
    for e in entries:
        label = e.get("label", "")
        script_name = label.split(":")[0] if ":" in label else "unknown"
        by_script[script_name].append(e)

    for script_name, script_entries in by_script.items():
        script_cost = sum(e.get("cost_usd") or 0 for e in script_entries)
        print(f"\n  {script_name}:")
        for e in script_entries:
            paper = e.get("label", "").split(":", 1)[-1] if ":" in e.get("label", "") else "?"
            cost = e.get("cost_usd") or 0
            turns = e.get("num_turns", "?")
            dur = e.get("duration_ms") or 0
            print(f"    {paper}: ${cost:.4f}  ({turns} turns, {dur/1000:.1f}s)")
        print(f"    Subtotal: ${script_cost:.4f}")

    print(f"\n  {'─'*40}")
    print(f"  Grand Total: ${total_cost:.4f}")
    if total_input or total_output:
        print(f"  Input tokens:  {total_input:,}")
        print(f"  Output tokens: {total_output:,}")
    print(f"  Total API duration: {total_duration/1000:.1f}s")
    print(f"{'='*60}")


# ── Command builders ──────────────────────────────────────────────────────────

def build_cmd_script_first(script_idx: int, args: argparse.Namespace) -> list[str]:
    """Build command for script-first mode (all papers at once)."""
    accepts = SCRIPT_ACCEPTS[script_idx]
    cmd = [sys.executable, "-u", SCRIPTS[script_idx]]

    if "paper_id" in accepts and args.paper_ids:
        cmd += ["--paper-id"] + args.paper_ids

    if "exclude_paper_id" in accepts and args.exclude_paper_ids:
        cmd += ["--exclude-paper-id"] + args.exclude_paper_ids

    if "num_instances" in accepts and args.num_instances is not None:
        cmd += ["--num_instances", str(args.num_instances)]

    if "workers" in accepts and args.workers is not None:
        cmd += ["--workers", str(args.workers)]

    if "instances" in accepts and args.instances is not None:
        cmd += ["--instances"] + list(args.instances)

    if "time_limit" in accepts and args.time_limit is not None:
        cmd += ["--time_limit", str(args.time_limit)]

    if "rerun_null" in accepts and args.rerun_null is not None:
        cmd += ["--rerun_null", str(args.rerun_null)]

    if "solution_workers" in accepts and args.solution_workers is not None:
        cmd += ["--workers", str(args.solution_workers)]

    return cmd


def build_cmd_paper_first(script_idx: int, paper_id: str, args: argparse.Namespace) -> list[str]:
    """Build command for paper-first mode (one paper at a time)."""
    accepts = SCRIPT_ACCEPTS[script_idx]
    cmd = [sys.executable, "-u", SCRIPTS[script_idx]]

    if "paper_id" in accepts:
        cmd += ["--paper-id", paper_id]

    if "num_instances" in accepts and args.num_instances is not None:
        cmd += ["--num_instances", str(args.num_instances)]

    if "instances" in accepts and args.instances is not None:
        cmd += ["--instances"] + list(args.instances)

    if "time_limit" in accepts and args.time_limit is not None:
        cmd += ["--time_limit", str(args.time_limit)]

    if "rerun_null" in accepts and args.rerun_null is not None:
        cmd += ["--rerun_null", str(args.rerun_null)]

    if "solution_workers" in accepts and args.solution_workers is not None:
        cmd += ["--workers", str(args.solution_workers)]

    return cmd


# ── Paper-first mode ──────────────────────────────────────────────────────────

def run_paper_pipeline(
    paper_id: str,
    paper_num: int,
    n_papers: int,
    script_indices: list[int],
    n_scripts: int,
    args: argparse.Namespace,
) -> tuple[str, bool, str]:
    """Run all selected scripts for a single paper. Returns (paper_id, success, error_msg)."""
    tag = f"[{paper_id}]"
    _log(f"\n{'#'*60}\n[run_pipeline] Paper {paper_num}/{n_papers}: {paper_id}\n{'#'*60}")

    for step_num, script_idx in enumerate(script_indices, 1):
        _log(f"\n{'─'*50}\n{tag} Step {step_num}/{n_scripts}: {SCRIPT_NAMES[script_idx]}\n{'─'*50}")

        cmd = build_cmd_paper_first(script_idx, paper_id, args)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            env=env,
        )
        for line in proc.stdout:
            _log(f"{tag} {line.rstrip()}")
        proc.wait()

        if proc.returncode != 0:
            err = f"{SCRIPT_NAMES[script_idx]} failed (exit code {proc.returncode})"
            _log(f"{tag} ERROR: {err}", file=sys.stderr)
            return paper_id, False, err

        _log(f"{tag} Step {step_num}/{n_scripts} done: {SCRIPT_NAMES[script_idx]}")

    _log(f"\n[run_pipeline] Paper {paper_num}/{n_papers} completed: {paper_id}")
    return paper_id, True, ""


def run_paper_first(args: argparse.Namespace, script_indices: list[int]):
    """Paper-first loop: for each paper, run all scripts."""
    n_scripts = len(script_indices)

    # Resolve paper list (paper-first always needs an explicit list)
    if HARDCODED_PAPER_IDS:
        paper_ids = list(HARDCODED_PAPER_IDS)
    elif args.paper_ids:
        paper_ids = list(args.paper_ids)
    else:
        paper_ids = discover_paper_ids()

    # Apply exclusions
    all_excludes: list[str] = list(HARDCODED_PAPER_EXCLUDE_IDS)
    if args.exclude_paper_ids:
        all_excludes += args.exclude_paper_ids
    if all_excludes:
        exclude_set = {e.lower() for e in all_excludes}
        paper_ids = [p for p in paper_ids if p.lower() not in exclude_set]

    if not paper_ids:
        print("[run_pipeline] ERROR: No paper IDs to process after exclusions.", file=sys.stderr)
        sys.exit(1)

    # Apply --continue-from
    if args.continue_from:
        try:
            resume_idx = paper_ids.index(args.continue_from)
            skipped = paper_ids[:resume_idx]
            paper_ids = paper_ids[resume_idx:]
            print(f"[run_pipeline] Resuming from '{args.continue_from}', skipping {len(skipped)} paper(s).")
        except ValueError:
            print(f"[run_pipeline] ERROR: --continue-from paper '{args.continue_from}' not found.", file=sys.stderr)
            sys.exit(1)

    n_papers = len(paper_ids)
    n_workers = args.workers or 1

    print(f"\n[run_pipeline] Order: paper-first")
    print(f"[run_pipeline] Papers to process: {n_papers}")
    print(f"[run_pipeline] Scripts per paper: {n_scripts} (indices {script_indices[0]}-{script_indices[-1]})")
    print(f"[run_pipeline] Paper-level parallelism: {n_workers} worker(s)")
    print(f"[run_pipeline] Total steps: {n_papers} x {n_scripts} = {n_papers * n_scripts}")
    if all_excludes:
        print(f"[run_pipeline] Excluded IDs: {', '.join(all_excludes)}")

    if n_workers <= 1:
        for paper_num, paper_id in enumerate(paper_ids, 1):
            _, ok, err = run_paper_pipeline(paper_id, paper_num, n_papers, script_indices, n_scripts, args)
            if not ok:
                print(f"\n[run_pipeline] ERROR: {paper_id}: {err}. Stopping.", file=sys.stderr)
                sys.exit(1)
    else:
        failed: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_paper_pipeline, pid, num, n_papers, script_indices, n_scripts, args): pid
                for num, pid in enumerate(paper_ids, 1)
            }
            for future in as_completed(futures):
                pid, ok, err = future.result()
                if not ok:
                    failed.append((pid, err))

        if failed:
            print(f"\n[run_pipeline] {len(failed)} paper(s) FAILED:", file=sys.stderr)
            for pid, err in failed:
                print(f"  - {pid}: {err}", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[run_pipeline] Pipeline finished: {n_papers} papers x {n_scripts} scripts.")
    print(f"{'='*60}")


# ── Script-first mode ─────────────────────────────────────────────────────────

def run_script_first(args: argparse.Namespace, start: int, end: int):
    """Script-first loop: for each script, run across all papers."""
    # Resolve paper list
    if HARDCODED_PAPER_IDS:
        args.paper_ids = list(HARDCODED_PAPER_IDS)

    all_excludes: list[str] = list(HARDCODED_PAPER_EXCLUDE_IDS)
    if args.exclude_paper_ids:
        all_excludes += args.exclude_paper_ids
    if all_excludes and args.paper_ids:
        exclude_set = {e.lower() for e in all_excludes}
        args.paper_ids = [p for p in args.paper_ids if p.lower() not in exclude_set]
        if not args.paper_ids:
            print("[run_pipeline] ERROR: All paper IDs were excluded.", file=sys.stderr)
            sys.exit(1)
    args.exclude_paper_ids = all_excludes if all_excludes else None

    n_papers = len(args.paper_ids) if args.paper_ids else None

    print(f"\n[run_pipeline] Order: script-first")
    if n_papers is not None:
        print(f"[run_pipeline] Papers to process: {n_papers}")
    else:
        print("[run_pipeline] Papers: (auto-discover in each sub-script)")
    if args.exclude_paper_ids:
        print(f"[run_pipeline] Excluded IDs: {', '.join(args.exclude_paper_ids)}")

    _PAPER_DONE_RE = re.compile(r"^\[(DONE|SKIP|ERROR)\]\s+paper_id")
    _INSTANCE_HEADER_RE = re.compile(r"^===\s+instance_\d+")
    _PROCESSING_RE = re.compile(r"(?:Processing\s+(\d+)\s+paper|(\d+)\s+active\s+paper|across\s+(\d+)\s+paper)")

    total_steps = end - start + 1
    for step_num, idx in enumerate(range(start, end + 1), 1):
        paper_info = f" | {n_papers} papers" if n_papers else ""
        print(f"\n{'='*60}")
        print(f"[run_pipeline] Step {step_num}/{total_steps} (script {idx}): {SCRIPT_NAMES[idx]}{paper_info}")
        print(f"{'='*60}\n")

        cmd = build_cmd_script_first(idx, args)

        done_count = 0
        step_total = n_papers
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            env=env,
        )
        for line in proc.stdout:
            line_stripped = line.rstrip("\n")
            print(line_stripped, flush=True)
            if _INSTANCE_HEADER_RE.match(line_stripped):
                done_count = 0
            m = _PROCESSING_RE.search(line_stripped)
            if m:
                step_total = int(m.group(1) or m.group(2) or m.group(3))
            if _PAPER_DONE_RE.match(line_stripped):
                done_count += 1
                total_str = str(step_total) if step_total else "?"
                print(f"[run_pipeline] >>> Paper progress: {done_count}/{total_str}", flush=True)
        proc.wait()

        if proc.returncode != 0:
            print(
                f"\n[run_pipeline] ERROR: {SCRIPT_NAMES[idx]} exited with code {proc.returncode}. Stopping.",
                file=sys.stderr,
            )
            sys.exit(proc.returncode)

        print(f"\n[run_pipeline] Step {step_num}/{total_steps} completed: {SCRIPT_NAMES[idx]} ({done_count} papers processed)")

    print(f"\n{'='*60}")
    print(f"[run_pipeline] Pipeline finished (scripts {start}-{end}).")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline runner for paper reproduction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--order", type=str, choices=["paper", "script"], default="paper",
        help="Loop order: 'paper' (default) runs all scripts per paper; "
             "'script' runs each script across all papers.",
    )
    parser.add_argument(
        "--run_scripts", type=str, default="0-4",
        help="Which scripts to run, by index range. "
             "E.g. '0-4' (all, default), '0' (first only), '1-2'.",
    )
    parser.add_argument(
        "--paper-id", nargs="+", dest="paper_ids", default=None,
        help="Paper IDs to process.",
    )
    parser.add_argument(
        "--exclude-paper-id", nargs="+", dest="exclude_paper_ids", default=None,
        help="Paper IDs to exclude.",
    )
    parser.add_argument(
        "--num_instances", type=int, default=None,
        help="Number of instances (forwarded to scripts 0, 2).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="In paper-first mode: number of papers to process in parallel. "
             "In script-first mode: forwarded to sub-scripts as --workers.",
    )
    parser.add_argument(
        "--instances", nargs="+", default=None,
        help="Categorical instance names e.g. 'tiny large_1' (forwarded to script 3).",
    )
    parser.add_argument(
        "--time_limit", type=int, default=None,
        help="Time limit in seconds (forwarded to script 3).",
    )
    parser.add_argument(
        "--rerun_null", type=int, choices=[0, 1], default=None,
        help="Whether to rerun instances with null solution_status in script 3. "
             "1 = rerun, 0 = skip.",
    )
    parser.add_argument(
        "--solution-workers", type=int, dest="solution_workers", default=None,
        help="Number of parallel workers for script 3 (default 1 = serial).",
    )
    parser.add_argument(
        "--continue-from", type=str, dest="continue_from", default=None,
        help="Resume from a specific paper_id, skipping all papers before it "
             "(paper-first mode only).",
    )

    args = parser.parse_args()

    # Parse run range
    try:
        start, end = parse_run_scripts(args.run_scripts)
    except (ValueError, argparse.ArgumentTypeError) as e:
        parser.error(str(e))

    if not (0 <= start <= end <= 4):
        parser.error(f"--run_scripts range must be within 0-4, got {start}-{end}.")

    # Validate --continue-from
    if args.continue_from and args.order != "paper":
        parser.error("--continue-from is only supported in paper-first mode (--order paper).")

    # Clear token usage log
    open(USAGE_LOG, "w").close()

    # Dispatch
    script_indices = list(range(start, end + 1))
    if args.order == "paper":
        run_paper_first(args, script_indices)
    else:
        run_script_first(args, start, end)

    print_token_summary()


if __name__ == "__main__":
    main()
