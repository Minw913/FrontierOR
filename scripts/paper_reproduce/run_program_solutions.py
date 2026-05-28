"""
Run efficient_algorithm.py and gurobi_code.py for each paper and instance,
compute optimality gaps, and incrementally write results to ./solving_results_full.csv.

Parallelises across papers within each instance round via --workers
(default 1 = serial).

Paper selection priority (highest to lowest):
1) HARDCODED_PAPER_IDS (if non-empty)
2) --paper-id CLI arguments (if provided)
3) Auto-discovery by scanning data/paper_data/*/*.pdf

Exclusions (HARDCODED_PAPER_EXCLUDE_IDS and --exclude_id) always win.

CLI arguments:
    --paper-id ID [ID ...]      Paper IDs to process (folder names under data/paper_data/).
                                 Omit to auto-discover all papers with PDFs.
    --exclude_id ID [...]  Paper IDs to exclude (highest priority, always wins).
    --instances NAME [NAME ...] Categorical instance names to run (default: tiny large_1).
    --time_limit SECONDS         Time limit passed to each program via --time_limit;
                                 outer process kills after time_limit+30s (default: 3600).
    --workers N                  Number of papers to run in parallel per instance round
                                 (default: 1).
    --rerun_null {0,1}           Whether to rerun instances with null solution_status.
                                 1 = rerun (default), 0 = skip.
    --gurobi-only                Only run gurobi_code.py, skip efficient_algorithm.py.
    --skip-existing-gurobi       Skip instances where gurobi_time[i] already has a value
                                 in CSV.
    --force                      Force rerun even when the (paper, instance) row already
                                 exists in the CSV; overwrites the existing row with
                                 fresh results.
    --ignore-status2             Do not skip remaining instances when solution_status=2
                                 (gap > 5%); by default the paper is stopped early.

Execution logic:
    Outer loop iterates instance names sequentially (tiny -> large_1 -> ...).
    Inner loop runs all active papers in parallel (controlled by --workers).
    Each (paper, instance) result is flushed to solving_results_full.csv immediately.

Usage examples:
    python scripts/paper_reproduce/run_program_solutions.py --workers 6 --time_limit 3600 --instances large_1 --gurobi-only --ignore-status2
    python scripts/paper_reproduce/run_program_solutions.py --workers 4 --time_limit 3600 --instances tiny
    python scripts/paper_reproduce/run_program_solutions.py --workers 2
    python scripts/paper_reproduce/run_program_solutions.py --paper-id mingozzi1999 amaldi2013 --workers 2
"""

import argparse
import csv
import glob
import json
import os

import sys as _sys_for_paths
_sys_for_paths.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
import exec_backends  # noqa: E402
from instance_paths import (  # noqa: E402
    DEFAULT_INSTANCES,
    instance_path as _instance_path,
    gurobi_solution_path as _gurobi_solution_path,
    gurobi_log_path as _gurobi_log_path,
    gurobi_feasi_result_path as _gurobi_feasi_result_path,
    efficient_solution_path as _efficient_solution_path,
    efficient_log_path as _efficient_log_path,
    parse_instances_arg,
)
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAPER_DATA_DIR = os.path.join(BASE_DIR, "frontier-or")
CSV_PATH = os.path.join(BASE_DIR, "solving_results_full.csv")
GUROBI_CSV_PATH = os.path.join(BASE_DIR, "gurobi_solving_results.csv")

CSV_SCHEMAS = {
    "full": {
        "path": CSV_PATH,
        "columns": [
            "paper_id", "instance", "category",
            "resolve_success", "resolve_num",
            "solution_status", "timeout_status", "solve_time_diff", "gap_status",
            "gurobi_feasibility_status", "efficient_feasibility_status",
            "gurobi_time", "gurobi_solution",
            "efficient_time", "efficient_solution",
            "failure_reason", "failure_error",
        ],
    },
    "gurobi": {
        "path": GUROBI_CSV_PATH,
        "columns": [
            "paper_id", "instance",
            "gurobi_feasibility_status",
            "gurobi_solution", "solution_status",
            "gurobi_time", "time_limit",
            "failure_reason", "failure_error",
        ],
    },
}

# Per-instance default results CSV. Used when --csv-path is omitted and a
# single instance is selected: keeps each scale's results in its own file
# so parallel runs don't fight for the same lock and rows stay grouped.
# Match by exact instance name. Adding a new scale? Drop a line here.
_INSTANCE_RESULTS_CSV = {
    "tiny":     os.path.join(BASE_DIR, "gurobi_results_tiny.csv"),
    "large_11": os.path.join(BASE_DIR, "gurobi_results_11.csv"),
    "large_21": os.path.join(BASE_DIR, "gurobi_results_21.csv"),
    "large_31": os.path.join(BASE_DIR, "gurobi_results_31.csv"),
    "large_41": os.path.join(BASE_DIR, "gurobi_results_41.csv"),
    "large_51": os.path.join(BASE_DIR, "gurobi_results_51.csv"),
}

# Source for tag-based paper selection (--paper-tag).
_GUROBI_RESULTS_ALL = os.path.join(BASE_DIR, "gurobi_results_all_new.csv")


def _default_csv_for_instances(instance_names):
    """Return per-instance default CSV when exactly one mapped instance is
    selected; otherwise None (caller falls back to schema default)."""
    if not instance_names or len(instance_names) != 1:
        return None
    return _INSTANCE_RESULTS_CSV.get(instance_names[0])


def _load_paper_ids_by_tag(tags, csv_path=None):
    """Return sorted unique paper_ids whose 'tag' column in
    gurobi_results_all_new.csv exactly matches any of ``tags``. Stacked tags
    like 'C,F' are NOT matched by --paper-tag C; pass --paper-tag "C,F"
    explicitly to include them.
    """
    path = csv_path or _GUROBI_RESULTS_ALL
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"--paper-tag set but {path} not found. "
            f"Run scripts/data_management/merge_and_classify_gurobi_results.py first."
        )
    wanted = {t.strip() for t in tags if t and t.strip()}
    if not wanted:
        return []
    out = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row_tag = (row.get("tag") or "").strip()
            if row_tag in wanted:
                pid = (row.get("paper_id") or "").strip()
                if pid:
                    out.add(pid)
    return sorted(out)


DEFAULT_TIME_LIMIT_SECONDS = 3600  # forwarded to each program as --time_limit
GAP_THRESHOLD = 0.05    # 5 %
# Treat very small gaps as zero (floating-point / solver tolerance)
GAP_ZERO_EPS = 1e-6


# Hardcoded paper_id list. If non-empty, this takes highest priority.
HARDCODED_PAPER_IDS: list[str] = [
#    'araujo2015', 'bertsimas2009', 'bollapragada2001', 'chen2019', 'coelho2017', 'cote2021', 'delorme2020', 'schwerdfeger2016', 'forrest2006', 'hoffman1993', 'kobayashi2021', 'peng2022', 'reinhardt2016',
#    'vaziri2024'
  ]

# Hardcoded exclusion list. These IDs are always removed.
HARDCODED_PAPER_EXCLUDE_IDS: list[str] = [
    # "some_paper_id_to_skip",
]


def discover_paper_ids(paper_data_dir: str | None = None) -> list[str]:
    """Return paper_ids by scanning <paper_data_dir>/<paper>/ for *.pdf files.

    If no PDFs are found (e.g. when the root is ``frontier-or/``
    which may not ship PDFs), fall back to every immediate subdirectory that
    contains a ``gurobi_code.py``.
    """
    root = paper_data_dir or PAPER_DATA_DIR
    ids = {os.path.basename(os.path.dirname(p))
           for p in glob.glob(os.path.join(root, "*", "*.pdf"))}
    if not ids:
        ids = {
            os.path.basename(d)
            for d in glob.glob(os.path.join(root, "*"))
            if os.path.isfile(os.path.join(d, "gurobi_code.py"))
        }
    return sorted(ids)


def _run_log_path_for_solution(solution_path: str) -> str:
    """Derive per-instance run log path from solution path.

    e.g. '.../gurobi_solution/tiny_solution.json' -> '.../gurobi_solution/tiny_run.log'
         '.../efficient_solution/large_solution_1.json' -> '.../efficient_solution/large_run_1.log'
    """
    sol_dir = os.path.dirname(solution_path)
    base = os.path.basename(solution_path)
    stem = base[:-5] if base.endswith(".json") else base  # strip .json
    stem = stem.replace("_solution", "_run", 1)
    return os.path.join(sol_dir, stem + ".log")


def _write_run_log(run_log_path: str, header: str, stdout: str, stderr: str) -> None:
    """Write a per-instance run log: 'solving time = X (s)' line on top, then stdout/stderr."""
    try:
        os.makedirs(os.path.dirname(run_log_path), exist_ok=True)
        with open(run_log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(header + "\n")
            f.write("\n===== STDOUT =====\n")
            f.write(stdout or "")
            if not (stdout or "").endswith("\n"):
                f.write("\n")
            f.write("\n===== STDERR =====\n")
            f.write(stderr or "")
            if not (stderr or "").endswith("\n"):
                f.write("\n")
    except OSError:
        pass  # never fail the run because of log-write issues


def run_program(script_path: str, instance_path: str, solution_path: str,
                time_limit: int | None = None,
                log_path: str | None = None,
                backend: str = "bare",
                backend_cfg: dict | None = None) -> tuple[float | str, float | dict | None]:
    """
    Run a Python script with --instance_path and --solution_path (and optionally --time_limit, --log_path).

    The ``backend`` selects the execution environment:
      - ``bare``    — plain ``python script.py ...`` (no resource limits)
      - ``systemd`` — ``systemd-run`` with pinned cores + CPU/memory cgroups
      - ``docker``  — ``docker run`` with ``--cpuset-cpus`` (pinned core) +
                       ``--memory`` + ``--network=none`` in a sealed image
    ``backend_cfg`` overrides defaults (``cpus``, ``memory``, ``docker_image``,
    ``gurobi_lic``). See ``scripts/utils/exec_backends.py``.

    Also writes a per-instance run log next to the solution file (e.g. tiny_run.log)
    whose first line is ``solving time = <elapsed> (s)``, followed by captured
    stdout/stderr.

    Returns:
        (elapsed_time_or_'time_out', objective_value_or_None)
    """
    cfg = {"cpus": 1, "memory": "32G"}
    if backend_cfg:
        cfg.update(backend_cfg)
    try:
        builder = exec_backends.BUILDERS[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown backend: {backend!r}") from exc
    cmd = builder(
        script_path, instance_path, solution_path,
        time_limit if time_limit is not None else 3600,
        log_path, cfg,
    )
    run_log_path = _run_log_path_for_solution(solution_path)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(script_path),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=time_limit + 30 if time_limit is not None else None,
        )
        elapsed = round(time.time() - start, 2)
    except subprocess.TimeoutExpired as te:
        elapsed_actual = round(time.time() - start, 2)
        to_str = lambda v: v if isinstance(v, str) else (v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else "")
        _write_run_log(
            run_log_path,
            f"solving time = {elapsed_actual} (s) [timed out]",
            to_str(te.stdout), to_str(te.stderr),
        )
        # The outer wall timeout killed the python process, but gurobi_code.py
        # may have written an incumbent to disk before being killed (e.g., via
        # its internal Gurobi TimeLimit). Surface that incumbent if present.
        salvage_obj = None
        if os.path.isfile(solution_path):
            try:
                with open(solution_path, "r") as f:
                    salvage_obj = json.load(f).get("objective_value")
            except (OSError, json.JSONDecodeError):
                salvage_obj = None
        return "time_out", salvage_obj

    _write_run_log(
        run_log_path,
        f"solving time = {elapsed} (s)",
        result.stdout, result.stderr,
    )

    if result.returncode != 0:
        return "runtime_error", {
            "exit_code": result.returncode,
            "stderr": result.stderr.strip(),
            "stdout": result.stdout.strip(),
        }

    if os.path.isfile(solution_path):
        try:
            with open(solution_path, "r") as f:
                sol = json.load(f)
            obj = sol.get("objective_value")
            return elapsed, obj
        except (json.JSONDecodeError, KeyError):
            return elapsed, None
    return elapsed, None


def compute_gap(obj_gurobi, obj_efficient) -> float | None:
    """Compute optimality gap = |obj_gurobi - obj_efficient| / max(|obj_gurobi|, 1e-6)."""
    if not isinstance(obj_gurobi, (int, float)) or not isinstance(obj_efficient, (int, float)):
        return None
    return abs(obj_gurobi - obj_efficient) / max(abs(obj_gurobi), 1e-6)


def gap_status_code(gap: float | None) -> int | None:
    """0 if gap<=GAP_ZERO_EPS, 1 if gap<=GAP_THRESHOLD, 2 if gap>GAP_THRESHOLD, None if gap is None."""
    if gap is None:
        return None
    if gap <= GAP_ZERO_EPS:
        return 0
    elif gap <= GAP_THRESHOLD:
        return 1
    else:
        return 2


# --- CSV helpers (tidy long-format: 1 row per (paper_id, instance)) ---------
def build_csv_columns(schema: str = "full") -> list[str]:
    """Return canonical column order for the selected CSV schema."""
    try:
        return list(CSV_SCHEMAS[schema]["columns"])
    except KeyError as exc:
        raise ValueError(f"Unknown csv schema: {schema!r}") from exc


def load_existing_csv(csv_path: str) -> tuple[dict[tuple[str, str], dict], list[str]]:
    rows: dict[tuple[str, str], dict] = {}
    existing_columns: list[str] = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_columns = list(reader.fieldnames) if reader.fieldnames else []
            for row in reader:
                pid = row.get("paper_id", "")
                inst = row.get("instance", "")
                if pid and inst:
                    rows[(pid, inst)] = row
    return rows, existing_columns


def merge_columns(existing_columns: list[str], new_columns: list[str]) -> list[str]:
    seen = set(existing_columns)
    merged = list(existing_columns)
    for c in new_columns:
        if c not in seen:
            merged.append(c)
            seen.add(c)
    return merged


def write_csv(csv_path: str, columns: list[str], rows_dict: dict[tuple[str, str], dict]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows_dict.keys()):
            writer.writerow({c: rows_dict[key].get(c, "") for c in columns})


def flush_row_to_csv(csv_path: str, paper_id: str, instance: str,
                     row_data: dict, new_columns: list[str]):
    """Read-modify-write: upsert (paper_id, instance) row."""
    disk_rows, existing_columns = load_existing_csv(csv_path)
    columns = merge_columns(existing_columns, new_columns)
    disk_rows[(paper_id, instance)] = row_data
    write_csv(csv_path, columns, disk_rows)


def init_row(paper_id: str, instance: str, columns: list[str]) -> dict:
    row = {c: "" for c in columns}
    row["paper_id"] = paper_id
    row["instance"] = instance
    return row


def infer_gurobi_optimality(solution_path: str, *, time_ran=None, time_limit=None) -> str:
    """Read solution JSON and return 'optimal' / 'incumbent' / ''.

    Handles the inconsistent status conventions across paper solution files
    by checking (in order):
      1. Any key containing "status" or "termination" with value == "OPTIMAL"
         (case-insensitive) or int code 2 (GRB.OPTIMAL).
      2. Any zero-gap field: ``mip_gap`` / ``MIPGap`` / ``optimality_gap`` /
         ``gap`` with |value| < 1e-6.
      3. If ``objective_value`` exists AND runtime is well below the time
         limit (time_ran < 0.95 * time_limit), infer 'optimal' by exclusion
         (no TimeLimit/NodeLimit/SolutionLimit hit in default gurobi_code.py).
         Threshold matches ``reclassify_solution_status.TIMEOUT_FRACTION``.
      4. ``objective_value`` exists but runtime near / at limit -> 'incumbent'.
      5. No solution info -> ''.

    Broad field coverage is needed because papers save ``solver_status`` /
    ``status_name`` / ``mip_gap`` etc. instead of canonical ``status`` keys;
    a narrower lookup would mis-classify optimal solutions as 'incumbent'.
    """
    if not os.path.isfile(solution_path):
        return ""
    try:
        with open(solution_path, "r", encoding="utf-8") as f:
            sol = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""

    # (1) Any *status* / *termination* field saying OPTIMAL or int 2.
    for k in sol:
        kl = k.lower()
        if "status" not in kl and "termination" not in kl:
            continue
        v = sol[k]
        if isinstance(v, str) and v.strip().upper() == "OPTIMAL":
            return "optimal"
        if isinstance(v, int) and v == 2:  # GRB.OPTIMAL
            return "optimal"

    # (2) Zero-gap evidence.
    for k in ("mip_gap", "MIPGap", "optimality_gap", "gap"):
        v = sol.get(k)
        if v is None:
            continue
        try:
            if abs(float(v)) < 1e-6:
                return "optimal"
        except (TypeError, ValueError):
            pass

    # (3, 4, 5) Objective-based inference with runtime hint.
    if sol.get("objective_value") is not None:
        if time_ran is not None and time_limit:
            try:
                if float(time_ran) < 0.95 * float(time_limit):
                    return "optimal"
            except (TypeError, ValueError):
                pass
        return "incumbent"
    return ""


def run_gurobi_feasibility_check(paper_dir: str, paper_id: str, inst_name: str,
                                 instance_path: str, solution_path: str,
                                 timeout_s: int = 1200) -> str:
    """Run the paper's local feasibility_check.py on the just-produced gurobi
    solution and return the JSON's ``feasible`` field as 'True' / 'False' / ''
    (empty if it didn't run, errored, timed out, or had no boolean field).
    Synchronous so callers can write the result straight into the CSV row.
    """
    fc_script = os.path.join(paper_dir, "feasibility_check.py")
    if not os.path.isfile(fc_script):
        return ""
    if not os.path.isfile(solution_path):
        return ""
    if not os.path.isfile(instance_path):
        return ""
    result_path = _gurobi_feasi_result_path(paper_dir, inst_name)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    # Always re-run so the result reflects the new solution.
    try:
        if os.path.isfile(result_path):
            os.remove(result_path)
    except OSError:
        pass
    cmd = [
        sys.executable, fc_script,
        "--instance_path", instance_path,
        "--solution_path", solution_path,
        "--result_path", result_path,
    ]
    try:
        subprocess.run(
            cmd, cwd=paper_dir,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{paper_id}] {inst_name} – feasibility_check.py TIMEOUT")
        return ""
    except Exception as e:
        print(f"  [{paper_id}] {inst_name} – feasibility_check.py ERROR ({e})")
        return ""
    if not os.path.isfile(result_path):
        return ""
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ""
    feasible = data.get("feasible") if isinstance(data, dict) else None
    if isinstance(feasible, bool):
        return "True" if feasible else "False"
    return ""


def summarize_runtime_error(label: str, payload) -> str:
    if not isinstance(payload, dict):
        return f"{label}: runtime_error"
    parts = [f"{label}: exit={payload.get('exit_code')}"]
    stderr = (payload.get("stderr") or "").strip()
    stdout = (payload.get("stdout") or "").strip()
    detail = stderr or stdout
    if detail:
        # Keep the CSV cell single-line: collapse any embedded \r/\n to " | ".
        detail = detail.replace("\r\n", "\n").replace("\r", "\n")
        detail = " | ".join(line.strip() for line in detail.split("\n") if line.strip())
        parts.append(detail[:300])
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Run efficient_algorithm.py and gurobi_code.py on instances "
                    "(parallel across papers), compute gaps, and write solving_results_full.csv."
    )
    parser.add_argument(
        "--paper-id", nargs="+", dest="paper_ids", default=None,
        help="Paper IDs (folder names under data/paper_data/). "
             "Omit to auto-discover all papers with PDFs.",
    )
    parser.add_argument(
        "--paper-tag", "--paper_tag", nargs="+", dest="paper_tags", default=None,
        help="Select papers from gurobi_results_all_new.csv whose 'tag' column "
             "exactly matches any of these tags. Stacked tags like 'C,F' are "
             "NOT matched by --paper-tag C; pass them explicitly with quotes, "
             "e.g. --paper-tag C E \"C,F\". Common with --force to re-run a "
             "tag class. Ignored when --paper-id is given.",
    )
    parser.add_argument(
        "--exclude_id", nargs="+", dest="exclude_paper_ids", default=[],
        help="Paper IDs to exclude.",
    )
    parser.add_argument(
        "--instances", nargs="+", default=None,
        help="Categorical instance names to run (e.g., --instances tiny large_1). "
             f"Default: {' '.join(DEFAULT_INSTANCES)}.",
    )
    parser.add_argument(
        "--time_limit", type=int, default=DEFAULT_TIME_LIMIT_SECONDS,
        help="Forwarded to each program as --time_limit (seconds). "
             f"Default: {DEFAULT_TIME_LIMIT_SECONDS}.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of (paper, instance) tasks to run in parallel across the "
             "full cross product of --paper-id × --instances (default: 1).",
    )
    parser.add_argument(
        "--rerun_null", type=int, choices=[0, 1], default=1,
        help="Whether to rerun instances whose solution_status is null. "
             "1 = rerun (default), 0 = skip.",
    )
    parser.add_argument(
        "--gurobi-only", action="store_true", dest="gurobi_only",
        help="Only run gurobi_code.py, skip efficient_algorithm.py.",
    )
    parser.add_argument(
        "--skip-existing-gurobi", action="store_true", dest="skip_existing_gurobi",
        help="Skip instances where gurobi_time_i already has a value in CSV.",
    )
    parser.add_argument(
        "--force", action="store_true", dest="force",
        help="Force rerun even when a (paper, instance) row already exists in the CSV; "
             "the existing row is overwritten with fresh results.",
    )
    parser.add_argument(
        "--ignore-status2", action="store_true", dest="ignore_status2",
        help="Do not skip remaining instances when solution_status=2 (gap > threshold).",
    )
    parser.add_argument(
        "--backend", choices=sorted(exec_backends.BUILDERS.keys()), default="bare",
        help="Execution environment for each paper subprocess: "
             "'bare' (default, no resource limits); "
             "'systemd' (systemd-run scope: pinned core via AllowedCPUs + MemoryMax); "
             "'docker' (docker run: --cpuset-cpus + --memory + --network=none, "
             "requires the '%s' image to be built). "
             "With --workers > 1, each parallel case gets its own pinned core "
             "(round-robin across host CPUs)." % exec_backends.DEFAULT_DOCKER_IMAGE,
    )
    parser.add_argument(
        "--backend-cpus", type=int, default=1, dest="backend_cpus",
        help="Number of cores per case (default: 1). Applied as cpuset size.",
    )
    parser.add_argument(
        "--backend-memory", default="32G", dest="backend_memory",
        help="Memory cap per case (default: 32G). Forwarded to the backend's "
             "memory limit (systemd MemoryMax / docker --memory).",
    )
    parser.add_argument(
        "--docker-image", default=exec_backends.DEFAULT_DOCKER_IMAGE, dest="docker_image",
        help=f"Docker image to use with --backend docker (default: "
             f"{exec_backends.DEFAULT_DOCKER_IMAGE}).",
    )
    parser.add_argument(
        "--paper-dir", dest="paper_dir", default=None,
        help=f"Root directory containing one sub-folder per paper (with "
             f"gurobi_code.py, instance/, gurobi_solution/, ...). "
             f"Default: {PAPER_DATA_DIR}.",
    )
    parser.add_argument(
        "--schema", choices=sorted(CSV_SCHEMAS.keys()), default="full",
        help="CSV schema / output file: 'full' -> solving_results_full.csv (default, "
             "full legacy schema with efficient + gurobi + feasibility columns); "
             "'gurobi' -> gurobi_solving_results.csv (simplified: paper_id, instance, "
             "gurobi_feasibility_status, gurobi_time, gurobi_solution, failure_reason, "
             "time_limit). The 'gurobi' schema implies --gurobi-only.",
    )
    parser.add_argument(
        "--csv-path", dest="csv_path", default=None,
        help="Override the CSV output file (single file for ALL --instances). "
             "When omitted: each instance writes to its per-instance default "
             "(large_51 -> gurobi_results_51.csv, large_11 -> gurobi_results_11.csv, "
             "etc.); unmapped names fall back to the --schema default. "
             "See _INSTANCE_RESULTS_CSV.",
    )
    args = parser.parse_args()
    if args.schema == "gurobi":
        args.gurobi_only = True

    paper_data_dir = os.path.abspath(args.paper_dir) if args.paper_dir else PAPER_DATA_DIR
    if not os.path.isdir(paper_data_dir):
        print(f"ERROR: --paper-dir does not exist: {paper_data_dir}", file=sys.stderr)
        sys.exit(1)

    if HARDCODED_PAPER_IDS:
        paper_ids = list(HARDCODED_PAPER_IDS)
    elif args.paper_ids:
        paper_ids = args.paper_ids
    elif args.paper_tags:
        paper_ids = _load_paper_ids_by_tag(args.paper_tags)
        if not paper_ids:
            print(
                f"ERROR: --paper-tag {args.paper_tags} matched no rows in "
                f"{_GUROBI_RESULTS_ALL}.", file=sys.stderr
            )
            sys.exit(1)
        print(f"Loaded {len(paper_ids)} paper(s) with tag in {args.paper_tags} "
              f"from {_GUROBI_RESULTS_ALL}")
    else:
        paper_ids = discover_paper_ids(paper_data_dir)

    if HARDCODED_PAPER_EXCLUDE_IDS:
        hard_exclude = {e.lower() for e in HARDCODED_PAPER_EXCLUDE_IDS}
        paper_ids = [p for p in paper_ids if p.lower() not in hard_exclude]
    if args.exclude_paper_ids:
        exclude_set = {e.lower() for e in args.exclude_paper_ids}
        paper_ids = [p for p in paper_ids if p.lower() not in exclude_set]

    if not paper_ids:
        print("ERROR: No paper_ids to process.", file=sys.stderr)
        sys.exit(1)

    try:
        instance_names = parse_instances_arg(args.instances or DEFAULT_INSTANCES)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    time_limit = args.time_limit
    workers = args.workers
    new_columns = build_csv_columns(args.schema)
    # Resolve a CSV path per instance:
    #   1. --csv-path explicit  -> single file shared by all instances
    #   2. otherwise            -> per-instance default from _INSTANCE_RESULTS_CSV
    #                              (each scale lands in its own gurobi_results_*.csv;
    #                              unmapped names fall back to schema default)
    if args.csv_path:
        explicit = os.path.abspath(args.csv_path)
        csv_paths = {n: explicit for n in instance_names}
    else:
        csv_paths = {}
        for n in instance_names:
            d = _INSTANCE_RESULTS_CSV.get(n)
            csv_paths[n] = os.path.abspath(d) if d else CSV_SCHEMAS[args.schema]["path"]

    backend = args.backend
    backend_cfg = {
        "cpus": args.backend_cpus,
        "memory": args.backend_memory,
        "docker_image": args.docker_image,
    }

    # Writes use flush_row_to_csv() which re-reads from disk each time, so
    # concurrent changes by other processes are preserved.
    rows_dict: dict[tuple[str, str], dict] = {}
    existing_columns: list[str] = []
    for path in sorted(set(csv_paths.values())):
        rd, ec = load_existing_csv(path)
        rows_dict.update(rd)
        for c in ec:
            if c not in existing_columns:
                existing_columns.append(c)
    columns = merge_columns(existing_columns, new_columns)

    print(f"Papers to process: {paper_ids}")
    print(f"Instances to run: {instance_names}")
    print(f"Time limit passed to programs: {time_limit}s")
    print(f"Parallel workers: {workers}")
    print(f"Paper root dir: {paper_data_dir}")
    print(f"Backend: {backend} (cpus={backend_cfg['cpus']} pinned, memory={backend_cfg['memory']})"
          + (f"  image={backend_cfg['docker_image']}" if backend == "docker" else ""))
    print(f"CSV schema: {args.schema}")
    if len(set(csv_paths.values())) == 1:
        print(f"Results CSV: {next(iter(csv_paths.values()))}")
    else:
        print("Results CSV (per instance):")
        for n in instance_names:
            print(f"  {n}: {csv_paths[n]}")
    print("=" * 70)

    # Pre-check which papers are runnable. New (paper, instance) rows are
    # created lazily inside the worker.
    skip_papers: set[str] = set()
    skip_lock = threading.Lock()
    csv_lock = threading.Lock()
    paper_dirs: dict[str, str] = {}
    eff_scripts: dict[str, str] = {}
    gurobi_scripts: dict[str, str] = {}

    for paper_id in paper_ids:
        paper_dir = os.path.join(paper_data_dir, paper_id)
        paper_dirs[paper_id] = paper_dir
        if not os.path.isdir(paper_dir):
            print(f"[WARN] Directory not found for paper_id={paper_id}, skipping.")
            skip_papers.add(paper_id)
            continue

        eff_script = os.path.join(paper_dir, "efficient_algorithm.py")
        gurobi_script = os.path.join(paper_dir, "gurobi_code.py")
        eff_scripts[paper_id] = eff_script
        gurobi_scripts[paper_id] = gurobi_script

        if not args.gurobi_only and not os.path.isfile(eff_script):
            print(f"[WARN] efficient_algorithm.py not found in {paper_dir}, skipping.")
            skip_papers.add(paper_id)
            continue
        if not os.path.isfile(gurobi_script):
            print(f"[WARN] gurobi_code.py not found in {paper_dir}, skipping.")
            skip_papers.add(paper_id)
            continue

    completed_counter = {"n": 0, "total": 0}
    counter_lock = threading.Lock()

    def _progress():
        """Increment and return progress string like '(3/139)'."""
        with counter_lock:
            completed_counter["n"] += 1
            return f"({completed_counter['n']}/{completed_counter['total']})"

    def process_paper_instance(paper_id: str, inst_name: str, total_papers_count: int = 0):
        """Run efficient + gurobi for (paper_id, inst_name). Returns True if paper should be skipped hereafter."""
        paper_dir = paper_dirs[paper_id]
        row = rows_dict.get((paper_id, inst_name)) or init_row(paper_id, inst_name, columns)
        instance_path = _instance_path(paper_dir, inst_name)

        if not os.path.isfile(instance_path):
            print(f"  [WARN] {paper_id}: {inst_name} not found at {instance_path}, skipping remaining instances.")
            print(f"[DONE]  paper_id '{paper_id}' | {inst_name}  {_progress()}")
            return True  # skip

        # Skip if CSV has a record for this (paper, instance) unless it was OOM-killed (exit_code=-9)
        # or --force was passed (which overwrites any existing row).
        if (paper_id, inst_name) in rows_dict:
            failure_error = row.get("failure_error") or ""
            is_oom = '"exit_code": -9' in failure_error
            if is_oom:
                print(f"[RERUN] paper_id '{paper_id}' | {inst_name}: previous run OOM-killed, rerunning")
            elif args.force:
                existing_status = (row.get("solution_status") or "").strip()
                print(f"[FORCE] paper_id '{paper_id}' | {inst_name}: --force set, overwriting existing row (status={existing_status or 'N/A'})")
            else:
                existing_status = (row.get("solution_status") or "").strip()
                print(f"[SKIP]  paper_id '{paper_id}' | {inst_name}: already has result (status={existing_status or 'N/A'})  {_progress()}")
                return False  # continue to next instance

        # Skip instances where gurobi_time already exists
        if args.skip_existing_gurobi and (row.get("gurobi_time") or "").strip():
            print(f"[SKIP]  paper_id '{paper_id}' | {inst_name}: gurobi_time exists  {_progress()}")
            return False

        # About to run gurobi for this (paper, instance) — invalidate any stale
        # gurobi_feasibility_status carried over from a prior run_feasibility_check.py
        # pass; it is no longer authoritative for the new solution.
        row["gurobi_feasibility_status"] = ""

        eff_sol_path = _efficient_solution_path(paper_dir, inst_name)
        gurobi_sol_path = _gurobi_solution_path(paper_dir, inst_name)
        eff_log_path = _efficient_log_path(paper_dir, inst_name)
        gurobi_log_path = _gurobi_log_path(paper_dir, inst_name)
        for p in (eff_sol_path, gurobi_sol_path, eff_log_path, gurobi_log_path):
            os.makedirs(os.path.dirname(p), exist_ok=True)

        if args.gurobi_only:
            eff_time, eff_obj = None, None
        else:
            print(f"  [{paper_id}] {inst_name} – running efficient_algorithm.py ...")
            eff_time, eff_obj = run_program(
                eff_scripts[paper_id], instance_path, eff_sol_path,
                time_limit=time_limit, log_path=eff_log_path,
                backend=backend, backend_cfg=backend_cfg,
            )
            if eff_time == "time_out":
                print(f"  [{paper_id}] {inst_name} – efficient_algorithm.py TIME_OUT")
            elif eff_time == "runtime_error":
                print(f"  [{paper_id}] {inst_name} – efficient_algorithm.py FAILED (exit={eff_obj.get('exit_code')})")
            else:
                print(f"  [{paper_id}] {inst_name} – efficient_algorithm.py done ({eff_time}s, obj={eff_obj})")

            eff_sol_val = (
                eff_obj if eff_obj is not None else
                "time_out" if eff_time == "time_out" else
                eff_obj if eff_time == "runtime_error" else
                "N/A"
            )
            row["efficient_time"] = eff_time if eff_time is not None else ""
            row["efficient_solution"] = (
                json.dumps(eff_sol_val) if isinstance(eff_sol_val, dict)
                else (eff_sol_val if eff_sol_val is not None else "")
            )

        # --- Run gurobi_code.py ---
        print(f"  [{paper_id}] {inst_name} – running gurobi_code.py ...")
        gurobi_time, gurobi_obj = run_program(
            gurobi_scripts[paper_id], instance_path, gurobi_sol_path,
            time_limit=time_limit, log_path=gurobi_log_path,
            backend=backend, backend_cfg=backend_cfg,
        )
        if gurobi_time == "time_out":
            print(f"  [{paper_id}] {inst_name} – gurobi_code.py TIME_OUT")
        elif gurobi_time == "runtime_error":
            print(f"  [{paper_id}] {inst_name} – gurobi_code.py FAILED (exit={gurobi_obj.get('exit_code')})")
        else:
            print(f"  [{paper_id}] {inst_name} – gurobi_code.py done ({gurobi_time}s, obj={gurobi_obj})")

        gurobi_sol_val = (
            gurobi_obj if gurobi_obj is not None else
            "time_out" if gurobi_time == "time_out" else
            gurobi_obj if gurobi_time == "runtime_error" else
            "N/A"
        )
        row["gurobi_time"] = gurobi_time if gurobi_time is not None else ""
        row["gurobi_solution"] = (
            json.dumps(gurobi_sol_val) if isinstance(gurobi_sol_val, dict)
            else (gurobi_sol_val if gurobi_sol_val is not None else "")
        )

        # --- Compute gap ---
        gap = compute_gap(gurobi_obj, eff_obj)
        if eff_time == "runtime_error" or gurobi_time == "runtime_error":
            gs = None
            failure_reason = "runtime_error"
            failure_parts = []
            if eff_time == "runtime_error":
                failure_parts.append(summarize_runtime_error("efficient", eff_obj))
            if gurobi_time == "runtime_error":
                failure_parts.append(summarize_runtime_error("gurobi", gurobi_obj))
            failure_error = " || ".join(failure_parts)
        else:
            gs = gap_status_code(gap)
            failure_reason = None
            failure_error = None
        gap_rounded = round(gap, 6) if gap is not None else None

        row["gap_status"] = gap_rounded if gap_rounded is not None else ""
        row["solution_status"] = gs if gs is not None else ""
        row["failure_reason"] = failure_reason if failure_reason is not None else ""
        row["failure_error"] = failure_error if failure_error is not None else ""

        if isinstance(gurobi_time, (int, float)) and isinstance(eff_time, (int, float)):
            row["solve_time_diff"] = round(gurobi_time - eff_time, 2)

        if gap is not None:
            print(f"  [{paper_id}] {inst_name} – Gap = {gap_rounded} ({gs})")

        if args.schema == "gurobi":
            row["time_limit"] = time_limit
            row["solution_status"] = infer_gurobi_optimality(
                gurobi_sol_path,
                time_ran=gurobi_time if isinstance(gurobi_time, (int, float)) else None,
                time_limit=time_limit,
            )

        # --- Run feasibility_check.py on the new gurobi solution ---
        if gurobi_time != "runtime_error":
            # Allow the checker up to half the gurobi budget, capped at 1800s.
            # Some papers' feasibility_check.py loops over edges × OD pairs and
            # runs for many minutes on large instances.
            feas_timeout = max(120, min(int(time_limit) // 2, 1800)) if isinstance(time_limit, int) else 1200
            feas = run_gurobi_feasibility_check(
                paper_dir, paper_id, inst_name,
                instance_path, gurobi_sol_path,
                timeout_s=feas_timeout,
            )
            row["gurobi_feasibility_status"] = feas
            if feas:
                print(f"  [{paper_id}] {inst_name} – feasibility = {feas}")

        with csv_lock:
            flush_row_to_csv(csv_paths[inst_name], paper_id, inst_name, row, new_columns)

        if gap is not None and gap > GAP_THRESHOLD:
            print(f"  [{paper_id}] {inst_name} – Gap {gap_rounded} > {GAP_THRESHOLD}. Skipping remaining instances.")
            print(f"[DONE]  paper_id '{paper_id}' | {inst_name}  {_progress()}")
            return True  # skip remaining instances

        print(f"[DONE]  paper_id '{paper_id}' | {inst_name}  {_progress()}")
        return False  # continue

    # Flat (paper, instance) task pool. Submit order is instance-major so
    # smaller instances start first (gives gap-skip a chance to land before
    # large_* tasks are picked up). gap-skip propagation is best-effort: a
    # paper added to skip_papers only filters tasks not yet submitted.
    tasks = [
        (paper_id, inst_name)
        for inst_name in instance_names
        for paper_id in paper_ids
        if paper_id not in skip_papers
    ]
    completed_counter["n"] = 0
    completed_counter["total"] = len(tasks)
    print(
        f"\n=== {len(tasks)} (paper, instance) task(s): "
        f"{len(paper_ids)} paper(s) × {len(instance_names)} instance(s) ==="
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for paper_id, inst_name in tasks:
            with skip_lock:
                if paper_id in skip_papers:
                    completed_counter["total"] -= 1
                    continue
            futures[pool.submit(process_paper_instance, paper_id, inst_name, len(tasks))] = (paper_id, inst_name)
        for future in as_completed(futures):
            paper_id, inst_name = futures[future]
            try:
                should_skip = future.result()
                if should_skip:
                    with skip_lock:
                        skip_papers.add(paper_id)
            except Exception as e:
                print(f"  [ERROR] {paper_id} {inst_name}: {e}", file=sys.stderr)
                with skip_lock:
                    skip_papers.add(paper_id)

    print("\n" + "=" * 70)
    unique_paths = sorted(set(csv_paths.values()))
    if len(unique_paths) == 1:
        print(f"Done. Results written to {unique_paths[0]}")
    else:
        print("Done. Results written to:")
        for p in unique_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
