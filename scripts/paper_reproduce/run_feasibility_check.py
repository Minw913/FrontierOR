"""
Run feasibility checks for each paper via Claude Code Agent.

Calls Claude with the feasibility check prompt to generate feasibility_check.py
and run it on all instances for each paper.

Paper selection priority (highest to lowest):
1) HARDCODED_PAPER_IDS (if non-empty)
2) --paper-id CLI arguments (if provided)
3) Auto-discovery by scanning data/paper_data/*/*.pdf

Exclusions (HARDCODED_PAPER_EXCLUDE_IDS and --exclude-paper-id) always win.

Usage examples:
    python scripts/paper_reproduce/run_feasibility_check.py
    python scripts/paper_reproduce/run_feasibility_check.py --paper-id bard2002 mingozzi1999 amaldi2013
    python scripts/paper_reproduce/run_feasibility_check.py --exclude-paper-id roberti2021
    python scripts/paper_reproduce/run_feasibility_check.py --workers 4
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.utils.claude_utils import run_claude_tracked
PAPER_DATA_DIR = os.path.join(BASE_DIR, "data", "paper_data")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "paper_reproduce", "prompt_feasibility_check.txt")
CSV_PATH = os.path.join(BASE_DIR, "solving_results_full.csv")

# Per-instance default results CSV. Used when --csv-path is omitted and a
# single instance is selected via --instances: keeps each scale's results in
# its own file (mirrors run_program_solutions.py). Adding a new scale? Drop
# a line here.
_INSTANCE_RESULTS_CSV = {
    "tiny":     os.path.join(BASE_DIR, "gurobi_results_tiny.csv"),
    "large_11": os.path.join(BASE_DIR, "gurobi_results_11.csv"),
    "large_21": os.path.join(BASE_DIR, "gurobi_results_21.csv"),
    "large_31": os.path.join(BASE_DIR, "gurobi_results_31.csv"),
    "large_41": os.path.join(BASE_DIR, "gurobi_results_41.csv"),
    "large_51": os.path.join(BASE_DIR, "gurobi_results_51.csv"),
}


def _default_csv_for_instances(instance_names):
    """Return per-instance default CSV when exactly one mapped instance is
    selected; otherwise None (caller falls back to module default)."""
    if not instance_names or len(instance_names) != 1:
        return None
    return _INSTANCE_RESULTS_CSV.get(instance_names[0])

# Hardcoded paper_id list. If non-empty, this takes highest priority.
HARDCODED_PAPER_IDS: list[str] = [
]

# Hardcoded exclusion list. These IDs are always removed.
HARDCODED_PAPER_EXCLUDE_IDS: list[str] = [
    # "some_paper_id_to_skip",
]


def discover_paper_ids() -> list[str]:
    """Auto-discover paper_ids by scanning data/paper_data/*/*.pdf."""
    ids = set()
    for pdf in glob.glob(os.path.join(PAPER_DATA_DIR, "*", "*.pdf")):
        ids.add(os.path.basename(os.path.dirname(pdf)))
    return sorted(ids)


# --- CSV helpers (tidy long-format) -----------------------------------------
def load_existing_csv(csv_path: str) -> tuple[dict[tuple[str, str], dict], list[str]]:
    """Load tidy CSV into a dict keyed by (paper_id, instance)."""
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


# --- Discover solution instances on disk ------------------------------------
# The dataset has two layouts:
#   (a) Legacy (instances 2..9): solution files at paper_dir root, named
#       e.g. gurobi_solution_3.json, efficient_solution_3.json.
#   (b) New (tiny + large_N): solution files in category subdirs, e.g.
#       gurobi_solution/tiny_solution.json, efficient_solution/large_solution_1.json.
#
# Discovery functions yield "instance keys" — either an int (legacy idx) or a
# str (categorical name like "tiny" or "large_1"). Downstream code branches
# on type to resolve paths.
import sys as _sys_for_paths
_sys_for_paths.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from instance_paths import (  # noqa: E402
    instance_path as _new_instance_path,
    gurobi_solution_path as _new_gurobi_solution_path,
    gurobi_feasi_result_path as _new_gurobi_feasi_result_path,
    efficient_solution_path as _new_efficient_solution_path,
    efficient_feasi_result_path as _new_efficient_feasi_result_path,
)


def _resolve_paths(paper_dir: str, prefix: str, key):
    """Return (instance_path, solution_path, feasi_path) for either layout."""
    if isinstance(key, int):
        # legacy layout
        return (
            os.path.join(paper_dir, f"instance_{key}.json"),
            os.path.join(paper_dir, f"{prefix}_solution_{key}.json"),
            os.path.join(paper_dir, f"{prefix}_feasi_result_{key}.json"),
        )
    # new layout (categorical name)
    if prefix == "gurobi":
        sol = _new_gurobi_solution_path(paper_dir, key)
        feasi = _new_gurobi_feasi_result_path(paper_dir, key)
    else:
        sol = _new_efficient_solution_path(paper_dir, key)
        feasi = _new_efficient_feasi_result_path(paper_dir, key)
    return _new_instance_path(paper_dir, key), sol, feasi


def discover_solution_indices(paper_dir: str) -> list[int]:
    """Scan paper_dir for legacy gurobi_solution_*.json / efficient_solution_*.json
    at root (instances 2..9). Returns sorted list of integer indices."""
    indices: set[int] = set()
    for pattern in ("gurobi_solution_*.json", "efficient_solution_*.json"):
        for path in glob.glob(os.path.join(paper_dir, pattern)):
            m = re.search(r"_(\d+)\.json$", os.path.basename(path))
            if m:
                indices.add(int(m.group(1)))
    return sorted(indices)


def _schema_prefixes(schema: str) -> tuple[str, ...]:
    if schema == "gurobi":
        return ("gurobi",)
    if schema == "efficient":
        return ("efficient",)
    return ("gurobi", "efficient")


def find_missing_feasi(paper_dir: str, instances: list[str] | None = None,
                       schema: str = "both") -> list[tuple[str, object]]:
    """Return list of (prefix, key) pairs that have a solution file but no
    corresponding feasi_result file yet. `key` is int (legacy 2..9) or
    str (new layout: 'tiny', 'large_N').

    If ``instances`` is provided, restrict to those keys only (string-compared).
    ``schema`` restricts which solution prefixes are considered.
    """
    missing: list[tuple[str, object]] = []
    inst_filter = set(instances) if instances else None
    prefixes = _schema_prefixes(schema)
    # Legacy layout: solutions at paper_dir root
    for prefix in prefixes:
        for path in glob.glob(os.path.join(paper_dir, f"{prefix}_solution_*.json")):
            m = re.search(r"_(\d+)\.json$", os.path.basename(path))
            if not m:
                continue
            idx = int(m.group(1))
            if inst_filter is not None and str(idx) not in inst_filter:
                continue
            feasi_path = os.path.join(paper_dir, f"{prefix}_feasi_result_{idx}.json")
            if not os.path.isfile(feasi_path):
                missing.append((prefix, idx))
    # New layout: solutions under {prefix}_solution/{tiny,large_N}_solution.json
    for prefix in prefixes:
        sub = os.path.join(paper_dir, f"{prefix}_solution")
        if not os.path.isdir(sub):
            continue
        for path in glob.glob(os.path.join(sub, "*.json")):
            base = os.path.basename(path)  # e.g. tiny_solution.json or large_solution_1.json
            if base == "tiny_solution.json":
                key = "tiny"
            else:
                m = re.match(r"large_solution_(\d+)\.json$", base)
                if not m:
                    continue
                key = f"large_{m.group(1)}"
            if inst_filter is not None and key not in inst_filter:
                continue
            _, _, feasi_path = _resolve_paths(paper_dir, prefix, key)
            if not os.path.isfile(feasi_path):
                missing.append((prefix, key))
    return sorted(missing, key=lambda x: (str(x[1]), x[0]))


def run_existing_feasibility_check(paper_dir: str, paper_id: str,
                                   instances: list[str] | None = None,
                                   schema: str = "both"):
    """Run the existing feasibility_check.py on solution files that lack feasi_result.

    When ``instances`` is given, only solutions whose key matches the filter are run.
    ``schema`` restricts which solution prefixes are considered.
    """
    fc_script = os.path.join(paper_dir, "feasibility_check.py")
    missing = find_missing_feasi(paper_dir, instances=instances, schema=schema)
    if not missing:
        scope = f" ({','.join(instances)})" if instances else ""
        print(f"[SKIP]  paper_id '{paper_id}': all solution files already have feasi_results{scope}.")
        return

    print(f"[RUN]   paper_id '{paper_id}': running feasibility_check.py on {len(missing)} missing result(s).")
    for prefix, key in missing:
        instance_path, solution_path, result_path = _resolve_paths(paper_dir, prefix, key)
        label = f"{prefix}_feasi_result[{key}]"

        if not os.path.isfile(instance_path):
            print(f"  [{paper_id}] WARN: {os.path.basename(instance_path)} not found, skipping {label}.")
            continue
        # Make sure the output dir exists for the new layout
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

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
                timeout=900,
            )
            status = "OK" if os.path.isfile(result_path) else "NO OUTPUT"
            print(f"  [{paper_id}] {label}: {status}")
        except subprocess.TimeoutExpired:
            print(f"  [{paper_id}] {label}: TIMEOUT")
        except Exception as e:
            print(f"  [{paper_id}] {label}: ERROR ({e})")


def collect_feasibility_results(paper_ids: list[str],
                                instances: list[str] | None = None,
                                schema: str = "both",
                                force: bool = False):
    """Read feasi_result JSON files and write results to CSV_PATH.

    When ``instances`` is given, only those keys are considered — no rows are
    created or updated for instances outside the filter (avoids polluting an
    instance-scoped CSV like gurobi_results_tiny.csv with large_* rows).
    ``schema`` restricts which feasibility columns are managed: "gurobi"
    suppresses (and drops from the CSV) ``efficient_feasibility_status``;
    "efficient" suppresses ``gurobi_feasibility_status``; "both" keeps both.
    ``force=True`` overwrites existing non-empty cells (default skips them
    so a stale rerun does not clobber fresh results).
    """
    rows_dict, existing_columns = load_existing_csv(CSV_PATH)

    schema_to_col = {
        "gurobi": "gurobi_feasibility_status",
        "efficient": "efficient_feasibility_status",
    }
    prefixes = _schema_prefixes(schema)
    new_columns = [schema_to_col[p] for p in prefixes]
    drop_columns = {col for s, col in schema_to_col.items() if s not in prefixes}
    if drop_columns:
        existing_columns = [c for c in existing_columns if c not in drop_columns]

    # Insert new columns right after gap_status if not already present
    if "gap_status" in existing_columns:
        gap_idx = existing_columns.index("gap_status")
        insert_cols = [c for c in new_columns if c not in existing_columns]
        for offset, c in enumerate(insert_cols):
            existing_columns.insert(gap_idx + 1 + offset, c)
        columns = existing_columns
    else:
        columns = merge_columns(existing_columns, new_columns)

    from instance_paths import DEFAULT_INSTANCES  # local import to avoid top-level shuffle

    inst_filter = set(instances) if instances else None

    for paper_id in paper_ids:
        paper_dir = os.path.join(PAPER_DATA_DIR, paper_id)

        if inst_filter is not None:
            # User explicitly named the instance keys (e.g. large_11). Use them
            # directly — do NOT intersect with DEFAULT_INSTANCES, which is
            # ["tiny","large_1"] and would drop large_11/21/31/41/51 etc.
            all_keys = sorted(inst_filter)
        else:
            # Per-instance keys to backfill: legacy 2..9 from disk + DEFAULT_INSTANCES.
            num_instances = max(discover_solution_indices(paper_dir), default=0)
            legacy_keys = [k for k in range(2, num_instances + 1) if k not in (1, 10)]
            all_keys = [*[str(k) for k in legacy_keys], *DEFAULT_INSTANCES]
        if not all_keys:
            print(f"[FEASI-CSV] Skipping '{paper_id}': no solution files found on disk.")
            continue

        for key in all_keys:
            row_key = (paper_id, key)
            row = rows_dict.get(row_key)
            if row is None:
                row = {c: "" for c in columns}
                row["paper_id"] = paper_id
                row["instance"] = key
                rows_dict[row_key] = row
            for prefix, col_name in [
                (p, schema_to_col[p]) for p in prefixes
            ]:
                if not force and (row.get(col_name) or "") != "":
                    continue
                # Path resolver expects int for legacy keys; convert if numeric str.
                resolver_key = int(key) if isinstance(key, str) and key.isdigit() else key
                _, _, feasi_path = _resolve_paths(paper_dir, prefix, resolver_key)
                value = ""
                if os.path.isfile(feasi_path):
                    try:
                        with open(feasi_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        v = data.get("feasible")
                        value = "" if v is None else v
                    except (json.JSONDecodeError, KeyError):
                        value = ""
                row[col_name] = value

    write_csv(CSV_PATH, columns, rows_dict)
    print(f"[FEASI-CSV] Feasibility results written to {CSV_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Run feasibility checks for each paper via Claude Code Agent."
    )
    parser.add_argument(
        "--paper-id", nargs="+", dest="paper_ids", default=None,
        help="Paper IDs (folder names under data/paper_data/). "
             "Omit to auto-discover all papers with PDFs.",
    )
    parser.add_argument(
        "--exclude-paper-id", nargs="+", dest="exclude_paper_ids", default=[],
        help="Paper IDs to exclude.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1, i.e. sequential).",
    )
    parser.add_argument(
        "--paper-dir", type=str, default=None,
        help="Override the paper data directory (default: data/paper_data). "
             "Use frontier-or for the new layout.",
    )
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help="Override the output CSV path. When omitted: a single mapped "
             "--instances picks its per-instance default (large_51 -> "
             "gurobi_results_51.csv, etc.); otherwise falls back to "
             "solving_results_full.csv. See _INSTANCE_RESULTS_CSV.",
    )
    parser.add_argument(
        "--instances", nargs="+", default=None,
        help="Restrict to these instance keys (e.g. 'tiny', 'large_11'). "
             "When set, feasibility_check.py only runs for these solution files "
             "and only their rows are touched in the CSV. Omit for all instances.",
    )
    parser.add_argument(
        "--schema", choices=["both", "gurobi", "efficient"], default="both",
        help="Which solution prefix(es) to process. 'gurobi' skips efficient "
             "feasibility runs and drops efficient_feasibility_status from the CSV; "
             "'efficient' is the mirror image. Default: both.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing non-empty feasibility cells in the CSV. "
             "Default behaviour preserves any pre-existing value to avoid "
             "clobbering fresh results with stale ones; use --force after a "
             "checker fix when the new feasi_result.json should win.",
    )
    args = parser.parse_args()

    # Override module-level paths from CLI before anything reads them.
    global CSV_PATH, PAPER_DATA_DIR
    if args.paper_dir:
        PAPER_DATA_DIR = os.path.abspath(os.path.expanduser(args.paper_dir))
    instances_filter = list(args.instances) if args.instances else None
    if args.csv_path:
        CSV_PATH = os.path.abspath(os.path.expanduser(args.csv_path))
    else:
        inst_default = _default_csv_for_instances(instances_filter)
        if inst_default:
            CSV_PATH = os.path.abspath(inst_default)
    schema = args.schema

    if HARDCODED_PAPER_IDS:
        paper_ids = list(HARDCODED_PAPER_IDS)
    elif args.paper_ids:
        paper_ids = args.paper_ids
    else:
        paper_ids = discover_paper_ids()

    if HARDCODED_PAPER_EXCLUDE_IDS:
        hard_exclude = {e.lower() for e in HARDCODED_PAPER_EXCLUDE_IDS}
        paper_ids = [p for p in paper_ids if p.lower() not in hard_exclude]
    if args.exclude_paper_ids:
        exclude_set = {e.lower() for e in args.exclude_paper_ids}
        paper_ids = [p for p in paper_ids if p.lower() not in exclude_set]

    if not paper_ids:
        print("ERROR: No paper_ids to process.", file=sys.stderr)
        sys.exit(1)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    valid_items: list[tuple[str, str]] = []  # (paper_dir, paper_id)
    for paper_id in paper_ids:
        paper_dir = os.path.join(PAPER_DATA_DIR, paper_id)
        if not os.path.isdir(paper_dir):
            print(f"[WARN] Skipping paper_id '{paper_id}': directory not found: {paper_dir}",
                  file=sys.stderr)
            continue
        valid_items.append((paper_dir, paper_id))

    if not valid_items:
        print("ERROR: No valid paper directories found.", file=sys.stderr)
        sys.exit(1)

    def process_paper(paper_dir: str, paper_id: str):
        fc_script = os.path.join(paper_dir, "feasibility_check.py")

        if os.path.isfile(fc_script):
            run_existing_feasibility_check(paper_dir, paper_id,
                                           instances=instances_filter,
                                           schema=schema)
            return

        # Generate feasibility_check.py via Claude, which also runs it on existing solutions
        print(f"[START] paper_id: {paper_id}")
        prompt_body = prompt_template.replace("{paper_id}", paper_id)
        full_prompt = (
            f"Please read the files in the directory at the absolute path below:\n"
            f"{paper_dir}\n\n"
            f"{prompt_body}\n\n"
            f"All outputs must be saved inside the absolute path below:\n"
            f"{paper_dir}\n"
        )
        try:
            run_claude_tracked(full_prompt, label=f"feasibility_check:{paper_id}")
            print(f"[DONE]  paper_id: {paper_id}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[WARN]  claude failed for paper_id '{paper_id}': {e}", file=sys.stderr)

    csv_lock = threading.Lock()
    print(f"Processing {len(valid_items)} paper(s) with {args.workers} worker(s).")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_paper, paper_dir, paper_id): paper_id
            for paper_dir, paper_id in valid_items
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] paper_id '{paper_id}': {e}", file=sys.stderr)
                continue
            with csv_lock:
                collect_feasibility_results([paper_id], instances=instances_filter,
                                            schema=schema, force=args.force)


if __name__ == "__main__":
    main()
