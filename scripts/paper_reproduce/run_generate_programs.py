"""
Generate gurobi_code.py + efficient_algorithm.py for each paper via Claude Code Agent.

Paper selection priority (highest to lowest):
1) HARDCODED_PAPER_IDS (if non-empty)
2) --paper-id CLI arguments (if provided)
3) Auto-discovery by scanning data/paper_data/*/*.pdf

Exclusions (HARDCODED_PAPER_EXCLUDE_IDS and --exclude_id) always win.

Usage examples:
    python scripts/paper_reproduce/run_generate_programs.py --paper-id mingozzi1999 SomeOtherPaper --num_instances 10
    python scripts/paper_reproduce/run_generate_programs.py
    python scripts/paper_reproduce/run_generate_programs.py --exclude_id mingozzi1999
    python scripts/paper_reproduce/run_generate_programs.py --workers 2
"""

import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.utils.claude_utils import run_claude_tracked
PAPER_DATA_DIR = os.path.join(BASE_DIR, "data", "paper_data")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "paper_reproduce", "prompt_generate_programs.txt")

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


def resolve_pdfs(paper_ids: list[str]) -> list[tuple[str, str, str]]:
    """
    For each paper_id, find the PDF file(s) under its directory.
    Returns list of (pdf_path, paper_dir, paper_id).
    """
    results = []
    seen = set()
    for paper_id in paper_ids:
        paper_dir = os.path.join(PAPER_DATA_DIR, paper_id)
        if not os.path.isdir(paper_dir):
            print(f"[WARN] Skipping paper_id '{paper_id}' because directory not found: {paper_dir}",
                  file=sys.stderr)
            continue
        for pdf in sorted(glob.glob(os.path.join(paper_dir, "*.pdf"))):
            if pdf not in seen:
                seen.add(pdf)
                results.append((pdf, paper_dir, paper_id))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate gurobi_code.py + efficient_algorithm.py via Claude Code Agent."
    )
    parser.add_argument(
        "--paper-id", nargs="+", dest="paper_ids", default=None,
        help="Paper IDs (folder names under data/paper_data/). "
             "Omit to auto-discover all papers with PDFs.",
    )
    parser.add_argument(
        "--exclude_id", nargs="+", dest="exclude_paper_ids", default=[],
        help="Paper IDs to exclude. Always wins over --paper-id and "
             "auto-discovery, so listed IDs are dropped even when other "
             "selectors would include them.",
    )
    parser.add_argument(
        "--num_instances", type=int, default=10,
        help="Number of instances (used to fill {num_instances} in prompt, default: 10).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1, i.e. sequential).",
    )
    args = parser.parse_args()

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

    pdf_items = resolve_pdfs(paper_ids)
    if not pdf_items:
        print(f"ERROR: No PDFs found for paper_ids: {', '.join(paper_ids)}",
              file=sys.stderr)
        sys.exit(1)

    num_instances = args.num_instances

    def process_paper(pdf_path: str, paper_dir: str, paper_id: str):
        required = ("gurobi_code.py", "efficient_algorithm.py")
        if all(os.path.isfile(os.path.join(paper_dir, f)) for f in required):
            print(f"[SKIP]  paper_id '{paper_id}': {', '.join(required)} already exist. Skipping.")
            return
        print(f"[START] paper_id: {paper_id}")
        prompt_body = (prompt_template
                       .replace("{paper_id}", paper_id)
                       .replace("{num_instances}", str(num_instances)))
        full_prompt = (
            f"Please read the PDF file at the absolute path below:\n"
            f"{pdf_path}\n\n"
            f"{prompt_body}\n\n"
            f"All outputs must be saved inside the absolute path below:\n"
            f"{paper_dir}\n"
        )
        try:
            run_claude_tracked(full_prompt, label=f"generate_programs:{paper_id}")
            print(f"[DONE]  paper_id: {paper_id}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[WARN]  claude failed for paper_id '{paper_id}': {e}", file=sys.stderr)

    print(f"Processing {len(pdf_items)} paper(s) with {args.workers} worker(s).")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_paper, pdf_path, paper_dir, paper_id): paper_id
            for pdf_path, paper_dir, paper_id in pdf_items
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] paper_id '{paper_id}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
