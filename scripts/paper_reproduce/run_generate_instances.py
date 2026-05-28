"""
Generate test instances for each paper via the Claude Code Agent.

Paper selection priority (highest to lowest):
1) Start from one of the following sources:
   - `HARDCODED_PAPER_IDS` (if non-empty)
   - otherwise `--paper-id` command-line arguments (if provided)
   - otherwise auto-discovery by scanning `data/paper_data/*/*.pdf`
2) Apply exclusions. If an ID appears in an exclude list, it is always removed
   even if it also appears in an include list (exclusions win).

To quickly control which papers to run without changing CLI args, edit the
`HARDCODED_PAPER_IDS` / `HARDCODED_PAPER_EXCLUDE_IDS` lists near the top of
this file. Leave them as `[]` to fall back to CLI args or auto-discovery.

Usage examples:
    python scripts/paper_reproduce/run_generate_instances.py --paper-id mingozzi1999 SomeOtherPaper --num_instances 10
    python scripts/paper_reproduce/run_generate_instances.py --num_instances 10
    python scripts/paper_reproduce/run_generate_instances.py --num_instances 10 --exclude-paper-id mingozzi1999 SomeOtherPaper
    python scripts/paper_reproduce/run_generate_instances.py --num_instances 10 --workers 2
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.utils.claude_utils import run_claude_tracked
PAPER_DATA_DIR = os.path.join(BASE_DIR, "data", "paper_data")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "paper_reproduce", "prompt_generate_instances.txt")

# Hardcoded paper_id list. If non-empty, this takes highest priority.
HARDCODED_PAPER_IDS: list[str] = [
]

# Hardcoded paper_id exclusion list. If non-empty, these IDs are always excluded
# (even if they also appear in HARDCODED_PAPER_IDS or --paper-id).
HARDCODED_PAPER_EXCLUDE_IDS: list[str] = [
    "roberti2021",
    "mingozzi1999",
    "amaldi2013",
    "ceselli2009",
    "feillet2005",
    "hadjar2006",
]

def discover_paper_ids() -> list[str]:
    """Auto-discover paper_ids by scanning data/paper_data/*/*.pdf."""
    ids = set()
    for pdf in glob.glob(os.path.join(PAPER_DATA_DIR, "*", "*.pdf")):
        ids.add(os.path.basename(os.path.dirname(pdf)))
    return sorted(ids)


def max_existing_instance(paper_dir: str) -> int:
    """Return the largest x such that instance_x.json exists in paper_dir, or 0."""
    max_x = 0
    for path in glob.glob(os.path.join(paper_dir, "instance_*.json")):
        m = re.search(r"instance_(\d+)\.json$", os.path.basename(path))
        if m:
            max_x = max(max_x, int(m.group(1)))
    return max_x


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
        description="Generate test instances for each paper via Claude Code Agent."
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
        "--num_instances", type=int, default=10,
        help="Number of instances to generate per paper (default: 10).",
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
        hard_exclude_set = {e.lower() for e in HARDCODED_PAPER_EXCLUDE_IDS}
        paper_ids = [p for p in paper_ids if p.lower() not in hard_exclude_set]
    if args.exclude_paper_ids:
        exclude_set = {e.lower() for e in args.exclude_paper_ids}
        paper_ids = [p for p in paper_ids if p.lower() not in exclude_set]

    if not paper_ids:
        print("ERROR: No paper_ids to process. "
              "(--paper-id was empty or all were excluded by --exclude-paper-id.)",
              file=sys.stderr)
        sys.exit(1)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    pdf_items = resolve_pdfs(paper_ids)
    if not pdf_items:
        print(f"ERROR: No PDFs found for paper_ids: {', '.join(paper_ids)} "
              f"under data/paper_data/<paper_id>/*.pdf",
              file=sys.stderr)
        sys.exit(1)

    num_instances = args.num_instances

    def process_paper(pdf_path: str, paper_dir: str, paper_id: str):
        existing_max = max_existing_instance(paper_dir)
        if existing_max >= num_instances:
            print(f"[SKIP]  paper_id '{paper_id}': already has instance_1..instance_{existing_max}.json "
                  f"(>= {num_instances} requested). Skipping.")
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
            run_claude_tracked(full_prompt, label=f"generate_instances:{paper_id}")
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
