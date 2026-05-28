"""
Generate problem_description.txt for each paper via Claude.

Calls Claude via run_claude_tracked with prompt_generate_problem_description.txt
to generate problem_description.txt (natural-language problem description with
inline JSON field names).

Paper selection priority (highest to lowest):
1) HARDCODED_PAPER_IDS (if non-empty)
2) --paper-id CLI arguments (if provided)
3) Auto-discovery by scanning data/paper_data/*/*.pdf

Exclusions (HARDCODED_PAPER_EXCLUDE_IDS and --exclude-paper-id) always win.

Usage examples:
    python scripts/paper_reproduce/run_generate_problem_descriptions.py --paper-id belvaux2000
    python scripts/paper_reproduce/run_generate_problem_descriptions.py
    python scripts/paper_reproduce/run_generate_problem_descriptions.py --exclude-paper-id mingozzi1999
    python scripts/paper_reproduce/run_generate_problem_descriptions.py --workers 4
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
PROMPT_PATH = os.path.join(
    BASE_DIR, "prompts", "paper_reproduce", "prompt_generate_problem_description.txt"
)
REWRITE_PROMPT_PATH = os.path.join(
    BASE_DIR, "prompts", "paper_reproduce", "prompt_description_rewrite.txt"
)
DATA_SPEC_PROMPT_PATH = os.path.join(
    BASE_DIR, "prompts", "paper_reproduce", "prompt_generate_data_specification.txt"
)
REWRITE_CSV_PATH = os.path.join(BASE_DIR, "results", "data_reproduce", "rewrite_problem_description.csv")

# Hardcoded paper_id list. If non-empty, this takes highest priority.
HARDCODED_PAPER_IDS: list[str] = [
]

# Hardcoded exclusion list. These IDs are always removed.
HARDCODED_PAPER_EXCLUDE_IDS: list[str] = [
    # "some_paper_id_to_skip",
]


def discover_paper_ids(paper_data_dir: str | None = None) -> list[str]:
    """Auto-discover paper_ids by scanning <paper_data_dir>/*/*.pdf.

    Falls back to subdirectories containing ``gurobi_code.py`` when no PDFs
    are present (e.g. ``frontier-or/`` may not ship PDFs).
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


def resolve_pdfs(paper_ids: list[str], paper_data_dir: str | None = None) -> list[tuple[str, str, str]]:
    """For each paper_id, find the PDF file(s) under ``<paper_data_dir>/<paper_id>/``.

    If the folder has no PDFs (benchmark-only folders), still returns an entry
    with ``pdf_path=""`` so schema-only flows can proceed.
    Returns list of (pdf_path, paper_dir, paper_id).
    """
    root = paper_data_dir or PAPER_DATA_DIR
    results = []
    seen = set()
    for paper_id in paper_ids:
        paper_dir = os.path.join(root, paper_id)
        if not os.path.isdir(paper_dir):
            print(
                f"[WARN] Skipping paper_id '{paper_id}' because directory not found: {paper_dir}",
                file=sys.stderr,
            )
            continue
        pdfs = sorted(glob.glob(os.path.join(paper_dir, "*.pdf")))
        if pdfs:
            for pdf in pdfs:
                if pdf not in seen:
                    seen.add(pdf)
                    results.append((pdf, paper_dir, paper_id))
        else:
            results.append(("", paper_dir, paper_id))
    return results


def generate_problem_description(
    pdf_path: str, paper_dir: str, paper_id: str, prompt_template: str
) -> bool:
    """
    Call Claude to generate problem_description.txt.
    Returns True on success, False on failure.
    """
    prompt_body = prompt_template.replace("{paper_id}", paper_id)
    full_prompt = (
        f"Please read the PDF file at the absolute path below:\n"
        f"{pdf_path}\n\n"
        f"{prompt_body}\n\n"
        f"All outputs must be saved inside the absolute path below:\n"
        f"{paper_dir}\n"
    )
    try:
        run_claude_tracked(
            full_prompt, label=f"generate_problem_description:{paper_id}"
        )
        out = os.path.join(paper_dir, "problem_description.txt")
        if not os.path.isfile(out):
            print(
                f"[WARN]  paper_id '{paper_id}': Claude finished but problem_description.txt not found.",
                file=sys.stderr,
            )
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"[WARN]  Generation failed for paper_id '{paper_id}': {e}",
            file=sys.stderr,
        )
        return False


def rewrite_problem_description(
    pdf_path: str, paper_dir: str, paper_id: str, prompt_template: str
) -> bool:
    """
    Call Claude to review and revise problem_description.txt.
    Produces problem_description_v2.txt and appends to rephrase_problem_description.csv.
    Returns True on success, False on failure.
    """
    prompt_body = prompt_template.replace("{paper_id}", paper_id)
    prompt_body = prompt_body.replace("{csv_path}", REWRITE_CSV_PATH)
    full_prompt = (
        f"Please read the PDF file at the absolute path below:\n"
        f"{pdf_path}\n\n"
        f"{prompt_body}\n\n"
        f"All source files and outputs are in the absolute path below:\n"
        f"{paper_dir}\n"
    )
    try:
        run_claude_tracked(
            full_prompt, label=f"rewrite_problem_description:{paper_id}"
        )
        out = os.path.join(paper_dir, "problem_description_v2.txt")
        if not os.path.isfile(out):
            print(
                f"[WARN]  paper_id '{paper_id}': Claude finished but problem_description_v2.txt not found.",
                file=sys.stderr,
            )
            return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"[WARN]  Generation failed for paper_id '{paper_id}' (rewrite): {e}",
            file=sys.stderr,
        )
        return False


def generate_templates(
    paper_dir: str, paper_id: str, prompt_template: str,
    target: str = "both",
) -> bool:
    """
    Call Claude to generate instance_schema.json and/or solution_schema.json.

    ``target`` selects which schema file(s) to produce:
      - ``"both"``     — generate both (default)
      - ``"instance"`` — only instance_schema.json
      - ``"solution"`` — only solution_schema.json
    Requires ``problem_description.txt`` (or the PDF) to already exist.
    Returns True on success, False on failure.
    """
    instance_path = os.path.join(paper_dir, "instance", "tiny_instance.json")
    if not os.path.isfile(instance_path):
        print(
            f"[WARN]  paper_id '{paper_id}': instance/tiny_instance.json not found, "
            f"skipping templates.",
            file=sys.stderr,
        )
        return False

    with open(instance_path, "r", encoding="utf-8") as f:
        instance_json = f.read()

    # Try gurobi_solution first, then fall back to efficient_solution.
    solution_json = None
    for sub in ("gurobi_solution", "efficient_solution"):
        sol_path = os.path.join(paper_dir, sub, "tiny_solution.json")
        if os.path.isfile(sol_path):
            with open(sol_path, "r", encoding="utf-8") as f:
                solution_json = f.read()
            break
    if solution_json is None:
        print(
            f"[WARN]  paper_id '{paper_id}': no tiny_solution.json "
            f"(under gurobi_solution/ or efficient_solution/) found, skipping templates.",
            file=sys.stderr,
        )
        return False

    prompt_body = prompt_template.replace("{paper_id}", paper_id)
    prompt_body = prompt_body.replace("{instance_json}", instance_json)
    prompt_body = prompt_body.replace("{solution_json}", solution_json)
    if target == "solution":
        scope_directive = (
            "## FOCUS OVERRIDE\n"
            "Only produce `solution_schema.json`. Do NOT create or modify "
            "`instance_schema.json`; skip every instruction below that asks for it.\n\n"
        )
    elif target == "instance":
        scope_directive = (
            "## FOCUS OVERRIDE\n"
            "Only produce `instance_schema.json`. Do NOT create or modify "
            "`solution_schema.json`; skip every instruction below that asks for it.\n\n"
        )
    else:
        scope_directive = ""
    full_prompt = (
        f"{scope_directive}{prompt_body}\n\n"
        f"All source files and outputs are in the absolute path below:\n"
        f"{paper_dir}\n"
    )
    try:
        run_claude_tracked(
            full_prompt, label=f"generate_templates:{paper_id}:{target}"
        )
        inst_out = os.path.join(paper_dir, "instance_schema.json")
        sol_out = os.path.join(paper_dir, "solution_schema.json")
        ok = True
        if target in ("both", "instance") and not os.path.isfile(inst_out):
            print(
                f"[WARN]  paper_id '{paper_id}': instance_schema.json not found.",
                file=sys.stderr,
            )
            ok = False
        if target in ("both", "solution") and not os.path.isfile(sol_out):
            print(
                f"[WARN]  paper_id '{paper_id}': solution_schema.json not found.",
                file=sys.stderr,
            )
            ok = False
        return ok
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"[WARN]  Generation failed for paper_id '{paper_id}' (templates): {e}",
            file=sys.stderr,
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate problem_description.txt (via Claude) for each paper."
    )
    parser.add_argument(
        "--paper-id",
        nargs="+",
        dest="paper_ids",
        default=None,
        help="Paper IDs (folder names under data/paper_data/). "
        "Omit to auto-discover all papers with PDFs.",
    )
    parser.add_argument(
        "--exclude-paper-id",
        nargs="+",
        dest="exclude_paper_ids",
        default=[],
        help="Paper IDs to exclude.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, i.e. sequential).",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 = skip existing files (default), 1 = overwrite and regenerate.",
    )
    parser.add_argument(
        "--paper-dir",
        dest="paper_dir",
        default=None,
        help=f"Root directory containing one sub-folder per paper. "
             f"Default: {PAPER_DATA_DIR}. "
             f"Pass 'frontier-or' to write schemas there.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        dest="schema_only",
        help="Skip Step 1 (problem_description.txt) and Step 2 (problem_description_v2.txt); "
             "run only Step 3 (schema generation). Reuses any existing description files "
             "(or the PDF) as the knowledge source.",
    )
    parser.add_argument(
        "--schema-target",
        choices=["both", "instance", "solution"],
        default="both",
        dest="schema_target",
        help="Which schema file(s) to generate in Step 3 (default: both).",
    )
    args = parser.parse_args()

    paper_data_dir = os.path.abspath(args.paper_dir) if args.paper_dir else PAPER_DATA_DIR
    if not os.path.isdir(paper_data_dir):
        print(f"ERROR: --paper-dir does not exist: {paper_data_dir}", file=sys.stderr)
        sys.exit(1)

    if HARDCODED_PAPER_IDS:
        paper_ids = list(HARDCODED_PAPER_IDS)
    elif args.paper_ids:
        paper_ids = args.paper_ids
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

    # Step 1/2 templates are not needed in schema-only mode.
    prompt_template = ""
    rewrite_prompt_template = ""
    if not args.schema_only:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        with open(REWRITE_PROMPT_PATH, "r", encoding="utf-8") as f:
            rewrite_prompt_template = f.read()
    with open(DATA_SPEC_PROMPT_PATH, "r", encoding="utf-8") as f:
        template_prompt_template = f.read()

    pdf_items = resolve_pdfs(paper_ids, paper_data_dir)
    if not pdf_items:
        print(
            f"ERROR: No paper folders found for paper_ids: {', '.join(paper_ids)}",
            file=sys.stderr,
        )
        sys.exit(1)

    overwrite = bool(args.overwrite)
    schema_only = args.schema_only
    schema_target = args.schema_target

    def process_paper(pdf_path: str, paper_dir: str, paper_id: str):
        if not schema_only:
            # Step 1: Generate problem_description.txt
            desc_path = os.path.join(paper_dir, "problem_description.txt")
            if os.path.isfile(desc_path) and not overwrite:
                print(
                    f"[SKIP]  paper_id '{paper_id}': problem_description.txt already exists."
                )
            else:
                print(f"[START] paper_id '{paper_id}': generating problem_description.txt")
                ok = generate_problem_description(
                    pdf_path, paper_dir, paper_id, prompt_template
                )
                if ok:
                    print(f"[DONE]  paper_id '{paper_id}': problem_description.txt complete.")
                else:
                    print(f"[FAIL]  paper_id '{paper_id}': problem_description.txt generation failed.")
                    return  # data_specification depends on problem_description

            # Step 2: Review and rewrite → problem_description_v2.txt
            v2_path = os.path.join(paper_dir, "problem_description_v2.txt")
            if os.path.isfile(v2_path) and not overwrite:
                print(
                    f"[SKIP]  paper_id '{paper_id}': problem_description_v2.txt already exists."
                )
            else:
                print(f"[START] paper_id '{paper_id}': rewriting problem_description_v2.txt")
                ok = rewrite_problem_description(
                    pdf_path, paper_dir, paper_id, rewrite_prompt_template
                )
                if ok:
                    print(f"[DONE]  paper_id '{paper_id}': problem_description_v2.txt complete.")
                else:
                    print(f"[FAIL]  paper_id '{paper_id}': problem_description rewrite failed.")

        # Step 3: Generate schema(s) per --schema-target
        inst_tpl = os.path.join(paper_dir, "instance_schema.json")
        sol_tpl = os.path.join(paper_dir, "solution_schema.json")
        already_have = True
        if schema_target in ("both", "instance") and not os.path.isfile(inst_tpl):
            already_have = False
        if schema_target in ("both", "solution") and not os.path.isfile(sol_tpl):
            already_have = False
        if already_have and not overwrite:
            print(
                f"[SKIP]  paper_id '{paper_id}': required schema(s) for target='{schema_target}' already exist."
            )
            return

        print(f"[START] paper_id '{paper_id}': generating schema (target={schema_target})")
        ok = generate_templates(
            paper_dir, paper_id, template_prompt_template, target=schema_target
        )
        if ok:
            print(f"[DONE]  paper_id '{paper_id}': schema (target={schema_target}) complete.")
        else:
            print(f"[FAIL]  paper_id '{paper_id}': schema generation failed.")

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
