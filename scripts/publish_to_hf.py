#!/usr/bin/env python3
"""Publish the local FrontierOR/ dataset to Hugging Face with cleaner names.

Cosmetic renames applied at publish time (local working tree is left untouched):

    large_instance_X1.json      -> large_instance_X.json      (X=1..5)
    large_solution_X1.json      -> large_solution_X.json
    large_feasi_result_X1.json  -> large_feasi_result_X.json
    large_log_X1.jsonl          -> large_log_X.jsonl

Locally we keep the two-digit `_ij` suffix because (i = diversity, j = scale)
is the contract the prompts/ pipeline relies on; this script only collapses
the public-facing j=1 slice to single digits.

Idempotent: re-runs converge on the same HF state. Designed for monthly
re-publishing. Hardlinks are used in the staging tree so an extra 32 GB of
disk is NOT consumed.

Usage:
    python scripts/publish_to_hf.py              # publish
    python scripts/publish_to_hf.py --dry-run    # preview, no remote changes
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi

REPO_ID_DEFAULT = "SmartOR/FrontierOR"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ROOT_DEFAULT = PROJECT_ROOT / "FrontierOR"
STAGING_DEFAULT = PROJECT_ROOT / "_publish_staging"

RENAME_RULES = [
    (re.compile(r"^(large_(?:instance|solution|feasi_result))_([1-5])1\.json$"),
     r"\1_\2.json"),
    (re.compile(r"^(large_log)_([1-5])1\.jsonl$"),
     r"\1_\2.jsonl"),
]

LEGACY_RE = re.compile(
    r"(?:^|/)large_(?:instance|solution|feasi_result)_[1-5]1\.json$"
    r"|(?:^|/)large_log_[1-5]1\.jsonl$"
)


def transform_name(name: str) -> str:
    for pat, sub in RENAME_RULES:
        if pat.match(name):
            return pat.sub(sub, name)
    return name


def build_staging(local_root: Path, staging: Path) -> tuple[int, int]:
    """Mirror local_root into staging via hardlinks, applying the rename rules.

    Preserves staging/.cache/huggingface across runs so upload-large-folder
    can resume from its prior hash/upload state.
    """
    cache_src = staging / ".cache" / "huggingface"
    cache_backup = staging.parent / "_publish_cache_backup"
    if cache_src.exists():
        if cache_backup.exists():
            shutil.rmtree(cache_backup)
        shutil.move(str(cache_src), str(cache_backup))

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if cache_backup.exists():
        (staging / ".cache").mkdir(parents=True, exist_ok=True)
        shutil.move(str(cache_backup), str(staging / ".cache" / "huggingface"))

    n_files = n_renamed = 0
    for src in local_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(local_root)
        if rel.parts and rel.parts[0] == ".cache":
            continue
        new_rel = rel.with_name(transform_name(rel.name))
        dst = staging / new_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n_files += 1
        if new_rel.name != rel.name:
            n_renamed += 1
    return n_files, n_renamed


def cleanup_legacy_on_remote(repo_id: str, local_root: Path, dry_run: bool) -> int:
    """Delete legacy _X1 files from HF.

    Safety net: only touches paper folders that ALSO exist locally — drift
    folders (remote-only papers) are left untouched. Detect & warn about
    them so they can be reviewed separately.
    """
    api = HfApi()
    remote = list(api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE))
    local_papers = {p.name for p in local_root.iterdir() if p.is_dir() and p.name != ".cache"}

    remote_papers = {p.split("/", 1)[0] for p in remote if "/" in p}
    remote_only = sorted(remote_papers - local_papers)
    if remote_only:
        print(f"      WARN: {len(remote_only)} paper folder(s) exist on remote but not locally:")
        for p in remote_only:
            print(f"        - {p}  (untouched; review manually)")

    leftover = [p for p in remote
                if LEGACY_RE.search("/" + p)
                and p.split("/", 1)[0] in local_papers]
    if not leftover:
        print("      No legacy _X1 files to clean (within local paper set).")
        return 0
    print(f"      Found {len(leftover)} legacy _X1 files to delete.")
    if dry_run:
        for p in leftover[:5]:
            print(f"        would delete: {p}")
        if len(leftover) > 5:
            print(f"        ... and {len(leftover) - 5} more")
        return len(leftover)
    ops = [CommitOperationDelete(path_in_repo=p) for p in leftover]
    api.create_commit(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        operations=ops,
        commit_message=f"Cleanup: remove {len(leftover)} legacy _X1 suffix files",
    )
    return len(leftover)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT_DEFAULT)
    parser.add_argument("--staging", type=Path, default=STAGING_DEFAULT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done, make no remote changes")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Build staging only; skip HF upload and cleanup")
    args = parser.parse_args()

    if not args.local_root.is_dir():
        print(f"ERROR: local root not found: {args.local_root}", file=sys.stderr)
        return 1

    print(f"Source : {args.local_root}")
    print(f"Staging: {args.staging}")
    print(f"Repo   : {args.repo_id}")
    print()

    print("[1/3] Building staging tree via hardlinks (zero extra disk)...")
    n_files, n_renamed = build_staging(args.local_root, args.staging)
    print(f"      {n_files} files staged, {n_renamed} with renamed paths.")
    print()

    if args.skip_upload:
        print("--skip-upload set, stopping after staging build.")
        return 0

    print("[2/3] hf upload-large-folder (idempotent; unchanged files are skipped)...")
    cmd = ["hf", "upload-large-folder", args.repo_id, str(args.staging),
           "--repo-type=" + REPO_TYPE, "--no-bars"]
    if args.dry_run:
        print(f"      [dry-run] would run: {' '.join(cmd)}")
    else:
        subprocess.run(cmd, check=True)
    print()

    print("[3/3] Cleanup legacy _X1 files left on remote from prior layout...")
    cleanup_legacy_on_remote(args.repo_id, args.local_root, dry_run=args.dry_run)
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
