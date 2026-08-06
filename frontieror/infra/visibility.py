"""Materialize the exact benchmark view exposed to an agent system."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from frontieror.infra.contracts import public_scoring_contract, visibility_contract
from frontieror.infra.files import copy_regular_file, sha256_regular_file
from scripts.utils.instance_paths import instance_path, is_valid_instance_name


PUBLIC_TASK_FILES = (
    "problem_description.txt",
    "instance_schema.json",
    "solution_schema.json",
)
MAX_PUBLIC_FILE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PublicPaperView:
    paper_id: str
    source_paper_dir: str
    public_paper_dir: str
    copied: list[str]
    withheld: list[str]


def validate_instance_content_split(
    *,
    paper_dir: str,
    dev_instances: Iterable[str],
    final_instances: Iterable[str],
) -> None:
    """Reject renamed or duplicated JSON across the dev/final boundary."""
    source_root = Path(paper_dir).resolve()

    def digest_for(instance: str, role: str) -> str:
        if not is_valid_instance_name(instance):
            raise ValueError(f"invalid {role} instance name: {instance!r}")
        path = Path(instance_path(os.fspath(source_root), instance))
        return sha256_regular_file(
            path,
            max_bytes=MAX_PUBLIC_FILE_BYTES,
            label=f"{role} instance {instance}",
        )

    dev_by_digest: dict[str, str] = {}
    for instance in dev_instances:
        digest = digest_for(str(instance), "dev")
        if digest in dev_by_digest:
            raise ValueError(
                "agent mode dev instances contain identical JSON content: "
                f"{dev_by_digest[digest]}, {instance}"
            )
        dev_by_digest[digest] = str(instance)

    final_by_digest: dict[str, str] = {}
    for instance in final_instances:
        digest = digest_for(str(instance), "final")
        if digest in dev_by_digest:
            raise ValueError(
                "agent mode requires content-disjoint dev/final instances; "
                f"{dev_by_digest[digest]} and {instance} have identical JSON"
            )
        if digest in final_by_digest:
            raise ValueError(
                "agent mode final instances contain identical JSON content: "
                f"{final_by_digest[digest]}, {instance}"
            )
        final_by_digest[digest] = str(instance)


def _copy(source: Path, destination: Path, relative: str, copied: list[str]) -> None:
    if not os.path.lexists(source):
        raise FileNotFoundError(f"required public task file is missing: {relative}")
    copy_regular_file(
        source,
        destination,
        max_bytes=MAX_PUBLIC_FILE_BYTES,
        label=f"public task file {relative}",
        mode=0o600,
    )
    copied.append(relative)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def materialize_public_paper_view(
    *,
    paper_dir: str,
    public_root: str,
    paper_id: str,
    instances: Iterable[str],
    seed_code_path: str | None = None,
    stage_boundary: float = 0.01,
) -> PublicPaperView:
    """Create a fresh dev workspace containing no trusted grading material.

    ``instances`` means stage-1/dev instances only. Final instances are never
    accepted by this API's caller until after the agent is stopped and code.py
    is frozen.
    """
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", paper_id) is None:
        raise ValueError(f"invalid paper id: {paper_id!r}")
    source_root = Path(paper_dir).resolve()
    destination = (Path(public_root).resolve() / paper_id)
    if not source_root.is_dir():
        raise FileNotFoundError(f"paper directory does not exist: {source_root}")
    if (
        source_root == destination
        or source_root in destination.parents
        or destination in source_root.parents
    ):
        raise ValueError("public workspace and trusted paper directory must not overlap")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, mode=0o700)

    copied: list[str] = []
    for filename in PUBLIC_TASK_FILES:
        _copy(source_root / filename, destination / filename, filename, copied)

    for instance in dict.fromkeys(str(value) for value in instances):
        if not is_valid_instance_name(instance):
            raise ValueError(f"invalid public instance name: {instance!r}")
        source = Path(instance_path(os.fspath(source_root), instance))
        relative = os.path.join("instance", source.name)
        _copy(source, destination / relative, relative, copied)

    if seed_code_path:
        _copy(
            Path(seed_code_path).resolve(),
            destination / "code.py",
            "code.py",
            copied,
        )

    contract_name = "benchmark_contract.json"
    _write_json(
        destination / contract_name,
        {
            "scoring": public_scoring_contract(stage_boundary=stage_boundary),
            "visibility": visibility_contract(),
        },
    )
    copied.append(contract_name)

    manifest = {
        "schema_version": 2,
        "paper_id": paper_id,
        "files": {
            relative: _sha256(destination / relative)
            for relative in sorted(copied)
        },
        "workspace_role": "agent_dev",
    }
    _write_json(destination / "public_manifest.json", manifest)
    # Keep the old filename for runs created by earlier Infra revisions. Its
    # schema remains stable for compatibility; new consumers use the v2 file.
    _write_json(
        destination / "anti_hack_manifest.json",
        {
            "schema_version": 1,
            "paper_id": paper_id,
            "copied": sorted(copied),
            "sha256": {
                relative: _sha256(destination / relative)
                for relative in sorted(copied)
            },
        },
    )

    return PublicPaperView(
        paper_id=paper_id,
        source_paper_dir=os.fspath(source_root),
        public_paper_dir=os.fspath(destination),
        copied=sorted(copied),
        withheld=[
            "final instances",
            "reference solutions and runtimes",
            "feasibility checker",
            "private evaluation traces",
        ],
    )
