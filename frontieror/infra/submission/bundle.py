"""Submission bundle validation for official FrontierOR evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from frontieror.infra.files import SecureFileError, read_regular_file


REQUIRED_METADATA = (
    "author",
    "model_or_agent",
    "framework",
    "paper_id",
    "track",
    "created_at",
)

SUPPORTED_FRAMEWORKS = {"one_shot", "openevolve", "eoh", "coral", "manual"}
SUPPORTED_TRACKS = {"official"}

PRIVATE_MARKERS = (
    "gurobi_solution",
    "feasibility_check.py",
    "gurobi_results_",
    "FRONTIER_OR_DATA_DIR",
    "OPENROUTER_API_KEY",
    ".coral/private",
)

MAX_CODE_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_METADATA_VALUE_CHARS = 512


class SubmissionBundleError(ValueError):
    """Raised when a submission bundle is malformed or rule-ineligible."""


@dataclass(frozen=True)
class SubmissionBundle:
    root: Path
    metadata_path: Path
    code_path: Path
    code_bytes: bytes
    metadata: Dict[str, Any]
    code_sha256: str
    submission_id: str
    manifest: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = read_regular_file(
            path,
            max_bytes=MAX_METADATA_BYTES,
            label="submission.json",
        )
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise SubmissionBundleError("submission.json must be UTF-8 text") from e
    except json.JSONDecodeError as e:
        raise SubmissionBundleError(f"invalid submission.json: {e}") from e
    except SecureFileError as e:
        raise SubmissionBundleError(str(e)) from e
    if not isinstance(data, dict):
        raise SubmissionBundleError("submission.json must contain a JSON object")
    return data


def _sanitize_token(raw: Any) -> str:
    text = str(raw).strip().lower()[:MAX_METADATA_VALUE_CHARS]
    parts = re.findall(r"[a-z0-9]+", text)
    return ("-".join(parts) or "unknown")[:96]


def _check_required_metadata(metadata: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_METADATA if not metadata.get(key)]
    if missing:
        raise SubmissionBundleError(f"submission.json missing required fields: {', '.join(missing)}")
    for key in REQUIRED_METADATA:
        value = metadata[key]
        if not isinstance(value, str):
            raise SubmissionBundleError(f"submission.json field {key!r} must be a string")
        if len(value) > MAX_METADATA_VALUE_CHARS:
            raise SubmissionBundleError(
                f"submission.json field {key!r} exceeds "
                f"{MAX_METADATA_VALUE_CHARS} characters"
            )
    framework = str(metadata["framework"])
    if framework not in SUPPORTED_FRAMEWORKS:
        raise SubmissionBundleError(
            f"unsupported framework {framework!r}; expected one of {sorted(SUPPORTED_FRAMEWORKS)}"
        )
    track = str(metadata["track"])
    if track not in SUPPORTED_TRACKS:
        raise SubmissionBundleError(
            f"unsupported track {track!r}; expected one of {sorted(SUPPORTED_TRACKS)}"
        )
    paper_id = str(metadata["paper_id"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", paper_id) is None:
        raise SubmissionBundleError("submission.json field 'paper_id' is invalid")


def _check_private_markers(code_text: str) -> None:
    for marker in PRIVATE_MARKERS:
        if marker in code_text:
            raise SubmissionBundleError(
                f"code.py contains private benchmark marker {marker!r}"
            )


def load_submission_bundle(
    bundle_dir: os.PathLike[str] | str,
    *,
    expected_paper_id: Optional[str] = None,
) -> SubmissionBundle:
    """Load, validate, and fingerprint a final-code submission bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir():
        raise SubmissionBundleError(f"submission directory does not exist: {root}")

    metadata_path = root / "submission.json"
    code_path = root / "code.py"
    if not metadata_path.exists():
        raise SubmissionBundleError("missing submission.json")
    if not code_path.exists():
        raise SubmissionBundleError("missing code.py")

    metadata = _read_json(metadata_path)
    _check_required_metadata(metadata)

    paper_id = str(metadata["paper_id"])
    if expected_paper_id is not None and paper_id != expected_paper_id:
        raise SubmissionBundleError(
            f"paper_id mismatch: submission has {paper_id!r}, expected {expected_paper_id!r}"
        )

    try:
        code_bytes = read_regular_file(
            code_path,
            max_bytes=MAX_CODE_BYTES,
            label="code.py",
        )
        code_text = code_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SubmissionBundleError("code.py must be UTF-8 text") from e
    except SecureFileError as e:
        raise SubmissionBundleError(str(e)) from e
    _check_private_markers(code_text)

    code_sha256 = hashlib.sha256(code_bytes).hexdigest()
    submission_id = "-".join(
        (
            _sanitize_token(metadata["paper_id"]),
            _sanitize_token(metadata["framework"]),
            _sanitize_token(metadata["model_or_agent"]),
            code_sha256[:12],
        )
    )

    manifest = {
        "submission_id": submission_id,
        "bundle_dir": str(root),
        "metadata_path": str(metadata_path),
        "code_path": str(code_path),
        "code_sha256": code_sha256,
        "metadata": dict(metadata),
    }
    return SubmissionBundle(
        root=root,
        metadata_path=metadata_path,
        code_path=code_path,
        code_bytes=code_bytes,
        metadata=dict(metadata),
        code_sha256=code_sha256,
        submission_id=submission_id,
        manifest=manifest,
    )
