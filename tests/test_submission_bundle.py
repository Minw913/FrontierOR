import json
from pathlib import Path

import pytest


def _write_bundle(root: Path, *, paper_id: str = "paper1", code: str = "print('ok')\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "submission.json").write_text(
        json.dumps(
            {
                "author": "Team Frontier",
                "model_or_agent": "gpt-5.3-codex",
                "framework": "coral",
                "paper_id": paper_id,
                "track": "official",
                "created_at": "2026-06-30T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (root / "code.py").write_text(code, encoding="utf-8")
    return root


def test_load_submission_bundle_computes_stable_identity(tmp_path):
    from scripts.utils.submission_bundle import load_submission_bundle

    bundle_dir = _write_bundle(tmp_path / "submission")

    first = load_submission_bundle(bundle_dir, expected_paper_id="paper1")
    second = load_submission_bundle(bundle_dir, expected_paper_id="paper1")

    assert first.code_sha256 == second.code_sha256
    assert first.submission_id == second.submission_id
    assert first.submission_id.startswith("paper1-coral-gpt-5-3-codex-")
    assert first.manifest["code_sha256"] == first.code_sha256
    assert first.manifest["metadata"]["author"] == "Team Frontier"


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        ("gurobi_solution", "private benchmark marker"),
        ("feasibility_check.py", "private benchmark marker"),
        ("FRONTIER_OR_DATA_DIR", "private benchmark marker"),
        ("OPENROUTER_API_KEY", "private benchmark marker"),
        (".coral/private", "private benchmark marker"),
    ],
)
def test_submission_bundle_rejects_private_leak_markers(tmp_path, marker, message):
    from scripts.utils.submission_bundle import SubmissionBundleError, load_submission_bundle

    bundle_dir = _write_bundle(tmp_path / "submission", code=f"# {marker}\n")

    with pytest.raises(SubmissionBundleError, match=message):
        load_submission_bundle(bundle_dir, expected_paper_id="paper1")


def test_submission_bundle_rejects_malformed_layout(tmp_path):
    from scripts.utils.submission_bundle import SubmissionBundleError, load_submission_bundle

    bundle_dir = tmp_path / "submission"
    bundle_dir.mkdir()
    (bundle_dir / "submission.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SubmissionBundleError, match="missing code.py"):
        load_submission_bundle(bundle_dir, expected_paper_id="paper1")

    _write_bundle(bundle_dir, paper_id="other")
    with pytest.raises(SubmissionBundleError, match="paper_id mismatch"):
        load_submission_bundle(bundle_dir, expected_paper_id="paper1")


def test_submission_bundle_rejects_symlink_code(tmp_path):
    from scripts.utils.submission_bundle import SubmissionBundleError, load_submission_bundle

    bundle_dir = _write_bundle(tmp_path / "submission")
    target = tmp_path / "outside.py"
    target.write_text("print('outside')\n", encoding="utf-8")
    (bundle_dir / "code.py").unlink()
    (bundle_dir / "code.py").symlink_to(target)

    with pytest.raises(SubmissionBundleError, match="symlink"):
        load_submission_bundle(bundle_dir, expected_paper_id="paper1")


def test_submission_bundle_keeps_validated_code_bytes(tmp_path):
    from scripts.utils.submission_bundle import load_submission_bundle

    bundle_dir = _write_bundle(tmp_path / "submission", code="print('first')\n")
    bundle = load_submission_bundle(bundle_dir, expected_paper_id="paper1")
    (bundle_dir / "code.py").write_text("print('changed')\n", encoding="utf-8")

    assert bundle.code_bytes == b"print('first')\n"


def test_submission_bundle_rejects_paper_path_traversal(tmp_path):
    from scripts.utils.submission_bundle import SubmissionBundleError, load_submission_bundle

    bundle_dir = _write_bundle(tmp_path / "submission", paper_id="../../private")

    with pytest.raises(SubmissionBundleError, match="paper_id.*invalid"):
        load_submission_bundle(bundle_dir)
