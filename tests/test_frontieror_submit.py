import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _stub_checker_security_preflight(monkeypatch):
    import frontieror_submit

    monkeypatch.setattr(
        frontieror_submit,
        "validate_objective_checker",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        frontieror_submit,
        "resolve_docker_image",
        lambda _image: "sha256:" + ("a" * 64),
    )


def _write_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "submission.json").write_text(
        json.dumps(
            {
                "author": "Team Frontier",
                "model_or_agent": "gpt-5.3-codex",
                "framework": "coral",
                "paper_id": "paper1",
                "track": "official",
                "created_at": "2026-06-30T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (root / "code.py").write_text("print('candidate')\n", encoding="utf-8")
    return root


def test_frontieror_submit_writes_private_traces_and_redacted_public_row(tmp_path, monkeypatch):
    import frontieror_submit

    bundle = _write_bundle(tmp_path / "submission")
    output_root = tmp_path / "out"
    captured = {}

    def fake_evaluate_candidate_code(
        paper_id,
        model_name,
        instances,
        code_path,
        output_dir,
        time_limit,
        exec_mode,
        exec_cfg,
        t_max,
    ):
        captured.update(
            {
                "paper_id": paper_id,
                "model_name": model_name,
                "instances": list(instances),
                "code_path": code_path,
                "output_dir": output_dir,
                "time_limit": time_limit,
                "exec_mode": exec_mode,
                "exec_cfg": dict(exec_cfg),
                "t_max": t_max,
            }
        )
        return {
            "large_4": {
                "status": "pass",
                "feasible": True,
                "llm_obj": 90.0,
                "solve_time": 5.0,
                "aocc": 0.2,
            }
        }

    monkeypatch.setattr(frontieror_submit.eval_core, "evaluate_candidate_code", fake_evaluate_candidate_code)
    monkeypatch.setattr(
        frontieror_submit.eval_core,
        "load_gurobi_csv_data",
        lambda paper_id, **_kwargs: {
            "large_4": {"solution": 100.0, "time": 10.0}
        },
    )
    monkeypatch.setattr(frontieror_submit.eval_core, "get_paper_direction", lambda paper_id: "min")

    rc = frontieror_submit.main(
        [
            str(bundle),
            "--paper-id",
            "paper1",
            "--final-instances",
            "large_4",
            "--output-root",
            str(output_root),
            "--time-limit",
            "3600",
            "--cpus",
            "1",
            "--memory",
            "1G",
        ]
    )

    assert rc == 0
    assert captured["exec_mode"] == "docker"
    assert captured["exec_cfg"]["anti_hack"] is True
    assert captured["exec_cfg"]["docker_image"] == "sha256:" + ("a" * 64)
    assert captured["exec_cfg"]["docker_image_ref"] == "frontieror-candidate:1"
    assert captured["exec_cfg"]["wls_egress"] == "off"
    assert captured["instances"] == ["large_4"]
    assert Path(captured["code_path"]).read_text(encoding="utf-8") == "print('candidate')\n"

    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    traces = [
        json.loads(line)
        for line in (run_dir / "private" / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert traces[0]["instance"] == "large_4"
    assert traces[0]["score"] == 1.6
    assert traces[0]["score_debug"]["stage_id"] == 2

    verdict = json.loads((run_dir / "private" / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["valid"] is True
    assert verdict["aggregate_score"] == 1.6
    manifest = json.loads(
        (run_dir / "private" / "submission_manifest.json").read_text(encoding="utf-8")
    )
    provenance = manifest["verifier_provenance"]
    assert provenance["docker_image_id"] == "sha256:" + ("a" * 64)
    assert provenance["scorer_sha256"]
    assert provenance["cpu_model"]

    public_row = json.loads((run_dir / "public" / "leaderboard_row.json").read_text(encoding="utf-8"))
    assert public_row["aggregate_score"] == 1.6
    assert public_row["docker_image_id"] == "sha256:" + ("a" * 64)
    public_text = json.dumps(public_row)
    assert "large_4" not in public_text
    assert "100.0" not in public_text
    assert "gurobi" not in public_text.lower()
    assert "private" not in public_text.lower()


def test_frontieror_submit_rejects_empty_final_instances(tmp_path):
    import frontieror_submit

    bundle = _write_bundle(tmp_path / "submission")

    rc = frontieror_submit.main(
        [
            str(bundle),
            "--paper-id",
            "paper1",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 2


def test_frontieror_submit_rejects_malformed_bundle(tmp_path):
    import frontieror_submit

    bundle = tmp_path / "submission"
    bundle.mkdir()

    rc = frontieror_submit.main(
        [
            str(bundle),
            "--paper-id",
            "paper1",
            "--final-instances",
            "large_4",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 2


def test_frontieror_submit_rejects_invalid_final_instance_name(tmp_path):
    import frontieror_submit

    bundle = _write_bundle(tmp_path / "submission")

    rc = frontieror_submit.main(
        [
            str(bundle),
            "--paper-id",
            "paper1",
            "--final-instances",
            "not_an_instance",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 2


def test_frontieror_submit_rejects_untrusted_aocc_log_scoring(tmp_path):
    import frontieror_submit

    bundle = _write_bundle(tmp_path / "submission")

    rc = frontieror_submit.main(
        [
            str(bundle),
            "--paper-id",
            "paper1",
            "--final-instances",
            "large_4",
            "--scorer",
            "aocc",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 2


def test_reference_row_falls_back_to_private_reference_solution(
    tmp_path, monkeypatch
):
    import frontieror_submit

    reference = tmp_path / "tiny_solution.json"
    reference.write_text(
        json.dumps({"objective_value": 17, "wall_time": 2.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        frontieror_submit,
        "gurobi_solution_path",
        lambda _paper_dir, _instance: str(reference),
    )

    row = frontieror_submit._reference_row(
        paper_dir=str(tmp_path),
        instance="tiny",
        csv_row={},
    )

    assert row == {"solution": 17.0, "time": 2.5}


def test_frontieror_submit_refuses_to_overwrite_existing_audit_run(
    tmp_path, monkeypatch
):
    import frontieror_submit
    from scripts.utils.submission_bundle import load_submission_bundle

    bundle = _write_bundle(tmp_path / "submission")
    loaded = load_submission_bundle(bundle, expected_paper_id="paper1")
    output_root = tmp_path / "out"
    (output_root / loaded.submission_id).mkdir(parents=True)
    monkeypatch.setattr(
        frontieror_submit.eval_core,
        "evaluate_candidate_code",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evaluation must not start")
        ),
    )

    rc = frontieror_submit.main(
        [
            str(bundle),
            "--paper-id",
            "paper1",
            "--final-instances",
            "large_4",
            "--output-root",
            str(output_root),
        ]
    )

    assert rc == 2
