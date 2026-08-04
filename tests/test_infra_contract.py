from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from frontieror.infra.contracts import public_scoring_contract, visibility_contract
from frontieror.infra.policy import AgentModePolicy, hardened_agent_argv


ROOT = Path(__file__).resolve().parents[1]


def test_scoring_equation_is_public_but_instance_baselines_are_not() -> None:
    contract = public_scoring_contract(stage_boundary=0.01)

    assert contract["scorer"] == "staged_qte"
    assert contract["aggregation"] == "arithmetic_mean_over_declared_instances"
    assert contract["aggregation_details"]["missing_or_failed_instance_score"] == 0.0
    assert contract["runtime_measurement"] == "trusted_host_wall_clock"
    encoded_formula = json.dumps(contract["instance_score"], sort_keys=True)
    assert "signed_gap" in encoded_formula
    assert contract["instance_score"]["scale"]["near_zero_reference"] == 1e-3
    assert contract["instance_score"]["rounding"]["per_instance_decimal_places"] == 6
    assert "max(0,g)<=b" in contract["instance_score"]["quality_and_speed"]
    assert "reference_values" not in contract
    assert "reference_runtime_seconds" not in contract


def test_public_scoring_constants_match_the_scorer_implementation() -> None:
    from test_time_self_evolution.scoring.building_blocks import NEAR_ZERO_REF
    from test_time_self_evolution.scoring.staged_qte import STAGE_BOUNDARY

    contract = public_scoring_contract(stage_boundary=STAGE_BOUNDARY)

    assert contract["instance_score"]["variables"]["b"] == STAGE_BOUNDARY
    assert (
        contract["instance_score"]["scale"]["near_zero_reference"]
        == NEAR_ZERO_REF
    )


def test_visibility_contract_distinguishes_agent_and_solver_runtime() -> None:
    artifacts = visibility_contract()["artifacts"]

    assert artifacts["dev_instance_json"] == "dev_workspace"
    assert artifacts["dev_aggregate_feedback"] == "dev_workspace"
    assert artifacts["dev_per_instance_score"] == "trusted_only"
    assert artifacts["final_instance_json"] == "solver_runtime_only"
    assert artifacts["final_per_instance_score"] == "trusted_only"
    assert artifacts["reference_solution"] == "trusted_only"
    assert artifacts["feasibility_checker"] == "trusted_only"


def test_agent_mode_rejects_dev_final_overlap() -> None:
    with pytest.raises(ValueError, match="dev/final isolation"):
        AgentModePolicy().validate_split(
            stage1_instances=["tiny"],
            dev_instances=["large_1"],
            final_instances=["large_1"],
        )


def test_agent_mode_rejects_duplicate_instance_weighting() -> None:
    with pytest.raises(ValueError, match="final instances must not contain duplicates"):
        AgentModePolicy().validate_split(
            stage1_instances=["tiny"],
            dev_instances=["large_1"],
            final_instances=["large_2", "large_2"],
        )


def test_agent_mode_forces_non_overridable_security_profile() -> None:
    forwarded = hardened_agent_argv(
        ["--paper-id", "paper1", "--coral-agent-count", "2"]
    )

    assert forwarded[-15:] == [
        "--modes",
        "self_evolve",
        "--framework",
        "coral",
        "--exec-mode",
        "docker",
        "--stage2-scorer",
        "staged_qte",
        "--coral-agent-isolation",
        "docker",
        "--coral-model-access",
        "proxy",
        "--coral-agent-image",
        "frontieror-coral-agent:0.1",
        "--anti-hack",
    ]
    with pytest.raises(ValueError, match="fixes --exec-mode=docker"):
        hardened_agent_argv(["--exec-mode", "systemd"])
    with pytest.raises(ValueError, match="legacy CORAL gateway"):
        hardened_agent_argv(["--coral-gateway"])
    with pytest.raises(ValueError, match="fixes --coral-model-access=proxy"):
        hardened_agent_argv(["--coral-model-access", "local-auth"])


def test_agent_cli_exposes_upstream_arguments() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "frontieror.infra", "agent", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "--coral-agent-count" in completed.stdout
    assert "--test-set" in completed.stdout
    assert "--exec-mode" not in completed.stdout
    assert "--anti-hack" not in completed.stdout
    assert "--coral-model-access" not in completed.stdout
    assert "--coral-agent-image" not in completed.stdout
    assert "--resume" not in completed.stdout


def test_agent_cli_rejects_security_downgrade_flags() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "frontieror.infra",
            "agent",
            "--coral-model-access",
            "local-auth",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr


def test_black_box_security_check_uses_official_candidate_path(
    tmp_path, monkeypatch
) -> None:
    from frontieror.infra import security_check

    image_digest = "sha256:" + ("c" * 64)
    monkeypatch.setattr(
        security_check.execution,
        "resolve_docker_image",
        lambda image: image_digest,
    )
    calls = []

    def fake_run(
        code_path,
        instance_path,
        solution_path,
        time_limit,
        *,
        log_path,
        cfg,
    ):
        del instance_path, log_path
        probe = Path(code_path).parent.name
        calls.append((probe, time_limit, dict(cfg)))
        if probe == "boundary":
            Path(solution_path).write_text(
                json.dumps(
                    {
                        "host_file": "blocked",
                        "host_env": "blocked",
                        "root_write": "blocked",
                        "network": "blocked",
                        "objective_value": 0,
                    }
                ),
                encoding="utf-8",
            )
            return True, "", 0.1
        if probe == "timeout":
            return False, "Execution timed out after 0.5 seconds", 0.6
        if probe == "stdout":
            Path(solution_path).write_text(
                '{"objective_value": 0}', encoding="utf-8"
            )
            return True, "[output truncated; 1048576 bytes omitted]", 0.1
        raise AssertionError(f"unexpected probe: {probe}")

    monkeypatch.setattr(security_check.execution, "run_docker", fake_run)

    def fake_checker(**kwargs):
        Path(kwargs["result_file"]).write_text(
            json.dumps(
                {
                    "feasible": True,
                    "host_file": "blocked",
                    "host_env": "blocked",
                    "root_write": "blocked",
                    "network": "blocked",
                }
            ),
            encoding="utf-8",
        )
        return True, "", 0.1

    monkeypatch.setattr(security_check, "run_checker_isolated", fake_checker)

    report = security_check.run_security_checks("frontieror-candidate:test")

    assert report["passed"] is True
    assert report["candidate_image"]["digest"] == image_digest
    assert [probe["id"] for probe in report["probes"]] == [
        "isolation_boundary",
        "checker_boundary",
        "wall_clock_timeout",
        "bounded_output",
    ]
    assert all(cfg["anti_hack"] is True for _, _, cfg in calls)
    assert all(cfg["wls_egress"] == "off" for _, _, cfg in calls)


def test_security_check_cli_propagates_failed_report(monkeypatch, capsys) -> None:
    from frontieror.infra import cli
    from frontieror.infra import security_check

    monkeypatch.setattr(
        security_check,
        "run_security_checks",
        lambda _image: {"schema_version": 1, "passed": False, "probes": []},
    )

    assert cli.main(["security-check"]) == 1
    assert '"passed": false' in capsys.readouterr().out


def test_security_check_cli_reports_setup_error_without_traceback(
    monkeypatch, capsys
) -> None:
    from frontieror.infra import cli
    from frontieror.infra import security_check

    monkeypatch.setattr(
        security_check,
        "run_security_checks",
        lambda _image: (_ for _ in ()).throw(RuntimeError("image unavailable")),
    )

    assert cli.main(["security-check"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert '"error": "image unavailable"' in captured.err
