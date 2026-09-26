import json
from pathlib import Path

import pytest


def configure_paths(monkeypatch, module, tmp_path):
    instance = tmp_path / "paper" / "instance.json"
    reference = tmp_path / "paper" / "reference.json"
    code = tmp_path / "code.py"
    output = tmp_path / "output"
    instance.parent.mkdir(parents=True)
    instance.write_text("{}", encoding="utf-8")
    reference.write_text('{"objective_value": 1}', encoding="utf-8")
    code.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(module, "_instance_path", lambda _paper, _idx: str(instance))
    monkeypatch.setattr(module, "_gurobi_solution_path", lambda _paper, _idx: str(reference))
    monkeypatch.setattr(module, "get_paper_dir", lambda _paper: str(instance.parent))
    monkeypatch.setattr(module, "get_paper_direction", lambda _paper: "min")
    return code, output


@pytest.mark.parametrize(
    ("success", "solution_text", "expected_reason"),
    [
        (True, '{"objective_value": 1e100}', "invalid_obj"),
        (False, "{malformed", "runtime_error"),
    ],
)
def test_early_failure_with_solution_still_runs_checker(
    tmp_path, monkeypatch, success, solution_text, expected_reason
):
    import one_shot_eval

    code, output = configure_paths(monkeypatch, one_shot_eval, tmp_path)
    checker_calls = []

    def fake_run(_code, solution_path, _instance, _limit, _log, **_kwargs):
        Path(solution_path).write_text(solution_text, encoding="utf-8")
        return success, "candidate failed" if not success else "", 0.1

    def fake_checker(_paper, _instance, solution_path, result_path, **_kwargs):
        checker_calls.append(solution_path)
        Path(result_path).write_text('{"feasible": false}', encoding="utf-8")
        return False, None, None

    monkeypatch.setattr(one_shot_eval, "run_generated_code", fake_run)
    monkeypatch.setattr(one_shot_eval, "run_feasibility_check", fake_checker)

    result, _ = one_shot_eval.run_and_evaluate_instance(
        "paper", "model", "tiny", str(code), 1, "systemd", {}, None,
        output_dir=str(output),
    )

    assert result["status"] == "fail"
    assert result["fail_reason"] == expected_reason
    assert result["feasible"] is False
    assert len(checker_calls) == 1
    assert json.loads((output / "feasi_result_tiny.json").read_text()) == {"feasible": False}


def test_early_failure_without_solution_clears_stale_checker_result(
    tmp_path, monkeypatch
):
    import one_shot_eval

    code, output = configure_paths(monkeypatch, one_shot_eval, tmp_path)
    output.mkdir()
    stale = output / "feasi_result_tiny.json"
    stale.write_text('{"feasible": true}', encoding="utf-8")

    monkeypatch.setattr(
        one_shot_eval,
        "run_generated_code",
        lambda *_args, **_kwargs: (True, "", 0.1),
    )

    def unexpected_checker(*_args, **_kwargs):
        raise AssertionError("checker must not run without a solution file")

    monkeypatch.setattr(one_shot_eval, "run_feasibility_check", unexpected_checker)

    result, _ = one_shot_eval.run_and_evaluate_instance(
        "paper", "model", "tiny", str(code), 1, "systemd", {}, None,
        output_dir=str(output),
    )

    assert result["status"] == "fail"
    assert result["fail_reason"] == "invalid_solution"
    assert result["feasible"] is None
    assert not stale.exists()
