import csv
import json

import pytest

from test_time_self_evolution import eval_modes
from test_time_self_evolution import run_eval_modes
from test_time_self_evolution.scoring import building_blocks


def test_median_dev_set_falls_back_when_tau_metadata_is_absent(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        building_blocks,
        "pick_median_tau_g_instance",
        lambda _paper_id: None,
    )
    monkeypatch.setattr(
        building_blocks,
        "list_large_instances",
        lambda _paper_id: [
            "large_1",
            "large_2",
            "large_3",
            "large_4",
            "large_5",
        ],
    )

    resolved = run_eval_modes._resolve_dev_set_for_paper(
        "paper1",
        [run_eval_modes._DEV_SET_SENTINEL_MEDIAN],
    )

    assert resolved == ["large_3"]
    assert "no Gurobi τ_g metadata" in capsys.readouterr().out


def test_max_dev_set_falls_back_when_tau_metadata_is_absent(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        building_blocks,
        "pick_max_tau_g_instance",
        lambda _paper_id: None,
    )
    monkeypatch.setattr(
        building_blocks,
        "list_large_instances",
        lambda _paper_id: ["large_1", "large_2", "large_3"],
    )

    resolved = run_eval_modes._resolve_dev_set_for_paper(
        "paper1",
        [run_eval_modes._DEV_SET_SENTINEL_MAX],
    )

    assert resolved == ["large_3"]
    assert "no Gurobi τ_g metadata" in capsys.readouterr().out


def test_default_final_set_preserves_preset_and_excludes_dev():
    resolved = run_eval_modes._resolve_test_set_for_paper(
        "paper1",
        run_eval_modes.SELF_EVOLVE_TEST_INSTANCES,
        ["large_3"],
    )

    assert resolved == ["large_2", "large_4", "large_5"]


def test_eoh_rejects_empty_population_budget_before_startup():
    with pytest.raises(SystemExit, match="population count"):
        run_eval_modes.main(
            [
                "--modes",
                "self_evolve",
                "--framework",
                "eoh",
                "--eoh-n-pop",
                "0",
            ]
        )


def test_self_evolve_csv_dedup_preserves_independent_runs(tmp_path):
    csv_path = tmp_path / "results.csv"
    columns = ["run_id", "paper_id", "model", "instance", "status"]
    base = {"paper_id": "paper1", "model": "model1", "instance": "large_1"}

    eval_modes._write_self_evolve_csv_with_dedup(
        str(csv_path), columns, {**base, "run_id": "run-a", "status": "pass"}
    )
    eval_modes._write_self_evolve_csv_with_dedup(
        str(csv_path), columns, {**base, "run_id": "run-b", "status": "fail"}
    )
    eval_modes._write_self_evolve_csv_with_dedup(
        str(csv_path), columns, {**base, "run_id": "run-a", "status": "updated"}
    )

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["run_id"], row["status"]) for row in rows] == [
        ("run-b", "fail"),
        ("run-a", "updated"),
    ]


def test_write_episode_result_is_private_and_atomic(tmp_path, monkeypatch):
    destination = tmp_path / "private" / "episode.json"
    monkeypatch.setenv("FRONTIER_OR_EPISODE_RESULT_PATH", str(destination))

    run_eval_modes._write_episode_result(
        {"schema_version": 1, "result": {"candidate_id": "best"}}
    )

    assert json.loads(destination.read_text())["result"]["candidate_id"] == "best"
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not list(destination.parent.glob("*.tmp"))


def test_shared_final_scorer_populates_staged_qte_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_modes.eval_core, "get_paper_direction", lambda _paper: "min"
    )
    monkeypatch.setattr(
        building_blocks, "lookup_gurobi_time", lambda _paper, _instance: 20.0
    )
    results = {
        "large_1": {
            "feasible": True,
            "llm_obj": 101.0,
            "gurobi_obj": 100.0,
            "solve_time": 5.0,
        }
    }

    scored = eval_modes.augment_results_with_staged_qte(
        results,
        "paper",
        str(tmp_path),
        time_limits={"large_1": 10},
    )

    assert scored["large_1"]["score"] == 1.74
    assert scored["large_1"]["stage_id"] == 2.0
    assert scored["large_1"]["speed_part"] == 0.75
