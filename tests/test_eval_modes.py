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
