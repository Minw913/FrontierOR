import csv

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_time_self_evolution.scoring import building_blocks


def _write_references(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "paper1", "instance": "tiny",
                    "runtime": 2.5, "time_limit": 300.0,
                },
                {
                    "task_id": "paper1", "instance": "large_1",
                    "runtime": 9.0, "time_limit": 3600.0,
                },
                {
                    "task_id": "bad", "instance": "tiny",
                    "runtime": float("nan"), "time_limit": 300.0,
                },
            ]
        ),
        path,
    )


def test_lookup_gurobi_time_defaults_to_reference_table(tmp_path, monkeypatch):
    path = tmp_path / "metadata" / "gurobi_references.parquet"
    _write_references(path)
    monkeypatch.setenv("FRONTIER_OR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FRONTIER_OR_GUROBI_REFERENCE", raising=False)
    monkeypatch.delenv("FRONTIER_OR_GUROBI_TIME_SOURCE", raising=False)

    assert building_blocks.lookup_gurobi_time("paper1", "tiny") == 2.5
    assert building_blocks.lookup_gurobi_time("paper1", "large_1") == 9.0
    assert building_blocks.lookup_gurobi_time_limit("paper1", "tiny") == 300.0
    assert building_blocks.lookup_gurobi_time_limit("paper1", "large_1") == 3600.0
    assert building_blocks.lookup_gurobi_time("missing", "tiny") is None
    assert building_blocks.lookup_gurobi_time("bad", "tiny") is None


def test_lookup_gurobi_time_does_not_fall_back_to_log(tmp_path, monkeypatch):
    path = tmp_path / "references.parquet"
    _write_references(path)
    monkeypatch.setenv("FRONTIER_OR_GUROBI_REFERENCE", str(path))
    monkeypatch.setattr(
        building_blocks, "gurobi_log_path_for", lambda *_args: "unexpected.jsonl"
    )

    assert building_blocks.lookup_gurobi_time("missing", "large_1") is None


def test_lookup_gurobi_time_supports_explicit_legacy_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "gurobi_results_1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["paper_id", "instance", "gurobi_time"]
        )
        writer.writeheader()
        writer.writerow(
            {"paper_id": "paper1", "instance": "large_1", "gurobi_time": "7.5"}
        )
    monkeypatch.setattr(building_blocks, "ROOT_DIR", str(tmp_path))

    assert (
        building_blocks.lookup_gurobi_time(
            "paper1", "large_1", source="legacy_csv"
        )
        == 7.5
    )


def test_lookup_gurobi_time_rejects_unknown_source():
    with pytest.raises(ValueError, match="Gurobi time source"):
        building_blocks.lookup_gurobi_time("paper1", "tiny", source="log")


def test_shared_qte_time_comparison_caps_budgets_and_allows_small_jitter():
    assert building_blocks.qte_time_is_fast_enough(
        3608.0,
        3605.0,
        candidate_time_limit=3600.0,
        gurobi_time_limit=3600.0,
    )
    assert building_blocks.qte_time_is_fast_enough(
        1000.8,
        1000.0,
        candidate_time_limit=3600.0,
        gurobi_time_limit=3600.0,
    )
    assert not building_blocks.qte_time_is_fast_enough(
        1002.0,
        1000.0,
        candidate_time_limit=3600.0,
        gurobi_time_limit=3600.0,
    )
