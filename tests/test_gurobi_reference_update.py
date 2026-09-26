import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "paper_reproduce" / "run_program_solutions.py"
SPEC = importlib.util.spec_from_file_location("run_program_solutions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reference_upsert_writes_matching_tables_atomically(tmp_path):
    data_dir = tmp_path / "frontier-or"
    tasks_dir = data_dir / "tasks"
    metadata = data_dir / "metadata"
    task = tasks_dir / "paper-a"
    (task / "gurobi_solution").mkdir(parents=True)
    (task / "gurobi_feasi_result").mkdir()
    metadata.mkdir()

    original = pd.DataFrame([
        {
            "task_id": "paper-a", "instance": "tiny", "objective_value": 1.0,
            "runtime": 2.0, "feasible": True, "status": "optimal",
            "time_limit": 300, "solution_path": "tasks/paper-a/gurobi_solution/tiny_solution.json",
        },
        {
            "task_id": "paper-b", "instance": "tiny", "objective_value": 3.0,
            "runtime": 4.0, "feasible": True, "status": "optimal",
            "time_limit": 300, "solution_path": "tasks/paper-b/gurobi_solution/tiny_solution.json",
        },
    ])
    original.to_parquet(metadata / "gurobi_references.parquet", index=False)
    original.to_csv(metadata / "gurobi_references.csv.gz", index=False, compression="gzip")
    (task / "gurobi_solution" / "tiny_solution.json").write_text(
        json.dumps({"objective_value": 10.0, "runtime": 5.0, "status": "optimal"})
    )
    (task / "gurobi_feasi_result" / "tiny_feasi_result.json").write_text(
        json.dumps({"feasible": True})
    )

    count, _, _ = MODULE.update_gurobi_reference_tables(tasks_dir, ["paper-a"], ["tiny"])

    parquet = pd.read_parquet(metadata / "gurobi_references.parquet")
    csv = pd.read_csv(metadata / "gurobi_references.csv.gz")
    assert count == 1
    assert len(parquet) == 2
    assert parquet.set_index(["task_id", "instance"]).loc[("paper-a", "tiny"), "objective_value"] == 10.0
    pd.testing.assert_frame_equal(parquet, csv, check_dtype=False)
