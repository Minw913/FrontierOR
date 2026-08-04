import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKER = ROOT_DIR / "trusted_checkers" / "bierwirth2017" / "feasibility_check.py"
ADULYASAK_CHECKER = (
    ROOT_DIR / "trusted_checkers" / "adulyasak2015" / "feasibility_check.py"
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    paper_dir = tmp_path / "bierwirth2017"
    instance_path = paper_dir / "instance" / "tiny_instance.json"
    instance_path.parent.mkdir(parents=True)
    instance_path.write_text(
        json.dumps(
            {
                "num_jobs": 1,
                "num_machines": 1,
                "jobs": [
                    {
                        "job_id": 0,
                        "weight": 2,
                        "release_date": 0,
                        "due_date": 3,
                        "operations": [
                            {"machine": 0, "processing_time": 5},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checker_path = paper_dir / "feasibility_check.py"
    checker_path.write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--instance_path")
parser.add_argument("--solution_path")
parser.add_argument("--result_path")
args = parser.parse_args()
with open(args.solution_path, encoding="utf-8") as handle:
    solution = json.load(handle)
operation = solution["schedule"][0]["operations"][0]
valid = (
    operation["processing_time"] == 5
    and operation["end_time"] == 5
    and solution["schedule"][0]["completion_time"] == 5
)
with open(args.result_path, "w", encoding="utf-8") as handle:
    json.dump({"feasible": valid, "violations": []}, handle)
""",
        encoding="utf-8",
    )
    return paper_dir, instance_path


def _candidate() -> dict:
    return {
        "objective_value": 4,
        "schedule": [
            {
                "job_id": 0,
                "completion_time": 5,
                "tardiness": 2,
                "operations": [
                    {
                        "machine": 0,
                        "start_time": 0,
                    }
                ],
            }
        ],
    }


def _run_checker(
    tmp_path: Path,
    instance_path: Path,
    solution: dict,
) -> dict:
    solution_path = tmp_path / "solution.json"
    result_path = tmp_path / "result.json"
    solution_path.write_text(json.dumps(solution), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--instance_path",
            str(instance_path),
            "--solution_path",
            str(solution_path),
            "--result_path",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(result_path.read_text(encoding="utf-8"))


def test_bierwirth_overlay_accepts_the_public_solution_schema(tmp_path):
    _, instance_path = _fixture(tmp_path)

    result = _run_checker(tmp_path, instance_path, _candidate())

    assert result["feasible"] is True


def test_bierwirth_overlay_rejects_missing_operations(tmp_path):
    _, instance_path = _fixture(tmp_path)
    solution = _candidate()
    solution["schedule"][0]["operations"] = []

    result = _run_checker(tmp_path, instance_path, solution)

    assert result["feasible"] is False
    assert "exactly 1 operations" in result["violations"][0]


def test_bierwirth_overlay_rejects_forged_objective(tmp_path):
    _, instance_path = _fixture(tmp_path)
    solution = _candidate()
    solution["objective_value"] = 0

    result = _run_checker(tmp_path, instance_path, solution)

    assert result["feasible"] is False
    assert "objective_value does not match" in result["violations"][0]


def test_checker_resolver_prefers_versioned_overlay(tmp_path):
    from scripts.utils.checker_paths import feasibility_checker_path

    paper_dir = tmp_path / "bierwirth2017"
    resolved = feasibility_checker_path(paper_dir=str(paper_dir))

    assert Path(resolved) == CHECKER


def _adulyasak_fixture(tmp_path: Path) -> tuple[Path, Path]:
    paper_dir = tmp_path / "adulyasak2015"
    instance_path = paper_dir / "instance" / "tiny_instance.json"
    instance_path.parent.mkdir(parents=True)
    instance_path.write_text(
        json.dumps(
            {
                "n": 1,
                "T": 1,
                "m": 1,
                "Q": 10,
                "C": 10,
                "f": 5,
                "u": 2,
                "h": [1, 1],
                "L": [10, 10],
                "I0": [0, 0],
                "sigma": [100],
                "transportation_costs": [[0, 3], [3, 0]],
                "scenario_probabilities": [1],
                "demand_scenarios": [[[4]]],
                "n_scenarios": 1,
            }
        ),
        encoding="utf-8",
    )
    checker_path = paper_dir / "feasibility_check.py"
    checker_path.write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--instance_path")
parser.add_argument("--solution_path")
parser.add_argument("--result_path")
args = parser.parse_args()
with open(args.result_path, "w", encoding="utf-8") as handle:
    json.dump({"feasible": True, "violations": []}, handle)
""",
        encoding="utf-8",
    )
    return paper_dir, instance_path


def _adulyasak_candidate() -> dict:
    return {
        "objective_value": 19,
        "y": {"1": 1},
        "z": {"0_1_1": 1, "1_1_1": 1},
        "x": {"0_1_1_1": 2},
    }


def _run_adulyasak_checker(
    tmp_path: Path,
    instance_path: Path,
    solution: dict,
) -> dict:
    solution_path = tmp_path / "adulyasak_solution.json"
    result_path = tmp_path / "adulyasak_result.json"
    solution_path.write_text(json.dumps(solution), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ADULYASAK_CHECKER),
            "--instance_path",
            str(instance_path),
            "--solution_path",
            str(solution_path),
            "--result_path",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(result_path.read_text(encoding="utf-8"))


def test_adulyasak_overlay_accepts_certified_recourse_objective(tmp_path):
    _, instance_path = _adulyasak_fixture(tmp_path)

    result = _run_adulyasak_checker(
        tmp_path,
        instance_path,
        _adulyasak_candidate(),
    )

    assert result["feasible"] is True


def test_adulyasak_overlay_rejects_forged_recourse_objective(tmp_path):
    _, instance_path = _adulyasak_fixture(tmp_path)
    solution = _adulyasak_candidate()
    solution["objective_value"] = 18

    result = _run_adulyasak_checker(tmp_path, instance_path, solution)

    assert result["feasible"] is False
    assert "fixed-decision recourse solve" in result["violations"][0]


def test_adulyasak_checker_resolver_prefers_versioned_overlay(tmp_path):
    from scripts.utils.checker_paths import feasibility_checker_path

    paper_dir = tmp_path / "adulyasak2015"
    resolved = feasibility_checker_path(paper_dir=str(paper_dir))

    assert Path(resolved) == ADULYASAK_CHECKER
