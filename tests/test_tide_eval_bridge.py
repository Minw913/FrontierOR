import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontieror.infra import cli as infra_cli
from frontieror.infra import tide_eval_driver, tide_eval_worker


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "paper_id", "model", "instance", "score_staged"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_driver_decodes_typed_worker_response():
    result = tide_eval_driver._decode_worker_response(
        json.dumps(
            {
                "schema_version": 1,
                "result": {
                    "rewards": {"reward": 0.75},
                    "trace": [{"t": 1.0, "score": 0.5, "data": {"generation": 1}}],
                    "usage": {"n_submissions": 1},
                    "uri": "/tmp/run",
                    "error": None,
                },
            }
        ).encode()
    )

    assert result["rewards"] == {"reward": 0.75}
    assert result["trace"][0]["data"] == {"generation": 1}


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b'{"schema_version": 2, "result": {}}',
        b'{"schema_version": 1, "result": {"rewards": {"reward": "1"}}}',
        b'{"schema_version": 1, "result": {"rewards": {"reward": NaN}}}',
        b'{"schema_version": 1, "result": {"rewards": {}, "trace": [{"t": true, "score": 1}]}}',
    ],
)
def test_driver_rejects_malformed_worker_response(payload):
    with pytest.raises(tide_eval_driver.WorkerProtocolError):
        tide_eval_driver._decode_worker_response(payload)


def test_final_reward_is_scoped_by_run(tmp_path):
    _write_csv(
        tmp_path / "eval" / "eval_test_results_eoh.csv",
        [
            {
                "run_id": "run-1",
                "paper_id": "paper",
                "model": "gpt-5.4",
                "instance": "large_1",
                "score_staged": "1.5",
            },
            {
                "run_id": "other-run",
                "paper_id": "paper",
                "model": "gpt-5.4",
                "instance": "large_2",
                "score_staged": "9",
            },
        ],
    )

    rewards, missing = tide_eval_worker._final_reward(
        root=tmp_path,
        framework="eoh",
        run_id="run-1",
        paper_id="paper",
        model_name="gpt-5.4",
        expected_instances=["large_1", "large_2"],
    )

    assert rewards == {"reward": 0.75}
    assert missing == ["large_2"]


def test_eoh_trace_converts_minimized_objective_to_score(tmp_path):
    output = tmp_path / "eoh_run" / "results" / "pops_best"
    output.mkdir(parents=True)
    (output / "population_generation_1.json").write_text(
        json.dumps({"objective": -0.42, "code": "pass"}), encoding="utf-8"
    )

    assert tide_eval_worker._eoh_trace(tmp_path) == [
        {"t": 1.0, "score": 0.42, "data": {"generation": 1}}
    ]


def test_execute_eoh_episode_writes_auditable_manifest(tmp_path, monkeypatch):
    argv = [
        "--modes",
        "self_evolve",
        "--framework",
        "eoh",
        "--paper-id",
        "paper",
        "--primary-model",
        "openai/gpt-5.4",
        "--stage1-instances",
        "tiny",
        "--dev-set",
        "large_1",
        "--test-set",
        "large_2",
        "--run-id",
        "run-1",
        "--exec-mode",
        "docker",
        "--stage2-scorer",
        "staged_qte",
        "--anti-hack",
    ]

    def fake_run(command, **kwargs):
        assert command[:3] == [
            tide_eval_worker.sys.executable,
            "-m",
            "test_time_self_evolution.run_eval_modes",
        ]
        run_dir = tmp_path / "eval" / "eoh" / "run-1" / "paper" / "gpt-5.4"
        pops = run_dir / "eoh_run" / "results" / "pops_best"
        pops.mkdir(parents=True)
        (pops / "population_generation_1.json").write_text(
            json.dumps({"objective": -0.8, "code": "pass"}), encoding="utf-8"
        )
        (run_dir / "selected_code.py").write_text("print('ok')\n")
        _write_csv(
            tmp_path / "eval" / "eval_test_results_eoh.csv",
            [
                {
                    "run_id": "run-1",
                    "paper_id": "paper",
                    "model": "gpt-5.4",
                    "instance": "large_2",
                    "score_staged": "0.9",
                }
            ],
        )
        Path(kwargs["env"]["FRONTIER_OR_EPISODE_RESULT_PATH"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ok",
                    "run_id": "run-1",
                    "paper_id": "paper",
                    "framework": "eoh",
                    "model": "gpt-5.4",
                    "result": {
                        "candidate_id": "eoh_best",
                        "results": {"large_2": {"score": 0.9}},
                    },
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tide_eval_worker.subprocess, "run", fake_run)
    response = tide_eval_worker.execute_episode(
        {
            "schema_version": 1,
            "episode": {
                "task": "frontieror/paper",
                "agent": {
                    "name": "eoh",
                    "entrypoint": "runner",
                    "frontieror_argv": argv,
                },
                "overrides": {},
            },
        },
        root=tmp_path,
    )

    assert response["result"]["rewards"] == {"reward": 0.9}
    assert response["result"]["trace"][0]["score"] == 0.8
    manifest = Path(response["result"]["uri"]) / "tide_episode_result.json"
    payload = json.loads(manifest.read_text())
    assert payload["artifact_sha256"]
    assert payload["orchestrator"] == "tide-eval"


def test_worker_rejects_path_like_identity():
    with pytest.raises(tide_eval_worker.WorkerRequestError, match="invalid paper id"):
        tide_eval_worker.execute_episode(
            {
                "schema_version": 1,
                "episode": {
                    "task": "frontieror/../../private",
                    "agent": {
                        "name": "eoh",
                        "entrypoint": "runner",
                        "frontieror_argv": [
                            "--modes",
                            "self_evolve",
                            "--framework",
                            "eoh",
                            "--paper-id",
                            "../../private",
                            "--primary-model",
                            "gpt-5.4",
                            "--test-set",
                            "large_2",
                            "--run-id",
                            "run-1",
                            "--exec-mode",
                            "docker",
                            "--stage2-scorer",
                            "staged_qte",
                            "--anti-hack",
                        ],
                    },
                    "overrides": {},
                },
            }
        )


def test_worker_rejects_framework_identity_mismatch():
    with pytest.raises(
        tide_eval_worker.WorkerRequestError,
        match="agent name does not match",
    ):
        tide_eval_worker.execute_episode(
            {
                "schema_version": 1,
                "episode": {
                    "task": "frontieror/paper",
                    "agent": {
                        "name": "openevolve",
                        "entrypoint": "runner",
                        "frontieror_argv": [
                            "--modes",
                            "self_evolve",
                            "--framework",
                            "eoh",
                            "--paper-id",
                            "paper",
                            "--primary-model",
                            "gpt-5.4",
                            "--test-set",
                            "large_2",
                            "--run-id",
                            "run-1",
                            "--exec-mode",
                            "docker",
                            "--stage2-scorer",
                            "staged_qte",
                            "--anti-hack",
                        ],
                    },
                    "overrides": {},
                },
            }
        )


def test_tide_eval_cli_builds_eoh_episode(tmp_path, monkeypatch):
    checkout = tmp_path / "tide-eval"
    tide_python = checkout / ".venv" / "bin" / "python"
    tide_python.parent.mkdir(parents=True)
    tide_python.write_text("")
    (checkout / "tide").mkdir()
    (checkout / "tide" / "__init__.py").write_text("")
    (checkout / "pyproject.toml").write_text("[project]\nname='tide-eval'\n")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        observed["request"] = json.loads(Path(command[-1]).read_text())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(infra_cli.subprocess, "run", fake_run)
    rc = infra_cli._run_tide_eval(
        [
            "--framework",
            "eoh",
            "--tide-python",
            str(tide_python),
            "--lab",
            str(tmp_path / "lab"),
            "--",
            "--paper-id",
            "paper",
            "--primary-model",
            "gpt-5.4",
            "--test-set",
            "large_2",
            "--run-id",
            "run-1",
        ]
    )

    assert rc == 0
    call = observed["request"]["calls"][0]
    assert call["agent"]["entrypoint"] == "runner"
    assert call["agent"]["name"] == "eoh"
    assert call["agent"]["frontieror_argv"].count("--anti-hack") == 1
    assert observed["cwd"] == checkout


def test_tide_eval_cli_rejects_path_like_run_id(tmp_path):
    tide_python = tmp_path / "python"
    tide_python.write_text("")
    with pytest.raises(SystemExit):
        infra_cli._run_tide_eval(
            [
                "--framework",
                "coral",
                "--tide-python",
                str(tide_python),
                "--lab",
                str(tmp_path / "lab"),
                "--",
                "--paper-id",
                "paper",
                "--test-set",
                "large_2",
                "--run-id",
                "../../escape",
            ]
        )
