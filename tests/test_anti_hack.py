import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from scripts.utils.exec_backends import (
    _exec,
    _canonical_license_path,
    _restricted_wls_egress,
    _wls_execution_slot,
    _wls_license_fields,
    build_docker_cmd,
    validate_docker_wls,
)
from scripts.utils.restricted_egress_proxy import host_allowed
from test_time_self_evolution.coral import runner as coral_runner


def _write(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _paper_fixture(tmp_path: Path) -> Path:
    paper = tmp_path / "frontier-or" / "paper1"
    _write(paper / "problem_description.txt", "public problem")
    _write(paper / "instance_schema.json", "{}")
    _write(paper / "solution_schema.json", "{}")
    _write(paper / "feasibility_check.py", "raise SystemExit('private')\n")
    _write(paper / "instance" / "tiny_instance.json", '{"n": 1}')
    _write(paper / "instance" / "large_instance_1.json", '{"n": 2}')
    _write(paper / "gurobi_solution" / "tiny_solution.json", '{"objective_value": 1}')
    _write(paper / "gurobi_solution_log" / "tiny_log.jsonl", "{}\n")
    return paper


def test_gurobi_license_resolution_uses_portable_home(
    tmp_path, monkeypatch
):
    import one_shot_eval

    home = tmp_path / "home"
    license_path = _write(home / "gurobi.lic", "test-license")
    monkeypatch.delenv("GRB_LICENSE_FILE", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(one_shot_eval, "ROOT_DIR", str(tmp_path / "repo"))

    assert one_shot_eval.configure_gurobi_license() == str(license_path)
    assert os.environ["GRB_LICENSE_FILE"] == str(license_path)


def test_wls_license_detection_rejects_incomplete_and_linked_files(tmp_path):
    complete = _write(
        tmp_path / "complete.lic",
        "WLSACCESSID=access\nWLSSECRET=secret\nLICENSEID=123\n",
    )
    incomplete = _write(tmp_path / "incomplete.lic", "LICENSEID=123\n")
    linked = tmp_path / "linked.lic"
    linked.symlink_to(complete)

    assert set(_wls_license_fields(str(complete))) == {
        "WLSACCESSID",
        "WLSSECRET",
        "LICENSEID",
    }
    assert _wls_license_fields(str(incomplete)) == {}
    assert _wls_license_fields(str(linked)) == {}
    assert _canonical_license_path(str(linked)) == str(complete)


def test_wls_proxy_allowlist_requires_exact_token_host():
    allowed = {"token.gurobi.com"}

    assert host_allowed("TOKEN.GUROBI.COM.", allowed)
    assert not host_allowed("license.gurobi.com", allowed)
    assert not host_allowed("token.gurobi.com.attacker.invalid", allowed)
    assert not host_allowed("gurobi.com", allowed)


def test_restricted_wls_sidecar_lifecycle_does_not_pass_credentials(
    tmp_path, monkeypatch
):
    from scripts.utils import exec_backends

    license_path = _write(
        tmp_path / "gurobi.lic",
        "WLSACCESSID=access-value\n"
        "WLSSECRET=secret-value\n"
        "LICENSEID=123\n",
    )
    calls = []

    def fake_docker_control(args, *, timeout=30):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            ["docker", *args],
            0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(exec_backends, "_docker_control", fake_docker_control)
    with _restricted_wls_egress(
        {
            "gurobi_lic": str(license_path),
            "docker_image": "sha256:" + ("a" * 64),
            "wls_egress": "required",
        }
    ) as egress:
        assert egress["network"].startswith("frontieror-wls-")
        assert egress["proxy_url"] == "http://frontieror-wls-egress:3128"

    flattened = " ".join(part for call in calls for part in call)
    assert "token.gurobi.com" in flattened
    assert "access-value" not in flattened
    assert "secret-value" not in flattened
    assert "--cpus 0.25" in flattened
    assert "--ulimit nofile=128:128" in flattened
    assert any(call[:2] == ["network", "create"] for call in calls)
    assert any(call[:2] == ["rm", "-f"] for call in calls)
    assert any(call[:2] == ["network", "rm"] for call in calls)


def test_wls_preflight_modes_fail_closed_before_candidate_execution(
    tmp_path, monkeypatch
):
    from scripts.utils import exec_backends

    monkeypatch.setattr(
        exec_backends,
        "run_docker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("off mode must not execute a probe")
        ),
    )
    validate_docker_wls({"wls_egress": "off"})

    with pytest.raises(RuntimeError, match="no complete WLS license"):
        validate_docker_wls(
            {
                "wls_egress": "required",
                "gurobi_lic": str(tmp_path / "missing.lic"),
            }
        )


def test_wls_preflight_uses_restricted_runner(tmp_path, monkeypatch):
    from scripts.utils import exec_backends

    license_path = _write(
        tmp_path / "gurobi.lic",
        "WLSACCESSID=access\nWLSSECRET=secret\nLICENSEID=123\n",
    )
    captured = {}

    def fake_run(
        code_path,
        instance_path,
        solution_path,
        timeout,
        *,
        log_path,
        cfg,
    ):
        captured.update(
            {
                "code_path": code_path,
                "instance_path": instance_path,
                "solution_path": solution_path,
                "timeout": timeout,
                "log_path": log_path,
                "cfg": dict(cfg),
            }
        )
        return True, "FRONTIEROR_WLS_READY", 0.1

    monkeypatch.setattr(exec_backends, "run_docker", fake_run)
    validate_docker_wls(
        {
            "wls_egress": "auto",
            "gurobi_lic": str(license_path),
            "anti_hack": True,
        }
    )

    assert captured["cfg"]["wls_egress"] == "required"
    assert captured["cfg"]["anti_hack"] is True
    assert captured["cfg"]["gurobi_lic"] == str(license_path)


def test_wls_execution_slots_serialize_candidate_processes(tmp_path, monkeypatch):
    license_path = _write(
        tmp_path / "gurobi.lic",
        "WLSACCESSID=access\nWLSSECRET=secret\nLICENSEID=123\n",
    )
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    config = {
        "gurobi_lic": str(license_path),
        "wls_egress": "required",
        "wls_concurrency": 1,
    }
    acquired = threading.Event()

    def wait_for_slot():
        with _wls_execution_slot(config):
            acquired.set()

    with _wls_execution_slot(config):
        waiter = threading.Thread(target=wait_for_slot)
        waiter.start()
        time.sleep(0.15)
        assert acquired.is_set() is False
    waiter.join(timeout=2)
    assert acquired.is_set() is True


def test_wls_execution_slots_are_disabled_by_default(tmp_path, monkeypatch):
    license_path = _write(
        tmp_path / "gurobi.lic",
        "WLSACCESSID=access\nWLSSECRET=secret\nLICENSEID=123\n",
    )
    monkeypatch.delenv("FRONTIER_OR_WLS_CONCURRENCY", raising=False)

    with _wls_execution_slot(
        {
            "gurobi_lic": str(license_path),
            "wls_egress": "required",
        }
    ) as slot:
        assert slot is None


def test_public_view_excludes_private_grader_and_reference_files(tmp_path):
    from scripts.utils.anti_hack import materialize_public_paper_view

    paper = _paper_fixture(tmp_path)
    public = tmp_path / "public"

    view = materialize_public_paper_view(
        paper_dir=str(paper),
        public_root=str(public),
        paper_id="paper1",
        instances=["tiny", "large_1"],
    )

    view_root = Path(view.public_paper_dir)
    assert (view_root / "problem_description.txt").read_text(encoding="utf-8") == "public problem"
    assert (view_root / "instance" / "tiny_instance.json").exists()
    assert (view_root / "instance" / "large_instance_1.json").exists()
    assert not (view_root / "feasibility_check.py").exists()
    assert not (view_root / "gurobi_solution").exists()
    assert not (view_root / "gurobi_solution_log").exists()

    manifest = json.loads((view_root / "anti_hack_manifest.json").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest)
    assert manifest["schema_version"] == 1
    contract = json.loads(
        (view_root / "benchmark_contract.json").read_text(encoding="utf-8")
    )
    assert contract["scoring"]["scorer"] == "staged_qte"
    assert contract["visibility"]["artifacts"]["dev_instance_json"] == "dev_workspace"
    assert contract["visibility"]["artifacts"]["final_instance_json"] == "solver_runtime_only"
    assert "reference_values" not in contract["scoring"]
    assert "reference_runtime_seconds" not in contract["scoring"]
    assert set(manifest["sha256"]) == set(manifest["copied"])
    assert str(paper) not in serialized
    assert str(public) not in serialized
    assert "withheld" not in manifest
    assert "feasibility_check" not in serialized
    assert "gurobi_solution" not in serialized


def test_public_view_contains_no_final_instance_name_or_canary(tmp_path):
    from frontieror.infra.visibility import materialize_public_paper_view

    paper = _paper_fixture(tmp_path)
    final_canary = "FRONTIEROR_FINAL_CANARY_7f33b4"
    _write(
        paper / "instance" / "large_instance_2.json",
        json.dumps({"private": final_canary}),
    )

    view = materialize_public_paper_view(
        paper_dir=str(paper),
        public_root=str(tmp_path / "public"),
        paper_id="paper1",
        instances=["tiny", "large_1"],
    )

    public_root = Path(view.public_paper_dir)
    assert not (public_root / "instance" / "large_instance_2.json").exists()
    public_bytes = b"\n".join(
        path.read_bytes() for path in public_root.rglob("*") if path.is_file()
    )
    assert b"large_2" not in public_bytes
    assert final_canary.encode() not in public_bytes


def test_public_view_rejects_symlinked_source_files(tmp_path):
    from scripts.utils.anti_hack import materialize_public_paper_view
    from scripts.utils.secure_files import SecureFileError

    paper = _paper_fixture(tmp_path)
    secret = _write(tmp_path / "host-secret.txt", "secret")
    (paper / "problem_description.txt").unlink()
    (paper / "problem_description.txt").symlink_to(secret)

    with pytest.raises(SecureFileError, match="symlink"):
        materialize_public_paper_view(
            paper_dir=str(paper),
            public_root=str(tmp_path / "public"),
            paper_id="paper1",
            instances=["tiny"],
        )


def test_public_view_fails_closed_on_missing_declared_instance(tmp_path):
    from frontieror.infra.visibility import materialize_public_paper_view

    paper = _paper_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="large_instance_2.json"):
        materialize_public_paper_view(
            paper_dir=str(paper),
            public_root=str(tmp_path / "public"),
            paper_id="paper1",
            instances=["large_2"],
        )


def test_agent_split_rejects_renamed_duplicate_instance_json(tmp_path):
    from frontieror.infra.visibility import validate_instance_content_split

    paper = _paper_fixture(tmp_path)
    duplicate = paper / "instance" / "large_instance_2.json"
    duplicate.write_bytes((paper / "instance" / "large_instance_1.json").read_bytes())

    with pytest.raises(ValueError, match="content-disjoint dev/final"):
        validate_instance_content_split(
            paper_dir=str(paper),
            dev_instances=["large_1"],
            final_instances=["large_2"],
        )


def test_reused_coral_seed_is_nofollow_and_has_relative_provenance(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(coral_runner, "ROOT_DIR", str(tmp_path))
    source = _write(
        tmp_path / "eval" / "eval_papers" / "paper1" / "model1" / "code_attempt0.py",
        "print('seed')\n",
    )
    seed_dir = tmp_path / "seed"

    reused = coral_runner._try_reuse_oneshot_seed(
        "paper1",
        "model1",
        str(seed_dir),
    )

    assert reused == str(seed_dir / "code.py")
    assert (seed_dir / "code.py").read_text(encoding="utf-8") == "print('seed')\n"
    provenance = (seed_dir / "_seed_source.txt").read_text(encoding="utf-8")
    assert str(tmp_path) not in provenance

    source.unlink()
    source.symlink_to(tmp_path / "host-secret.py")
    _write(tmp_path / "host-secret.py", "print('secret')\n")
    assert coral_runner._try_reuse_oneshot_seed(
        "paper1",
        "model1",
        str(tmp_path / "other-seed"),
    ) is None


def test_reused_coral_seed_records_seed_model_override(tmp_path, monkeypatch):
    monkeypatch.setattr(coral_runner, "ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("FRONTIER_OR_CORAL_SEED_MODEL", "seed-model")
    _write(
        tmp_path
        / "eval"
        / "eval_papers"
        / "paper1"
        / "seed-model"
        / "code_attempt0.py",
        "print('seed')\n",
    )
    seed_dir = tmp_path / "seed"

    reused = coral_runner._try_reuse_oneshot_seed(
        "paper1",
        "agent-model",
        str(seed_dir),
    )

    assert reused == str(seed_dir / "code.py")
    provenance = (seed_dir / "_seed_source.txt").read_text(encoding="utf-8")
    assert "seed_model: seed-model" in provenance
    assert "agent_model: agent-model" in provenance


def test_anti_hack_rejects_non_docker_and_empty_final_test():
    from scripts.utils.anti_hack import validate_anti_hack_runtime

    with pytest.raises(ValueError, match="requires --exec-mode docker"):
        validate_anti_hack_runtime(enabled=True, exec_mode="systemd", final_test_instances=["large_2"])

    with pytest.raises(ValueError, match="requires a non-empty final test set"):
        validate_anti_hack_runtime(enabled=True, exec_mode="docker", final_test_instances=[])

    with pytest.raises(ValueError, match="does not accept.*AOCC"):
        validate_anti_hack_runtime(
            enabled=True,
            exec_mode="docker",
            final_test_instances=["large_2"],
            scorer="aocc",
        )

    validate_anti_hack_runtime(enabled=True, exec_mode="docker", final_test_instances=["large_2"])


def test_anti_hack_exec_config_pins_the_candidate_image(monkeypatch):
    from frontieror.infra import policy
    from scripts.utils import exec_backends

    policy._resolve_image.cache_clear()
    monkeypatch.setattr(
        exec_backends,
        "resolve_docker_image",
        lambda image: "sha256:" + ("b" * 64),
    )

    config = policy.with_anti_hack_exec_cfg(
        {"docker_image": "frontier-or:test", "cpus": 1},
        True,
    )

    assert config["docker_image_ref"] == "frontier-or:test"
    assert config["docker_image"] == "sha256:" + ("b" * 64)
    assert "wls_concurrency" not in config


def test_checker_security_preflight_rejects_objective_blind_checker(tmp_path):
    from scripts.utils.anti_hack import validate_objective_checker

    paper = tmp_path / "paper"
    _write(paper / "instance" / "tiny_instance.json", '{"expected": 7}')
    _write(paper / "gurobi_solution" / "tiny_solution.json", '{"objective_value": 7}')
    _write(
        paper / "feasibility_check.py",
        """
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--instance_path")
p.add_argument("--solution_path")
p.add_argument("--result_path")
a = p.parse_args()
with open(a.result_path, "w") as f:
    json.dump({"feasible": True}, f)
""",
    )

    with pytest.raises(ValueError, match="does not reject a forged objective_value"):
        validate_objective_checker(paper_dir=str(paper), instance="tiny")


def test_checker_security_preflight_accepts_objective_consistency_check(tmp_path):
    from scripts.utils.anti_hack import validate_objective_checker

    paper = tmp_path / "paper"
    _write(paper / "instance" / "tiny_instance.json", '{"expected": 7}')
    _write(paper / "gurobi_solution" / "tiny_solution.json", '{"objective_value": 7}')
    _write(
        paper / "feasibility_check.py",
        """
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--instance_path")
p.add_argument("--solution_path")
p.add_argument("--result_path")
a = p.parse_args()
with open(a.instance_path) as f:
    instance = json.load(f)
with open(a.solution_path) as f:
    solution = json.load(f)
with open(a.result_path, "w") as f:
    json.dump({"feasible": solution.get("objective_value") == instance["expected"]}, f)
""",
    )

    validate_objective_checker(paper_dir=str(paper), instance="tiny")


def test_hardened_docker_cmd_uses_clean_security_boundary(tmp_path, monkeypatch):
    code_dir = tmp_path / "code"
    code = _write(code_dir / "code.py", "print('ok')\n")
    instance = _write(tmp_path / "public" / "instance.json", "{}")
    solution = tmp_path / "out" / "solution.json"
    log = tmp_path / "logs" / "log.jsonl"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("FRONTIER_OR_DATA_DIR", "/private/data")

    cmd = build_docker_cmd(
        str(code),
        str(instance),
        str(solution),
        10,
        log_path=str(log),
        cfg={"anti_hack": True, "cpus": 1, "memory": "1G"},
    )

    flat = " ".join(cmd)
    assert f"--user={os.getuid()}:{os.getgid()}" in cmd
    assert "--name" in cmd
    assert any(part.startswith("frontieror-candidate-") for part in cmd)
    assert "--log-driver=none" in cmd
    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt" in cmd
    assert "no-new-privileges" in cmd
    assert "--tmpfs" in cmd
    assert "OPENROUTER_API_KEY" not in flat
    assert "FRONTIER_OR_DATA_DIR" not in flat
    assert str(tmp_path / "frontier-or") not in flat
    assert f"{code}:/workspace/code.py:ro" in flat
    assert "/workspace/codedir" not in flat
    assert "type=bind,src=" + str(solution) in flat
    assert "/workspace/output:rw,nosuid,nodev,size=64m" in flat
    assert f"{solution.parent}:/workspace/output" not in flat
    assert "--ulimit" in cmd


def test_hardened_docker_cmd_uses_only_restricted_wls_proxy(tmp_path):
    code = _write(tmp_path / "code.py", "print('ok')\n")
    instance = _write(tmp_path / "instance.json", "{}")
    solution = tmp_path / "solution.json"
    log = tmp_path / "log.jsonl"

    cmd = build_docker_cmd(
        str(code),
        str(instance),
        str(solution),
        10,
        log_path=str(log),
        cfg={
            "anti_hack": True,
            "_restricted_network": "frontieror-wls-test",
            "_restricted_proxy": "http://frontieror-wls-egress:3128",
        },
    )

    flat = " ".join(cmd)
    assert "--network=none" not in cmd
    assert "--network frontieror-wls-test" in flat
    assert "HTTPS_PROXY=http://frontieror-wls-egress:3128" in flat
    assert "HTTP_PROXY=http://frontieror-wls-egress:3128" in flat
    assert "token.gurobi.com" not in flat
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd


def test_hardened_checker_cmd_mounts_only_trusted_inputs(tmp_path):
    from frontieror.infra.checkers import build_isolated_checker_cmd

    paper = _paper_fixture(tmp_path)
    checker = paper / "feasibility_check.py"
    instance = paper / "instance" / "tiny_instance.json"
    solution = _write(tmp_path / "candidate" / "solution.json", "{}")
    result = _write(tmp_path / "checker" / "result.json", "")

    command = build_isolated_checker_cmd(
        checker_path=str(checker),
        paper_dir=str(paper),
        instance_file=str(instance),
        solution_file=str(solution),
        result_file=str(result),
        cfg={"docker_image": "sha256:" + ("d" * 64)},
    )

    flat = " ".join(command)
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "no-new-privileges" in command
    assert f"src={paper},dst=/workspace/paper,readonly" in flat
    assert f"src={solution},dst=/workspace/solution.json,readonly" in flat
    assert f"src={result},dst=/workspace/result.json" in flat
    assert "/workspace/checker.py" in command


def test_anti_hack_feasibility_check_uses_isolated_checker(tmp_path, monkeypatch):
    import one_shot_eval

    paper = _paper_fixture(tmp_path)
    solution = _write(tmp_path / "solution.json", '{"objective_value": 1}')
    result = tmp_path / "result.json"
    captured = {}

    monkeypatch.setattr(one_shot_eval, "get_paper_dir", lambda _paper: str(paper))
    monkeypatch.setattr(
        one_shot_eval,
        "run_bounded_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("host checker must not run")
        ),
    )

    def fake_isolated(**kwargs):
        captured.update(kwargs)
        Path(kwargs["result_file"]).write_text(
            '{"feasible": true}', encoding="utf-8"
        )
        return True, "", 0.1

    monkeypatch.setattr(one_shot_eval, "run_checker_isolated", fake_isolated)

    feasible, reason, error = one_shot_eval.run_feasibility_check(
        "paper1",
        str(paper / "instance" / "tiny_instance.json"),
        str(solution),
        str(result),
        exec_cfg={"anti_hack": True, "docker_image": "image"},
    )

    assert (feasible, reason, error) == (True, None, None)
    assert captured["paper_dir"] == str(paper)
    assert captured["cfg"]["anti_hack"] is True


def test_wls_off_does_not_mount_platform_credentials(tmp_path, monkeypatch):
    from scripts.utils import exec_backends

    license_path = _write(
        tmp_path / "gurobi.lic",
        "WLSACCESSID=access\nWLSSECRET=secret\nLICENSEID=123\n",
    )
    code = _write(tmp_path / "code.py", "print('ok')\n")
    instance = _write(tmp_path / "instance.json", "{}")
    solution = _write(tmp_path / "solution.json", "")
    captured = {}

    def fake_build(*args, **kwargs):
        captured["cfg"] = dict(args[-1])
        return [
            "docker",
            "run",
            "--name",
            "frontieror-candidate-test",
            "image",
        ]

    monkeypatch.setattr(exec_backends, "build_docker_cmd", fake_build)
    monkeypatch.setattr(
        exec_backends,
        "_exec",
        lambda *_args, **_kwargs: (True, "ok", 0.1),
    )
    success, _, _ = exec_backends.run_docker(
        str(code),
        str(instance),
        str(solution),
        1,
        cfg={
            "gurobi_lic": str(license_path),
            "wls_egress": "off",
        },
    )

    assert success
    assert captured["cfg"]["gurobi_lic"] == ""


def test_candidate_solution_symlink_is_rejected_before_hidden_grader_reads_it(
    tmp_path, monkeypatch
):
    import one_shot_eval

    instance = _write(tmp_path / "private" / "instance.json", "{}")
    reference = _write(
        tmp_path / "private" / "reference.json",
        '{"objective_value": 1}',
    )
    hidden = _write(
        tmp_path / "private" / "hidden.json",
        '{"objective_value": 1}',
    )
    code = _write(tmp_path / "submission" / "code.py", "print('candidate')\n")
    output = tmp_path / "output"

    monkeypatch.setattr(one_shot_eval, "_instance_path", lambda _paper, _idx: str(instance))
    monkeypatch.setattr(
        one_shot_eval,
        "_gurobi_solution_path",
        lambda _paper, _idx: str(reference),
    )
    monkeypatch.setattr(one_shot_eval, "get_paper_dir", lambda _paper: str(tmp_path))
    monkeypatch.setattr(one_shot_eval, "get_paper_direction", lambda _paper: "min")

    def fake_run(
        _code_path,
        solution_path,
        _instance_path,
        _time_limit,
        _log_path,
        **_kwargs,
    ):
        Path(solution_path).unlink()
        Path(solution_path).symlink_to(hidden)
        return True, "", 0.01

    monkeypatch.setattr(one_shot_eval, "run_generated_code", fake_run)

    result, _ = one_shot_eval.run_and_evaluate_instance(
        "paper1",
        "model",
        "tiny",
        str(code),
        1,
        "docker",
        {"anti_hack": True},
        None,
        output_dir=str(output),
    )

    assert result["status"] == "fail"
    assert result["fail_reason"] == "invalid_solution"
    assert "symlink" in result["error"]
    assert not (output / "solution_tiny.json").exists()


def test_subprocess_capture_is_bounded():
    success, output, _ = _exec(
        [sys.executable, "-c", "print('x' * (2 * 1024 * 1024))"],
        time_limit=5,
    )

    assert success is True
    assert len(output.encode("utf-8")) < 1100 * 1024
    assert "output truncated" in output


def test_subprocess_time_limit_has_no_compute_grace_period():
    success, output, elapsed = _exec(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        time_limit=0.2,
    )

    assert success is False
    assert "timed out" in output
    assert elapsed < 2


def test_coral_anti_hack_task_uses_public_seed_and_no_private_path_tips(
    tmp_path, monkeypatch
):
    paper = _paper_fixture(tmp_path)
    monkeypatch.setattr(coral_runner.eval_core, "get_paper_dir", lambda _paper: str(paper))
    task = coral_runner.write_coral_task(
        base_dir=str(tmp_path),
        paper_id="paper1",
        prompt="public prompt",
        model_name="model",
        primary_model="openai/model",
        stage1_instances=["tiny"],
        stage2_instances=["large_1"],
        stage1_time_limit=5,
        stage2_time_limit=5,
        stage1_gap_threshold=0.1,
        exec_mode="docker",
        exec_cfg={"anti_hack": True},
        t_max=None,
        stage2_scorer="staged_qte",
        agent_runtime="codex",
        agent_count=1,
        agent_model=None,
        max_turns=1,
        max_steps=20,
        gateway_enabled=False,
        anti_hack=True,
    )

    task_yaml = Path(task.config_path).read_text(encoding="utf-8")
    config = yaml.safe_load(task_yaml)
    seed = Path(config["workspace"]["repo_path"])
    assert seed.name == "public_seed"
    assert (seed / "README.md").exists()
    assert not (seed / "eval" / "grader.py").exists()
    assert "gurobi_solution" not in task_yaml
    assert "frontier-or/<paper>" not in task_yaml
    assert ".coral/private" not in task_yaml
    assert config["agents"]["runtime_options"]["web_search"] == "disabled"
    assert config["agents"]["max_turns"] == 20
    assert config["grader"]["args"]["agent_native_max_steps"] == 20


def test_cli_rejects_anti_hack_without_docker():
    proc = subprocess.run(
        [
            sys.executable,
            "one_shot_eval.py",
            "--paper_id",
            "paper1",
            "--anti-hack",
            "--exec-mode",
            "systemd",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    assert "anti-hack mode requires --exec-mode docker" in proc.stderr


def test_coral_runner_rejects_empty_final_test_in_anti_hack():
    with pytest.raises(ValueError, match="requires a non-empty final test set"):
        coral_runner.run_self_evolve(
            run_id="r",
            paper_id="paper1",
            primary_model="openai/model",
            prompt="prompt",
            config={},
            stage1_instances=["tiny"],
            stage2_instances=["large_1"],
            test_instances=[],
            stage1_time_limit=1,
            stage2_time_limit=1,
            test_time_limit=1,
            stage1_gap_threshold=0.1,
            exec_mode="docker",
            exec_cfg={"anti_hack": True},
            t_max=None,
            attempts=1,
            max_seconds=1,
            anti_hack=True,
        )


def test_coral_runner_rejects_public_and_final_instance_overlap():
    with pytest.raises(ValueError, match="requires dev/final isolation"):
        coral_runner.run_self_evolve(
            run_id="r",
            paper_id="paper1",
            primary_model="openai/model",
            prompt="prompt",
            config={},
            stage1_instances=["tiny"],
            stage2_instances=["large_1"],
            test_instances=["large_1"],
            stage1_time_limit=1,
            stage2_time_limit=1,
            test_time_limit=1,
            stage1_gap_threshold=0.1,
            exec_mode="docker",
            exec_cfg={"anti_hack": True},
            t_max=None,
            attempts=1,
            max_seconds=1,
            agent_count=2,
            anti_hack=True,
        )


def test_coral_best_attempt_requires_finalized_broker_score(tmp_path):
    attempts = tmp_path / ".coral" / "public" / "attempts"
    _write(
        attempts / ("a" * 40 + ".json"),
        json.dumps(
            {
                "commit_hash": "a" * 40,
                "agent_id": "agent-1",
                "status": "pending",
                "score": 100,
            }
        ),
    )
    _write(
        attempts / ("b" * 40 + ".json"),
        json.dumps(
            {
                "commit_hash": "b" * 40,
                "agent_id": "agent-2",
                "status": "completed",
                "score": 2,
            }
        ),
    )

    best = coral_runner.read_best_attempt(str(tmp_path / ".coral"))

    assert best is not None
    assert best["commit_hash"] == "b" * 40
    assert best["agent_id"] == "agent-2"


def test_coral_local_model_preflight_fails_before_agent_launch(monkeypatch):
    monkeypatch.setattr(
        coral_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=json.dumps(
                {
                    "type": "error",
                    "message": "model is not supported by this account",
                }
            ),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="not supported by this account"):
        coral_runner.validate_local_codex_model("gpt-5.3-codex")


def test_coral_local_model_preflight_accepts_exact_requested_model(monkeypatch):
    monkeypatch.setattr(
        coral_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "FRONTIEROR_MODEL_READY",
                        },
                    }
                )
                + "\n"
            ),
            stderr="",
        ),
    )

    coral_runner.validate_local_codex_model("gpt-5.3-codex")


def test_coral_openrouter_preflight_accepts_exact_requested_model(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps(
                {
                    "id": "generation-1",
                    "model": "openai/gpt-5.3-codex",
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(coral_runner.urllib.request, "urlopen", fake_urlopen)

    coral_runner.validate_openrouter_model(
        "openai/gpt-5.3-codex",
        "sk-test-only",
        timeout=12,
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["body"]["model"] == "openai/gpt-5.3-codex"
    assert captured["authorization"] == "Bearer sk-test-only"
    assert captured["timeout"] == 12


def test_coral_openrouter_preflight_rejects_invalid_credential(monkeypatch):
    class FakeResponse:
        status = 401

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps({"error": {"message": "User not found."}}).encode()

    monkeypatch.setattr(
        coral_runner.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match=r"HTTP 401.*User not found"):
        coral_runner.validate_openrouter_model(
            "openai/gpt-5.3-codex",
            "sk-invalid",
        )


def test_coral_anti_hack_rejects_legacy_gateway():
    with pytest.raises(ValueError, match="credential-isolating platform model proxy"):
        coral_runner.run_self_evolve(
            run_id="r",
            paper_id="paper1",
            primary_model="openai/gpt-5.3-codex",
            prompt="prompt",
            config={"OPENROUTER_API_KEY": "sk-test"},
            stage1_instances=["tiny"],
            stage2_instances=["large_1"],
            test_instances=["large_2"],
            stage1_time_limit=1,
            stage2_time_limit=1,
            test_time_limit=1,
            stage1_gap_threshold=0.1,
            exec_mode="docker",
            exec_cfg={"anti_hack": True},
            t_max=None,
            attempts=1,
            max_seconds=1,
            anti_hack=True,
            model_access="proxy",
            gateway_enabled=True,
        )


def test_official_submit_has_no_non_docker_escape_hatch(tmp_path):
    import frontieror_submit

    bundle = tmp_path / "submission"
    bundle.mkdir()
    _write(
        bundle / "submission.json",
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
    )
    _write(bundle / "code.py", "print('ok')\n")

    with pytest.raises(SystemExit) as exc:
        frontieror_submit.main(
            [
                str(bundle),
                "--paper-id",
                "paper1",
                "--final-instances",
                "large_1",
                "--exec-mode",
                "systemd",
            ]
        )

    assert exc.value.code == 2
