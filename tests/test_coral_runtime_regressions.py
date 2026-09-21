import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from trusted_eval_infra.agent import submit
from trusted_eval_infra.agent.container_handle import ContainerAgentHandle
from trusted_eval_infra.agent.grader import public_stage1_feedback
from trusted_eval_infra.execution import run_docker
from trusted_eval_infra.files import SecureFileError, read_regular_file


def test_checker_ignores_inherited_license_and_network(tmp_path, monkeypatch):
    from trusted_eval_infra import execution
    from trusted_eval_infra.checkers import run_checker_isolated
    paper = tmp_path / "paper"
    paper.mkdir()
    checker = paper / "feasibility_check.py"
    checker.write_text("pass\n")
    instance = paper / "instance.json"
    instance.write_text("{}")
    solution = tmp_path / "solution.json"
    solution.write_text("{}")
    license_path = tmp_path / "gurobi.lic"
    license_path.write_text("WLSACCESSID=test\nWLSSECRET=test\nLICENSEID=123\n")
    monkeypatch.setenv("GRB_LICENSE_FILE", str(license_path))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure checker entered candidate license infrastructure")
    monkeypatch.setattr(execution, "_wls_execution_slot", forbidden)
    monkeypatch.setattr(execution, "_restricted_wls_egress", forbidden)
    commands = []
    def execute(command, timeout):
        commands.append(command)
        return True, "checker completed", 0.01
    monkeypatch.setattr(execution, "_exec", execute)
    cfg = {"gurobi_lic": str(license_path), "wls_egress": "required",
           "_restricted_network": "candidate-network", "_restricted_proxy": "http://candidate-proxy:3128"}
    before = dict(cfg)
    result = run_checker_isolated(checker_path=str(checker), paper_dir=str(paper),
        instance_file=str(instance), solution_file=str(solution), result_file=str(tmp_path / "result.json"), cfg=cfg)
    assert result[0]
    assert cfg == before
    flat = " ".join(commands[0])
    assert "--network=none" in commands[0]
    assert "GRB_LICENSE_FILE=" in commands[0]
    assert str(license_path) not in flat
    assert "candidate-network" not in flat and "candidate-proxy" not in flat
    assert "HTTPS_PROXY" not in flat


@pytest.mark.skipif(os.environ.get("FRONTIEROR_DOCKER_TESTS") != "1", reason="explicit Docker integration opt-in")
def test_real_checker_runs_while_all_license_slots_are_held(tmp_path, monkeypatch):
    from trusted_eval_infra.execution import _wls_execution_slot
    from trusted_eval_infra.checkers import run_checker_isolated
    paper = tmp_path / "paper"
    paper.mkdir()
    checker = paper / "feasibility_check.py"
    checker.write_text('''import argparse, json, os
p=argparse.ArgumentParser()
for name in ('instance_path','solution_path','result_path'): p.add_argument('--'+name)
a=p.parse_args()
assert not os.environ.get('GRB_LICENSE_FILE')
assert not os.environ.get('HTTPS_PROXY')
assert not os.path.exists('/opt/gurobi/gurobi.lic')
with open(a.result_path,'w') as f: json.dump({'feasible':True},f)
''')
    instance = paper / "instance.json"
    instance.write_text("{}")
    solution = tmp_path / "solution.json"
    solution.write_text("{}")
    license_path = tmp_path / "gurobi.lic"
    license_path.write_text("WLSACCESSID=test\nWLSSECRET=test\nLICENSEID=123\n")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("FRONTIER_OR_WLS_QUEUE_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("GRB_LICENSE_FILE", str(license_path))
    cfg = {"gurobi_lic": str(license_path), "wls_egress": "required", "wls_concurrency": 1,
           "docker_image": "frontieror-candidate:1"}
    result = tmp_path / "result.json"
    with _wls_execution_slot(cfg):
        success, output, _ = run_checker_isolated(checker_path=str(checker), paper_dir=str(paper),
            instance_file=str(instance), solution_file=str(solution), result_file=str(result), cfg=cfg)
    assert success, output
    assert json.loads(result.read_text())["feasible"] is True


def test_license_cooldown_survives_slot_release(tmp_path, monkeypatch):
    import time
    from trusted_eval_infra.execution import _wls_execution_slot
    license_path = tmp_path / "gurobi.lic"
    license_path.write_text("WLSACCESSID=test\nWLSSECRET=test\nLICENSEID=123\n")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    cfg = {"gurobi_lic": str(license_path), "wls_concurrency": 1, "wls_release_delay_seconds": 0.2}
    with _wls_execution_slot(cfg):
        pass
    started = time.monotonic()
    with _wls_execution_slot(cfg):
        assert time.monotonic() - started >= 0.18


def test_independent_licenses_do_not_share_cooldown(tmp_path, monkeypatch):
    from trusted_eval_infra.execution import _wls_execution_slot
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("FRONTIER_OR_WLS_QUEUE_TIMEOUT_SECONDS", "0.15")
    paths = [tmp_path / "a.lic", tmp_path / "b.lic"]
    for index, path in enumerate(paths):
        path.write_text(f"WLSACCESSID=test\nWLSSECRET=test\nLICENSEID={index + 1}\n")
    for path in paths:
        with _wls_execution_slot({"gurobi_lic": str(path), "wls_concurrency": 1,
                                  "wls_release_delay_seconds": 10}):
            pass


def test_large_seed_inputs_do_not_waive_new_file_limits(tmp_path, monkeypatch):
    from trusted_eval_infra.agent import broker
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "code.py").write_text("pass\n")
    (repo / "paper").mkdir()
    (repo / "paper" / "instance.json").write_text("x" * 1024)
    trusted = broker.trusted_seed_blobs(str(repo))
    git("add", ".")
    git("commit", "-qm", "seed")
    monkeypatch.setattr(broker, "MAX_TREE_FILE_BYTES", 100)
    assert broker._validate_tree(repo, git("rev-parse", "HEAD"), trusted)[0]
    (repo / "paper" / "instance.json").write_text("tampered")
    git("commit", "-qam", "tamper")
    assert not broker._validate_tree(repo, git("rev-parse", "HEAD"), trusted)[0]


@pytest.mark.skipif(os.environ.get("FRONTIEROR_DOCKER_TESTS") != "1", reason="explicit Docker integration opt-in")
def test_repeated_container_interrupt_cleans_detached_tools(tmp_path):
    import time
    import uuid
    name = "frontieror-lifecycle-test-" + uuid.uuid4().hex[:12]
    fake_codex = tmp_path / "codex"
    fake_codex.write_text('''#!/usr/bin/python3
import json,subprocess,time,signal,sys
subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],start_new_session=True,env={'PATH':'/usr/bin'},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print(json.dumps({'type':'thread.started','thread_id':'test-session'}),flush=True)
signal.signal(signal.SIGINT,lambda *_:sys.exit(0))
time.sleep(300)
''')
    fake_codex.chmod(0o755)
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    def docker(*args):
        return subprocess.check_output(["docker", *args], text=True).strip()
    try:
        docker("run", "-d", "--rm", "--init", "--name", name, "--network=none",
               "--pids-limit=64", "--cap-drop=ALL", "--security-opt=no-new-privileges",
               "--mount", f"type=bind,src={fake_codex},dst=/usr/local/bin/codex,readonly",
               "--mount", f"type=bind,src={auth},dst=/frontieror/codex-auth.json,readonly",
               "frontieror-coral-agent:0.1", "python3", "-c", "import time;time.sleep(300)")
        baseline = len(docker("top", name, "-eo", "pid").splitlines())
        for index in range(8):
            token = uuid.uuid4().hex
            log_path = tmp_path / f"agent.{index}.log"
            log = log_path.open("w")
            proc = subprocess.Popen([
                "docker", "exec", "--env", f"FRONTIER_OR_INVOCATION={token}",
                "--env", "FRONTIER_OR_AGENT_ID=agent-1", "--env", "HOME=/tmp/test-agent",
                "--env", "CODEX_HOME=/tmp/test-codex", name,
                "python3", "/opt/frontieror/secure_codex_entrypoint.py", "--model", "test",
                "--prompt", "test", "--max-steps", "10"], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            handle = ContainerAgentHandle("agent-1", proc, tmp_path, log_path,
                _log_file=log, container_name=name, invocation=token)
            deadline = time.monotonic() + 10
            while "thread.started" not in log_path.read_text() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert "thread.started" in log_path.read_text(), log_path.read_text()
            assert handle.interrupt() == "test-session"
            deadline = time.monotonic() + 3
            while len(docker("top", name, "-eo", "pid").splitlines()) > baseline and time.monotonic() < deadline:
                time.sleep(0.05)
            assert len(docker("top", name, "-eo", "pid").splitlines()) == baseline
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_feedback_only_uses_fixed_public_categories():
    feedback = public_stage1_feedback({
        "stage1_failure_runtime_crash": 1,
        "stage1_runtime_key": 1,
        "error": "PRIVATE_REFERENCE_SECRET",
        "stage1_runtime_PRIVATE_REFERENCE_SECRET": 1,
    })
    assert "crashed" in feedback and "KeyError" in feedback
    assert "PRIVATE_REFERENCE_SECRET" not in feedback


def test_nonregular_fifo_read_does_not_block(tmp_path):
    path = tmp_path / "solution.json"
    os.mkfifo(path)
    with pytest.raises(SecureFileError, match="regular file"):
        read_regular_file(path, max_bytes=100, label="test")


def test_public_history_and_checkout_keep_branch_and_local_work(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (repo / "code.py").write_text("original\n")
    git("add", "code.py")
    git("commit", "-qm", "original")
    first = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / f"{first}.json").write_text(json.dumps({
        "commit_hash": first, "agent_id": "agent-1", "score": 0.5,
        "status": "improved", "feedback": "feasible", "private_debug": "HIDDEN",
    }))
    monkeypatch.setenv("FRONTIER_OR_ATTEMPTS_DIR", str(attempts))
    monkeypatch.chdir(repo)
    monkeypatch.setattr("sys.argv", ["coral", "log", "--agent", "agent-1"])
    assert submit.main() == 0
    assert "HIDDEN" not in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["coral", "show", first[:8]])
    assert submit.main() == 0
    assert "feasible" in capsys.readouterr().out
    (repo / "code.py").write_text("changed\n")
    (repo / "notes.txt").write_text("keep me\n")
    monkeypatch.setattr("sys.argv", ["coral", "checkout", first[:8]])
    assert submit.main() == 0
    assert (repo / "code.py").read_text() == "original\n"
    assert (repo / "notes.txt").read_text() == "keep me\n"
    assert git("branch", "--show-current") == branch
    assert git("show", "HEAD:code.py") == "changed"


@pytest.mark.skipif(os.environ.get("FRONTIEROR_DOCKER_TESTS") != "1", reason="explicit Docker integration opt-in")
@pytest.mark.parametrize("mode", ["atomic", "symlink", "hardlink", "fifo"])
def test_docker_atomic_output_and_malicious_output(tmp_path, mode):
    code = tmp_path / "code.py"
    code.write_text('''import argparse,json,os
from solution_logger import SolutionLogger
assert callable(SolutionLogger.log_solution)
p=argparse.ArgumentParser()
for name in ('instance_path','solution_path','time_limit','log_path'): p.add_argument('--'+name)
a=p.parse_args()
mode=''' + repr(mode) + '''
if mode == 'atomic':
    with open(a.solution_path+'.tmp','w') as f: json.dump({'objective_value': 1},f)
    os.replace(a.solution_path+'.tmp',a.solution_path)
elif mode == 'symlink': os.symlink('/workspace/instance.json',a.solution_path)
elif mode == 'hardlink':
    with open(a.solution_path+'.tmp','w') as f: f.write('{}')
    os.link(a.solution_path+'.tmp',a.solution_path)
else: os.mkfifo(a.solution_path)
''')
    instance = tmp_path / "instance.json"
    instance.write_text("{}")
    solution = tmp_path / "solution.json"
    success, output, _ = run_docker(str(code), str(instance), str(solution), 10,
        cfg={"anti_hack": True, "docker_image": "frontieror-candidate:1", "wls_egress": "off"})
    assert success is (mode == "atomic"), output
    if success:
        assert json.loads(solution.read_text())["objective_value"] == 1
    else:
        assert not solution.exists()
