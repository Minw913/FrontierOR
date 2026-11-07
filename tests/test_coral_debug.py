import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from trusted_eval_infra.agent.broker import drain_eval_requests
from trusted_eval_infra.agent.debug import DebugRunner


def make_task(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    (repo / 'code.py').write_text("print('test')\n")
    git('add', '.')
    git('commit', '-qm', 'seed')
    git('checkout', '-qb', 'coral/agent-1')
    seed = tmp_path / 'seed'
    (seed / 'example/instance').mkdir(parents=True)
    (seed / 'example/instance/tiny_instance.json').write_text('{}')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'grader': {'args': {'paper_id': 'example', 'exec_cfg': {'memory': '100G', 'docker_image': 'candidate'}}}}))
    task = SimpleNamespace(repo_dir=str(repo), seed_dir=str(seed), config_path=str(config), coral_dir=str(tmp_path / '.coral'), agent_ids=('agent-1',))
    request = {'kind': 'debug', 'schema_version': 1, 'nonce': 'a' * 32, 'agent_id': 'agent-1', 'commit_hash': git('rev-parse', 'HEAD'), 'message': 'debug test'}
    return task, request


def wait(runner):
    deadline = time.monotonic() + 5
    while runner.pending and time.monotonic() < deadline:
        time.sleep(.01)
        runner.poll()
    assert not runner.pending


def test_debug_executes_tiny_without_score_or_attempt_even_when_eval_budget_exhausted(tmp_path, monkeypatch):
    task, request = make_task(tmp_path)
    seen = {}
    def run(code, instance, solution, seconds, log, cfg):
        seen.update(instance=instance, seconds=seconds, cfg=cfg)
        Path(solution).write_text('{"objective_value": 7}')
        return True, 'worked', .1
    monkeypatch.setattr('trusted_eval_infra.execution.run_docker', run)
    inbox = Path(task.coral_dir) / 'private/eval_requests/inbox'
    inbox.mkdir(parents=True)
    (inbox / ('a' * 32 + '.json')).write_text(json.dumps(request))
    runner = DebugRunner(task)
    try:
        drain_eval_requests(task, max_attempts=0, debug_runner=runner)
        wait(runner)
        data = json.loads((runner.responses / ('a' * 32 + '.json')).read_text())
        assert data['success'] and data['counts_as_attempt'] is False
        assert 'score' not in data and 'feasible' not in data
        assert data['solution']['objective_value'] == 7
        assert seen['instance'].endswith('/example/instance/tiny_instance.json')
        assert seen['seconds'] == 30 and seen['cfg']['anti_hack'] is True
        assert seen['cfg']['gurobi_lic'] == ''
        assert seen['cfg']['wls_egress'] == 'off'
        assert '_restricted_network' not in seen['cfg']
        assert '_restricted_proxy' not in seen['cfg']
        assert not list((Path(task.coral_dir) / 'public/attempts').glob('*.json'))
        assert not (Path(task.coral_dir) / 'private/audit/eval_requests.jsonl').exists()
    finally:
        runner.close()


def test_debug_rejects_instance_override_and_unregistered_agent(tmp_path):
    task, request = make_task(tmp_path)
    runner = DebugRunner(task)
    try:
        for update in ({'instance': 'large_5'}, {'agent_id': 'agent-2'}, {'agent_id': []}):
            assert not runner.submit({**request, **update}, 'a' * 32 + '.json', frozenset({'agent-1'}))
        assert not runner.pending
    finally:
        runner.close()


def test_debug_quota_persists_and_crash_output_is_available(tmp_path, monkeypatch):
    task, request = make_task(tmp_path)
    monkeypatch.setattr('trusted_eval_infra.execution.run_docker', lambda *args: (False, 'TypeError: wrong argument', .2))
    monkeypatch.setattr('trusted_eval_infra.agent.debug.MAX_DEBUG_RUNS', 1)
    runner = DebugRunner(task)
    runner.submit(request, 'a' * 32 + '.json', frozenset({'agent-1'}))
    wait(runner)
    data = json.loads((runner.responses / ('a' * 32 + '.json')).read_text())
    assert data['success'] is False and 'TypeError' in data['output']
    runner.close()
    resumed = DebugRunner(task)
    try:
        assert not resumed.submit({**request, 'nonce': 'b' * 32}, 'b' * 32 + '.json', frozenset({'agent-1'}))
        assert 'budget exhausted' in json.loads((resumed.responses / ('b' * 32 + '.json')).read_text())['reason']
    finally:
        resumed.close()
