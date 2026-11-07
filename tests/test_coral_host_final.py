"""Final-stage backend selection must not relax development isolation."""
import json
from pathlib import Path

import pytest

from test_time_self_evolution import run_eval_modes
from test_time_self_evolution.coral import runner


def test_final_backend_cli_preserves_original_default():
    assert run_eval_modes.parse_args([]).coral_final_exec_mode == 'same'
    assert run_eval_modes.parse_args(['--coral-final-exec-mode', 'bare']).coral_final_exec_mode == 'bare'
    args = run_eval_modes.parse_args(['--coral-final-exec-mode', 'systemd', '--coral-final-cpus', '2', '--coral-final-memory', '256M'])
    assert args.coral_final_cpus == 2 and args.coral_final_memory == '256M'


def test_final_backend_does_not_mutate_development_config():
    original = {'anti_hack': True, 'memory': '100G', 'cpus': 1, 'docker_image': 'sha256:abc', 'docker_image_ref': 'candidate'}
    mode, cfg = runner._resolve_final_execution('docker', original, 'bare')
    assert mode == 'bare' and cfg['anti_hack'] is False
    assert 'docker_image' not in cfg
    assert original['anti_hack'] is True and original['docker_image'] == 'sha256:abc'
    assert runner._resolve_final_execution('docker', original, 'same') == ('docker', original)
    with pytest.raises(ValueError):
        runner._resolve_final_execution('docker', original, 'invalid')


@pytest.mark.parametrize('override', [None, 'bare', 'systemd'])
def test_only_final_stage_changes_and_audit_records_isolation(tmp_path, monkeypatch, override):
    base = tmp_path / 'base'
    seed = base / 'seed'
    seed.mkdir(parents=True)
    code = seed / 'code.py'
    code.write_text("print('benign test fixture')\n")
    task = runner.CoralTask('task', str(base), str(seed), str(base/'config.yaml'), str(base/'run'), str(base/'run/.coral'), str(base/'repo'), str(base/'start.log'), ('agent-1',))
    best = {'commit_hash': 'a'*40, 'agent_id': 'agent-1', 'score': 1.0, 'status': 'improved'}
    captured = {}
    def write_task(**kw):
        captured['development'] = kw
        return task
    def evaluate(*args, **kw):
        captured['final_mode'], captured['final_cfg'], captured['instances'], captured['workers'] = args[8], args[9], args[3], kw['max_workers']
        return {i: {'status': 'pass'} for i in args[3]}
    monkeypatch.setattr(runner.eval_modes, 'mode_run_dir', lambda *a: str(base))
    monkeypatch.setattr(runner, 'write_coral_task', write_task)
    monkeypatch.setattr(runner, '_try_reuse_oneshot_seed', lambda *a: str(code))
    monkeypatch.setattr(runner, 'prepare_coral_env', lambda *a: {})
    monkeypatch.setattr(runner, '_validate_systemd_final_resources', lambda: {'cpu_max': '100000 100000', 'memory_max': '134217728'})
    monkeypatch.setattr(runner, 'run_coral_until_done', lambda *a, **kw: {'stop_reason': 'attempt_budget_complete'})
    monkeypatch.setattr(runner, 'read_best_attempt', lambda *a: best)
    monkeypatch.setattr(runner, 'read_attempts', lambda *a: [best])
    monkeypatch.setattr(runner, 'extract_attempt_code', lambda *a: str(code))
    monkeypatch.setattr(runner, 'validate_instance_content_split', lambda **kw: None)
    monkeypatch.setattr(runner, 'validate_objective_checker', lambda **kw: None)
    monkeypatch.setattr(runner.eval_core, 'get_paper_dir', lambda *a: str(tmp_path))
    monkeypatch.setattr(runner.eval_modes, '_resolve_test_time_limits', lambda *a: {i: 10 for i in a[1]})
    monkeypatch.setattr(runner.eval_modes, 'evaluate_best_on_test_set', evaluate)
    monkeypatch.setattr(runner.eval_modes, 'augment_results_with_staged_qte', lambda r, *a, **kw: r)
    monkeypatch.setattr(runner.eval_modes, 'write_self_evolve_results', lambda **kw: None)
    monkeypatch.setattr(runner.eval_modes, 'write_api_cost_row', lambda *a, **kw: None)
    original = {'anti_hack': True, 'cpus': 1, 'memory': '100G', 'docker_image': 'sha256:abc'}
    kw = {} if override is None else {'final_exec_mode': override}
    if override == 'systemd':
        kw.update(final_cpus=2, final_memory='256M')
    result = runner.run_self_evolve(
        run_id='unit', paper_id='paper', primary_model='openai/model', prompt='test', config={},
        stage1_instances=['tiny'], stage2_instances=['large_1'], test_instances=['large_2','large_3','large_4','large_5'],
        stage1_time_limit=10, stage2_time_limit=10, test_time_limit=10,
        stage1_gap_threshold=.1, exec_mode='docker', exec_cfg=original, t_max=None,
        anti_hack=True, test_instance_workers=4, **kw,
    )
    assert captured['development']['exec_mode'] == 'docker'
    assert captured['development']['exec_cfg']['anti_hack'] is True
    assert original['anti_hack'] is True
    assert captured['workers'] == 4 and len(captured['instances']) == 4
    host = override is not None
    assert captured['final_mode'] == (override if host else 'docker')
    assert captured['final_cfg']['anti_hack'] is (not host)
    execution = json.loads((base/'final_eval/execution.json').read_text())
    assert execution['full_anti_hack'] is (not host)
    assert execution['filesystem_isolated'] is (not host)
    assert execution['memory_limit_enforced'] is (override != 'bare')
    if override == 'systemd':
        assert execution['cpus_per_instance'] == 2
        assert execution['memory_per_instance'] == '256M'
        assert captured['development']['exec_cfg']['memory'] == '100G'
    audit = json.loads((Path(task.coral_dir)/'private/audit/final_artifact.json').read_text())
    assert audit['final_execution'] == execution
    assert result['verification_tier'] == (f'mixed-agent-docker-final-{override}' if host else 'isolated-shared-container-local-auth')


def test_systemd_wls_exception_preserves_default_deny_and_resource_caps(monkeypatch):
    from trusted_eval_infra import execution
    monkeypatch.setattr(execution, '_ensure_logger', lambda *a: None)
    monkeypatch.setattr(execution, '_allocate_cores', lambda n: '0,1')
    monkeypatch.setattr(execution, '_wls_license_fields', lambda p: {'LICENSEID': 'test'})
    monkeypatch.setattr(execution.socket, 'getaddrinfo', lambda *a, **kw: [(None, None, None, None, ('203.0.113.7', 443))])
    default = execution.build_systemd_cmd('code.py', 'instance.json', 'out.json', 10, cfg={'cpus': 2, 'memory': '256M'})
    assert 'CPUQuota=200%' in default and 'MemoryMax=256M' in default
    assert 'IPAddressDeny=any' in default
    assert not any('IPAddressAllow=' in a for a in default)
    wls = execution.build_systemd_cmd('code.py', 'instance.json', 'out.json', 10, cfg={'cpus': 2, 'memory': '256M', 'systemd_allow_wls': True})
    assert 'IPAddressDeny=any' in wls and 'IPAddressAllow=203.0.113.7/32' in wls


def test_systemd_unavailable_does_not_fall_back(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stdout=''))
    with pytest.raises(RuntimeError, match='no bare fallback'):
        runner._validate_systemd_final_resources()
