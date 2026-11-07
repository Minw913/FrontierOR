import json
from pathlib import Path
import pytest
import yaml
from test_time_self_evolution.openevolve import runner, evaluator
from test_time_self_evolution.run_eval_modes import parse_args


def test_queue_timeout_covers_both_stages(monkeypatch):
    monkeypatch.setenv('FRONTIER_OR_WLS_CONCURRENCY', '2')
    monkeypatch.setenv('FRONTIER_OR_WLS_QUEUE_TIMEOUT_SECONDS', '14400')
    budget = runner.evaluation_timeout(['tiny'], ['large_3'], 300, 3600, {})
    assert budget > 300 + 3600 + 2 * 14400


def test_config_preserves_larger_timeout_and_sets_concurrency(tmp_path):
    p = tmp_path / 'config.yaml'
    runner.write_openevolve_config(str(p), 'openai/gpt-5.6-sol',
                                  evaluator_timeout=33180, parallel_evaluations=1)
    d = yaml.safe_load(p.read_text())
    assert d['evaluator']['timeout'] == 33180
    assert d['evaluator']['parallel_evaluations'] == 1
    assert d['llm']['primary_model'] == 'openai/gpt-5.6-sol'


def test_systemd_wls_flag_reaches_evaluator(monkeypatch):
    monkeypatch.setenv('EFFICIENT_OR_PAPER_ID', 'task')
    monkeypatch.setenv('EFFICIENT_OR_EXEC_SYSTEMD_ALLOW_WLS', 'True')
    cfg = evaluator._common_env()[-1]
    assert cfg['systemd_allow_wls'] is True
    monkeypatch.delenv('EFFICIENT_OR_EXEC_SYSTEMD_ALLOW_WLS')
    assert evaluator._common_env()[-1]['systemd_allow_wls'] is False


def test_selected_fable_seed_and_resume_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT_DIR', str(tmp_path))
    source = tmp_path / 'eval/seed_programs/task/claude-fable-5'
    source.mkdir(parents=True)
    (source / 'code.py').write_text('print("final")')
    seed = tmp_path / 'seed'
    result = runner.prepare_explicit_starter('task', 'openai/gpt-5.6-sol', 'claude-fable-5', str(seed))
    assert Path(result['code_path']).read_text() == 'print("final")'
    assert json.loads((seed / '_seed_source.json').read_text())['starter_mode'] == 'best-one-shot-program'
    runner.prepare_explicit_starter('task', 'openai/gpt-5.6-sol', 'claude-fable-5', str(seed), resume=True)
    (seed / 'code.py').write_text('changed')
    with pytest.raises(ValueError, match='mismatch'):
        runner.prepare_explicit_starter('task', 'openai/gpt-5.6-sol', 'claude-fable-5', str(seed), resume=True)


def test_explicit_missing_seed_never_generates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT_DIR', str(tmp_path))
    with pytest.raises(ValueError, match='no generation fallback'):
        runner.prepare_explicit_starter('task', 'openai/gpt-5.6-sol', 'claude-fable-5', str(tmp_path / 'seed'))


def test_cli_exposes_openevolve_controls():
    args = parse_args(['--framework', 'openevolve', '--systemd-allow-wls',
                       '--openevolve-starter-mode', 'best-one-shot-program',
                       '--openevolve-parallel-evaluations', '1'])
    assert args.systemd_allow_wls
    assert args.openevolve_starter_model == 'claude-fable-5'
    assert args.openevolve_parallel_evaluations == 1


def test_billing_audit_keeps_missing_cost_unknown_and_avoids_double_counting():
    from test_time_self_evolution.openevolve.usage_audit import summarize
    rows = {'id': {'usage': {'prompt_tokens': 10, 'completion_tokens': 7,
                            'completion_tokens_details': {'reasoning_tokens': 3}}}}
    result = summarize(rows, {})
    assert result['prompt_tokens'] == 10 and result['completion_tokens'] == 7
    assert result['reasoning_tokens'] == 3
    assert result['missing_response_costs'] == result['missing_billing_costs'] == 1
    assert not result['complete_for_observed_responses']
    rows['id']['usage']['cost'] = 0.0123
    verified = summarize(rows, {'id': {'total_cost': 0.0123, 'tokens_prompt': 8}})
    assert verified['complete_for_observed_responses']
    assert verified['billing_cost_usd_verified'] == '0.0123'
    assert verified['billing_normalized_prompt_tokens'] == 8
