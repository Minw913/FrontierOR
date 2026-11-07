"""Explicit cross-model starters must not silently fall back or change on resume."""
import json
from pathlib import Path

import pytest

from test_time_self_evolution import run_eval_modes
from test_time_self_evolution.coral import runner


def test_cli_preserves_default_and_exposes_best_program():
    assert run_eval_modes.parse_args([]).coral_starter_mode == 'same-model'
    args = run_eval_modes.parse_args(['--coral-starter-mode', 'best-one-shot-program'])
    assert args.coral_starter_model == 'claude-fable-5'


def test_best_program_ignores_old_override_and_records_source(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT_DIR', str(tmp_path))
    monkeypatch.setenv('FRONTIER_OR_CORAL_SEED_MODEL', 'wrong-model')
    source = tmp_path / 'eval/seed_programs/task/claude-fable-5'
    source.mkdir(parents=True)
    (source / 'code.py').write_text('print("Fable final")\n')
    out = tmp_path / 'seed'
    result = runner._prepare_best_oneshot_seed('task', 'openai/gpt-5.6-sol', str(out), 'claude-fable-5')
    assert Path(result).read_bytes() == (source / 'code.py').read_bytes()
    metadata = json.loads((out / '_seed_source.json').read_text())
    assert metadata['model'] == 'claude-fable-5'
    assert metadata['agent_model'] == 'openai/gpt-5.6-sol'
    assert metadata['starter_mode'] == 'best-one-shot-program'
    assert runner._prepare_best_oneshot_seed('task', 'openai/gpt-5.6-sol', str(out), 'claude-fable-5', resume=True) == result
    with pytest.raises(ValueError, match='Resume starter'):
        runner._prepare_best_oneshot_seed('task', 'agent', str(out), 'other-model', resume=True)
    Path(result).write_text('changed')
    with pytest.raises(ValueError, match='Resume starter'):
        runner._prepare_best_oneshot_seed('task', 'agent', str(out), 'claude-fable-5', resume=True)


def test_missing_final_program_is_error_even_if_attempt_zero_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT_DIR', str(tmp_path))
    source = tmp_path / 'eval/eval_tasks/task/claude-fable-5'
    source.mkdir(parents=True)
    (source / 'code_attempt0.py').write_text('print("not final")')
    with pytest.raises(ValueError, match='no generation fallback'):
        runner._prepare_best_oneshot_seed('task', 'agent', str(tmp_path / 'seed'), 'claude-fable-5')
