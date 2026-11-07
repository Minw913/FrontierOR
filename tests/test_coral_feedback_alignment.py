"""Public evaluation feedback must survive the CORAL venv and grader bridge."""
import json
from types import SimpleNamespace

import pytest
from test_time_self_evolution.openevolve import evaluator
from trusted_eval_infra.agent import grader


def make_grader(tmp_path):
    (tmp_path / 'code.py').write_text('pass\n')
    return SimpleNamespace(
        codebase_path=str(tmp_path),
        args=dict(base_dir=str(tmp_path), paper_id='example', stage1_instances=['tiny'],
                  stage1_time_limit=300, stage1_gap_threshold=.1,
                  stage2_instances=['large_3'], stage2_time_limit=3600),
        bundle=lambda score, message, feedback='': dict(score=score, message=message, feedback=feedback),
        fail=lambda message: pytest.fail(message),
    )


def test_feedback_survives_without_openevolve(monkeypatch):
    monkeypatch.setattr(evaluator, '_HAS_EVALUATION_RESULT', False)
    result = evaluator._maybe_wrap_with_artifacts(
        {'combined_score': 0}, {'summary': 'tiny failed', 'failure_breakdown': 'KeyError: demand'})
    assert result.get('combined_score') == 0
    assert json.loads(json.dumps(result)) == {'combined_score': 0}
    assert grader.evaluation_feedback(result) == 'tiny failed\n\nKeyError: demand'


@pytest.mark.parametrize('kind,error', [
    ('runtime_error', 'Traceback:\nKeyError: demand'),
    ('checker_error', 'JSONDecodeError: Expecting value'),
])
def test_stage1_forwards_shared_diagnostics(tmp_path, monkeypatch, kind, error):
    monkeypatch.setattr(evaluator, '_HAS_EVALUATION_RESULT', False)
    artifacts = evaluator._build_stage1_artifacts(
        {'status': 'fail', 'fail_reason': kind, 'error': error}, str(tmp_path), 'example')
    result = evaluator._maybe_wrap_with_artifacts({'combined_score': 0}, artifacts)
    monkeypatch.setattr(evaluator, 'evaluate_stage1', lambda path: result)
    monkeypatch.setattr(evaluator, 'evaluate_stage2', lambda path: pytest.fail('tiny failure must gate dev'))
    monkeypatch.setattr(grader.subprocess, 'check_output', lambda *a, **kw: 'abc\n')
    monkeypatch.setenv('EFFICIENT_OR_PAPER_ID', 'original')
    response = grader.Grader.evaluate(make_grader(tmp_path))
    assert error in response['feedback']
    assert response['score'] == 0
    import os
    assert os.environ['EFFICIENT_OR_PAPER_ID'] == 'original'
    sidecar = json.loads((tmp_path / 'coral_eval/attempt_metadata/abc.json').read_text())
    assert sidecar['feedback'] == response['feedback']
    assert sidecar['feedback_protocol'] == grader.FEEDBACK_PROTOCOL


def test_stage2_preserves_score_and_both_stage_artifacts(tmp_path, monkeypatch):
    s1 = evaluator.MetricsWithArtifacts({'combined_score': 1}, {'summary': 'Tiny passed; gap=0; time=1s'})
    s2 = evaluator.MetricsWithArtifacts({'combined_score': 1.5, 'private_field': 'DO_NOT_FORWARD'},
        {'score_summary': 'Quality and speed score 1.5', 'failure_breakdown': 'Capacity constraint violated',
         'private_field': 'DO_NOT_FORWARD', 'evaluation_id': 'INTERNAL_ID'})
    monkeypatch.setattr(evaluator, 'evaluate_stage1', lambda path: s1)
    monkeypatch.setattr(evaluator, 'evaluate_stage2', lambda path: s2)
    monkeypatch.setattr(grader.subprocess, 'check_output', lambda *a, **kw: 'def\n')
    result = grader.Grader.evaluate(make_grader(tmp_path))
    assert result['score'] == 1.5
    assert result['feedback'] == 'Tiny passed; gap=0; time=1s\n\nQuality and speed score 1.5\n\nCapacity constraint violated'
    assert 'DO_NOT_FORWARD' not in result['feedback']
    assert 'INTERNAL_ID' not in result['feedback']


def test_constraint_details_reach_coral(tmp_path, monkeypatch):
    (tmp_path / 'feasi_result_tiny.json').write_text(json.dumps({
        'feasible': False, 'violations': ['Constraint (5): terminal setup state sum=0, expected=1']}))
    artifacts = evaluator._build_stage1_artifacts(
        {'status': 'fail', 'fail_reason': 'infeasible', 'feasible': False}, str(tmp_path), 'example')
    result = evaluator.MetricsWithArtifacts({'combined_score': 0}, artifacts)
    monkeypatch.setattr(evaluator, 'evaluate_stage1', lambda path: result)
    monkeypatch.setattr(grader.subprocess, 'check_output', lambda *a, **kw: 'ghi\n')
    response = grader.Grader.evaluate(make_grader(tmp_path))
    assert 'terminal setup state sum=0, expected=1' in response['feedback']
