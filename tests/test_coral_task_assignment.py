import json
from pathlib import Path
import pytest
from trusted_eval_infra.agent import task_assignment as assignment


def test_assignment_is_scoped_to_managed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(assignment,'ROOT',tmp_path)
    manifest=tmp_path/'assignment.json'
    manifest.write_text(json.dumps({'managed_runs':['old','new'],'owners':{'task':'new'}}))
    assert assignment.assigned_elsewhere(tmp_path/'eval/coral/old/task/model','task',manifest)=='new'
    assert assignment.assigned_elsewhere(tmp_path/'eval/coral/new/task/model','task',manifest) is None
    assert assignment.assigned_elsewhere(tmp_path/'eval/coral/unrelated/task/model','task',manifest) is None
    assert assignment.assigned_elsewhere(tmp_path/'elsewhere/task','task',manifest) is None


def test_native_guard_prevents_old_queued_start_without_running_agent(tmp_path, monkeypatch):
    import yaml
    root=tmp_path/'local_docs/coral-runs';root.mkdir(parents=True)
    monkeypatch.setattr(assignment,'ROOT',tmp_path)
    manifest=root/'assignment.json'
    manifest.write_text(json.dumps({'managed_runs':['old','new'],'owners':{'task':'new'}}))
    monkeypatch.setenv('FRONTIER_OR_TASK_ASSIGNMENT',str(manifest))
    config=tmp_path/'task.yaml';config.write_text(yaml.safe_dump({'grader':{'args':{'base_dir':str(tmp_path/'eval/coral/old/task/model'),'paper_id':'task'}}}))
    with pytest.raises(SystemExit) as error:
        assignment.guard_native_start(['start','-c',str(config)])
    assert error.value.code==78
    data=json.loads((root/'task-handoffs-20261030/task.json').read_text())
    assert data['agent_started'] is False and data['evaluation_started'] is False


def test_cohort_handoff_preserves_historical_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(assignment, 'ROOT', tmp_path)
    manifest = tmp_path / 'assignment.json'
    manifest.write_text(json.dumps({'managed_runs': ['historical', 'new_a', 'new_b'],
        'owners': {'task': 'historical'},
        'run_owners': {'new_a': {'task': 'new_b'}, 'new_b': {'task': 'new_b'}}}))
    assert assignment.assigned_elsewhere(tmp_path/'eval/coral/historical/task/model', 'task', manifest) is None
    assert assignment.assigned_elsewhere(tmp_path/'eval/coral/new_a/task/model', 'task', manifest) == 'new_b'
    assert assignment.assigned_elsewhere(tmp_path/'eval/coral/new_b/task/model', 'task', manifest) is None
