import json
import pytest
from test_time_self_evolution.eoh.task_handoff import skip_transferred_paper


def test_handoff_is_scoped_and_preserves_retained_tasks(tmp_path, capsys):
    folder = tmp_path / 'eval/eoh/source'; folder.mkdir(parents=True)
    (folder / 'task-handoff.json').write_text(json.dumps({
        'source_run_id': 'source', 'transferred_tasks': ['moved']}))
    assert skip_transferred_paper(tmp_path, 'source', ['moved'])
    assert 'no model call or evaluation started' in capsys.readouterr().out
    assert not skip_transferred_paper(tmp_path, 'source', ['retained'])
    assert not skip_transferred_paper(tmp_path, 'destination', ['moved'])
    assert not skip_transferred_paper(tmp_path, None, ['moved'])
    with pytest.raises(ValueError):
        skip_transferred_paper(tmp_path, 'source', ['moved', 'retained'])


def test_corrupt_manifest_fails_closed(tmp_path):
    folder = tmp_path / 'eval/eoh/source'; folder.mkdir(parents=True)
    (folder / 'task-handoff.json').write_text('{')
    with pytest.raises(ValueError):
        skip_transferred_paper(tmp_path, 'source', ['moved'])
