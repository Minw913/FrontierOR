from pathlib import Path
import one_shot_eval as ose


def test_versioned_output_root_preserves_originals(tmp_path, monkeypatch):
    old = tmp_path / 'eval/eval_tasks/fragkos2016/fable'
    old.mkdir(parents=True)
    (old / 'code.py').write_text('original')
    monkeypatch.setattr(ose, 'ROOT_DIR', str(tmp_path))
    monkeypatch.setattr(ose, 'OUTPUT_ROOT', str(tmp_path / 'rerun/programs'))
    monkeypatch.setattr(ose, 'CODE_ROOT', None)
    new = Path(ose.get_model_eval_dir('fragkos2016', 'fable'))
    assert new == tmp_path / 'rerun/programs/fragkos2016/fable'
    assert Path(ose.get_model_code_dir('fragkos2016', 'fable')) == new
    (new / 'code.py').write_text('corrected')
    assert (old / 'code.py').read_text() == 'original'


def test_explicit_read_root_remains_independent_of_output_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ose, 'CODE_ROOT', str(tmp_path / 'seeds'))
    monkeypatch.setattr(ose, 'OUTPUT_ROOT', str(tmp_path / 'outputs'))
    assert Path(ose.get_model_code_dir('task', 'model')) == tmp_path / 'seeds/task/model'
    assert Path(ose.get_model_eval_dir('task', 'model')) == tmp_path / 'outputs/task/model'
