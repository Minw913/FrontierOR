from pathlib import Path
from types import SimpleNamespace
from test_time_self_evolution.coral import runner


def test_mixed_batch_only_resumes_existing_checkpoint(tmp_path):
    assert not runner._resume_existing_run(True, str(tmp_path))
    config=tmp_path/'coral_run/.coral/config.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('task: {}')
    assert runner._resume_existing_run(True, str(tmp_path))
    assert not runner._resume_existing_run(False, str(tmp_path))


def test_native_cli_finds_custom_run_using_breadcrumb(tmp_path):
    from coral.cli._helpers import find_coral_dir
    import os
    run=tmp_path/'custom'/'coral_run'
    coral=run/'.coral';coral.mkdir(parents=True)
    task=SimpleNamespace(run_dir=str(run),coral_dir=str(coral))
    cwd=runner._control_cwd(task)
    previous=os.getcwd()
    try:
        os.chdir(cwd)
        assert find_coral_dir(None,None)==coral.resolve()
    finally:
        os.chdir(previous)
