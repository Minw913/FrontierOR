import json
from types import SimpleNamespace
import pytest
from test_time_self_evolution.public_context import prepare_public_context
from test_time_self_evolution.openevolve.runner import write_openevolve_config
from test_time_self_evolution.eoh.problem_adapter import EfficientOptPrompts
from test_time_self_evolution.eoh.runner import _build_eoh_system_message
import yaml


def task(tmp_path):
    p=tmp_path/'task';(p/'instance').mkdir(parents=True)
    for f in ['problem_description.txt','instance_schema.json','solution_schema.json']:
        (p/f).write_text('{}')
    for name,value in [('tiny_instance.json','TINY_PUBLIC'),('large_instance_1.json','DEV_PUBLIC'),('large_instance_2.json','HIDDEN_FINAL')]:
        (p/'instance'/name).write_text(json.dumps({'data':value}))
    (p/'feasibility_check.py').write_text('PRIVATE_CHECKER')
    return p


def test_actual_prompts_include_dev_and_never_final(tmp_path):
    p=task(tmp_path)
    context=prepare_public_context(p,tmp_path/'run','task',['tiny'],['large_1'],['large_2'])
    config=tmp_path/'oe.yaml';write_openevolve_config(str(config),'model',public_context=context)
    prompts=EfficientOptPrompts('TASK',context)
    eoh=_build_eoh_system_message(SimpleNamespace(prompt_other_inf=prompts.get_other_inf()))
    for text in [yaml.safe_load(config.read_text())['prompt']['system_message'],prompts.get_task(),eoh]:
        assert 'TINY_PUBLIC' in text and 'DEV_PUBLIC' in text
        assert 'HIDDEN_FINAL' not in text and 'PRIVATE_CHECKER' not in text
        assert '"stage_boundary": 0.01' in text
    # Resume uses the original frozen context, even if input files later change.
    (p/'instance/large_instance_1.json').write_text('{}')
    assert prepare_public_context(p,tmp_path/'run','task',['tiny'],['large_1'],['large_2'],resume=True)==context


def test_overlap_rejected_and_legacy_resume_preserved(tmp_path):
    p=task(tmp_path)
    with pytest.raises(ValueError,match='overlap'):
        prepare_public_context(p,tmp_path/'run','task',['tiny'],['large_1'],['large_1'])
    assert prepare_public_context(p,tmp_path/'old','task',['tiny'],['large_1'],['large_2'],resume=True)==''


def test_large_inputs_fail_before_silent_truncation(tmp_path,monkeypatch):
    p=task(tmp_path)
    monkeypatch.setattr('test_time_self_evolution.public_context.MAX_INLINE_BYTES',1)
    with pytest.raises(ValueError,match='Chunked retrieval'):
        prepare_public_context(p,tmp_path/'run','task',['tiny'],['large_1'],['large_2'])
