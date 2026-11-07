import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_time_self_evolution.public_context import prepare_public_context
from test_time_self_evolution.openevolve.public_tools import configure_public_tools, generate_with_public_tools


def public_task(tmp_path):
    task = tmp_path / 'task'
    (task / 'instance').mkdir(parents=True)
    for name in ['problem_description.txt', 'instance_schema.json', 'solution_schema.json']:
        (task / name).write_text('{}')
    (task / 'instance/tiny_instance.json').write_text('{}')
    (task / 'instance/large_instance_1.json').write_text(json.dumps({'data': 'X' * 300000 + 'TAIL'}))
    (task / 'instance/large_instance_2.json').write_text('HIDDEN_FINAL')
    (task / 'feasibility_check.py').write_text('PRIVATE_CHECKER')
    run = tmp_path / 'run'
    context = prepare_public_context(task, run, 'task', ['tiny'], ['large_1'], ['large_2'], allow_retrieval=True)
    env = {}
    configure_public_tools(env, run, 'task')
    return Path(env['FRONTIER_OR_OPENEVOLVE_PUBLIC_ROOT']), context, run


def reply(content=None, calls=None, number=0):
    from openai.types.chat import ChatCompletion
    return ChatCompletion.model_validate({
        'id': f'response-{number}', 'object': 'chat.completion', 'created': 1, 'model': 'test',
        'choices': [{'index': 0, 'finish_reason': 'tool_calls' if calls else 'stop',
                     'message': {'role': 'assistant', 'content': content, 'tool_calls': calls}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13, 'cost': 0.002},
    })


def tool(path, offset=0, number=0):
    return {'id': f'tool-{number}', 'type': 'function', 'function': {
        'name': 'read_public_file', 'arguments': json.dumps({'path': path, 'offset': offset, 'length': 64})}}


def test_native_client_reads_large_dev_and_audits_every_response(tmp_path, monkeypatch):
    from openevolve.llm.openai import OpenAILLM
    root, context, run = public_task(tmp_path)
    assert len(context) < 20000 and 'HIDDEN_FINAL' not in context
    manifest = json.loads((root / 'public_manifest.json').read_text())['files']
    dev = next(name for name in manifest if 'large_instance_1' in name)
    responses = iter([reply(calls=[tool(dev, 299980)], number=1), reply('print("candidate")', number=2)])
    requests = []
    def create(**params):
        import copy
        requests.append(copy.deepcopy(params))
        return next(responses)
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setenv('FRONTIER_OR_OPENEVOLVE_PUBLIC_ROOT', str(root))
    monkeypatch.setenv('FRONTIER_OR_OPENEVOLVE_USAGE_PATH', str(run / 'usage.jsonl'))
    params = {'model': 'test', 'messages': [{'role': 'user', 'content': context}], 'temperature': 0.8, 'max_tokens': 100}
    assert asyncio.run(llm._call_api(params)) == 'print("candidate")'
    assert len(params['messages']) == 1  # retries don't inherit partial tool exchanges
    result = json.loads(requests[1]['messages'][-1]['content'])
    assert 'TAIL' in result['text'] and result['offset'] == 299980
    assert requests[1]['temperature'] == 0.8 and requests[1]['max_tokens'] == 100
    usage = [json.loads(line) for line in (run / 'usage.jsonl').read_text().splitlines()]
    assert [r['response_id'] for r in usage] == ['response-1', 'response-2']
    assert sum(r['usage']['cost'] for r in usage) == 0.004
    assert len(list((run / 'public_context/tool_calls').glob('*/*-response.json'))) == 2


def test_private_paths_and_escaping_symlinks_rejected(tmp_path):
    root, context, run = public_task(tmp_path)
    hidden = tmp_path / 'secret'; hidden.write_text('SECRET')
    (root / 'linked').symlink_to(hidden)
    manifest_path = root / 'public_manifest.json'
    manifest = json.loads(manifest_path.read_text()); manifest['files']['linked'] = 'unused'
    manifest_path.write_text(json.dumps(manifest))
    calls = [tool(name, number=i) for i, name in enumerate([
        '../secret', str(hidden), 'instance/large_instance_2.json', 'feasibility_check.py', 'linked'])]
    responses = iter([reply(calls=calls), reply('program')]); requests = []
    def create(**params):
        import copy
        requests.append(copy.deepcopy(params)); return next(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert asyncio.run(generate_with_public_tools(client, {'messages': []}, root)) == 'program'
    results = [json.loads(m['content']) for m in requests[1]['messages'] if m['role'] == 'tool']
    assert len(results) == 5 and all('error' in r for r in results)
    assert 'SECRET' not in json.dumps(results)


def test_tool_budget_forces_text_only_and_retains_usage_on_failure(tmp_path):
    root, _, run = public_task(tmp_path)
    responses = iter([reply(calls=[tool('invalid')], number=1), reply(calls=[tool('invalid')], number=2)])
    requests = []
    def create(**params):
        import copy
        requests.append(copy.deepcopy(params)); return next(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match='budget exhausted'):
        asyncio.run(generate_with_public_tools(client, {'messages': []}, root, run / 'usage.jsonl', max_tool_rounds=1))
    assert 'tools' in requests[0] and 'tools' not in requests[1]
    assert len((run / 'usage.jsonl').read_text().splitlines()) == 2


def test_resume_does_not_enable_tools_for_legacy_or_inline_runs(tmp_path):
    env = {'FRONTIER_OR_OPENEVOLVE_PUBLIC_ROOT': '/another/task'}
    configure_public_tools(env, tmp_path, 'task')
    assert not env
    folder = tmp_path / 'public_context'; folder.mkdir()
    (folder / 'protocol.json').write_text(json.dumps({'version': 'public-dev-context-v1'}))
    configure_public_tools(env, tmp_path, 'task')
    assert not env
