"""Bounded development-file retrieval for OpenEvolve's OpenAI-compatible client."""
import asyncio
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from test_time_self_evolution.eoh.model_io import TOOL, append_json, read_public_file

MAX_TOOL_ROUNDS = 16
MAX_READS = 32


def configure_public_tools(env, base_dir, paper_id):
    """Use the saved protocol, so legacy/inline resumes never gain new inputs."""
    env.pop('FRONTIER_OR_OPENEVOLVE_PUBLIC_ROOT', None)
    protocol = Path(base_dir) / 'public_context/protocol.json'
    if protocol.exists():
        saved = json.loads(protocol.read_text())
        if saved.get('transport') == 'bounded-read-only-tools':
            root = protocol.parent / paper_id
            if not (root / 'public_manifest.json').is_file():
                raise ValueError('Saved public retrieval manifest is missing')
            env['FRONTIER_OR_OPENEVOLVE_PUBLIC_ROOT'] = str(root.resolve())


async def generate_with_public_tools(client, params, public_root, usage_path=None,
                                     *, max_tool_rounds=MAX_TOOL_ROUNDS):
    """Preserve native sampling parameters and audit every charged response.

    Limits are per generation, not per paper. At the limit a final text-only
    response is requested; it must contain program text rather than tool calls.
    No shell, arbitrary path, checker or final-instance access is offered.
    """
    request = copy.deepcopy(params)
    output = Path(public_root).parent / 'tool_calls' / uuid.uuid4().hex
    output.mkdir(parents=True, mode=0o700)
    call_id = output.name
    reads = 0
    for turn in range(max_tool_rounds + 1):
        tools_enabled = turn < max_tool_rounds and reads < MAX_READS
        if tools_enabled:
            request['tools'] = [TOOL]
            request['tool_choice'] = 'auto'
        else:
            request.pop('tools', None)
            request.pop('tool_choice', None)
        # Request body contains no API credentials. These traces retain actual
        # tool exchanges, including earlier responses when a later call fails.
        (output / f'{turn:02d}-request.json').write_text(json.dumps(request, indent=2) + '\n')
        response = await asyncio.to_thread(client.chat.completions.create, **request)
        data = response.model_dump(mode='json')
        if usage_path:
            append_json(usage_path, {
                'at': datetime.now(timezone.utc).isoformat(),
                'response_id': data.get('id'), 'model': data.get('model'),
                'usage': data.get('usage') or {}, 'call_id': call_id, 'tool_turn': turn,
            })
        (output / f'{turn:02d}-response.json').write_text(json.dumps(data, indent=2) + '\n')
        message = data['choices'][0]['message']
        calls = message.get('tool_calls') or []
        if not calls:
            content = message.get('content')
            if not content:
                raise RuntimeError('Model returned no program text')
            return content
        if not tools_enabled:
            raise RuntimeError('Public-file tool budget exhausted')
        # Preserve reasoning/provider fields needed by compatible endpoints.
        request['messages'].append(message)
        for call in calls:
            try:
                if reads >= MAX_READS:
                    raise ValueError('Public file read budget exhausted; return the program now')
                reads += 1
                if call['function']['name'] != 'read_public_file':
                    raise ValueError('Unknown public tool')
                arguments = json.loads(call['function']['arguments'])
                result = read_public_file(public_root, arguments)
            except (ValueError, TypeError, KeyError) as exc:
                result = {'error': str(exc)}
            except OSError:
                result = {'error': 'Public file is unavailable'}
            request['messages'].append({'role': 'tool', 'tool_call_id': call['id'],
                                        'content': json.dumps(result)})
    raise RuntimeError('Public-file tool budget exhausted')
