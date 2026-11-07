"""Shared final-one-shot seed lookup (never substitutes attempt zero)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from trusted_eval_infra.files import SecureFileError, copy_regular_file, read_regular_file

MAX_SEED_BYTES = 4 * 1024 * 1024


def copy_final_oneshot_seed(root: str, paper_id: str, model: str, seed_dir: str) -> Optional[str]:
    """Prefer the published final snapshot, then a runtime final code.py.

    FRONTIER_OR_FINAL_ONESHOT_ROOT explicitly selects a versioned
    <task>/<model> artifact tree; missing overrides never use an older seed.

    The snapshot makes a fresh clone usable without copying eval_tasks first.
    Invalid existing sources are rejected, not replaced by a different version.
    Preserve the adjacent logger and record the exact source/hash. No candidate
    program is imported or executed here.
    """
    for value in (paper_id, model):
        if not value or value in ('.', '..') or '/' in value or '\\' in value:
            raise ValueError('Seed task/model must be a single path component')
    repo = Path(root)
    override = os.environ.get('FRONTIER_OR_FINAL_ONESHOT_ROOT')
    choices = ([Path(override).expanduser().resolve()/paper_id/model/'code.py'] if override else
               [repo/'eval/seed_programs'/paper_id/model/'code.py',
                repo/'eval/eval_tasks'/paper_id/model/'code.py'])
    if not override and model == 'grok-4.20':
        choices.append(repo/'eval/eval_tasks'/paper_id/'grok-4.20-beta/code.py')
    source = next((path for path in choices if os.path.lexists(path)), None)
    if source is None:
        return None
    folder = Path(seed_dir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'code.py'
    try:
        code = read_regular_file(source, max_bytes=MAX_SEED_BYTES, label='final one-shot code.py')
        if not code:
            return None
        expected = None
        manifest_path = repo/'eval/seed_programs/manifest.json'
        if not override and source == choices[0] and manifest_path.is_file():
            manifest = json.loads(read_regular_file(manifest_path, max_bytes=16*1024*1024,
                                                   label='seed manifest'))
            expected = next((row['files'] for row in manifest['programs']
                             if row['paper_id'] == paper_id and row['model'] == model), None)
            if expected is None or hashlib.sha256(code).hexdigest() != expected['code.py']['sha256']:
                raise SecureFileError('Final seed does not match its published manifest')
        copy_regular_file(source, target, max_bytes=MAX_SEED_BYTES, label='final one-shot code.py')
        copied = read_regular_file(target, max_bytes=MAX_SEED_BYTES, label='copied final seed')
        if copied != code:
            raise SecureFileError('Final seed changed while copying')
        helper = source.parent/'solution_logger.py'
        helper_hash = None
        if os.path.lexists(helper):
            data = read_regular_file(helper, max_bytes=MAX_SEED_BYTES, label='seed solution logger')
            helper_hash = hashlib.sha256(data).hexdigest()
            if expected and expected.get('solution_logger.py', {}).get('sha256') != helper_hash:
                raise SecureFileError('Seed logger does not match its published manifest')
            copy_regular_file(helper, folder/helper.name, max_bytes=MAX_SEED_BYTES, label='seed solution logger')
        elif expected and 'solution_logger.py' in expected:
            raise SecureFileError('Published seed logger is missing')
    except SecureFileError as error:
        target.unlink(missing_ok=True)
        print(f'[warn] rejected final one-shot seed {paper_id}/{model}: {error}')
        return None
    provenance = dict(seed_version='final_one_shot', paper_id=paper_id, model=model,
                      source=str(source.relative_to(repo)) if source.is_relative_to(repo) else str(source),
                      sha256=hashlib.sha256(code).hexdigest(),
                      solution_logger_sha256=helper_hash)
    (folder/'_seed_source.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (folder/'_seed_source.txt').write_text(
        f'reused from final one-shot: {provenance["source"]}\n'
        f'seed_version: final_one_shot\nsha256: {provenance["sha256"]}\n')
    print(f'[reuse-oneshot] {paper_id}/{model}: copied final one-shot → {target}')
    return str(target)
