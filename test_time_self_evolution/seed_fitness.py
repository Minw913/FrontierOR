"""Host-side seed-only fitness cache; never reuse one-shot aggregate scores.

Cache identities include source/data/scoring contents, runtime settings and host.
Stage evidence is copied on a hit, so downstream feedback/reporting still works.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def configure_seed_cache(env, seed_path, paper_dir, root):
    """Configure only trusted seed bytes; cache is outside agent workspaces."""
    root, paper = Path(root), Path(paper_dir)
    paths = set()
    for folder in (paper, root / 'test_time_self_evolution/scoring'):
        paths.update(p for p in folder.glob('*') if p.is_file())
    for name in ('one_shot_eval.py', 'test_time_self_evolution/openevolve/evaluator.py',
                 'test_time_self_evolution/openevolve/evidence.py',
                 'test_time_self_evolution/seed_fitness.py'):
        paths.add(root / name)
    # Hash execution/checker infrastructure too, not merely its version label.
    for name in ('scripts/utils', 'trusted_eval_infra'):
        paths.update((root / name).rglob('*.py'))
    names = set()
    for stage in ('STAGE1', 'STAGE2'):
        names.update(env.get(f'EFFICIENT_OR_{stage}_INSTANCES', '').split(','))
    for name in names - {''}:
        filename = 'tiny_instance.json' if name == 'tiny' else name.replace('large_', 'large_instance_') + '.json'
        paths.add(paper / 'instance' / filename)
        # References/checker outputs may contain the objective as well as runtime.
        paths.update((paper / 'gurobi_solution').glob('*.json'))
    data_root = Path(env.get('FRONTIER_OR_DATA_DIR', str(root / 'frontier-or')))
    paths.add(data_root / 'metadata/paper_meta_info.json')
    paths.add(Path(env.get('FRONTIER_OR_GUROBI_REFERENCE',
                           str(data_root / 'metadata/gurobi_references.parquet'))))
    paths.update((root / 'eval').glob('gurobi*.csv'))
    assets = {str(p.resolve()): digest(p) if p.is_file() else None for p in sorted(paths)}
    settings = {k: v for k, v in env.items() if k.startswith((
        'EFFICIENT_OR_STAGE', 'EFFICIENT_OR_EXEC_', 'FRONTIER_OR_GUROBI_'))}
    settings.update(t_max=env.get('EFFICIENT_OR_T_MAX'),
                    backend=env.get('EFFICIENT_OR_EXEC_MODE'),
                    host=platform.node(), machine=platform.machine(),
                    python=platform.python_version(), cpu=platform.processor())
    if hasattr(os, 'sched_getaffinity'):
        settings['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    try:
        import gurobipy
        settings['gurobi'] = gurobipy.gurobi.version()
    except ImportError:
        settings['gurobi'] = None
    identity = hashlib.sha256(json.dumps({'assets': assets, 'settings': settings},
                                        sort_keys=True).encode()).hexdigest()
    env['FRONTIER_OR_SEED_SHA256'] = hashlib.sha256(Path(seed_path).read_text().strip().encode()).hexdigest()
    env['FRONTIER_OR_SEED_FITNESS_ID'] = identity
    env['FRONTIER_OR_SEED_FITNESS_CACHE'] = str(root / '.experiment-cache/seed-fitness-v1')


def cache_path(source_digest, stage):
    if source_digest != os.environ.get('FRONTIER_OR_SEED_SHA256'):
        return None
    root = os.environ.get('FRONTIER_OR_SEED_FITNESS_CACHE')
    identity = os.environ.get('FRONTIER_OR_SEED_FITNESS_ID')
    if not root or not identity:
        return None
    return Path(root) / identity / source_digest / f'{stage}.json'
