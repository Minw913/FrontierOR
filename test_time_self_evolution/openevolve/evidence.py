"""Immutable, per-invocation evaluation evidence for native OpenEvolve."""
from __future__ import annotations
import contextvars
import contextlib
import datetime
import fcntl
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from trusted_eval_infra.files import read_regular_file

_CURRENT = contextvars.ContextVar('openevolve_evidence', default=None)


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_number': str(value)}
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(_safe(value), indent=2, default=str, allow_nan=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def output_directory(base, stage):
    current = _CURRENT.get()
    return str(current['directory']) if current else str(Path(base) / stage)


def capture_results(results):
    current = _CURRENT.get()
    if current:
        _write(current['directory'] / 'results.json', results)


def attach_artifacts(artifacts):
    current = _CURRENT.get()
    if current:
        artifacts = {**artifacts, 'evaluation_id': current['id'],
                     f"{current['stage']}_evaluation_id": current['id']}
        current['artifacts'] = artifacts
    return artifacts


def isolated_evaluation(stage, environment_reader):
    def decorate(function):
        @functools.wraps(function)
        def evaluate(program_path):
            paper, model, backend, base, _, execution = environment_reader()
            code = read_regular_file(Path(program_path), max_bytes=4 * 1024 * 1024,
                                     label='OpenEvolve candidate source')
            digest = hashlib.sha256(code).hexdigest()
            ident = uuid.uuid4().hex
            directory = Path(base) / stage / digest / ident
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            source = directory / 'code.py'
            source.write_bytes(code)
            source.chmod(0o600)
            record = {'schema_version': 1, 'adapter_version': 'feedback-evidence-v2-20261102',
                      'evaluation_id': ident, 'stage': stage,
                      'paper_id': paper, 'model_name': model, 'backend': backend,
                      'source_path': str(Path(program_path).resolve()), 'source_sha256': digest,
                      'started_at': datetime.datetime.now(datetime.UTC).isoformat(), 'status': 'running',
                      'execution': {k: v for k, v in execution.items() if k in {
                          'cpus', 'memory', 'anti_hack', 'systemd_allow_wls', 'wls_egress',
                          'docker_image', 'docker_image_ref'}},
                      'settings': {k: v for k, v in os.environ.items() if k.startswith((
                          'EFFICIENT_OR_STAGE1_', 'EFFICIENT_OR_STAGE2_'))}}
            _write(directory / 'evaluation.json', record)
            current = {'id': ident, 'stage': stage, 'directory': directory, 'artifacts': {}}
            token = _CURRENT.set(current)
            try:
                from test_time_self_evolution.seed_fitness import cache_path
                normalized_digest = hashlib.sha256(code.decode('utf-8').strip().encode()).hexdigest()
                cached_path = cache_path(normalized_digest, stage)
                with contextlib.ExitStack() as stack:
                    if cached_path is not None:
                        cached_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        lock = stack.enter_context(cached_path.with_suffix('.lock').open('a'))
                        fcntl.flock(lock, fcntl.LOCK_EX)
                    cached = None
                    if cached_path is not None and cached_path.exists():
                        try:
                            cached = json.loads(cached_path.read_text())
                            if (not isinstance(cached, dict)
                                    or not isinstance(cached.get('metrics'), dict)
                                    or not isinstance(cached.get('results'), dict)
                                    or not isinstance(cached.get('artifacts'), dict)
                                    or not isinstance(cached.get('evaluation_id'), str)):
                                cached = None
                        except (ValueError, OSError):
                            pass
                    if cached is not None:
                        from test_time_self_evolution.openevolve import evaluator
                        artifacts = {k: v for k, v in cached['artifacts'].items()
                                     if not k.endswith('evaluation_id')}
                        artifacts['seed_fitness_cache_source'] = cached['evaluation_id']
                        artifacts = attach_artifacts(artifacts)
                        capture_results(cached['results'])
                        metrics = dict(cached['metrics'])
                        for key in list(metrics):
                            if key.endswith('evaluation_id'):
                                metrics[key] = ident
                        result = (evaluator.EvaluationResult(metrics=metrics, artifacts=artifacts)
                                  if evaluator._HAS_EVALUATION_RESULT else metrics)
                        record['seed_fitness_cache_hit'] = True
                        record['seed_fitness_cache_source'] = cached['evaluation_id']
                    else:
                        result = function(str(source))
                        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                            raise RuntimeError('Evaluated candidate source changed during execution')
                        results_path = directory / 'results.json'
                        metrics = getattr(result, 'metrics', result)
                        if (cached_path is not None and results_path.exists()
                                and not any(k.endswith('_error') for k in metrics)):
                            _write(cached_path, dict(metrics=metrics, artifacts=current['artifacts'],
                                                    results=json.loads(results_path.read_text()),
                                                    evaluation_id=ident))
                if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Evaluated candidate source changed during execution')
                _write(directory / 'metrics.json', getattr(result, 'metrics', result))
                _write(directory / 'artifacts.json', current['artifacts'])
                record['status'] = 'completed'
                return result
            except BaseException as error:
                record.update(status='failed', error_type=type(error).__name__)
                raise
            finally:
                record['finished_at'] = datetime.datetime.now(datetime.UTC).isoformat()
                try:
                    _write(directory / 'evaluation.json', record)
                finally:
                    _CURRENT.reset(token)
        return evaluate
    return decorate
