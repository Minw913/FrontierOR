"""Unscored public-tiny execution; never calls a grader or writes attempts."""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from trusted_eval_infra.agent.broker import _atomic_json, _git, _validate_request, trusted_seed_blobs
from trusted_eval_infra.files import read_regular_file

DEBUG_SECONDS = 30
MAX_DEBUG_RUNS = 30  # Separate per-task quota, also bounded by task wall time.


class DebugRunner:
    def __init__(self, task):
        self.task = task
        self.directory = Path(task.coral_dir) / 'private' / 'debug'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.responses = Path(task.coral_dir) / 'public' / 'request_status'
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='coral-debug')
        self.pending = {}

    def submit(self, request, filename, allowed_agents):
        self.poll()
        normalized = {k: v for k, v in request.items() if k != 'kind'}
        valid, _, _ = _validate_request(
            Path(self.task.repo_dir), normalized, allowed_agent_ids=allowed_agents,
            request_name=filename,
            trusted_files=trusted_seed_blobs(self.task.seed_dir),
        )
        reason = None
        if not valid:
            reason = 'invalid debug request'
        elif self.pending:
            reason = 'a debug execution is already pending; wait before retrying'
        elif sum(1 for _ in self.directory.glob('*.json')) >= MAX_DEBUG_RUNS:
            reason = 'separate debug budget exhausted'
        if reason:
            payload = {'status': 'rejected', 'success': False, 'reason': reason, 'counts_as_attempt': False}
        else:
            nonce = request['nonce']
            # Trusted persistent admission record, including the immutable code commit.
            _atomic_json(self.directory / f'{nonce}.json', {'request': request, 'status': 'running'})
            self.pending[nonce] = self.pool.submit(self._execute, normalized)
            payload = {'status': 'running', 'counts_as_attempt': False}
        _atomic_json(self.responses / filename, payload)
        with (self.directory / 'admissions.jsonl').open('a') as audit:
            audit.write(json.dumps({'request': request, 'accepted': reason is None, 'reason': reason}) + '\n')
        return reason is None

    def _execute(self, request):
        from trusted_eval_infra.execution import run_docker
        config = yaml.safe_load(Path(self.task.config_path).read_text())
        args = config['grader']['args']
        task_id = args['paper_id']
        # Fixed trusted path: requests cannot select final tests or arbitrary files.
        instance = Path(self.task.seed_dir) / task_id / 'instance' / 'tiny_instance.json'
        if not instance.is_file():
            return {'status': 'completed', 'success': False, 'reason': 'public tiny instance unavailable', 'counts_as_attempt': False}
        code = _git(Path(self.task.repo_dir), 'show', f"{request['commit_hash']}:code.py")
        if code.returncode:
            raise RuntimeError('debug source unavailable')
        cfg = dict(args['exec_cfg'])
        cfg['anti_hack'] = True
        # Explicit empty license prevents fallback to host GRB_LICENSE_FILE.
        # Debug cannot acquire a licensed WLS environment.
        cfg['gurobi_lic'] = ''
        cfg['wls_egress'] = 'off'
        cfg.pop('_restricted_network', None)
        cfg.pop('_restricted_proxy', None)
        with tempfile.TemporaryDirectory(prefix='frontieror-debug-') as directory:
            root = Path(directory)
            (root / 'code.py').write_text(code.stdout)
            helper = Path(self.task.seed_dir) / 'solution_logger.py'
            if helper.is_file():
                (root / helper.name).write_bytes(read_regular_file(helper, max_bytes=1024 * 1024, label='public logger'))
            for name in ('solution.json', 'log.jsonl'):
                (root / name).touch()
            ok, output, elapsed = run_docker(
                str(root / 'code.py'), str(instance), str(root / 'solution.json'),
                DEBUG_SECONDS, str(root / 'log.jsonl'), cfg,
            )
            result = {
                'status': 'completed', 'success': ok, 'counts_as_attempt': False,
                'commit_hash': request['commit_hash'], 'instance': 'tiny',
                'time_limit_seconds': DEBUG_SECONDS, 'elapsed_seconds': elapsed,
                'output': output[-8192:],
                'note': 'Execution only; no checker, score, reference objective or feasibility verdict. Debug has no Gurobi license or network access. Use coral eval for licensed Gurobi execution. Debug consumes compute and task wall time.',
            }
            try:
                raw = read_regular_file(root / 'solution.json', max_bytes=64 * 1024, label='debug solution')
                result['solution'] = json.loads(raw) if raw else None
            except (OSError, ValueError, RuntimeError):
                result['solution_note'] = 'No valid solution within the 64 KiB debug preview limit.'
            return result

    def poll(self):
        for nonce, future in list(self.pending.items()):
            if not future.done():
                continue
            try:
                result = future.result()
            except Exception as exc:
                result = {'status': 'completed', 'success': False, 'counts_as_attempt': False,
                          'reason': 'debug infrastructure error; consult host audit'}
                record_path = self.directory / f'{nonce}.json'
                record = json.loads(record_path.read_text())
                record['infrastructure_error_type'] = type(exc).__name__
                _atomic_json(record_path, record)
            audit = self.directory / f'{nonce}.json'
            record = json.loads(audit.read_text())
            record['result'] = result
            record['status'] = 'completed'
            _atomic_json(audit, record)
            _atomic_json(self.responses / f'{nonce}.json', result)
            del self.pending[nonce]

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.poll()
