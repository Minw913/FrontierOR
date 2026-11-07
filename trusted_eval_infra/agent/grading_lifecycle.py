"""Refresh grading budgets per attempt and clean containers on grader failure."""
from pathlib import Path
import subprocess
import json
import logging


def checkout_containers(checkout):
    """Select only exact code mounts; never match by a task-name substring."""
    expected = str(Path(checkout).resolve() / 'code.py')
    result = subprocess.run(['docker', 'ps', '-q', '--filter', 'name=frontieror-candidate-'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError('Docker listing failed')
    selected = []
    for ident in result.stdout.split():
        data = subprocess.run(['docker', 'inspect', ident], capture_output=True, text=True, timeout=10)
        if data.returncode:
            continue
        row = json.loads(data.stdout)[0]
        if not row.get('Name', '').lstrip('/').startswith('frontieror-candidate-'):
            continue
        if any(m['Destination'] == '/workspace/code.py' and m['Source'] == expected for m in row.get('Mounts', [])):
            selected.append(ident)
    return selected


def cleanup_checkout(checkout):
    removed = []
    for ident in checkout_containers(checkout):
        result = subprocess.run(['docker', 'rm', '-f', ident], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            removed.append(ident)
    return removed


def install_grading_lifecycle():
    from coral.grader import daemon
    from coral.config import CoralConfig
    if getattr(daemon, '_frontieror_lifecycle', False):
        return
    grade_one = daemon._grade_one
    run = daemon._run_grader_with_timeout

    def refreshed(attempt, config_path, coral_dir, config):
        return grade_one(attempt, config_path, coral_dir, CoralConfig.from_yaml(config_path))

    def guarded(config_path, coral_dir, codebase_path, tasks, timeout):
        try:
            return run(config_path, coral_dir, codebase_path, tasks, timeout)
        finally:
            try:
                cleanup_checkout(codebase_path)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                logging.getLogger(__name__).exception("Candidate cleanup failed; external watchdog must retry")

    daemon._grade_one = refreshed
    daemon._run_grader_with_timeout = guarded
    daemon._frontieror_lifecycle = True
