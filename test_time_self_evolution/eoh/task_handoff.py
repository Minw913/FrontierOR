"""Skip explicitly transferred papers before a fresh EOH child starts work."""
import json
from pathlib import Path


def skip_transferred_paper(root, run_id, paper_ids):
    if not run_id:
        return False
    path = Path(root) / 'eval/eoh' / run_id / 'task-handoff.json'
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    if manifest.get('source_run_id') != run_id:
        raise ValueError('EOH handoff run ID mismatch')
    deferred = set(manifest['transferred_tasks'])
    if len(paper_ids or []) != 1:
        if deferred.intersection(paper_ids or []):
            raise ValueError('Run transferred-task checks separately for each EOH paper')
        return False
    task = paper_ids[0]
    if task not in deferred:
        return False
    print(f'[task-handoff] SKIPPED {task}: reserved for another server; '
          'no model call or evaluation started', flush=True)
    return True
