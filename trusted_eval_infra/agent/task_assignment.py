"""Optional host-owned assignment for a live hardset handoff."""
import json
from pathlib import Path
import datetime
import os

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT/'local_docs/coral-runs/hardset-license-assignment-20261030.json'


def assigned_elsewhere(base_dir, paper_id, manifest_path=None):
    path = Path(manifest_path or os.environ.get('FRONTIER_OR_TASK_ASSIGNMENT', DEFAULT_MANIFEST))
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    base = Path(base_dir).resolve()
    eval_root = ROOT/'eval/coral'
    if not base.is_relative_to(eval_root):
        return None
    run = base.relative_to(eval_root).parts[0]
    if run not in manifest['managed_runs']:
        return None
    # New experimental cohorts may hand off tasks without overwriting the
    # historical owners used to select published result rows.
    owners = manifest.get('run_owners', {}).get(run, manifest['owners'])
    owner = owners.get(paper_id)
    return owner if owner and owner != run else None


def guard_native_start(argv):
    """Protect already-enqueued old parent processes, which cannot reload code."""
    if not argv or argv[0] != 'start' or '-c' not in argv:
        return
    import yaml
    config = yaml.safe_load(Path(argv[argv.index('-c')+1]).read_text())
    args = config.get('grader', {}).get('args', {})
    if not args.get('base_dir') or not args.get('paper_id'):
        return
    owner = assigned_elsewhere(args['base_dir'], args['paper_id'])
    if owner:
        event = {'at':datetime.datetime.now(datetime.UTC).isoformat(), 'task':args['paper_id'],
                 'source_base':args['base_dir'], 'owner_run':owner, 'reason':'intentional_license_handoff',
                 'agent_started':False, 'evaluation_started':False}
        folder=ROOT/'local_docs/coral-runs/task-handoffs-20261030'
        folder.mkdir(exist_ok=True)
        (folder/f"{args['paper_id']}.json").write_text(json.dumps(event,indent=2)+'\n')
        print(f"[task-transfer] SKIPPED {args['paper_id']}: assigned to {owner}; no agent or evaluation started", flush=True)
        raise SystemExit(78)
