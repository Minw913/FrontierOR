"""Online compatibility for pre-fix grader coordinators; agents/solvers keep running.

SIGSTOP applies ONLY to an old coordinator while its independent worker runs.
This delays the old join timeout; when the worker exits we SIGCONT the coordinator
so native code collects and scores its result. New coordinators need no protection.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def proc_state(pid):
    try:
        values = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': values[0], 'parent': int(values[1]), 'start_ticks': int(values[19])}
    except (OSError, ValueError, IndexError):
        return None


def age_seconds(state):
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    return uptime - state['start_ticks'] / os.sysconf('SC_CLK_TCK')


def signal_same(pid, start_ticks, sig):
    state = proc_state(pid)
    if state and state['start_ticks'] == start_ticks:
        os.kill(pid, sig)
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    out = manifest_path.parent
    held = {}
    if args.release:
        path = out/'held.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        for pid, state in previous.items():
            signal_same(int(pid), state['start_ticks'], signal.SIGCONT)
        path.write_text('{}\n')
        return
    def save(name, value):
        path = out / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)
    def event(kind, **data):
        with (out/'events.jsonl').open('a') as file:
            file.write(json.dumps({'at':datetime.datetime.now(datetime.UTC).isoformat(),'event':kind,**data})+'\n')
    def release_all(*_):
        for pid, state in list(held.items()):
            signal_same(pid, state['start_ticks'], signal.SIGCONT)
        held.clear()
        save('held.json', {})
    def terminate(*_):
        release_all()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        while True:
            for row in manifest['coordinators']:
                pid=row['pid'];state=proc_state(pid)
                if not state or state['start_ticks']!=row['start_ticks']:
                    continue
                children=Path(f'/proc/{pid}/task/{pid}/children')
                workers=[]
                for text in children.read_text().split():
                    child=int(text);cs=proc_state(child)
                    if cs and cs['parent']==pid:
                        workers.append((child,cs))
                live=[(child,cs) for child,cs in workers if cs['state']!='Z']
                if pid in held:
                    entry=held[pid]
                    worker=proc_state(entry['worker_pid'])
                    done=not worker or worker['start_ticks']!=entry['worker_ticks'] or worker['state']=='Z'
                    expired=worker is not None and age_seconds(worker)>row['new_timeout']+30
                    if done or expired:
                        signal_same(pid,row['start_ticks'],signal.SIGCONT)
                        event('coordinator_released',task=row['task'],worker_done=done,new_deadline_expired=expired)
                        held.pop(pid,None);save('held.json',held)
                elif live:
                    worker,cs=live[0]
                    if age_seconds(cs)>=row['old_timeout']-120 and age_seconds(cs)<row['new_timeout']:
                        if signal_same(pid,row['start_ticks'],signal.SIGSTOP):
                            held[pid]={'start_ticks':row['start_ticks'],'worker_pid':worker,'worker_ticks':cs['start_ticks']}
                            save('held.json',held)
                            event('old_coordinator_deadline_deferred',task=row['task'],worker_pid=worker)
            # Remove only candidates whose exact run/task/commit attempt is already terminal.
            result=subprocess.run(['docker','ps','-q','--filter','name=frontieror-candidate-'],capture_output=True,text=True,timeout=10)
            if result.returncode==0:
                for ident in result.stdout.split():
                    inspected=subprocess.run(['docker','inspect',ident],capture_output=True,text=True,timeout=10)
                    if inspected.returncode:continue
                    info=json.loads(inspected.stdout)[0]
                    code=next((m['Source'] for m in info.get('Mounts',[]) if m['Destination']=='/workspace/code.py'),None)
                    if not code:continue
                    for run in manifest['run_dirs']:
                        root=Path(run).resolve()
                        path=Path(code)
                        if not path.is_relative_to(root):continue
                        parts=path.relative_to(root).parts
                        # task/model/coral_run/.coral/private/grader_checkouts/commit/code.py
                        if len(parts)!=8 or parts[2:6]!=('coral_run','.coral','private','grader_checkouts') or parts[-1]!='code.py':continue
                        attempt=root/parts[0]/parts[1]/'coral_run/.coral/public/attempts'/f'{parts[6]}.json'
                        try:record=json.loads(attempt.read_text())
                        except (OSError,ValueError):continue
                        if record.get('commit_hash')!=parts[6]:continue
                        terminal=record.get('status') not in (None,'pending')
                        command=info.get('Config',{}).get('Cmd',[])
                        overdue=False
                        if '--time_limit' in command:
                            limit=float(command[command.index('--time_limit')+1])
                            started=datetime.datetime.fromisoformat(info['State']['StartedAt'].replace('Z','+00:00'))
                            overdue=(datetime.datetime.now(datetime.UTC)-started).total_seconds()>limit+60
                        if not terminal and not overdue:continue
                        removed=subprocess.run(['docker','rm','-f',ident],capture_output=True,text=True,timeout=15)
                        if removed.returncode==0:event('candidate_container_removed',task=parts[0],commit=parts[6],container=ident,status=record['status'],reason='terminal_attempt' if terminal else 'candidate_wall_deadline')
            save('status.json',{'at':datetime.datetime.now(datetime.UTC).isoformat(),'phase':'monitoring','held_coordinators':len(held)})
            time.sleep(2)
    finally:
        release_all()


if __name__=='__main__':
    main()
