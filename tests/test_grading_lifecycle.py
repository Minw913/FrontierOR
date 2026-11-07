import json
from pathlib import Path
from types import SimpleNamespace

from trusted_eval_infra.agent import grading_lifecycle as lifecycle


def test_cleanup_selects_exact_checkout_only(monkeypatch, tmp_path):
    target=tmp_path/'checkout'
    def run(args, **kw):
        if args[1]=='ps':return SimpleNamespace(returncode=0,stdout='ours\nother\n')
        ident=args[-1]
        source=str(target/'code.py') if ident=='ours' else str(tmp_path/'checkout-copy/code.py')
        return SimpleNamespace(returncode=0,stdout=json.dumps([{'Name':'/frontieror-candidate-'+ident,'Mounts':[{'Destination':'/workspace/code.py','Source':source}]}]))
    monkeypatch.setattr(lifecycle.subprocess,'run',run)
    assert lifecycle.checkout_containers(target)==['ours']


def test_refreshed_attempt_reads_new_timeout_and_cleans_after_failure(tmp_path, monkeypatch):
    from coral.grader import daemon
    from coral.config import CoralConfig
    path=tmp_path/'config.yaml'
    config=CoralConfig.from_dict({'task':{'name':'test','description':'test fixture'}});config.grader.timeout=4380;config.to_yaml(path)
    seen=[]
    monkeypatch.delattr(daemon,'_frontieror_lifecycle',raising=False)
    monkeypatch.setattr(daemon,'_grade_one',lambda attempt,p,c,cfg:seen.append(cfg.grader.timeout))
    def fail(*args):raise TimeoutError('test timeout')
    monkeypatch.setattr(daemon,'_run_grader_with_timeout',fail)
    monkeypatch.setattr(lifecycle,'cleanup_checkout',lambda checkout:seen.append(checkout))
    lifecycle.install_grading_lifecycle()
    new=CoralConfig.from_yaml(path);new.grader.timeout=33180;new.to_yaml(path)
    daemon._grade_one(None,path,None,config)
    import pytest
    with pytest.raises(TimeoutError):daemon._run_grader_with_timeout(path,None,'exact-checkout',[],33180)
    assert seen==[33180,'exact-checkout']


def test_coordinator_delay_preserves_worker_and_collects_result(tmp_path):
    import subprocess,sys,time,signal
    from trusted_eval_infra.agent.live_eval_watchdog import proc_state,signal_same
    script=tmp_path/'coordinator.py'
    script.write_text('''import multiprocessing,time,sys
from pathlib import Path
child=multiprocessing.Process(target=time.sleep,args=(3,))
child.start()
Path(sys.argv[1]).write_text(str(child.pid))
child.join(timeout=1)
if child.is_alive():
 child.kill();child.join();sys.exit(3)
sys.exit(0)
''')
    pidfile=tmp_path/'worker.pid'
    proc=subprocess.Popen([sys.executable,str(script),str(pidfile)])
    try:
        deadline=time.monotonic()+5
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
        worker=int(pidfile.read_text());state=proc_state(proc.pid)
        assert signal_same(proc.pid,state['start_ticks'],signal.SIGSTOP)
        deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            cs=proc_state(worker)
            if cs is None or cs['state']=='Z':break
            time.sleep(.05)
        assert proc_state(worker)['state']=='Z'
        signal_same(proc.pid,state['start_ticks'],signal.SIGCONT)
        assert proc.wait(timeout=3)==0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGCONT);proc.kill();proc.wait()
