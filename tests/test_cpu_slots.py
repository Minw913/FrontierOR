import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import pytest

from trusted_eval_infra.cpu_slots import reserve_cpus


def hold_cpu(folder, affinity, ready, release, count=1):
    os.sched_setaffinity(0, affinity)
    with reserve_cpus(count, timeout=5, lock_dir=folder) as cores:
        ready.put(cores)
        release.wait(10)


def test_spawned_processes_hold_distinct_cpus_and_release_on_exit(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) < 4:
        pytest.skip('Needs four logical CPUs')
    ctx = mp.get_context('spawn'); ready = ctx.Queue(); release = ctx.Event()
    workers = [ctx.Process(target=hold_cpu, args=(str(tmp_path), cpus, ready, release)) for _ in cpus]
    try:
        for p in workers: p.start()
        held = [ready.get(timeout=10) for _ in cpus]
        assert len(set(held)) == 4
        with pytest.raises(TimeoutError):
            # Restrict this process so it cannot claim a fifth free CPU.
            old = os.sched_getaffinity(0)
            try:
                os.sched_setaffinity(0, cpus)
                with reserve_cpus(1, timeout=0.1, lock_dir=tmp_path): pass
            finally: os.sched_setaffinity(0, old)
        release.set()
        for p in workers: p.join(10); assert p.exitcode == 0
        with reserve_cpus(4, timeout=0.1, lock_dir=tmp_path) as cores:
            assert len(set(cores.split(','))) == 4
    finally:
        release.set()
        for p in workers:
            if p.is_alive(): p.kill(); p.join()


def test_killed_owner_releases_lease_and_multi_cpu_timeout_leaves_no_partial_lock(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) < 2: pytest.skip('Needs two CPUs')
    ctx = mp.get_context('spawn'); ready = ctx.Queue(); release = ctx.Event()
    p = ctx.Process(target=hold_cpu, args=(str(tmp_path), [cpus[1]], ready, release))
    old = os.sched_getaffinity(0)
    try:
        p.start(); assert ready.get(timeout=10) == str(cpus[1])
        os.sched_setaffinity(0, cpus)
        with pytest.raises(TimeoutError):
            with reserve_cpus(2, timeout=0.1, lock_dir=tmp_path): pass
        with reserve_cpus(1, timeout=0, lock_dir=tmp_path) as core:
            assert core == str(cpus[0])
        p.kill(); p.join(10)
        with reserve_cpus(2, timeout=0.2, lock_dir=tmp_path) as cores:
            assert set(map(int, cores.split(','))) == set(cpus)
    finally:
        os.sched_setaffinity(0, old)
        if p.is_alive(): p.kill(); p.join()


def test_invalid_request(tmp_path):
    for n in [0, -1, True, 1.5, len(os.sched_getaffinity(0)) + 1]:
        with pytest.raises(ValueError):
            with reserve_cpus(n, lock_dir=tmp_path): pass


def run_systemd_probe(folder, ready, start):
    from trusted_eval_infra.execution import run_systemd
    root = Path(folder); ready.put(os.getpid()); start.wait(10)
    ok, output, elapsed = run_systemd(str(root/'code.py'), str(root/'instance.json'),
                                     str(root/'solution.json'), 10,
                                     cfg={'cpus': 1, 'memory': '128M', 'wls_egress': 'off'})
    if not ok: raise RuntimeError(output)


def test_real_systemd_candidates_in_independent_processes_do_not_share_cpu(tmp_path):
    if len(os.sched_getaffinity(0)) < 4: pytest.skip('Needs four CPUs')
    import subprocess
    if subprocess.run(['systemctl','--user','show-environment'], capture_output=True).returncode:
        pytest.skip('No user systemd')
    code = '''import argparse,json,os,time
p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit');p.add_argument('--log_path');a=p.parse_args()
start=time.time();affinity=sorted(os.sched_getaffinity(0));time.sleep(2)
json.dump({'cpus':affinity,'start':start,'end':time.time()},open(a.solution_path,'w'))
'''
    ctx = mp.get_context('spawn'); ready = ctx.Queue(); start = ctx.Event(); workers=[]
    try:
        for i in range(4):
            folder=tmp_path/str(i);folder.mkdir();(folder/'code.py').write_text(code);(folder/'instance.json').write_text('{}')
            p=ctx.Process(target=run_systemd_probe,args=(str(folder),ready,start));p.start();workers.append(p)
        for p in workers: ready.get(timeout=10)
        start.set()
        for p in workers: p.join(20); assert p.exitcode==0
        rows=[json.loads((tmp_path/str(i)/'solution.json').read_text()) for i in range(4)]
        assert max(r['start'] for r in rows) < min(r['end'] for r in rows)
        assert len({tuple(r['cpus']) for r in rows})==4
        for i,r in enumerate(rows):
            evidence=json.loads((tmp_path/str(i)/'solution.json.cpu.json').read_text())
            assert evidence['cpus']==r['cpus'] and evidence['released_at'] is not None
    finally:
        start.set()
        for p in workers:
            if p.is_alive(): p.kill();p.join()


@pytest.mark.parametrize('framework', ['eoh', 'openevolve'])
def test_self_evolve_forwards_systemd_wls_flag(framework, monkeypatch):
    from test_time_self_evolution import run_eval_modes as cli
    from test_time_self_evolution.openevolve import evaluator
    args=cli.parse_args(['--framework', framework, '--paper-id','task',
                         '--dev-set','large_1','--test-set','large_2',
                         '--exec-mode','systemd','--systemd-allow-wls'])
    target=cli.eoh_runner if framework=='eoh' else cli.openevolve_runner
    monkeypatch.setattr(target,'run_self_evolve',lambda **kw: kw)
    result=cli.run_self_evolve_mode(args,'run',{},'task','prompt','model','model')
    assert result['exec_cfg']['systemd_allow_wls'] is True
    for k,v in result['exec_cfg'].items():
        monkeypatch.setenv('EFFICIENT_OR_EXEC_'+k.upper(),str(v))
    monkeypatch.setenv('EFFICIENT_OR_PAPER_ID','task')
    monkeypatch.setenv('EFFICIENT_OR_MODEL_NAME','model')
    assert evaluator._common_env()[-1]['systemd_allow_wls'] is True
