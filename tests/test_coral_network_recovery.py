import json
from types import SimpleNamespace
import pytest
from trusted_eval_infra.docker_networks import choose_subnet,create_internal_network
from trusted_eval_infra.agent.startup_cleanup import install_startup_cleanup


def test_subnet_does_not_overlap_docker_or_host_routes():
    assert choose_subnet(['10.240.0.0/28','10.240.0.16/28'])=='10.240.0.32/28'
    with pytest.raises(RuntimeError):choose_subnet(['10.0.0.0/8'])


def test_network_creation_is_small_internal_and_reports_stderr(tmp_path,monkeypatch):
    import trusted_eval_infra.docker_networks as networks
    monkeypatch.setenv('XDG_RUNTIME_DIR',str(tmp_path))
    monkeypatch.setattr(networks.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='[{"dst":"10.128.0.1"}]'))
    calls=[]
    def docker(args):
        calls.append(args)
        if args[:2]==['network','ls']:return SimpleNamespace(returncode=0,stdout='abc')
        if args[:2]==['network','inspect']:return SimpleNamespace(returncode=0,stdout=json.dumps([{'IPAM':{'Config':[{'Subnet':'10.240.0.0/28'}]}}]))
        return SimpleNamespace(returncode=1,stderr='diagnostic failure')
    with pytest.raises(RuntimeError,match='diagnostic failure'):create_internal_network('frontieror-agent-test',docker)
    create=calls[-1]
    assert '--internal' in create and create[create.index('--subnet')+1]=='10.240.0.16/28'


@pytest.mark.parametrize('method',['start_all','resume_all'])
def test_failed_startup_stops_grader_and_reraises(method):
    class Manager:
        def start_all(self):raise RuntimeError('network setup failed')
        def resume_all(self):raise RuntimeError('network setup failed')
        def stop_all(self):self.cleaned=True
    install_startup_cleanup(Manager);install_startup_cleanup(Manager)
    manager=Manager();manager.cleaned=False
    with pytest.raises(RuntimeError,match='network setup failed'):getattr(manager,method)()
    assert manager.cleaned


def test_native_grader_is_reaped_after_startup_exception():
    import multiprocessing
    from coral.agent.manager import AgentManager
    class Manager:
        _stop_grader_daemon=AgentManager._stop_grader_daemon
        def start_all(self):
            self._grader_stop_event=multiprocessing.Event()
            self._grader_proc=multiprocessing.Process(target=self._grader_stop_event.wait,args=(60,))
            self._grader_proc.start()
            self.child_pid=self._grader_proc.pid
            raise RuntimeError('network setup failed')
        resume_all=start_all
        def stop_all(self):self._stop_grader_daemon(timeout=2)
    install_startup_cleanup(Manager)
    manager=Manager();manager.paths=None
    with pytest.raises(RuntimeError):manager.start_all()
    assert manager._grader_proc is None
    assert manager.child_pid not in [p.pid for p in multiprocessing.active_children()]
