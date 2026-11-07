import json
import pytest
from trusted_eval_infra import execution
from trusted_eval_infra.wls_gate import can_admit


def test_admission_counts_backend_and_unobserved_reservations():
    assert can_admit(2, 0, 1)
    assert not can_admit(2, 1, 1)
    assert not can_admit(2, 2, 0)
    assert not can_admit(0, 0, 0)


def test_gate_rejects_unlimited_slots(tmp_path, monkeypatch):
    p=tmp_path/'gurobi.lic';p.write_text('LICENSEID=1\nWLSACCESSID=test\nWLSSECRET=test\n')
    with pytest.raises(ValueError, match='positive'):
        with execution._wls_execution_slot({'gurobi_lic':str(p),'wls_backend_gate':1,'wls_concurrency':0}):
            pytest.fail('must not start')


def test_backend_gate_fail_closed(tmp_path, monkeypatch):
    from trusted_eval_infra import wls_gate
    p=tmp_path/'gurobi.lic';p.write_text('LICENSEID=1\nWLSACCESSID=test\nWLSSECRET=test\n')
    monkeypatch.setenv('XDG_RUNTIME_DIR',str(tmp_path))
    monkeypatch.setattr(wls_gate,'capacity',lambda fields, **kwargs: (_ for _ in ()).throw(OSError('unavailable')))
    monkeypatch.setattr(execution.time,'sleep',lambda seconds: None)
    with pytest.raises(RuntimeError,match='queue timed out'):
        with execution._wls_execution_slot({'gurobi_lic':str(p),'wls_backend_gate':1,'wls_concurrency':2,'wls_queue_timeout_seconds':0.03}):
            pytest.fail('must not start on backend failure')


def test_capacity_deduplicates_environments_and_ignores_expired_tokens(monkeypatch):
    import io
    from trusted_eval_infra import wls_gate
    monkeypatch.setattr(wls_gate.time, 'time', lambda: 100)
    payloads = {
        'licenses': [{'id': 1, 'maxContainers': 2}],
        'sessions': [{'info': {'licenseId': 1}, 'containerId': 'a', 'startedAt': 90000, 'endedAt': 110000}],
        'tokens?length=5000': [
            {'licenseId': 1, 'containerId': 'a', 'expiresAt': 110000, 'hostname': 'candidate-a'},
            {'licenseId': 1, 'containerId': 'a', 'expiresAt': 120000},
            {'licenseId': 1, 'containerId': 'expired', 'expiresAt': 99000},
            {'licenseId': 2, 'containerId': 'other-license', 'expiresAt': 120000},
        ],
    }
    monkeypatch.setattr(wls_gate.urllib.request, 'urlopen', lambda req, timeout: io.BytesIO(json.dumps(payloads[req.full_url.removeprefix(wls_gate.BASE)]).encode()))
    assert wls_gate.capacity({'LICENSEID': '1', 'WLSACCESSID': 'test', 'WLSSECRET': 'test'}) == (2, 1)
    assert wls_gate.capacity({'LICENSEID': '1', 'WLSACCESSID': 'test', 'WLSSECRET': 'test'},
                             include_environments=True) == (2, {'a'}, {'candidate-a': {'a'}})


def test_identity_matching_is_conservative():
    from trusted_eval_infra.wls_gate import unobserved_reservations
    hosts = {'local': {'env-1'}, 'ambiguous': {'env-2', 'env-3'}}
    assert unobserved_reservations(['local'], hosts) == 0
    assert unobserved_reservations([None, 'unknown', 'ambiguous'], hosts) == 3
    assert unobserved_reservations(['local', 'local'], hosts) == 1
    assert can_admit(2, 1, unobserved_reservations(['local'], hosts))
    assert not can_admit(2, 2, unobserved_reservations(['local'], hosts))


def test_two_local_candidates_share_backend_count_without_double_charge(tmp_path, monkeypatch):
    from trusted_eval_infra import wls_gate
    p = tmp_path / 'gurobi.lic'
    p.write_text('LICENSEID=1\nWLSACCESSID=test\nWLSSECRET=test\n')
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    backend = {}
    monkeypatch.setattr(wls_gate, 'capacity', lambda fields, **kwargs:
                        (2, set().union(*backend.values()) if backend else set(), dict(backend)))
    cfg = {'gurobi_lic': str(p), 'wls_backend_gate': 1, 'wls_concurrency': 2,
           'wls_queue_timeout_seconds': 0.05, '_wls_candidate_hostname': 'local-1'}
    with execution._wls_execution_slot(cfg):
        backend['local-1'] = {'env-1'}
        with execution._wls_execution_slot({**cfg, '_wls_candidate_hostname': 'local-2'}) as slot:
            assert slot == 1


def test_foreign_environment_still_blocks_second_candidate(tmp_path, monkeypatch):
    from trusted_eval_infra import wls_gate
    p = tmp_path / 'gurobi.lic'
    p.write_text('LICENSEID=1\nWLSACCESSID=test\nWLSSECRET=test\n')
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    monkeypatch.setattr(execution.time, 'sleep', lambda _: None)
    backend = {}
    monkeypatch.setattr(wls_gate, 'capacity', lambda fields, **kwargs:
                        (2, set().union(*backend.values()) if backend else set(), dict(backend)))
    cfg = {'gurobi_lic': str(p), 'wls_backend_gate': 1, 'wls_concurrency': 2,
           'wls_queue_timeout_seconds': 0.03, '_wls_candidate_hostname': 'local-1'}
    with execution._wls_execution_slot(cfg):
        backend.update({'local-1': {'env-1'}, 'foreign': {'env-foreign'}})
        with pytest.raises(RuntimeError, match='queue timed out'):
            with execution._wls_execution_slot({**cfg, '_wls_candidate_hostname': 'local-2'}):
                pytest.fail('must not exceed capacity')
