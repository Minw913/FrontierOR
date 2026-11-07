import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import pytest

from test_time_self_evolution import eval_modes, seeds
from test_time_self_evolution.coral import runner as coral
from test_time_self_evolution.eoh import runner as eoh
from test_time_self_evolution.openevolve import runner as openevolve


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def fake_eoh(monkeypatch):
    class EVOL:
        def __init__(self, paras):
            random.seed(2024)
            np.random.seed(2024)

    class EOH(EVOL):
        pass

    class Evolution:
        def i1(self):
            return ['generated', 'LLM']

    module = types.ModuleType('eoh')
    module.eoh = types.SimpleNamespace(EVOL=EVOL)
    monkeypatch.setitem(sys.modules, 'eoh', module)
    monkeypatch.setitem(sys.modules, 'eoh.methods.eoh.eoh', types.SimpleNamespace(EOH=EOH))
    monkeypatch.setitem(sys.modules, 'eoh.methods.eoh.eoh_evolution', types.SimpleNamespace(Evolution=Evolution))
    return EVOL, EOH, Evolution


@pytest.mark.parametrize('framework', ['coral', 'openevolve', 'eoh'])
def test_all_frameworks_choose_final_snapshot_not_attempt_zero(tmp_path, monkeypatch, framework):
    source = tmp_path/'eval/seed_programs/task/model'
    write(source/'code.py', "print('final')\n")
    write(source/'code_attempt0.py', "print('first')\n")
    write(source/'solution_logger.py', '# helper\n')
    write(tmp_path/'eval/eval_tasks/task/model/code.py', "print('stale runtime')\n")
    out = tmp_path/'run'
    module = dict(coral=coral, openevolve=openevolve, eoh=eoh)[framework]
    monkeypatch.setattr(module, 'ROOT_DIR', str(tmp_path))
    monkeypatch.delenv('FRONTIER_OR_CORAL_SEED_MODEL', raising=False)
    if framework == 'eoh':
        _, _, evolution = fake_eoh(monkeypatch)
        assert eoh.patch_eoh_i1_with_oneshot_seed('task', 'model', str(out))
        destination = out/'seed_oneshot'
        assert evolution().i1()[0] == "print('final')\n"
        assert evolution().i1()[0] == 'generated'
    else:
        assert module._try_reuse_oneshot_seed('task', 'model', str(out)) == str(out/'code.py')
        destination = out
    assert (destination/'code.py').read_text() == "print('final')\n"
    assert (destination/'solution_logger.py').read_text() == '# helper\n'
    provenance = json.loads((destination/'_seed_source.json').read_text())
    assert provenance['seed_version'] == 'final_one_shot'
    assert provenance['source'] == 'eval/seed_programs/task/model/code.py'
    assert provenance['sha256'] == hashlib.sha256((source/'code.py').read_bytes()).hexdigest()


def test_never_substitute_attempt_zero(tmp_path):
    write(tmp_path/'eval/eval_tasks/task/model/code_attempt0.py', 'print(0)\n')
    assert seeds.copy_final_oneshot_seed(str(tmp_path), 'task', 'model', str(tmp_path/'seed')) is None


def test_runtime_final_and_grok_alias(tmp_path):
    write(tmp_path/'eval/eval_tasks/task/grok-4.20-beta/code.py', 'print(1)\n')
    out = tmp_path/'seed'
    assert seeds.copy_final_oneshot_seed(str(tmp_path), 'task', 'grok-4.20', str(out))
    assert (out/'code.py').read_text() == 'print(1)\n'


def test_manifest_mismatch_rejected_without_downgrade(tmp_path):
    write(tmp_path/'eval/seed_programs/task/model/code.py', 'print(1)\n')
    write(tmp_path/'eval/seed_programs/manifest.json', json.dumps(dict(programs=[dict(
        paper_id='task', model='model', files={'code.py': {'sha256': 'wrong'}})])))
    write(tmp_path/'eval/eval_tasks/task/model/code.py', 'print(2)\n')
    assert seeds.copy_final_oneshot_seed(str(tmp_path), 'task', 'model', str(tmp_path/'seed')) is None
    assert not (tmp_path/'seed/code.py').exists()


def test_manifest_hashes_and_helpers(tmp_path):
    code = write(tmp_path/'eval/seed_programs/task/model/code.py', 'print(1)\n')
    helper = write(code.parent/'solution_logger.py', '# helper\n')
    files = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (code, helper)}
    write(tmp_path/'eval/seed_programs/manifest.json', json.dumps(dict(programs=[dict(
        paper_id='task', model='model', files=files)])))
    assert seeds.copy_final_oneshot_seed(str(tmp_path), 'task', 'model', str(tmp_path/'seed'))


def test_symlink_is_not_followed(tmp_path):
    source = write(tmp_path/'private.py', 'secret')
    code = tmp_path/'eval/seed_programs/task/model/code.py'
    code.parent.mkdir(parents=True)
    code.symlink_to(source)
    assert seeds.copy_final_oneshot_seed(str(tmp_path), 'task', 'model', str(tmp_path/'seed')) is None


def test_status_uses_final_not_first_attempt(tmp_path, monkeypatch):
    path = tmp_path/'eval/eval_results_model.csv'
    path.parent.mkdir()
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['paper_id', 'model', 'instance', 'status', 'first_status'])
        writer.writeheader()
        writer.writerow(dict(paper_id='task', model='model', instance='large_1', status='pass', first_status='fail'))
    monkeypatch.setattr(eval_modes, 'ROOT_DIR', str(tmp_path))
    assert eval_modes._load_oneshot_seed_status('task', 'model', 'large_1') == 'pass'


def test_eoh_rng_survives_hardcoded_constructor_and_repatch(monkeypatch):
    evol, method, _ = fake_eoh(monkeypatch)
    for chosen in (17, 29, 17):
        eoh.patch_eoh_rng_seed(chosen)
        for cls in (evol, method):
            cls(types.SimpleNamespace(ec_seed=chosen))
            assert random.random() == random.Random(chosen).random()
            assert np.random.random() == np.random.RandomState(chosen).random_sample()
    # A per-run Paras value takes precedence over the latest patch closure.
    evol(types.SimpleNamespace(ec_seed=42))
    assert random.random() == random.Random(42).random()


def test_eoh_rng_none_is_noop_and_invalid_seed_rejected(monkeypatch):
    monkeypatch.delitem(sys.modules, 'eoh', raising=False)
    random.seed(99)
    eoh.patch_eoh_rng_seed(None)
    assert random.random() == random.Random(99).random()
    with pytest.raises(ValueError, match='EoH seed'):
        eoh._set_eoh_rng_seed(-1)


def test_seed_cli_help():
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'test_time_self_evolution.run_eval_modes', '--help'],
                            capture_output=True, text=True, check=True)
    assert '--eoh-seed' in result.stdout


@pytest.mark.parametrize('config_seed,cli_seed,expected', [(None, None, 2024), (71, None, 71), (71, 92, 92)])
def test_run_eoh_seed_precedence_without_llm(tmp_path, monkeypatch, config_seed, cli_seed, expected):
    evol, _, _ = fake_eoh(monkeypatch)
    samples = []

    def run(self):
        samples.append((random.random(), np.random.random()))

    monkeypatch.setattr(evol, 'run', run, raising=False)

    class Paras:
        def set_paras(self, **values):
            self.__dict__.update(values)

    monkeypatch.setitem(sys.modules, 'eoh.utils.getParas', types.SimpleNamespace(Paras=Paras))
    for name in ('_ensure_eoh_importable', 'patch_eoh_remote_api_path', 'patch_eoh_prompt_templates',
                 'patch_eoh_get_alg', 'patch_eoh_n_create', 'patch_eoh_i1_with_oneshot_seed',
                 'patch_eoh_prompts_with_artifact'):
        monkeypatch.setattr(eoh, name, lambda *a, **k: None)
    config = dict(OPENROUTER_API_KEY='unit-test-not-a-real-key', eoh={})
    if config_seed is not None:
        config['eoh']['evolution'] = dict(seed=config_seed)
    eoh.run_eoh(types.SimpleNamespace(paper_id='task', model_name='model'), str(tmp_path), 'model',
                config, pop_size=1, n_pop=1, workers=1, timeout=1, seed=cli_seed)
    assert samples == [(random.Random(expected).random(), np.random.RandomState(expected).random_sample())]
    assert json.loads((tmp_path/'_eoh_rng.json').read_text())['seed'] == expected


def test_explicit_versioned_root_does_not_fall_back(tmp_path, monkeypatch):
    from test_time_self_evolution.seeds import copy_final_oneshot_seed
    root = tmp_path / 'repo'
    old = root / 'eval/seed_programs/task/model'
    old.mkdir(parents=True)
    (old / 'code.py').write_text('old')
    revised = tmp_path / 'revised'
    monkeypatch.setenv('FRONTIER_OR_FINAL_ONESHOT_ROOT', str(revised))
    target = tmp_path / 'target'
    assert copy_final_oneshot_seed(str(root), 'task', 'model', str(target)) is None
    source = revised / 'task/model'
    source.mkdir(parents=True)
    (source / 'code.py').write_text('corrected')
    result = copy_final_oneshot_seed(str(root), 'task', 'model', str(target))
    assert Path(result).read_text() == 'corrected'
    assert (old / 'code.py').read_text() == 'old'
