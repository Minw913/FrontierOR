import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_time_self_evolution.eoh.budget import EvaluationBudget, patch_batch_admission
from test_time_self_evolution.seed_fitness import configure_seed_cache


def test_atomic_budget_excludes_seed_and_survives_resume(tmp_path):
    path = tmp_path / 'budget.json'
    budget = EvaluationBudget(path, 20, 'seed', context='settings')
    assert all(budget.admit('seed') for _ in range(3))
    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(budget.admit, [str(i) for i in range(40)]))
    assert sum(admitted) == 20
    assert budget.remaining == 0
    assert EvaluationBudget(path, 20, 'seed', resume=True, context='settings').remaining == 0
    with pytest.raises(RuntimeError, match='different'):
        EvaluationBudget(path, 20, 'seed', resume=True, context='changed')


def test_trim_batch_counts_twenty_new_submissions(tmp_path):
    budget = EvaluationBudget(tmp_path / 'budget.json', 20, 'seed')
    class Problem:
        candidate_budget = budget
    class Interface:
        interface_eval = Problem()
        pop_size = 2
        generated = 0
        first = True
        def get_algorithm(self, population, operator):
            offspring = []
            for _ in range(self.pop_size):
                if self.first:
                    code = 'seed'
                    self.first = False
                else:
                    self.generated += 1
                    code = str(self.generated)
                assert budget.admit(code)
                offspring.append(code)
            return [], offspring
    patch_batch_admission(Interface)
    interface = Interface()
    _, initial = interface.get_algorithm([], 'i1')
    assert initial == ['seed', '1']
    sizes = []
    for _ in range(20):
        _, offspring = interface.get_algorithm(initial, 'e1')
        sizes.append(len(offspring))
    assert interface.generated == 20
    assert sizes[:10] == [2] * 9 + [1]
    assert sizes[10:] == [0] * 10
    assert interface.pop_size == 2


def test_seed_cache_replays_evidence_not_solver(tmp_path, monkeypatch):
    from test_time_self_evolution.openevolve.evidence import (
        isolated_evaluation, capture_results, attach_artifacts,
    )
    from test_time_self_evolution.openevolve import evaluator
    source = tmp_path / 'seed.py'
    source.write_text('print(1)\n')
    monkeypatch.setenv('FRONTIER_OR_SEED_SHA256', hashlib.sha256(b'print(1)').hexdigest())
    monkeypatch.setenv('FRONTIER_OR_SEED_FITNESS_ID', 'same-settings')
    monkeypatch.setenv('FRONTIER_OR_SEED_FITNESS_CACHE', str(tmp_path / 'cache'))
    calls = []
    def environment():
        return 'task', 'model', 'bare', str(tmp_path / 'evidence'), None, {}
    @isolated_evaluation('stage1', environment)
    def evaluate(path):
        calls.append(path)
        capture_results({'tiny': {'feasible': True}})
        artifacts = attach_artifacts({'summary': 'passed'})
        metrics = {'combined_score': 1.0}
        return (evaluator.EvaluationResult(metrics=metrics, artifacts=artifacts)
                if evaluator._HAS_EVALUATION_RESULT else metrics)
    first = evaluate(str(source))
    second = evaluate(str(source))
    assert getattr(first, 'metrics', first) == getattr(second, 'metrics', second)
    assert len(calls) == 1
    records = [json.loads(p.read_text()) for p in (tmp_path / 'evidence').rglob('evaluation.json')]
    assert sum(r.get('seed_fitness_cache_hit', False) for r in records) == 1
    assert len(list((tmp_path / 'evidence').rglob('results.json'))) == 2
    monkeypatch.setenv('FRONTIER_OR_SEED_FITNESS_ID', 'different-settings')
    evaluate(str(source))
    source.write_text('print(2)')
    evaluate(str(source))
    assert len(calls) == 3


def test_cache_identity_changes_with_inputs_and_protocol(tmp_path):
    paper = tmp_path / 'task'
    (paper / 'instance').mkdir(parents=True)
    source = tmp_path / 'seed.py'
    source.write_text('pass')
    instance = paper / 'instance/large_instance_1.json'
    instance.write_text('{}')
    def identity(limit='3600'):
        env = {'EFFICIENT_OR_STAGE2_INSTANCES': 'large_1',
               'EFFICIENT_OR_STAGE2_TIME_LIMIT': limit}
        configure_seed_cache(env, source, paper, tmp_path)
        return env['FRONTIER_OR_SEED_FITNESS_ID']
    before = identity()
    assert identity() == before
    assert identity('120') != before
    instance.write_text('{"changed": true}')
    assert identity() != before


def test_eoh_seed_is_pre_evaluated_and_failures_consume_budget(tmp_path):
    from test_time_self_evolution.eoh.problem_adapter import EohBenchmarkProblem, code_hash
    counts = []
    def stage1(path):
        counts.append(path)
        return {'combined_score': 1.0}
    problem = EohBenchmarkProblem(
        paper_id='mock', model_name='mock', prompt='task', base_output=str(tmp_path),
        stage1_instances=['tiny'], stage2_instances=['large_1'],
        stage1_time_limit=300, stage2_time_limit=3600, stage1_gap_threshold=.1,
        exec_mode='bare', exec_cfg={}, t_max=None, stage2_scorer='staged_qte',
        stage1_fn=stage1, stage2_fn=lambda path: {'combined_score': 1.5})
    problem.candidate_budget = EvaluationBudget(tmp_path / 'budget.json', 2, code_hash('seed'))
    assert problem.evaluate('seed') == -1.5
    problem.seed_fitness_ready = True
    assert problem.evaluate('seed') == -1.5
    assert len(counts) == 1 and problem.candidate_budget.remaining == 2
    assert problem.evaluate('new') == -1.5
    def broken(path):
        raise RuntimeError('mock failure')
    problem.stage1_fn = broken
    assert problem.evaluate('broken') == 1e9
    assert problem.candidate_budget.remaining == 0
    with pytest.raises(RuntimeError, match='exhausted'):
        problem.evaluate('third')


def test_real_eoh_loop_has_twenty_new_evaluations_plus_seed(tmp_path, monkeypatch):
    """Exercise upstream population/selection/checkpoint loop without API/solver calls."""
    from test_time_self_evolution.eoh import runner
    from test_time_self_evolution.eoh.problem_adapter import EohBenchmarkProblem
    runner._ensure_eoh_importable()
    from eoh.methods.eoh import eoh_interface_EC as ec
    from eoh.methods.eoh.eoh_evolution import Evolution
    monkeypatch.setattr(Evolution, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ec.time, 'sleep', lambda seconds: None)
    for name in ('patch_eoh_remote_api_path', 'patch_eoh_prompt_templates',
                 'patch_eoh_get_alg', 'patch_eoh_prompts_with_artifact', 'patch_eoh_rng_seed'):
        monkeypatch.setattr(runner, name, lambda *a, **k: None)
    output = tmp_path / 'run'
    def prepare(*args, **kwargs):
        folder = output / 'seed_oneshot'
        folder.mkdir(parents=True)
        (folder / 'code.py').write_text('seed')
        return True
    monkeypatch.setattr(runner, 'patch_eoh_i1_with_oneshot_seed', prepare)
    monkeypatch.setattr(runner.eval_core, 'get_paper_dir', lambda paper: str(tmp_path / 'task'))
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-no-network')
    generated = []
    def generate(self, population, operator):
        code = 'seed' if not generated else f'candidate{len(generated)}'
        generated.append(code)
        return None, dict(code=code, algorithm='mock', objective=None, other_inf=None)
    monkeypatch.setattr(ec.InterfaceEC, '_get_alg', generate)
    evaluated = []
    def stage1(path):
        evaluated.append(open(path).read())
        return {'combined_score': 1.0}
    problem = EohBenchmarkProblem(
        paper_id='mock', model_name='mock', prompt='task', base_output=str(tmp_path / 'adapter'),
        stage1_instances=['tiny'], stage2_instances=['large_1'], stage1_time_limit=300,
        stage2_time_limit=3600, stage1_gap_threshold=.1, exec_mode='bare', exec_cfg={},
        t_max=None, stage2_scorer='staged_qte', stage1_fn=stage1,
        stage2_fn=lambda path: {'combined_score': .5 + len(evaluated) / 100})
    problem.new_candidate_evaluations = 20
    runner.run_eoh(problem, str(output), 'mock', {'eoh': {}},
                   pop_size=2, n_pop=1, workers=2, timeout=60, operators=['e1','e2','m2'])
    assert evaluated[0] == 'seed' and evaluated.count('seed') == 1
    assert len(evaluated) == 21
    assert len(generated) == 21
    assert problem.candidate_budget.remaining == 0
    best = runner.find_best_eoh_individual(str(output))
    assert best['code'].startswith('candidate')
