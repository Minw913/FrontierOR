import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from test_time_self_evolution.openevolve import evaluator as ev


def test_feedback_uses_actual_default_and_custom_boundaries(tmp_path):
    args=('staged_qte', {}, ['large_3'], {'large_3':100}, {}, str(tmp_path), 'task')
    default=ev._build_stage2_artifacts(*args)['score_summary']
    assert '0.01 (1%)' in default and '[0, 0.99]' in default
    assert 'max(0, (τ_g - t_solve) / τ_g)' in default
    custom=ev._build_stage2_artifacts(*args, stage_boundary=.05)['score_summary']
    assert '0.05 (5%)' in custom and '[0, 0.95]' in custom
    assert '0.01 (1%)' not in custom


def test_tiny_feedback_uses_configured_budget_and_gap(tmp_path):
    r={'status':'fail','fail_reason':'runtime_error','error':'Execution timed out after 17s','feasible':False}
    text=ev._build_stage1_artifacts(r,str(tmp_path),'task',time_limit=17,gap_threshold=.03)['summary']
    assert '17s' in text and '300s' not in text
    r={'status':'fail','fail_reason':'gap_exceeds','feasible':True,'gap':.5}
    assert 'gap > 3%' in ev._build_stage1_artifacts(r,str(tmp_path),'task',gap_threshold=.03)['summary']


def setup_stage2(tmp_path,monkeypatch):
    monkeypatch.setenv('EFFICIENT_OR_PAPER_ID','task')
    monkeypatch.setenv('EFFICIENT_OR_STAGE2_INSTANCES','large_3')
    monkeypatch.setenv('EFFICIENT_OR_OUTPUT_DIR',str(tmp_path/'out'))
    monkeypatch.setenv('EFFICIENT_OR_STAGE2_STAGE_BOUNDARY','.03')
    monkeypatch.setattr(ev.eval_core,'get_paper_direction',lambda _: 'min')
    monkeypatch.setattr('test_time_self_evolution.scoring.building_blocks.lookup_gurobi_time',lambda *a:100)
    monkeypatch.setattr('test_time_self_evolution.scoring.building_blocks.lookup_gurobi_time_limit',lambda *a:100)
    def run(paper,model,instances,program_path,output_dir,*args):
        out=Path(output_dir);code=Path(program_path).read_text()
        assert Path(program_path).parent==out
        result={'status':'pass','feasible':True,'llm_obj':100,'gurobi_obj':100,'solve_time':1,'gap':0,'aocc':0}
        (out/'solution_large_3.json').write_text(json.dumps({'source_marker':code}))
        (out/'log_large_3.jsonl').write_text('{"time":1,"objective_value":100}\n')
        (out/'feasi_result_large_3.json').write_text('{"feasible":true}')
        return {'large_3':result}
    monkeypatch.setattr(ev.eval_core,'evaluate_candidate_code',run)


def test_concurrent_candidates_and_identical_repeats_preserve_evidence(tmp_path,monkeypatch):
    setup_stage2(tmp_path,monkeypatch)
    a=tmp_path/'a.py';a.write_text('# marker A\n')
    b=tmp_path/'b.py';b.write_text('# marker B\n')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(ev.evaluate_stage2,[str(a),str(b),str(a)]))
    paths=list((tmp_path/'out/stage2').glob('*/*/evaluation.json'))
    assert len(paths)==3
    ids=set();sources=[]
    for path in paths:
        folder=path.parent;record=json.loads(path.read_text());ids.add(record['evaluation_id'])
        assert record['status']=='completed'
        assert hashlib.sha256((folder/'code.py').read_bytes()).hexdigest()==record['source_sha256']
        source=(folder/'code.py').read_text();sources.append(source)
        assert json.loads((folder/'solution_large_3.json').read_text())['source_marker']==source
        assert (folder/'results.json').is_file() and (folder/'metrics.json').is_file()
        artifacts=json.loads((folder/'artifacts.json').read_text())
        assert artifacts['evaluation_id']==record['evaluation_id']
        assert artifacts['stage2_evaluation_id']==record['evaluation_id']
        assert '0.03 (3%)' in artifacts['score_summary']
    assert len(ids)==3 and sources.count(a.read_text())==2 and sources.count(b.read_text())==1
    assert not (tmp_path/'out/stage2/solution_large_3.json').exists()


def test_failed_evaluation_retains_partial_output_and_source(tmp_path,monkeypatch):
    setup_stage2(tmp_path,monkeypatch)
    def failed(*args):
        (Path(args[4])/'partial-output.json').write_text('{"partial":true}')
        raise RuntimeError('simulated evaluator failure')
    monkeypatch.setattr(ev.eval_core,'evaluate_candidate_code',failed)
    p=tmp_path/'program.py';p.write_text('print(1)')
    with pytest.raises(RuntimeError,match='simulated'):
        ev.evaluate_stage2(str(p))
    path=next((tmp_path/'out/stage2').glob('*/*/evaluation.json'))
    record=json.loads(path.read_text())
    assert record['status']=='failed' and record['error_type']=='RuntimeError'
    assert (path.parent/'partial-output.json').exists() and (path.parent/'code.py').exists()
