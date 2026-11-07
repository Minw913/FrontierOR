import json,os,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
from test_time_self_evolution.eoh import model_io,runner
from test_time_self_evolution.eoh.problem_adapter import EohBenchmarkProblem
from test_time_self_evolution.public_context import prepare_public_context
from test_time_self_evolution.run_eval_modes import parse_args,eoh_paper_child_command


def public(tmp_path,monkeypatch):
    task=tmp_path/'source';(task/'instance').mkdir(parents=True)
    for name in ['problem_description.txt','instance_schema.json','solution_schema.json']:(task/name).write_text('{}')
    for name in ['tiny_instance.json','large_instance_1.json','large_instance_2.json']:(task/'instance'/name).write_text(json.dumps({'name':name,'data':'x'*80}))
    monkeypatch.setattr('test_time_self_evolution.public_context.MAX_INLINE_BYTES',1)
    context=prepare_public_context(task,tmp_path/'run','task',['tiny'],['large_1'],['large_2'],allow_retrieval=True)
    return tmp_path/'run/public_context/task',context


def test_large_context_has_real_bounded_tools_and_hides_final(tmp_path,monkeypatch):
    root,context=public(tmp_path,monkeypatch)
    assert 'read_public_file' in context and 'large_instance_1.json' in context
    assert 'large_instance_2.json' not in context
    r=model_io.read_public_file(root,{'path':'instance/large_instance_1.json','offset':5,'length':8})
    assert len(r['text'])==8 and r['next_offset']==13
    for name in ['../source/instance/large_instance_2.json','instance/large_instance_2.json','/etc/passwd']:
        with pytest.raises(ValueError):model_io.read_public_file(root,{'path':name})
    for length in [0,8193,True]:
        with pytest.raises(ValueError):model_io.read_public_file(root,{'path':'instance/tiny_instance.json','length':length})
    target=root/'instance/tiny_instance.json';target.unlink();target.symlink_to(tmp_path/'source/instance/large_instance_2.json')
    with pytest.raises(ValueError):model_io.read_public_file(root,{'path':'instance/tiny_instance.json'})


class Response:
    def __init__(self,data,status=200):self.data=data;self.status_code=status
    def json(self):return self.data
    def raise_for_status(self):
        if self.status_code>=400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def test_tool_responses_all_accounted_and_actual_prompts_retained(tmp_path,monkeypatch):
    root,_=public(tmp_path,monkeypatch);sent=[]
    responses=[{'id':'tool-response','model':'mock','usage':{'prompt_tokens':10,'completion_tokens':2,'cost':.01},'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':'call1','type':'function','function':{'name':'read_public_file','arguments':json.dumps({'path':'instance/large_instance_1.json','length':16})}}]}}]}, {'id':'final-response','model':'mock','usage':{'prompt_tokens':20,'completion_tokens':4,'cost':.02},'choices':[{'message':{'role':'assistant','content':'def main(): pass'}}]}]
    def post(url,**kw):sent.append(kw['json']);return Response(responses.pop(0))
    assert model_io.generate([{'role':'user','content':'evolve'}],model='mock',key='FAKE_SECRET',endpoint='http://localhost/mock',output_dir=tmp_path/'calls',public_root=root,post=post)=='def main(): pass'
    usage=[json.loads(l) for l in (tmp_path/'calls/api_usage.jsonl').read_text().splitlines()]
    assert len(usage)==2 and sum(r['usage']['cost'] for r in usage)==pytest.approx(.03)
    assert any(m['role']=='tool' and 'large_instance_1.json' in m['content'] for m in sent[-1]['messages'])
    assert 'FAKE_SECRET' not in ''.join(p.read_text() for p in (tmp_path/'calls').rglob('*.json'))


def test_credit_error_stops_further_calls(tmp_path):
    count=[]
    def post(*a,**k):count.append(1);return Response({},402)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='credit'):
            model_io.generate([],model='mock',key='fake',endpoint='http://localhost/mock',output_dir=tmp_path,post=post)
    assert len(count)==1 and (tmp_path/'credit-abort.json').exists()


def test_cli_and_process_command_preserve_settings():
    argv=['--framework','eoh','--paper-id','a','b','--paper-workers','25','--run-id','old','--eoh-workers','2','--eoh-enable-artifact','--eoh-starter-mode','best-one-shot-program']
    cmd=eoh_paper_child_command(argv,'b','new');args=parse_args(cmd[2:])
    assert args.paper_ids==['b'] and args.paper_workers==1 and args.run_id=='new'
    assert args.eoh_workers==2 and args.eoh_enable_artifact and args.eoh_starter_model=='claude-fable-5'


def test_explicit_fable_starter_never_silently_falls_back(tmp_path,monkeypatch):
    from test_time_self_evolution.openevolve import runner as oe
    monkeypatch.setattr(oe,'ROOT_DIR',str(tmp_path));monkeypatch.setattr(runner,'ROOT_DIR',str(tmp_path));monkeypatch.delenv('FRONTIER_OR_FINAL_ONESHOT_ROOT',raising=False)
    runner._ensure_eoh_importable()
    from eoh.methods.eoh.eoh_evolution import Evolution
    monkeypatch.setattr(Evolution,'i1',Evolution.i1)
    with pytest.raises(ValueError,match='no generation fallback'):
        runner.patch_eoh_i1_with_oneshot_seed('task','gpt-5.6-sol',str(tmp_path/'run'),starter_mode='best-one-shot-program')
    seed=tmp_path/'eval/seed_programs/task/claude-fable-5';seed.mkdir(parents=True);(seed/'code.py').write_text('def main(): pass\n# FABLE')
    assert runner.patch_eoh_i1_with_oneshot_seed('task','gpt-5.6-sol',str(tmp_path/'run'),starter_mode='best-one-shot-program')
    assert '# FABLE' in Evolution.i1(None)[0]


def test_real_parallel_candidate_processes_have_separate_evidence(tmp_path,monkeypatch):
    monkeypatch.setenv('EFFICIENT_OR_PAPER_ID','parent-unchanged')
    code='''import argparse,json
 def main():
  p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit');p.add_argument('--log_path');a=p.parse_args()
  i=json.load(open(a.instance_path));v=i['graph']['vertices'][0]
  json.dump({'objective_value':0,'clique_vertices':[v],'clique_edges':[]},open(a.solution_path,'w'))
 if __name__=='__main__':main()
 '''.replace('\n ','\n')
    def run(tag):
        p=EohBenchmarkProblem(paper_id='segundo2019',model_name='mock',prompt='task',base_output=str(tmp_path/tag),stage1_instances=['tiny'],stage2_instances=['large_1'],stage1_time_limit=17,stage2_time_limit=17,stage1_gap_threshold=1e9,exec_mode='systemd',exec_cfg={'cpus':1,'memory':'256M'},t_max=None,stage2_scorer='staged_qte',enable_artifact=True)
        score=p.evaluate(code);cache=p.read_cached_metrics(code)
        assert cache is not None and cache['stage2_metrics']['stage2_total']==1
        assert '1%' in cache['stage2_artifact']['score_summary']
        return cache
    with ThreadPoolExecutor(max_workers=2) as ex:a,b=list(ex.map(run,['a','b']))
    assert a['stage2_artifact']['stage2_evaluation_id']!=b['stage2_artifact']['stage2_evaluation_id']
    assert os.environ['EFFICIENT_OR_PAPER_ID']=='parent-unchanged'
