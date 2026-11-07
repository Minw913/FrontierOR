"""One candidate evaluation per OS process; no shared environment with siblings."""
from pathlib import Path
import json,sys,traceback
from test_time_self_evolution.openevolve import evaluator
from test_time_self_evolution.eoh.problem_adapter import _to_metrics,_to_artifacts

def main():
    code,result=sys.argv[1:3]
    try:
        s1=evaluator.evaluate_stage1(code);m1=_to_metrics(s1)
        row={'stage1_metrics':m1,'stage1_artifact':_to_artifacts(s1),'stage2_metrics':{},'stage2_artifact':{}}
        if float(m1.get('combined_score',0))>=1:
            s2=evaluator.evaluate_stage2(code);row.update(stage2_metrics=_to_metrics(s2),stage2_artifact=_to_artifacts(s2))
        Path(result).write_text(json.dumps(row,indent=2)+'\n')
    except BaseException:
        Path(result).write_text(json.dumps({'worker_error':traceback.format_exc()})+'\n');raise
if __name__=='__main__':main()
