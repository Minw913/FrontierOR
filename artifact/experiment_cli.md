# One-shot eval command: runs gpt-5.6-sol on only tiny instances from hardset_id.json with Docker anti-hack mode and trusted snapshot-validated AOCC.

```bash
FRONTIER_OR_DATA_DIR=/home/qua/Code/FrontierOR-run/frontier-or \
python -u one_shot_eval.py \
  --paper_id $(python -c 'import json; print(" ".join(json.load(open("/home/qua/Code/FrontierOR-run/artifact/hardset_id.json"))))') \
  --models gpt-5.6-sol \
  --max_debug_retries 5 \
  --instances tiny \
  --tiny_time_limit 300 \
  --t_max gurobi \
  --exec-mode docker \
  --anti-hack \
  --paper_workers 10 \
  --model_workers 2 \
  --instance_workers 20
```
