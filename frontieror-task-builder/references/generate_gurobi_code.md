# Generate Gurobi Code

Use after `original_formulation.tex` and at least `instance/tiny_instance.json` exist.

This reference inherits the critical logic of `prompts/paper_reproduce/prompt_generate_gurobi_code.txt`.

## Inputs

- `original_formulation.tex`
- `instance/tiny_instance.json`
- optional paper PDF or submitter notes for ambiguity resolution
- optional `instance_schema.json`
- optional `solution_schema.json`

## Output

- `gurobi_code.py`

## Requirements

- Implement the confirmed mathematical formulation faithfully.
- If solver-driven reformulation is necessary, explain it in comments and project output back to original decision-variable fields.
- Do not change the problem semantics to make solving easier.
- If a detail is ambiguous, first consult available source material; otherwise mark the assumption clearly in comments.

## CLI Contract

The program must use `argparse` and support:

```bash
python gurobi_code.py \
  --instance_path PATH \
  --solution_path PATH \
  --time_limit SECONDS
```

Use `--time_limit` to set Gurobi's time limit. If optimality is not proven, write the best feasible solution found so far when available.

## Solution Output

- Write JSON to `--solution_path`.
- Include top-level `objective_value`.
- Include only original decision-variable fields needed by `feasibility_check.py`.
- Do not include solver-internal variables, logs, status codes, or hidden reference data in the solution JSON.

## Important

When generating `gurobi_code.py`, do not execute it. Execution belongs to the validation stage.
