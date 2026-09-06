# Generate Feasibility Checker

Use after a confirmed formulation and schemas exist. If tiny instance and Gurobi solution already exist, use them as concrete examples.

This reference inherits the critical logic of `scripts/paper_reproduce/prompts/prompt_feasibility_check.txt`.

## Inputs

- `original_formulation.tex`
- `instance_schema.json`
- `solution_schema.json`
- optional `gurobi_code.py`
- optional `instance/tiny_instance.json`
- optional `gurobi_solution/tiny_solution.json`

## Output

- `feasibility_check.py`

## CLI Contract

The checker must use `argparse` and support:

```bash
python feasibility_check.py \
  --instance_path PATH \
  --solution_path PATH \
  --result_path PATH
```

Do not hard-code or inline any instance or solution data.

## Required Checks

- Check every hard constraint in `original_formulation.tex` one by one.
- Ignore soft constraints, penalties, and heuristic rules unless they are hard feasibility requirements.
- Check variable domains explicitly:
  - binary values are 0/1 within tolerance;
  - integer values are integral within tolerance;
  - nonnegative values are at least `-tol`;
  - explicit bounds are enforced.
- Do not assume JSON representation automatically satisfies domains; check duplicates, index validity, and missing values.

## Violation Magnitudes

Use `tol = 1e-5` and `eps = 1e-5`.

For each violated constraint:

- for `<=` or `<`: `max(0, lhs - rhs)`;
- for `>=` or `>`: `max(0, rhs - lhs)`;
- for equality: `abs(lhs - rhs)`;
- normalized ratio: `raw_excess / max(abs(rhs), eps)`.

## Output JSON

Write:

```json
{
  "feasible": false,
  "violated_constraints": [1],
  "violations": ["Capacity exceeded on route 3"],
  "violation_magnitudes": [
    {
      "constraint": 1,
      "lhs": 25.0,
      "rhs": 20.0,
      "raw_excess": 5.0,
      "normalizer": 20.0,
      "ratio": 0.25
    }
  ]
}
```

`feasible` is true if and only if no hard constraint is violated.
