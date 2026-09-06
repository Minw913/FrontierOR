# FrontierOR Task Artifact Contract

Read this before creating or editing task artifacts.

## Required Files

```text
<task_dir>/
├── problem_description.txt
├── original_formulation.tex
├── instance_schema.json
├── solution_schema.json
├── feasibility_check.py
├── generate_multiple_instances.py
├── gurobi_code.py
└── instance/
    ├── tiny_instance.json
    ├── large_instance_1.json
    ├── large_instance_2.json
    ├── large_instance_3.json
    ├── large_instance_4.json
    └── large_instance_5.json
```

Reference outputs generated during validation:

```text
gurobi_solution/<name>_solution.json
gurobi_feasi_result/<name>_feasi_result.json
gurobi_solution_log/<name>_log.jsonl
```

where `<name>` is `tiny` or `large_1` through `large_5`.

## Problem Description Contract

`problem_description.txt` must be concise natural language that lets another model reconstruct the optimization problem without the original source. It must include:

- entities and counts;
- input data categories;
- operational choices to make;
- every hard rule from the formulation expressed verbally;
- objective direction and named cost/revenue components.

It must not include algorithm hints, solver hints, reformulation guidance, examples, worked instances, or commentary on solution difficulty.

## Formulation Contract

`original_formulation.tex` must contain the original optimization model:

- sets, indices, parameters;
- decision variables and domains;
- objective;
- all hard constraints in stable order;
- reproduction-critical assumptions and markings for reconstructed or unspecified elements.

Do not replace the original problem with a Gurobi-specific reformulation.

## Gurobi Code CLI Contract

`gurobi_code.py` must support:

```bash
python gurobi_code.py \
  --instance_path instance/tiny_instance.json \
  --solution_path gurobi_solution/tiny_solution.json \
  --time_limit 3600
```

It must write a JSON solution with top-level `objective_value` and only original decision-variable fields needed by `feasibility_check.py`.

## Feasibility Checker CLI Contract

`feasibility_check.py` must support:

```bash
python feasibility_check.py \
  --instance_path instance/tiny_instance.json \
  --solution_path gurobi_solution/tiny_solution.json \
  --result_path gurobi_feasi_result/tiny_feasi_result.json
```

The result JSON must include:

```json
{
  "feasible": true,
  "violated_constraints": [],
  "violations": [],
  "violation_magnitudes": []
}
```

For violations, include normalized magnitudes with `constraint`, `lhs`, `rhs`, `raw_excess`, `normalizer`, and `ratio`.

## Validation Gates

Do not call a task complete until:

- all required files exist;
- all instances match `instance_schema.json`;
- all Gurobi solutions match `solution_schema.json`;
- every solution has numeric `objective_value`;
- `feasibility_check.py` passes all Gurobi reference solutions;
- objective direction is known and registered in task metadata;
- checker and solver read only CLI-provided paths and do not access hidden answers or network.

## Schema-First Contract Rule

Generate `instance_schema.json` and `solution_schema.json` from the confirmed `problem_description.txt` and `original_formulation.tex` before generating downstream programs. The schemas are the interface contract for:

- `generate_multiple_instances.py`;
- `gurobi_code.py`;
- `feasibility_check.py`.

After actual instances and solutions exist, validate schema/program/checker consistency. It is acceptable to revise schema field names, shape descriptions, or JSON nesting when the draft interface is impractical. It is not acceptable to change schema semantics in a way that changes the confirmed problem description or formulation.
