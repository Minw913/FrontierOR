# Generate Data Specification

Use after a confirmed `problem_description.txt` and `original_formulation.tex` exist, before generating downstream programs.

This reference inherits the underlying logic of `prompts/paper_reproduce/prompt_generate_data_specification.txt`.

## Inputs

Read from the task directory:

- `problem_description.txt`
- `original_formulation.tex`
- optional `instance/tiny_instance.json`
- optional `gurobi_solution/tiny_solution.json`
- optional `generate_multiple_instances.py`
- optional `gurobi_code.py`

If example JSON files exist, preserve their field names unless they contradict the confirmed description/formulation. If examples do not exist yet, generate draft schemas from confirmed semantics and available instance-setting notes. Downstream programs must implement this draft interface.

## Outputs

Write:

- draft or final `instance_schema.json`
- draft or final `solution_schema.json`

## Schema Style

Both files should be JSON objects that mirror the corresponding instance or solution nesting structure. Replace each leaf value with:

```text
"<type, shape> Description."
```

Use operational/business descriptions. Do not use modeling jargon when describing fields.

## Instance Schema Rules

- Include every input data field needed to instantiate the formulation.
- Include sets and sizes, scalar parameters, vectors, matrices, tensors, graph data, time data, capacity data, demand data, cost data, distances, probabilities, and scenarios as needed.
- Do not include decision variables.
- Do not include hidden reference objectives, known optimal solutions, solver status, random seeds, timestamps, or metadata that does not feed the optimization model.

## Solution Schema Rules

- Include top-level `objective_value`.
- Include all original decision variables needed to verify every hard constraint and variable domain.
- Exclude solver-internal auxiliary variables unless they correspond to original formulation variables.
- For dict-valued fields where keys encode index tuples, describe key format using semantic names.

## Consistency Checks

- Every formulation parameter maps to an instance field.
- Every formulation decision variable maps to a solution field.
- Every description field appears in one of the schemas or is justified as non-model metadata.
- `feasibility_check.py` must be able to read all fields needed for validation.

## Finalization

After `generate_multiple_instances.py`, `gurobi_code.py`, and `feasibility_check.py` exist and actual JSON artifacts have been produced, rerun schema consistency validation. Only interface-level schema refinements are allowed; do not change the confirmed problem semantics.
