# Flow 2: Problem Description Only

Use when the submitter provides a natural-language problem description but no mathematical formulation.

## Inputs

- `problem_description.txt`
- optional metadata: direction, problem class, application field, submitter notes

## Plan

1. Normalize the problem description.
   - Remove algorithm hints, solver hints, examples, implementation advice, and heuristic suggestions.
   - Preserve entities, input data, operational choices, hard rules, and objective.
   - Output: `problem_description.normalized.txt`.

2. Generate an algorithm-hint agnostic original formulation.
   - Read `generate_formulation_from_description.md`.
   - Input: normalized description.
   - Output: `original_formulation.draft.tex`.

3. Run consistency review.
   - Read `judge_description_formulation_consistency.md`.
   - Inputs: normalized description and draft formulation.
   - Output: `consistency_report.md`.

4. Ask submitter to validate or revise.
   - The submitter must confirm that the draft formulation exactly captures the problem description, or provide edits.
   - Do not continue with downstream artifacts until the formulation is confirmed.

5. Freeze confirmed pair.
   - Output final `problem_description.txt`.
   - Output final `original_formulation.tex`.

6. Continue with the common downstream chain.
   - Generate draft schemas first: `generate_data_specification.md`.
   - Generate instance generator against `instance_schema.json`: `generate_instances_from_schema.md`.
   - Generate canonical instances and validate them against the schema.
   - Generate Gurobi code against both schemas: `generate_gurobi_code.md`.
   - Generate checker against both schemas: `generate_feasibility_checker.md`.
   - Run Gurobi/checker validation and finalize schemas.

## Flow-Specific Requirements

- There is no paper-derived `all_instances.json`; do not require it.
- Schemas are generated from confirmed semantics before programs. Later validation may revise schema interface details, but not task semantics.
- `generate_multiple_instances.py` must define scale choices, deterministic seed strategy, and feasibility prechecks.
- If the description lacks enough detail to define a formulation, stop and request submitter clarification.
