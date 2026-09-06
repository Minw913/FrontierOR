# Flow 1: Paper PDF Only

Use when the submitter provides only a paper PDF, or when FrontierOR is reproducing a task from a paper.

## Inputs

- `paper.pdf`
- optional metadata: `paper_id`, title, source link, direction, problem class

## Plan

1. Extract original formulation.
   - Read `extract_original_formulation.md`.
   - Input: paper PDF.
   - Output: `original_formulation.tex`.
   - Preserve the first original formulation from the paper. Do not substitute a solver reformulation.

2. Generate algorithm-hint agnostic problem description.
   - Use the underlying rules from `prompts/paper_reproduce/prompt_generate_problem_description.txt`.
   - Inputs: paper PDF and `original_formulation.tex`.
   - Output: `problem_description.txt`.
   - Do not rely on later implementation files to define semantics.

3. Judge description/formulation consistency.
   - Read `judge_description_formulation_consistency.md`.
   - Inputs: `problem_description.txt` and `original_formulation.tex`.
   - Output: `consistency_report.md`.
   - Repair any semantic mismatch before proceeding.

4. Generate paper-derived instance settings.
   - Use the first-stage logic from `prompts/paper_reproduce/prompt_generate_instances.txt`.
   - Input: paper PDF and computational-experiment settings.
   - Output: `all_instances.json`.

5. Generate draft schemas as the interface contract.
   - Read `generate_data_specification.md`.
   - Inputs: confirmed `problem_description.txt`, confirmed `original_formulation.tex`, and `all_instances.json`.
   - Outputs: draft `instance_schema.json`, draft `solution_schema.json`.
   - The schemas are the contract for downstream code; do not wait to extract them from generated programs.

6. Generate instance generator against `instance_schema.json`.
   - Use the second-stage logic from `prompts/paper_reproduce/prompt_generate_instances.txt`, plus schema-first constraints.
   - Inputs: `all_instances.json`, confirmed description/formulation, and draft schemas.
   - Output: `generate_multiple_instances.py`.

7. Generate canonical instances and validate against `instance_schema.json`.
   - Output: `instance/tiny_instance.json` and `instance/large_instance_1.json` through `instance/large_instance_5.json`.

8. Generate Gurobi reference solver against schemas.
   - Read `generate_gurobi_code.md`.
   - Inputs: `original_formulation.tex`, schemas, `instance/tiny_instance.json`, and paper PDF for ambiguity resolution.
   - Output: `gurobi_code.py`.

9. Generate feasibility checker against schemas.
   - Read `generate_feasibility_checker.md`.
   - Inputs: `original_formulation.tex`, `instance_schema.json`, and `solution_schema.json`.
   - Output: `feasibility_check.py`.

10. Run Gurobi reference solver.
   - Inputs: `gurobi_code.py` and all canonical instances.
   - Outputs: `gurobi_solution/*` and `gurobi_solution_log/*`.

11. Run feasibility checker on Gurobi solutions.
   - Output: `gurobi_feasi_result/*`.

12. Validate complete task and finalize schemas.
   - Run `scripts/check_artifact_contract.py`.
   - Run `scripts/validate_task.py` when solver/checker execution is available.
   - Allow schema edits only for interface details such as field names, shapes, and nesting. Do not change confirmed semantics.

## Flow-Specific Risks

- Paper algorithms can leak into `problem_description.txt`; remove algorithm, heuristic, decomposition, and solver hints.
- Paper may present multiple formulations; use the first original formulation unless the paper explicitly adopts an equivalent formulation inside the formulation section.
- Paper instances may be underspecified; document deterministic assumptions in `generate_multiple_instances.py`.
- This flow intentionally improves on legacy paper-reproduce ordering: schemas are generated before downstream programs as a contract, then validated against actual JSON artifacts later.
