# Flow 3: Mathematical Formulation Only

Use when the submitter provides `original_formulation.tex` but no natural-language problem description.

## Inputs

- `original_formulation.tex`
- optional metadata: direction, problem class, application field, symbol glossary

## Plan

1. Normalize the formulation.
   - Preserve submitter semantics.
   - Ensure sets, indices, parameters, decision variables, domains, objective, and constraints are explicit.
   - Output: `original_formulation.normalized.tex`.

2. Generate algorithm-hint agnostic problem description.
   - Read `generate_problem_description_from_formulation.md`.
   - Input: normalized formulation and optional glossary.
   - Output: `problem_description.draft.txt`.

3. Run consistency review.
   - Read `judge_description_formulation_consistency.md`.
   - Inputs: draft description and normalized formulation.
   - Output: `consistency_report.md`.

4. Ask submitter to validate or revise.
   - The submitter must confirm that the problem description faithfully states the formulation.
   - Allow edits to terminology, business context, objective wording, and rule descriptions.

5. Freeze confirmed pair.
   - Output final `problem_description.txt`.
   - Output final `original_formulation.tex`.

6. Continue with the common downstream chain.
   - Generate draft schemas from confirmed semantics.
   - Generate instance generator against `instance_schema.json`.
   - Generate canonical instances and validate them against the schema.
   - Generate Gurobi code against both schemas.
   - Generate feasibility checker against both schemas.
   - Run Gurobi/checker validation and finalize schemas.

## Flow-Specific Requirements

- The generated description must not expose mathematical notation, solver-specific reformulation, or algorithm hints.
- If the formulation omits parameter meanings or business semantics, preserve mathematical fidelity and mark unclear operational terms for submitter review.
- Schemas are contract-first artifacts, not extracted after programs are generated.
