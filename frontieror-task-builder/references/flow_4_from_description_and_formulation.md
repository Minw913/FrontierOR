# Flow 4: Problem Description and Mathematical Formulation

Use when the submitter provides both `problem_description.txt` and `original_formulation.tex`.

## Inputs

- `problem_description.txt`
- `original_formulation.tex`
- optional metadata: direction, problem class, application field

## Plan

1. Normalize both inputs.
   - Remove algorithm and solver hints from the description.
   - Clean formulation formatting without changing semantics.
   - Outputs: `problem_description.normalized.txt`, `original_formulation.normalized.tex`.

2. Run LLM-as-judge consistency validation.
   - Read `judge_description_formulation_consistency.md`.
   - Inputs: normalized description and normalized formulation.
   - Output: `consistency_report.md`.

3. If inconsistent, produce modification suggestions.
   - Identify whether each issue is better fixed in the description, formulation, or both.
   - Do not silently create final corrected files.

4. Ask submitter to validate and revise.
   - If submitter changes either file, rerun consistency validation.
   - Do not proceed until the pair is confirmed consistent.

5. Freeze confirmed pair.
   - Output final `problem_description.txt`.
   - Output final `original_formulation.tex`.

6. Continue with the common downstream chain.
   - Generate draft schemas from the confirmed pair.
   - Generate instance generator against `instance_schema.json`.
   - Generate canonical instances and validate them against the schema.
   - Generate Gurobi code against both schemas.
   - Generate feasibility checker against both schemas.
   - Run Gurobi/checker validation and finalize schemas.

## Flow-Specific Requirements

- This is the preferred community submission path because it provides both semantic and mathematical sources.
- The main work is consistency validation and submitter-in-the-loop repair, not unnecessary regeneration.
- Schemas are contract-first artifacts. Use actual generated JSON later to validate and refine interface details only.
