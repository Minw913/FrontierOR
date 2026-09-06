# Repair Validation Failures

Use when task validation fails.

## Diagnose First

Collect:

- failing command;
- stdout/stderr;
- instance JSON path;
- solution JSON path;
- feasibility result JSON;
- relevant schema snippets;
- relevant formulation constraints.

Do not edit multiple core artifacts at once unless the failure clearly requires it.

## Common Failures

### Instance Does Not Match Schema

Likely owner: `generate_multiple_instances.py` or `instance_schema.json`.

Fix by aligning field names, nesting, types, and dimensions. Prefer changing the generator if the schema reflects the confirmed formulation.

### Gurobi Code Cannot Read Instance

Likely owner: `gurobi_code.py`.

Fix parsing logic to match `instance_schema.json` and concrete instance files.

### Gurobi Solution Does Not Match Solution Schema

Likely owner: `gurobi_code.py` or `solution_schema.json`.

Ensure `objective_value` exists and all original decision variables needed by the checker are emitted.

### Feasibility Checker Rejects Gurobi Solution

Possible owners:

- checker implementation;
- Gurobi formulation;
- solution projection from reformulated variables;
- instance infeasibility;
- confirmed formulation ambiguity.

Compare the violated constraint with `original_formulation.tex`. If Gurobi code intentionally uses a reformulation, verify output is projected back to original variables before checking.

### Tiny Instance Too Hard

Likely owner: `generate_multiple_instances.py`.

Scale down tiny until it is a reliable smoke test.

## Repair Loop

1. Identify one failing invariant.
2. Patch the most likely owning artifact.
3. Rerun the narrowest validation command.
4. Only then continue to broader validation.
