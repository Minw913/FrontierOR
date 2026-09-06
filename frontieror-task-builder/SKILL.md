---
name: frontieror-task-builder
description: "Create, repair, and validate standard FrontierOR benchmark tasks from one of four submitter inputs: paper PDF only, problem_description.txt only, original_formulation.tex only, or both description and formulation. Use when Codex needs to build task artifacts such as problem_description.txt, original_formulation.tex, instance_schema.json, solution_schema.json, feasibility_check.py, generate_multiple_instances.py, canonical instance JSONs, gurobi_code.py, Gurobi reference solutions, or task validation reports."
---

# FrontierOR Task Builder

Use this skill to turn submitter materials into a standard FrontierOR task. First inspect what the submitter provided, then route to exactly one flow.

## Routing

- Only a paper PDF: read `references/flow_1_from_pdf.md`.
- Only `problem_description.txt`: read `references/flow_2_from_problem_description.md`.
- Only `original_formulation.tex`: read `references/flow_3_from_mathematical_formulation.md`.
- Both `problem_description.txt` and `original_formulation.tex`: read `references/flow_4_from_description_and_formulation.md`.
- After a confirmed description/formulation pair exists, follow the shared artifact contract in `references/artifact_contract.md`.

## Global Rules

- Treat `problem_description.txt` as algorithm-hint agnostic: it describes entities, data, choices, rules, and objective, not solution methods.
- Treat `original_formulation.tex` as algorithm-hint agnostic: it expresses the original optimization problem, not a Gurobi reformulation, heuristic, decomposition, or implementation shortcut.
- Do not proceed from draft description/formulation to schemas, checker, instances, or Gurobi code until submitter validation or LLM-as-judge consistency validation has produced a confirmed pair.
- Generate canonical instances as exactly one tiny instance and five large instances unless the user explicitly asks otherwise.
- Prefer deterministic scripts for validation and execution. Use `scripts/summarize_task_state.py`, `scripts/check_artifact_contract.py`, and `scripts/validate_task.py` when applicable.

## Common Output

A complete task should contain:

```text
<task_dir>/
├── problem_description.txt
├── original_formulation.tex
├── instance_schema.json
├── solution_schema.json
├── feasibility_check.py
├── generate_multiple_instances.py
├── gurobi_code.py
├── instance/
│   ├── tiny_instance.json
│   ├── large_instance_1.json
│   ├── large_instance_2.json
│   ├── large_instance_3.json
│   ├── large_instance_4.json
│   └── large_instance_5.json
├── gurobi_solution/
├── gurobi_feasi_result/
└── gurobi_solution_log/
```

## References

- `references/artifact_contract.md`: standard files, CLI contracts, JSON result contracts, and validation gates.
- `references/judge_description_formulation_consistency.md`: consistency checks for description/formulation pairs.
- `references/generate_data_specification.md`: schema generation rules inherited from `prompt_generate_data_specification.txt`.
- `references/generate_feasibility_checker.md`: checker generation rules inherited from `prompt_feasibility_check.txt`.
- `references/generate_instances_from_schema.md`: instance generator rules for non-PDF flows.
- `references/generate_gurobi_code.md`: Gurobi solver generation rules inherited from `prompt_generate_gurobi_code.txt`.
- `references/repair_validation_failures.md`: repair loop for validation failures.
