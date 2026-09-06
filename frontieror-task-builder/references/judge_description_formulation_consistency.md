# Judge Description/Formulation Consistency

Use this to compare a problem description with `original_formulation.tex`.

## Inputs

- `problem_description.txt` or normalized draft
- `original_formulation.tex` or normalized draft

## Output

Write `consistency_report.md` with:

- overall status: `consistent`, `minor_issues`, or `inconsistent`;
- issue table;
- recommended owner for each fix: description, formulation, or submitter clarification;
- blocking questions if the mismatch cannot be resolved safely.

## Checks

1. Entities and sets:
   - Every entity in the description has a corresponding set/index in the formulation.
   - Every set/index in the formulation has operational meaning in the description.

2. Input data:
   - Every parameter in the formulation appears as input data in the description.
   - Description-only data fields are either mapped to formulation parameters or marked unnecessary.

3. Decisions:
   - Every decision variable maps to an operational choice in the description.
   - Every described choice has a corresponding variable or derived expression.

4. Objective:
   - Direction matches: minimize or maximize.
   - All cost, penalty, reward, or revenue components match.

5. Hard rules:
   - Every formulation constraint has a natural-language counterpart.
   - Every description rule has a mathematical counterpart.
   - Variable domains and bounds match.

6. Algorithm-hint leakage:
   - Flag solver names, decomposition, heuristics, search procedures, dynamic programming, cuts, warm starts, or implementation advice unless they are part of the problem definition.

## Report Format

```markdown
# Consistency Report

Overall status: inconsistent

| Severity | Location | Issue | Suggested fix |
| --- | --- | --- | --- |
| blocking | objective | Description says maximize profit; formulation minimizes cost. | Ask submitter which direction is intended. |

## Submitter Questions

1. ...
```

Do not convert suggestions into final artifacts without submitter validation.
