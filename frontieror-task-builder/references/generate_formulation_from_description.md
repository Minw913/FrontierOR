# Generate Formulation From Problem Description

Use in Flow 2 when only a problem description is provided.

## Inputs

- normalized `problem_description.txt`
- optional direction, problem class, application field, submitter notes

## Output

- `original_formulation.draft.tex`

## Rules

- Produce an algorithm-hint agnostic mathematical formulation of the original optimization problem.
- Include sets, indices, parameters, decision variables, domains, objective, and all hard constraints.
- Keep constraints aligned with the description's operational rules.
- Do not introduce solver reformulations, valid inequalities, heuristics, decomposition, or implementation shortcuts.
- If the description is ambiguous, mark the ambiguity and ask a submitter question instead of inventing a hidden assumption.

## Output Format

Use LaTeX-style sections:

```tex
\paragraph{Sets and indices.}
...

\paragraph{Parameters.}
...

\paragraph{Decision variables.}
...

\paragraph{Objective.}
...

\paragraph{Constraints.}
...
```

Number constraints in stable order so `feasibility_check.py` can report violations against those indices.
