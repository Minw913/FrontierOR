# Generate Problem Description From Formulation

Use in Flow 3 when only `original_formulation.tex` is provided.

## Inputs

- `original_formulation.tex`
- optional symbol glossary or submitter notes

## Output

- `problem_description.draft.txt`

## Rules

Inherit the logic of `scripts/paper_reproduce/prompts/prompt_generate_problem_description.txt`:

- begin with `# Problem Description`;
- write continuous prose paragraphs after the header;
- do not use LaTeX, equation blocks, bullets, or additional section headers;
- describe mathematical relationships verbally;
- avoid modeling jargon such as "decision variable", "constraint", "objective function", "binary variable", and "nonnegativity";
- do not include examples, worked instances, algorithmic hints, or commentary on difficulty.

## Required Content

The description must encode:

- entities and how many of each;
- every category of input data;
- operational choices to make;
- every hard rule from the formulation;
- objective direction and named objective components.

If business semantics are unclear from mathematical symbols, write the most faithful neutral wording possible and add a submitter-review note outside the final description.
