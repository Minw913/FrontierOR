# One-shot seed programs: 180 tasks × 12 models

This is a source-only snapshot of the evaluated programs in
`efficient_opt_bench/eval/eval_papers`. No instance data, solutions, evaluation
logs, credentials, or historical model directories are included.

Layout:

```text
eval/seed_programs/<paper_id>/<model>/
  code.py              # final saved one-shot program (may include self-debugging)
  code_attempt0.py      # original first attempt, where available
  solution_logger.py   # adjacent local helper used by generated programs
```

All 2,160 final programs and helpers are included without modification.
`manifest.json` records SHA-256 hashes, syntax errors under the export interpreter,
tiny evaluation status, and the original directory mapping. Grok directories are
normalized to `grok-4.20`; most originally used `grok-4.20-beta`.

## Choosing a self-evolution starter

CORAL, EoH, and OpenEvolve now reuse the **final `code.py`** directly from this
snapshot. No copying to `eval/eval_tasks` is required. If a final snapshot entry
is absent, they look for the same model's runtime
`eval/eval_tasks/<paper_id>/<model>/code.py`. They never substitute attempt zero.
The adjacent logger is copied, and `_seed_source.json` / `_seed_source.txt`
record the exact source, final-one-shot version, and SHA-256 hash.

This defines improvement as evolved score minus the matching final one-shot
score, not minus a pre-debugging first-attempt score. Re-evaluate the seed under
the same current checker, instances, time budget and metric when checking this
comparison. A successful copy alone does not guarantee numerical score parity.
Missing/rejected seeds still use the runner's existing live-generation fallback;
check provenance to exclude those runs from a paired one-shot comparison.
Resume and explicit keep-existing-seed options preserve old runs; use a fresh
run ID to adopt this policy rather than silently replacing a resumed seed.

There are 2,148 original first attempts. The following 12 combinations have no
saved `code_attempt0.py`: tasks `bard2002` and `kress2019`, each for
`claude-opus-4.6`, `gemini-3.1-pro-preview`,
`gpt-5.3-codex`, `grok-4.20`, `llama-4-maverick`, and `qwen3-coder-plus`.
Consult the manifest for authoritative entries.
No substitute first-attempt files are fabricated. They are retained only for
historical auditing and are not the starters used by the current adapters.

## EoH evolutionary random seed

Set `--eoh-seed 2025` on `python -m test_time_self_evolution.run_eval_modes ...`
to override `eoh.evolution.seed` in YAML (default `2024`). This controls Python
and NumPy framework RNGs, including reseeding after upstream constructors that
hard-code seed 2024. It does not control LLM sampling. Thread interleaving and
parallel tasks share process-global RNG state, so this is not a guarantee of
bitwise reproducibility or statistically independent parallel task streams.
For seed-based repeats use separate processes/run IDs, preferably one paper
worker and one EoH worker. Resume restarts RNGs from the chosen seed rather than
restoring a checkpointed RNG state. `_eoh_rng.json` records the configured seed.

## Quality and execution caveats

Coverage does not imply correctness: this snapshot preserves failed programs
as well as successful ones. Four final programs have syntax errors under Python
3.13: `wang2018`, `martin-iradi2024`, `gruson2021`, and `riley2025`, all for
`llama-4-maverick`. First-attempt syntax issues are recorded separately.
No algorithms were repaired or executed during export. Use the benchmark's
isolated evaluator and current checker when running generated code.

Programs require their normal third-party libraries and the benchmark
instance/schema; these are not standalone task packages. Most use Gurobi,
which requires an appropriate license. Some also import NumPy, SciPy,
NetworkX, ijson, or orjson. Task definitions are distributed separately in
the FrontierOR dataset.

## Rebuilding the snapshot

```bash
python scripts/export_seed_programs.py \
  --source-root /path/to/efficient_opt_bench/eval/eval_papers \
  --metadata /path/to/frontier-or/metadata/paper_meta_info.json
```

The export reads the 12 named `eval/eval_results_*.csv` files to identify models
and tiny outcomes, and refuses to overwrite different already-exported sources.
