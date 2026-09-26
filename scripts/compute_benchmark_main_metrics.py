"""Compute the per-model benchmark metrics on an auto-anchored paper set.

Anchor (denominator) policy:
  Take the strict intersection over all selected/registered models of paper_ids
  that have COMPLETE records --- i.e. one row each for ``tiny`` and every
  large instance in ``LARGE_INSTANCES``. This guarantees that every model
  column in the paper's main table reports its metric over the same set
  of papers ("one anchor for all model columns"). Add or remove a model
  from the registry or CLI selection to expand/contract the anchor.

Metric columns written to ``compute_benchmark_main_metrics.csv``:
  1. Success rate    : fraction of papers passing the tiny-instance gate.
  2. Execution rate  : fraction of first-shot tiny programs that run without
                       triggering self-debug retries.
  3. Feasibility     : fraction of (paper, large_*) cells that return a
                       feasible solution within T_max (gate_fail / missing /
                       infeasible all count as 0).
  4. Solution quality: fraction of (paper, large_*) cells that are both
                       feasible AND have gap <= 1% to the Gurobi reference
                       (matches main.tex's binary Sol. quality definition).
  5. Solve-time ratio: geometric mean of (gurobi_time / effective_model_time)
                       over feasible cells plus LLM-program-timeout cells.
  6. QTE (continuous): per-cell  max(0, 1 - max(gap, 0) * t_solve / tau_g),
                       infeasible/missing cells = 0; mean over all cells.
                       (Legacy continuous QTE; the binary QTE used in the
                       paper is ``beat_gurobi_1``.)
  7. beat_gurobi_0/1/5/10:
                       per-cell binary indicator (feasible AND gap <= thr%
                       AND effective model time is no slower than effective
                       Gurobi time within a small clock tolerance); mean over
                       all cells. Times are capped at their declared budgets.
                       ``beat_gurobi_1`` is the binary QTE in main.tex.
  Plus diagnostic columns: solution_quality_raw / median, n_gap_outliers,
  solve_time_ratio_arith, n_ratio_cells, n_timeout_cells, n_feasible_cells.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval"
HARD_TOP_K = 50
_HARD_SET_PATH: Path | None = None

LARGE_INSTANCES = ["large_1", "large_2", "large_3", "large_4", "large_5"]
GUROBI_INSTANCE_ALIASES = {
    "large_11": "large_1",
    "large_21": "large_2",
    "large_31": "large_3",
    "large_41": "large_4",
    "large_51": "large_5",
}

# Cells whose Gurobi reference is a checker-feasible, proven optimal zero.
# The tolerance is the absolute objective-consistency tolerance used by the
# corresponding hardened checker at a zero objective.  Zero-reference quality
# cannot be expressed as a relative gap, so SQ/QTE use this cell-specific
# absolute comparison instead.  Do not add time-limited zero incumbents here.
PROVEN_ZERO_OPTIMUM_TOLERANCE = {
    **{("araujo2020", f"large_{i}"): 0.5 for i in (1, 3, 5)},
    **{("chen1999", f"large_{i}"): 0.5 for i in (4, 5)},
    **{("dienstknecht2024", f"large_{i}"): 0.5 for i in range(1, 6)},
    **{("earl2005", f"large_{i}"): 1e-6 for i in range(1, 6)},
    ("forrest2006", "large_1"): 1e-3,
    **{("frey2017", f"large_{i}"): 1e-3 for i in range(1, 6)},
    ("kowalczyk2024", "large_2"): 0.5,
    **{("oliveira2020", f"large_{i}"): 1e-5 for i in range(1, 6)},
    **{("wangk2020", f"large_{i}"): 1e-3 for i in range(1, 6)},
}

# Wall-clock budget for large instances (matches gurobi_results_*.csv time_limit).
T_MAX_LARGE = 3600.0
DEFAULT_QTE_TIME_TOLERANCE_SECONDS = 0.01
DEFAULT_QTE_TIME_TOLERANCE_FRACTION = 0.001

def load_gurobi_large_csv(csv_dir: Path) -> pd.DataFrame:
    """Return one clean Gurobi reference row per (paper_id, large instance).

    For 'time_out' rows we use the declared time_limit so the budget tau_g is
    well defined; 'runtime_error' rows stay NaN and are excluded downstream.
    Some historical CSVs contain placeholder duplicate rows with empty Gurobi
    fields (e.g., historical misplaced placeholder rows);
    these are dropped before metric merges so denominators stay exactly
    |paper_set| * len(LARGE_INSTANCES).
    """
    paths = [csv_dir / f"gurobi_results_{i}.csv" for i in range(1, 6)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Gurobi reference CSV(s): {missing}")
    parts = [pd.read_csv(path) for path in paths]
    df = pd.concat(parts, ignore_index=True)

    ref_cols = ["gurobi_time", "solution_status", "gurobi_solution"]
    all_ref_empty = df[ref_cols].isna().all(axis=1) | df[ref_cols].astype(str).apply(
        lambda col: col.str.strip().isin(["", "nan", "None"]), axis=0
    ).all(axis=1)
    df = df[~all_ref_empty].copy()

    raw_time = df["gurobi_time"]
    numeric = pd.to_numeric(raw_time, errors="coerce")
    is_timeout = raw_time.astype(str).str.strip() == "time_out"
    numeric = numeric.where(~is_timeout, df["time_limit"])
    out = df[["paper_id", "instance"]].copy()
    out["instance"] = out["instance"].replace(GUROBI_INSTANCE_ALIASES)
    out["gurobi_time"] = numeric
    out["gurobi_time_limit"] = pd.to_numeric(df["time_limit"], errors="coerce")
    out["_has_time"] = out["gurobi_time"].notna()
    out = (
        out.sort_values(["paper_id", "instance", "_has_time"], ascending=[True, True, False])
        .drop_duplicates(["paper_id", "instance"], keep="first")
        .drop(columns=["_has_time"])
        .reset_index(drop=True)
    )
    return out


def load_gurobi_large_json(data_dir: Path) -> pd.DataFrame:
    """Read Gurobi objective/runtime directly from canonical solution JSON."""
    tasks_dir = data_dir / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"FrontierOR tasks directory not found: {tasks_dir}")
    rows = []
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        for index in range(1, 6):
            path = task_dir / "gurobi_solution" / f"large_solution_{index}.json"
            if not path.is_file():
                continue
            try:
                if path.stat().st_size > 8 * 1024 * 1024:
                    output = subprocess.check_output(
                        ["jq", "-c", "{objective_value, runtime}", str(path)], text=True
                    )
                    solution = json.loads(output)
                else:
                    with path.open(encoding="utf-8") as handle:
                        solution = json.load(handle)
            except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
                raise ValueError(f"Cannot read Gurobi reference {path}: {exc}") from exc
            rows.append({
                "paper_id": task_dir.name,
                "instance": f"large_{index}",
                "gurobi_solution": pd.to_numeric(solution.get("objective_value"), errors="coerce"),
                "gurobi_time": pd.to_numeric(solution.get("runtime"), errors="coerce"),
                "gurobi_time_limit": pd.to_numeric(
                    solution.get("time_limit", T_MAX_LARGE), errors="coerce"
                ),
            })
    return pd.DataFrame(rows)


def load_gurobi_references(path: Path) -> pd.DataFrame:
    """Load the compact public reference table and return canonical large rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Gurobi reference table not found: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"task_id", "instance", "objective_value", "runtime", "time_limit"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Gurobi reference table missing columns: {sorted(missing)}")
    frame = frame[frame["instance"].isin(LARGE_INSTANCES)].copy()
    frame = frame.rename(columns={
        "task_id": "paper_id",
        "objective_value": "gurobi_solution",
        "runtime": "gurobi_time",
        "time_limit": "gurobi_time_limit",
    })
    if frame[["paper_id", "instance"]].duplicated().any():
        raise ValueError(f"Duplicate Gurobi reference keys in {path}")
    return frame[[
        "paper_id", "instance", "gurobi_solution", "gurobi_time",
        "gurobi_time_limit",
    ]]


def qte_time_is_fast_enough(
    candidate_time: float,
    gurobi_time: float,
    *,
    candidate_time_limit: float = T_MAX_LARGE,
    gurobi_time_limit: float = T_MAX_LARGE,
    tolerance_seconds: float = DEFAULT_QTE_TIME_TOLERANCE_SECONDS,
    tolerance_fraction: float = DEFAULT_QTE_TIME_TOLERANCE_FRACTION,
) -> bool:
    """Compare budget-capped wall times with bounded clock-jitter tolerance."""
    values = pd.to_numeric(
        pd.Series([
            candidate_time, gurobi_time, candidate_time_limit, gurobi_time_limit
        ]),
        errors="coerce",
    )
    if values.isna().any() or (values <= 0).any():
        return False
    candidate_effective = min(float(values.iloc[0]), float(values.iloc[2]))
    gurobi_effective = min(float(values.iloc[1]), float(values.iloc[3]))
    if tolerance_seconds < 0 or tolerance_fraction < 0:
        raise ValueError("QTE time tolerances must be non-negative")
    tolerance = max(tolerance_seconds, tolerance_fraction * gurobi_effective)
    return candidate_effective <= gurobi_effective + tolerance


def load_gurobi_large(reference_path: Path | None = None) -> pd.DataFrame:
    """Compatibility API used by auxiliary metrics scripts; defaults to Parquet."""
    if reference_path is None:
        data_dir = os.environ.get("FRONTIER_OR_DATA_DIR")
        if not data_dir:
            raise ValueError("Set FRONTIER_OR_DATA_DIR or pass a Gurobi reference path")
        reference_path = Path(data_dir) / "metadata" / "gurobi_references.parquet"
    return load_gurobi_references(Path(reference_path).expanduser().resolve())


def _default_registry_path() -> Path | None:
    candidates = [
        REPO / "configs" / "oneshot.yaml",
        Path.cwd() / "configs" / "oneshot.yaml",
        REPO.parent / "FrontierOR-run" / "configs" / "oneshot.yaml",
    ]
    existing = []
    for path in candidates:
        if not path.is_file() or path in existing:
            continue
        existing.append(path)
        try:
            with path.open(encoding="utf-8") as handle:
                if (yaml.safe_load(handle) or {}).get("model_results"):
                    return path
        except (OSError, yaml.YAMLError):
            continue
    return existing[0] if existing else None


def load_registered_models(registry: Path, eval_dir: Path) -> dict[str, Path]:
    with registry.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    metadata = config.get("model_results") or {}
    models = {}
    complete_counts = {}
    for model_id in config.get("models") or []:
        short_name = str(model_id).rsplit("/", 1)[-1]
        entry = metadata.get(short_name, {})
        if isinstance(entry, str):
            entry = {"csv": entry}
        display_name = entry.get("name", short_name)
        relative_csv = entry.get("csv", f"eval/eval_results_{short_name}.csv")
        csv_path = Path(relative_csv)
        if not csv_path.is_absolute():
            csv_path = eval_dir / csv_path.name
        if csv_path.is_file():
            complete_count = len(papers_with_complete_records(csv_path))
            if not complete_count:
                print(f"Skipping registered model without complete task records: {short_name} ({csv_path})")
                continue
            if display_name in models and complete_counts[display_name] >= complete_count:
                print(f"Skipping less complete duplicate {display_name}: {csv_path}")
                continue
            if display_name in models:
                print(f"Replacing less complete duplicate {display_name}: {models[display_name]}")
            models[display_name] = csv_path
            complete_counts[display_name] = complete_count
        else:
            print(f"Skipping registered model without results CSV: {short_name} ({csv_path})")
    return models


def resolve_models(args: argparse.Namespace) -> dict[str, Path]:
    if args.model_csv:
        names = args.model_name or []
        if names and len(names) != len(args.model_csv):
            raise ValueError("--model-name and --model-csv must be supplied the same number of times")
        if not names:
            names = [re.sub(r"^eval_results_|\.csv$", "", Path(path).name) for path in args.model_csv]
        return {name: Path(path).expanduser().resolve() for name, path in zip(names, args.model_csv)}
    registry = Path(args.model_registry).expanduser().resolve() if args.model_registry else _default_registry_path()
    if registry is None:
        raise FileNotFoundError("No model registry found; pass --model-registry or --model-csv")
    return load_registered_models(registry, Path(args.eval_dir).expanduser().resolve())


def papers_with_complete_records(model_csv: Path) -> set[str]:
    """Return paper_ids for which this model has a row for tiny AND every
    large_* instance. (Status / feasibility do not matter; only that the
    evaluation actually attempted the cell.)"""
    df = pd.read_csv(model_csv)
    required = {"tiny", *LARGE_INSTANCES}
    counts = df.groupby("paper_id")["instance"].agg(set)
    return set(counts[counts.apply(lambda s: required.issubset(s))].index)


def auto_anchor_paper_set(active_models: dict) -> set[str]:
    """Intersect the per-model complete-record sets across active models."""
    per_model = {name: papers_with_complete_records(csv) for name, csv in active_models.items()}
    sizes = {name: len(s) for name, s in per_model.items()}
    print("Per-model complete-record counts:")
    for name, n in sizes.items():
        print(f"  {name:24s} {n:4d}")
    intersected = set.intersection(*per_model.values()) if per_model else set()
    return intersected


def load_canonical_hard_set(anchor: set[str] | None = None) -> set[str]:
    """Load the fixed, model-independent Hard task IDs used by the paper."""
    if _HARD_SET_PATH is None:
        raise ValueError("Hard-set metadata path is not configured")
    payload = json.loads(_HARD_SET_PATH.read_text())
    ordered = payload["task_ids"]
    if len(ordered) != HARD_TOP_K or len(set(ordered)) != HARD_TOP_K:
        raise ValueError(
            f"Expected {HARD_TOP_K} unique IDs in {_HARD_SET_PATH}, found "
            f"{len(ordered)} rows / {len(set(ordered))} unique IDs"
        )
    hard = set(ordered)
    return hard if anchor is None else hard & anchor


def compute_metrics(
    model_csv: Path,
    paper_set: set[str],
    gurobi: pd.DataFrame,
    *,
    qte_time_tolerance_seconds: float = DEFAULT_QTE_TIME_TOLERANCE_SECONDS,
    qte_time_tolerance_fraction: float = DEFAULT_QTE_TIME_TOLERANCE_FRACTION,
) -> dict:
    df = pd.read_csv(model_csv)
    df = df[df["paper_id"].isin(paper_set)].copy()

    # Success rate ----------------------------------------------------------
    tiny = df[df["instance"] == "tiny"]
    success_rate = (tiny["status"] == "pass").sum() / len(paper_set)

    # Execution rates -------------------------------------------------------
    # execution_rate: the final tiny program completed candidate execution
    # within the configured five-debug budget.  A checker/model-quality
    # failure still means the Python program executed; runtime_error does not.
    #
    # first_execution_rate: the initially generated tiny program completed
    # candidate execution before any debug attempt.  first_* is preserved by
    # reuse-code runs and is therefore the authoritative first-attempt record.
    debug_col = pd.to_numeric(tiny["debug_retries"], errors="coerce").fillna(0)
    final_executed = tiny["status"].notna() & tiny["fail_reason"].ne("runtime_error")
    execution_rate = (final_executed & debug_col.le(5)).sum() / len(paper_set)

    first_executed = tiny["first_status"].notna() & tiny["first_fail_reason"].ne("runtime_error")
    first_execution_rate = first_executed.sum() / len(paper_set)

    # Build full (paper, large_*) grid, left-join model + gurobi --------
    # The one-shot protocol is tiny-gated: if a paper/model fails tiny, its
    # large rows are not eligible for feasibility, quality, runtime, or QTE
    # metrics even if historical artifacts/CSV rows exist.
    tiny_pass_papers = set(tiny.loc[tiny["status"] == "pass", "paper_id"])
    grid = pd.DataFrame(
        [(p, i) for p in sorted(paper_set) for i in LARGE_INSTANCES],
        columns=["paper_id", "instance"],
    )
    large = df[
        df["instance"].str.startswith("large_")
        & df["paper_id"].isin(tiny_pass_papers)
    ]
    merged = grid.merge(large, on=["paper_id", "instance"], how="left")
    merged = merged.merge(gurobi, on=["paper_id", "instance"], how="left")

    # Feasibility -----------------------------------------------------------
    feas_col = merged["feasible"]
    if feas_col.dtype == object:
        feas_col = feas_col.map({True: True, False: False, "True": True, "False": False})
    merged["is_feasible"] = feas_col.eq(True)
    feasibility = merged["is_feasible"].mean()

    feas = merged[merged["is_feasible"]].copy()

    # Solution quality (NEW DEFINITION) ----------------------------------------
    # Per-cell binary: 1 iff the cell is feasible AND gap <= 1% (== 0.01);
    # aggregated as mean over the full (paper, large_*) grid. Infeasible cells
    # count as 0 even when their (invalid) objective happens to yield gap <= 1%
    # -- an infeasible solution has no valid objective. NaN gap -> False -> 0.
    import numpy as np
    gap_arr_full = pd.to_numeric(merged["gap"], errors="coerce")
    obj_arr_full = pd.to_numeric(merged["obj"], errors="coerce")
    zero_tol = pd.Series(
        [
            PROVEN_ZERO_OPTIMUM_TOLERANCE.get((paper_id, instance))
            for paper_id, instance in zip(merged["paper_id"], merged["instance"])
        ],
        index=merged.index,
        dtype="float64",
    )
    is_proven_zero = zero_tol.notna()
    zero_quality_ok = (obj_arr_full.abs() <= zero_tol) & merged["is_feasible"]
    relative_quality_ok = (gap_arr_full <= 0.01) & merged["is_feasible"]
    quality_ok_1pct = relative_quality_ok.where(~is_proven_zero, zero_quality_ok).fillna(False)
    sol_quality = float(quality_ok_1pct.mean())

    # Legacy diagnostics for the previous signed-gap definition (kept for
    # debugging; not surfaced in the LaTeX table any more).
    gap_vals = feas["gap"].replace([np.inf, -np.inf], np.nan).dropna()
    gap_clipped = gap_vals.clip(-1.0, 1.0)
    sol_quality_signed_gap_clipped = gap_clipped.mean()
    sol_quality_raw_mean = gap_vals.mean()
    sol_quality_median = gap_vals.median()
    n_gap_outliers = int((gap_vals.abs() > 1.0).sum())

    # Solve-time ratio: geomean of (gurobi_time / effective_model_time). -----
    # We include two kinds of cells:
    #   (a) feasible cells -> effective_time = actual model time
    #   (b) cells where the LLM-generated program ran out of time without
    #       producing a solution (error message contains "timed out")
    #       -> effective_time = T_max (the full 1h budget)
    # All other failure modes (constraint-violation infeasibility, runtime
    # crashes other than timeout, missing rows, gate_fail) are EXCLUDED from
    # the ratio because their solve time is not informative.
    err_str = merged["error"].astype(str)
    timeout_no_sol = (~merged["is_feasible"]) & err_str.str.contains("timed out", case=False, na=False)
    eff_time = pd.Series(np.nan, index=merged.index)
    eff_time.loc[merged["is_feasible"]] = merged.loc[merged["is_feasible"], "time"]
    eff_time.loc[timeout_no_sol] = T_MAX_LARGE
    valid_mask = (merged["gurobi_time"] > 0) & eff_time.notna() & (eff_time > 0)
    ratios = (merged.loc[valid_mask, "gurobi_time"] / eff_time[valid_mask]).to_numpy()
    if len(ratios) == 0:
        solve_time_ratio = float("nan")
    else:
        solve_time_ratio = float(np.exp(np.log(ratios).mean()))
    solve_time_ratio_arith = float(ratios.mean()) if len(ratios) else float("nan")
    n_ratio_cells = int(valid_mask.sum())
    n_timeout_cells = int(timeout_no_sol.sum())

    # QTE -------------------------------------------------------------------
    def qte_row(r) -> float:
        if not r["is_feasible"]:
            return 0.0
        g = max(r["gap"], 0.0) if pd.notna(r["gap"]) else 0.0
        t = r["time"]
        tg = r["gurobi_time"]
        if pd.isna(t) or pd.isna(tg) or tg <= 0:
            return 0.0
        return max(0.0, 1.0 - g * t / tg)
    merged["qte"] = merged.apply(qte_row, axis=1)
    qte = merged["qte"].mean()

    # Beat-Gurobi indicators ------------------------------------------------
    # Per-cell binary: 1 iff the cell is feasible AND gap <= threshold AND
    # its budget-capped wall time is no slower than Gurobi within the bounded
    # clock-jitter tolerance. Infeasible cells count as 0 (an infeasible solution
    # cannot "beat" Gurobi regardless of its invalid objective / time).
    # Missing/invalid quality or time -> False -> 0. Aggregate as mean over the full
    # (paper, large_*) grid so the denominator equals 5 * |paper_set|.
    gap_arr = pd.to_numeric(merged["gap"], errors="coerce")
    candidate_times = pd.to_numeric(merged["time"], errors="coerce")
    gurobi_times = pd.to_numeric(merged["gurobi_time"], errors="coerce")
    gurobi_limits = pd.to_numeric(
        merged["gurobi_time_limit"], errors="coerce"
    ).fillna(T_MAX_LARGE)
    candidate_effective = candidate_times.clip(upper=T_MAX_LARGE)
    gurobi_effective = gurobi_times.where(
        gurobi_times <= gurobi_limits, gurobi_limits
    )
    time_tolerance = (
        qte_time_tolerance_fraction * gurobi_effective
    ).clip(lower=qte_time_tolerance_seconds)
    fast_enough = (
        candidate_effective.notna()
        & gurobi_effective.notna()
        & candidate_effective.gt(0)
        & gurobi_effective.gt(0)
        & candidate_effective.le(gurobi_effective + time_tolerance)
    )
    beat_metrics = {}
    for thr_pct, thr_val in [(0, 0.0), (1, 0.01), (5, 0.05), (10, 0.10)]:
        relative_quality_ok = (gap_arr <= thr_val) & merged["is_feasible"]
        quality_ok = relative_quality_ok.where(
            ~is_proven_zero, zero_quality_ok
        ).fillna(False)
        beat = (quality_ok & fast_enough).fillna(False)
        beat_metrics[f"beat_gurobi_{thr_pct}"] = float(beat.mean())

    return {
        "success_rate":            success_rate,
        "execution_rate":          execution_rate,
        "first_execution_rate":    first_execution_rate,
        "feasibility":             feasibility,
        "solution_quality":        sol_quality,             # clipped mean (primary)
        "solution_quality_raw":    sol_quality_raw_mean,    # un-clipped (diagnostic)
        "solution_quality_median": sol_quality_median,
        "n_gap_outliers":          n_gap_outliers,
        "solve_time_ratio_geom":   solve_time_ratio,
        "solve_time_ratio_arith":  solve_time_ratio_arith,
        "n_ratio_cells":           n_ratio_cells,
        "n_timeout_cells":         n_timeout_cells,
        "qte":                     qte,
        "n_feasible_cells":        int(merged["is_feasible"].sum()),
        **beat_metrics,
    }


def format_latex(table: pd.DataFrame, n_papers: int) -> str:
    """Render a model x metric LaTeX table (rows=models, cols=metrics)."""
    metric_specs = [
        (r"Tiny pass $\uparrow$",        "success_rate",          "{:.2f}"),
        (r"Feasibility $\uparrow$",      "feasibility",           "{:.2f}"),
        (r"Solution quality $\uparrow$", "solution_quality",      "{:.2f}"),
        (r"Solve-time ratio $\uparrow$", "solve_time_ratio_geom", "{:.2f}"),
        (r"QTE $\uparrow$",              "beat_gurobi_1",         "{:.2f}"),
    ]
    header = (
        " & ".join([r"\textbf{Model}"] + [r"\textbf{" + lbl + "}" for lbl, _, _ in metric_specs])
        + r" \\"
    )
    models = list(table.columns)
    rows = []
    for m in models:
        cells = [fmt.format(table.loc[key, m]) for _, key, fmt in metric_specs]
        rows.append(r"\textit{" + m + "} & " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    column_spec = "l" + "c" * len(metric_specs)
    out = (
        r"\begin{table}[t]" "\n"
        r"\caption{One-shot performance of the active base models on the "
        f"{n_papers}-paper benchmark subset (papers with complete tiny + 5 large "
        f"records across all active models). "
        r"\emph{Optimality gap} is the mean relative gap to Gurobi over "
        r"feasible cells (negative = strictly better than Gurobi). "
        r"\emph{Solve-time ratio} is the geometric mean of $\tau_g/t_{\text{solve}}$ "
        r"over feasible cells plus LLM-program-timeout cells (where "
        f"$t_{{\\text{{solve}}}}{{=}}T_{{\\max}}{{=}}{int(T_MAX_LARGE)}$\\,s); "
        r"other failure modes are excluded. "
        r"\emph{QTE} is the binary quality-time pass rate: the fraction of "
        r"(paper, large-instance) cells that are feasible, within a $1\%$ gap "
        r"to Gurobi, and faster than Gurobi (infeasible cells count as $0$).}" "\n"
        r"\label{tab:oneshot-main}" "\n"
        r"\centering" "\n"
        r"\scriptsize" "\n"
        r"\setlength{\tabcolsep}{4pt}" "\n"
        r"\begin{threeparttable}" "\n"
        r"\begin{tabular}{" + column_spec + "}\n"
        r"\toprule" "\n"
        + header + "\n"
        r"\midrule" "\n"
        + body + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{threeparttable}" "\n"
        r"\end{table}" "\n"
    )
    return out


def load_standard_hard_split(anchor: set[str]) -> tuple[set[str], set[str]]:
    """Standard Set = the full anchor (every paper with complete records across
    all active models). Hard Set = the fixed rule-based 50-task list
    intersected with the anchor."""
    canonical_hard = load_canonical_hard_set()
    hard = canonical_hard & anchor
    standard = set(anchor)
    print(
        f"\nStandard/Hard split:\n"
        f"  Standard Set : full anchor                            = {len(standard)}\n"
        f"  Hard Set     : fixed rule-based top-{HARD_TOP_K} ∩ anchor = {len(hard)}\n"
        f"  Canonical Hard IDs absent from anchor                    = {len(canonical_hard - anchor)}"
    )
    return standard, hard


def format_latex_split(standard_df: pd.DataFrame, hard_df: pd.DataFrame, n_standard: int, n_hard: int) -> str:
    """Side-by-side Standard / Hard table (rows = models, two column groups of 4)."""
    metric_specs = [
        (r"Exec. rate $\uparrow$",       "execution_rate",        "{:.2f}"),
        (r"Feas. $\uparrow$",            "feasibility",           "{:.2f}"),
        (r"Sol. quality $\uparrow$",     "solution_quality",      "{:.2f}"),
        (r"QTE $\uparrow$",              "beat_gurobi_1",         "{:.2f}"),
    ]
    metric_header = " & ".join(r"\textbf{" + lbl + "}" for lbl, _, _ in metric_specs)
    header_top = (
        r"\textbf{Model} & "
        + r"\multicolumn{4}{c|}{\textbf{Standard Set} ($n=" + str(n_standard) + r"$)} & "
        + r"\multicolumn{4}{c}{\textbf{Hard Set} ($n=" + str(n_hard) + r"$)} \\"
    )
    header_mid = r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}"
    header_metrics = r" & " + metric_header + " & " + metric_header + r" \\"

    models = list(standard_df.columns)
    body_rows = []
    for m in models:
        standard_cells = [fmt.format(standard_df.loc[key, m]) for _, key, fmt in metric_specs]
        hard_cells = [fmt.format(hard_df.loc[key, m]) for _, key, fmt in metric_specs]
        body_rows.append(
            r"\textit{" + m + "} & " + " & ".join(standard_cells)
            + " & " + " & ".join(hard_cells) + r" \\"
        )
    body = "\n".join(body_rows)
    column_spec = "l" + "cccc" + "|" + "cccc"
    return (
        r"\begin{table}[t]" "\n"
        r"\caption{One-shot performance on the FrontierOR Standard / Hard split "
        f"(Standard: full anchor, $n={n_standard}$ papers; Hard: top-{HARD_TOP_K} "
        f"Gurobi-saturated tasks, $n={n_hard}$ papers). Metric definitions follow "
        r"Section~\ref{subsec:eval-metrics}.}" "\n"
        r"\label{tab:oneshot-standard-hard}" "\n"
        r"\centering" "\n"
        r"\scriptsize" "\n"
        r"\setlength{\tabcolsep}{4pt}" "\n"
        r"\begin{tabular}{" + column_spec + "}\n"
        r"\toprule" "\n"
        + header_top + "\n"
        + header_mid + "\n"
        + header_metrics + "\n"
        r"\midrule" "\n"
        + body + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )


def _compute_for_set(
    models: dict[str, Path],
    paper_set: set[str],
    gurobi: pd.DataFrame,
    *,
    qte_time_tolerance_seconds: float = DEFAULT_QTE_TIME_TOLERANCE_SECONDS,
    qte_time_tolerance_fraction: float = DEFAULT_QTE_TIME_TOLERANCE_FRACTION,
) -> pd.DataFrame:
    return pd.DataFrame({
        name: compute_metrics(
            csv,
            paper_set,
            gurobi,
            qte_time_tolerance_seconds=qte_time_tolerance_seconds,
            qte_time_tolerance_fraction=qte_time_tolerance_fraction,
        )
        for name, csv in models.items()
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-registry", help="one_shot YAML registry; auto-detected by default")
    parser.add_argument("--eval-dir", default=str(EVAL_DIR), help="directory containing model result CSVs")
    parser.add_argument("--model-csv", action="append", help="explicit model result CSV; repeat for multiple models")
    parser.add_argument("--model-name", action="append", help="display name paired with --model-csv")
    parser.add_argument(
        "--gurobi-source", choices=("references", "solutions", "csv"), default="references",
        help="Gurobi reference source (default: compact Parquet reference table)",
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("FRONTIER_OR_DATA_DIR"),
        help="FrontierOR dataset root containing tasks/; required for --gurobi-source solutions",
    )
    parser.add_argument(
        "--gurobi-reference",
        help="Parquet/CSV reference table; defaults to <data-dir>/metadata/gurobi_references.parquet",
    )
    parser.add_argument("--hard-set", help="hard split JSON; defaults to <data-dir>/metadata/splits/hard.json")
    parser.add_argument(
        "--gurobi-csv-dir", default=str(REPO),
        help="directory containing gurobi_results_1.csv ... gurobi_results_5.csv",
    )
    parser.add_argument(
        "--qte-time-tolerance-seconds", type=float,
        default=DEFAULT_QTE_TIME_TOLERANCE_SECONDS,
        help="absolute wall-clock tolerance for binary QTE (default: 0.01 seconds)",
    )
    parser.add_argument(
        "--qte-time-tolerance-fraction", type=float,
        default=DEFAULT_QTE_TIME_TOLERANCE_FRACTION,
        help="relative wall-clock tolerance for binary QTE (default: 0.001)",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).parent))
    return parser.parse_args()


def main() -> None:
    global _HARD_SET_PATH
    args = parse_args()
    if args.qte_time_tolerance_seconds < 0 or args.qte_time_tolerance_fraction < 0:
        raise ValueError("QTE time tolerances must be non-negative")
    models = resolve_models(args)
    if not models:
        raise ValueError("No model result CSVs were selected")
    if args.hard_set:
        _HARD_SET_PATH = Path(args.hard_set).expanduser().resolve()
    elif args.data_dir:
        _HARD_SET_PATH = (
            Path(args.data_dir).expanduser().resolve() / "metadata" / "splits" / "hard.json"
        )
    else:
        raise ValueError("Pass --data-dir or --hard-set to locate hard split metadata")
    # Use the strict common intersection: a paper is counted only if every
    # evaluated model has a row for tiny + 5 large instances.
    paper_set = auto_anchor_paper_set(models)
    n_papers = len(paper_set)
    print(f"\nAnchor paper set (intersection of all {len(models)} models): {n_papers} papers")

    if args.gurobi_source == "csv":
        gurobi = load_gurobi_large_csv(Path(args.gurobi_csv_dir).expanduser().resolve())
    elif args.gurobi_source == "solutions":
        if not args.data_dir:
            raise ValueError("Set FRONTIER_OR_DATA_DIR or pass --data-dir for solution JSON references")
        gurobi = load_gurobi_large_json(Path(args.data_dir).expanduser().resolve())
    else:
        if args.gurobi_reference:
            reference_path = Path(args.gurobi_reference).expanduser().resolve()
        elif args.data_dir:
            reference_path = Path(args.data_dir).expanduser().resolve() / "metadata" / "gurobi_references.parquet"
        else:
            raise ValueError("Set FRONTIER_OR_DATA_DIR, pass --data-dir, or pass --gurobi-reference")
        gurobi = load_gurobi_references(reference_path)

    metric_options = {
        "qte_time_tolerance_seconds": args.qte_time_tolerance_seconds,
        "qte_time_tolerance_fraction": args.qte_time_tolerance_fraction,
    }
    df = _compute_for_set(models, paper_set, gurobi, **metric_options)
    print("\n=== Raw numbers (overall anchor set) ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "compute_benchmark_main_metrics.csv"
    df.to_csv(out_csv)
    print(f"\nSaved raw metrics to {out_csv}")

    print("\n=== LaTeX table (overall) ===\n")
    print(format_latex(df, n_papers))

    # ------------------------------------------------------------------
    # Standard / Hard split  (Standard = full anchor; Hard = top-K subset)
    # ------------------------------------------------------------------
    standard_set, hard_set = load_standard_hard_split(paper_set)
    df_standard = df  # already computed on the full anchor above
    df_hard = _compute_for_set(models, hard_set, gurobi, **metric_options)

    out_standard_csv = output_dir / "compute_benchmark_main_metrics_standard.csv"
    out_hard_csv = output_dir / "compute_benchmark_main_metrics_hard.csv"
    df_standard.to_csv(out_standard_csv)
    df_hard.to_csv(out_hard_csv)
    print(f"\nSaved standard metrics to {out_standard_csv}")
    print(f"Saved hard metrics to {out_hard_csv}")

    print("\n=== LaTeX table (Standard / Hard split) ===\n")
    print(format_latex_split(df_standard, df_hard, len(standard_set), len(hard_set)))


if __name__ == "__main__":
    main()
