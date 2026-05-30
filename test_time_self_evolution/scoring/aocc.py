"""Backup Stage 2 scorer: pure 1 − AOCC.

AOCC = Area Over the Convergence Curve = (1/T)·∫_0^T gap(t) dt ∈ [0, 1].
Lower AOCC = faster convergence + better final quality. We report
``1 − AOCC`` so higher is better.

This is the classic "anytime performance" metric (smaller area =
better). Gurobi-time-independent: works even when τ_g is unavailable.

Use as ablation baseline when comparing against the staged_qte primary
scorer. Range strictly [0, 1] — no beat-Gurobi bonus baked in.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import Scorer, ScoreContext


class AoccScorer(Scorer):
    """Pure ``1 − AOCC`` scorer."""

    name = "aocc"

    def score_instance(
        self, result: Dict[str, Any], ctx: ScoreContext
    ) -> Tuple[float, Dict[str, Any]]:
        if result.get("feasible") is not True:
            return 0.0, {"reason": "infeasible", "aocc": 1.0}

        aocc = result.get("aocc")
        if aocc is None:
            aocc = 1.0
        aocc_clipped = max(0.0, min(float(aocc), 1.0))
        score = 1.0 - aocc_clipped
        return round(score, 6), {
            "aocc": round(aocc_clipped, 6),
        }
