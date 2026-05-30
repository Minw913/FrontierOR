"""Primary Stage 2 scorer: Staged QTE (quality-time-efficiency).

Two-stage piecewise score parameterized by the Stage1/Stage2 boundary
``b`` (= ``STAGE_BOUNDARY`` below; default 0.01 = 1% gap):

    if signed_gap > b:           # Stage 1: gap above the quality bar
        score = max(0, 1 - signed_gap)              ∈ [0, 1 - b]

    else:                        # Stage 2: gap close to (or beats) Gurobi
        score = (1 - signed_gap) + max(0, (τ_g - t_solve) / τ_g)
                                                    ∈ [1 - b, 2 + |beat_amount|]

Anchors (independent of ``b``):
    0.0  = LLM equally bad as the worst case (gap ≥ 1 OR infeasible)
    1-b  = LLM gap = b (just inside Stage 2 boundary, slow); concretely with
           b=0.01 this is 0.99
    1.0  = LLM matches Gurobi exactly + same wall time (parity)
    2.0  = LLM matches Gurobi + instantaneous solve (perfect)
    >2.0 = LLM strictly beats Gurobi (signed_gap < 0); upper bound = 2 + beat_amount

The "beat-Gurobi" bonus enters naturally via the signed (unclipped) relative
gap — no separate beat_amount term, no Stage 3.

Note: this is a weighted-sum form within Stage 2 (quality and speed both
contribute additively). It is NOT strict lexicographic — a fast-but-not-quite-
matched candidate (gap = b/2, t = 0.5 τ_g → score ≈ 1.5 - b/2) can outscore a
slow-but-matched one (gap=0, t=τ_g → score=1.0). This trade-off is intentional:
in practice OR Gurobi often timeouts at incumbent (not proven optimal), so a
fast heuristic with a small but non-zero gap may be more useful than a slow
one that just barely matches the incumbent.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import Scorer, ScoreContext
from .building_blocks import _scaled_denom

STAGE_BOUNDARY = 0.01  # 1% gap: a candidate must clear gap<=1% to enter Stage 2.


def signed_quality_gap(
    obj_llm: float, obj_gurobi: float, direction: str = "min"
) -> float:
    """Direction-aware signed relative gap (unclipped).

    >0  ⇔ LLM worse than Gurobi (standard "gap" semantics)
    =0  ⇔ matched
    <0  ⇔ LLM strictly beats Gurobi

    Uses scaled denominator so near-zero references don't blow up
    (see ``building_blocks._scaled_denom``).
    """
    sign = 1 if direction == "min" else -1
    denom = max(_scaled_denom(obj_gurobi, obj_llm), 1e-10)
    return sign * (obj_llm - obj_gurobi) / denom


class StagedQteScorer(Scorer):
    """Two-stage piecewise QTE scorer (default)."""

    name = "staged_qte"

    def __init__(self, stage_boundary: float = STAGE_BOUNDARY):
        self.stage_boundary = float(stage_boundary)

    def score_instance(
        self, result: Dict[str, Any], ctx: ScoreContext
    ) -> Tuple[float, Dict[str, Any]]:
        if result.get("feasible") is not True:
            return 0.0, self._empty_debug("infeasible")

        if ctx.gurobi_obj is None:
            return 0.0, self._empty_debug("missing_gurobi_baseline")

        obj_llm = result.get("llm_obj")
        if obj_llm is None:
            return 0.0, self._empty_debug("missing_llm_obj")

        # Signed gap: negative when LLM beats Gurobi.
        g = signed_quality_gap(float(obj_llm), float(ctx.gurobi_obj), ctx.direction)

        # Quality term (>1 when LLM beats Gurobi).
        quality = 1.0 - g

        # Clipped gap; negative gap still counts as Stage 2.
        g_pos = max(0.0, g)

        if g_pos > self.stage_boundary:
            # Stage 1: quality bar not met.
            score = max(0.0, quality)
            stage_id = 1
            quality_part = score
            speed_part = 0.0
        else:
            # Stage 2: gap ≤ boundary, includes beat-Gurobi territory
            t_solve_raw = result.get("solve_time")
            tau_g = ctx.gurobi_time
            try:
                t_solve = (
                    float(t_solve_raw) if t_solve_raw is not None
                    else float(ctx.time_limit)
                )
            except (TypeError, ValueError):
                t_solve = float(ctx.time_limit)

            if tau_g is None or tau_g <= 0:
                # No baseline τ_g — fall back to no speed bonus
                speed_part = 0.0
            else:
                speed_part = max(0.0, 1.0 - t_solve / float(tau_g))

            quality_part = quality          # ≥ 1-stage_boundary by stage condition
            score = quality_part + speed_part
            stage_id = 2

        return round(score, 6), {
            "stage_id": stage_id,
            "quality_part": round(quality_part, 6),
            "speed_part": round(speed_part, 6),
            "signed_gap": round(g, 6),
            "beat_amount": round(max(0.0, -g), 6),
            "matched": bool(g_pos < 1e-4),
            "beat_gurobi": bool(g < -1e-4),
        }

    def _empty_debug(self, reason: str) -> Dict[str, Any]:
        return {
            "stage_id": 0,
            "quality_part": 0.0,
            "speed_part": 0.0,
            "signed_gap": 1.0,
            "beat_amount": 0.0,
            "matched": False,
            "beat_gurobi": False,
            "reason": reason,
        }
