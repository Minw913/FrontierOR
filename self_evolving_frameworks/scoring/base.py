"""Scoring interface shared by all Stage 2 scoring schemes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScoreContext:
    """Per-(paper, instance) context needed by a Scorer.

    Attributes:
        time_limit:   stage2 time limit T (seconds LLM code is allowed to run).
        gurobi_time:  τ_g. Gurobi's actual solve time on this instance. Case A:
                      time-to-proven-optimal. Case B: Gurobi's time_limit (timeout).
                      None if not available.
        gurobi_obj:   Reference objective value. Case A: proven optimal. Case B:
                      Gurobi's best incumbent at timeout. None if not available.
        direction:    "min" or "max". Needed so "LLM beats Gurobi" is scored
                      correctly regardless of problem direction.
        log_path:     Filesystem path to the LLM's convergence log JSONL.
        paper_id:     For debugging and lookup fallbacks.
        instance:     For debugging and lookup fallbacks.
        extra:        Scorer-specific extra parameters.
    """
    time_limit: int
    gurobi_time: Optional[float] = None
    gurobi_obj: Optional[float] = None
    direction: str = "min"
    log_path: Optional[str] = None
    paper_id: Optional[str] = None
    instance: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Scorer:
    """Stateless stage2 scorer. Subclasses implement score_instance()."""

    name: str = "base"

    def score_instance(self, result: Dict[str, Any], ctx: ScoreContext) -> Tuple[float, Dict[str, Any]]:
        """Score a single (paper, instance) evaluation result.

        Args:
            result: per-instance result dict produced by eval_core.evaluate_candidate_code.
                    Expected keys (not all required by every scorer):
                        feasible, gap, aocc, llm_obj, gurobi_obj, solve_time, error
            ctx:    ScoreContext (see above)

        Returns:
            (score ∈ [0, 1], debug_metrics) — debug_metrics is a flat dict of
            scalar or serialisable fields to be written to CSV.
        """
        raise NotImplementedError

    def aggregate(self, per_instance_scores: List[float]) -> float:
        """Aggregate per-instance scores into stage2 combined_score. Default = mean."""
        valid = [s for s in per_instance_scores if s is not None]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)
