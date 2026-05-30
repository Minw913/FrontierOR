"""Pluggable Stage 2 scoring schemes.

Public API:
    - get_scorer(name, **kwargs) -> Scorer
    - SCORERS registry dict
    - ScoreContext dataclass
    - Scorer base class

Two implementations:
    - "staged_qte"  (default) — two-stage piecewise QTE with anchors at
                                 0/0.8/1.0/2.0; beat-Gurobi enters via signed
                                 gap. See scoring/staged_qte.py.
    - "aocc"        (backup)  — pure 1 − AOCC, anytime baseline. Range [0, 1].
"""

from __future__ import annotations

from typing import Dict, Type

from .aocc import AoccScorer
from .base import Scorer, ScoreContext
from .staged_qte import StagedQteScorer

SCORERS: Dict[str, Type[Scorer]] = {
    StagedQteScorer.name: StagedQteScorer,
    AoccScorer.name: AoccScorer,
}

DEFAULT_SCORER = "staged_qte"


def get_scorer(name: str, **kwargs) -> Scorer:
    cls = SCORERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown scorer: {name!r}. Available: {sorted(SCORERS)}")
    return cls(**kwargs)


__all__ = [
    "Scorer",
    "ScoreContext",
    "SCORERS",
    "DEFAULT_SCORER",
    "get_scorer",
    "StagedQteScorer",
    "AoccScorer",
]
