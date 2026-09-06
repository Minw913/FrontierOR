"""Trusted evaluation infrastructure for FrontierOR.

The upstream research pipeline remains under ``one_shot_eval.py`` and
``test_time_self_evolution``.  This package owns the additional trust boundary
needed when the code-producing system is not trusted.
"""

from .contracts import public_scoring_contract, visibility_contract
from .policy import AgentModePolicy, EvaluationProfile

__all__ = [
    "AgentModePolicy",
    "EvaluationProfile",
    "public_scoring_contract",
    "visibility_contract",
]
