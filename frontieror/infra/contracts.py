"""Versioned public contracts for official FrontierOR evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


CONTRACT_SCHEMA_VERSION = 1
SCORING_CONTRACT_VERSION = "staged-qte-v1"
NEAR_ZERO_REFERENCE = 1e-3
INSTANCE_SCORE_DECIMALS = 6
CANDIDATE_SHUTDOWN_RESERVE_SECONDS = 2


class Visibility(StrEnum):
    """When an artifact is visible to the submitted system."""

    PUBLIC = "public"
    DEV_WORKSPACE = "dev_workspace"
    SOLVER_RUNTIME_ONLY = "solver_runtime_only"
    AGGREGATE_AFTER_RUN = "aggregate_after_run"
    TRUSTED_ONLY = "trusted_only"


def public_scoring_contract(*, stage_boundary: float = 0.01) -> dict[str, Any]:
    """Return the scorer definition without any instance-specific baselines.

    A benchmark should disclose what it optimizes.  Security comes from
    withholding final instances, reference objectives, reference runtimes, and
    the checker implementation, not from obscuring the score equation.
    """
    boundary = float(stage_boundary)
    if not 0.0 <= boundary < 1.0:
        raise ValueError("stage_boundary must be in [0, 1)")
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_version": SCORING_CONTRACT_VERSION,
        "scorer": "staged_qte",
        "optimization_direction": "maximize_score",
        "instance_score": {
            "infeasible_or_invalid": 0.0,
            "variables": {
                "c": "candidate_objective",
                "r": "reference_objective",
                "t": "trusted_host_wall_time_seconds",
                "tau": "reference_runtime_seconds",
                "b": boundary,
            },
            "scale": {
                "symbol": "D",
                "near_zero_reference": NEAR_ZERO_REFERENCE,
                "formula": (
                    "abs(r) if abs(r)>=0.001 else max(abs(c),0.001)"
                ),
            },
            "signed_gap": {
                "symbol": "g",
                "minimize": "(c-r)/D",
                "maximize": "(r-c)/D",
                "meaning": "negative means the candidate beats the reference",
            },
            "stage_boundary": boundary,
            "quality_only": "max(0,1-g) when max(0,g)>b",
            "quality_and_speed": (
                "(1-g)+max(0,1-t/tau) when max(0,g)<=b and tau>0"
            ),
            "missing_or_nonpositive_reference_runtime": (
                "speed term is 0; score is 1-g when max(0,g)<=b"
            ),
            "rounding": {
                "per_instance_decimal_places": INSTANCE_SCORE_DECIMALS,
                "mode": "Python round (round-half-to-even)",
            },
        },
        "aggregation": "arithmetic_mean_over_declared_instances",
        "aggregation_details": {
            "missing_or_failed_instance_score": 0.0,
        },
        "runtime_measurement": "trusted_host_wall_clock",
        "runtime_budget": {
            "host_hard_deadline": "declared_time_limit_seconds",
            "candidate_argument": "max(1, floor(declared_time_limit_seconds)-2)",
            "serialization_reserve_seconds": CANDIDATE_SHUTDOWN_RESERVE_SECONDS,
            "compute_grace_seconds": 0,
        },
        "candidate_reported_timestamps_trusted": False,
        "private_parameters": [
            "reference_objective",
            "reference_runtime",
            "checker_implementation",
            "final_instance_membership",
        ],
    }


def visibility_contract() -> dict[str, Any]:
    """Return the role-based artifact visibility policy."""
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "artifacts": {
            "problem_description": Visibility.PUBLIC,
            "instance_schema": Visibility.PUBLIC,
            "solution_schema": Visibility.PUBLIC,
            "scoring_formula": Visibility.PUBLIC,
            "dev_instance_json": Visibility.DEV_WORKSPACE,
            "dev_aggregate_feedback": Visibility.DEV_WORKSPACE,
            "dev_per_instance_score": Visibility.TRUSTED_ONLY,
            "final_instance_json": Visibility.SOLVER_RUNTIME_ONLY,
            "final_aggregate_score": Visibility.AGGREGATE_AFTER_RUN,
            "final_per_instance_score": Visibility.TRUSTED_ONLY,
            "final_instance_membership": Visibility.TRUSTED_ONLY,
            "final_instance_provenance": Visibility.TRUSTED_ONLY,
            "reference_solution": Visibility.TRUSTED_ONLY,
            "reference_objective": Visibility.TRUSTED_ONLY,
            "reference_runtime": Visibility.TRUSTED_ONLY,
            "feasibility_checker": Visibility.TRUSTED_ONLY,
            "private_trace": Visibility.TRUSTED_ONLY,
            "candidate_stdout_stderr": Visibility.TRUSTED_ONLY,
        },
        "final_runtime_rule": (
            "After code.py is frozen and the agent system is stopped, the "
            "candidate solver receives one read-only final instance at an "
            "opaque path. No score or reference data is returned to the agent."
        ),
        "final_data_requirement": (
            "Final instances are unpublished server-only artifacts and are "
            "content-disjoint from every stage-1/dev instance."
        ),
    }
