"""Compatibility imports for the pre-package anti-hack module path."""

from frontieror.infra.checkers import validate_objective_checker
from frontieror.infra.policy import (
    _resolve_docker_image_cached,
    validate_anti_hack_runtime,
    with_anti_hack_exec_cfg,
)
from frontieror.infra.visibility import (
    PublicPaperView,
    materialize_public_paper_view,
)

__all__ = [
    "PublicPaperView",
    "materialize_public_paper_view",
    "_resolve_docker_image_cached",
    "validate_anti_hack_runtime",
    "validate_objective_checker",
    "with_anti_hack_exec_cfg",
]
