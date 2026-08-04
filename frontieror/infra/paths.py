"""Repository paths shared by the infrastructure package."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_CHECKER_ROOT = REPO_ROOT / "trusted_checkers"
