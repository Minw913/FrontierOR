"""Compatibility alias for the packaged trusted grader."""

import sys
from trusted_eval_infra.agent import grader as _implementation

sys.modules[__name__] = _implementation
