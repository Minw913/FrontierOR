"""Compatibility alias for the packaged trusted grader."""

import sys
from infra.agent import grader as _implementation

sys.modules[__name__] = _implementation
