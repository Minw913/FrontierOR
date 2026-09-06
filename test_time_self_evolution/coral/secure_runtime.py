"""Compatibility alias for the packaged agent runtime."""

import sys
from trusted_eval_infra.agent import runtime as _implementation

sys.modules[__name__] = _implementation
