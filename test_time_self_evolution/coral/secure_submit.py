"""Compatibility alias for the packaged agent submit command."""

import sys
from trusted_eval_infra.agent import submit as _implementation

sys.modules[__name__] = _implementation
