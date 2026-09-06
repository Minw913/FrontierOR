"""Compatibility alias for the packaged CORAL instructions."""

import sys
from trusted_eval_infra.agent import instructions as _implementation

sys.modules[__name__] = _implementation
