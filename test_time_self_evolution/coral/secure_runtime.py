"""Compatibility alias for the packaged agent runtime."""

import sys
from frontieror.infra.agent import runtime as _implementation

sys.modules[__name__] = _implementation
