"""Compatibility alias for the packaged CORAL instructions."""

import sys
from frontieror.infra.agent import instructions as _implementation

sys.modules[__name__] = _implementation
