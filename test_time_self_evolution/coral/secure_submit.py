"""Compatibility alias for the packaged agent submit command."""

import sys
from infra.agent import submit as _implementation

sys.modules[__name__] = _implementation
