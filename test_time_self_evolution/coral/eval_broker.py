"""Compatibility alias for the packaged evaluation broker."""

import sys
from infra.agent import broker as _implementation

sys.modules[__name__] = _implementation
