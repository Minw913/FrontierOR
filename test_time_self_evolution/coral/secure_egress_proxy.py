"""Compatibility alias for the packaged model egress proxy."""

import sys
from frontieror.infra.agent import egress_proxy as _implementation

sys.modules[__name__] = _implementation
