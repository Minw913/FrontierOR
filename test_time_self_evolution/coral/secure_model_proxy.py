"""Compatibility alias for the packaged model proxy."""

import sys
from infra.agent import model_proxy as _implementation

sys.modules[__name__] = _implementation
