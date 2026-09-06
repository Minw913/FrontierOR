"""Compatibility alias for the packaged evaluation broker."""

import sys
from trusted_eval_infra.agent import broker as _implementation

sys.modules[__name__] = _implementation
