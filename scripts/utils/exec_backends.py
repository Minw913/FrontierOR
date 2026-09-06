"""Compatibility alias for the packaged execution backends."""

import sys

from trusted_eval_infra import execution as _implementation

sys.modules[__name__] = _implementation
