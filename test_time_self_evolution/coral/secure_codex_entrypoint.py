"""Compatibility alias for the packaged Codex entrypoint."""

import sys
from infra.agent import codex_entrypoint as _implementation

sys.modules[__name__] = _implementation
