#!/usr/bin/env python3
"""Compatibility alias for the packaged official submission verifier."""

import sys
from frontieror.infra.submission import cli as _implementation

sys.modules[__name__] = _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
