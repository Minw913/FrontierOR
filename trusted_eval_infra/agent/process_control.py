"""Signal one invocation inside a shared agent container."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def signal_invocation(token: str, signum: int) -> int:
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("invalid invocation token")
    marker = f"FRONTIER_OR_INVOCATION={token}".encode()
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            if marker not in (entry / "environ").read_bytes().split(b"\0"):
                continue
            # Let the wrapper forward graceful signals and save the session.
            # Forced cleanup also reaches detached tool processes.
            if signum != signal.SIGKILL:
                cmd = (entry / "cmdline").read_bytes().split(b"\0")
                if b"/opt/frontieror/secure_codex_entrypoint.py" not in cmd:
                    continue
            os.kill(int(entry.name), signum)
            count += 1
        except (OSError, ProcessLookupError):
            continue
    return count


if __name__ == "__main__":
    signal_invocation(sys.argv[1], int(sys.argv[2]))
