"""Cooperative CPU leases shared by independent benchmark processes."""
import contextlib
import fcntl
import os
from pathlib import Path
import time


@contextlib.contextmanager
def reserve_cpus(count, *, timeout=86400, lock_dir=None):
    """Reserve distinct allowed logical CPUs for the entire candidate lifetime.

    Lock files persist; the kernel releases flock ownership when the owning
    process exits. All backends use the same per-user directory, independently
    of framework, run ID, license, and XDG_RUNTIME_DIR.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("cpus must be a positive integer")
    available = sorted(os.sched_getaffinity(0))
    if count > len(available):
        raise ValueError(f"Requested {count} CPUs but only {len(available)} are allowed")
    timeout = float(timeout)
    if not 0 <= timeout < float('inf'):
        raise ValueError("CPU queue timeout must be finite and nonnegative")
    folder = Path(lock_dir) if lock_dir is not None else Path("/tmp") / f"frontieror-cpu-slots-{os.getuid()}"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    held = []
    try:
        while True:
            selected = []
            for cpu in available:
                fd = os.open(folder / f"cpu-{cpu}.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                except BaseException:
                    os.close(fd)
                    raise
                held.append(fd)
                selected.append(cpu)
                if len(selected) == count:
                    yield ",".join(map(str, selected))
                    return
            # Never wait while holding a partial multi-CPU reservation.
            for fd in held:
                os.close(fd)
            held.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for free benchmark CPUs")
            time.sleep(min(0.05, remaining))
    finally:
        for fd in held:
            os.close(fd)
