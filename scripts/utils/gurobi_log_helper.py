"""
Gurobi convergence logger via class-level monkey-patch.

Patches gurobipy.Model.optimize so that every optimize() call — with or
without an existing callback — automatically logs each new incumbent to
a JSONL file via SolutionLogger.

Usage in gurobi_code.py (3 lines added):
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    from scripts.utils.gurobi_log_helper import install_gurobi_logger

    # ... after argparse ...
    install_gurobi_logger(args.log_path)
"""

import os
import sys


# --- Hard wall-clock guard -------------------------------------------------
# ``--time_limit`` only feeds Gurobi's TimeLimit param, which bounds
# optimize() ONLY. A memory-heavy model can hang for hours in the data-load /
# model-build phase BEFORE optimize() is ever reached. The benchmark runners
# already enforce a wall-clock deadline, but a bare ``python3 gurobi_code.py``
# invocation has no outer guard. This arms a SIGALRM watchdog inside the
# process itself so every gurobi_code.py is bounded regardless of caller.
_WALLTIME_MARGIN_S = 300       # extra budget for data-load + model-build + write
_walltime_guard_armed = False


def _arm_walltime_guard():
    """Arm a one-shot SIGALRM watchdog that hard-exits the process if total
    runtime exceeds ``--time_limit`` (parsed from sys.argv) plus a build/IO
    margin. No-op if already armed, if argv carries no --time_limit, or if
    signal handling is unavailable (non-main thread / non-POSIX)."""
    global _walltime_guard_armed
    if _walltime_guard_armed:
        return
    tl = None
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--time_limit" and i + 1 < len(argv):
            try:
                tl = int(argv[i + 1])
            except (ValueError, TypeError):
                pass
        elif a.startswith("--time_limit="):
            try:
                tl = int(a.split("=", 1)[1])
            except (ValueError, TypeError):
                pass
    if tl is None:
        return  # not a --time_limit-driven invocation; leave it alone
    budget = max(60, tl) + _WALLTIME_MARGIN_S
    try:
        import signal

        def _on_alarm(signum, frame):
            msg = (f"\n[walltime-guard] hard timeout: runtime exceeded "
                   f"{budget}s (--time_limit {tl}s + {_WALLTIME_MARGIN_S}s "
                   f"margin) — model-build/solve hang; killing process.\n")
            try:
                os.write(2, msg.encode())
            except Exception:
                pass
            os._exit(124)

        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(budget)
        _walltime_guard_armed = True
    except (ValueError, OSError, AttributeError, ImportError):
        # signal.signal/alarm only on the main thread / POSIX — skip elsewhere
        pass


def install_gurobi_logger(log_path):
    """Monkey-patch gurobipy.Model.optimize at class level to log incumbents.

    Also arms a hard wall-clock watchdog (see ``_arm_walltime_guard``) so the
    process is bounded even when invoked bare, with no outer runner timeout.

    Args:
        log_path: Path to JSONL output file, or None to skip logger setup
                  (the wall-clock guard is armed either way).
    """
    _arm_walltime_guard()
    if log_path is None:
        return

    import gurobipy as gp
    from gurobipy import GRB

    if getattr(gp.Model.optimize, '_bench_patched', False):
        return

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from scripts.utils.solution_logger import SolutionLogger

    logger = SolutionLogger(log_path, sense="minimize")
    _original_optimize = gp.Model.optimize

    def _patched_optimize(self, callback=None):
        # Read sense from the model right before solving
        logger.sense = "maximize" if self.ModelSense == -1 else "minimize"

        def _combined_callback(model, where):
            if callback is not None:
                callback(model, where)
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)

        _original_optimize(self, _combined_callback)

    _patched_optimize._bench_patched = True
    gp.Model.optimize = _patched_optimize
