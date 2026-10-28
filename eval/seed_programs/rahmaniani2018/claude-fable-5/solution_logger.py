"""
Convergence logger for optimization algorithms.

Records incumbent solutions with timestamps to a JSONL file.
This module is provided to LLM-generated programs. Call `log(objective_value)`
when only the objective is available, or `log_solution(objective_value,
solution_dict)` when the full incumbent solution is available.

Usage in generated code:
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize")  # or "maximize"
    # ... inside algorithm loop:
    logger.log(objective_value)
    logger.log_solution(objective_value, solution_dict)
"""

import json
import time


def _json_default(value):
    """Convert NumPy-style scalar/array values without importing NumPy."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class SolutionLogger:
    def __init__(self, log_path, sense="minimize"):
        self.log_path = log_path
        self.sense = sense
        self.start_time = time.time()
        self.best_obj = None
        self.min_interval = 0.1  # seconds; throttle writes

        self._last_log_time = 0.0
        with open(self.log_path, "w") as f:
            pass

    def _improves(self, objective_value):
        if objective_value is None:
            return False
        if self.best_obj is not None:
            if self.sense == "minimize" and objective_value >= self.best_obj:
                return False
            if self.sense == "maximize" and objective_value <= self.best_obj:
                return False
        return True

    def _write_event(self, objective_value, extra=None):
        elapsed = time.time() - self.start_time

        if self.best_obj is not None and elapsed - self._last_log_time < self.min_interval:
            self.best_obj = objective_value
            return

        self.best_obj = objective_value
        self._last_log_time = elapsed

        event = {"time": round(elapsed, 3), "objective_value": objective_value}
        if extra:
            event.update(extra)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event, default=_json_default) + "\n")

    def log(self, objective_value):
        """Record a new incumbent if it improves on the best known."""
        if not self._improves(objective_value):
            return
        self._write_event(objective_value)

    def log_solution(self, objective_value, solution):
        """Record a full incumbent solution snapshot when it improves.

        Hardened evaluators can verify these snapshots with the trusted
        feasibility checker before using them for AOCC.
        """
        if not self._improves(objective_value):
            return
        self._write_event(objective_value, {"solution": solution})
