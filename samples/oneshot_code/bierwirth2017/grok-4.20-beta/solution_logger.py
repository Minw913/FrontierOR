"""
Convergence logger for optimization algorithms.

Records incumbent solutions with timestamps to a JSONL file.
This module is provided to LLM-generated programs — they only need to call
`log(objective_value)` whenever a better feasible solution is found.

Usage in generated code:
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize")  # or "maximize"
    # ... inside algorithm loop:
    logger.log(objective_value)
"""

import json
import time


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

    def log(self, objective_value):
        """Record a new incumbent if it improves on the best known."""
        if objective_value is None:
            return

        if self.best_obj is not None:
            if self.sense == "minimize" and objective_value >= self.best_obj:
                return
            if self.sense == "maximize" and objective_value <= self.best_obj:
                return

        elapsed = time.time() - self.start_time

        if self.best_obj is not None and elapsed - self._last_log_time < self.min_interval:
            self.best_obj = objective_value
            return

        self.best_obj = objective_value
        self._last_log_time = elapsed

        with open(self.log_path, "a") as f:
            f.write(json.dumps({"time": round(elapsed, 3),
                                "objective_value": objective_value}) + "\n")
