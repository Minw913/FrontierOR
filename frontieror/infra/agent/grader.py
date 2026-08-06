"""Trusted CORAL grader bridge for FrontierOR.

This module is imported by CORAL's hidden grader shim. It must run outside the
agent-editable seed repository and is responsible for invoking the benchmark
evaluator with private grader/reference access.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from coral.grader import TaskGrader  # type: ignore
except ImportError:  # pragma: no cover - only used when CORAL is absent in unit tests.
    class TaskGrader:  # type: ignore
        args = {}
        codebase_path = "."

        def fail(self, message):
            return {"score": 0.0, "message": message}

        def bundle(self, score, message, feedback=""):
            return {"score": score, "message": message, "feedback": feedback}


ROOT_DIR = os.fspath(Path(__file__).resolve().parents[3])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class Grader(TaskGrader):
    def evaluate(self):
        args = self.args
        code_path = Path(self.codebase_path) / "code.py"
        if not code_path.exists():
            return self.fail("code.py not found")

        output_dir = Path(args["base_dir"]) / "coral_eval"
        output_dir.mkdir(parents=True, exist_ok=True)

        sidecar_dir = output_dir / "attempt_metadata"
        sidecar_dir.mkdir(parents=True, exist_ok=True)

        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(self.codebase_path),
                text=True, timeout=5,
            ).strip()
        except Exception:
            commit_hash = None

        env = {
            "EFFICIENT_OR_ROOT": ROOT_DIR,
            "EFFICIENT_OR_PAPER_ID": args["paper_id"],
            "EFFICIENT_OR_MODEL_NAME": args.get("model_name", "coral"),
            "EFFICIENT_OR_STAGE1_INSTANCES": ",".join(args["stage1_instances"]),
            "EFFICIENT_OR_STAGE1_TIME_LIMIT": str(args["stage1_time_limit"]),
            "EFFICIENT_OR_STAGE1_GAP_THRESHOLD": str(args["stage1_gap_threshold"]),
            "EFFICIENT_OR_STAGE2_INSTANCES": ",".join(args["stage2_instances"]),
            "EFFICIENT_OR_STAGE2_TIME_LIMIT": str(args["stage2_time_limit"]),
            "EFFICIENT_OR_STAGE2_TIME_POLICY": args.get("stage2_time_policy", "uniform"),
            "EFFICIENT_OR_STAGE2_TIME_BUFFER": str(args.get("stage2_time_buffer", 0)),
            "EFFICIENT_OR_STAGE2_SCORER": args.get("stage2_scorer", "staged_qte"),
            "EFFICIENT_OR_STAGE2_STAGE_BOUNDARY": str(args.get("stage2_stage_boundary", 0.01)),
            "EFFICIENT_OR_EXEC_MODE": args.get("exec_mode", "bare"),
            "EFFICIENT_OR_T_MAX": "" if args.get("t_max") is None else str(args["t_max"]),
            "EFFICIENT_OR_OUTPUT_DIR": str(output_dir),
            "EFFICIENT_OR_ANTI_HACK": "1" if args.get("anti_hack") else "0",
        }
        for key, value in (args.get("exec_cfg") or {}).items():
            env[f"EFFICIENT_OR_EXEC_{key.upper()}"] = str(value)

        old_env = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            from test_time_self_evolution.openevolve import evaluator

            def _metrics(result):
                return getattr(result, "metrics", result) or {}

            def _write_sidecar(payload):
                if commit_hash:
                    (sidecar_dir / f"{commit_hash}.json").write_text(
                        json.dumps(payload, default=str),
                        encoding="utf-8",
                    )

            stage1 = _metrics(evaluator.evaluate_stage1(str(code_path)))
            if float(stage1.get("combined_score", 0.0)) < 1.0:
                _write_sidecar({
                    "stage1_combined_score": stage1.get("combined_score", 0.0),
                    "stage1": stage1,
                    "stage2_skipped": True,
                })
                return self.bundle(
                    0.0,
                    "Stage1 gate failed",
                    feedback=(
                        f"Stage1 failed: combined_score={stage1.get('combined_score', 0):.3f}, "
                        f"worst_gap={stage1.get('stage1_worst_gap', 'N/A')}"
                    ),
                )

            stage2 = _metrics(evaluator.evaluate_stage2(str(code_path)))
            score = float(stage2.get("combined_score", 0.0))
            metadata_full = dict(stage2)
            metadata_full["stage1_combined_score"] = stage1.get("combined_score", 0.0)
            _write_sidecar(metadata_full)
            return self.bundle(
                score,
                f"Stage2 score {score:.6f}",
                feedback=f"Stage2 score {score:.6f}",
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
