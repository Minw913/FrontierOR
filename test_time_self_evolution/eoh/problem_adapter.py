"""EoH problem adapter for frontier-or self-evolve runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from typing import Callable, Dict, Iterable, List, Optional

from test_time_self_evolution.openevolve import evaluator


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _split_instances(value: Iterable[str]) -> List[str]:
    return [str(item).strip() for item in value if str(item).strip()]


def code_hash(code_string: str) -> str:
    return hashlib.sha256(code_string.encode("utf-8")).hexdigest()[:16]


def _to_metrics(result) -> Dict:
    """Normalize evaluator return into a plain dict.

    OpenEvolve's evaluator (``test_time_self_evolution.openevolve.evaluator``)
    returns ``EvaluationResult(metrics, artifacts)`` when the ``openevolve``
    package is importable, and a bare ``dict`` otherwise. ``EvaluationResult``
    is a dataclass without ``.get()``, so callers that read scores via
    ``result.get("combined_score")`` need to unwrap first. We also accept
    ``None`` as an empty dict for defensive handling.
    """
    if result is None:
        return {}
    if hasattr(result, "metrics"):
        return dict(result.metrics)
    return dict(result)


def _to_artifacts(result) -> Dict[str, str]:
    """Extract the LLM-feedback artifact text dict from evaluator return.

    OpenEvolve's evaluator wraps results in ``EvaluationResult`` whose
    ``.artifacts`` field is the natural-language feedback channel
    (``summary``, ``failure_breakdown``, ``score_summary``). When the result
    is a bare dict (i.e. openevolve not installed), we return ``{}``.
    """
    if result is None:
        return {}
    if hasattr(result, "artifacts"):
        return dict(result.artifacts or {})
    return {}


def strip_markdown_code(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def materialize_candidate(code_string: str, output_path: str) -> str:
    """Write the LLM-generated full standalone Python script to disk.

    OpenEvolve-aligned (path 3): the LLM produces a complete CLI script
    (argparse + SolutionLogger) — we strip markdown fences and write the
    rest verbatim. ``solution_logger.py`` is auto-staged next to the
    candidate by ``scripts.utils.exec_backends._ensure_logger`` before each
    subprocess invocation, so ``from solution_logger import SolutionLogger``
    resolves correctly.
    """
    code = strip_markdown_code(code_string)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    return output_path


@contextlib.contextmanager
def patched_env(updates: Dict[str, str]):
    previous = {}
    missing = object()
    for key, value in updates.items():
        previous[key] = os.environ.get(key, missing)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EfficientOptPrompts:
    """Prompt scaffolding expected by upstream EoH custom problems.

    OpenEvolve-aligned (path 3): the LLM is prompted to produce a *complete
    standalone CLI script* (with argparse + SolutionLogger), not just a
    ``solve(instance)`` function body. ``get_task()`` returns the OpenEvolve
    system_message + the paper's benchmark spec, used directly inside the
    forked prompt templates installed by ``runner.patch_eoh_prompt_templates``.
    The four scaffolding methods (``get_func_name`` / ``inputs`` / ``outputs``
    / ``inout_inf``) are stubs because the forked templates don't reference
    them. Upstream ``Evolution.__init__`` still calls them so the interface
    must exist; we return innocuous defaults.
    """

    def __init__(self, benchmark_prompt: str):
        self.benchmark_prompt = benchmark_prompt

    def get_task(self) -> str:
        # Returns the raw benchmark prompt (no "Benchmark specification:"
        # header) — the prompt itself already contains a problem description
        # + instance/solution schemas + TASK_SPECIFICATION sections, so an
        # extra "Benchmark specification:" label is redundant. The CLI
        # CONTRACT preamble lives in runner._CLI_CONTRACT_PREAMBLE and is
        # prepended only for e/m operators; i1 receives just this raw prompt
        # so the first generation can explore freely.
        return self.benchmark_prompt.strip()

    # --- stubs (kept only for upstream EoH __init__ compatibility) -----
    def get_func_name(self) -> str:
        return ""

    def get_func_inputs(self) -> List[str]:
        return [""]

    def get_func_outputs(self) -> List[str]:
        return [""]

    def get_inout_inf(self) -> str:
        return ""

    def get_other_inf(self) -> str:
        return ""


class EohBenchmarkProblem:
    """Custom EoH problem that minimizes negative benchmark stage2 score."""

    def __init__(
        self,
        paper_id: str,
        model_name: str,
        prompt: str,
        base_output: str,
        stage1_instances: List[str],
        stage2_instances: List[str],
        stage1_time_limit: int,
        stage2_time_limit: int,
        stage1_gap_threshold: float,
        exec_mode: str,
        exec_cfg: Dict,
        t_max,
        stage2_scorer: str,
        stage2_stage_boundary: float = 0.01,
        stage2_time_policy: str = "uniform",
        stage2_time_buffer: int = 0,
        stage1_fn: Optional[Callable[[str], Dict]] = None,
        stage2_fn: Optional[Callable[[str], Dict]] = None,
        enable_artifact: bool = False,
    ):
        self.paper_id = paper_id
        self.model_name = model_name
        self.prompts = EfficientOptPrompts(prompt)
        self.base_output = base_output
        self.stage1_instances = _split_instances(stage1_instances)
        self.stage2_instances = _split_instances(stage2_instances)
        self.stage1_time_limit = stage1_time_limit
        self.stage2_time_limit = stage2_time_limit
        self.stage1_gap_threshold = stage1_gap_threshold
        self.exec_mode = exec_mode
        self.exec_cfg = exec_cfg or {}
        self.t_max = t_max
        self.stage2_scorer = stage2_scorer
        self.stage2_stage_boundary = stage2_stage_boundary
        self.stage2_time_policy = stage2_time_policy
        self.stage2_time_buffer = stage2_time_buffer
        self.stage1_fn = stage1_fn or evaluator.evaluate_stage1
        self.stage2_fn = stage2_fn or evaluator.evaluate_stage2
        # When True, capture OpenEvolve evaluator's `.artifacts` dict
        # (failure_breakdown / score_summary text) into the cache, so the
        # next-generation prompt builder can inject it back as feedback.
        self.enable_artifact = bool(enable_artifact)

    def _candidate_path(self, candidate_hash: str) -> str:
        return os.path.join(self.base_output, "candidates", f"{candidate_hash}.py")

    def _cache_path(self, candidate_hash: str) -> str:
        return os.path.join(self.base_output, "eoh_eval_cache", f"{candidate_hash}.json")

    def _env(self, candidate_hash: str) -> Dict[str, str]:
        env = {
            "EFFICIENT_OR_ROOT": ROOT_DIR,
            "EFFICIENT_OR_PAPER_ID": self.paper_id,
            "EFFICIENT_OR_MODEL_NAME": self.model_name,
            "EFFICIENT_OR_STAGE1_INSTANCES": ",".join(self.stage1_instances),
            "EFFICIENT_OR_STAGE1_TIME_LIMIT": str(self.stage1_time_limit),
            "EFFICIENT_OR_STAGE1_GAP_THRESHOLD": str(self.stage1_gap_threshold),
            "EFFICIENT_OR_STAGE2_INSTANCES": ",".join(self.stage2_instances),
            "EFFICIENT_OR_STAGE2_TIME_LIMIT": str(self.stage2_time_limit),
            "EFFICIENT_OR_STAGE2_TIME_POLICY": self.stage2_time_policy,
            "EFFICIENT_OR_STAGE2_TIME_BUFFER": str(self.stage2_time_buffer),
            "EFFICIENT_OR_STAGE2_SCORER": self.stage2_scorer,
            "EFFICIENT_OR_STAGE2_STAGE_BOUNDARY": str(self.stage2_stage_boundary),
            "EFFICIENT_OR_EXEC_MODE": self.exec_mode,
            "EFFICIENT_OR_T_MAX": "" if self.t_max is None else str(self.t_max),
            "EFFICIENT_OR_OUTPUT_DIR": os.path.join(self.base_output, "eoh_eval", candidate_hash),
        }
        for key, value in self.exec_cfg.items():
            env[f"EFFICIENT_OR_EXEC_{key.upper()}"] = str(value)
        return env

    def evaluate(self, code_string: str) -> float:
        # Diagnostic: confirm evaluate is actually being invoked.
        print(f"[EohBenchmarkProblem.evaluate] CALLED with code_string len={len(code_string)}", flush=True)
        try:
            return self._evaluate_inner(code_string)
        except Exception as exc:
            # Return a finite sentinel objective on any evaluation failure
            # (instead of raising). Upstream EoH otherwise catches the
            # exception, sets objective=None, and filters the individual out
            # of the population. With pop_size=2, two simultaneous failures
            # empty the population entirely and the next save crashes at
            # `population[0]` (IndexError). A very large objective keeps the
            # candidate around (so the population never empties) while
            # guaranteeing it loses to any real-scoring candidate.
            import traceback
            print(f"[EohBenchmarkProblem.evaluate] EXCEPTION caught (objective=1e9 sentinel): {exc!r}", flush=True)
            traceback.print_exc()
            return 1e9

    def _evaluate_inner(self, code_string: str) -> float:
        candidate_hash = code_hash(strip_markdown_code(code_string))
        program_path = materialize_candidate(code_string, self._candidate_path(candidate_hash))

        with patched_env(self._env(candidate_hash)):
            # Capture the raw evaluator return (may be EvaluationResult or
            # dict). We split into metrics (always needed for objective) and
            # artifacts (only used when enable_artifact=True).
            s1_raw = self.stage1_fn(program_path)
            stage1_metrics = _to_metrics(s1_raw)
            stage1_artifact = _to_artifacts(s1_raw) if self.enable_artifact else {}
            if float(stage1_metrics.get("combined_score", 0.0)) < 1.0:
                objective = 1.0
                self._write_cache(candidate_hash, program_path, objective,
                                  stage1_metrics, {}, stage1_artifact, {})
                return objective
            s2_raw = self.stage2_fn(program_path)
            stage2_metrics = _to_metrics(s2_raw)
            stage2_artifact = _to_artifacts(s2_raw) if self.enable_artifact else {}

        stage2_score = float(stage2_metrics.get("combined_score", 0.0))
        # OpenEvolve's staged_qte scorer can return combined_score > 1.0 (the
        # "beat-Gurobi" stage), so we don't clip — just negate so EoH's
        # internal "lower is better" sort lines up with our "higher is better".
        objective = round(-stage2_score, 6)
        self._write_cache(candidate_hash, program_path, objective,
                          stage1_metrics, stage2_metrics,
                          stage1_artifact, stage2_artifact)
        return objective

    def get_artifact_text(self, code_string: str) -> Optional[str]:
        """Format cached evaluator artifacts into one feedback block.

        Returns ``None`` when artifacts are disabled, the candidate isn't
        cached yet, or no artifact text exists. Returned text is plain
        natural language suitable for direct injection into LLM prompts —
        same shape as OpenEvolve's per-candidate feedback channel.
        """
        if not self.enable_artifact:
            return None
        cached = self.read_cached_metrics(code_string)
        if not cached:
            return None
        s1 = cached.get("stage1_artifact") or {}
        s2 = cached.get("stage2_artifact") or {}
        parts: List[str] = []
        if s1.get("summary"):
            parts.append(str(s1["summary"]).strip())
        if s1.get("failure_breakdown"):
            parts.append(str(s1["failure_breakdown"]).strip())
        if s2.get("score_summary"):
            parts.append(str(s2["score_summary"]).strip())
        if s2.get("failure_breakdown"):
            parts.append(str(s2["failure_breakdown"]).strip())
        if not parts:
            return None
        return "\n\n".join(parts)

    def read_cached_metrics(self, code_string: str) -> Optional[Dict]:
        cache_path = self._cache_path(code_hash(strip_markdown_code(code_string)))
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    def _write_cache(
        self,
        candidate_hash: str,
        program_path: str,
        objective: float,
        stage1_metrics: Dict,
        stage2_metrics: Dict,
        stage1_artifact: Optional[Dict] = None,
        stage2_artifact: Optional[Dict] = None,
    ):
        cache_path = self._cache_path(candidate_hash)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "code_hash": candidate_hash,
                    "program_path": program_path,
                    "objective": objective,
                    "stage1_metrics": stage1_metrics,
                    "stage2_metrics": stage2_metrics,
                    # Artifact dicts are written even when empty so cache schema
                    # is stable across enable_artifact runs.
                    "stage1_artifact": dict(stage1_artifact or {}),
                    "stage2_artifact": dict(stage2_artifact or {}),
                },
                f,
                indent=2,
                sort_keys=True,
            )
