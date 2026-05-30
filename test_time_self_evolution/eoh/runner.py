"""EoH-specific orchestration for self-evolving benchmark runs."""

from __future__ import annotations

import glob
import http.client
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional
from urllib.parse import urlparse

import yaml
from joblib import parallel_backend

import one_shot_eval as eval_core
from test_time_self_evolution import eval_modes
from test_time_self_evolution.eoh.problem_adapter import (
    EohBenchmarkProblem,
    materialize_candidate,
    patched_env,
)
from test_time_self_evolution.openevolve.runner import reconstruct_results_from_metrics


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EOH_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "eoh.yaml")


def load_eoh_config(path: Optional[str] = None) -> Dict:
    config_path = path or EOH_CONFIG_PATH
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _endpoint_host(endpoint: Optional[str]) -> str:
    endpoint = (endpoint or "openrouter.ai").strip()
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        parsed = urlparse(endpoint)
        return parsed.netloc or parsed.path
    return endpoint.strip("/").split("/")[0]


def chat_completions_path(endpoint: Optional[str]) -> str:
    host = _endpoint_host(endpoint)
    if host == "openrouter.ai":
        return "/api/v1/chat/completions"
    return "/v1/chat/completions"


def prepare_eoh_env(base_env: Optional[Dict[str, str]], config: Dict) -> Dict[str, str]:
    env = dict(base_env or os.environ)
    key = env.get("OPENROUTER_API_KEY") or config.get("OPENROUTER_API_KEY")
    if key:
        env["OPENROUTER_API_KEY"] = key
    if os.environ.get("GRB_LICENSE_FILE"):
        env["GRB_LICENSE_FILE"] = os.environ["GRB_LICENSE_FILE"]
    return env


def _ensure_eoh_importable():
    source_paths = [
        os.path.join(ROOT_DIR, "external", "eoh", "eoh", "src"),
        os.path.join(ROOT_DIR, "external", "eoh", "eoh"),
    ]
    for path in reversed(source_paths):
        if os.path.exists(path):
            while path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
    local_eoh_dir = os.path.join(ROOT_DIR, "test_time_self_evolution", "eoh")
    loaded = sys.modules.get("eoh")
    loaded_path = os.path.abspath(getattr(loaded, "__file__", "")) if loaded else ""
    if loaded_path.startswith(os.path.abspath(local_eoh_dir) + os.sep):
        for name in list(sys.modules):
            if name == "eoh" or name.startswith("eoh."):
                sys.modules.pop(name, None)


# Sentinel embedded in prompt_content by patched get_prompt_* methods to
# encode a (system, user) pair as a single string. The http patcher
# (patch_eoh_remote_api_path) splits on this and emits a 2-message payload.
# Choosing a long unlikely-to-collide string instead of a unicode separator
# so prompts remain debuggable in plain logs.
_EOH_SYSTEM_USER_SEP = "\n<<<__EOH_SYSTEM_USER_SEP__>>>\n"


def patch_eoh_remote_api_path():
    """Make upstream EoH's remote API client work with OpenRouter's /api/v1 path."""
    try:
        from eoh.llm import api_general
    except Exception:
        return

    def get_response(self, prompt_content):
        # Temperature aligned with configs/openevolve.yaml (llm.temperature: 0.8)
        # so EoH and OpenEvolve use the same LLM-sampling distribution.
        # Read from the EoH-side instance's ``temperature`` attribute when set
        # (e.g., when our runner overrides it), else default to 0.8.
        temperature = float(getattr(self, "temperature", 0.8))
        # If the patched prompt builder embedded the sentinel, decode it
        # into separate system + user messages. i1 omits the sentinel and
        # is sent as a user-only message; e/m operators always include it.
        if _EOH_SYSTEM_USER_SEP in prompt_content:
            sys_msg, user_msg = prompt_content.split(_EOH_SYSTEM_USER_SEP, 1)
            messages = [
                {"role": "system", "content": sys_msg.strip()},
                {"role": "user", "content": user_msg.strip()},
            ]
        else:
            messages = [{"role": "user", "content": prompt_content}]
        payload = json.dumps({
            "model": self.model_LLM,
            "messages": messages,
            "temperature": temperature,
        })
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "frontier-or/1.0",
            "Content-Type": "application/json",
        }
        response = None
        for _ in range(getattr(self, "n_trial", 5)):
            try:
                conn = http.client.HTTPSConnection(_endpoint_host(self.api_endpoint))
                conn.request(
                    "POST",
                    chat_completions_path(self.api_endpoint),
                    payload,
                    headers,
                )
                data = json.loads(conn.getresponse().read())
                message = data["choices"][0]["message"]
                response = message.get("content") or message.get("reasoning")
                if response:
                    break
            except Exception:
                if getattr(self, "debug_mode", False):
                    print("Error in API. Restarting the process...")
                continue
        return response

    api_general.InterfaceAPI.get_response = get_response


_CLI_CONTRACT_PREAMBLE = """\
You are improving a Python program that solves a combinatorial optimization problem. Each iteration receives the parent program's metrics, recent execution output, and full code. Rewrite the program to MAXIMIZE the score.

CLI CONTRACT (every program MUST satisfy):
- argparse args: --instance_path, --solution_path, --time_limit (int seconds), --log_path (optional, JSONL)
- Read all instance data from --instance_path; never hard-code instance data
- Write solution JSON to --solution_path matching the solution_schema
- Use solution_logger.SolutionLogger; call logger.log(objective_value) on every improving incumbent:
  ```python
  from solution_logger import SolutionLogger
  logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None  # or sense="maximize"
  if logger:
      logger.log(objective_value)
  ```
- If using MIP/LP, use Gurobi (gurobipy) ONLY — no CPLEX/SCIP/CBC/PuLP/OR-Tools/etc.

FAIRNESS CONSTRAINT: keep model.setParam('Threads', 1) (and any other single-thread settings) unchanged. Do NOT spawn parallel workers — this benchmark compares algorithms under a single-CPU budget."""

_OUTPUT_FORMAT_BLOCK_I1 = (
    "First, describe your new algorithm in one sentence. The description "
    "must be enclosed in a brace, like {your one-sentence description}. "
    "Next, output the standalone Python script wrapped in ```python``` "
    "markdown fences. Do not give additional explanations."
)
_OUTPUT_FORMAT_BLOCK = (
    "First, describe your new algorithm in one sentence. The description "
    "must be enclosed in a brace, like {your one-sentence description}. "
    "Next, output the COMPLETE standalone Python script (every CLI-CONTRACT "
    "requirement included) wrapped in ```python``` markdown fences. "
    "Do not give additional explanations."
)
_OUTPUT_FORMAT_BLOCK_M3 = (
    "First, describe your revised algorithm in one sentence. The description "
    "must be enclosed in a brace, like {your one-sentence description}. "
    "Next, output the COMPLETE revised standalone Python script (every "
    "CLI-CONTRACT requirement preserved) wrapped in ```python``` markdown "
    "fences. Do not give additional explanations."
)


def _format_parent_block(indivs) -> str:
    """Render parent algorithms+codes for crossover/mutation prompts.

    Each parent rendered as: ``No.{i} algorithm and the corresponding code:\n
    <description>\n```python\n<code>\n``` ``` `` ``. Single-parent variants
    (m1/m2/m3) skip the numbering.
    """
    multi = isinstance(indivs, list) and len(indivs) > 1
    indivs_list = indivs if isinstance(indivs, list) else [indivs]
    blocks = []
    for i, ind in enumerate(indivs_list, 1):
        if not ind:
            continue
        algo = (ind.get("algorithm") or "").strip()
        code = (ind.get("code") or "").strip()
        header = (
            f"No.{i} algorithm and the corresponding code are:"
            if multi
            else "Algorithm description:"
        )
        blocks.append(
            f"{header} {algo}\n```python\n{code}\n```"
        )
    return "\n\n".join(blocks)


_EOH_SYSTEM_SPEC_MODE: str = "cli_only"  # "cli_only" | "full"


def _build_eoh_system_message(self_evolution) -> str:
    """Build the system-message content for the current mode.

    "cli_only" (default, OpenEvolve-aligned): system message contains only
    the CLI CONTRACT preamble; the per-paper benchmark spec is NOT sent —
    LLM relies on parent code carrying problem context implicitly (matches
    OpenEvolve's prompt structure exactly).

    "full" (--eoh-system-include-spec): system message is the full raw
    benchmark prompt — problem_description.txt + instance_schema.json +
    solution_schema.json + TASK_SPECIFICATION (which itself contains a
    duplicate CLI CONTRACT). Use when EoH's per-iteration crossover/mutation
    decisions benefit from re-seeing the problem description even though
    the parent code is already attached.
    """
    if _EOH_SYSTEM_SPEC_MODE == "full":
        return self_evolution.prompt_task  # = paper_benchmark_spec
    return _CLI_CONTRACT_PREAMBLE


def patch_eoh_prompt_templates(system_spec_mode: str = "cli_only"):
    """Install OpenEvolve-aligned prompt templates on EoH's Evolution class.

    Path 3 (full-script paradigm): each of ``get_prompt_{i1,e1,e2,m1,m2,m3}``
    is replaced so the LLM is asked to output a *complete standalone CLI
    Python program* (with argparse, SolutionLogger, Gurobi-only) instead of
    a bare ``solve(instance)`` function.

    e/m operators emit a 2-part prompt (system + user, encoded via
    ``_EOH_SYSTEM_USER_SEP`` and decoded by ``patch_eoh_remote_api_path``):
      - system: depends on ``system_spec_mode`` (see ``_build_eoh_system_message``)
      - user: parent block + algorithm-specific instruction + output format

    i1 has no system message and stays user-only (just ``self.prompt_task``
    + ``_OUTPUT_FORMAT_BLOCK_I1``) — first generation explores freely.

    Idempotent: re-installing replaces the previous patched method.
    """
    if system_spec_mode not in ("cli_only", "full"):
        raise ValueError(
            f"system_spec_mode must be 'cli_only' or 'full', got {system_spec_mode!r}"
        )
    global _EOH_SYSTEM_SPEC_MODE
    _EOH_SYSTEM_SPEC_MODE = system_spec_mode
    try:
        from eoh.methods.eoh.eoh_evolution import Evolution
    except Exception:
        return

    def make_i1():
        # i1 (initial generation, no parent) sees only the benchmark spec —
        # no CLI CONTRACT preamble. Lets the LLM explore freely on the first
        # generation; CLI compliance gets enforced in subsequent e/m
        # operators which DO carry the preamble alongside the parent code.
        def get_prompt_i1(self):
            return (
                f"{self.prompt_task}\n\n"
                f"{_OUTPUT_FORMAT_BLOCK_I1}"
            )
        return get_prompt_i1

    def make_e1():
        def get_prompt_e1(self, indivs):
            n = len(indivs)
            parents = _format_parent_block(indivs)
            sys_msg = _build_eoh_system_message(self)
            user_msg = (
                f"I have {n} existing algorithms with their codes as follows:\n"
                f"{parents}\n\n"
                f"Please help me create a new algorithm that has a totally "
                f"different form from the given ones.\n\n"
                f"{_OUTPUT_FORMAT_BLOCK}"
            )
            return f"{sys_msg}{_EOH_SYSTEM_USER_SEP}{user_msg}"
        return get_prompt_e1

    def make_e2():
        def get_prompt_e2(self, indivs):
            n = len(indivs)
            parents = _format_parent_block(indivs)
            sys_msg = _build_eoh_system_message(self)
            user_msg = (
                f"I have {n} existing algorithms with their codes as follows:\n"
                f"{parents}\n\n"
                f"Please help me create a new algorithm that has a totally "
                f"different form from the given ones but can be motivated "
                f"from them. Firstly, identify the common backbone idea in "
                f"the provided algorithms. Secondly, based on the backbone "
                f"idea design your new algorithm.\n\n"
                f"{_OUTPUT_FORMAT_BLOCK}"
            )
            return f"{sys_msg}{_EOH_SYSTEM_USER_SEP}{user_msg}"
        return get_prompt_e2

    def make_m1():
        def get_prompt_m1(self, indiv1):
            parent = _format_parent_block(indiv1)
            sys_msg = _build_eoh_system_message(self)
            user_msg = (
                f"I have one algorithm with its code as follows.\n"
                f"{parent}\n\n"
                f"Please assist me in creating a new algorithm that has a "
                f"different form but can be a modified version of the "
                f"algorithm provided.\n\n"
                f"{_OUTPUT_FORMAT_BLOCK}"
            )
            return f"{sys_msg}{_EOH_SYSTEM_USER_SEP}{user_msg}"
        return get_prompt_m1

    def make_m2():
        def get_prompt_m2(self, indiv1):
            parent = _format_parent_block(indiv1)
            sys_msg = _build_eoh_system_message(self)
            user_msg = (
                f"I have one algorithm with its code as follows.\n"
                f"{parent}\n\n"
                f"Please identify the main algorithm parameters and assist "
                f"me in creating a new algorithm that has different parameter "
                f"settings of the score function provided.\n\n"
                f"{_OUTPUT_FORMAT_BLOCK}"
            )
            return f"{sys_msg}{_EOH_SYSTEM_USER_SEP}{user_msg}"
        return get_prompt_m2

    def make_m3():
        def get_prompt_m3(self, indiv1):
            parent = _format_parent_block(indiv1)
            sys_msg = _build_eoh_system_message(self)
            user_msg = (
                f"I have one algorithm with its code as follows.\n"
                f"{parent}\n\n"
                f"First, identify the main components in the program above. "
                f"Next, analyze whether any of these components can be overfit "
                f"to the in-distribution instances. Then, based on your "
                f"analysis, simplify the components to enhance generalization "
                f"to potential out-of-distribution instances. Finally, "
                f"provide the revised complete CLI script.\n\n"
                f"{_OUTPUT_FORMAT_BLOCK_M3}"
            )
            return f"{sys_msg}{_EOH_SYSTEM_USER_SEP}{user_msg}"
        return get_prompt_m3

    Evolution.get_prompt_i1 = make_i1()
    Evolution.get_prompt_e1 = make_e1()
    Evolution.get_prompt_e2 = make_e2()
    Evolution.get_prompt_m1 = make_m1()
    Evolution.get_prompt_m2 = make_m2()
    Evolution.get_prompt_m3 = make_m3()


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DESC_RE = re.compile(r"\{(.*?)\}", re.DOTALL)


def _parse_eoh_response(response: str):
    """Extract (algorithm_description, code) from a full-script LLM response.

    Description: first non-greedy ``{...}`` match (one-sentence sentence the
    LLM puts before the script).
    Code: prefer the LAST ```` ```python ```` fenced block (LLM may echo
    parents but its new code comes last); fallback to everything after the
    first ``}``; last resort is the raw response. Strip stray fence
    remnants if present.
    """
    text = response or ""
    desc_m = _DESC_RE.search(text)
    fences = _FENCE_RE.findall(text)
    if fences:
        code = fences[-1].strip()
    elif desc_m:
        code = text[desc_m.end():].strip()
    else:
        code = text.strip()
    code = re.sub(r"^```(?:python|py)?\s*\n", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\n```\s*$", "", code)
    algorithm = desc_m.group(1).strip() if desc_m else ""
    return algorithm, code


def patch_eoh_get_alg():
    """Replace upstream Evolution._get_alg with a full-script parser.

    Upstream's regex (``import.*return`` + greedy ``\\{(.*)\\}``) was tuned
    for ``def solve(...): return ...`` snippets and breaks on the path-3
    full-CLI scripts: it truncates at the last ``return`` (dropping
    ``if __name__ == "__main__": main()``) and the greedy brace match
    swallows everything when the script contains dict literals. Our
    replacement parses ``{description}`` (non-greedy) + ```` ```python ````
    block. Up to 3 retries when either is missing — matches upstream's
    retry budget at eoh_evolution.py:138.
    """
    try:
        from eoh.methods.eoh.eoh_evolution import Evolution
    except Exception:
        return

    def _get_alg(self, prompt_content):
        print(f"[_get_alg] called (prompt len={len(prompt_content)})", flush=True)
        response = self.interface_llm.get_response(prompt_content)
        print(f"[_get_alg] got response (len={len(response or '')})", flush=True)
        algorithm, code = _parse_eoh_response(response)
        print(f"[_get_alg] parsed: algo_len={len(algorithm)} code_len={len(code)} has_def={'def ' in code}", flush=True)
        n_retry = 1
        while (not algorithm or "def " not in code) and n_retry <= 3:
            print(f"[_get_alg] retry {n_retry} (algo or code missing)", flush=True)
            response = self.interface_llm.get_response(prompt_content)
            algorithm, code = _parse_eoh_response(response)
            print(f"[_get_alg] retry {n_retry} parsed: algo_len={len(algorithm)} code_len={len(code)} has_def={'def ' in code}", flush=True)
            n_retry += 1
        if not algorithm:
            algorithm = "(no description)"
        return [code, algorithm]

    Evolution._get_alg = _get_alg


def patch_eoh_prompts_with_artifact(problem: EohBenchmarkProblem):
    """Inject per-parent evaluator artifact text into EoH's mutation /
    crossover prompts.

    This monkey-patches ``Evolution.get_prompt_{e1,e2,m1,m2,m3}`` so that
    each parent's cached failure_breakdown / score_summary (built by
    OpenEvolve's evaluator and captured by ``EohBenchmarkProblem.evaluate``)
    is appended right before the trailing "Do not give additional
    explanations." instruction — making EoH consume the same LLM-feedback
    channel that OpenEvolve uses natively.

    No-op if the problem instance has ``enable_artifact=False``; in that
    case the original prompt builders are kept untouched.
    """
    if not problem.enable_artifact:
        return
    from eoh.methods.eoh.eoh_evolution import Evolution

    trailing_marker = "Do not give additional explanations."

    def make_patched(method_name: str, original):
        def patched(self_evolution, indivs):
            # m1/m2/m3 take a single dict; e1/e2 take a list of dicts.
            indivs_list = indivs if isinstance(indivs, list) else [indivs]
            base = original(self_evolution, indivs)
            arts = []
            for i, ind in enumerate(indivs_list, 1):
                code = (ind or {}).get("code") or ""
                if not code:
                    continue
                txt = problem.get_artifact_text(code)
                if not txt:
                    continue
                label = f"Algorithm No.{i}" if len(indivs_list) > 1 else "The algorithm above"
                arts.append(f"--- {label} evaluation feedback ---\n{txt}")
            if not arts:
                return base
            block = "\n\nEvaluation feedback for the candidate(s) shown above:\n\n" + "\n\n".join(arts) + "\n"
            idx = base.rfind(trailing_marker)
            if idx >= 0:
                return base[:idx] + block + "\n" + base[idx:]
            return base + block
        patched.__name__ = method_name
        return patched

    for method_name in ("get_prompt_e1", "get_prompt_e2",
                        "get_prompt_m1", "get_prompt_m2", "get_prompt_m3"):
        original = getattr(Evolution, method_name, None)
        if original is None:
            continue
        setattr(Evolution, method_name, make_patched(method_name, original))


def patch_eoh_n_create(n_create: int = 1):
    """Override upstream EoH's hardcoded ``n_create=2`` initial-population
    over-sampling factor.

    Upstream ``InterfaceEC.population_generation`` (eoh_interface_EC.py:62)
    runs ``n_create=2`` outer loops × ``pop_size`` parallel i1 calls each →
    generates ``2 * pop_size`` candidates, then ``population_management``
    trims back down to ``pop_size`` by objective.

    With n_create=1 we generate exactly ``pop_size`` candidates and skip
    the trim step. Halves initial LLM cost (4 i1 calls instead of 8 at
    pop_size=4) and doubles the seed-reuse savings ratio (1/4 = 25%
    instead of 1/8 = 12.5%). Trade-off: no over-sample + select step, so
    initial population variance is higher.
    """
    try:
        from eoh.methods.eoh.eoh_interface_EC import InterfaceEC
    except Exception:
        return

    def population_generation(self):
        population = []
        for _ in range(n_create):
            _, pop = self.get_algorithm([], "i1")
            for p in pop:
                population.append(p)
        return population

    InterfaceEC.population_generation = population_generation


def patch_eoh_i1_with_oneshot_seed(
    paper_id: str, model_short: str, output_dir: str
) -> bool:
    """Make the FIRST ``Evolution.i1()`` invocation reuse one-shot's
    ``code_attempt0.py`` instead of calling LLM. All subsequent i1() calls
    fall through to the original (which then runs the patched template +
    LLM flow). Mirrors OpenEvolve's ``_try_reuse_oneshot_seed`` so the same
    one-shot v0 attempt anchors the initial population in both frameworks.

    Returns True if seed reuse was wired up (one-shot artifact found and
    copied + Evolution.i1 monkey-patched); False if no one-shot data exists
    for ``(paper_id, model_short)``.

    EoH builds the initial population via ``n_create=2`` outer loops × joblib
    Parallel(``pop_size`` offspring per loop) — total ``2 × pop_size`` i1
    calls per run (default 8 when pop_size=4). To preserve i1 diversity for
    the evolution operators, only the FIRST i1 call across all workers
    consumes the seed; the rest go to LLM. Cross-process atomicity uses
    ``O_CREAT|O_EXCL`` on a marker file in ``<output_dir>/seed_oneshot/``,
    the standard POSIX mutex primitive.

    Path lookup matches OpenEvolve's helper (``_try_reuse_oneshot_seed``):
    ``eval/eval_papers/<paper>/<model_short>/code_attempt0.py``.
    """
    src = os.path.join(
        ROOT_DIR, "eval", "eval_papers", paper_id, model_short, "code_attempt0.py",
    )
    if not os.path.exists(src) or os.path.getsize(src) == 0:
        return False

    seed_dir = os.path.join(output_dir, "seed_oneshot")
    os.makedirs(seed_dir, exist_ok=True)
    seed_code_path = os.path.join(seed_dir, "code.py")
    shutil.copyfile(src, seed_code_path)
    # Provenance record so anyone inspecting the run can see where the
    # initial seed came from.
    with open(os.path.join(seed_dir, "_seed_source.txt"), "w", encoding="utf-8") as f:
        f.write(f"reused from one-shot v0: {src}\n")

    claim_marker = os.path.join(seed_dir, ".claimed")
    # Clean any stale marker from a prior interrupted run so this run gets
    # to consume the seed exactly once.
    try:
        os.unlink(claim_marker)
    except FileNotFoundError:
        pass

    try:
        from eoh.methods.eoh.eoh_evolution import Evolution
    except Exception:
        return False

    original_i1 = Evolution.i1

    def patched_i1(self):
        print(f"[i1] patched_i1 invoked (pid {os.getpid()})", flush=True)
        # Atomic claim across joblib workers: O_CREAT|O_EXCL is the standard
        # POSIX mutex primitive. The first caller (across all loky processes)
        # successfully creates the marker and consumes the seed; subsequent
        # callers get FileExistsError and fall through to the original LLM
        # path, preserving population diversity.
        try:
            fd = os.open(claim_marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print(f"[i1] seed already claimed; calling original_i1 (LLM path)", flush=True)
            return original_i1(self)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(f"claimed by pid {os.getpid()}\n")
            with open(seed_code_path, encoding="utf-8") as f:
                code = f.read()
            algorithm = "Reused from one-shot attempt 0 (no LLM call)"
            print(f"[i1] reused one-shot seed: {seed_code_path} (code len={len(code)})", flush=True)
            return [code, algorithm]
        except Exception as exc:
            # Defensive: if seed reading fails for any reason after we've
            # claimed it, log + fall back to LLM so the run doesn't crash.
            print(f"[i1] seed reuse failed ({exc!r}); falling back to LLM", flush=True)
            return original_i1(self)

    Evolution.i1 = patched_i1
    return True


def run_eoh(
    problem: EohBenchmarkProblem,
    output_dir: str,
    primary_model: str,
    config: Dict,
    pop_size: int,
    n_pop: int,
    workers: int,
    timeout: Optional[int],
    operators: Optional[List[str]] = None,
    system_include_spec: bool = False,
    resume: bool = False,
):
    _ensure_eoh_importable()
    try:
        from eoh import eoh as eoh_module
        from eoh.utils.getParas import Paras
    except Exception as exc:
        raise RuntimeError(
            "EoH is not importable. Run test_time_self_evolution/eoh/setup.sh "
            "or install external/eoh/eoh with pip -e."
        ) from exc
    patch_eoh_remote_api_path()
    # Path-3 alignment with OpenEvolve: install full-script prompt templates
    # and a parser that handles complete CLI scripts. These run BEFORE the
    # artifact patcher so it wraps our new templates (artifact text is
    # spliced in front of the trailing "Do not give additional explanations.").
    # system_include_spec=True puts the full <paper_benchmark_spec> in the
    # system message of e/m operators; default False puts only the CLI
    # CONTRACT preamble (matches OpenEvolve, which doesn't send the per-paper
    # spec to LLM but relies on parent code carrying that context).
    patch_eoh_prompt_templates(
        system_spec_mode="full" if system_include_spec else "cli_only"
    )
    patch_eoh_get_alg()
    # Override upstream's n_create=2 over-sampling. With n_create=1 we
    # generate exactly pop_size i1 candidates (vs 2*pop_size) and skip the
    # population_management trim step — halves initial LLM cost.
    patch_eoh_n_create(n_create=1)
    # Reuse one-shot's code_attempt0.py for the FIRST i1 call (saves 1 LLM
    # call); other pop_size-1 i1 calls go to LLM as usual to keep init
    # population diverse. No-op when no matching one-shot artifact exists.
    patch_eoh_i1_with_oneshot_seed(
        problem.paper_id, problem.model_name, output_dir,
    )
    # When the problem was constructed with enable_artifact=True, splice
    # cached evaluator artifacts back into mutation/crossover prompts.
    patch_eoh_prompts_with_artifact(problem)

    llm_cfg = (config.get("eoh") or {}).get("llm", {})
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or config.get("OPENROUTER_API_KEY")
        or llm_cfg.get("api_key")
    )
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for EoH runs.")

    api_endpoint = _endpoint_host(
        llm_cfg.get("api_endpoint")
        or config.get("OPENROUTER_API_ENDPOINT")
        or config.get("OPENROUTER_BASE_URL")
        or "openrouter.ai"
    )
    eoh_cfg = config.get("eoh") or {}

    # Resume support: EoH's native ``exp_use_continue`` reloads a prior
    # population JSON and continues from generation ``exp_continue_id``. We
    # find the highest-numbered population_generation_<N>.json that the
    # previous run wrote and feed that path back to the EVOL constructor.
    resume_kwargs = {}
    if resume:
        pops_dir = os.path.join(output_dir, "results", "pops")
        if not os.path.isdir(pops_dir):
            raise RuntimeError(
                f"Cannot resume EoH at {output_dir}: pops directory not found "
                f"({pops_dir}). Run was never started here."
            )
        pop_files = sorted(
            glob.glob(os.path.join(pops_dir, "population_generation_*.json")),
            key=_generation_number,
        )
        if not pop_files:
            raise RuntimeError(
                f"Cannot resume EoH at {output_dir}: no population_generation_*.json "
                f"in {pops_dir}."
            )
        latest = pop_files[-1]
        latest_n = _generation_number(latest)
        print(f"[resume:eoh] continuing from generation {latest_n} at {latest}")
        resume_kwargs = {
            "exp_use_continue": True,
            "exp_continue_path": latest,
            "exp_continue_id": latest_n,
        }

    # ec_operators: precedence is CLI flag > yaml > default [e1,e2,m1,m2,m3].
    # Note we enable m3 by default — it's wired up in upstream EoH
    # (eoh_evolution.py:266 + eoh_interface_EC.py:125) but excluded from
    # getParas.py:72's default list. Its prompt asks the LLM to "simplify
    # components to enhance generalization to potential out-of-distribution
    # instances" — useful for our setting since stage2 uses smaller instances
    # and the final test uses larger ones, i.e. an OOD evaluation. With m3
    # enabled, total LLM evals per run = pop_size + 5 * pop_size * n_pop.
    if operators is None:
        operators = list(eoh_cfg.get("operators") or ["e1", "e2", "m1", "m2", "m3"])
    else:
        operators = list(operators)

    paras = Paras()
    paras.set_paras(
        method=eoh_cfg.get("method", "eoh"),
        problem=problem,
        llm_api_endpoint=api_endpoint,
        llm_api_key=api_key,
        llm_model=primary_model,
        ec_pop_size=pop_size,
        ec_n_pop=n_pop,
        ec_operators=operators,
        exp_n_proc=workers,
        exp_output_path=output_dir,
        exp_debug_mode=False,
        eva_timeout=timeout,
        eva_numba_decorator=False,
        **resume_kwargs,
    )
    # Force joblib's "threading" backend so every Parallel(n_jobs=...) call
    # inside upstream EoH runs in this same process. The default "loky"
    # backend spawns subprocess workers, which DO NOT inherit our runtime
    # monkey-patches (Evolution.i1, Evolution._get_alg, get_response, etc.).
    # Result with loky: workers fall back to upstream's brittle parsers
    # (regex `import.*return`) → can't parse our full-script LLM responses
    # → IndexError on empty match → every offspring objective=None →
    # population_management filters all → IndexError at end.
    # Threading is safe here: LLM calls and subprocess evaluations release
    # the GIL, so this still parallelizes effectively.
    with parallel_backend("threading"):
        eoh_module.EVOL(paras).run()


def _generation_number(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _best_from_json(data):
    if isinstance(data, list):
        if not data:
            raise RuntimeError("EoH population file is empty.")
        return min(data, key=lambda item: float(item.get("objective", 0.0)))
    if isinstance(data, dict):
        return data
    raise RuntimeError("Unsupported EoH population JSON shape.")


def find_best_eoh_individual(run_dir: str) -> Dict:
    pattern = os.path.join(run_dir, "results", "pops_best", "population_generation_*.json")
    candidates = sorted(glob.glob(pattern), key=_generation_number)
    if not candidates:
        raise RuntimeError(f"No EoH best-population artifacts found under {pattern}")
    latest = candidates[-1]
    with open(latest, encoding="utf-8") as f:
        individual = _best_from_json(json.load(f))
    if not individual.get("code"):
        raise RuntimeError(f"EoH best artifact has no code field: {latest}")
    individual = dict(individual)
    individual["_source_path"] = latest
    individual["_generation"] = _generation_number(latest)
    return individual


def run_self_evolve(
    run_id: str,
    paper_id: str,
    primary_model: str,
    prompt: str,
    config: Dict,
    stage1_instances: List[str],
    stage2_instances: List[str],
    test_instances: List[str],
    stage1_time_limit: int,
    stage2_time_limit: int,
    test_time_limit: int,
    stage1_gap_threshold: float,
    exec_mode: str,
    exec_cfg: Dict,
    t_max,
    secondary_model: Optional[str] = None,
    stage2_scorer: str = "staged_qte",
    stage2_stage_boundary: float = 0.01,
    stage2_time_policy: str = "uniform",
    stage2_time_buffer: int = 0,
    test_time_policy: str = "uniform",
    test_time_buffer: int = 0,
    test_instance_workers: int = 4,
    pop_size: int = 4,
    n_pop: int = 4,
    workers: int = 1,
    timeout: Optional[int] = None,
    operators: Optional[List[str]] = None,
    resume: bool = False,
    enable_artifact: bool = False,
    system_include_spec: bool = False,
):
    del secondary_model
    merged_config = dict(load_eoh_config())
    merged_config.update(config or {})

    model_name = eval_core.get_model_short_name(primary_model)
    base_dir = eval_modes.mode_run_dir(run_id, "eoh", paper_id, model_name)
    eoh_run_dir = os.path.join(base_dir, "eoh_run")
    eval_dir = os.path.join(base_dir, "eoh_adapter")
    selection_instance = stage1_instances[0] if stage1_instances else (
        stage2_instances[0] if stage2_instances else "tiny"
    )
    final_instances = list(test_instances)
    reporting_instances = final_instances or list(stage2_instances)
    timeout = timeout if timeout is not None else stage2_time_limit + 60

    problem = EohBenchmarkProblem(
        paper_id=paper_id,
        model_name=model_name,
        prompt=prompt,
        base_output=eval_dir,
        stage1_instances=stage1_instances,
        stage2_instances=stage2_instances,
        stage1_time_limit=stage1_time_limit,
        stage2_time_limit=stage2_time_limit,
        stage1_gap_threshold=stage1_gap_threshold,
        exec_mode=exec_mode,
        exec_cfg=exec_cfg,
        t_max=t_max,
        stage2_scorer=stage2_scorer,
        stage2_stage_boundary=stage2_stage_boundary,
        stage2_time_policy=stage2_time_policy,
        stage2_time_buffer=stage2_time_buffer,
        enable_artifact=enable_artifact,
    )

    env = prepare_eoh_env(os.environ, merged_config)
    with patched_env(env):
        run_eoh(
            problem=problem,
            output_dir=eoh_run_dir,
            primary_model=primary_model,
            config=merged_config,
            pop_size=pop_size,
            n_pop=n_pop,
            workers=workers,
            timeout=timeout,
            operators=operators,
            system_include_spec=system_include_spec,
            resume=resume,
        )

    best = find_best_eoh_individual(eoh_run_dir)
    best_program_path = materialize_candidate(
        best["code"],
        os.path.join(base_dir, "best", "best_program.py"),
    )

    if final_instances:
        final_results = eval_modes.evaluate_best_on_test_set(
            paper_id, model_name, best_program_path, final_instances,
            test_time_limit, test_time_policy, test_time_buffer,
            os.path.join(base_dir, "selected_eval"),
            exec_mode, exec_cfg, t_max,
            max_workers=test_instance_workers,
        )
    else:
        cached = problem.read_cached_metrics(best["code"])
        stage2_metrics = (cached or {}).get("stage2_metrics")
        if stage2_metrics:
            final_results = reconstruct_results_from_metrics(stage2_metrics, reporting_instances)
            print(
                f"[final:eoh] reconstructed {len(final_results)} rows from EoH eval cache "
                f"(generation={best.get('_generation', '?')})"
            )
        else:
            final_results = {}
            print("[final:eoh] no cached stage2 metrics found; dev results CSV rows will be empty")

    selected_code = eval_modes.copy_selected_code(best_program_path, base_dir)

    # iteration_found = best individual's generation number (each EoH
    # generation evolves the whole population); generation (lineage depth)
    # is left blank since EoH individuals don't store parent_id in saved JSON.
    cached = problem.read_cached_metrics(best["code"])
    stage2_metrics = (cached or {}).get("stage2_metrics") or {}
    if stage2_metrics:
        dev_results_for_csv = reconstruct_results_from_metrics(
            stage2_metrics, list(stage2_instances),
        )
    else:
        dev_results_for_csv = {}
    test_results_for_csv = final_results if final_instances else {}
    eval_modes.write_self_evolve_results(
        paper_id=paper_id,
        model_name=model_name,
        framework="eoh",
        dev_instances=list(stage2_instances),
        dev_results=dev_results_for_csv,
        dev_seed_results={},   # EoH has no single "seed" concept (initial pop is operator-i1)
        test_instances=list(test_instances),
        test_results=test_results_for_csv,
        iteration_found=best.get("_generation"),
        generation=None,
        run_id=run_id,
    )

    eval_modes.write_api_cost_row(
        run_id,
        "self_evolve",
        paper_id,
        primary_model,
        model_name,
        {},
        note="EoH internal LLM usage is tracked by EoH logs.",
    )
    return {"candidate_id": "eoh_best", "results": final_results, "code_path": selected_code}
