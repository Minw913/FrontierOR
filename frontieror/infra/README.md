# FrontierOR Trusted Evaluation Infrastructure

This package is an additive trust boundary around the upstream FrontierOR
research pipeline. The original one-shot and self-evolution commands remain
available for reproducibility. Use this package whenever the code-producing
agent or the submitted solver is not trusted.

## Entry points

```bash
# Show the versioned scoring and visibility contracts.
python -m frontieror.infra contract

# Run the platform CORAL adapter under the mandatory agent profile.
python -m frontieror.infra agent --help

# Verify an immutable code-only submission.
python -m frontieror.infra submission --help

# Attack the built candidate image through the official execution path.
python -m frontieror.infra security-check --candidate-image frontieror-candidate:1
```

The `agent` command fixes these settings and does not expose downgrade flags:

- CORAL is the supported platform adapter.
- Agent processes and candidate solvers run in Docker.
- Model credentials remain in a trusted, model-pinned proxy sidecar.
- Dev evaluation goes through a trusted request broker.
- `staged_qte` uses trusted host wall-clock time.
- A non-empty final set is evaluated only after `code.py` is frozen.
- Dev and final instances must be disjoint.

The old `--anti-hack` option remains an internal compatibility switch for
existing scripts. It is not the public agent-mode interface.

## Tide-eval orchestration

FrontierOR integrates with
[`Human-Agent-Society/tide-eval`](https://github.com/Human-Agent-Society/tide-eval)
through its public `Executor` protocol. Tide-eval owns episode scheduling,
stable-key resume, budgets, traces, and the SQLite Lab. The trusted FrontierOR
worker still owns Agent Docker, Candidate Docker, brokered dev scoring,
artifact freezing, and hidden final grading. This division keeps untrusted
`code.py` outside the hidden checker's filesystem and protection domain.

Install tide-eval in a separate environment so its optional Harbor dependency
does not change the benchmark environment:

```bash
git clone https://github.com/Human-Agent-Society/tide-eval reference/tide-eval
uv sync --project reference/tide-eval --dev
export TIDE_EVAL_PYTHON="$PWD/reference/tide-eval/.venv/bin/python"
```

Install each self-evolution framework before selecting it:

```bash
bash test_time_self_evolution/openevolve/setup.sh
bash test_time_self_evolution/eoh/setup.sh
bash test_time_self_evolution/coral/setup.sh
```

One command creates one Tide episode per paper. CORAL uses the official agent
profile; OpenEvolve and EoH use the same hardened Docker evaluator and hidden
final scorer:

```bash
python -m frontieror.infra tide-eval \
  --framework coral \
  --lab eval/tide/coral \
  --concurrency 1 \
  -- \
  --paper-id bierwirth2017 \
  --primary-model openai/gpt-5.4 \
  --stage1-instances tiny \
  --dev-set large_1 \
  --test-set large_2 \
  --coral-agent-count 1 \
  --coral-attempts 10 \
  --run-id tide-coral-smoke

python -m frontieror.infra tide-eval \
  --framework openevolve \
  --lab eval/tide/openevolve \
  -- \
  --paper-id bierwirth2017 \
  --primary-model gpt-5.4 \
  --model-backend local-codex \
  --openevolve-iterations 1 \
  --stage1-instances tiny \
  --dev-set large_1 \
  --test-set large_2 \
  --run-id tide-openevolve-smoke
```

Use `--framework eoh` with the corresponding `--eoh-*` controls for EoH.
The local Codex bridge is loopback-only, model-pinned, concurrency-bounded,
and starts Codex with a read-only empty workspace and optional capabilities
disabled. Candidate solver code never receives access to this bridge.

This integration does not use the unrelated `gauthierpiarrette/tide` fleet
manager. Current tide-eval exposes `Lab` and `Executor`; FrontierOR implements
that interface directly instead of maintaining a fork.

## Package boundaries

The security-sensitive implementation is owned by this package:

- `contracts.py` and `policy.py` define the public score/visibility contract
  and the non-overridable Agent profile;
- `visibility.py` builds the exact dev workspace exposed to the Agent;
- `execution.py` owns candidate isolation, bounded process execution, image
  pinning, and restricted WLS egress;
- `agent/` owns the model proxy, request broker, trusted grader bridge, Agent
  container lifecycle, and audit records;
- `submission/` owns immutable Code-only bundles and public/private traces;
- `security_check.py` runs deploy-time black-box attacks against a built image.

Files retained at the old `scripts/utils` and `test_time_self_evolution/coral`
paths are compatibility aliases. Upstream research commands keep their import
surface, while official behavior has one canonical implementation under
`frontieror/infra`.

## Container images

Install the pinned CORAL checkout and configure the platform-owned model key:

```bash
bash test_time_self_evolution/coral/setup.sh
export OPENROUTER_API_KEY="<platform-openrouter-key>"
```

The official proxy profile expects a full OpenRouter route such as
`openai/gpt-5.4`. The route is reduced to the Codex short model name only
inside the Agent container; the trusted proxy retains the full upstream route
and credential.

Run `setup.sh` after activating the benchmark Python environment. It fails if
that interpreter imports any CORAL installation other than the pinned checkout.
Use `--coral-max-seconds auto` for normal runs: the wall-clock cap covers both
Agent reasoning and brokered dev evaluation, so a manually shortened cap may
expire while an otherwise valid request is still being graded.

Build the three independently versioned images from the repository root:

```bash
docker build \
  -f frontieror/infra/docker/candidate.Dockerfile \
  -t frontieror-candidate:1 .

docker build \
  -f frontieror/infra/docker/agent.Dockerfile \
  -t frontieror-coral-agent:0.1 .

docker build \
  -f frontieror/infra/docker/model-proxy.Dockerfile \
  -t frontieror-coral-model-proxy:0.1 .
```

The candidate image is intentionally separate from the upstream `Dockerfile`.
This keeps the research environment compatible with FrontierOR while allowing
the official runner dependencies and image digest to be frozen independently.
The three infra Dockerfiles pin their base-image index digests; update those
digests only as an explicit runner release and rerun `security-check` afterward.

## Visibility contract

| Artifact | Agent/framework while developing | Frozen solver at final runtime | Trusted grader |
|---|---:|---:|---:|
| Problem and input/output schemas | Yes | Yes | Yes |
| Instance-level scoring formula | Yes | Yes | Yes |
| Stage-1 and dev instance JSON | Yes | As needed | Yes |
| Final instance JSON | No | Current instance only, read-only | Yes |
| Dev aggregate feedback | Yes | No | Yes |
| Dev per-instance score and trace | No | No | Yes |
| Final aggregate score | Only after the run ends | No | Yes |
| Final per-instance score | No | No | Yes |
| Final instance membership | No | No | Yes |
| Reference objective/runtime | No | No | Yes |
| Reference solution and checker | No | No | Yes |
| Private per-instance trace | No | No | Yes |

The distinction around final JSON is important. An optimization algorithm must
read the instance it is solving, so the frozen solver necessarily sees that
single JSON at execution time. The agent and its optimization framework do not
see final instances before selecting and freezing `code.py`, and receive no
final per-instance feedback.

The scoring equation is public because it is part of the benchmark definition.
The values substituted into it for a final instance are private. Hiding the
equation would reduce reproducibility without preventing leakage or reward
hacking.

Dev feedback is deliberately an optimization signal, not a secret. The API
returns an aggregate and with a one-instance dev set that value is necessarily
the instance score; what remains withheld is the reference objective/runtime,
checker output, and detailed trace. This does not weaken the final boundary,
because neither final per-instance values nor final feedback are available
during artifact selection.

Filename separation alone is not a hidden test. Official final instances must
come from an unpublished server-only pool; a file already shipped in the public
Hugging Face dataset remains public even when omitted from a particular Agent
workspace. Startup hashes the configured JSON files and rejects renamed or
duplicated content across dev and final.

Every public dev workspace contains:

```text
<paper_id>/
  problem_description.txt
  instance_schema.json
  solution_schema.json
  instance/<stage1-or-dev>.json
  benchmark_contract.json
  public_manifest.json
```

It does not contain final instances, feasibility checkers, Gurobi solutions or
logs, aggregate reference CSVs, or private traces.

## Evaluation boundary

```text
public dev workspace
        |
        v
untrusted agent container(s) -- commit + nonce --> trusted request broker
        ^                                            |
        |                                  aggregate dev feedback
        +--------------------------------------------+

agent stopped -> selected commit -> frozen code.py hash
                                    |
                                    v
                     one final instance per candidate container
                                    |
                                    v
                   trusted checker + private reference + trace
                                    |
                                    v
                        aggregate public result only
```

Isolation is the primary control. Detection is defense in depth:

- submission and broker schemas reject links, traversal, oversized files,
  unregistered agents, duplicate commits, and commits outside agent branches;
- candidate containers have a read-only root, no capabilities, no privilege
  escalation, PID/CPU/RAM/file limits, bounded output capture, and no general
  network route;
- output files are pre-created and promoted through no-follow bounded reads;
- trusted feasibility code parses the promoted solution in a second read-only,
  resource-bounded checker container, not in the Agent or candidate process;
- checker preflight requires the trusted reference to pass and objective-only
  tampering in both directions to fail;
- final scoring uses host wall-clock time and never trusts candidate-written
  convergence timestamps;
- the hardened runner passes `max(1, floor(T)-2)` to `code.py` while retaining
  the declared `T`-second host hard deadline, reserving two seconds for Docker
  startup and output serialization without granting compute grace;
- private traces record the artifact hash, image digest, policy, result, and
  selected commit. Public rows contain aggregate fields only.

`security-check` is the image release gate for the candidate boundary. It runs
host-file/environment canaries plus root-write and public-network probes against
both candidate and checker containers, then adds candidate timeout-escape and
output-flood probes. This is deliberately black-box: a passing result depends
on observed container behavior, not only on the generated Docker command.

## Model and Gurobi access

The official `agent` entry point keeps the upstream model credential in a
trusted sidecar, pins one model ID, issues per-agent ephemeral tokens, and
records request/response hashes. It does not expose a credential downgrade
flag. The upstream research CLI still supports `local-auth` for controlled
internal experiments; because that mode mounts an existing Codex login into
the Agent container, it is intentionally outside the official profile.

Candidate networking is disabled unless WLS is enabled. With WLS, the solver
network is internal and its only egress path is an exact-host CONNECT proxy for
`token.gurobi.com:443`. The model proxy and WLS proxy are separate trust
domains. A WLS credential is still readable by solver code inside its container;
use a dedicated, revocable evaluation license.

## Reference designs

The architecture follows established benchmark patterns while tightening the
boundary for FrontierOR's executable solver setting:

- [OpenAI MLE-bench](https://github.com/openai/mle-bench) gives agents public
  competition data and places answers behind a grading service. FrontierOR uses
  a separate host broker so the submitted system never shares a filesystem
  protection domain with final references.
- [NVIDIA SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) uses formal
  problem/workload/solution/trace schemas and tests concrete reward-hacking
  behaviors. FrontierOR likewise treats per-instance traces and malicious
  submissions as testable contracts.
- [Inspect](https://inspect.aisi.org.uk/sandboxing.html) models sandboxes per
  sample, bounds file/output operations, and supports network-disabled
  containers. FrontierOR applies the same per-instance execution boundary.

MLE-bench also documents real leakage failures in public test artifacts and has
paused new leaderboard submissions while improving fairness and comparability.
The practical lesson is that sandboxing alone is insufficient: split manifests,
checker conformance, versioned contracts, and attack-regression tests must all
be release gates.

## Scope

The current `agent` command supports the platform-maintained CORAL adapter. It
does not yet claim to execute an arbitrary third-party framework container.
Adding such adapters requires the entire submitted orchestrator, not only its
model subprocess, to implement this same artifact, network, budget, and final
freeze contract.

Docker is a process boundary, not protection against host-kernel or Docker
runtime vulnerabilities. A public multi-tenant service should place each run in
a disposable VM or equivalent microVM, use authenticated queues and rate limits,
deduplicate final evaluations by frozen artifact hash, and store signed
append-only audit records outside the runner. Submission quotas are part of the
privacy boundary: unlimited aggregate final queries can turn the leaderboard
into an oracle even when per-instance feedback is hidden.
