# FrontierOR Trusted Evaluation Infrastructure

This package is an additive trust boundary around the upstream FrontierOR
research pipeline. The original one-shot and self-evolution commands remain
available for reproducibility. Use this package whenever the code-producing
agent or the submitted solver is not trusted.

## Entry points

```bash
# Show the versioned scoring and visibility contracts.
python -m trusted_eval_infra contract

# Run the platform CORAL adapter under the mandatory agent profile.
python -m trusted_eval_infra agent --help

# Verify an immutable code-only submission.
python -m trusted_eval_infra submission --help

# Attack the built candidate image through the official execution path.
python -m trusted_eval_infra security-check --candidate-image frontieror-candidate:1
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
python -m trusted_eval_infra tide-eval \
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

python -m trusted_eval_infra tide-eval \
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
`trusted_eval_infra`.

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
  -f trusted_eval_infra/docker/candidate.Dockerfile \
  -t frontieror-candidate:1 .

docker build \
  -f trusted_eval_infra/docker/agent.Dockerfile \
  -t frontieror-coral-agent:0.1 .

docker build \
  -f trusted_eval_infra/docker/model-proxy.Dockerfile \
  -t frontieror-coral-model-proxy:0.1 .
```

The candidate image is intentionally separate from the upstream `Dockerfile`.
This keeps the research environment compatible with FrontierOR while allowing
the official runner dependencies and image digest to be frozen independently.
The three trusted-eval Dockerfiles pin their base-image index digests; update those
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
                   dataset checker + private reference + trace
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

### CORAL unscored development tests

In secure CORAL runs, agents can use `coral debug -m "check tiny execution"`
before submitting with `coral eval`. The trusted broker runs the committed
`code.py` on the **public tiny instance only**, in the candidate Docker image
with the configured candidate memory/CPU limits and existing restricted WLS
license access. The agent image does not need Gurobi or license credentials for
this command. Rebuild `trusted_eval_infra/docker/agent.Dockerfile` after updating
the CLI shim.

Debug is a container-backed development test, not execution inside the agent
shell. It returns bounded runtime output and a solution preview (64 KiB limit),
without a checker, reference objective, feasibility verdict or score. It creates
no formal attempt and is not eligible for final-program selection. Limits are
30 seconds per debug execution and 30 admitted debug requests per task, with
one pending debug execution per task; these are separate from the scored attempt
budget and consume compute and task wall time. A WLS queue wait is bounded to
60 seconds and can fail if the execution slot remains occupied.

Debug requests and results are recorded under `.coral/private/debug/`; bounded
responses are published through the existing read-only request-status mount.
The host validates the agent branch, commit tree and unchanged public inputs.
There is no agent-selectable instance path, shell command, host path or test-set
parameter. Rejected requests do not consume a scored attempt. Original
benchmark runs are not changed by enabling this command.

### CORAL final tests on the host (optional mixed isolation)

The original all-Docker path remains the default: `--coral-final-exec-mode same`
inherits `--exec-mode`. To keep the agent, debug tests and development evaluator
in Docker, but evaluate the selected fixed final program on host systemd scopes,
add these options to the CORAL self-evolve command:

```bash
--exec-mode docker --anti-hack \
--coral-final-exec-mode systemd \
--coral-final-cpus 1 \
--coral-final-memory 100G \
--test-instance-workers 4 \
--test-time-limit 3600 --test-time-policy uniform
```

CPU and memory overrides apply **per final instance**, not per task or shared
across the four instances. If omitted they inherit `--cpus` and `--memory`.
Systemd enforces CPU quota, CPU affinity and a cgroup memory cap; the existing
trusted executor enforces the wall deadline. A resource preflight checks real
user systemd CPU/memory cgroup enforcement before evolving a program, and fails
instead of falling back to bare execution. WLS-enabled final scopes retain
`IPAddressDeny=any` and add resolved WLS endpoint IPs plus configured DNS resolver
IPs. These are IP-level permissions, not a full network or filesystem sandbox;
endpoint changes can require refreshed configuration on a subsequent execution.
The global remaining WLS allowance must still be checked separately.

Candidate execution now reserves logical CPUs through cross-process `flock`
leases shared by bare, systemd, and Docker backends. Leases are held for execution
and cleanup, and CPUs outside the evaluator's allowed affinity are never chosen.
Independent paper/evaluator processes therefore cannot reserve the same logical
CPU concurrently. The coordination directory is `/tmp/frontieror-cpu-slots-<uid>`;
lock files persist, but active ownership is maintained by the kernel. This is
cooperative per-user allocation, not exclusion of unrelated host processes or SMT
siblings. Do not delete lock files while evaluations are running. Existing
processes started before this change do not participate in the lease protocol.

Each instance writes `<solution_path>.cpu.json` with assigned CPUs, owner PID,
and acquisition/release timestamps. CPU admission waits occur before the
candidate execution timer; they default to 86400 seconds and can be bounded with
`exec_cfg.cpu_queue_timeout_seconds`. Normal completion, exceptions and execution
timeouts release the reservation; stopping a run must also stop its candidate
systemd scopes/containers before admitting a replacement run. The self-evolve CLI
now forwards `--systemd-allow-wls` into evaluator and final-test configuration so
configured WLS execution slots are actually enforced.


`--coral-final-exec-mode bare` is also available, but does not enforce a cgroup
memory cap. Neither host mode provides Docker filesystem isolation. Both remain
explicit opt-ins and do not change the agent runtime, debug runner, development
scoring, selected program, or final-instance worker/time settings. The original
anti-hack validator still forbids a non-Docker development evaluator.

Each final evaluation writes `final_eval/execution.json` with its backend,
per-instance resource/time configuration and isolation status. Secure CORAL's
`final_artifact.json` includes the same record. A host-final result is labelled
`mixed-agent-docker-final-systemd` or `mixed-agent-docker-final-bare`, with
`full_anti_hack=false`; it must not be reported as a fully hardened final test.
Use a fresh run ID when changing the final backend to avoid mixing historical
results with a different execution protocol. These host options are CORAL-only.

### CORAL best one-shot program starter

Keep the evolving agent on GPT-5.6-Sol + Codex while starting from Claude Fable 5's
final one-shot program for each task:

```bash
--primary-model openai/gpt-5.6-sol \
--coral-agent-runtime codex \
--coral-starter-mode best-one-shot-program \
--coral-attempts 20
```

`--coral-starter-model` defaults to `claude-fable-5`. This is a fixed source model,
selected from the aggregate one-shot comparison; it does not choose a separate
winner per task or run a development-set selection experiment. Only the source
program and adjacent logger are copied, using the existing final snapshot/manifest
validation. Source model, path, code hash, and agent model are recorded in
`seed/_seed_source.json`. Missing or invalid programs fail without LLM generation
fallback. Resume checks the recorded source model and seed hash; switching starter
requires a new run ID. This mode takes priority over the legacy seed-model and
reuse-existing-seed environment variables.

The default `--coral-starter-mode same-model` preserves the previous behavior,
including `FRONTIER_OR_CORAL_SEED_MODEL` overrides and seed generation fallback.

### WLS admission gate for independent candidate environments

For the two-license hardset launchers, set `FRONTIER_OR_WLS_CONCURRENCY=2`,
`FRONTIER_OR_WLS_BACKEND_GATE=1`, and
`FRONTIER_OR_WLS_RELEASE_DELAY_SECONDS=310`. Each license ID has independent
cross-process locks. Docker preflight, development candidates, and WLS-enabled
systemd final evaluation share this gate. Codex/model preflight now precedes the
Gurobi Docker probe. Debug has no license access.

Admission checks read-only WLS sessions and unexpired tokens, deduplicate their
environment IDs, and include local reservations to cover backend visibility lag.
Backend query failures wait without starting execution. Completed executions retain
a reservation for at least 310 seconds; actual longer-lived tokens still block
admission. Local reservations may duplicate environments already visible to WLS,
so this intentionally sacrifices throughput. Queue time is outside the candidate
solve timer but still counts against the overall task wall-clock limit. External
clients not using these locks can still consume license capacity; retain backend
and program-error monitoring. Default execution behavior is unchanged unless the
gate is enabled.

### Grader deadline and orphan cleanup

CORAL's trusted CLI adapter refreshes the saved grader config before every
attempt, and resume synchronizes its grading timeout from the current task YAML.
The overall evaluation deadline includes per-instance WLS queue allowances,
execution budgets, checker/setup overhead, and a margin. Candidate solve limits
are unchanged. The adapter removes containers mounted from the exact grader
checkout on exit, including grader timeouts.

For the already-running 2026-10-29 split runs, the independent
`coral-live-eval-watchdog-20261029` user service provides online compatibility.
It removes containers belonging to terminal attempts and enforces a separate
candidate wall deadline. Old coordinators already holding the short join timeout
are temporarily suspended near that timeout while their independent worker and
Docker solver continue. The coordinator resumes when the worker exits, or at the
new bounded deadline. PID start times are verified before signals; service stop
also releases suspended coordinators. Agents, candidate processes, and the two
self-evolving services are not restarted. Already-started workers may still have
an old inner timer; historical affected attempts are recorded separately under
private/audit/infrastructure_timeouts.json and are not rewritten as successes.

### Live hardset license reassignment (2026-10-30)

The optional host-owned assignment file
`local_docs/coral-runs/hardset-license-assignment-20261030.json` assigns each of the
50 task IDs to exactly one managed run. It transfers 13 of 38 unstarted tasks to
the ZZM license, leaving all previously started tasks in their original runs.
New batch processes skip tasks assigned elsewhere before task execution. Existing
in-memory queues cannot reload that parent logic, so the native `start` adapter
also checks ownership before creating an agent or grader. An old queued start
exits with code 78 and writes a handoff record under
`local_docs/coral-runs/task-handoffs-20261030/`. Those old parent processes may log
this as a task failure and eventually exit nonzero; interpret these explicit
handoff records as intentional skips, not solver failures. Evaluate transferred
tasks using the owner run only. Unmanaged run IDs are unaffected.

The new ZZM group runs with two paper workers under systemd, its own WLS gate,
license monitor and independent cleanup watchdog. Its host subprocess affinity
is restricted to CPUs 40–55, so candidate allocation starts in that range.

ZZM launch outcome: the real Gurobi Docker preflight rejected the license as
expired (backend expiration: 2026-07-23 23:59:59 UTC). No ZZM task or agent started.
The proposed assignment was archived as
`hardset-license-assignment-20261030-blocked-zzm.json`, and the active assignment
was rolled back to all original JCH/KMW owners. Renew the ZZM license before
attempting this handoff again; the original two runs continue unchanged.

The replacement LJY license passed backend validity and actual Gurobi Docker
preflight. On 2026-10-30 the same 13 unstarted tasks were assigned to
`hardset_ljy_13_fable_seed_gpt56sol_a20_20261030` and started with two paper workers.
The active owner distribution is JCH 18 / KMW 19 / LJY 13 tasks. ZZM remains unused.
The original JCH/KMW services were not restarted. LJY has independent monitoring,
cleanup, and the same WLS gate; preflight token cooldown can delay first candidates.

After renewal, ZZM passed real Docker preflight and started a fourth group:
`hardset_zzm_9_fable_seed_gpt56sol_a20_20261030`, paper workers=2. This transfers
9/36 then-unstarted tasks. Current ownership is JCH 14 / KMW 16 / LJY 11 / ZZM 9;
all 50 task IDs remain uniquely assigned. Existing three services were not stopped.
The ZZM host subprocess affinity is CPUs 56–59, with independent WLS monitoring
and cleanup. The earlier failed ZZM 13-task launch remains historical only.

LZK passed real Gurobi Docker preflight and started a fifth group,
`hardset_lzk_7_fable_seed_gpt56sol_a20_20261030`, with two paper workers.
It receives 7/33 then-unstarted tasks, keeping all started tasks in place.
Current unique ownership is JCH 12 / KMW 14 / LJY 9 / ZZM 8 / LZK 7 = 50.
LZK uses host subprocess CPUs 32–39 and independent WLS/cleanup monitors.

OYWB passed actual Gurobi Docker preflight and started the sixth group,
`hardset_oywb_4_fable_seed_gpt56sol_a20_20261030`, with two paper workers and
independent WLS/cleanup monitoring. It receives 4/23 then-unstarted tasks:
bard2002, delage2022, fischetti1997, martin-iradi2024. Current unique ownership is
JCH 12 / KMW 12 / LJY 7 / ZZM 8 / LZK 7 / OYWB 4 = 50. Host subprocess CPU
range is 24–31. Previously running services were not restarted.

### Network exhaustion and failed-startup recovery (2026-10-31)

Agent and candidate WLS networks now request explicit non-overlapping /28 subnets
from `10.240.0.0/16` (overridable with `FRONTIER_OR_DOCKER_NETWORK_POOL`). Allocation
is protected by a cross-process lock and excludes current Docker subnets and host
routes. It does not restart Docker or change existing networks. This avoids
exhausting Docker's small number of default large address pools. Network errors
now preserve the daemon diagnostic instead of only a CalledProcessError.

The trusted manager adapter wraps start/resume failures: stop agents, gateway and
grader before re-raising, so an idle non-daemon grader cannot keep a failed CLI
alive. Existing failed CLI/idle-grader pairs were verified to have no attempts and
no agent runtime audit before termination. Recovery queues resume bard2002,
adulyasak2015, amaldi2013, wang2019 and ozturk2020 in their original owner run IDs.
They run under `coral-network-recover-{oywb,lzk,ljy,kmw}-20261031` user services,
with separate monitors and shared per-license admission gates. Original batch
workers continue onto their queues; recovery may temporarily add agent-level
concurrency but never raises the per-license WLS admission limit.

Validation: 67 relevant tests passed, including real child-grader shutdown on
startup failure. Three real explicit network creations passed. No old empty
CORAL network met cleanup criteria. Two hour-old WLS networks retaining only an
orphan proxy and no candidate were rechecked and removed. Live agent networks,
all unrelated networks, and candidate evaluations were left in place. Audit files
are under `local_docs/coral-runs/network-startup-repair-20261031/`.

QA passed actual Gurobi Docker preflight on 2026-10-31 and received one unstarted
task, roberti2018, from LZK. KMW/LZK tied at six unfinished tasks; LZK had more
remaining attempts. New run `hardset_qa_1_fable_seed_gpt56sol_a20_20261031` uses
one paper worker, host CPUs 8–15, and independent WLS/cleanup monitors. Unique
ownership is JCH 12 / KMW 12 / LJY 7 / ZZM 8 / LZK 6 / OYWB 4 / QA 1 = 50.
All existing experiments and recovery queues remain running.

### Closeout repairs (2026-10-31)

The evolution wall deadline includes WLS queue time; `--coral-max-seconds auto`
resolved to 85,200 seconds for the current 20-attempt runs. It is independent
of the attempt count, and excludes the subsequent final evaluation.

Workspace quota counting excludes only the host-private broker checkout root,
whose immutable input copies previously prematurely stopped `fragkos2021`.
Agent-created directories remain counted. New supervisors persist
`private/audit/termination.json` even when no selected artifact exists and stop
with `agent_no_submission_progress` after eight runtime restarts without a new
submission; pending evaluations are exempt. Running processes are not patched.

The local `fragkos2016` task now explicitly describes priced, variable initial
inventory and its checker enforces the reference implementation's initial-stock
lower bound. The dataset is outside parent Git tracking; portable patch,
backups, test evidence and recovery limitations are in
`local_docs/coral-runs/closeout-repair-20261031/`. Historical results are retained;
a corrected task needs a separately identified reevaluation.
