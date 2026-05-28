# CORAL → frontier-or 接入计划

> 参照 `self_evolving_frameworks/openevolve/` 的接入形态，把 CORAL 这套
> multi-agent 编排框架接到本 bench 的 `self_evolve` 模式上来。
>
> CORAL 与 OpenEvolve 在三件事上根本不同，决定了接入工作量比 OpenEvolve 大：
>
> 1. **进化主体不同**：OpenEvolve 自己就是算法（LLM 当 mutation 算子）；CORAL
>    本身**不进化**，它编排若干"真编码 agent"（Claude Code / Codex / OpenCode）
>    在 git worktree 里循环改代码 + commit + 评分。
> 2. **驱动方式不同**：OpenEvolve 一个 `iterations=N` 跑完就停；CORAL **没有
>    自然终止条件**——agent 一直循环直到你停它，需要我们外加 stop policy。
> 3. **LLM 调用层不同**：OpenEvolve 用 OpenAI SDK 直调；CORAL 通过 **litellm
>    gateway**（默认 `localhost:4000`）让 agent runtime 透明换模型。OpenRouter
>    要走 litellm 配置而不是 SDK base_url。


---
# User Instruction

---

# Task

接入 CORAL 要做的工程清单。每条都对照 OpenEvolve 已有的等价产物，方便实现时套用。

## T1. 前置环境

| 依赖 | 处理 |
|---|---|
| **CORAL 源码** | `external/coral/`，已 clone（pin 到 `e633b552…`，setup.sh 已就位） |
| **uv 包管理器** | CORAL 文档推荐 `uv sync`。若 venv 用 pip 也可，但 `pyproject.toml` 没声明 console_script，需要 `pip install -e external/coral` 后才有 `coral` CLI |
| **编码 agent runtime（必选其一）** | OpenCode / Claude Code / Codex。**推荐 OpenCode**——开源 + 可走 litellm gateway + 与现有 OpenRouter 路径一致；Claude Code 需 Anthropic 直接 auth，Codex 需 OpenAI |
| **litellm gateway** | CORAL 的 LLM 抽象层。我们需要写一个 `litellm_config.yaml` 把 CORAL 看到的模型名（`claude-sonnet`、`gpt-5-codex` 等）路由到 OpenRouter 端点 |
| **`coral` CLI 在 PATH** | `pip install -e external/coral` 之后 `coral` 命令应可用；启动时 `--output-dir` 决定 `.coral/` 落点 |

**setup.sh 修改**：开启 `pip install -e ${TARGET_DIR}`（当前注释掉），加一行 sanity check 确认 `coral --version` 可执行。

## T2. 要新写的文件（`self_evolving_frameworks/coral/`）

参照 `self_evolving_frameworks/openevolve/` 的产物，对应映射：

| OpenEvolve | CORAL 等价物 | 职责 |
|---|---|---|
| `openevolve/runner.py` | **`coral/runner.py`**（新写 ~300–400 行） | 主编排：搭脚手架 → 起 `coral start` 子进程 → 等终止 → 取 best attempt → final eval → 写 CSV |
| `openevolve/evaluator.py` | **`coral/efficient_or_grader.py`**（新写 ~150 行） | TaskGrader 子类，实现 `evaluate()`，bridge 到 `eval_core.evaluate_candidate_code()` + `scoring/` |
| `configs/openevolve.yaml` | **`coral/task_template.yaml`** | 每 paper 跑前用 `format()`/yaml 生成具体的 `task.yaml` |
| —（OpenEvolve 不需要） | **`coral/litellm_template.yaml`** | OpenRouter 路由配置模板，每 run 实例化 |
| —（OpenEvolve 不需要） | **`coral/seed_template/opencode.json`** | OpenCode 配置（指向 litellm gateway 的 baseURL `http://localhost:4000/v1`） |

新增的两类模板（litellm yaml + agent runtime json）是 CORAL 特有的成本，OpenEvolve 没这层。

## T3. 每跑 1 个 paper 的脚手架（`runner.run_self_evolve` 内部）

CORAL 要求每个任务是一个完整目录树。我们给每个 `(run_id, paper_id, model)` 组合
动态生成一份在 `eval/modes/<run_id>/self_evolve/<paper_id>/<model>/coral_task/` 下：

```
coral_task/
├── task.yaml                    # 由 task_template.yaml 渲染：把 prompt 填进 task.description
├── eval/
│   └── grader.py                # ★ 不复制 grader 代码，而是 import-shim：
│                                #   `from self_evolving_frameworks.coral.efficient_or_grader import Grader`
│                                #   靠 PYTHONPATH 让 CORAL 子进程能找到我们的 bench 代码
├── seed/
│   ├── solve.py                 # eval_core.generate_candidate_code 生的 seed solve()，
│   │                            #   外面包一层 `# EVOLVE-BLOCK-START / END`
│   └── opencode.json            # 由 seed_template/opencode.json 复制，gateway baseURL/model 实例化
├── litellm_config.yaml          # 由 litellm_template.yaml 渲染，注入 OPENROUTER_API_KEY
└── results/                     # CORAL 写 .coral/{public/attempts,private,...} 进这里
```

**关键 trade-off — 共用 grader 代码 vs 复制**：

- 复制 grader 代码到 `eval/grader.py`：CORAL 标准做法，自包含但每改一次 grader 都要 regenerate
- import-shim：`eval/grader.py` 只一行 `from self_evolving_frameworks.coral.efficient_or_grader import Grader`，CORAL 跑前我们设 `PYTHONPATH=<bench_root>:$PYTHONPATH`。**推荐这个**——单一事实来源、改 grader 不用 regenerate per-paper

## T4. 起子进程 + 生命周期管理

OpenEvolve 是 `subprocess.run(['python', 'openevolve-run.py', ...], check=True)`——同步、自然终止。CORAL 不一样：

- `coral start -c task.yaml` 会**启动 tmux/docker session 跑 agents**（默认 detached），命令本身**立刻返回**
- 要等结束需要主动调 `coral status` 轮询、或读 `.coral/public/attempts/` 的更新
- 没有原生的 "stop after N attempts"——需要外部驱动

**方案选项**（plan 里需要拍板）：

| 选项 | 实现 | 优点 | 缺点 |
|---|---|---|---|
| **A. wall-clock budget** | `coral start ...` → sleep `iterations * 60` → `coral stop` | 简单 | 时间换 attempt 数不稳定（agent 速度波动大） |
| **B. attempts-count poll** | poll `.coral/public/attempts/` 文件数，达到 `iterations` 个时调 `coral stop` | 直接对应 OpenEvolve 的 `iterations` 参数 | 需要写 polling loop；如果 agent 卡住会无限等 |
| **C. attempts-count poll + wall-clock 兜底** | B + 软超时（比如 `max_wall = iterations * 120s`） | 健壮 | 代码量略增 |

**推荐 C**——3–4 行 loop 的事，不复杂。

`coral stop` 之后还要：
1. 读 `<task_dir>/results/.coral/public/attempts/*.json`，按 `score` 降序排第一名
2. 拿 `attempt.commit_hash`，在该 worktree 里 `git checkout <hash>` → 拷贝 `solve.py` 到我们 bench 的 `selected/code.py`
3. 调 `eval_core.evaluate_candidate_code(test_instances, ...)` 做 final test eval
4. `eval_modes.write_summary_rows(...)` / `write_candidate_rows(...)` 落 CSV

## T5. 与 `run_eval_modes.py` 的对接

加一行 dispatch（参考 OpenEvolve 现有 pattern）：

```python
# self_evolving_frameworks/__init__.py 或 run_eval_modes.py 顶部
from self_evolving_frameworks.coral import runner as coral_runner

FRAMEWORKS = {
    "openevolve": openevolve_runner.run_self_evolve,
    "coral": coral_runner.run_self_evolve,
}
```

`runner.run_self_evolve(...)` 的签名**保持与 OpenEvolve 一致**——这样 `run_eval_modes.py` 不区分 framework 实现细节。

---

# Eval

CORAL 的评分契约（`Grader.evaluate() -> ScoreBundle`）和我们的 `eval_core` + `scoring/` 怎么对齐。

## E1. Stage1 / Stage2 / Test 三阶映射

OpenEvolve 原生支持 cascade evaluator（先 stage1 quick gate，再 stage2 main score）。**CORAL 不支持**——每个 attempt 只能跑一次 grader、返回一个 ScoreBundle。

折中方案：在我们的 `Grader.evaluate()` **内部按需求自串两阶**：

```python
def evaluate(self) -> ScoreBundle:
    # Stage1: 在 stage1_instances 上跑，feasibility-gate
    s1_results = eval_core.evaluate_candidate_code(paper_id, ..., stage1_instances, ...)
    if any_infeasible(s1_results) or worst_gap(s1_results) > stage1_gap_threshold:
        return self._make_bundle(0.0, feedback="stage1 failed")

    # Stage2: 通过 gate 后才在 stage2_instances 上跑主要打分
    s2_results = eval_core.evaluate_candidate_code(paper_id, ..., stage2_instances, ...)
    score = scorer.aggregate([scorer.score_instance(r, ctx) for r in s2_results])
    return self._make_bundle(score, feedback=f"avg_g_at_τ={...}")
```

成本：每个 attempt 都跑两层评估。OpenEvolve 用 cascade 是为了**早期 reject** 便宜，CORAL 没原生支持就只能多花一些 stage1 时间。但因为 stage1 用 `tiny` instance + 短 time_limit，成本可控。

## E2. ScoreBundle ↔ scoring/ 的桥

CORAL `ScoreBundle.aggregated` 是单一 float，CORAL 框架按 `direction: maximize/minimize` 排序 attempts。我们 `scoring/` 输出 0–1 高优分。

- 在 `task.yaml` 里写 `direction: maximize`
- `evaluate()` 把 `scorer.aggregate([...])` 直接当 `aggregated` 返回
- 顺手把每 instance 的 `g_at_τ_g`、`time_penalty_weight`、`feasible` 写进 `Score.metadata` 字段，CORAL 的 dashboard 和 attempt JSON 里就能看到详情

复用：**`scoring/` 一行不改**——`fixed_gurobi` / `convergence_speed` 都直接用。

## E3. Final eval（best attempt → test_instances）

CORAL 里 attempts 评分用的是 stage2_instances（可能比较小、训练用）。最终我们要在 `test_instances` 上重跑一遍 best attempt 拿 reportable 数字——和 OpenEvolve 流程一致：

```python
# In runner.run_self_evolve, after coral stop:
best = pick_best_attempt(coral_dir)            # 读 .coral/public/attempts/*.json
best_code_path = checkout_attempt(best, worktree_dir)  # git checkout + 拷贝 solve.py
final_results = eval_core.evaluate_candidate_code(
    paper_id, ..., test_instances, best_code_path, final_dir,
    per_instance_tl, exec_mode, exec_cfg, t_max,
)
write_candidate_rows(...)
write_summary_rows(...)
```

如果 `test_instances` 为空（采用 OpenEvolve 优化过的"读 checkpoint metric"路径），CORAL 等价做法是**直接读 best attempt 的 ScoreBundle.metadata 里我们埋的 per-instance 信息**——但 stage2_instances 可能 ≠ test_instances，所以这条路只在两者相等时安全。**v1 实现里强制要求 test_instances 非空**，后续再做无 test 优化。

## E4. 时间策略 & 每实例 budget

复用 OpenEvolve 那套 env 变量+`scoring.building_blocks.lookup_gurobi_time` 的逻辑：
- `EFFICIENT_OR_STAGE2_TIME_POLICY ∈ {uniform, gurobi_time, gurobi_time_plus_buffer}`
- `EFFICIENT_OR_STAGE2_TIME_BUFFER`（秒）
- `EFFICIENT_OR_STAGE2_TIME_LIMIT`（cap）

但传入 grader 的方式不一样——OpenEvolve 通过 env vars，CORAL 通过 `task.yaml.grader.args` 字段。两条路：

- **(a)** task.yaml 里塞 `args: { paper_id, instances, time_limit, ... }`，grader 读 `self.args`
- **(b)** 仍用 env vars，runner 起 `coral start` 时把 env vars 设好，grader 子进程继承

**推荐 (a)** — task.yaml 是 CORAL 原生的"per-task config" 容器，把这些参数写进去更符合 CORAL 的设计。也能直接写进 attempt JSON 让 dashboard 看到。

## E5. LLM 成本核算

OpenEvolve 我们追踪的是 **seed 生成那 1 次 LLM call** 的 token usage（OpenEvolve 内部进化的 LLM 调用单独有它自己的 logs）。CORAL 同理：
- Seed 生成（用 `eval_core.generate_candidate_code`）→ 走我们 bench 的 cost 计算
- CORAL agent 内部所有 LLM 调用 → **走 litellm gateway**——litellm 自带 logging（默认写到 `litellm_logs/`），我们可以在 run 结束时聚合

**v1 处理**：`write_api_cost_row` 只记 seed 那次（同 OpenEvolve），note 字段写 "Coral internal LLM usage tracked by litellm gateway logs at <path>"。后续可加 litellm log 解析器。

## E6. Iteration / Stop 条件

如 T4 所述，**采用 attempts-count poll + wall-clock 兜底**：

```python
target_attempts = iterations
max_wall_seconds = iterations * 180  # adjustable
deadline = time.time() + max_wall_seconds

while time.time() < deadline:
    n_attempts = count_attempts(coral_dir)
    if n_attempts >= target_attempts:
        break
    time.sleep(15)

subprocess.run(["coral", "stop"], cwd=task_dir)
```

意义对照：OpenEvolve 的 `iterations=20` ≈ "尝试 20 个候选程序"；CORAL 的"20 个 attempt" ≈ "agent 20 次 commit-and-eval"。语义稍有不同（CORAL agent 一个 attempt 可能改了多个文件做综合改进，单 attempt 信息密度更高），但作为预算单位可比。

## E7. 落 CSV / 落表

完全复用 `eval_modes.write_*_rows()`。每行的 `code_path` 字段指向我们最终拷出的 `selected/code.py`（best attempt 的 commit checkout 后取出的版本）。

---

# Open Questions（实现前需要拍板）

1. **编码 agent 选哪个？** 推荐 **OpenCode**（开源 + litellm 友好 + 与现有 OpenRouter 路径一致）；如果你已经在本机有 Claude Code 配置好的，用 Claude Code 也可。
2. **uv 还是 pip？** 简洁起见 `pip install -e external/coral`（无新依赖管理器）。如果未来要用 CORAL 的 swebench/datasets 选装包再考虑 uv。
3. **是否每次 reuse 同一份 coral_task 目录？** 同一 paper 多次跑可以重用 task.yaml + seed/，让 `.coral/` 共享 attempts、notes、skills（这就是 CORAL 的 cross-run learning 卖点）。但 reproducibility 上每次 fresh 更好。**推荐：默认 fresh，加一个 `--reuse-coral-state` flag。**
4. **multi-paper 并发？** 同一台机器跑多 paper 时 litellm gateway 端口冲突（默认 4000）。要么每 paper 一个端口（`port = 4000 + hash(paper_id) % 1000`），要么强制 sequential。**v1 推荐 sequential**；并发以后做。
5. **CORAL 的 sharing 要不要打开？** `.coral/notes/` 和 `.coral/skills/` 是 CORAL 的核心特性（agent 经验跨 attempt 复用）。打开它符合论文意图但增加 reproducibility 复杂度。**推荐：v1 打开**（CORAL 的卖点就是这个，关掉就和 OpenCode 单 agent 没区别），只在 paper 内复用，paper 之间不串。
6. **`Grader.evaluate()` 的运行时**——CORAL 在哪个进程跑 grader？看 `task_grader.py` 的 `get_python_command()` 实现：grader 默认在 CORAL 主进程的 worker pool 里跑（async）。这意味着我们的 grader 必须 import 得到 `eval_core` 和 `scoring/`——意味着 **CORAL 进程的 PYTHONPATH 必须包含 bench root**。`runner.py` 里设好 env 即可。

---

# 验收清单（v1 接入完成的标志）

- [ ] `bash self_evolving_frameworks/coral/setup.sh` 一次运行 → `coral --version` 可用
- [ ] `coral/runner.py:run_self_evolve(...)` 签名与 OpenEvolve 完全一致
- [ ] `coral/efficient_or_grader.py` 通过 unit test（mock `eval_core.evaluate_candidate_code` 验 ScoreBundle 形状）
- [ ] 一个 smoke test 脚本 `coral/smoke_test.py`，类似 ReEvo 的——用极小 budget（attempts=2，wall=120s）跑通一个 paper（建议 bodur2017 这种小问题）
- [ ] `python run_eval_modes.py self_evolve --framework coral --paper-id <x> --iterations 2 ...` 端到端跑通，落 `eval/mode_summary.csv` 一行
- [ ] 已有的 `tests/test_eval_modes.py` + `tests/test_scoring.py` 全绿（不被 CORAL 接入破坏）

---

# 时间预估（参考 OpenEvolve 接入约 2 天）

| 阶段 | 预估 | 备注 |
|---|---|---|
| T1 + T2 (前置 + 模板/grader) | 1 天 | grader 子类 + task_template.yaml + litellm/opencode 配置 |
| T3 (脚手架 + import-shim 接好) | 0.5 天 | 写 + 调 PYTHONPATH 等坑 |
| T4 (subprocess lifecycle + best 提取) | 1 天 | poll + stop + git checkout attempt commit 是 CORAL 特有逻辑 |
| Smoke test + 调 OpenRouter gateway | 0.5 天 | litellm 配置最易踩坑 |
| 完整集成测试（至少 1 个 paper 走通） | 0.5 天 | |
| **合计** | **3.5 天** | 比 OpenEvolve 多 1.5 天，主要在 CORAL agent runtime + litellm 配置 |
