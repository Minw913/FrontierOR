import argparse
import json
import time
import random
from typing import Dict, List, Tuple, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_operation_index(instance: Dict[str, Any]):
    jobs = instance["jobs"]
    n = instance["num_jobs"]

    job_ops: List[List[int]] = [[] for _ in range(n)]
    op_job: List[int] = []
    op_idx_in_job: List[int] = []
    op_machine: List[int] = []
    op_p: List[float] = []

    machine_to_ops: Dict[int, List[int]] = {m: [] for m in range(instance["num_machines"])}

    op_id = 0
    for j in range(n):
        ops = jobs[j]["operations"]
        for k, op in enumerate(ops):
            job_ops[j].append(op_id)
            op_job.append(j)
            op_idx_in_job.append(k)
            m = int(op["machine"])
            p = float(op["processing_time"])
            op_machine.append(m)
            op_p.append(p)
            machine_to_ops[m].append(op_id)
            op_id += 1

    return job_ops, op_job, op_idx_in_job, op_machine, op_p, machine_to_ops


def evaluate_schedule(instance: Dict[str, Any], start_times_by_job: List[List[float]]) -> Tuple[float, List[float], List[float]]:
    jobs = instance["jobs"]
    n = instance["num_jobs"]

    completion = [0.0] * n
    tardiness = [0.0] * n
    obj = 0.0

    for j in range(n):
        ops = jobs[j]["operations"]
        last_k = len(ops) - 1
        c = start_times_by_job[j][last_k] + float(ops[last_k]["processing_time"])
        completion[j] = c
        t = max(0.0, c - float(jobs[j]["due_date"]))
        tardiness[j] = t
        obj += float(jobs[j]["weight"]) * t

    return obj, completion, tardiness


def greedy_construct(instance: Dict[str, Any], rng: random.Random, randomized: bool = False) -> List[List[float]]:
    """
    Serial schedule generation:
    At each step, choose one next operation among all jobs.
    """
    n = instance["num_jobs"]
    m = instance["num_machines"]
    jobs = instance["jobs"]

    next_op = [0] * n
    job_ready = [float(jobs[j]["release_date"]) for j in range(n)]
    machine_ready = [0.0] * m

    start_times = [[0.0] * len(jobs[j]["operations"]) for j in range(n)]
    remaining = sum(len(j["operations"]) for j in jobs)

    while remaining > 0:
        candidates = []
        for j in range(n):
            k = next_op[j]
            if k >= len(jobs[j]["operations"]):
                continue
            op = jobs[j]["operations"][k]
            mach = int(op["machine"])
            p = float(op["processing_time"])
            est = max(job_ready[j], machine_ready[mach])

            # Priority metrics
            due = float(jobs[j]["due_date"])
            w = float(jobs[j]["weight"])
            slack = due - (est + p)
            crit = w / max(1.0, p)
            candidates.append((j, k, mach, p, est, due, slack, crit))

        if not candidates:
            break

        if not randomized:
            # Deterministic: earliest start, then higher criticality, then earlier due date
            candidates.sort(key=lambda x: (x[4], -x[7], x[5], x[0]))
            chosen = candidates[0]
        else:
            # Randomized restricted candidate list around earliest start
            min_est = min(c[4] for c in candidates)
            threshold = min_est + rng.uniform(0.0, 3.0)
            rcl = [c for c in candidates if c[4] <= threshold]
            if not rcl:
                rcl = candidates
            # Weighted pick favoring higher crit and tighter slack
            scores = []
            for c in rcl:
                # larger is better
                score = (2.0 * c[7]) + (1.0 / max(1.0, c[6] + 5.0))
                score *= rng.uniform(0.85, 1.15)
                scores.append(max(1e-6, score))
            ssum = sum(scores)
            r = rng.random() * ssum
            acc = 0.0
            idx = 0
            for i, sc in enumerate(scores):
                acc += sc
                if acc >= r:
                    idx = i
                    break
            chosen = rcl[idx]

        j, k, mach, p, est, _, _, _ = chosen
        start_times[j][k] = est
        end = est + p
        job_ready[j] = end
        machine_ready[mach] = end
        next_op[j] += 1
        remaining -= 1

    return start_times


def convert_start_by_job_to_op_array(start_by_job, job_ops):
    num_ops = sum(len(x) for x in job_ops)
    op_starts = [0.0] * num_ops
    for j, ops in enumerate(job_ops):
        for k, op_id in enumerate(ops):
            op_starts[op_id] = float(start_by_job[j][k])
    return op_starts


def build_output(instance: Dict[str, Any], start_by_job: List[List[float]]) -> Dict[str, Any]:
    obj, completion, tardiness = evaluate_schedule(instance, start_by_job)
    schedule_out = []
    for j, job in enumerate(instance["jobs"]):
        ops_out = []
        for k, op in enumerate(job["operations"]):
            ops_out.append({
                "machine": int(op["machine"]),
                "start_time": float(start_by_job[j][k])
            })
        schedule_out.append({
            "job_id": int(job["job_id"]),
            "completion_time": float(completion[j]),
            "tardiness": float(tardiness[j]),
            "operations": ops_out
        })
    return {"objective_value": float(obj), "schedule": schedule_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["num_jobs"])
    m = int(instance["num_machines"])
    jobs = instance["jobs"]

    job_ops, op_job, op_idx_in_job, op_machine, op_p, machine_to_ops = build_operation_index(instance)
    num_ops = len(op_p)

    # ---------- Heuristic phase ----------
    rng = random.Random(42)
    best_start = greedy_construct(instance, rng, randomized=False)
    best_obj, _, _ = evaluate_schedule(instance, best_start)
    if logger:
        logger.log(best_obj)

    heuristic_budget = min(2.0, 0.2 * float(args.time_limit))
    while time.time() - t0 < heuristic_budget:
        cand = greedy_construct(instance, rng, randomized=True)
        cand_obj, _, _ = evaluate_schedule(instance, cand)
        if cand_obj + 1e-9 < best_obj:
            best_obj = cand_obj
            best_start = cand
            if logger:
                logger.log(best_obj)

    elapsed = time.time() - t0
    remaining = max(0.0, float(args.time_limit) - elapsed)

    # If no time left, output heuristic
    if remaining <= 0.05:
        sol = build_output(instance, best_start)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # ---------- MIP phase ----------
    model = gp.Model("jobshop_weighted_tardiness")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = remaining
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    s = model.addVars(num_ops, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    C = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="C")
    T = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="T")

    max_release = max(float(j["release_date"]) for j in jobs) if n > 0 else 0.0
    total_p = sum(op_p)
    Mbig = max_release + total_p + 1.0

    # Job constraints
    for j in range(n):
        first_op = job_ops[j][0]
        model.addConstr(s[first_op] >= float(jobs[j]["release_date"]), name=f"rel_{j}")

        for k in range(1, len(job_ops[j])):
            prev_op = job_ops[j][k - 1]
            cur_op = job_ops[j][k]
            model.addConstr(s[cur_op] >= s[prev_op] + op_p[prev_op], name=f"prec_{j}_{k}")

        last_op = job_ops[j][-1]
        model.addConstr(C[j] == s[last_op] + op_p[last_op], name=f"comp_{j}")
        model.addConstr(T[j] >= C[j] - float(jobs[j]["due_date"]), name=f"tard_{j}")

    # Machine disjunctive constraints
    y = {}
    pair_list = []
    for mach in range(m):
        ops = machine_to_ops[mach]
        ln = len(ops)
        for i in range(ln):
            a = ops[i]
            for k in range(i + 1, ln):
                b = ops[k]
                y[(a, b)] = model.addVar(vtype=GRB.BINARY, name=f"y_{a}_{b}")
                pair_list.append((a, b))
    model.update()

    for (a, b) in pair_list:
        yab = y[(a, b)]
        model.addConstr(s[a] + op_p[a] <= s[b] + Mbig * (1 - yab), name=f"mach1_{a}_{b}")
        model.addConstr(s[b] + op_p[b] <= s[a] + Mbig * yab, name=f"mach2_{a}_{b}")

    model.setObjective(gp.quicksum(float(jobs[j]["weight"]) * T[j] for j in range(n)), GRB.MINIMIZE)

    # Warm start from heuristic
    op_start_ws = convert_start_by_job_to_op_array(best_start, job_ops)
    for op in range(num_ops):
        s[op].Start = op_start_ws[op]

    ws_obj, ws_completion, ws_tard = evaluate_schedule(instance, best_start)
    for j in range(n):
        C[j].Start = ws_completion[j]
        T[j].Start = ws_tard[j]

    # Set y start according to warm-start order on each machine
    for (a, b) in pair_list:
        sa = op_start_ws[a]
        sb = op_start_ws[b]
        if sa < sb - 1e-9:
            y[(a, b)].Start = 1.0
        elif sb < sa - 1e-9:
            y[(a, b)].Start = 0.0
        else:
            # tie-break by op id
            y[(a, b)].Start = 1.0 if a < b else 0.0

    # Callback logging incumbents
    model._logger = logger
    model._best_logged = best_obj

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < mdl._best_logged:
                mdl._best_logged = obj_val
                if mdl._logger:
                    mdl._logger.log(float(obj_val))

    model.optimize(cb)

    # ---------- Extract best solution ----------
    if model.SolCount > 0:
        start_by_job = [[0.0] * len(jobs[j]["operations"]) for j in range(n)]
        for j in range(n):
            for k, op_id in enumerate(job_ops[j]):
                start_by_job[j][k] = float(s[op_id].X)
        solution = build_output(instance, start_by_job)
    else:
        # Fallback to heuristic
        solution = build_output(instance, best_start)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()