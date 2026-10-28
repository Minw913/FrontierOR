import argparse
import json
import math
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_prefix_counts(pi_k: List[int], D: int) -> List[int]:
    """prefix[t] = number of periods in Pi_k that are <= t, for t in [0..D]."""
    mark = [0] * (D + 1)
    for p in pi_k:
        if 1 <= p <= D:
            mark[p] = 1
    prefix = [0] * (D + 1)
    run = 0
    for t in range(D + 1):
        run += mark[t]
        prefix[t] = run
    return prefix


def overlap_count(prefix: List[int], start_t: int, proc_time: int, D: int) -> int:
    """Activity running on periods start_t+1 .. start_t+proc_time."""
    if proc_time <= 0:
        return 0
    left = min(max(start_t, 0), D)
    right = min(max(start_t + proc_time, 0), D)
    if right <= left:
        return 0
    return prefix[right] - prefix[left]


def earliest_start_relaxation(
    activity_ids: List[int],
    temporal_constraints: List[Tuple[int, int, int]],
    start_activity: int,
    D: int,
) -> Dict[int, int]:
    """
    Simple longest-path style relaxation for lower bounds:
      S_j >= S_i + lag, S_i >= 0, and S_start fixed to 0.
    Used only as a MIP start heuristic.
    """
    es = {i: 0 for i in activity_ids}
    es[start_activity] = 0

    n = len(activity_ids)
    for _ in range(n - 1):
        updated = False
        for i, j, lag in temporal_constraints:
            cand = es[i] + lag
            if cand > es[j]:
                es[j] = cand
                updated = True
        # nonnegative starts
        for i in activity_ids:
            if es[i] < 0:
                es[i] = 0
        # fix dummy start at 0
        es[start_activity] = 0
        if not updated:
            break

    # clamp to modeling horizon for time-indexed vars
    for i in activity_ids:
        es[i] = max(0, min(D, int(es[i])))

    return es


def solve_instance(instance: dict, time_limit: int, logger: SolutionLogger = None) -> dict:
    num_real = int(instance["num_real_activities"])
    num_total = int(instance["num_activities_total"])
    D = int(instance["max_project_duration"])
    end_activity = num_real + 1
    start_activity = 0

    activities = instance["activities"]
    proc_time = {int(a["id"]): int(a["processing_time"]) for a in activities}
    activity_ids = sorted(proc_time.keys())

    temporal_constraints = []
    for c in instance["temporal_constraints"]:
        i = int(c["from_activity"])
        j = int(c["to_activity"])
        lag = int(c["time_lag"])
        temporal_constraints.append((i, j, lag))

    resources = instance["resources"]

    model = gp.Model("prcpsp_ps")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(max(1, time_limit))

    # x[i,t] = 1 if activity i starts at time t, with t in 0..D
    x = {}
    for i in activity_ids:
        for t in range(D + 1):
            x[(i, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    model.update()

    # Each activity starts exactly once
    for i in activity_ids:
        model.addConstr(gp.quicksum(x[(i, t)] for t in range(D + 1)) == 1, name=f"one_start_{i}")

    # Dummy start fixed at 0
    if (start_activity, 0) in x:
        model.addConstr(x[(start_activity, 0)] == 1, name="fix_start_0")
    for t in range(1, D + 1):
        if (start_activity, t) in x:
            model.addConstr(x[(start_activity, t)] == 0, name=f"fix_start_{t}")

    # Start time expressions
    S_expr = {
        i: gp.quicksum(t * x[(i, t)] for t in range(D + 1))
        for i in activity_ids
    }

    # Temporal constraints: S_j - S_i >= lag
    for idx, (i, j, lag) in enumerate(temporal_constraints):
        model.addConstr(S_expr[j] - S_expr[i] >= lag, name=f"temp_{idx}")

    # Partially renewable resource constraints
    # sum_i sum_t demand(i,k) * overlap(i,t,Pi_k) * x[i,t] <= capacity_k
    for rk, r in enumerate(resources):
        pi_k = [int(v) for v in r["Pi_k"]]
        cap = int(r["capacity"])
        demands_raw = r["demands"]  # keys likely strings

        prefix = build_prefix_counts(pi_k, D)

        lhs_terms = []
        for i in activity_ids:
            dem = int(demands_raw.get(str(i), demands_raw.get(i, 0)))
            if dem == 0:
                continue
            p_i = int(proc_time.get(i, 0))
            if p_i <= 0:
                continue
            for t in range(D + 1):
                ov = overlap_count(prefix, t, p_i, D)
                if ov == 0:
                    continue
                coeff = dem * ov
                lhs_terms.append(coeff * x[(i, t)])

        if lhs_terms:
            model.addConstr(gp.quicksum(lhs_terms) <= cap, name=f"res_{rk}")
        else:
            model.addConstr(0 <= cap, name=f"res_{rk}_trivial")

    # Objective: minimize start time of project-end activity
    model.setObjective(S_expr[end_activity], GRB.MINIMIZE)

    # Heuristic MIP start
    es = earliest_start_relaxation(activity_ids, temporal_constraints, start_activity, D)
    for i in activity_ids:
        t0 = es[i]
        for t in range(D + 1):
            x[(i, t)].Start = 1.0 if t == t0 else 0.0

    # Incumbent logging callback
    model._logger = logger
    model._last_logged_obj = math.inf

    def cb(m: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < m._last_logged_obj:
                m._last_logged_obj = obj
                m._logger.log(float(obj))

    model.optimize(cb)

    # Extract best solution found
    schedule = {}
    if model.SolCount > 0:
        for i in activity_ids:
            chosen_t = 0
            best_val = -1.0
            for t in range(D + 1):
                val = x[(i, t)].X
                if val > best_val:
                    best_val = val
                    chosen_t = t
            schedule[str(i)] = int(chosen_t)

        objective_value = float(schedule.get(str(end_activity), int(round(model.ObjVal))))
    else:
        # Fallback (format-valid) output
        for i in activity_ids:
            schedule[str(i)] = 0
        objective_value = float(schedule.get(str(end_activity), 0))

    return {
        "objective_value": objective_value,
        "schedule": schedule
    }


def main():
    parser = argparse.ArgumentParser(description="Solve partially renewable RCPSP with temporal lags.")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()