import argparse
import json
import math
from bisect import bisect_right

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_activity_data(activities_obj, act_id):
    """Robustly fetch activity record whether keys are stored as str or int."""
    if str(act_id) in activities_obj:
        return activities_obj[str(act_id)]
    if act_id in activities_obj:
        return activities_obj[act_id]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    num_activities = int(inst["num_activities"])
    d_bar = int(inst["d_bar"])

    start_id = 0
    end_id = n + 1 if (n + 1) < num_activities else (num_activities - 1)

    activity_ids = list(range(num_activities))

    # Parse durations and demands
    activities_obj = inst["activities"]
    durations = {}
    demands = {}  # demands[i][k] = int

    resource_ids = []
    resources_obj = inst["resources"]
    # Keep resource ids as strings (schema keys are strings)
    for rk in resources_obj.keys():
        resource_ids.append(str(rk))
    resource_ids.sort(key=lambda x: int(x) if str(x).lstrip("-").isdigit() else str(x))

    for i in activity_ids:
        rec = get_activity_data(activities_obj, i)
        if rec is None:
            # Default for missing fictitious activity definitions
            durations[i] = 0
            demands[i] = {k: 0 for k in resource_ids}
        else:
            durations[i] = int(rec.get("processing_time", 0))
            rd = rec.get("resource_demands", {})
            di = {}
            for k in resource_ids:
                if k in rd:
                    di[k] = int(rd[k])
                elif str(k) in rd:
                    di[k] = int(rd[str(k)])
                else:
                    di[k] = 0
            demands[i] = di

    # Parse temporal constraints
    temporal_constraints = []
    for c in inst["temporal_constraints"]:
        i = int(c["i"])
        j = int(c["j"])
        delta = int(c["delta"])
        temporal_constraints.append((i, j, delta))

    # Build resource metadata
    capacities = {}
    periods_sorted = {}
    for k in resource_ids:
        rk = resources_obj[k] if k in resources_obj else resources_obj[str(k)]
        capacities[k] = int(rk["capacity"])
        plist = [int(p) for p in rk.get("periods", [])]
        plist.sort()
        periods_sorted[k] = plist

    # Simple bound tightening for start times (optional but helpful)
    LB = {i: 0 for i in activity_ids}
    UB = {i: d_bar for i in activity_ids}
    LB[start_id] = 0
    UB[start_id] = 0
    UB[end_id] = min(UB[end_id], d_bar)

    # Forward relax for lower bounds (difference constraints)
    for _ in range(num_activities - 1):
        changed = False
        for i, j, delta in temporal_constraints:
            cand = LB[i] + delta
            if cand > LB[j]:
                LB[j] = cand
                changed = True
        LB[start_id] = 0
        if not changed:
            break

    # Backward relax for upper bounds
    for _ in range(num_activities - 1):
        changed = False
        for i, j, delta in temporal_constraints:
            cand = UB[j] - delta
            if cand < UB[i]:
                UB[i] = cand
                changed = True
        UB[start_id] = 0
        UB[end_id] = min(UB[end_id], d_bar)
        if not changed:
            break

    # Create domains
    domains = {}
    infeasible_pre = False
    for i in activity_ids:
        if i == start_id:
            domains[i] = [0]
            continue
        lo = max(0, int(math.ceil(LB[i])))
        hi = min(d_bar, int(math.floor(UB[i])))
        if lo > hi:
            infeasible_pre = True
            domains[i] = []
        else:
            domains[i] = list(range(lo, hi + 1))

    if infeasible_pre:
        # Write fallback output (no feasible domain)
        out = {
            "objective_value": 0,
            "start_times": {str(i): 0 for i in activity_ids},
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    # Build MIP
    model = gp.Model("rcpsp_partially_renewable")
    model.Params.TimeLimit = int(args.time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    x = {}
    for i in activity_ids:
        for t in domains[i]:
            x[(i, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    model.update()

    # Exactly one start time per activity
    for i in activity_ids:
        model.addConstr(gp.quicksum(x[(i, t)] for t in domains[i]) == 1, name=f"assign_{i}")

    # Start-time expressions
    s_expr = {
        i: gp.quicksum(t * x[(i, t)] for t in domains[i])
        for i in activity_ids
    }

    # Temporal constraints: s_j - s_i >= delta
    for idx, (i, j, delta) in enumerate(temporal_constraints):
        model.addConstr(s_expr[j] - s_expr[i] >= delta, name=f"tc_{idx}")

    # Partially renewable resource capacity constraints
    for k in resource_ids:
        cap = capacities[k]
        plist = periods_sorted[k]
        expr = gp.LinExpr()

        if len(plist) > 0 and cap < GRB.INFINITY:
            for i in activity_ids:
                dur = durations[i]
                dem = demands[i].get(k, 0)
                if dur <= 0 or dem <= 0:
                    continue
                for t in domains[i]:
                    # count periods p in plist such that t < p <= t + dur
                    cnt = bisect_right(plist, t + dur) - bisect_right(plist, t)
                    if cnt > 0:
                        expr += (dem * cnt) * x[(i, t)]

        model.addConstr(expr <= cap, name=f"res_{k}")

    # Objective: minimize start time of project-end activity
    model.setObjective(s_expr[end_id], GRB.MINIMIZE)

    # Optional MIP start from lower bounds
    for i in activity_ids:
        target = 0 if i == start_id else max(domains[i][0], min(domains[i][-1], int(math.ceil(LB[i]))))
        for t in domains[i]:
            x[(i, t)].Start = 1.0 if t == target else 0.0

    # Callback for incumbent logging
    model._logger = logger
    model._best_obj = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-6 < m._best_obj:
                m._best_obj = obj
                if m._logger:
                    m._logger.log(int(round(obj)))

    model.optimize(cb)

    # Extract best available solution
    start_times = {}
    if model.SolCount > 0:
        for i in activity_ids:
            chosen_t = None
            for t in domains[i]:
                if x[(i, t)].X > 0.5:
                    chosen_t = t
                    break
            if chosen_t is None:
                # fallback in case of numerical issues
                chosen_t = min(domains[i], key=lambda tt: abs(x[(i, tt)].X - 1.0))
            start_times[str(i)] = int(chosen_t)
        objective_value = int(start_times[str(end_id)])
    else:
        # No feasible incumbent found within time limit
        for i in activity_ids:
            start_times[str(i)] = 0
        objective_value = 0

    out = {
        "objective_value": int(objective_value),
        "start_times": start_times
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()