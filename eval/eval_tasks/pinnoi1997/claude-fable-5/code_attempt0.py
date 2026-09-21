import argparse
import json
import math
import sys

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["num_tasks"]
    c = data["cycle_time"]
    m = data["num_stations"]
    times = {int(k): int(v) for k, v in data["task_processing_times"].items()}
    tasks = sorted(times.keys())
    arcs = [(int(a), int(b)) for a, b in data.get("precedence_arcs", [])]
    E = {int(k): int(v) for k, v in data["earliest_station"].items()}
    L = {int(k): int(v) for k, v in data["latest_station"].items()}

    total_time = sum(times[i] for i in tasks)
    # Lower bound on worst-case idle time
    lb_W = max(0, math.floor((m * c - total_time) / c))

    model = gp.Model("salbp_balance")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, args.time_limit - 5)
    model.Params.OutputFlag = 0

    # x[i, s] = 1 if task i assigned to station s
    x = {}
    for i in tasks:
        lo = max(1, E.get(i, 1))
        hi = min(m, L.get(i, m))
        if lo > hi:
            lo, hi = 1, m  # fallback if bounds inconsistent
        for s in range(lo, hi + 1):
            x[i, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}")

    W = model.addVar(vtype=GRB.INTEGER, lb=lb_W, ub=c, name="W")

    stations_of = {i: sorted(s for (j, s) in x.keys() if j == i) for i in tasks}

    # Each task assigned exactly once
    for i in tasks:
        model.addConstr(gp.quicksum(x[i, s] for s in stations_of[i]) == 1)

    # Station capacity and worst-case idle
    for s in range(1, m + 1):
        load = gp.quicksum(times[i] * x[i, s] for i in tasks if (i, s) in x)
        model.addConstr(load <= c)
        model.addConstr(load + W >= c)

    # Precedence: station(a) <= station(b)
    # Strong form: sum_{s<=k} x[b,s] <= sum_{s<=k} x[a,s] for each relevant k
    for (a, b) in arcs:
        sa = stations_of[a]
        sb = stations_of[b]
        lo = max(min(sa), min(sb))
        hi = min(max(sa), max(sb))
        if lo > hi:
            # bounds already enforce precedence
            continue
        for k in range(lo, hi + 1):
            model.addConstr(
                gp.quicksum(x[b, s] for s in sb if s <= k)
                <= gp.quicksum(x[a, s] for s in sa if s <= k)
            )

    model.setObjective(W, GRB.MINIMIZE)

    best = {"obj": None, "assign": None}

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best["obj"] is not None and obj >= best["obj"] - 1e-9:
                return
            vals = mdl.cbGetSolution([x[key] for key in x])
            assign = {}
            for (key, v) in zip(x.keys(), vals):
                if v > 0.5:
                    assign[str(key[0])] = int(key[1])
            best["obj"] = obj
            best["assign"] = assign
            if logger:
                logger.log_solution(float(obj), {
                    "objective_value": float(obj),
                    "task_assignments": assign,
                })

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        assign = {}
        for (i, s), var in x.items():
            if var.X > 0.5:
                assign[str(i)] = int(s)
        obj = float(model.ObjVal)
        if best["obj"] is None or obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["assign"] = assign
            if logger:
                logger.log_solution(obj, {
                    "objective_value": obj,
                    "task_assignments": assign,
                })

    if best["assign"] is None:
        # Fallback: trivial (possibly infeasible) assignment to earliest stations
        assign = {str(i): max(1, min(m, E.get(i, 1))) for i in tasks}
        loads = {s: 0 for s in range(1, m + 1)}
        for i in tasks:
            loads[int(assign[str(i)])] += times[i]
        obj = float(max(c - loads[s] for s in loads))
        best["obj"] = obj
        best["assign"] = assign
        if logger:
            logger.log_solution(obj, {
                "objective_value": obj,
                "task_assignments": assign,
            })

    solution = {
        "objective_value": float(best["obj"]),
        "task_assignments": best["assign"],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()