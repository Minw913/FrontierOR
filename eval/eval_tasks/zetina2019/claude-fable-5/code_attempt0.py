import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_nodes = data["num_nodes"]
    arcs = [tuple(a) for a in data["arcs"]]
    fixed_costs = data["fixed_costs"]
    variable_costs = data["variable_costs"]
    commodities = data["commodities"]
    K = len(commodities)

    def akey(a):
        return f"{a[0]}_{a[1]}"

    model = gp.Model("fcnd")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining = max(1.0, args.time_limit - (time.time() - start_time) - 2.0)
    model.Params.TimeLimit = remaining

    y = {}
    for a in arcs:
        y[a] = model.addVar(vtype=GRB.BINARY, obj=float(fixed_costs[akey(a)]), name=f"y_{akey(a)}")

    x = {}
    for k, com in enumerate(commodities):
        d = float(com["demand"])
        vc = {a: float(variable_costs[akey(a)][k]) for a in arcs}
        for a in arcs:
            x[a, k] = model.addVar(lb=0.0, ub=1.0, obj=d * vc[a], name=f"x_{akey(a)}_{k}")

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # Flow conservation
    in_arcs = {n: [] for n in range(num_nodes)}
    out_arcs = {n: [] for n in range(num_nodes)}
    for a in arcs:
        out_arcs[a[0]].append(a)
        in_arcs[a[1]].append(a)

    for k, com in enumerate(commodities):
        o, dst = com["origin"], com["destination"]
        for n in range(num_nodes):
            rhs = 0.0
            if n == o:
                rhs = -1.0
            elif n == dst:
                rhs = 1.0
            model.addConstr(
                gp.quicksum(x[a, k] for a in in_arcs[n])
                - gp.quicksum(x[a, k] for a in out_arcs[n])
                == rhs
            )

    # Forcing constraints
    for a in arcs:
        for k in range(K):
            model.addConstr(x[a, k] <= y[a])

    def build_solution(yvals, xvals, obj):
        open_arcs = {}
        for a in arcs:
            if yvals[a] > 0.5:
                open_arcs[akey(a)] = 1
        routings = {}
        for k in range(K):
            rk = {}
            for a in arcs:
                v = xvals[a, k]
                if v > 1e-6:
                    rk[akey(a)] = round(v, 9)
            routings[str(k)] = rk
        return {
            "objective_value": obj,
            "open_arcs": open_arcs,
            "routings": routings,
        }

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    yvals = {a: m.cbGetSolution(y[a]) for a in arcs}
                    xvals = {(a, k): m.cbGetSolution(x[a, k]) for a in arcs for k in range(K)}
                    sol = build_solution(yvals, xvals, obj)
                    logger.log_solution(obj, sol)
                except Exception:
                    logger.log(obj)

    model.optimize(callback)

    if model.SolCount > 0:
        obj = model.ObjVal
        yvals = {a: y[a].X for a in arcs}
        xvals = {(a, k): x[a, k].X for a in arcs for k in range(K)}
        sol = build_solution(yvals, xvals, obj)
        if logger:
            logger.log_solution(obj, sol)
    else:
        # Fallback: open all arcs, no meaningful routing found (should be rare)
        sol = {
            "objective_value": float("inf"),
            "open_arcs": {akey(a): 1 for a in arcs},
            "routings": {str(k): {} for k in range(K)},
        }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()