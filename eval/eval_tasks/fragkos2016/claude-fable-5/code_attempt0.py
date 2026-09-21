import argparse
import json

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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["n_items"]
    T = data["n_periods"]
    cap = data["capacity"]
    init_inv = data["initial_inventory"]
    d = data["demands"]
    sc = data["setup_costs"]
    hc = data["holding_costs"]
    vc = data["variable_costs"]
    st = data["setup_times"]
    vt = data["variable_times"]
    bigM = data["big_M"]

    m = gp.Model("clsp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, args.time_limit - 5)

    # Variables
    x = m.addVars(n, T, lb=0.0, name="x")
    s = m.addVars(n, T + 1, lb=0.0, name="s")
    y = m.addVars(n, T, vtype=GRB.BINARY, name="y")

    # Fix initial inventory and final inventory to zero
    for i in range(n):
        s[i, 0].LB = init_inv[i]
        s[i, 0].UB = init_inv[i]
        s[i, T].UB = 0.0

    # Inventory balance
    for i in range(n):
        for t in range(T):
            m.addConstr(s[i, t] + x[i, t] == d[i][t] + s[i, t + 1])

    # Setup linking: tighten big-M with remaining demand
    for i in range(n):
        rem = sum(d[i][t] for t in range(T))
        for t in range(T):
            M = bigM[i][t]
            # remaining demand from t onward is also a valid bound
            M = min(M, rem) if rem > 0 else M
            if vt[i][t] > 0:
                M = min(M, max(0.0, (cap[t] - st[i][t]) / vt[i][t]))
            m.addConstr(x[i, t] <= M * y[i, t])
            rem -= d[i][t]

    # Capacity constraints
    for t in range(T):
        m.addConstr(
            gp.quicksum(st[i][t] * y[i, t] + vt[i][t] * x[i, t] for i in range(n))
            <= cap[t]
        )

    # Objective
    m.setObjective(
        gp.quicksum(
            sc[i][t] * y[i, t] + vc[i][t] * x[i, t] + hc[i][t] * s[i, t + 1]
            for i in range(n)
            for t in range(T)
        ),
        GRB.MINIMIZE,
    )

    def build_solution(xv, sv, yv, obj):
        return {
            "objective_value": obj,
            "x": [[max(0.0, xv[i][t]) for t in range(T)] for i in range(n)],
            "s": [[max(0.0, sv[i][t]) for t in range(T + 1)] for i in range(n)],
            "y": [[int(round(yv[i][t])) for t in range(T)] for i in range(n)],
        }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            xv = [[model.cbGetSolution(x[i, t]) for t in range(T)] for i in range(n)]
            sv = [[model.cbGetSolution(s[i, t]) for t in range(T + 1)] for i in range(n)]
            yv = [[model.cbGetSolution(y[i, t]) for t in range(T)] for i in range(n)]
            sol = build_solution(xv, sv, yv, obj)
            if logger:
                logger.log_solution(obj, sol)

    m.optimize(callback if logger else None)

    if m.SolCount > 0:
        xv = [[x[i, t].X for t in range(T)] for i in range(n)]
        sv = [[s[i, t].X for t in range(T + 1)] for i in range(n)]
        yv = [[y[i, t].X for t in range(T)] for i in range(n)]
        sol = build_solution(xv, sv, yv, m.ObjVal)
        if logger:
            logger.log_solution(m.ObjVal, sol)
    else:
        # No feasible solution found; output an empty/trivial structure
        sol = {
            "objective_value": float("inf"),
            "x": [[0.0] * T for _ in range(n)],
            "s": [[0.0] * (T + 1) for _ in range(n)],
            "y": [[0] * T for _ in range(n)],
        }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()