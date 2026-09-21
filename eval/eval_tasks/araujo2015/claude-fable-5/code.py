import argparse
import json
import math

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

    n = data["num_items"]
    T = data["num_periods"]
    demand = data["demand"]
    setup_cost = data["setup_cost"]
    holding_cost = data["holding_cost"]
    variable_cost = data["variable_cost"]
    setup_time = data["setup_time"]
    variable_time = data["variable_time"]
    capacity = data["capacity"]
    init_inv_cost = data["initial_inventory_cost"]

    # Remaining demand from period t to end for each item
    rem_dem = [[0.0] * (T + 1) for _ in range(n)]
    for i in range(n):
        for t in range(T - 1, -1, -1):
            rem_dem[i][t] = rem_dem[i][t + 1] + demand[i][t]

    m = gp.Model("clsp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, args.time_limit - 5)

    # Big-M / upper bounds for production
    ub = [[0.0] * T for _ in range(n)]
    for i in range(n):
        for t in range(T):
            if variable_time[i] > 0:
                cap_bound = (capacity[t] - setup_time[i]) / variable_time[i]
            else:
                cap_bound = float("inf")
            cap_bound = max(0.0, cap_bound)
            ub[i][t] = min(cap_bound, rem_dem[i][t])

    x = m.addVars(n, T, lb=0.0, name="x")
    y = m.addVars(n, T, vtype=GRB.BINARY, name="y")
    s = m.addVars(n, T + 1, lb=0.0, name="s")  # s[i,0] initial inventory

    for i in range(n):
        # Ending inventory must be zero
        m.addConstr(s[i, T] == 0)
        # Initial inventory cannot exceed total demand (implied optimality bound)
        s[i, 0].UB = rem_dem[i][0]
        for t in range(T):
            m.addConstr(s[i, t] + x[i, t] == demand[i][t] + s[i, t + 1])
            m.addConstr(x[i, t] <= ub[i][t] * y[i, t])
            x[i, t].UB = ub[i][t]

    for t in range(T):
        m.addConstr(
            gp.quicksum(setup_time[i] * y[i, t] + variable_time[i] * x[i, t]
                        for i in range(n)) <= capacity[t]
        )

    obj = (
        gp.quicksum(init_inv_cost[i] * s[i, 0] for i in range(n))
        + gp.quicksum(setup_cost[i] * y[i, t] for i in range(n) for t in range(T))
        + gp.quicksum(variable_cost[i] * x[i, t] for i in range(n) for t in range(T))
        + gp.quicksum(holding_cost[i] * s[i, t + 1] for i in range(n) for t in range(T))
    )
    m.setObjective(obj, GRB.MINIMIZE)

    def build_solution(prod, setup, inv, obj_val):
        return {
            "objective_value": obj_val,
            "production": prod,
            "setup": setup,
            "inventory": inv,
        }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            xv = model.cbGetSolution(x)
            yv = model.cbGetSolution(y)
            sv = model.cbGetSolution(s)
            prod = [[max(0.0, xv[i, t]) for t in range(T)] for i in range(n)]
            setup = [[1 if yv[i, t] > 0.5 else 0 for t in range(T)] for i in range(n)]
            inv = [[max(0.0, sv[i, t]) for t in range(T + 1)] for i in range(n)]
            if logger:
                logger.log_solution(obj_val, build_solution(prod, setup, inv, obj_val))

    m.optimize(callback if logger else None)

    if m.SolCount > 0:
        prod = [[max(0.0, x[i, t].X) for t in range(T)] for i in range(n)]
        setup = [[1 if y[i, t].X > 0.5 else 0 for t in range(T)] for i in range(n)]
        inv = [[max(0.0, s[i, t].X) for t in range(T + 1)] for i in range(n)]
        obj_val = m.ObjVal
        sol = build_solution(prod, setup, inv, obj_val)
        if logger:
            logger.log_solution(obj_val, sol)
    else:
        # Fallback: trivial (likely infeasible-free) solution: produce demand each period
        prod = [[float(demand[i][t]) for t in range(T)] for i in range(n)]
        setup = [[1 if demand[i][t] > 0 else 0 for t in range(T)] for i in range(n)]
        inv = [[0.0] * (T + 1) for _ in range(n)]
        obj_val = sum(
            setup_cost[i] * setup[i][t] + variable_cost[i] * prod[i][t]
            for i in range(n) for t in range(T)
        )
        sol = build_solution(prod, setup, inv, float(obj_val))
        if logger:
            logger.log_solution(float(obj_val), sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()