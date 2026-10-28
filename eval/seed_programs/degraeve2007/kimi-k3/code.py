import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser(
        description="Multi-product capacitated lot-sizing problem solver"
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    start = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    P = int(inst["dimensions"]["num_products"])
    T = int(inst["dimensions"]["num_periods"])
    par = inst["parameters"]

    demand = [[float(v) for v in row] for row in par["demand"]]
    holding = [[float(v) for v in row] for row in par["holding_cost"]]
    setup_cost = [[float(v) for v in row] for row in par["setup_cost"]]
    var_cost = [[float(v) for v in row] for row in par["variable_production_cost"]]
    var_time = [[float(v) for v in row] for row in par["variable_production_time"]]
    setup_time = [[float(v) for v in row] for row in par["setup_time"]]
    capacity = [float(v) for v in par["capacity"]]
    init_cost = [float(v) for v in par["initial_inventory_cost"]]

    # Cumulative remaining demand from period t through the horizon.
    cum_dem = [[0.0] * T for _ in range(P)]
    for i in range(P):
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc += demand[i][t]
            cum_dem[i][t] = acc

    EPS = 1e-9

    def make_solution(prod, setp, inv, init):
        """Clean raw values, enforce consistency, and recompute the objective."""
        prod_c = [[float(v) if v > EPS else 0.0 for v in row] for row in prod]
        setp_c = [[1 if v >= 0.5 else 0 for v in row] for row in setp]
        inv_c = [[float(v) if v > EPS else 0.0 for v in row] for row in inv]
        init_c = [float(v) if v > EPS else 0.0 for v in init]
        for i in range(P):
            for t in range(T):
                if prod_c[i][t] > EPS and setp_c[i][t] == 0:
                    setp_c[i][t] = 1
        obj = 0.0
        for i in range(P):
            obj += init_cost[i] * init_c[i]
            for t in range(T):
                obj += (
                    setup_cost[i][t] * setp_c[i][t]
                    + var_cost[i][t] * prod_c[i][t]
                    + holding[i][t] * inv_c[i][t]
                )
        return obj, {
            "objective_value": obj,
            "variables": {
                "production": prod_c,
                "setup": setp_c,
                "inventory": inv_c,
                "initial_inventory": init_c,
            },
        }

    # Trivially feasible plan: cover all demand from (penalized) initial inventory.
    t_prod = [[0.0] * T for _ in range(P)]
    t_setp = [[0] * T for _ in range(P)]
    t_inv = [[0.0] * T for _ in range(P)]
    t_init = [cum_dem[i][0] for i in range(P)]
    for i in range(P):
        run = 0.0
        for t in range(T):
            run += demand[i][t]
            t_inv[i][t] = cum_dem[i][0] - run

    best_obj, best_sol = make_solution(t_prod, t_setp, t_inv, t_init)
    if logger is not None:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    try:
        model = gp.Model("mlclsp")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = max(0.2, args.time_limit - (time.time() - start) - 0.5)

        # Decision variables
        x = model.addVars(P, T, lb=0.0, name="x")          # production
        y = model.addVars(P, T, vtype=GRB.BINARY, name="y")  # setup
        s = model.addVars(P, T, lb=0.0, name="s")          # ending inventory
        I0 = model.addVars(P, lb=0.0, name="I0")           # initial inventory

        model.setObjective(
            gp.quicksum(init_cost[i] * I0[i] for i in range(P))
            + gp.quicksum(setup_cost[i][t] * y[i, t] for i in range(P) for t in range(T))
            + gp.quicksum(var_cost[i][t] * x[i, t] for i in range(P) for t in range(T))
            + gp.quicksum(holding[i][t] * s[i, t] for i in range(P) for t in range(T)),
            GRB.MINIMIZE,
        )

        # Inventory balance constraints
        for i in range(P):
            model.addConstr(I0[i] + x[i, 0] - s[i, 0] == demand[i][0])
            for t in range(1, T):
                model.addConstr(s[i, t - 1] + x[i, t] - s[i, t] == demand[i][t])

        # Setup forcing constraints with tightened big-M
        for i in range(P):
            for t in range(T):
                vt = var_time[i][t]
                cap_rem = capacity[t] - setup_time[i][t]
                cap_based = (
                    cap_rem / vt
                    if vt > 1e-9
                    else (float("inf") if cap_rem >= 0.0 else 0.0)
                )
                big_m = cap_based if cap_based < cum_dem[i][t] else cum_dem[i][t]
                if big_m <= 1e-9:
                    y[i, t].UB = 0.0
                    x[i, t].UB = 0.0
                else:
                    model.addConstr(x[i, t] <= big_m * y[i, t])

        # Shared resource capacity constraints
        for t in range(T):
            model.addConstr(
                gp.quicksum(
                    setup_time[i][t] * y[i, t] + var_time[i][t] * x[i, t]
                    for i in range(P)
                )
                <= capacity[t]
            )

        # MIP start from the trivially feasible plan
        for i in range(P):
            I0[i].Start = t_init[i]
            for t in range(T):
                x[i, t].Start = 0.0
                y[i, t].Start = 0.0
                s[i, t].Start = t_inv[i][t]

        all_vars = (
            [x[i, t] for i in range(P) for t in range(T)]
            + [y[i, t] for i in range(P) for t in range(T)]
            + [s[i, t] for i in range(P) for t in range(T)]
            + [I0[i] for i in range(P)]
        )
        nxt = P * T

        def incumbent_cb(m, where):
            if where == GRB.Callback.MIPSOL and logger is not None:
                try:
                    vals = m.cbGetSolution(all_vars)
                    xv = vals[0:nxt]
                    yv = vals[nxt:2 * nxt]
                    sv = vals[2 * nxt:3 * nxt]
                    iv = vals[3 * nxt:3 * nxt + P]
                    prod = [[xv[i * T + t] for t in range(T)] for i in range(P)]
                    setp = [[yv[i * T + t] for t in range(T)] for i in range(P)]
                    inv = [[sv[i * T + t] for t in range(T)] for i in range(P)]
                    init = [iv[i] for i in range(P)]
                    obj, sol = make_solution(prod, setp, inv, init)
                    logger.log_solution(obj, sol)
                except Exception:
                    pass

        if logger is not None:
            model.optimize(incumbent_cb)
        else:
            model.optimize()

        if model.SolCount > 0:
            obj, sol = make_solution(
                [[x[i, t].X for t in range(T)] for i in range(P)],
                [[y[i, t].X for t in range(T)] for i in range(P)],
                [[s[i, t].X for t in range(T)] for i in range(P)],
                [I0[i].X for i in range(P)],
            )
            if obj < best_obj - 1e-9:
                best_obj, best_sol = obj, sol
    except Exception:
        pass

    if logger is not None:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()