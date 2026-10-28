import argparse
import json
import time

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

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    NI = data["parameters"]["NI"]
    NK = data["parameters"]["NK"]
    NT = data["parameters"]["NT"]

    f_cost = data["costs"]["setup_cost_f"]
    g_cost = data["costs"]["startup_cost_g"]
    h = data["costs"]["holding_cost_h"]
    e = data["costs"]["backlogging_cost_e"]

    C = data["machine_data"]["capacity_C"]
    sigma = data["machine_data"]["startup_time_sigma"]

    demand = data["demand"]
    s0 = data["initial_conditions"]["initial_stock_s0"]
    r0 = data["initial_conditions"]["initial_backlog_r0"]

    m = gp.Model("clsp_startup")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    elapsed = time.time() - start_time
    remaining = max(5.0, args.time_limit - elapsed - 5.0)
    m.Params.TimeLimit = remaining

    # Variables
    x = m.addVars(NI, NK, NT, lb=0.0, name="x")          # production
    s = m.addVars(NI, NT, lb=0.0, name="s")              # stock at end of period
    r = m.addVars(NI, NT, lb=0.0, name="r")              # backlog at end of period
    y = m.addVars(NI, NK, NT, vtype=GRB.BINARY, name="y")  # setup
    z = m.addVars(NI, NK, NT, vtype=GRB.BINARY, name="z")  # startup

    # Balance constraints
    for i in range(NI):
        for t in range(NT):
            prev_s = s0[i] if t == 0 else s[i, t - 1]
            prev_r = r0[i] if t == 0 else r[i, t - 1]
            m.addConstr(
                prev_s + gp.quicksum(x[i, k, t] for k in range(NK)) - prev_r
                == demand[i][t] + s[i, t] - r[i, t]
            )

    # Machine can be set up for at most one item per period
    for k in range(NK):
        for t in range(NT):
            m.addConstr(gp.quicksum(y[i, k, t] for i in range(NI)) <= 1)

    # Capacity linking with startup loss (at most one item per machine so per-item ok)
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                m.addConstr(x[i, k, t] + sigma[k] * z[i, k, t] <= C[k] * y[i, k, t])

    # Startup logic
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                if t == 0:
                    # No setup before horizon -> startup = setup in period 0
                    m.addConstr(z[i, k, t] == y[i, k, t])
                else:
                    m.addConstr(z[i, k, t] >= y[i, k, t] - y[i, k, t - 1])
                    m.addConstr(z[i, k, t] <= y[i, k, t])
                    m.addConstr(z[i, k, t] <= 1 - y[i, k, t - 1])

    # Objective
    obj = (
        f_cost * gp.quicksum(y[i, k, t] for i in range(NI) for k in range(NK) for t in range(NT))
        + g_cost * gp.quicksum(z[i, k, t] for i in range(NI) for k in range(NK) for t in range(NT))
        + gp.quicksum(h[i] * s[i, t] for i in range(NI) for t in range(NT))
        + gp.quicksum(e[i] * r[i, t] for i in range(NI) for t in range(NT))
    )
    m.setObjective(obj, GRB.MINIMIZE)

    def build_solution(xv, sv, rv, yv, zv, obj_val):
        prod = {}
        setup = {}
        startup = {}
        for i in range(NI):
            for k in range(NK):
                for t in range(NT):
                    key = f"{i}_{k}_{t}"
                    prod[key] = max(0.0, float(xv[i, k, t]))
                    setup[key] = int(round(yv[i, k, t]))
                    startup[key] = int(round(zv[i, k, t]))
        stock = {}
        backlog = {}
        for i in range(NI):
            for t in range(NT):
                key = f"{i}_{t}"
                stock[key] = max(0.0, float(sv[i, t]))
                backlog[key] = max(0.0, float(rv[i, t]))
        return {
            "objective_value": float(obj_val),
            "production": prod,
            "stock": stock,
            "backlog": backlog,
            "setup": setup,
            "startup": startup,
        }

    best = {"obj": float("inf"), "sol": None}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best["obj"] - 1e-9:
                xv = model.cbGetSolution(x)
                sv = model.cbGetSolution(s)
                rv = model.cbGetSolution(r)
                yv = model.cbGetSolution(y)
                zv = model.cbGetSolution(z)
                sol = build_solution(xv, sv, rv, yv, zv, obj_val)
                best["obj"] = obj_val
                best["sol"] = sol
                if logger:
                    logger.log_solution(obj_val, sol)

    m.optimize(callback)

    final_sol = None
    if m.SolCount > 0:
        obj_val = m.ObjVal
        xv = m.getAttr("X", x)
        sv = m.getAttr("X", s)
        rv = m.getAttr("X", r)
        yv = m.getAttr("X", y)
        zv = m.getAttr("X", z)
        final_sol = build_solution(xv, sv, rv, yv, zv, obj_val)
        if obj_val < best["obj"] - 1e-9:
            if logger:
                logger.log_solution(obj_val, final_sol)
    elif best["sol"] is not None:
        final_sol = best["sol"]

    if final_sol is None:
        # Fallback: no production, all demand backlogged (always feasible)
        prod = {f"{i}_{k}_{t}": 0.0 for i in range(NI) for k in range(NK) for t in range(NT)}
        setup = {f"{i}_{k}_{t}": 0 for i in range(NI) for k in range(NK) for t in range(NT)}
        startup = dict(setup)
        stock = {}
        backlog = {}
        total = 0.0
        for i in range(NI):
            cur_s = s0[i]
            cur_r = r0[i]
            for t in range(NT):
                net = cur_s - cur_r - demand[i][t]
                cur_s = max(0.0, net)
                cur_r = max(0.0, -net)
                stock[f"{i}_{t}"] = cur_s
                backlog[f"{i}_{t}"] = cur_r
                total += h[i] * cur_s + e[i] * cur_r
        final_sol = {
            "objective_value": total,
            "production": prod,
            "stock": stock,
            "backlog": backlog,
            "setup": setup,
            "startup": startup,
        }
        if logger:
            logger.log_solution(total, final_sol)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()