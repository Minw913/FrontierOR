import argparse
import json

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
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

    P = data["dimensions"]["num_products"]
    T = data["dimensions"]["num_periods"]
    prm = data["parameters"]
    d = prm["demand"]
    hc = prm["holding_cost"]
    sc = prm["setup_cost"]
    vc = prm["variable_production_cost"]
    vt = prm["variable_production_time"]
    st = prm["setup_time"]
    cap = prm["capacity"]
    ic = prm["initial_inventory_cost"]

    # cumulative remaining demand
    cumd = [[0.0] * T for _ in range(P)]
    for i in range(P):
        run = 0.0
        for t in range(T - 1, -1, -1):
            run += d[i][t]
            cumd[i][t] = run

    m = gp.Model("clsp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, args.time_limit - 5)

    x = m.addVars(P, T, lb=0.0, name="x")
    y = m.addVars(P, T, vtype=GRB.BINARY, name="y")
    s = m.addVars(P, T, lb=0.0, name="s")
    I0 = m.addVars(P, lb=0.0, name="I0")

    # flow balance
    for i in range(P):
        m.addConstr(I0[i] + x[i, 0] == d[i][0] + s[i, 0])
        for t in range(1, T):
            m.addConstr(s[i, t - 1] + x[i, t] == d[i][t] + s[i, t])

    # setup forcing with big-M
    for i in range(P):
        for t in range(T):
            cap_bound = (cap[t] - st[i][t]) / vt[i][t] if vt[i][t] > 0 else cumd[i][t]
            M = max(0.0, min(cap_bound, cumd[i][t]))
            m.addConstr(x[i, t] <= M * y[i, t])

    # capacity
    for t in range(T):
        m.addConstr(
            gp.quicksum(st[i][t] * y[i, t] + vt[i][t] * x[i, t] for i in range(P)) <= cap[t]
        )

    obj = (
        gp.quicksum(ic[i] * I0[i] for i in range(P))
        + gp.quicksum(sc[i][t] * y[i, t] + vc[i][t] * x[i, t] + hc[i][t] * s[i, t]
                      for i in range(P) for t in range(T))
    )
    m.setObjective(obj, GRB.MINIMIZE)

    def extract(getval):
        prod = [[max(0.0, getval(x[i, t])) for t in range(T)] for i in range(P)]
        setup = [[1 if getval(y[i, t]) > 0.5 else 0 for t in range(T)] for i in range(P)]
        inv = [[max(0.0, getval(s[i, t])) for t in range(T)] for i in range(P)]
        init = [max(0.0, getval(I0[i])) for i in range(P)]
        objv = sum(ic[i] * init[i] for i in range(P))
        for i in range(P):
            for t in range(T):
                objv += sc[i][t] * setup[i][t] + vc[i][t] * prod[i][t] + hc[i][t] * inv[i][t]
        sol = {
            "objective_value": objv,
            "variables": {
                "production": prod,
                "setup": setup,
                "inventory": inv,
                "initial_inventory": init,
            },
        }
        return objv, sol

    best = {"obj": float("inf"), "sol": None}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model.getVars())
            vmap = {v: vals[k] for k, v in enumerate(model.getVars())}
            objv, sol = extract(lambda var: vmap[var])
            if objv < best["obj"] - 1e-9:
                best["obj"] = objv
                best["sol"] = sol
                if logger:
                    logger.log_solution(objv, sol)

    m.optimize(callback)

    if m.SolCount > 0:
        objv, sol = extract(lambda var: var.X)
        if objv < best["obj"] - 1e-9:
            best["obj"] = objv
            best["sol"] = sol
            if logger:
                logger.log_solution(objv, sol)

    if best["sol"] is None:
        # fallback: satisfy everything from initial inventory (always feasible)
        init = [float(sum(d[i])) for i in range(P)]
        prod = [[0.0] * T for _ in range(P)]
        setup = [[0] * T for _ in range(P)]
        inv = [[0.0] * T for _ in range(P)]
        for i in range(P):
            rem = init[i]
            for t in range(T):
                rem -= d[i][t]
                inv[i][t] = rem
        objv = sum(ic[i] * init[i] for i in range(P))
        for i in range(P):
            for t in range(T):
                objv += hc[i][t] * inv[i][t]
        best["sol"] = {
            "objective_value": objv,
            "variables": {
                "production": prod,
                "setup": setup,
                "inventory": inv,
                "initial_inventory": init,
            },
        }
        best["obj"] = objv
        if logger:
            logger.log_solution(objv, best["sol"])

    with open(args.solution_path, "w") as f:
        json.dump(best["sol"], f)


if __name__ == "__main__":
    main()