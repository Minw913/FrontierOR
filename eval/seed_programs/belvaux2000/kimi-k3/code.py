import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def clean_number(v, tol=1e-9):
    """Clip tiny numerical noise and round for output."""
    if v is None:
        return 0.0
    if -tol < v < tol:
        return 0.0
    return float(round(v, 6))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-item capacitated lot-sizing with setups, startups, "
                    "and backlogging (MIP via Gurobi)."
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (SolutionLogger(args.log_path, sense="minimize")
              if args.log_path else None)

    t_start = time.time()

    # ----------------------- read instance -----------------------
    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    params = inst["parameters"]
    NI, NK, NT = params["NI"], params["NK"], params["NT"]

    costs = inst["costs"]
    f = costs["setup_cost_f"]
    g = costs["startup_cost_g"]
    h = costs["holding_cost_h"]
    e = costs["backlogging_cost_e"]

    mach = inst["machine_data"]
    C = mach["capacity_C"]
    sigma = mach["startup_time_sigma"]

    d = inst["demand"]

    init = inst["initial_conditions"]
    s0 = init["initial_stock_s0"]
    r0 = init["initial_backlog_r0"]

    I, K, T = range(NI), range(NK), range(NT)
    x_keys = [(i, k, t) for i in I for k in K for t in T]
    s_keys = [(i, t) for i in I for t in T]

    # ----------------------- build model -----------------------
    model = gp.Model("CLSP_backlogging")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addVars(NI, NK, NT, lb=0.0, name="x")          # production qty
    s = model.addVars(NI, NT, lb=0.0, name="s")              # ending inventory
    r = model.addVars(NI, NT, lb=0.0, name="r")              # ending backlog
    y = model.addVars(NI, NK, NT, vtype=GRB.BINARY, name="y")  # setup
    z = model.addVars(NI, NK, NT, vtype=GRB.BINARY, name="z")  # startup
    model.update()

    # Inventory/backlog flow balance:
    # s[t-1] + sum_k x[i,k,t] - r[t-1] = d[i,t] + s[t] - r[t]
    model.addConstrs(
        ((s0[i] if t == 0 else s[i, t - 1])
         + x.sum(i, "*", t)
         - (r0[i] if t == 0 else r[i, t - 1])
         == d[i][t] + s[i, t] - r[i, t]
         for i in I for t in T),
        name="balance")

    # At most one item per machine per period.
    model.addConstrs((y.sum("*", k, t) <= 1 for k in K for t in T),
                     name="one_item")

    # Capacity incl. startup capacity loss; production only if set up.
    model.addConstrs(
        (x[i, k, t] + sigma[k] * z[i, k, t] <= C[k] * y[i, k, t]
         for i in I for k in K for t in T),
        name="capacity")

    # Startup definition: z[t] = 1 iff y[t]=1 and y[t-1]=0 (y[-1] := 0).
    if NT > 0:
        model.addConstrs((z[i, k, 0] == y[i, k, 0] for i in I for k in K),
                         name="startup_t0")
    if NT > 1:
        Tn = range(1, NT)
        model.addConstrs((z[i, k, t] >= y[i, k, t] - y[i, k, t - 1]
                          for i in I for k in K for t in Tn),
                         name="startup_lb")
        model.addConstrs((z[i, k, t] <= y[i, k, t]
                          for i in I for k in K for t in Tn),
                         name="startup_ub1")
        model.addConstrs((z[i, k, t] <= 1 - y[i, k, t - 1]
                          for i in I for k in K for t in Tn),
                         name="startup_ub2")

    # Objective: setup + startup + holding + backlogging costs.
    model.setObjective(
        f * y.sum() + g * z.sum()
        + gp.quicksum(h[i] * s[i, t] for i in I for t in T)
        + gp.quicksum(e[i] * r[i, t] for i in I for t in T),
        GRB.MINIMIZE)

    # ------------------- trivial feasible MIP start -------------------
    # (produce nothing, carry everything through inventory/backlog)
    for key in x_keys:
        x[key].Start = 0.0
        y[key].Start = 0.0
        z[key].Start = 0.0
    for i in I:
        net = float(s0[i]) - float(r0[i])
        for t in T:
            net -= d[i][t]
            s[i, t].Start = net if net > 0.0 else 0.0
            r[i, t].Start = -net if net < 0.0 else 0.0

    # ------------------- solution assembly helpers -------------------
    x_vars = [x[key] for key in x_keys]
    y_vars = [y[key] for key in x_keys]
    z_vars = [z[key] for key in x_keys]
    s_vars = [s[key] for key in s_keys]
    r_vars = [r[key] for key in s_keys]

    def build_solution_dict(xv, yv, zv, sv, rv, obj_val):
        production = {}
        setup = {}
        startup = {}
        for (i, k, t), vx, vy, vz in zip(x_keys, xv, yv, zv):
            key = "{}_{}_{}".format(i, k, t)
            production[key] = clean_number(vx)
            setup[key] = int(round(vy))
            startup[key] = int(round(vz))
        stock = {}
        backlog = {}
        for (i, t), vs, vr in zip(s_keys, sv, rv):
            key = "{}_{}".format(i, t)
            stock[key] = clean_number(vs)
            backlog[key] = clean_number(vr)
        return {
            "objective_value": float(obj_val),
            "production": production,
            "stock": stock,
            "backlog": backlog,
            "setup": setup,
            "startup": startup,
        }

    def trivial_solution():
        production = {"{}_{}_{}".format(i, k, t): 0.0
                      for i in I for k in K for t in T}
        setup = {"{}_{}_{}".format(i, k, t): 0
                 for i in I for k in K for t in T}
        startup = {"{}_{}_{}".format(i, k, t): 0
                   for i in I for k in K for t in T}
        stock = {}
        backlog = {}
        obj_val = 0.0
        for i in I:
            net = float(s0[i]) - float(r0[i])
            for t in T:
                net -= d[i][t]
                sv = net if net > 0.0 else 0.0
                rv = -net if net < 0.0 else 0.0
                stock["{}_{}".format(i, t)] = sv
                backlog["{}_{}".format(i, t)] = rv
                obj_val += h[i] * sv + e[i] * rv
        return obj_val, {
            "objective_value": float(obj_val),
            "production": production,
            "stock": stock,
            "backlog": backlog,
            "setup": setup,
            "startup": startup,
        }

    # ------------------- incumbent logging callback -------------------
    state = {"last_logged": float("inf")}

    def mipsol_callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            except gp.GurobiError:
                return
            if obj < state["last_logged"] - 1e-9:
                try:
                    xv = m.cbGetSolution(x_vars)
                    yv = m.cbGetSolution(y_vars)
                    zv = m.cbGetSolution(z_vars)
                    sv = m.cbGetSolution(s_vars)
                    rv = m.cbGetSolution(r_vars)
                except gp.GurobiError:
                    return
                sol = build_solution_dict(xv, yv, zv, sv, rv, obj)
                state["last_logged"] = obj
                if logger is not None:
                    logger.log_solution(float(obj), sol)

    # ------------------- solve (respecting time limit) -------------------
    remaining = args.time_limit - (time.time() - t_start)
    model.Params.TimeLimit = max(1.0, remaining)

    model.optimize(mipsol_callback)

    # ------------------- extract best solution -------------------
    if model.SolCount > 0:
        xv = [v.X for v in x_vars]
        yv = [v.X for v in y_vars]
        zv = [v.X for v in z_vars]
        sv = [v.X for v in s_vars]
        rv = [v.X for v in r_vars]
        obj_val = float(model.ObjVal)
        solution = build_solution_dict(xv, yv, zv, sv, rv, obj_val)
    else:
        # Fallback: trivial feasible all-backlog plan.
        obj_val, solution = trivial_solution()

    if logger is not None and obj_val < state["last_logged"] - 1e-9:
        logger.log_solution(obj_val, solution)
        state["last_logged"] = obj_val

    # ------------------- write output -------------------
    with open(args.solution_path, "w") as fh:
        json.dump(solution, fh)


if __name__ == "__main__":
    main()