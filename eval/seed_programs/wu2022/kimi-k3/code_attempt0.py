import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start = time.time()
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    dims = data["dimensions"]
    I = int(dims["num_items"])
    T = int(dims["num_periods"])
    P = int(dims["num_plants"])
    C = int(dims["num_customers"])

    capable = [[int(p) for p in lst] for lst in data["plant_capabilities"]]
    plant_items = [[] for _ in range(P)]
    for i in range(I):
        for p in capable[i]:
            plant_items[p].append(i)

    demand = data["demand"]
    setup_cost = data["setup_cost"]
    production_cost = data["production_cost"]
    holding_cost = data["holding_cost"]
    setup_time = data["setup_time"]
    production_time = data["production_time"]
    capacity = data["capacity"]
    tcost = data["transportation_cost"]

    # Total demand per item and period (aggregated across customers).
    D = [[0] * T for _ in range(I)]
    for i in range(I):
        di = demand[i]
        for c in range(C):
            row = di[c]
            for t in range(T):
                D[i][t] += row[t]

    # Big-M: remaining demand from period t onward (upper bound on production in t).
    M = [[0] * T for _ in range(I)]
    for i in range(I):
        running = 0
        for t in range(T - 1, -1, -1):
            running += D[i][t]
            M[i][t] = running

    # Plant opening cost (charged once if the plant is used at all).
    if data.get("plant_opening_cost") is not None:
        opening = [float(v) for v in data["plant_opening_cost"]]
    else:
        pop = data["plant_opening_cost_per_period"]
        opening = [float(sum(pop[p])) for p in range(P)]

    # Transportation cost implied by assigning item i to plant p
    # (single sourcing: the assigned plant serves all demand of that item).
    assign_cost = [[0.0] * P for _ in range(I)]
    for i in range(I):
        di = demand[i]
        tci = tcost[i]
        for p in capable[i]:
            val = 0.0
            for c in range(C):
                drow = di[c]
                trow = tci[c][p]
                for t in range(T):
                    if drow[t]:
                        val += trow[t] * drow[t]
            assign_cost[i][p] = val

    # ---------------- MIP model ----------------
    m = gp.Model("single_sourcing_lotsizing")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    z = m.addVars(P, vtype=GRB.BINARY, name="Z")
    upairs = [(i, p) for i in range(I) for p in capable[i]]
    u = m.addVars(upairs, vtype=GRB.BINARY, name="U")
    ytrip = [(i, p, t) for (i, p) in upairs for t in range(T)]
    y = m.addVars(ytrip, vtype=GRB.BINARY, name="Y")
    x = m.addVars(ytrip, lb=0.0, name="X")       # production quantity
    inv = m.addVars(ytrip, lb=0.0, name="Inv")   # end-of-period inventory

    obj = gp.LinExpr()
    for p in range(P):
        obj += opening[p] * z[p]
    for (i, p) in upairs:
        obj += assign_cost[i][p] * u[i, p]
    for (i, p, t) in ytrip:
        obj += setup_cost[i][p][t] * y[i, p, t]
        obj += production_cost[i][p][t] * x[i, p, t]
        obj += holding_cost[i][p][t] * inv[i, p, t]
    m.setObjective(obj, GRB.MINIMIZE)

    # Each item assigned to exactly one capable plant.
    for i in range(I):
        m.addConstr(gp.quicksum(u[i, p] for p in capable[i]) == 1.0)

    # Assignment only to open plants.
    for (i, p) in upairs:
        m.addConstr(u[i, p] <= z[p])

    for (i, p, t) in ytrip:
        m.addConstr(y[i, p, t] <= u[i, p])
        mbig = M[i][t]
        if mbig > 0:
            m.addConstr(x[i, p, t] <= mbig * y[i, p, t])
        else:
            m.addConstr(x[i, p, t] <= 0.0)
        prev = inv[i, p, t - 1] if t > 0 else 0.0
        # Inventory balance: for the assigned plant (u=1), cumulative production
        # must cover cumulative demand (no backlogging); elsewhere all are zero.
        m.addConstr(prev + x[i, p, t] - inv[i, p, t] == D[i][t] * u[i, p])

    for p in range(P):
        items = plant_items[p]
        if not items:
            continue
        for t in range(T):
            expr = gp.quicksum(
                setup_time[i][p][t] * y[i, p, t]
                + production_time[i][p][t] * x[i, p, t]
                for i in items
            )
            m.addConstr(expr <= capacity[p][t] * z[p])

    # ---------------- solution extraction / logging ----------------
    z_vars = [z[p] for p in range(P)]
    u_vars = [u[k] for k in upairs]
    y_vars = [y[k] for k in ytrip]

    def assemble(objval, zvals, uvals, yvals):
        Z = {str(p): int(round(zvals[p])) for p in range(P)}
        U = {}
        for idx, (i, p) in enumerate(upairs):
            U[f"{i}_{p}"] = int(round(uvals[idx]))
        Y = {}
        for idx, (i, p, t) in enumerate(ytrip):
            Y[f"{i}_{p}_{t}"] = int(round(yvals[idx]))
        return {"objective_value": float(objval), "Z": Z, "U": U, "Y": Y}

    state = {"last": None}

    def try_log(objval, sol):
        if logger is None:
            return
        if state["last"] is not None and objval >= state["last"] - 1e-9:
            return
        state["last"] = objval
        logger.log_solution(objval, sol)

    def log_cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                zv = model.cbGetSolution(z_vars)
                uv = model.cbGetSolution(u_vars)
                yv = model.cbGetSolution(y_vars)
            except gp.GurobiError:
                return
            try_log(objval, assemble(objval, zv, uv, yv))

    remaining = args.time_limit - (time.time() - start)
    m.Params.TimeLimit = max(1.0, remaining)

    try:
        m.optimize(log_cb)
    except gp.GurobiError:
        pass

    # ---------------- final output ----------------
    if m.SolCount > 0:
        objval = float(m.ObjVal)
        zvals = [v.X for v in z_vars]
        uvals = [v.X for v in u_vars]
        yvals = [v.X for v in y_vars]
        sol = assemble(objval, zvals, uvals, yvals)
    else:
        # Emergency fallback (used only if the solver produced no incumbent):
        # open all plants, assign each item to its cheapest capable plant,
        # and set a setup in every period with positive demand.
        Z = {str(p): 1 for p in range(P)}
        U = {}
        Y = {}
        objval = sum(opening)
        for i in range(I):
            best_p = None
            best_c = None
            for p in capable[i]:
                cst = assign_cost[i][p]
                for t in range(T):
                    cst += production_cost[i][p][t] * D[i][t]
                    if D[i][t] > 0:
                        cst += setup_cost[i][p][t]
                if best_c is None or cst < best_c:
                    best_c = cst
                    best_p = p
            objval += best_c if best_c is not None else 0.0
            for p in capable[i]:
                U[f"{i}_{p}"] = 1 if p == best_p else 0
                for t in range(T):
                    Y[f"{i}_{p}_{t}"] = 1 if (p == best_p and D[i][t] > 0) else 0
        sol = {"objective_value": float(objval), "Z": Z, "U": U, "Y": Y}

    try_log(sol["objective_value"], sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()