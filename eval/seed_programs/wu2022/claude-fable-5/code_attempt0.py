import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    dims = data["dimensions"]
    nI = dims["num_items"]
    nT = dims["num_periods"]
    nJ = dims["num_plants"]
    nC = dims["num_customers"]

    capab = data["plant_capabilities"]          # per item, list of plants
    demand = data["demand"]                     # [i][c][t]
    setup_cost = data["setup_cost"]             # [i][j][t]
    prod_cost = data["production_cost"]         # [i][j][t]
    hold_cost = data["holding_cost"]            # [i][j][t]
    setup_time = data["setup_time"]             # [i][j][t]
    prod_time = data["production_time"]         # [i][j][t]
    capacity = data["capacity"]                 # [j][t]
    open_cost = data["plant_opening_cost"]      # [j]
    trans_cost = data["transportation_cost"]    # [i][c][j][t]

    # Aggregate demand per item per period
    d = [[0] * nT for _ in range(nI)]
    for i in range(nI):
        for c in range(nC):
            row = demand[i][c]
            for t in range(nT):
                d[i][t] += row[t]

    # Remaining (tail) demand for each item from period t to end
    rem = [[0] * (nT + 1) for _ in range(nI)]
    for i in range(nI):
        for t in range(nT - 1, -1, -1):
            rem[i][t] = rem[i][t + 1] + d[i][t]

    # Transportation cost per (item, plant) if item assigned to plant
    TC = [[0.0] * nJ for _ in range(nI)]
    for i in range(nI):
        for c in range(nC):
            dem_row = demand[i][c]
            tc_row = trans_cost[i][c]
            for j in range(nJ):
                s = 0.0
                tcj = tc_row[j]
                for t in range(nT):
                    if dem_row[t]:
                        s += dem_row[t] * tcj[t]
                TC[i][j] += s

    m = gp.Model("plfp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    # Variables
    Z = m.addVars(nJ, vtype=GRB.BINARY, name="Z")
    # (i, j) pairs restricted to capable plants
    IJ = [(i, j) for i in range(nI) for j in capab[i]]
    U = m.addVars(IJ, vtype=GRB.BINARY, name="U")

    Y = {}
    X = {}
    Ivar = {}
    for (i, j) in IJ:
        for t in range(nT):
            # Big-M for production quantity
            ub = rem[i][t]
            pt = prod_time[i][j][t]
            if pt > 0:
                cap_b = (capacity[j][t] - setup_time[i][j][t]) / pt
                if cap_b < ub:
                    ub = max(0, int(cap_b))
            if ub < 0:
                ub = 0
            Y[i, j, t] = m.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{j}_{t}")
            X[i, j, t] = m.addVar(lb=0.0, ub=ub, name=f"X_{i}_{j}_{t}")
            Ivar[i, j, t] = m.addVar(lb=0.0, ub=rem[i][t + 1] if rem[i][t + 1] > 0 else 0.0,
                                     name=f"I_{i}_{j}_{t}")
            if ub <= 0:
                m.addConstr(Y[i, j, t] == 0)

    # Objective
    obj = gp.LinExpr()
    for j in range(nJ):
        obj += open_cost[j] * Z[j]
    for (i, j) in IJ:
        obj += TC[i][j] * U[i, j]
        for t in range(nT):
            obj += setup_cost[i][j][t] * Y[i, j, t]
            obj += prod_cost[i][j][t] * X[i, j, t]
            obj += hold_cost[i][j][t] * Ivar[i, j, t]
    m.setObjective(obj, GRB.MINIMIZE)

    # Assignment constraints
    for i in range(nI):
        m.addConstr(gp.quicksum(U[i, j] for j in capab[i]) == 1)
    for (i, j) in IJ:
        m.addConstr(U[i, j] <= Z[j])

    # Setup / production linking, inventory balance
    for (i, j) in IJ:
        for t in range(nT):
            ub = X[i, j, t].UB
            if ub > 0:
                m.addConstr(X[i, j, t] <= ub * Y[i, j, t])
            m.addConstr(Y[i, j, t] <= U[i, j])
            prev = Ivar[i, j, t - 1] if t > 0 else 0.0
            m.addConstr(Ivar[i, j, t] == prev + X[i, j, t] - d[i][t] * U[i, j])

    # Capacity constraints
    items_at_plant = [[] for _ in range(nJ)]
    for (i, j) in IJ:
        items_at_plant[j].append(i)
    for j in range(nJ):
        for t in range(nT):
            expr = gp.LinExpr()
            for i in items_at_plant[j]:
                expr += setup_time[i][j][t] * Y[i, j, t]
                expr += prod_time[i][j][t] * X[i, j, t]
            m.addConstr(expr <= capacity[j][t] * Z[j])

    # Time limit
    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    m.Params.TimeLimit = remaining

    best = {"obj": float("inf"), "sol": None}

    def build_solution(zvals, uvals, yvals, objv):
        sol = {"objective_value": float(objv)}
        sol["Z"] = {str(j): int(round(zvals[j])) for j in range(nJ)}
        sol["U"] = {f"{i}_{j}": int(round(uvals[i, j])) for (i, j) in IJ}
        sol["Y"] = {f"{i}_{j}_{t}": int(round(yvals[i, j, t]))
                    for (i, j) in IJ for t in range(nT)}
        return sol

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objv < best["obj"] - 1e-9:
                zvals = {j: model.cbGetSolution(Z[j]) for j in range(nJ)}
                uvals = {(i, j): model.cbGetSolution(U[i, j]) for (i, j) in IJ}
                yvals = {(i, j, t): model.cbGetSolution(Y[i, j, t])
                         for (i, j) in IJ for t in range(nT)}
                sol = build_solution(zvals, uvals, yvals, objv)
                best["obj"] = objv
                best["sol"] = sol
                if logger:
                    try:
                        logger.log_solution(float(objv), sol)
                    except Exception:
                        pass

    m.optimize(callback)

    # Final solution
    final_sol = None
    if m.SolCount > 0:
        objv = m.ObjVal
        zvals = {j: Z[j].X for j in range(nJ)}
        uvals = {(i, j): U[i, j].X for (i, j) in IJ}
        yvals = {(i, j, t): Y[i, j, t].X for (i, j) in IJ for t in range(nT)}
        final_sol = build_solution(zvals, uvals, yvals, objv)
        if objv < best["obj"] - 1e-9 and logger:
            try:
                logger.log_solution(float(objv), final_sol)
            except Exception:
                pass
    elif best["sol"] is not None:
        final_sol = best["sol"]
    else:
        # No feasible solution found; write an empty-shell fallback
        final_sol = {
            "objective_value": float("inf"),
            "Z": {str(j): 0 for j in range(nJ)},
            "U": {f"{i}_{j}": 0 for (i, j) in IJ},
            "Y": {f"{i}_{j}_{t}": 0 for (i, j) in IJ for t in range(nT)},
        }

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()