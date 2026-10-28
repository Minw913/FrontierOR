import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(
        description="Multi-item capacitated lot-sizing solver"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON will be written")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds")
    parser.add_argument("--log_path", default=None,
                        help="Optional path for the JSONL convergence log")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # ----------------------------- Load instance -----------------------------
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n_items = int(data["n_items"])
    T = int(data["n_periods"])
    capacity = [float(v) for v in data["capacity"]]
    init_inv = [float(v) for v in data["initial_inventory"]]
    demand = [[float(v) for v in row] for row in data["demands"]]
    setup_cost = [[float(v) for v in row] for row in data["setup_costs"]]
    holding_cost = [[float(v) for v in row] for row in data["holding_costs"]]
    var_cost = [[float(v) for v in row] for row in data["variable_costs"]]
    setup_time = [[float(v) for v in row] for row in data["setup_times"]]
    var_time = [[float(v) for v in row] for row in data["variable_times"]]
    big_M = [[float(v) for v in row] for row in data["big_M"]]

    # ----------------------------- Build model -------------------------------
    model = gp.Model("MICLS")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining = max(1.0, float(args.time_limit) - (time.time() - start_time))
    model.Params.TimeLimit = remaining

    x = model.addVars(n_items, T, lb=0.0, name="x")          # production qty
    y = model.addVars(n_items, T, vtype=GRB.BINARY, name="y")  # setup indicator
    s = model.addVars(n_items, T + 1, lb=0.0, name="s")        # inventory

    # Boundary conditions: given initial inventory, zero terminal inventory.
    for i in range(n_items):
        s[i, 0].lb = float(init_inv[i])
        s[i, 0].ub = float(init_inv[i])
        s[i, T].lb = 0.0
        s[i, T].ub = 0.0

    # Inventory balance: s[i,t] + x[i,t] = demand[i,t] + s[i,t+1]
    model.addConstrs(
        (s[i, t] + x[i, t] == demand[i][t] + s[i, t + 1]
         for i in range(n_items) for t in range(T)),
        name="balance",
    )

    # Production only if set up.
    model.addConstrs(
        (x[i, t] <= big_M[i][t] * y[i, t]
         for i in range(n_items) for t in range(T)),
        name="setup_link",
    )

    # Period capacity on setup + production time.
    model.addConstrs(
        (gp.quicksum(setup_time[i][t] * y[i, t] + var_time[i][t] * x[i, t]
                     for i in range(n_items)) <= capacity[t]
         for t in range(T)),
        name="capacity",
    )

    # Objective: setup + variable production + holding costs.
    # holding_cost[i][t] applies to inventory carried from period t to t+1,
    # i.e. the end-of-period-t inventory s[i, t+1].
    model.setObjective(
        gp.quicksum(setup_cost[i][t] * y[i, t]
                    for i in range(n_items) for t in range(T))
        + gp.quicksum(var_cost[i][t] * x[i, t]
                      for i in range(n_items) for t in range(T))
        + gp.quicksum(holding_cost[i][t] * s[i, t + 1]
                      for i in range(n_items) for t in range(T)),
        GRB.MINIMIZE,
    )

    x_flat = [x[i, t] for i in range(n_items) for t in range(T)]
    y_flat = [y[i, t] for i in range(n_items) for t in range(T)]

    # -------------------- Solution assembly / projection ---------------------
    def assemble(xvals, yvals):
        xsol = []
        ysol = []
        ssol = []
        for i in range(n_items):
            xrow = []
            for t in range(T):
                v = float(xvals[i * T + t])
                if -1e-7 < v < 0.0:
                    v = 0.0
                xrow.append(v)
            xsol.append(xrow)
            ysol.append([1 if yvals[i * T + t] >= 0.5 else 0 for t in range(T)])
        # Recompute inventories exactly from the balance equations so the
        # projected solution is internally consistent.
        for i in range(n_items):
            srow = [float(init_inv[i])]
            for t in range(T):
                val = srow[t] + xsol[i][t] - demand[i][t]
                if -1e-6 < val < 0.0:
                    val = 0.0
                srow.append(val)
            if abs(srow[T]) < 1e-6:
                srow[T] = 0.0
            ssol.append(srow)
        obj_val = 0.0
        for i in range(n_items):
            for t in range(T):
                obj_val += (setup_cost[i][t] * ysol[i][t]
                            + var_cost[i][t] * xsol[i][t]
                            + holding_cost[i][t] * ssol[i][t + 1])
        solution = {
            "objective_value": float(obj_val),
            "x": xsol,
            "s": ssol,
            "y": ysol,
        }
        return obj_val, solution

    # ------------------- Lot-for-lot heuristic MIP start ---------------------
    def lot_for_lot():
        xv = [0.0] * (n_items * T)
        yv = [0.0] * (n_items * T)
        for i in range(n_items):
            inv = float(init_inv[i])
            for t in range(T):
                prod = demand[i][t] - inv
                if prod < 0.0:
                    prod = 0.0
                xv[i * T + t] = prod
                yv[i * T + t] = 1.0 if prod > 1e-9 else 0.0
                inv = inv + prod - demand[i][t]
        return xv, yv

    xv0, yv0 = lot_for_lot()
    for k in range(n_items * T):
        x_flat[k].Start = xv0[k]
        y_flat[k].Start = yv0[k]

    # ------------------------- Incumbent logging -----------------------------
    def incumbent_callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                xv = m.cbGetSolution(x_flat)
                yv = m.cbGetSolution(y_flat)
                obj_val, sol = assemble(xv, yv)
                logger.log_solution(obj_val, sol)
            except Exception:
                pass

    # ------------------------------- Solve -----------------------------------
    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        xv = model.getAttr("X", x_flat)
        yv = model.getAttr("X", y_flat)
        obj_val, solution = assemble(xv, yv)
    else:
        # Fallback (should not normally happen): use the lot-for-lot plan so
        # that a schema-valid solution is always produced.
        xv, yv = lot_for_lot()
        obj_val, solution = assemble(xv, yv)

    if logger:
        logger.log_solution(obj_val, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()