#!/usr/bin/env python3
"""
Stochastic unit commitment on a scenario tree (single thermal generator).

Solves the problem as a mixed-integer linear program with Gurobi:
  - x[i] : generation at node i (MW)
  - y[i] : on/off status at node i (binary)
  - u[i] : startup indicator at non-root node i (binary)
  - shutdown at node i is the linear expression y[parent] - y[i] + u[i]

Reads the instance from --instance_path, writes the solution to
--solution_path, enforces --time_limit, and logs incumbents via
solution_logger.
"""

import argparse
import json

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    g = inst["generator"]
    t = inst["scenario_tree"]

    C_lo = float(g["C_lower_MW"])
    C_up = float(g["C_upper_MW"])
    V_up = float(g["V_plus_MW_per_h"])
    V_dn = float(g["V_minus_MW_per_h"])
    c_start = float(g["U_bar_startup_cost_dollars"])
    c_shut = float(g["U_lower_shutdown_cost_dollars"])
    fuel_a = float(g["fuel_a_dollars_per_h"])       # fixed cost per period when on
    fuel_b = float(g["fuel_b_dollars_per_MWh"])     # variable cost per MWh

    L_up = max(1, int(t["L_min_up_time"]))
    L_dn = max(1, int(t["ell_min_down_time"]))
    y0 = int(t["initial_generator_status_y0"])
    x0 = float(t["initial_generation_x0_MW"])

    nodes = t["nodes"]
    ids = []
    parent = {}
    prob = {}
    price = {}
    root = None
    for n in nodes:
        i = n["id"]
        ids.append(i)
        parent[i] = n["parent_id"]
        prob[i] = float(n["probability"])
        price[i] = float(n["electricity_price_dollars_per_MWh"])
        if n["parent_id"] is None:
            root = i
    nonroot = [i for i in ids if parent[i] is not None]

    def fallback_solution():
        # All-off schedule is always feasible in this formulation and has
        # objective value 0 (no startup/shutdown is charged at the root).
        return {
            "objective_value": 0.0,
            "x": {str(i): 0.0 for i in ids},
            "y": {str(i): 0 for i in ids},
            "u": {str(i): 0 for i in nonroot},
        }

    def compute_objective(xd, yd, ud):
        val = 0.0
        for i in ids:
            val += prob[i] * ((price[i] - fuel_b) * xd[i] - fuel_a * yd[i])
        for i in nonroot:
            pi = parent[i]
            shut = yd[pi] - yd[i] + ud[i]
            val -= prob[i] * (c_start * ud[i] + c_shut * shut)
        return val

    best_sol = None

    try:
        model = gp.Model("scenario_tree_unit_commitment")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = float(max(1, args.time_limit))

        # ---------------- Variables ----------------
        x = model.addVars(ids, lb=0.0, ub=C_up, name="x")
        y = model.addVars(ids, vtype=GRB.BINARY, name="y")
        u = model.addVars(nonroot, vtype=GRB.BINARY, name="u")

        # ---------------- Objective ----------------
        obj = gp.LinExpr()
        for i in ids:
            obj += prob[i] * ((price[i] - fuel_b) * x[i] - fuel_a * y[i])
        for i in nonroot:
            pi = parent[i]
            # startup cost + shutdown cost (shutdown = y_p - y_i + u_i)
            obj -= prob[i] * (c_start * u[i] + c_shut * (y[pi] - y[i] + u[i]))
        model.setObjective(obj, GRB.MAXIMIZE)

        # ---------------- Generation limits ----------------
        for i in ids:
            model.addConstr(x[i] <= C_up * y[i], name=f"gen_up_{i}")
            model.addConstr(x[i] >= C_lo * y[i], name=f"gen_lo_{i}")

        # ---------------- Startup logic ----------------
        # u_i = 1 iff y_i = 1 and y_parent = 0
        for i in nonroot:
            pi = parent[i]
            model.addConstr(u[i] >= y[i] - y[pi], name=f"start_def_{i}")
            model.addConstr(u[i] <= y[i], name=f"start_on_{i}")
            model.addConstr(u[i] <= 1 - y[pi], name=f"start_off_{i}")

        # ---------------- Minimum up / down times ----------------
        # For node i, consider the window of non-root nodes on the path from
        # i up to its ancestor at distance (L-1).  A startup (shutdown) in
        # that window forces node i to be on (off).
        def anc_window(i, length):
            w = []
            cur = i
            while cur is not None and len(w) < length:
                if parent[cur] is not None:
                    w.append(cur)
                cur = parent[cur]
            return w

        for i in ids:
            wu = anc_window(i, L_up)
            if wu:
                model.addConstr(y[i] >= gp.quicksum(u[m] for m in wu),
                                name=f"minup_{i}")
            wd = anc_window(i, L_dn)
            if wd:
                model.addConstr(
                    1 - y[i] >= gp.quicksum(y[parent[m]] - y[m] + u[m] for m in wd),
                    name=f"mindn_{i}",
                )

        # ---------------- Ramping ----------------
        # Ramp up: free jump when previously off (u_i = 1), else <= V_up.
        # Ramp down: free drop when shutting down (y_i = 0), else <= V_dn.
        for i in nonroot:
            pi = parent[i]
            model.addConstr(x[i] - x[pi] <= V_up * y[pi] + C_up * u[i],
                            name=f"rampup_{i}")
            model.addConstr(x[pi] - x[i] <= V_dn * y[i] + C_up * (1 - y[i]),
                            name=f"rampdn_{i}")
        # Ramp from the pre-period initial state into the root node.
        if root is not None and y0 == 1:
            model.addConstr(x[root] - x0 <= V_up, name="rampup_root")
            model.addConstr(x0 - x[root] <= V_dn * y[root] + C_up * (1 - y[root]),
                            name="rampdn_root")

        # ---------------- Simple MIP start ----------------
        for i in ids:
            on = 1.0 if price[i] > fuel_b else 0.0
            y[i].Start = on
            x[i].Start = C_up * on
        for i in nonroot:
            pi = parent[i]
            yi = 1.0 if price[i] > fuel_b else 0.0
            yp = 1.0 if price[pi] > fuel_b else 0.0
            u[i].Start = max(0.0, yi - yp)

        x_list = [x[i] for i in ids]
        y_list = [y[i] for i in ids]
        u_list = [u[i] for i in nonroot]

        # ---------------- Incumbent logging callback ----------------
        def callback(m, where):
            if where == GRB.Callback.MIPSOL and logger is not None:
                try:
                    objval = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
                    xv = m.cbGetSolution(x_list)
                    yv = m.cbGetSolution(y_list)
                    uv = m.cbGetSolution(u_list)
                    sol = {
                        "objective_value": objval,
                        "x": {str(i): max(0.0, float(v)) for i, v in zip(ids, xv)},
                        "y": {str(i): int(round(v)) for i, v in zip(ids, yv)},
                        "u": {str(i): int(round(v)) for i, v in zip(nonroot, uv)},
                    }
                    logger.log_solution(objval, sol)
                except Exception:
                    pass

        model.optimize(callback)

        if model.SolCount > 0:
            xd = {i: min(C_up, max(0.0, float(x[i].X))) for i in ids}
            yd = {i: int(round(y[i].X)) for i in ids}
            ud = {i: int(round(u[i].X)) for i in nonroot}
            objval = float(compute_objective(xd, yd, ud))
            best_sol = {
                "objective_value": objval,
                "x": {str(i): xd[i] for i in ids},
                "y": {str(i): yd[i] for i in ids},
                "u": {str(i): ud[i] for i in nonroot},
            }
    except Exception:
        best_sol = None

    if best_sol is None:
        best_sol = fallback_solution()

    # Record the final incumbent.
    if logger:
        try:
            logger.log_solution(float(best_sol["objective_value"]), best_sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()