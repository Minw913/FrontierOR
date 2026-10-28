import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def build_solution_dict(node_ids, xv, yv, uv, obj):
    sol = {
        "objective_value": float(obj),
        "x": {},
        "y": {},
        "u": {},
        "sparse_defaults": {"x": 0, "y": 0, "u": 0},
    }
    for nid in node_ids:
        if abs(xv[nid]) > 1e-9:
            sol["x"][str(nid)] = float(xv[nid])
        if yv[nid] > 0:
            sol["y"][str(nid)] = int(yv[nid])
        if uv.get(nid, 0) > 0:
            sol["u"][str(nid)] = int(uv[nid])
    return sol


def compute_objective(nodes_by_id, node_ids, xv, yv, uv, y0, a, b, SU, SD):
    total = 0.0
    for nid in node_ids:
        n = nodes_by_id[nid]
        prob = n["probability"]
        price = n["electricity_price_dollars_per_MWh"]
        pid = n["parent_id"]
        yp = y0 if pid is None else yv[pid]
        s = yp - yv[nid] + uv.get(nid, 0)
        rev = price * xv[nid]
        cost = a * yv[nid] + b * xv[nid] + SU * uv.get(nid, 0) + SD * s
        total += prob * (rev - cost)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    gen = data["generator"]
    tree = data["scenario_tree"]
    nodes = tree["nodes"]
    nodes_by_id = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]

    Cl = float(gen["C_lower_MW"])
    Cu = float(gen["C_upper_MW"])
    Vp = float(gen["V_plus_MW_per_h"])
    Vm = float(gen["V_minus_MW_per_h"])
    SU = float(gen["U_bar_startup_cost_dollars"])
    SD = float(gen["U_lower_shutdown_cost_dollars"])
    a = float(gen.get("fuel_a_dollars_per_h", 0.0) or 0.0)
    b = float(gen.get("fuel_b_dollars_per_MWh", 0.0) or 0.0)

    L = int(tree["L_min_up_time"])
    ell = int(tree["ell_min_down_time"])
    y0 = int(tree["initial_generator_status_y0"])
    x0 = float(tree["initial_generation_x0_MW"])

    m = gp.Model("stochastic_uc")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    # Decision variables
    x = {}
    y = {}
    u = {}
    for nid in node_ids:
        x[nid] = m.addVar(lb=0.0, ub=Cu, vtype=GRB.CONTINUOUS, name=f"x_{nid}")
        y[nid] = m.addVar(vtype=GRB.BINARY, name=f"y_{nid}")
        u[nid] = m.addVar(vtype=GRB.BINARY, name=f"u_{nid}")

    # Shutdown expressions
    s_expr = {}
    for nid in node_ids:
        n = nodes_by_id[nid]
        pid = n["parent_id"]
        yp = y0 if pid is None else y[pid]
        s_expr[nid] = yp - y[nid] + u[nid]

    for nid in node_ids:
        n = nodes_by_id[nid]
        pid = n["parent_id"]

        # Generation bounds
        m.addConstr(x[nid] <= Cu * y[nid])
        m.addConstr(x[nid] >= Cl * y[nid])

        if pid is None:
            yp_const = y0
            xp_expr = x0
            # Startup logic (root, vs initial status)
            m.addConstr(u[nid] >= y[nid] - yp_const)
            m.addConstr(u[nid] <= y[nid])
            m.addConstr(u[nid] <= 1 - yp_const)
            # Ramp up
            m.addConstr(x[nid] - xp_expr <= Vp + (Cu - Vp) * (1 - yp_const))
            # Ramp down
            m.addConstr(xp_expr - x[nid] <= Vm + (Cu - Vm) * (1 - y[nid]))
        else:
            m.addConstr(u[nid] >= y[nid] - y[pid])
            m.addConstr(u[nid] <= y[nid])
            m.addConstr(u[nid] <= 1 - y[pid])
            m.addConstr(x[nid] - x[pid] <= Vp * y[pid] + Cu * (1 - y[pid]))
            m.addConstr(x[pid] - x[nid] <= Vm * y[nid] + Cu * (1 - y[nid]))

        # Minimum up time: y_n >= sum of startups on path within L-1 periods
        if L > 1:
            terms = []
            cur = n
            steps = 0
            while cur is not None and steps <= L - 1:
                terms.append(u[cur["id"]])
                p = cur["parent_id"]
                cur = nodes_by_id[p] if p is not None else None
                steps += 1
            if len(terms) > 1:
                m.addConstr(gp.quicksum(terms) <= y[nid])

        # Minimum down time: 1 - y_n >= sum of shutdowns on path within ell-1 periods
        if ell > 1:
            terms = []
            cur = n
            steps = 0
            while cur is not None and steps <= ell - 1:
                terms.append(s_expr[cur["id"]])
                p = cur["parent_id"]
                cur = nodes_by_id[p] if p is not None else None
                steps += 1
            if len(terms) > 1:
                m.addConstr(gp.quicksum(terms) <= 1 - y[nid])

    # Objective
    obj = gp.LinExpr()
    for nid in node_ids:
        n = nodes_by_id[nid]
        prob = n["probability"]
        price = n["electricity_price_dollars_per_MWh"]
        obj += prob * (price * x[nid] - a * y[nid] - b * x[nid]
                       - SU * u[nid] - SD * s_expr[nid])
    m.setObjective(obj, GRB.MAXIMIZE)

    # Time limit
    elapsed = time.time() - t_start
    remaining = max(2.0, args.time_limit - elapsed - 3.0)
    m.Params.TimeLimit = remaining

    # Ordered lists for callback extraction
    xvars = [x[nid] for nid in node_ids]
    yvars = [y[nid] for nid in node_ids]
    uvars = [u[nid] for nid in node_ids]

    best_logged = [-float("inf")]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objval <= best_logged[0] + 1e-9:
                return
            best_logged[0] = objval
            if logger is None:
                return
            try:
                xs = model.cbGetSolution(xvars)
                ys = model.cbGetSolution(yvars)
                us = model.cbGetSolution(uvars)
                xv = {}
                yv = {}
                uv = {}
                for i, nid in enumerate(node_ids):
                    yv[nid] = 1 if ys[i] > 0.5 else 0
                    uv[nid] = 1 if us[i] > 0.5 else 0
                    xv[nid] = xs[i] if yv[nid] == 1 else 0.0
                true_obj = compute_objective(nodes_by_id, node_ids, xv, yv, uv,
                                             y0, a, b, SU, SD)
                sol = build_solution_dict(node_ids, xv, yv, uv, true_obj)
                logger.log_solution(true_obj, sol)
            except Exception:
                try:
                    logger.log(objval)
                except Exception:
                    pass

    m.optimize(callback)

    # Extract final solution
    if m.SolCount > 0:
        xv = {}
        yv = {}
        uv = {}
        for nid in node_ids:
            yval = 1 if y[nid].X > 0.5 else 0
            uval = 1 if u[nid].X > 0.5 else 0
            yv[nid] = yval
            uv[nid] = uval
            xv[nid] = x[nid].X if yval == 1 else 0.0
        true_obj = compute_objective(nodes_by_id, node_ids, xv, yv, uv,
                                     y0, a, b, SU, SD)
        sol = build_solution_dict(node_ids, xv, yv, uv, true_obj)
    else:
        # Fallback: everything off
        xv = {nid: 0.0 for nid in node_ids}
        yv = {nid: 0 for nid in node_ids}
        uv = {nid: 0 for nid in node_ids}
        true_obj = compute_objective(nodes_by_id, node_ids, xv, yv, uv,
                                     y0, a, b, SU, SD)
        sol = build_solution_dict(node_ids, xv, yv, uv, true_obj)

    if logger:
        try:
            logger.log_solution(true_obj, sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()