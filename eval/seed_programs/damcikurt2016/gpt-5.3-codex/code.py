import argparse
import json
import math
from typing import Dict, Any, List, Optional

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def build_solution_dict(
    objective_value: float,
    node_ids: List[int],
    non_root_ids: List[int],
    x_vals: Dict[int, float],
    y_vals: Dict[int, float],
    u_vals: Dict[int, float],
    tol: float = 1e-6,
) -> Dict[str, Any]:
    x_out = {}
    y_out = {}
    u_out = {}

    for n in node_ids:
        xv = x_vals.get(n, 0.0)
        yv = y_vals.get(n, 0.0)

        if abs(xv) > tol:
            x_out[str(n)] = float(xv)
        if yv > 0.5:
            y_out[str(n)] = 1

    for n in non_root_ids:
        uv = u_vals.get(n, 0.0)
        if uv > 0.5:
            u_out[str(n)] = 1

    return {
        "objective_value": float(objective_value),
        "x": x_out,
        "y": y_out,
        "u": u_out,
        "sparse_defaults": {"x": 0, "y": 0, "u": 0},
    }


def solve_instance(instance: Dict[str, Any], time_limit: int, logger: Optional[SolutionLogger]) -> Dict[str, Any]:
    gen = instance["generator"]
    tree = instance["scenario_tree"]

    C_lower = float(gen["C_lower_MW"])
    C_upper = float(gen["C_upper_MW"])
    V_plus = float(gen["V_plus_MW_per_h"])
    V_minus = float(gen["V_minus_MW_per_h"])
    startup_cost = float(gen["U_bar_startup_cost_dollars"])
    shutdown_cost = float(gen["U_lower_shutdown_cost_dollars"])
    fuel_a = float(gen.get("fuel_a_dollars_per_h", 0.0) or 0.0)
    fuel_b = float(gen.get("fuel_b_dollars_per_MWh", 0.0) or 0.0)

    L_min_up = int(tree["L_min_up_time"])
    ell_min_down = int(tree["ell_min_down_time"])
    y0 = int(tree["initial_generator_status_y0"])
    x0 = float(tree["initial_generation_x0_MW"])

    nodes = tree["nodes"]
    node_ids = [int(nd["id"]) for nd in nodes]
    node_data = {int(nd["id"]): nd for nd in nodes}

    parent = {}
    time_of = {}
    prob = {}
    price = {}
    root_id = None

    for nd in nodes:
        nid = int(nd["id"])
        pid = nd["parent_id"]
        parent[nid] = None if pid is None else int(pid)
        time_of[nid] = int(nd["time"])
        prob[nid] = float(nd["probability"])
        price[nid] = float(nd["electricity_price_dollars_per_MWh"])
        if pid is None:
            root_id = nid

    if root_id is None:
        raise ValueError("Scenario tree has no root node (parent_id is null).")

    non_root_ids = [n for n in node_ids if n != root_id]
    max_window = max(L_min_up, ell_min_down)

    model = gp.Model("stochastic_unit_commitment_single_gen")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(time_limit)

    # Decision variables
    x = {n: model.addVar(lb=0.0, ub=C_upper, vtype=GRB.CONTINUOUS, name=f"x_{n}") for n in node_ids}
    y = {n: model.addVar(vtype=GRB.BINARY, name=f"y_{n}") for n in node_ids}

    # Startup variable defined at all nodes internally (root included for initial transition modeling)
    u = {n: model.addVar(vtype=GRB.BINARY, name=f"u_{n}") for n in node_ids}

    model.update()

    # Startup logical constraints
    # Root uses initial state y0
    model.addConstr(u[root_id] >= y[root_id] - y0, name=f"startup_lb_root_{root_id}")
    model.addConstr(u[root_id] <= y[root_id], name=f"startup_ub1_root_{root_id}")
    model.addConstr(u[root_id] <= 1 - y0, name=f"startup_ub2_root_{root_id}")

    for n in non_root_ids:
        p = parent[n]
        model.addConstr(u[n] >= y[n] - y[p], name=f"startup_lb_{n}")
        model.addConstr(u[n] <= y[n], name=f"startup_ub1_{n}")
        model.addConstr(u[n] <= 1 - y[p], name=f"startup_ub2_{n}")

    # Generation bounds linked to commitment
    for n in node_ids:
        model.addConstr(x[n] <= C_upper * y[n], name=f"gen_ub_{n}")
        model.addConstr(x[n] >= C_lower * y[n], name=f"gen_lb_{n}")

    # Ramp constraints
    # Root against initial generation/state
    model.addConstr(
        x[root_id] - x0 <= V_plus * y0 + C_upper * (1 - y0),
        name=f"ramp_up_root_{root_id}",
    )
    model.addConstr(
        x0 - x[root_id] <= V_minus * y[root_id] + C_upper * (1 - y[root_id]),
        name=f"ramp_down_root_{root_id}",
    )

    for n in non_root_ids:
        p = parent[n]
        model.addConstr(
            x[n] - x[p] <= V_plus * y[p] + C_upper * (1 - y[p]),
            name=f"ramp_up_{n}",
        )
        model.addConstr(
            x[p] - x[n] <= V_minus * y[n] + C_upper * (1 - y[n]),
            name=f"ramp_down_{n}",
        )

    # Shutdown expressions (linear expressions, no explicit variable needed)
    w_expr = {}
    w_expr[root_id] = y0 - y[root_id] + u[root_id]
    for n in non_root_ids:
        p = parent[n]
        w_expr[n] = y[p] - y[n] + u[n]

    # Minimum up/down constraints along ancestor chain
    for m in node_ids:
        expr_up = gp.LinExpr()
        expr_down = gp.LinExpr()

        cur = m
        while cur is not None:
            dt = time_of[m] - time_of[cur]
            if dt < L_min_up:
                expr_up += u[cur]
            if dt < ell_min_down:
                expr_down += w_expr[cur]

            if dt >= max_window - 1:
                break
            cur = parent[cur]

        model.addConstr(expr_up <= y[m], name=f"min_up_{m}")
        model.addConstr(expr_down <= 1 - y[m], name=f"min_down_{m}")

    # Objective: expected profit
    obj = gp.LinExpr()
    for n in node_ids:
        pn = prob[n]
        obj += pn * (price[n] * x[n] - fuel_b * x[n] - fuel_a * y[n] - startup_cost * u[n] - shutdown_cost * w_expr[n])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Callback for incumbent logging
    model._logger = logger
    model._best_logged = -1e100
    model._node_ids = node_ids
    model._non_root_ids = non_root_ids
    model._x_vars = x
    model._y_vars = y
    model._u_vars = u

    def cb(m: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            incumbent_obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj > m._best_logged + 1e-9:
                x_vals_list = m.cbGetSolution([m._x_vars[n] for n in m._node_ids])
                y_vals_list = m.cbGetSolution([m._y_vars[n] for n in m._node_ids])
                u_vals_list = m.cbGetSolution([m._u_vars[n] for n in m._node_ids])

                x_vals = {n: x_vals_list[i] for i, n in enumerate(m._node_ids)}
                y_vals = {n: y_vals_list[i] for i, n in enumerate(m._node_ids)}
                u_vals = {n: u_vals_list[i] for i, n in enumerate(m._node_ids)}

                sol_dict = build_solution_dict(
                    incumbent_obj,
                    m._node_ids,
                    m._non_root_ids,
                    x_vals,
                    y_vals,
                    u_vals,
                )
                m._logger.log_solution(incumbent_obj, sol_dict)
                m._best_logged = incumbent_obj

    model.optimize(cb)

    if model.SolCount > 0:
        x_vals = {n: x[n].X for n in node_ids}
        y_vals = {n: y[n].X for n in node_ids}
        u_vals = {n: u[n].X for n in node_ids}
        best_obj = float(model.ObjVal)
        sol = build_solution_dict(best_obj, node_ids, non_root_ids, x_vals, y_vals, u_vals)
        if logger:
            logger.log_solution(best_obj, sol)
        return sol

    # Fallback (unlikely)
    fallback = {
        "objective_value": 0.0,
        "x": {},
        "y": {},
        "u": {},
        "sparse_defaults": {"x": 0, "y": 0, "u": 0},
    }
    if logger:
        logger.log_solution(0.0, fallback)
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Solve stochastic unit commitment on a scenario tree.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()