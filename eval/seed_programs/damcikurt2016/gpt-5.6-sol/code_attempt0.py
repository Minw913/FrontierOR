import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(node_ids, nonroot_ids, x_values, y_values, u_values, objective):
    x_out = {}
    y_out = {}
    u_out = {}

    for node_id in node_ids:
        xv = float(x_values[node_id])
        yv = 1 if float(y_values[node_id]) >= 0.5 else 0

        if abs(xv) < 1e-7:
            xv = 0.0
        if xv != 0.0:
            x_out[str(node_id)] = xv
        if yv != 0:
            y_out[str(node_id)] = yv

    for node_id in nonroot_ids:
        uv = 1 if float(u_values[node_id]) >= 0.5 else 0
        if uv != 0:
            u_out[str(node_id)] = uv

    return {
        "objective_value": float(objective),
        "x": x_out,
        "y": y_out,
        "u": u_out,
        "sparse_defaults": {
            "x": 0,
            "y": 0,
            "u": 0
        }
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    gen = instance["generator"]
    tree = instance["scenario_tree"]
    nodes_data = tree["nodes"]

    node_by_id = {int(n["id"]): n for n in nodes_data}
    node_ids = [int(n["id"]) for n in nodes_data]
    roots = [n for n in nodes_data if n["parent_id"] is None]
    if len(roots) != 1:
        raise ValueError("The scenario tree must contain exactly one root.")
    root_id = int(roots[0]["id"])

    parent = {
        int(n["id"]): (None if n["parent_id"] is None else int(n["parent_id"]))
        for n in nodes_data
    }
    nonroot_ids = [i for i in node_ids if i != root_id]

    c_lower = float(gen["C_lower_MW"])
    c_upper = float(gen["C_upper_MW"])
    ramp_up = float(gen["V_plus_MW_per_h"])
    ramp_down = float(gen["V_minus_MW_per_h"])
    startup_cost = float(gen["U_bar_startup_cost_dollars"])
    shutdown_cost = float(gen["U_lower_shutdown_cost_dollars"])
    fuel_a = float(gen.get("fuel_a_dollars_per_h", 0.0))
    fuel_b = float(gen.get("fuel_b_dollars_per_MWh", 0.0))

    min_up = max(1, int(tree["L_min_up_time"]))
    min_down = max(1, int(tree["ell_min_down_time"]))
    initial_y = int(tree["initial_generator_status_y0"])
    initial_x = float(tree["initial_generation_x0_MW"])

    # The root is treated as the initial-period state corresponding to the
    # supplied initial commitment. Transitions, startups, and shutdowns begin
    # at non-root nodes, as required by the output schema.
    if initial_y == 1:
        fallback_x_level = min(c_upper, max(c_lower, initial_x))
    else:
        fallback_x_level = 0.0

    fallback_x = {i: fallback_x_level for i in node_ids}
    fallback_y = {i: initial_y for i in node_ids}
    fallback_u = {i: 0 for i in nonroot_ids}

    fallback_objective = 0.0
    for i in node_ids:
        n = node_by_id[i]
        probability = float(n["probability"])
        price = float(n["electricity_price_dollars_per_MWh"])
        fallback_objective += probability * (
            price * fallback_x[i]
            - fuel_b * fallback_x[i]
            - fuel_a * fallback_y[i]
        )

    fallback_solution = make_solution(
        node_ids,
        nonroot_ids,
        fallback_x,
        fallback_y,
        fallback_u,
        fallback_objective,
    )

    best_logged_objective = -math.inf
    if logger:
        try:
            logger.log_solution(fallback_objective, fallback_solution)
            best_logged_objective = fallback_objective
        except Exception:
            pass

    model = gp.Model("stochastic_unit_commitment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.monotonic() - start_time
    remaining = max(0.01, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    x = {
        i: model.addVar(lb=0.0, ub=c_upper, vtype=GRB.CONTINUOUS, name=f"x[{i}]")
        for i in node_ids
    }
    y = {
        i: model.addVar(vtype=GRB.BINARY, name=f"y[{i}]")
        for i in node_ids
    }
    u = {
        i: model.addVar(vtype=GRB.BINARY, name=f"u[{i}]")
        for i in nonroot_ids
    }
    shutdown = {
        i: model.addVar(vtype=GRB.BINARY, name=f"d[{i}]")
        for i in nonroot_ids
    }

    # Root commitment is the supplied initial status. Root generation may be
    # redispatched subject to the initial generation and ramp limits.
    model.addConstr(y[root_id] == initial_y, name="initial_commitment")
    model.addConstr(
        x[root_id] >= c_lower * y[root_id],
        name="root_min_generation",
    )
    model.addConstr(
        x[root_id] <= c_upper * y[root_id],
        name="root_max_generation",
    )
    model.addConstr(
        x[root_id] - initial_x
        <= ramp_up * initial_y + c_upper * (1 - initial_y),
        name="root_ramp_up",
    )
    model.addConstr(
        initial_x - x[root_id]
        <= ramp_down * y[root_id] + c_upper * (1 - y[root_id]),
        name="root_ramp_down",
    )

    for i in nonroot_ids:
        p = parent[i]

        model.addConstr(x[i] >= c_lower * y[i], name=f"min_generation[{i}]")
        model.addConstr(x[i] <= c_upper * y[i], name=f"max_generation[{i}]")

        # Exact startup and shutdown transition logic.
        model.addConstr(
            y[i] - y[p] == u[i] - shutdown[i],
            name=f"transition[{i}]",
        )
        model.addConstr(u[i] <= y[i], name=f"startup_current_on[{i}]")
        model.addConstr(u[i] <= 1 - y[p], name=f"startup_parent_off[{i}]")
        model.addConstr(shutdown[i] <= y[p], name=f"shutdown_parent_on[{i}]")
        model.addConstr(shutdown[i] <= 1 - y[i], name=f"shutdown_current_off[{i}]")

        # Ramp limits are relaxed on startup and shutdown, respectively.
        model.addConstr(
            x[i] - x[p]
            <= ramp_up * y[p] + c_upper * (1 - y[p]),
            name=f"ramp_up[{i}]",
        )
        model.addConstr(
            x[p] - x[i]
            <= ramp_down * y[i] + c_upper * (1 - y[i]),
            name=f"ramp_down[{i}]",
        )

    # Path-based minimum up/down constraints. At a node, recent startups on
    # its unique ancestor path require it to be on, while recent shutdowns
    # require it to be off.
    for i in node_ids:
        recent_startups = []
        current = i
        distance = 0
        while current is not None and distance < min_up:
            if current != root_id:
                recent_startups.append(u[current])
            current = parent[current]
            distance += 1

        if recent_startups:
            model.addConstr(
                gp.quicksum(recent_startups) <= y[i],
                name=f"minimum_up[{i}]",
            )

        recent_shutdowns = []
        current = i
        distance = 0
        while current is not None and distance < min_down:
            if current != root_id:
                recent_shutdowns.append(shutdown[current])
            current = parent[current]
            distance += 1

        if recent_shutdowns:
            model.addConstr(
                gp.quicksum(recent_shutdowns) <= 1 - y[i],
                name=f"minimum_down[{i}]",
            )

    objective_terms = []
    for i in node_ids:
        n = node_by_id[i]
        probability = float(n["probability"])
        price = float(n["electricity_price_dollars_per_MWh"])

        node_profit = (
            (price - fuel_b) * x[i]
            - fuel_a * y[i]
        )
        if i != root_id:
            node_profit -= startup_cost * u[i]
            node_profit -= shutdown_cost * shutdown[i]

        objective_terms.append(probability * node_profit)

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)

    # Provide a simple feasible incumbent whenever the supplied initial state
    # is physically valid.
    for i in node_ids:
        x[i].Start = fallback_x[i]
        y[i].Start = fallback_y[i]
    for i in nonroot_ids:
        u[i].Start = 0.0
        shutdown[i].Start = 0.0

    ordered_x_vars = [x[i] for i in node_ids]
    ordered_y_vars = [y[i] for i in node_ids]
    ordered_u_vars = [u[i] for i in nonroot_ids]

    callback_state = {
        "best": best_logged_objective
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return

        try:
            incumbent_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if incumbent_obj <= callback_state["best"] + 1e-7:
                return

            x_vals_list = cb_model.cbGetSolution(ordered_x_vars)
            y_vals_list = cb_model.cbGetSolution(ordered_y_vars)
            u_vals_list = cb_model.cbGetSolution(ordered_u_vars)

            x_vals = {i: x_vals_list[k] for k, i in enumerate(node_ids)}
            y_vals = {i: y_vals_list[k] for k, i in enumerate(node_ids)}
            u_vals = {i: u_vals_list[k] for k, i in enumerate(nonroot_ids)}

            solution = make_solution(
                node_ids,
                nonroot_ids,
                x_vals,
                y_vals,
                u_vals,
                incumbent_obj,
            )
            logger.log_solution(incumbent_obj, solution)
            callback_state["best"] = incumbent_obj
        except Exception:
            # Logging must not interrupt the optimization.
            pass

    try:
        model.optimize(incumbent_callback)
    except gp.GurobiError:
        # If optimization cannot proceed, return the supplied feasible start.
        pass

    if model.SolCount > 0:
        final_x = {i: x[i].X for i in node_ids}
        final_y = {i: y[i].X for i in node_ids}
        final_u = {i: u[i].X for i in nonroot_ids}
        final_objective = float(model.ObjVal)

        final_solution = make_solution(
            node_ids,
            nonroot_ids,
            final_x,
            final_y,
            final_u,
            final_objective,
        )
    else:
        final_solution = fallback_solution
        final_objective = fallback_objective

    if logger and final_objective > callback_state["best"] + 1e-7:
        try:
            logger.log_solution(final_objective, final_solution)
        except Exception:
            pass

    write_solution(args.solution_path, final_solution)


if __name__ == "__main__":
    main()