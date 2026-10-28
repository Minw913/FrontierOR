import argparse
import json
import math
import os
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def write_solution(path, solution):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


def make_solution(node_ids, nonroot_ids, x_values, y_values, u_values, objective):
    x_output = {}
    y_output = {}
    u_output = {}

    for node_id in node_ids:
        value = float(x_values.get(node_id, 0.0))
        if abs(value) <= 1e-7:
            value = 0.0
        if value != 0.0:
            x_output[str(node_id)] = value

        value = 1 if float(y_values.get(node_id, 0.0)) >= 0.5 else 0
        if value:
            y_output[str(node_id)] = value

    for node_id in nonroot_ids:
        value = 1 if float(u_values.get(node_id, 0.0)) >= 0.5 else 0
        if value:
            u_output[str(node_id)] = value

    return {
        "objective_value": float(objective),
        "x": x_output,
        "y": y_output,
        "u": u_output,
        "sparse_defaults": {
            "x": 0,
            "y": 0,
            "u": 0
        }
    }


def evaluate_solution(
    node_ids,
    nonroot_ids,
    node_by_id,
    parent,
    x_values,
    y_values,
    u_values,
    fuel_a,
    fuel_b,
    startup_cost,
    shutdown_cost
):
    objective = 0.0

    for node_id in node_ids:
        node = node_by_id[node_id]
        probability = float(node["probability"])
        price = float(node["electricity_price_dollars_per_MWh"])

        x_value = float(x_values.get(node_id, 0.0))
        y_value = 1 if float(y_values.get(node_id, 0.0)) >= 0.5 else 0

        node_profit = (
            price * x_value
            - fuel_b * x_value
            - fuel_a * y_value
        )

        if node_id in u_values:
            startup = 1 if float(u_values[node_id]) >= 0.5 else 0
            shutdown = (
                (1 if float(y_values[parent[node_id]]) >= 0.5 else 0)
                - y_value
                + startup
            )
            node_profit -= startup_cost * startup
            node_profit -= shutdown_cost * shutdown

        objective += probability * node_profit

    return objective


def build_fallback_solution(
    node_ids,
    nonroot_ids,
    node_by_id,
    parent,
    initial_y,
    initial_x,
    c_lower,
    c_upper,
    fuel_a,
    fuel_b,
    startup_cost,
    shutdown_cost
):
    # Keeping the initial state unchanged throughout the tree is feasible for
    # valid instance data and provides a guaranteed incumbent if Gurobi is
    # unavailable or no MIP incumbent is found.
    if initial_y:
        level = min(c_upper, max(c_lower, initial_x))
    else:
        level = 0.0

    x_values = {node_id: level for node_id in node_ids}
    y_values = {node_id: int(initial_y) for node_id in node_ids}
    u_values = {node_id: 0 for node_id in nonroot_ids}

    objective = evaluate_solution(
        node_ids,
        nonroot_ids,
        node_by_id,
        parent,
        x_values,
        y_values,
        u_values,
        fuel_a,
        fuel_b,
        startup_cost,
        shutdown_cost
    )

    return make_solution(
        node_ids,
        nonroot_ids,
        x_values,
        y_values,
        u_values,
        objective
    ), x_values, y_values, u_values


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

    generator = instance["generator"]
    tree = instance["scenario_tree"]
    nodes_data = tree["nodes"]

    node_by_id = {int(node["id"]): node for node in nodes_data}
    node_ids = [int(node["id"]) for node in nodes_data]

    roots = [
        int(node["id"])
        for node in nodes_data
        if node["parent_id"] is None
    ]
    if len(roots) != 1:
        raise ValueError("The scenario tree must have exactly one root.")

    root_id = roots[0]
    parent = {
        int(node["id"]): (
            None if node["parent_id"] is None else int(node["parent_id"])
        )
        for node in nodes_data
    }
    nonroot_ids = [node_id for node_id in node_ids if node_id != root_id]

    c_lower = float(generator["C_lower_MW"])
    c_upper = float(generator["C_upper_MW"])
    ramp_up = float(generator["V_plus_MW_per_h"])
    ramp_down = float(generator["V_minus_MW_per_h"])
    startup_cost = float(generator["U_bar_startup_cost_dollars"])
    shutdown_cost = float(generator["U_lower_shutdown_cost_dollars"])
    fuel_a = float(generator.get("fuel_a_dollars_per_h", 0.0))
    fuel_b = float(generator.get("fuel_b_dollars_per_MWh", 0.0))

    min_up = max(1, int(tree["L_min_up_time"]))
    min_down = max(1, int(tree["ell_min_down_time"]))
    initial_y = int(tree["initial_generator_status_y0"])
    initial_x = float(tree["initial_generation_x0_MW"])

    fallback_solution, fallback_x, fallback_y, fallback_u = (
        build_fallback_solution(
            node_ids,
            nonroot_ids,
            node_by_id,
            parent,
            initial_y,
            initial_x,
            c_lower,
            c_upper,
            fuel_a,
            fuel_b,
            startup_cost,
            shutdown_cost
        )
    )

    best_logged_objective = -math.inf
    if logger:
        try:
            logger.log_solution(
                fallback_solution["objective_value"],
                fallback_solution
            )
            best_logged_objective = fallback_solution["objective_value"]
        except Exception:
            pass

    # Gurobi can be importable even when model construction fails because no
    # license is available. In that case, return the feasible fallback rather
    # than terminating without an output file.
    if gp is None or GRB is None:
        write_solution(args.solution_path, fallback_solution)
        return

    try:
        model = gp.Model("stochastic_unit_commitment")
    except Exception:
        write_solution(args.solution_path, fallback_solution)
        return

    try:
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        elapsed = time.monotonic() - start_time
        remaining_time = max(0.01, float(args.time_limit) - elapsed)
        model.Params.TimeLimit = remaining_time

        x = {
            node_id: model.addVar(
                lb=0.0,
                ub=c_upper,
                vtype=GRB.CONTINUOUS,
                name=f"x[{node_id}]"
            )
            for node_id in node_ids
        }
        y = {
            node_id: model.addVar(
                vtype=GRB.BINARY,
                name=f"y[{node_id}]"
            )
            for node_id in node_ids
        }
        u = {
            node_id: model.addVar(
                vtype=GRB.BINARY,
                name=f"u[{node_id}]"
            )
            for node_id in nonroot_ids
        }
        shutdown = {
            node_id: model.addVar(
                vtype=GRB.BINARY,
                name=f"d[{node_id}]"
            )
            for node_id in nonroot_ids
        }

        model.addConstr(
            y[root_id] == initial_y,
            name="initial_commitment"
        )
        model.addConstr(
            x[root_id] >= c_lower * y[root_id],
            name="root_minimum_generation"
        )
        model.addConstr(
            x[root_id] <= c_upper * y[root_id],
            name="root_maximum_generation"
        )

        # Ramp from the supplied pre-horizon generation into the root period.
        model.addConstr(
            x[root_id] - initial_x
            <= ramp_up * initial_y + c_upper * (1 - initial_y),
            name="root_ramp_up"
        )
        model.addConstr(
            initial_x - x[root_id]
            <= ramp_down * y[root_id] + c_upper * (1 - y[root_id]),
            name="root_ramp_down"
        )

        for node_id in nonroot_ids:
            parent_id = parent[node_id]

            model.addConstr(
                x[node_id] >= c_lower * y[node_id],
                name=f"minimum_generation[{node_id}]"
            )
            model.addConstr(
                x[node_id] <= c_upper * y[node_id],
                name=f"maximum_generation[{node_id}]"
            )

            # y_parent - y_current + startup is the shutdown variable.
            model.addConstr(
                shutdown[node_id]
                == y[parent_id] - y[node_id] + u[node_id],
                name=f"shutdown_definition[{node_id}]"
            )

            model.addConstr(
                u[node_id] <= y[node_id],
                name=f"startup_requires_on[{node_id}]"
            )
            model.addConstr(
                u[node_id] <= 1 - y[parent_id],
                name=f"startup_requires_parent_off[{node_id}]"
            )
            model.addConstr(
                u[node_id] >= y[node_id] - y[parent_id],
                name=f"startup_lower_bound[{node_id}]"
            )

            # Startup permits a free jump from zero; otherwise ramp-up is
            # limited while the parent is running.
            model.addConstr(
                x[node_id] - x[parent_id]
                <= ramp_up * y[parent_id]
                + c_upper * (1 - y[parent_id]),
                name=f"ramp_up[{node_id}]"
            )

            # Shutdown permits a free drop to zero; otherwise ramp-down is
            # limited while the current node is running.
            model.addConstr(
                x[parent_id] - x[node_id]
                <= ramp_down * y[node_id]
                + c_upper * (1 - y[node_id]),
                name=f"ramp_down[{node_id}]"
            )

        # For every node, collect startup and shutdown events among the recent
        # nodes on its unique root path.
        for node_id in node_ids:
            recent_startups = []
            cursor = node_id
            distance = 0

            while cursor is not None and distance < min_up:
                if cursor != root_id:
                    recent_startups.append(u[cursor])
                cursor = parent[cursor]
                distance += 1

            if recent_startups:
                model.addConstr(
                    gp.quicksum(recent_startups) <= y[node_id],
                    name=f"minimum_up_time[{node_id}]"
                )

            recent_shutdowns = []
            cursor = node_id
            distance = 0

            while cursor is not None and distance < min_down:
                if cursor != root_id:
                    recent_shutdowns.append(shutdown[cursor])
                cursor = parent[cursor]
                distance += 1

            if recent_shutdowns:
                model.addConstr(
                    gp.quicksum(recent_shutdowns) <= 1 - y[node_id],
                    name=f"minimum_down_time[{node_id}]"
                )

        objective_terms = []

        for node_id in node_ids:
            node = node_by_id[node_id]
            probability = float(node["probability"])
            price = float(node["electricity_price_dollars_per_MWh"])

            node_profit = (
                (price - fuel_b) * x[node_id]
                - fuel_a * y[node_id]
            )

            if node_id != root_id:
                node_profit -= startup_cost * u[node_id]
                node_profit -= shutdown_cost * shutdown[node_id]

            objective_terms.append(probability * node_profit)

        model.setObjective(
            gp.quicksum(objective_terms),
            GRB.MAXIMIZE
        )

        for node_id in node_ids:
            x[node_id].Start = fallback_x[node_id]
            y[node_id].Start = fallback_y[node_id]

        for node_id in nonroot_ids:
            u[node_id].Start = fallback_u[node_id]
            shutdown[node_id].Start = 0.0

        ordered_x = [x[node_id] for node_id in node_ids]
        ordered_y = [y[node_id] for node_id in node_ids]
        ordered_u = [u[node_id] for node_id in nonroot_ids]

        callback_state = {"best": best_logged_objective}

        def incumbent_callback(callback_model, where):
            if logger is None or where != GRB.Callback.MIPSOL:
                return

            try:
                objective_value = float(
                    callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                )

                if objective_value <= callback_state["best"] + 1e-7:
                    return

                x_list = callback_model.cbGetSolution(ordered_x)
                y_list = callback_model.cbGetSolution(ordered_y)
                u_list = callback_model.cbGetSolution(ordered_u)

                x_values = {
                    node_id: x_list[index]
                    for index, node_id in enumerate(node_ids)
                }
                y_values = {
                    node_id: y_list[index]
                    for index, node_id in enumerate(node_ids)
                }
                u_values = {
                    node_id: u_list[index]
                    for index, node_id in enumerate(nonroot_ids)
                }

                incumbent_solution = make_solution(
                    node_ids,
                    nonroot_ids,
                    x_values,
                    y_values,
                    u_values,
                    objective_value
                )

                logger.log_solution(
                    objective_value,
                    incumbent_solution
                )
                callback_state["best"] = objective_value
            except Exception:
                # Logging errors must never interrupt optimization.
                pass

        try:
            model.optimize(incumbent_callback)
        except Exception:
            write_solution(args.solution_path, fallback_solution)
            try:
                model.dispose()
            except Exception:
                pass
            return

        if model.SolCount > 0:
            final_x = {node_id: x[node_id].X for node_id in node_ids}
            final_y = {node_id: y[node_id].X for node_id in node_ids}
            final_u = {
                node_id: u[node_id].X
                for node_id in nonroot_ids
            }
            final_objective = float(model.ObjVal)

            final_solution = make_solution(
                node_ids,
                nonroot_ids,
                final_x,
                final_y,
                final_u,
                final_objective
            )
        else:
            final_solution = fallback_solution
            final_objective = fallback_solution["objective_value"]

        if logger and final_objective > callback_state["best"] + 1e-7:
            try:
                logger.log_solution(final_objective, final_solution)
            except Exception:
                pass

        write_solution(args.solution_path, final_solution)

    except Exception:
        # This also handles failures during model construction caused by
        # restricted-size or otherwise unusable Gurobi licenses.
        write_solution(args.solution_path, fallback_solution)
    finally:
        try:
            model.dispose()
        except Exception:
            pass


if __name__ == "__main__":
    main()