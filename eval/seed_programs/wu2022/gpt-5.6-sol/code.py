import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dims = data["dimensions"]
    num_items = int(dims["num_items"])
    num_periods = int(dims["num_periods"])
    num_plants = int(dims["num_plants"])
    num_customers = int(dims["num_customers"])

    items = range(num_items)
    periods = range(num_periods)
    plants = range(num_plants)
    customers = range(num_customers)

    capabilities = []
    for i in items:
        capable = sorted({
            int(p) for p in data["plant_capabilities"][i]
            if 0 <= int(p) < num_plants
        })
        capabilities.append(capable)

    demand_data = data["demand"]
    total_demand = [
        [
            sum(float(demand_data[i][c][t]) for c in customers)
            for t in periods
        ]
        for i in items
    ]

    remaining_demand = [[0.0] * num_periods for _ in items]
    future_demand = [[0.0] * num_periods for _ in items]
    for i in items:
        running = 0.0
        for t in reversed(range(num_periods)):
            future_demand[i][t] = running
            running += total_demand[i][t]
            remaining_demand[i][t] = running

    model = gp.Model("single_source_capacitated_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    z = {
        p: model.addVar(vtype=GRB.BINARY, name=f"Z_{p}")
        for p in plants
    }

    u = {}
    y = {}
    x = {}
    inventory = {}

    for i in items:
        for p in capabilities[i]:
            u[i, p] = model.addVar(
                vtype=GRB.BINARY, name=f"U_{i}_{p}"
            )
            for t in periods:
                y[i, p, t] = model.addVar(
                    vtype=GRB.BINARY, name=f"Y_{i}_{p}_{t}"
                )
                x[i, p, t] = model.addVar(
                    lb=0.0,
                    ub=remaining_demand[i][t],
                    vtype=GRB.CONTINUOUS,
                    name=f"X_{i}_{p}_{t}",
                )
                if t < num_periods - 1:
                    inventory[i, p, t] = model.addVar(
                        lb=0.0,
                        ub=future_demand[i][t],
                        vtype=GRB.CONTINUOUS,
                        name=f"I_{i}_{p}_{t}",
                    )

    model.update()

    # Each item is assigned to exactly one capable plant.
    for i in items:
        model.addConstr(
            gp.quicksum(u[i, p] for p in capabilities[i]) == 1,
            name=f"single_source_{i}",
        )

    # An assignment is possible only at an open plant.
    for i in items:
        for p in capabilities[i]:
            model.addConstr(
                u[i, p] <= z[p],
                name=f"assignment_open_{i}_{p}",
            )

    # Do not open plants to which no item is assigned.
    capable_items_by_plant = {
        p: [i for i in items if p in capabilities[i]]
        for p in plants
    }
    for p in plants:
        if capable_items_by_plant[p]:
            model.addConstr(
                z[p] <= gp.quicksum(
                    u[i, p] for i in capable_items_by_plant[p]
                ),
                name=f"open_if_used_{p}",
            )
        else:
            model.addConstr(z[p] == 0, name=f"unusable_plant_{p}")

    # Setup and production linking.
    for i in items:
        for p in capabilities[i]:
            for t in periods:
                model.addConstr(
                    y[i, p, t] <= u[i, p],
                    name=f"setup_assignment_{i}_{p}_{t}",
                )
                model.addConstr(
                    x[i, p, t]
                    <= remaining_demand[i][t] * y[i, p, t],
                    name=f"production_setup_{i}_{p}_{t}",
                )
                if remaining_demand[i][t] <= 0.0:
                    model.addConstr(
                        y[i, p, t] == 0,
                        name=f"zero_demand_setup_{i}_{p}_{t}",
                    )

            for t in range(num_periods - 1):
                model.addConstr(
                    inventory[i, p, t]
                    <= future_demand[i][t] * u[i, p],
                    name=f"inventory_assignment_{i}_{p}_{t}",
                )

    # Aggregate material balance for each item and period.
    for i in items:
        for t in periods:
            lhs = gp.quicksum(
                x[i, p, t] for p in capabilities[i]
            )
            if t > 0:
                lhs += gp.quicksum(
                    inventory[i, p, t - 1] for p in capabilities[i]
                )

            rhs = total_demand[i][t]
            if t < num_periods - 1:
                rhs += gp.quicksum(
                    inventory[i, p, t] for p in capabilities[i]
                )

            model.addConstr(
                lhs == rhs,
                name=f"balance_{i}_{t}",
            )

    # Plant capacity in each period.
    setup_time = data["setup_time"]
    production_time = data["production_time"]
    capacity = data["capacity"]

    for p in plants:
        relevant_items = capable_items_by_plant[p]
        for t in periods:
            consumed_time = gp.quicksum(
                float(setup_time[i][p][t]) * y[i, p, t]
                + float(production_time[i][p][t]) * x[i, p, t]
                for i in relevant_items
            )
            model.addConstr(
                consumed_time <= float(capacity[p][t]) * z[p],
                name=f"capacity_{p}_{t}",
            )

    setup_cost = data["setup_cost"]
    production_cost = data["production_cost"]
    holding_cost = data["holding_cost"]
    opening_cost = data["plant_opening_cost"]
    transportation_cost = data["transportation_cost"]

    objective = gp.LinExpr()

    for p in plants:
        objective += float(opening_cost[p]) * z[p]

    for i in items:
        for p in capabilities[i]:
            # Transportation is determined by the item's single source and
            # the customer demand in each demand period.
            transport_assignment_cost = 0.0
            for c in customers:
                for t in periods:
                    transport_assignment_cost += (
                        float(demand_data[i][c][t])
                        * float(transportation_cost[i][c][p][t])
                    )
            objective += transport_assignment_cost * u[i, p]

            for t in periods:
                objective += float(setup_cost[i][p][t]) * y[i, p, t]
                objective += float(production_cost[i][p][t]) * x[i, p, t]

                if t < num_periods - 1:
                    objective += (
                        float(holding_cost[i][p][t])
                        * inventory[i, p, t]
                    )

    model.setObjective(objective, GRB.MINIMIZE)

    z_entries = [(p, z[p]) for p in plants]
    u_entries = [
        ((i, p), u[i, p])
        for i in items
        for p in capabilities[i]
    ]
    y_entries = [
        ((i, p, t), y[i, p, t])
        for i in items
        for p in capabilities[i]
        for t in periods
    ]

    def make_solution(get_values, objective_value):
        z_values = get_values([var for _, var in z_entries])
        u_values = get_values([var for _, var in u_entries])
        y_values = get_values([var for _, var in y_entries])

        solution = {
            "objective_value": float(objective_value),
            "Z": {
                str(p): int(value >= 0.5)
                for (p, _), value in zip(z_entries, z_values)
            },
            "U": {
                f"{i}_{p}": int(value >= 0.5)
                for ((i, p), _), value in zip(u_entries, u_values)
            },
            "Y": {
                f"{i}_{p}_{t}": int(value >= 0.5)
                for ((i, p, t), _), value in zip(y_entries, y_values)
            },
        }
        return solution

    best_logged = [float("inf")]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        obj_value = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
        tolerance = 1e-9 * max(1.0, abs(best_logged[0]), abs(obj_value))
        if obj_value >= best_logged[0] - tolerance:
            return

        best_logged[0] = obj_value
        if logger:
            solution = make_solution(
                cb_model.cbGetSolution,
                obj_value,
            )
            logger.log_solution(obj_value, solution)

    elapsed = time.monotonic() - start_time
    remaining_time = max(0.0, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    if logger:
        model.optimize(incumbent_callback)
    else:
        model.optimize()

    if model.SolCount <= 0:
        raise RuntimeError(
            f"No feasible solution was found. Gurobi status: {model.Status}"
        )

    final_objective = float(model.ObjVal)
    final_solution = make_solution(
        lambda variables: [var.X for var in variables],
        final_objective,
    )

    tolerance = 1e-9 * max(
        1.0, abs(best_logged[0]), abs(final_objective)
    )
    if logger and final_objective < best_logged[0] - tolerance:
        logger.log_solution(final_objective, final_solution)
        best_logged[0] = final_objective

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()