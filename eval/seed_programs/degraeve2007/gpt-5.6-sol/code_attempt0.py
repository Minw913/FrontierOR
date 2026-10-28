import argparse
import json
import math
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


def clean_float(value):
    value = float(value)
    if abs(value) < 1e-9:
        return 0.0
    return value


def write_solution(path, solution):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dimensions = data["dimensions"]
    parameters = data["parameters"]

    num_products = int(dimensions["num_products"])
    num_periods = int(dimensions["num_periods"])

    demand = parameters["demand"]
    holding_cost = parameters["holding_cost"]
    setup_cost = parameters["setup_cost"]
    variable_cost = parameters["variable_production_cost"]
    production_time = parameters["variable_production_time"]
    setup_time = parameters["setup_time"]
    capacity = parameters["capacity"]
    initial_inventory_cost = parameters["initial_inventory_cost"]

    def objective_from_values(production, setup, inventory, initial_inventory):
        terms = []

        for i in range(num_products):
            terms.append(
                float(initial_inventory_cost[i]) * float(initial_inventory[i])
            )
            for t in range(num_periods):
                terms.append(
                    float(setup_cost[i][t]) * int(setup[i][t])
                )
                terms.append(
                    float(variable_cost[i][t]) * float(production[i][t])
                )
                terms.append(
                    float(holding_cost[i][t]) * float(inventory[i][t])
                )

        return float(math.fsum(terms))

    def make_solution(production_flat, setup_flat, inventory_flat, initial_flat):
        production = []
        setup = []
        inventory = []

        k = 0
        for i in range(num_products):
            production_row = []
            setup_row = []
            inventory_row = []

            for t in range(num_periods):
                production_row.append(clean_float(production_flat[k]))
                setup_row.append(1 if float(setup_flat[k]) >= 0.5 else 0)
                inventory_row.append(clean_float(inventory_flat[k]))
                k += 1

            production.append(production_row)
            setup.append(setup_row)
            inventory.append(inventory_row)

        initial_inventory = [
            clean_float(initial_flat[i]) for i in range(num_products)
        ]

        objective_value = objective_from_values(
            production, setup, inventory, initial_inventory
        )

        return {
            "objective_value": objective_value,
            "variables": {
                "production": production,
                "setup": setup,
                "inventory": inventory,
                "initial_inventory": initial_inventory,
            },
        }

    # A guaranteed feasible solution: satisfy all demand using initial inventory.
    baseline_production = [
        [0.0 for _ in range(num_periods)] for _ in range(num_products)
    ]
    baseline_setup = [
        [0 for _ in range(num_periods)] for _ in range(num_products)
    ]
    baseline_inventory = []
    baseline_initial = []

    for i in range(num_products):
        total_demand = float(math.fsum(float(v) for v in demand[i]))
        baseline_initial.append(total_demand)

        row = []
        remaining = total_demand
        for t in range(num_periods):
            remaining -= float(demand[i][t])
            row.append(clean_float(remaining))
        baseline_inventory.append(row)

    baseline_objective = objective_from_values(
        baseline_production,
        baseline_setup,
        baseline_inventory,
        baseline_initial,
    )
    best_solution = {
        "objective_value": baseline_objective,
        "variables": {
            "production": baseline_production,
            "setup": baseline_setup,
            "inventory": baseline_inventory,
            "initial_inventory": baseline_initial,
        },
    }
    best_objective = baseline_objective

    if logger:
        logger.log_solution(best_objective, best_solution)

    # If the time limit has already been exhausted, return the guaranteed
    # feasible baseline without starting the optimizer.
    if time.monotonic() - start_time >= max(0, args.time_limit):
        write_solution(args.solution_path, best_solution)
        return

    model = None

    try:
        model = gp.Model("capacitated_multi_item_lot_sizing")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        production = model.addVars(
            num_products,
            num_periods,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="production",
        )
        setup = model.addVars(
            num_products,
            num_periods,
            vtype=GRB.BINARY,
            name="setup",
        )
        inventory = model.addVars(
            num_products,
            num_periods,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="inventory",
        )
        initial_inventory = model.addVars(
            num_products,
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="initial_inventory",
        )

        cumulative_remaining_demand = [
            [0.0 for _ in range(num_periods)] for _ in range(num_products)
        ]
        for i in range(num_products):
            running = 0.0
            for t in range(num_periods - 1, -1, -1):
                running += float(demand[i][t])
                cumulative_remaining_demand[i][t] = running

        # Inventory-flow constraints.
        for i in range(num_products):
            model.addConstr(
                initial_inventory[i] + production[i, 0]
                == float(demand[i][0]) + inventory[i, 0],
                name=f"balance_{i}_0",
            )
            for t in range(1, num_periods):
                model.addConstr(
                    inventory[i, t - 1] + production[i, t]
                    == float(demand[i][t]) + inventory[i, t],
                    name=f"balance_{i}_{t}",
                )

        # Production/setup linking constraints with the specified tight upper
        # bound for each product-period pair.
        production_upper_bounds = {}
        for i in range(num_products):
            for t in range(num_periods):
                pt = float(production_time[i][t])
                available_after_setup = (
                    float(capacity[t]) - float(setup_time[i][t])
                )

                if pt > 0.0:
                    capacity_bound = max(0.0, available_after_setup / pt)
                else:
                    capacity_bound = cumulative_remaining_demand[i][t]

                upper_bound = max(
                    0.0,
                    min(
                        capacity_bound,
                        cumulative_remaining_demand[i][t],
                    ),
                )
                production_upper_bounds[i, t] = upper_bound

                model.addConstr(
                    production[i, t] <= upper_bound * setup[i, t],
                    name=f"setup_link_{i}_{t}",
                )

        # Shared resource capacity.
        for t in range(num_periods):
            model.addConstr(
                gp.quicksum(
                    float(setup_time[i][t]) * setup[i, t]
                    + float(production_time[i][t]) * production[i, t]
                    for i in range(num_products)
                )
                <= float(capacity[t]),
                name=f"capacity_{t}",
            )

        objective = gp.quicksum(
            float(initial_inventory_cost[i]) * initial_inventory[i]
            for i in range(num_products)
        )
        objective += gp.quicksum(
            float(setup_cost[i][t]) * setup[i, t]
            + float(variable_cost[i][t]) * production[i, t]
            + float(holding_cost[i][t]) * inventory[i, t]
            for i in range(num_products)
            for t in range(num_periods)
        )
        model.setObjective(objective, GRB.MINIMIZE)

        # Install the guaranteed feasible solution as a MIP start.
        for i in range(num_products):
            initial_inventory[i].Start = baseline_initial[i]
            for t in range(num_periods):
                production[i, t].Start = 0.0
                setup[i, t].Start = 0.0
                inventory[i, t].Start = baseline_inventory[i][t]

        production_vars = [
            production[i, t]
            for i in range(num_products)
            for t in range(num_periods)
        ]
        setup_vars = [
            setup[i, t]
            for i in range(num_products)
            for t in range(num_periods)
        ]
        inventory_vars = [
            inventory[i, t]
            for i in range(num_products)
            for t in range(num_periods)
        ]
        initial_vars = [
            initial_inventory[i] for i in range(num_products)
        ]

        incumbent_state = {
            "solver_objective": best_objective,
            "solution_objective": best_objective,
            "solution": best_solution,
        }

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                incumbent_obj = float(
                    cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                )

                if incumbent_obj >= incumbent_state["solver_objective"] - 1e-9:
                    return

                production_values = cb_model.cbGetSolution(production_vars)
                setup_values = cb_model.cbGetSolution(setup_vars)
                inventory_values = cb_model.cbGetSolution(inventory_vars)
                initial_values = cb_model.cbGetSolution(initial_vars)

                solution = make_solution(
                    production_values,
                    setup_values,
                    inventory_values,
                    initial_values,
                )

                incumbent_state["solver_objective"] = incumbent_obj
                incumbent_state["solution_objective"] = solution[
                    "objective_value"
                ]
                incumbent_state["solution"] = solution

                if logger:
                    logger.log_solution(
                        solution["objective_value"], solution
                    )
            except Exception:
                # Logging or solution serialization must not interrupt the MIP.
                return

        remaining_time = args.time_limit - (time.monotonic() - start_time)

        if remaining_time > 0.0:
            model.Params.TimeLimit = max(0.001, remaining_time)
            model.optimize(incumbent_callback)

        # Extract Gurobi's final incumbent. This also covers cases in which an
        # incumbent was accepted without a callback being delivered.
        if model.SolCount > 0:
            final_production = [v.X for v in production_vars]
            final_setup = [v.X for v in setup_vars]
            final_inventory = [v.X for v in inventory_vars]
            final_initial = [v.X for v in initial_vars]

            final_solution = make_solution(
                final_production,
                final_setup,
                final_inventory,
                final_initial,
            )

            if (
                final_solution["objective_value"]
                < incumbent_state["solution_objective"] - 1e-8
            ):
                incumbent_state["solution_objective"] = final_solution[
                    "objective_value"
                ]
                incumbent_state["solution"] = final_solution
                if logger:
                    logger.log_solution(
                        final_solution["objective_value"], final_solution
                    )
            elif model.ObjVal <= incumbent_state["solver_objective"] + 1e-7:
                # Prefer the final solver values when they represent the same
                # incumbent, as they are generally the cleanest values.
                incumbent_state["solution"] = final_solution
                incumbent_state["solution_objective"] = final_solution[
                    "objective_value"
                ]

        if (
            incumbent_state["solution_objective"]
            <= best_objective + 1e-8
        ):
            best_solution = incumbent_state["solution"]
            best_objective = incumbent_state["solution_objective"]

    except gp.GurobiError:
        # The all-initial-inventory plan remains a valid fallback.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    best_solution["objective_value"] = float(
        best_solution["objective_value"]
    )
    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()