import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def clean_continuous(value, tolerance=1e-8):
    value = float(value)
    if abs(value) <= tolerance:
        return 0.0
    return value


def calculate_objective(data, production, setup, inventory):
    num_items = data["num_items"]
    num_periods = data["num_periods"]

    total = 0.0
    for i in range(num_items):
        total += float(data["initial_inventory_cost"][i]) * inventory[i][0]
        for t in range(num_periods):
            total += float(data["setup_cost"][i]) * setup[i][t]
            total += float(data["variable_cost"][i]) * production[i][t]
            total += float(data["holding_cost"][i]) * inventory[i][t + 1]
    return float(total)


def make_solution(data, production, setup, inventory):
    production = [
        [clean_continuous(value) for value in row]
        for row in production
    ]
    setup = [
        [int(round(value)) for value in row]
        for row in setup
    ]
    inventory = [
        [clean_continuous(value) for value in row]
        for row in inventory
    ]

    objective = calculate_objective(data, production, setup, inventory)
    return {
        "objective_value": objective,
        "production": production,
        "setup": setup,
        "inventory": inventory,
    }


def make_initial_inventory_solution(data):
    """A capacity-independent feasible solution using only initial inventory."""
    num_items = data["num_items"]
    num_periods = data["num_periods"]
    demand = data["demand"]

    production = [
        [0.0 for _ in range(num_periods)]
        for _ in range(num_items)
    ]
    setup = [
        [0 for _ in range(num_periods)]
        for _ in range(num_items)
    ]
    inventory = []

    for i in range(num_items):
        initial_stock = float(sum(demand[i]))
        row = [initial_stock]
        current = initial_stock
        for t in range(num_periods):
            current -= float(demand[i][t])
            if abs(current) <= 1e-10:
                current = 0.0
            row.append(current)
        inventory.append(row)

    return make_solution(data, production, setup, inventory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        data = json.load(instance_file)

    num_items = int(data["num_items"])
    num_periods = int(data["num_periods"])

    demand = data["demand"]
    setup_cost = data["setup_cost"]
    holding_cost = data["holding_cost"]
    variable_cost = data["variable_cost"]
    setup_time = data["setup_time"]
    variable_time = data["variable_time"]
    capacity = data["capacity"]
    initial_inventory_cost = data["initial_inventory_cost"]

    # This solution guarantees that a feasible result is available even if the
    # optimizer reaches the time limit before finding an incumbent.
    best_solution = make_initial_inventory_solution(data)
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    remaining_time = float(args.time_limit) - (time.monotonic() - start_time)

    if remaining_time > 0.01:
        try:
            model = gp.Model("capacitated_multi_item_lot_sizing")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining_time)

            # remaining_demand[i][t] is demand from period t through the end.
            remaining_demand = [
                [0.0 for _ in range(num_periods + 1)]
                for _ in range(num_items)
            ]
            for i in range(num_items):
                for t in range(num_periods - 1, -1, -1):
                    remaining_demand[i][t] = (
                        remaining_demand[i][t + 1] + float(demand[i][t])
                    )

            production_ub = {}
            for i in range(num_items):
                vt = float(variable_time[i])
                st = float(setup_time[i])

                for t in range(num_periods):
                    cap = float(capacity[t])
                    if st > cap + 1e-12:
                        capacity_bound = 0.0
                    elif vt > 0.0:
                        capacity_bound = max(0.0, (cap - st) / vt)
                    else:
                        capacity_bound = remaining_demand[i][t]

                    production_ub[i, t] = min(
                        remaining_demand[i][t],
                        capacity_bound,
                    )

            inventory_ub = {
                (i, t): remaining_demand[i][t]
                for i in range(num_items)
                for t in range(num_periods + 1)
            }

            production = model.addVars(
                num_items,
                num_periods,
                lb=0.0,
                ub=production_ub,
                vtype=GRB.CONTINUOUS,
                name="production",
            )
            setup = model.addVars(
                num_items,
                num_periods,
                vtype=GRB.BINARY,
                name="setup",
            )
            inventory = model.addVars(
                num_items,
                num_periods + 1,
                lb=0.0,
                ub=inventory_ub,
                vtype=GRB.CONTINUOUS,
                name="inventory",
            )

            for i in range(num_items):
                for t in range(num_periods):
                    model.addConstr(
                        inventory[i, t] + production[i, t]
                        == float(demand[i][t]) + inventory[i, t + 1],
                        name=f"balance_{i}_{t}",
                    )
                    model.addConstr(
                        production[i, t]
                        <= production_ub[i, t] * setup[i, t],
                        name=f"setup_link_{i}_{t}",
                    )

                model.addConstr(
                    inventory[i, num_periods] == 0.0,
                    name=f"final_inventory_{i}",
                )

            for t in range(num_periods):
                model.addConstr(
                    gp.quicksum(
                        float(setup_time[i]) * setup[i, t]
                        + float(variable_time[i]) * production[i, t]
                        for i in range(num_items)
                    )
                    <= float(capacity[t]),
                    name=f"capacity_{t}",
                )

            objective = gp.quicksum(
                float(initial_inventory_cost[i]) * inventory[i, 0]
                for i in range(num_items)
            )
            objective += gp.quicksum(
                float(setup_cost[i]) * setup[i, t]
                + float(variable_cost[i]) * production[i, t]
                + float(holding_cost[i]) * inventory[i, t + 1]
                for i in range(num_items)
                for t in range(num_periods)
            )
            model.setObjective(objective, GRB.MINIMIZE)

            # Warm start from the always-feasible initial-inventory solution.
            for i in range(num_items):
                for t in range(num_periods):
                    production[i, t].Start = 0.0
                    setup[i, t].Start = 0.0
                for t in range(num_periods + 1):
                    inventory[i, t].Start = best_solution["inventory"][i][t]

            model.update()

            production_vars = [
                production[i, t]
                for i in range(num_items)
                for t in range(num_periods)
            ]
            setup_vars = [
                setup[i, t]
                for i in range(num_items)
                for t in range(num_periods)
            ]
            inventory_vars = [
                inventory[i, t]
                for i in range(num_items)
                for t in range(num_periods + 1)
            ]

            incumbent_state = {
                "objective": best_objective,
                "solution": best_solution,
            }

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                raw_objective = float(
                    cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                )
                if raw_objective >= incumbent_state["objective"] - 1e-9:
                    return

                production_values = cb_model.cbGetSolution(production_vars)
                setup_values = cb_model.cbGetSolution(setup_vars)
                inventory_values = cb_model.cbGetSolution(inventory_vars)

                candidate_production = []
                offset = 0
                for _i in range(num_items):
                    candidate_production.append(
                        production_values[offset:offset + num_periods]
                    )
                    offset += num_periods

                candidate_setup = []
                offset = 0
                for _i in range(num_items):
                    candidate_setup.append(
                        setup_values[offset:offset + num_periods]
                    )
                    offset += num_periods

                candidate_inventory = []
                offset = 0
                inventory_width = num_periods + 1
                for _i in range(num_items):
                    candidate_inventory.append(
                        inventory_values[offset:offset + inventory_width]
                    )
                    offset += inventory_width

                candidate = make_solution(
                    data,
                    candidate_production,
                    candidate_setup,
                    candidate_inventory,
                )
                candidate_objective = candidate["objective_value"]

                if candidate_objective < incumbent_state["objective"] - 1e-9:
                    incumbent_state["objective"] = candidate_objective
                    incumbent_state["solution"] = candidate
                    if logger:
                        logger.log_solution(candidate_objective, candidate)

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                final_production = [
                    [production[i, t].X for t in range(num_periods)]
                    for i in range(num_items)
                ]
                final_setup = [
                    [setup[i, t].X for t in range(num_periods)]
                    for i in range(num_items)
                ]
                final_inventory = [
                    [inventory[i, t].X for t in range(num_periods + 1)]
                    for i in range(num_items)
                ]

                final_candidate = make_solution(
                    data,
                    final_production,
                    final_setup,
                    final_inventory,
                )

                if (
                    final_candidate["objective_value"]
                    < incumbent_state["objective"] - 1e-9
                ):
                    incumbent_state["objective"] = final_candidate["objective_value"]
                    incumbent_state["solution"] = final_candidate
                    if logger:
                        logger.log_solution(
                            final_candidate["objective_value"],
                            final_candidate,
                        )

            best_solution = incumbent_state["solution"]
            best_objective = incumbent_state["objective"]

        except gp.GurobiError:
            # Retain the guaranteed feasible fallback if Gurobi cannot run.
            pass

    best_solution["objective_value"] = float(best_objective)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as solution_file:
        json.dump(best_solution, solution_file, separators=(",", ":"))


if __name__ == "__main__":
    main()