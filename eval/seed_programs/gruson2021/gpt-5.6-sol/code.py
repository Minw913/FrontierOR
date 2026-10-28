import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def clean_number(value):
    value = float(value)
    if abs(value) <= 1e-8:
        return 0.0
    if -1e-6 < value < 0.0:
        return 0.0
    return value


def ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_empty_solution():
    return {
        "objective_value": 0.0,
        "setup_variables": {},
        "production_plant": {},
        "delivery_warehouse": {},
        "delivery_retailer": {},
        "inventory_plant": {},
        "inventory_warehouse": {},
        "inventory_retailer": {},
    }


def setup_key(facility, period):
    return f"y_{facility}_{period}"


def build_all_setups_fallback(data):
    """
    Construct a feasible just-in-time solution with every setup active.

    Initial inventory, if present, is consumed first at the retailer level,
    then at warehouses, and finally at the plant.
    """
    T = int(data["num_periods"])
    W = int(data["num_warehouses"])
    R = int(data["num_retailers"])
    S = int(data["num_scenarios"])

    assignment = [int(x) for x in data["retailer_to_warehouse"]]
    probabilities = [float(x) for x in data["scenario_probabilities"]]
    demands = data["demands"]
    initial_inventory = float(data.get("initial_inventory", 0))

    hp = float(data["holding_cost_plant"])
    hw = float(data["holding_cost_warehouse"])
    hr = [float(x) for x in data["holding_costs_retailers"]]

    solution = make_empty_solution()

    # Activate every facility in every period.
    for t in range(T):
        solution["setup_variables"][setup_key(0, t)] = 1
        for w in range(W):
            solution["setup_variables"][setup_key(1 + w, t)] = 1
        for r in range(R):
            solution["setup_variables"][setup_key(1 + W + r, t)] = 1

    expected_holding = 0.0

    for s in range(S):
        retail_inventory = [initial_inventory for _ in range(R)]
        warehouse_inventory = [initial_inventory for _ in range(W)]
        plant_inventory = initial_inventory
        scenario_holding = 0.0

        for t in range(T):
            retailer_deliveries = [0.0 for _ in range(R)]

            # Retailer inventory is used before requesting a new delivery.
            for r in range(R):
                demand = float(demands[s][r][t])
                available = retail_inventory[r]
                delivery = max(0.0, demand - available)
                ending_inventory = max(0.0, available - demand)

                retailer_deliveries[r] = delivery
                retail_inventory[r] = ending_inventory

                solution["delivery_retailer"][f"{r}_{t}_{s}"] = delivery
                solution["inventory_retailer"][f"{r}_{t}_{s}"] = ending_inventory
                scenario_holding += hr[r] * ending_inventory

            warehouse_receipts = [0.0 for _ in range(W)]

            # Each warehouse uses its own inventory before ordering upstream.
            for w in range(W):
                outgoing = sum(
                    retailer_deliveries[r]
                    for r in range(R)
                    if assignment[r] == w
                )
                available = warehouse_inventory[w]
                receipt = max(0.0, outgoing - available)
                ending_inventory = max(0.0, available - outgoing)

                warehouse_receipts[w] = receipt
                warehouse_inventory[w] = ending_inventory

                solution["delivery_warehouse"][f"{w}_{t}_{s}"] = receipt
                solution["inventory_warehouse"][f"{w}_{t}_{s}"] = ending_inventory
                scenario_holding += hw * ending_inventory

            # The plant uses its initial/carried inventory before producing.
            plant_outgoing = sum(warehouse_receipts)
            production = max(0.0, plant_outgoing - plant_inventory)
            plant_inventory = max(0.0, plant_inventory - plant_outgoing)

            solution["production_plant"][f"{t}_{s}"] = production
            solution["inventory_plant"][f"{t}_{s}"] = plant_inventory
            scenario_holding += hp * plant_inventory

        expected_holding += probabilities[s] * scenario_holding

    setup_cost = sum(float(x) for x in data["setup_costs_plant"])
    setup_cost += sum(
        float(data["setup_costs_warehouses"][w][t])
        for w in range(W)
        for t in range(T)
    )
    setup_cost += sum(
        float(data["setup_costs_retailers"][r][t])
        for r in range(R)
        for t in range(T)
    )

    solution["objective_value"] = float(setup_cost + expected_holding)
    return solution


def solve(data, time_limit, logger, start_time):
    T = int(data["num_periods"])
    W = int(data["num_warehouses"])
    R = int(data["num_retailers"])
    S = int(data["num_scenarios"])

    assignment = [int(x) for x in data["retailer_to_warehouse"]]
    probabilities = [float(x) for x in data["scenario_probabilities"]]
    demands = [
        [
            [float(data["demands"][s][r][t]) for t in range(T)]
            for r in range(R)
        ]
        for s in range(S)
    ]

    if len(assignment) != R or any(w < 0 or w >= W for w in assignment):
        raise ValueError("retailer_to_warehouse contains an invalid warehouse index")

    initial_inventory = float(data.get("initial_inventory", 0))
    impose_initial_setups = bool(data.get("initial_setups_imposed", False))

    hp = float(data["holding_cost_plant"])
    hw = float(data["holding_cost_warehouse"])
    hr = [float(x) for x in data["holding_costs_retailers"]]

    setup_plant = [float(x) for x in data["setup_costs_plant"]]
    setup_warehouse = [
        [float(data["setup_costs_warehouses"][w][t]) for t in range(T)]
        for w in range(W)
    ]
    setup_retailer = [
        [float(data["setup_costs_retailers"][r][t]) for t in range(T)]
        for r in range(R)
    ]

    retailers_by_warehouse = [
        [r for r in range(R) if assignment[r] == w]
        for w in range(W)
    ]

    # Remaining demand gives tight, scenario-specific setup-linking bounds.
    remaining_retailer = [
        [[0.0 for _ in range(T)] for _ in range(R)]
        for _ in range(S)
    ]
    remaining_warehouse = [
        [[0.0 for _ in range(T)] for _ in range(W)]
        for _ in range(S)
    ]
    remaining_total = [[0.0 for _ in range(T)] for _ in range(S)]

    future_retailer = [
        [[0.0 for _ in range(T)] for _ in range(R)]
        for _ in range(S)
    ]
    future_warehouse = [
        [[0.0 for _ in range(T)] for _ in range(W)]
        for _ in range(S)
    ]
    future_total = [[0.0 for _ in range(T)] for _ in range(S)]

    for s in range(S):
        for r in range(R):
            running = 0.0
            for t in range(T - 1, -1, -1):
                future_retailer[s][r][t] = running
                running += demands[s][r][t]
                remaining_retailer[s][r][t] = running

        for w in range(W):
            for t in range(T):
                remaining_warehouse[s][w][t] = sum(
                    remaining_retailer[s][r][t]
                    for r in retailers_by_warehouse[w]
                )
                future_warehouse[s][w][t] = sum(
                    future_retailer[s][r][t]
                    for r in retailers_by_warehouse[w]
                )

        for t in range(T):
            remaining_total[s][t] = sum(
                remaining_retailer[s][r][t] for r in range(R)
            )
            future_total[s][t] = sum(
                future_retailer[s][r][t] for r in range(R)
            )

    fallback = build_all_setups_fallback(data)
    best_solution = fallback
    best_objective = float(fallback["objective_value"])

    if logger:
        logger.log_solution(best_objective, fallback)

    model = gp.Model("stochastic_three_echelon_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.01, float(time_limit) - elapsed)

    y_plant = {}
    y_warehouse = {}
    y_retailer = {}

    production = {}
    delivery_warehouse = {}
    delivery_retailer = {}

    inventory_plant = {}
    inventory_warehouse = {}
    inventory_retailer = {}

    metadata = []

    # First-stage setup variables.
    for t in range(T):
        fixed = impose_initial_setups and t == 0
        y_plant[t] = model.addVar(
            vtype=GRB.BINARY,
            lb=1.0 if fixed else 0.0,
            ub=1.0,
            obj=setup_plant[t],
            name=f"y_plant_{t}",
        )
        metadata.append(("setup_variables", setup_key(0, t), y_plant[t]))

        for w in range(W):
            y_warehouse[w, t] = model.addVar(
                vtype=GRB.BINARY,
                lb=1.0 if fixed else 0.0,
                ub=1.0,
                obj=setup_warehouse[w][t],
                name=f"y_warehouse_{w}_{t}",
            )
            metadata.append(
                ("setup_variables", setup_key(1 + w, t), y_warehouse[w, t])
            )

        for r in range(R):
            y_retailer[r, t] = model.addVar(
                vtype=GRB.BINARY,
                lb=1.0 if fixed else 0.0,
                ub=1.0,
                obj=setup_retailer[r][t],
                name=f"y_retailer_{r}_{t}",
            )
            metadata.append(
                (
                    "setup_variables",
                    setup_key(1 + W + r, t),
                    y_retailer[r, t],
                )
            )

    # Scenario-dependent recourse variables.
    for s in range(S):
        probability = probabilities[s]

        for t in range(T):
            production[t, s] = model.addVar(
                lb=0.0,
                ub=remaining_total[s][t],
                name=f"production_{t}_{s}",
            )

            plant_inventory_ub = GRB.INFINITY
            if initial_inventory == 0.0:
                plant_inventory_ub = future_total[s][t]

            inventory_plant[t, s] = model.addVar(
                lb=0.0,
                ub=plant_inventory_ub,
                obj=probability * hp,
                name=f"inventory_plant_{t}_{s}",
            )

            metadata.append(
                ("production_plant", f"{t}_{s}", production[t, s])
            )
            metadata.append(
                ("inventory_plant", f"{t}_{s}", inventory_plant[t, s])
            )

            for w in range(W):
                delivery_warehouse[w, t, s] = model.addVar(
                    lb=0.0,
                    ub=remaining_warehouse[s][w][t],
                    name=f"delivery_warehouse_{w}_{t}_{s}",
                )

                warehouse_inventory_ub = GRB.INFINITY
                if initial_inventory == 0.0:
                    warehouse_inventory_ub = future_warehouse[s][w][t]

                inventory_warehouse[w, t, s] = model.addVar(
                    lb=0.0,
                    ub=warehouse_inventory_ub,
                    obj=probability * hw,
                    name=f"inventory_warehouse_{w}_{t}_{s}",
                )

                metadata.append(
                    (
                        "delivery_warehouse",
                        f"{w}_{t}_{s}",
                        delivery_warehouse[w, t, s],
                    )
                )
                metadata.append(
                    (
                        "inventory_warehouse",
                        f"{w}_{t}_{s}",
                        inventory_warehouse[w, t, s],
                    )
                )

            for r in range(R):
                delivery_retailer[r, t, s] = model.addVar(
                    lb=0.0,
                    ub=remaining_retailer[s][r][t],
                    name=f"delivery_retailer_{r}_{t}_{s}",
                )

                retailer_inventory_ub = GRB.INFINITY
                if initial_inventory == 0.0:
                    retailer_inventory_ub = future_retailer[s][r][t]

                inventory_retailer[r, t, s] = model.addVar(
                    lb=0.0,
                    ub=retailer_inventory_ub,
                    obj=probability * hr[r],
                    name=f"inventory_retailer_{r}_{t}_{s}",
                )

                metadata.append(
                    (
                        "delivery_retailer",
                        f"{r}_{t}_{s}",
                        delivery_retailer[r, t, s],
                    )
                )
                metadata.append(
                    (
                        "inventory_retailer",
                        f"{r}_{t}_{s}",
                        inventory_retailer[r, t, s],
                    )
                )

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # Setup-linking constraints and inventory balances.
    for s in range(S):
        for t in range(T):
            model.addConstr(
                production[t, s]
                <= remaining_total[s][t] * y_plant[t],
                name=f"link_plant_{t}_{s}",
            )

            previous_plant_inventory = (
                initial_inventory if t == 0 else inventory_plant[t - 1, s]
            )
            model.addConstr(
                previous_plant_inventory + production[t, s]
                == gp.quicksum(
                    delivery_warehouse[w, t, s] for w in range(W)
                )
                + inventory_plant[t, s],
                name=f"balance_plant_{t}_{s}",
            )

            for w in range(W):
                model.addConstr(
                    delivery_warehouse[w, t, s]
                    <= remaining_warehouse[s][w][t] * y_warehouse[w, t],
                    name=f"link_warehouse_{w}_{t}_{s}",
                )

                previous_warehouse_inventory = (
                    initial_inventory
                    if t == 0
                    else inventory_warehouse[w, t - 1, s]
                )
                model.addConstr(
                    previous_warehouse_inventory
                    + delivery_warehouse[w, t, s]
                    == gp.quicksum(
                        delivery_retailer[r, t, s]
                        for r in retailers_by_warehouse[w]
                    )
                    + inventory_warehouse[w, t, s],
                    name=f"balance_warehouse_{w}_{t}_{s}",
                )

            for r in range(R):
                model.addConstr(
                    delivery_retailer[r, t, s]
                    <= remaining_retailer[s][r][t] * y_retailer[r, t],
                    name=f"link_retailer_{r}_{t}_{s}",
                )

                previous_retailer_inventory = (
                    initial_inventory
                    if t == 0
                    else inventory_retailer[r, t - 1, s]
                )
                model.addConstr(
                    previous_retailer_inventory
                    + delivery_retailer[r, t, s]
                    == demands[s][r][t] + inventory_retailer[r, t, s],
                    name=f"balance_retailer_{r}_{t}_{s}",
                )

    # Warm-start the MIP with the guaranteed feasible all-setups solution.
    for field, key, var in metadata:
        value = fallback[field][key]
        var.Start = float(value)

    all_vars = [entry[2] for entry in metadata]

    def solution_from_values(values, objective):
        result = make_empty_solution()
        result["objective_value"] = float(objective)

        for (field, key, _), value in zip(metadata, values):
            if field == "setup_variables":
                result[field][key] = int(float(value) >= 0.5)
            else:
                result[field][key] = clean_number(value)

        return result

    model._metadata = metadata
    model._all_vars = all_vars
    model._logger = logger
    model._best_logged = best_objective
    model._solution_from_values = solution_from_values

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
        if objective >= cb_model._best_logged - 1e-7:
            return

        cb_model._best_logged = objective
        if cb_model._logger:
            values = cb_model.cbGetSolution(cb_model._all_vars)
            incumbent = cb_model._solution_from_values(values, objective)
            cb_model._logger.log_solution(objective, incumbent)

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        objective = float(model.ObjVal)
        values = model.getAttr("X", all_vars)
        candidate = solution_from_values(values, objective)

        if objective < best_objective - 1e-7:
            best_solution = candidate
            best_objective = objective

            if logger and objective < model._best_logged - 1e-7:
                logger.log_solution(objective, candidate)
                model._best_logged = objective

    return best_solution


def main():
    start_time = time.monotonic()

    parser = argparse.ArgumentParser(
        description="Solve the stochastic three-echelon lot-sizing problem."
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    if args.time_limit <= 0:
        raise ValueError("--time_limit must be a positive integer")

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    solution = solve(
        data=data,
        time_limit=args.time_limit,
        logger=logger,
        start_time=start_time,
    )

    ensure_parent_directory(args.solution_path)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()