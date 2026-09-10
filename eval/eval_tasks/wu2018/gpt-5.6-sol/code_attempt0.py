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


def clean_continuous(value):
    value = float(value)
    if abs(value) <= 1e-8:
        return 0.0
    if value < 0.0 and value >= -1e-6:
        return 0.0
    return value


def read_initial_inventory(data, num_items):
    raw = data.get("initial_inventory", data.get("initial_inventory_level"))
    if raw is None:
        return [0.0] * num_items

    if isinstance(raw, list):
        if len(raw) != num_items:
            raise ValueError("Initial inventory must have one value per item.")
        return [float(v) for v in raw]

    if isinstance(raw, dict):
        result = []
        for i in range(1, num_items + 1):
            value = raw.get(f"item_{i}", raw.get(str(i), 0.0))
            result.append(float(value))
        return result

    raise ValueError("Unsupported initial inventory format.")


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_items = int(data["num_items"])
    num_periods = int(data["num_periods"])
    num_machines = int(data["num_machines"])

    items = range(1, num_items + 1)
    periods = range(1, num_periods + 1)
    machines = range(1, num_machines + 1)

    demand = {
        (i, t): float(data["demand"][i - 1][t - 1])
        for i in items
        for t in periods
    }
    capacity = {
        (m, t): float(data["capacity"][m - 1][t - 1])
        for m in machines
        for t in periods
    }
    holding_cost = {
        i: float(data["holding_cost"][i - 1])
        for i in items
    }
    initial_inventory_values = read_initial_inventory(data, num_items)
    initial_inventory = {
        i: initial_inventory_values[i - 1]
        for i in items
    }
    max_machines = {
        i: int(data["max_machines_per_item_per_period"][i - 1])
        for i in items
    }

    capabilities = {}
    capable_pairs = []
    pairs_by_machine = {m: [] for m in machines}

    for i in items:
        machine_list = sorted(set(
            int(m) for m in data["machine_capabilities"][i - 1]
        ))
        for m in machine_list:
            if m < 1 or m > num_machines:
                raise ValueError(f"Invalid machine {m} in capabilities of item {i}.")
        capabilities[i] = machine_list

        for m in machine_list:
            capable_pairs.append((i, m))
            pairs_by_machine[m].append((i, m))

    production_time = {}
    production_cost = {}
    setup_time = {}
    setup_cost = {}

    for i, m in capable_pairs:
        key = f"item_{i}_machine_{m}"
        production_time[i, m] = float(data["production_time"][key])
        production_cost[i, m] = float(data["production_cost"][key])
        setup_time[i, m] = float(data["setup_time"][key])
        setup_cost[i, m] = float(data["setup_cost"][key])

    remaining_demand = {}
    for i in items:
        running = 0.0
        for t in range(num_periods, 0, -1):
            running += demand[i, t]
            remaining_demand[i, t] = running

    model = gp.Model("multi_item_multi_machine_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.monotonic() - start_time
    remaining_time = max(0.0, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    production = {}
    setups = {}
    inventory = {}
    production_upper_bound = {}

    for i, m in capable_pairs:
        ptime = production_time[i, m]
        stime = setup_time[i, m]

        for t in periods:
            demand_bound = max(0.0, remaining_demand[i, t])
            available_after_setup = capacity[m, t] - stime

            if available_after_setup < -1e-9:
                capacity_bound = 0.0
            elif ptime > 0.0:
                capacity_bound = max(0.0, available_after_setup / ptime)
            else:
                capacity_bound = demand_bound

            upper_bound = max(0.0, min(demand_bound, capacity_bound))
            production_upper_bound[i, m, t] = upper_bound

            production[i, m, t] = model.addVar(
                lb=0.0,
                ub=upper_bound,
                vtype=GRB.CONTINUOUS,
                obj=production_cost[i, m],
                name=f"X_item{i}_machine{m}_period{t}",
            )
            setups[i, m, t] = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=GRB.BINARY,
                obj=setup_cost[i, m],
                name=f"Y_item{i}_machine{m}_period{t}",
            )

    for i in items:
        for t in periods:
            inventory[i, t] = model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                obj=holding_cost[i],
                name=f"I_item{i}_period{t}",
            )

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # Production/setup linking constraints.
    for i, m in capable_pairs:
        for t in periods:
            model.addConstr(
                production[i, m, t]
                <= production_upper_bound[i, m, t] * setups[i, m, t],
                name=f"link_item{i}_machine{m}_period{t}",
            )

    # Machine capacity constraints.
    for m in machines:
        for t in periods:
            model.addConstr(
                gp.quicksum(
                    production_time[i, m] * production[i, m, t]
                    + setup_time[i, m] * setups[i, m, t]
                    for i, _ in pairs_by_machine[m]
                )
                <= capacity[m, t],
                name=f"capacity_machine{m}_period{t}",
            )

    # Inventory balance constraints.
    for i in items:
        for t in periods:
            previous_inventory = (
                initial_inventory[i] if t == 1 else inventory[i, t - 1]
            )
            model.addConstr(
                gp.quicksum(
                    production[i, m, t] for m in capabilities[i]
                )
                + previous_inventory
                - inventory[i, t]
                == demand[i, t],
                name=f"balance_item{i}_period{t}",
            )

    # Limit the number of machines simultaneously assigned to each item.
    for i in items:
        for t in periods:
            model.addConstr(
                gp.quicksum(
                    setups[i, m, t] for m in capabilities[i]
                )
                <= max_machines[i],
                name=f"machine_limit_item{i}_period{t}",
            )

    production_keys = [
        (i, m, t)
        for i, m in capable_pairs
        for t in periods
    ]
    inventory_keys = [
        (i, t)
        for i in items
        for t in periods
    ]
    setup_keys = list(production_keys)

    production_vars = [production[key] for key in production_keys]
    inventory_vars = [inventory[key] for key in inventory_keys]
    setup_vars = [setups[key] for key in setup_keys]

    def make_solution(objective_value, x_values, i_values, y_values):
        production_output = {
            f"X_item{i}_machine{m}_period{t}": clean_continuous(value)
            for (i, m, t), value in zip(production_keys, x_values)
        }
        inventory_output = {
            f"I_item{i}_period{t}": clean_continuous(value)
            for (i, t), value in zip(inventory_keys, i_values)
        }
        setup_output = {
            f"Y_item{i}_machine{m}_period{t}": int(float(value) >= 0.5)
            for (i, m, t), value in zip(setup_keys, y_values)
        }

        objective_value = float(objective_value)
        if abs(objective_value) <= 1e-9:
            objective_value = 0.0

        return {
            "objective_value": objective_value,
            "production": production_output,
            "inventory": inventory_output,
            "setups": setup_output,
        }

    best_logged = [math.inf]

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return

        try:
            objective_value = float(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            tolerance = 1e-7 * max(1.0, abs(objective_value))
            if objective_value >= best_logged[0] - tolerance:
                return

            x_values = cb_model.cbGetSolution(production_vars)
            i_values = cb_model.cbGetSolution(inventory_vars)
            y_values = cb_model.cbGetSolution(setup_vars)

            solution = make_solution(
                objective_value, x_values, i_values, y_values
            )
            logger.log_solution(objective_value, solution)
            best_logged[0] = objective_value
        except Exception:
            # Logging must not interrupt the optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    final_objective = float(model.ObjVal)
    final_x_values = [var.X for var in production_vars]
    final_i_values = [var.X for var in inventory_vars]
    final_y_values = [var.X for var in setup_vars]

    final_solution = make_solution(
        final_objective,
        final_x_values,
        final_i_values,
        final_y_values,
    )

    if logger is not None:
        tolerance = 1e-7 * max(1.0, abs(final_objective))
        if final_objective < best_logged[0] - tolerance:
            try:
                logger.log_solution(final_objective, final_solution)
                best_logged[0] = final_objective
            except Exception:
                pass

    solution_directory = os.path.dirname(
        os.path.abspath(args.solution_path)
    )
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()