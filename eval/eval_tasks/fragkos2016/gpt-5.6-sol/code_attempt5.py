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
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    n_items = int(data["n_items"])
    n_periods = int(data["n_periods"])

    capacity = [float(value) for value in data["capacity"]]
    initial_inventory = [float(value) for value in data["initial_inventory"]]

    demands = [[float(value) for value in row] for row in data["demands"]]
    setup_costs = [[float(value) for value in row] for row in data["setup_costs"]]
    holding_costs = [[float(value) for value in row] for row in data["holding_costs"]]
    variable_costs = [[float(value) for value in row] for row in data["variable_costs"]]
    setup_times = [[float(value) for value in row] for row in data["setup_times"]]
    variable_times = [[float(value) for value in row] for row in data["variable_times"]]
    instance_big_m = [[float(value) for value in row] for row in data["big_M"]]

    # A remaining-demand bound is valid for nonnegative inventory and zero
    # terminal inventory. Taking the maximum with the supplied bound prevents
    # accidentally cutting off feasible solutions if the supplied values have
    # been rounded down.
    big_m = [[0.0] * n_periods for _ in range(n_items)]
    for i in range(n_items):
        remaining_demand = 0.0
        initial_backlog = max(0.0, -initial_inventory[i])

        for t in range(n_periods - 1, -1, -1):
            remaining_demand += demands[i][t]
            big_m[i][t] = max(
                0.0,
                instance_big_m[i][t],
                remaining_demand + initial_backlog,
            )

    model = gp.Model("capacitated_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.TimeLimit = max(
        0.01,
        float(args.time_limit) - (time.monotonic() - start_time),
    )

    x = model.addVars(
        n_items,
        n_periods,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x",
    )
    y = model.addVars(
        n_items,
        n_periods,
        vtype=GRB.BINARY,
        name="y",
    )
    s = model.addVars(
        n_items,
        n_periods + 1,
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="s",
    )

    # Fix initial inventories. Inventory after every period, including the
    # terminal period, is constrained separately to be nonnegative.
    for i in range(n_items):
        model.addConstr(
            s[i, 0] == initial_inventory[i],
            name=f"initial_inventory_{i}",
        )

        for t in range(1, n_periods + 1):
            s[i, t].LB = 0.0

        model.addConstr(
            s[i, n_periods] == 0.0,
            name=f"terminal_inventory_{i}",
        )

    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i, t] + x[i, t]
                == demands[i][t] + s[i, t + 1],
                name=f"inventory_balance_{i}_{t}",
            )
            model.addConstr(
                x[i, t] <= big_m[i][t] * y[i, t],
                name=f"setup_link_{i}_{t}",
            )

    for t in range(n_periods):
        model.addConstr(
            gp.quicksum(
                setup_times[i][t] * y[i, t]
                + variable_times[i][t] * x[i, t]
                for i in range(n_items)
            )
            <= capacity[t],
            name=f"capacity_{t}",
        )

    model.setObjective(
        gp.quicksum(
            setup_costs[i][t] * y[i, t]
            + variable_costs[i][t] * x[i, t]
            + holding_costs[i][t] * s[i, t + 1]
            for i in range(n_items)
            for t in range(n_periods)
        ),
        GRB.MINIMIZE,
    )

    model.update()

    x_variables = [
        x[i, t]
        for i in range(n_items)
        for t in range(n_periods)
    ]
    s_variables = [
        s[i, t]
        for i in range(n_items)
        for t in range(n_periods + 1)
    ]
    y_variables = [
        y[i, t]
        for i in range(n_items)
        for t in range(n_periods)
    ]

    def make_solution(x_values, y_values):
        output_x = [[0.0] * n_periods for _ in range(n_items)]
        output_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
        output_y = [[0] * n_periods for _ in range(n_items)]

        for i in range(n_items):
            output_s[i][0] = float(initial_inventory[i])
            inventory = float(initial_inventory[i])

            for t in range(n_periods):
                production = float(x_values[i][t])
                if abs(production) <= 1e-8:
                    production = 0.0
                production = max(0.0, production)

                setup = 1 if float(y_values[i][t]) >= 0.5 else 0
                if production > 1e-7:
                    setup = 1

                output_x[i][t] = production
                output_y[i][t] = setup

                inventory += production - demands[i][t]
                if abs(inventory) <= 1e-7:
                    inventory = 0.0
                output_s[i][t + 1] = float(inventory)

        objective = 0.0
        for i in range(n_items):
            for t in range(n_periods):
                objective += setup_costs[i][t] * output_y[i][t]
                objective += variable_costs[i][t] * output_x[i][t]
                objective += holding_costs[i][t] * output_s[i][t + 1]

        return {
            "objective_value": float(objective),
            "x": output_x,
            "s": output_s,
            "y": output_y,
        }

    best_solution = None
    best_objective = math.inf

    def solution_is_feasible(solution, tolerance=1e-5):
        solution_x = solution["x"]
        solution_s = solution["s"]
        solution_y = solution["y"]

        for i in range(n_items):
            if abs(solution_s[i][0] - initial_inventory[i]) > tolerance:
                return False
            if abs(solution_s[i][n_periods]) > tolerance:
                return False

            for t in range(n_periods):
                if solution_x[i][t] < -tolerance:
                    return False
                if solution_s[i][t + 1] < -tolerance:
                    return False
                if solution_y[i][t] not in (0, 1):
                    return False
                if (
                    solution_x[i][t]
                    > big_m[i][t] * solution_y[i][t] + tolerance
                ):
                    return False

                residual = (
                    solution_s[i][t]
                    + solution_x[i][t]
                    - demands[i][t]
                    - solution_s[i][t + 1]
                )
                if abs(residual) > tolerance:
                    return False

        for t in range(n_periods):
            usage = sum(
                setup_times[i][t] * solution_y[i][t]
                + variable_times[i][t] * solution_x[i][t]
                for i in range(n_items)
            )
            if usage > capacity[t] + tolerance:
                return False

        return True

    def save_incumbent(solution):
        nonlocal best_solution, best_objective

        if not solution_is_feasible(solution):
            return

        objective = float(solution["objective_value"])
        tolerance = 1e-8 * max(1.0, abs(objective))

        if objective < best_objective - tolerance:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(objective, solution)

    callback_best = math.inf

    def callback(callback_model, where):
        nonlocal callback_best

        if time.monotonic() - start_time >= args.time_limit:
            callback_model.terminate()
            return

        if where != GRB.Callback.MIPSOL:
            return

        try:
            incumbent_objective = float(
                callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            tolerance = 1e-8 * max(1.0, abs(incumbent_objective))

            if incumbent_objective >= callback_best - tolerance:
                return

            flat_x = callback_model.cbGetSolution(x_variables)
            flat_y = callback_model.cbGetSolution(y_variables)

            x_values = [[0.0] * n_periods for _ in range(n_items)]
            y_values = [[0.0] * n_periods for _ in range(n_items)]

            position = 0
            for i in range(n_items):
                for t in range(n_periods):
                    x_values[i][t] = float(flat_x[position])
                    position += 1

            position = 0
            for i in range(n_items):
                for t in range(n_periods):
                    y_values[i][t] = float(flat_y[position])
                    position += 1

            solution = make_solution(x_values, y_values)
            if solution_is_feasible(solution):
                callback_best = solution["objective_value"]
                if logger:
                    logger.log_solution(
                        solution["objective_value"],
                        solution,
                    )
        except Exception:
            # Logging must not terminate optimization.
            pass

    model.optimize(callback)

    if model.SolCount > 0:
        final_x = [
            [float(x[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]
        final_y = [
            [float(y[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]
        save_incumbent(make_solution(final_x, final_y))

    if best_solution is None:
        status_text = {
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible or unbounded",
            GRB.TIME_LIMIT: "time limit reached before a feasible solution",
            GRB.INTERRUPTED: "interrupted before a feasible solution",
        }.get(model.Status, f"solver status {model.Status}")
        raise RuntimeError(f"No feasible solution found: {status_text}")

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(
            best_solution,
            file,
            separators=(",", ":"),
            allow_nan=False,
        )


if __name__ == "__main__":
    main()