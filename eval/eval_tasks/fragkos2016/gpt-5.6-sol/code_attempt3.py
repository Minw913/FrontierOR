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

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_items = int(data["n_items"])
    n_periods = int(data["n_periods"])

    capacity = [float(v) for v in data["capacity"]]
    initial_inventory = [float(v) for v in data["initial_inventory"]]
    demands = [[float(v) for v in row] for row in data["demands"]]
    setup_costs = [[float(v) for v in row] for row in data["setup_costs"]]
    holding_costs = [[float(v) for v in row] for row in data["holding_costs"]]
    variable_costs = [[float(v) for v in row] for row in data["variable_costs"]]
    setup_times = [[float(v) for v in row] for row in data["setup_times"]]
    variable_times = [[float(v) for v in row] for row in data["variable_times"]]

    # The supplied bounds account for both capacity and remaining requirements.
    # In particular, they may account for an initial backlog represented by a
    # negative initial inventory, so they must not be replaced by plain
    # remaining-demand bounds.
    supplied_big_m = [[float(v) for v in row] for row in data["big_M"]]

    big_m = [[0.0] * n_periods for _ in range(n_items)]
    for i in range(n_items):
        for t in range(n_periods):
            big_m[i][t] = max(0.0, supplied_big_m[i][t])

    def calculate_objective(x_values, s_values, y_values):
        return float(
            sum(
                setup_costs[i][t] * y_values[i][t]
                + variable_costs[i][t] * x_values[i][t]
                + holding_costs[i][t] * s_values[i][t + 1]
                for i in range(n_items)
                for t in range(n_periods)
            )
        )

    def make_solution(x_values, s_values, y_values):
        result_x = [[0.0] * n_periods for _ in range(n_items)]
        result_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
        result_y = [[0] * n_periods for _ in range(n_items)]

        for i in range(n_items):
            result_s[i][0] = float(initial_inventory[i])

            for t in range(n_periods):
                xv = float(x_values[i][t])
                if abs(xv) < 1e-9:
                    xv = 0.0
                result_x[i][t] = max(0.0, xv)
                result_y[i][t] = 1 if float(y_values[i][t]) >= 0.5 else 0

            for k in range(1, n_periods):
                sv = float(s_values[i][k])
                if abs(sv) < 1e-9:
                    sv = 0.0
                result_s[i][k] = max(0.0, sv)

            result_s[i][n_periods] = 0.0

        objective = calculate_objective(result_x, result_s, result_y)
        return {
            "objective_value": objective,
            "x": result_x,
            "s": result_s,
            "y": result_y,
        }

    def is_feasible(solution, tolerance=2e-5):
        x_values = solution["x"]
        s_values = solution["s"]
        y_values = solution["y"]

        for i in range(n_items):
            if abs(s_values[i][0] - initial_inventory[i]) > tolerance:
                return False
            if abs(s_values[i][n_periods]) > tolerance:
                return False

            for t in range(n_periods):
                if x_values[i][t] < -tolerance:
                    return False

                # s[i][0] is exogenous initial inventory and may represent an
                # initial backlog. All inventories after production starts are
                # constrained to be nonnegative.
                if t > 0 and s_values[i][t] < -tolerance:
                    return False
                if s_values[i][t + 1] < -tolerance:
                    return False

                if y_values[i][t] not in (0, 1):
                    return False

                if x_values[i][t] > big_m[i][t] * y_values[i][t] + tolerance:
                    return False

                balance = (
                    s_values[i][t]
                    + x_values[i][t]
                    - demands[i][t]
                    - s_values[i][t + 1]
                )
                if abs(balance) > tolerance:
                    return False

        for t in range(n_periods):
            used = sum(
                setup_times[i][t] * y_values[i][t]
                + variable_times[i][t] * x_values[i][t]
                for i in range(n_items)
            )
            if used > capacity[t] + tolerance:
                return False

        return True

    best_solution = None
    best_objective = math.inf

    def record_solution(solution):
        nonlocal best_solution, best_objective

        objective = float(solution["objective_value"])
        tolerance = 1e-8 * max(1.0, abs(objective))

        if is_feasible(solution) and objective < best_objective - tolerance:
            best_solution = solution
            best_objective = objective
            if logger:
                logger.log_solution(objective, solution)

    # Try a just-in-time initial solution. Negative initial inventory is treated
    # as initial backlog and is covered in the first period.
    warm_x = [[0.0] * n_periods for _ in range(n_items)]
    warm_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
    warm_y = [[0] * n_periods for _ in range(n_items)]

    for i in range(n_items):
        inventory = initial_inventory[i]
        warm_s[i][0] = inventory

        for t in range(n_periods):
            required_production = max(0.0, demands[i][t] - inventory)
            ending_inventory = inventory + required_production - demands[i][t]

            warm_x[i][t] = required_production
            warm_y[i][t] = 1 if required_production > 1e-9 else 0
            warm_s[i][t + 1] = max(0.0, ending_inventory)
            inventory = warm_s[i][t + 1]

    warm_solution = make_solution(warm_x, warm_s, warm_y)
    record_solution(warm_solution)

    model = gp.Model("capacitated_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.01, float(args.time_limit) - elapsed)

    x = {}
    s = {}
    y = {}

    for i in range(n_items):
        for t in range(n_periods):
            x[i, t] = model.addVar(
                lb=0.0,
                ub=big_m[i][t],
                vtype=GRB.CONTINUOUS,
                name=f"x_{i}_{t}",
            )
            y[i, t] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y_{i}_{t}",
            )

        # Initial inventory is fixed to the supplied value. It is exogenous and
        # can therefore be negative if the instance represents initial backlog.
        s[i, 0] = model.addVar(
            lb=initial_inventory[i],
            ub=initial_inventory[i],
            vtype=GRB.CONTINUOUS,
            name=f"s_{i}_0",
        )

        for k in range(1, n_periods):
            s[i, k] = model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"s_{i}_{k}",
            )

        s[i, n_periods] = model.addVar(
            lb=0.0,
            ub=0.0,
            vtype=GRB.CONTINUOUS,
            name=f"s_{i}_{n_periods}",
        )

    model.update()

    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1],
                name=f"balance_{i}_{t}",
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

    if best_solution is not None:
        for i in range(n_items):
            for t in range(n_periods):
                x[i, t].Start = best_solution["x"][i][t]
                y[i, t].Start = best_solution["y"][i][t]
            for k in range(n_periods + 1):
                s[i, k].Start = best_solution["s"][i][k]

    x_vars = [x[i, t] for i in range(n_items) for t in range(n_periods)]
    s_vars = [
        s[i, k]
        for i in range(n_items)
        for k in range(n_periods + 1)
    ]
    y_vars = [y[i, t] for i in range(n_items) for t in range(n_periods)]

    callback_best = best_objective

    def incumbent_callback(cb_model, where):
        nonlocal callback_best

        if logger is None or where != GRB.Callback.MIPSOL:
            return

        try:
            callback_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            tolerance = 1e-8 * max(1.0, abs(callback_obj))
            if callback_obj >= callback_best - tolerance:
                return

            flat_x = cb_model.cbGetSolution(x_vars)
            flat_s = cb_model.cbGetSolution(s_vars)
            flat_y = cb_model.cbGetSolution(y_vars)

            x_values = [[0.0] * n_periods for _ in range(n_items)]
            s_values = [[0.0] * (n_periods + 1) for _ in range(n_items)]
            y_values = [[0.0] * n_periods for _ in range(n_items)]

            index = 0
            for i in range(n_items):
                for t in range(n_periods):
                    x_values[i][t] = flat_x[index]
                    index += 1

            index = 0
            for i in range(n_items):
                for k in range(n_periods + 1):
                    s_values[i][k] = flat_s[index]
                    index += 1

            index = 0
            for i in range(n_items):
                for t in range(n_periods):
                    y_values[i][t] = flat_y[index]
                    index += 1

            solution = make_solution(x_values, s_values, y_values)
            if is_feasible(solution):
                logger.log_solution(solution["objective_value"], solution)
                callback_best = solution["objective_value"]
        except Exception:
            # Incumbent logging must not interrupt optimization.
            pass

    model.optimize(incumbent_callback if logger else None)

    if model.SolCount > 0:
        final_x = [
            [float(x[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]
        final_s = [
            [float(s[i, k].X) for k in range(n_periods + 1)]
            for i in range(n_items)
        ]
        final_y = [
            [float(y[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]
        record_solution(make_solution(final_x, final_s, final_y))

    if best_solution is None:
        status_names = {
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible or unbounded",
            GRB.TIME_LIMIT: "time limit reached before finding a solution",
        }
        status_text = status_names.get(model.Status, f"solver status {model.Status}")
        raise RuntimeError(f"No feasible solution found: {status_text}")

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(
            best_solution,
            f,
            separators=(",", ":"),
            allow_nan=False,
        )


if __name__ == "__main__":
    main()