import argparse
import json
import math
import os

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
    supplied_big_m = [[float(v) for v in row] for row in data["big_M"]]

    # Use the supplied valid big-M values, tightening them only with bounds
    # that remain valid regardless of the initial-inventory allocation.
    big_m = [[0.0] * n_periods for _ in range(n_items)]
    for i in range(n_items):
        remaining_demand = [0.0] * n_periods
        running = 0.0
        for t in range(n_periods - 1, -1, -1):
            running += demands[i][t]
            remaining_demand[t] = running

        for t in range(n_periods):
            m = max(0.0, supplied_big_m[i][t])
            m = min(m, max(0.0, remaining_demand[t]))

            if setup_times[i][t] > capacity[t] + 1e-9:
                m = 0.0
            elif variable_times[i][t] > 0.0:
                capacity_bound = (
                    capacity[t] - max(0.0, setup_times[i][t])
                ) / variable_times[i][t]
                m = min(m, max(0.0, capacity_bound))

            big_m[i][t] = max(0.0, m)

    def calculate_objective(x_values, s_values, y_values):
        objective = 0.0
        for i in range(n_items):
            for t in range(n_periods):
                objective += setup_costs[i][t] * y_values[i][t]
                objective += variable_costs[i][t] * x_values[i][t]
                objective += holding_costs[i][t] * s_values[i][t + 1]
        return float(objective)

    def make_solution(x_values, s_values, y_values):
        clean_x = [[0.0] * n_periods for _ in range(n_items)]
        clean_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
        clean_y = [[0] * n_periods for _ in range(n_items)]

        for i in range(n_items):
            clean_s[i][0] = float(initial_inventory[i])

            for t in range(n_periods):
                xv = float(x_values[i][t])
                if abs(xv) <= 1e-7:
                    xv = 0.0
                clean_x[i][t] = max(0.0, xv)
                clean_y[i][t] = 1 if float(y_values[i][t]) >= 0.5 else 0

            for k in range(1, n_periods):
                sv = float(s_values[i][k])
                if abs(sv) <= 1e-7:
                    sv = 0.0
                clean_s[i][k] = max(0.0, sv)

            clean_s[i][n_periods] = 0.0

        objective = calculate_objective(clean_x, clean_s, clean_y)
        return {
            "objective_value": objective,
            "x": clean_x,
            "s": clean_s,
            "y": clean_y,
        }

    def is_feasible_solution(solution, tolerance=1e-5):
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
                if s_values[i][t] < -tolerance:
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
            usage = sum(
                setup_times[i][t] * y_values[i][t]
                + variable_times[i][t] * x_values[i][t]
                for i in range(n_items)
            )
            if usage > capacity[t] + tolerance:
                return False

        return True

    best_solution = None
    best_objective = math.inf

    def accept_solution(solution):
        nonlocal best_solution, best_objective

        objective = float(solution["objective_value"])
        tolerance = 1e-8 * max(1.0, abs(objective))
        if (
            is_feasible_solution(solution)
            and objective < best_objective - tolerance
        ):
            best_solution = solution
            best_objective = objective
            if logger:
                logger.log_solution(objective, solution)

    # A just-in-time solution is a useful warm start on instances where it
    # satisfies capacity.
    jit_x = [[0.0] * n_periods for _ in range(n_items)]
    jit_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
    jit_y = [[0] * n_periods for _ in range(n_items)]

    for i in range(n_items):
        inventory = initial_inventory[i]
        jit_s[i][0] = inventory

        for t in range(n_periods):
            used_inventory = min(inventory, demands[i][t])
            inventory -= used_inventory
            production = demands[i][t] - used_inventory

            jit_x[i][t] = production
            jit_y[i][t] = 1 if production > 1e-9 else 0
            jit_s[i][t + 1] = inventory

    jit_solution = make_solution(jit_x, jit_s, jit_y)
    if is_feasible_solution(jit_solution):
        accept_solution(jit_solution)

    model = gp.Model("capacitated_lot_sizing")

    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, float(args.time_limit))
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.5

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
                lb=0.0,
                ub=0.0 if big_m[i][t] <= 1e-12 else 1.0,
                vtype=GRB.BINARY,
                name=f"y_{i}_{t}",
            )

        for k in range(n_periods + 1):
            if k == 0:
                lower = initial_inventory[i]
                upper = initial_inventory[i]
            elif k == n_periods:
                lower = 0.0
                upper = 0.0
            else:
                lower = 0.0
                upper = sum(demands[i][k:])

            s[i, k] = model.addVar(
                lb=lower,
                ub=upper,
                vtype=GRB.CONTINUOUS,
                name=f"s_{i}_{k}",
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

    objective_expression = gp.quicksum(
        setup_costs[i][t] * y[i, t]
        + variable_costs[i][t] * x[i, t]
        + holding_costs[i][t] * s[i, t + 1]
        for i in range(n_items)
        for t in range(n_periods)
    )
    model.setObjective(objective_expression, GRB.MINIMIZE)

    if best_solution is not None:
        for i in range(n_items):
            for t in range(n_periods):
                x[i, t].Start = best_solution["x"][i][t]
                y[i, t].Start = best_solution["y"][i][t]
            for k in range(n_periods + 1):
                s[i, k].Start = best_solution["s"][i][k]

    x_variables = [
        x[i, t] for i in range(n_items) for t in range(n_periods)
    ]
    s_variables = [
        s[i, k] for i in range(n_items) for k in range(n_periods + 1)
    ]
    y_variables = [
        y[i, t] for i in range(n_items) for t in range(n_periods)
    ]

    logged_callback_objective = best_objective

    def arrays_from_flat(x_flat, s_flat, y_flat):
        x_values = [[0.0] * n_periods for _ in range(n_items)]
        s_values = [[0.0] * (n_periods + 1) for _ in range(n_items)]
        y_values = [[0.0] * n_periods for _ in range(n_items)]

        index = 0
        for i in range(n_items):
            for t in range(n_periods):
                x_values[i][t] = x_flat[index]
                index += 1

        index = 0
        for i in range(n_items):
            for k in range(n_periods + 1):
                s_values[i][k] = s_flat[index]
                index += 1

        index = 0
        for i in range(n_items):
            for t in range(n_periods):
                y_values[i][t] = y_flat[index]
                index += 1

        return x_values, s_values, y_values

    def callback(cb_model, where):
        nonlocal logged_callback_objective

        if logger is None or where != GRB.Callback.MIPSOL:
            return

        try:
            incumbent_objective = float(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            tolerance = 1e-8 * max(1.0, abs(incumbent_objective))
            if incumbent_objective >= logged_callback_objective - tolerance:
                return

            x_flat = cb_model.cbGetSolution(x_variables)
            s_flat = cb_model.cbGetSolution(s_variables)
            y_flat = cb_model.cbGetSolution(y_variables)

            x_values, s_values, y_values = arrays_from_flat(
                x_flat, s_flat, y_flat
            )
            solution = make_solution(x_values, s_values, y_values)

            if is_feasible_solution(solution):
                logger.log_solution(
                    solution["objective_value"], solution
                )
                logged_callback_objective = incumbent_objective
        except Exception:
            # Logging must not interrupt optimization.
            pass

    model.optimize(callback if logger else None)

    if model.SolCount > 0:
        model_x = [
            [float(x[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]
        model_s = [
            [float(s[i, k].X) for k in range(n_periods + 1)]
            for i in range(n_items)
        ]
        model_y = [
            [float(y[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]

        solver_solution = make_solution(model_x, model_s, model_y)
        accept_solution(solver_solution)

    if best_solution is None:
        if model.Status == GRB.INFEASIBLE:
            raise RuntimeError("The input instance is infeasible.")
        raise RuntimeError(
            "No feasible solution was found within the specified time limit."
        )

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