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

    # Use a structurally valid production bound rather than tightening the
    # supplied big-M values further. Production in period t can never exceed
    # demand from t through the end because final inventory is fixed to zero.
    # This avoids accidental infeasibility caused by overly restrictive input
    # bounds or numerical tightening at capacity boundaries.
    big_m = [[0.0] * n_periods for _ in range(n_items)]
    for i in range(n_items):
        remaining = 0.0
        for t in range(n_periods - 1, -1, -1):
            remaining += max(0.0, demands[i][t])
            big_m[i][t] = remaining

    def objective_value(x_values, s_values, y_values):
        return float(
            sum(
                setup_costs[i][t] * y_values[i][t]
                + variable_costs[i][t] * x_values[i][t]
                + holding_costs[i][t] * s_values[i][t + 1]
                for i in range(n_items)
                for t in range(n_periods)
            )
        )

    def build_solution(x_values, s_values, y_values):
        out_x = [[0.0] * n_periods for _ in range(n_items)]
        out_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
        out_y = [[0] * n_periods for _ in range(n_items)]

        for i in range(n_items):
            out_s[i][0] = float(initial_inventory[i])

            for t in range(n_periods):
                xv = float(x_values[i][t])
                if abs(xv) <= 1e-8:
                    xv = 0.0
                out_x[i][t] = max(0.0, xv)
                out_y[i][t] = int(float(y_values[i][t]) >= 0.5)

            for k in range(1, n_periods):
                sv = float(s_values[i][k])
                if abs(sv) <= 1e-8:
                    sv = 0.0
                out_s[i][k] = max(0.0, sv)

            out_s[i][n_periods] = 0.0

        obj = objective_value(out_x, out_s, out_y)
        return {
            "objective_value": obj,
            "x": out_x,
            "s": out_s,
            "y": out_y,
        }

    def check_feasibility(solution, tolerance=1e-5):
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
                if y_values[i][t] not in (0, 1):
                    return False
                if x_values[i][t] > big_m[i][t] * y_values[i][t] + tolerance:
                    return False

                balance_error = (
                    s_values[i][t]
                    + x_values[i][t]
                    - demands[i][t]
                    - s_values[i][t + 1]
                )
                if abs(balance_error) > tolerance:
                    return False

        for t in range(n_periods):
            time_used = sum(
                setup_times[i][t] * y_values[i][t]
                + variable_times[i][t] * x_values[i][t]
                for i in range(n_items)
            )
            if time_used > capacity[t] + tolerance:
                return False

        return True

    best_solution = None
    best_objective = math.inf

    def record_solution(solution):
        nonlocal best_solution, best_objective

        obj = float(solution["objective_value"])
        improvement_tolerance = 1e-8 * max(1.0, abs(obj))

        if (
            check_feasibility(solution)
            and obj < best_objective - improvement_tolerance
        ):
            best_solution = solution
            best_objective = obj
            if logger:
                logger.log_solution(obj, solution)

    # Construct a simple just-in-time warm start where possible.
    warm_x = [[0.0] * n_periods for _ in range(n_items)]
    warm_s = [[0.0] * (n_periods + 1) for _ in range(n_items)]
    warm_y = [[0] * n_periods for _ in range(n_items)]

    for i in range(n_items):
        inventory = initial_inventory[i]
        warm_s[i][0] = inventory

        for t in range(n_periods):
            inventory_used = min(inventory, demands[i][t])
            inventory -= inventory_used
            production = demands[i][t] - inventory_used

            warm_x[i][t] = max(0.0, production)
            warm_y[i][t] = int(production > 1e-9)
            warm_s[i][t + 1] = max(0.0, inventory)

    warm_solution = build_solution(warm_x, warm_s, warm_y)
    if check_feasibility(warm_solution):
        record_solution(warm_solution)

    model = gp.Model("capacitated_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, args.time_limit)
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
                vtype=GRB.BINARY,
                name=f"y_{i}_{t}",
            )

        for k in range(n_periods + 1):
            if k == 0:
                s[i, k] = model.addVar(
                    lb=initial_inventory[i],
                    ub=initial_inventory[i],
                    vtype=GRB.CONTINUOUS,
                    name=f"s_{i}_{k}",
                )
            elif k == n_periods:
                s[i, k] = model.addVar(
                    lb=0.0,
                    ub=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"s_{i}_{k}",
                )
            else:
                s[i, k] = model.addVar(
                    lb=0.0,
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
                name=f"production_setup_{i}_{t}",
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

    def callback(cb_model, where):
        nonlocal callback_best

        if logger is None or where != GRB.Callback.MIPSOL:
            return

        try:
            incumbent_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
            tolerance = 1e-8 * max(1.0, abs(incumbent_obj))

            if incumbent_obj >= callback_best - tolerance:
                return

            flat_x = cb_model.cbGetSolution(x_vars)
            flat_s = cb_model.cbGetSolution(s_vars)
            flat_y = cb_model.cbGetSolution(y_vars)

            x_values = [[0.0] * n_periods for _ in range(n_items)]
            s_values = [[0.0] * (n_periods + 1) for _ in range(n_items)]
            y_values = [[0.0] * n_periods for _ in range(n_items)]

            p = 0
            for i in range(n_items):
                for t in range(n_periods):
                    x_values[i][t] = flat_x[p]
                    p += 1

            p = 0
            for i in range(n_items):
                for k in range(n_periods + 1):
                    s_values[i][k] = flat_s[p]
                    p += 1

            p = 0
            for i in range(n_items):
                for t in range(n_periods):
                    y_values[i][t] = flat_y[p]
                    p += 1

            solution = build_solution(x_values, s_values, y_values)
            if check_feasibility(solution):
                logger.log_solution(solution["objective_value"], solution)
                callback_best = incumbent_obj
        except Exception:
            # Logging must never terminate the optimization.
            pass

    model.optimize(callback if logger else None)

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

        record_solution(build_solution(final_x, final_s, final_y))

    if best_solution is None:
        status_names = {
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible or unbounded",
            GRB.TIME_LIMIT: "time limit reached",
        }
        status_text = status_names.get(model.Status, f"solver status {model.Status}")
        raise RuntimeError(f"No feasible solution found: {status_text}")

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(
            best_solution,
            f,
            separators=(",", ":"),
            allow_nan=False,
        )


if __name__ == "__main__":
    main()