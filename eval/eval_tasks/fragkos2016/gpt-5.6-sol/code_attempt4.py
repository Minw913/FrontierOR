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
    supplied_big_m = [[float(v) for v in row] for row in data["big_M"]]

    # Construct safe setup-linking bounds. Some instances contain big-M values
    # that are tighter than the actual feasible production requirement. Using
    # such a value can incorrectly make an otherwise feasible model infeasible.
    # Remaining demand is always a valid production upper bound when inventory
    # is nonnegative and terminal inventory is zero. The initial-deficit term
    # also supports data sets that encode initial backlog with a negative value.
    big_m = [[0.0] * n_periods for _ in range(n_items)]
    for i in range(n_items):
        initial_deficit = max(0.0, -initial_inventory[i])
        remaining_demand = 0.0
        for t in range(n_periods - 1, -1, -1):
            remaining_demand += demands[i][t]
            safe_demand_bound = remaining_demand + initial_deficit

            capacity_bound = math.inf
            vt = variable_times[i][t]
            st = setup_times[i][t]
            if vt > 0.0:
                capacity_bound = max(0.0, (capacity[t] - st) / vt)

            safe_bound = min(safe_demand_bound, capacity_bound)
            # Keep the supplied bound when it is larger; this remains safe
            # because the explicit capacity and flow constraints still apply.
            big_m[i][t] = max(
                0.0,
                safe_bound,
                supplied_big_m[i][t],
            )

    def objective_from_values(x_values, s_values, y_values):
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
                out_y[i][t] = 1 if float(y_values[i][t]) >= 0.5 else 0

            # Reconstruct inventory directly from the production plan. This
            # avoids tiny solver inconsistencies in reported balance variables.
            inventory = float(initial_inventory[i])
            for t in range(n_periods):
                inventory += out_x[i][t] - demands[i][t]
                if abs(inventory) <= 1e-7:
                    inventory = 0.0
                out_s[i][t + 1] = inventory

        objective = objective_from_values(out_x, out_s, out_y)
        return {
            "objective_value": objective,
            "x": out_x,
            "s": out_s,
            "y": out_y,
        }

    def is_feasible(solution, tolerance=1e-5):
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
                if s_values[i][t + 1] < -tolerance:
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
            used_capacity = sum(
                setup_times[i][t] * y_values[i][t]
                + variable_times[i][t] * x_values[i][t]
                for i in range(n_items)
            )
            if used_capacity > capacity[t] + tolerance:
                return False

        expected_objective = objective_from_values(x_values, s_values, y_values)
        if abs(expected_objective - solution["objective_value"]) > (
            tolerance * max(1.0, abs(expected_objective))
        ):
            return False

        return True

    best_solution = None
    best_objective = math.inf

    def record_solution(solution):
        nonlocal best_solution, best_objective

        if not is_feasible(solution):
            return

        objective = float(solution["objective_value"])
        improvement_tolerance = 1e-8 * max(1.0, abs(objective))

        if objective < best_objective - improvement_tolerance:
            best_solution = solution
            best_objective = objective
            if logger:
                logger.log_solution(objective, solution)

    model = gp.Model("capacitated_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    remaining_time = float(args.time_limit) - (time.monotonic() - start_time)
    model.Params.TimeLimit = max(0.01, remaining_time)

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

        s[i, 0] = model.addVar(
            lb=initial_inventory[i],
            ub=initial_inventory[i],
            vtype=GRB.CONTINUOUS,
            name=f"s_{i}_0",
        )

        for t in range(1, n_periods):
            s[i, t] = model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"s_{i}_{t}",
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

    flat_x_vars = [x[i, t] for i in range(n_items) for t in range(n_periods)]
    flat_s_vars = [
        s[i, t]
        for i in range(n_items)
        for t in range(n_periods + 1)
    ]
    flat_y_vars = [y[i, t] for i in range(n_items) for t in range(n_periods)]

    last_logged_objective = math.inf

    def callback(cb_model, where):
        nonlocal last_logged_objective

        if where == GRB.Callback.MIP:
            if time.monotonic() - start_time >= args.time_limit:
                cb_model.terminate()
            return

        if where != GRB.Callback.MIPSOL:
            return

        try:
            incumbent_objective = float(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            tolerance = 1e-8 * max(1.0, abs(incumbent_objective))
            if incumbent_objective >= last_logged_objective - tolerance:
                return

            flat_x = cb_model.cbGetSolution(flat_x_vars)
            flat_s = cb_model.cbGetSolution(flat_s_vars)
            flat_y = cb_model.cbGetSolution(flat_y_vars)

            x_values = [[0.0] * n_periods for _ in range(n_items)]
            s_values = [[0.0] * (n_periods + 1) for _ in range(n_items)]
            y_values = [[0.0] * n_periods for _ in range(n_items)]

            pos = 0
            for i in range(n_items):
                for t in range(n_periods):
                    x_values[i][t] = flat_x[pos]
                    pos += 1

            pos = 0
            for i in range(n_items):
                for t in range(n_periods + 1):
                    s_values[i][t] = flat_s[pos]
                    pos += 1

            pos = 0
            for i in range(n_items):
                for t in range(n_periods):
                    y_values[i][t] = flat_y[pos]
                    pos += 1

            solution = build_solution(x_values, s_values, y_values)
            if is_feasible(solution):
                last_logged_objective = solution["objective_value"]
                if logger:
                    logger.log_solution(
                        solution["objective_value"],
                        solution,
                    )
        except Exception:
            # Logging must never interrupt optimization.
            pass

    model.optimize(callback)

    if model.SolCount > 0:
        x_values = [
            [float(x[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]
        s_values = [
            [float(s[i, t].X) for t in range(n_periods + 1)]
            for i in range(n_items)
        ]
        y_values = [
            [float(y[i, t].X) for t in range(n_periods)]
            for i in range(n_items)
        ]
        record_solution(build_solution(x_values, s_values, y_values))

    if best_solution is None:
        status_names = {
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible or unbounded",
            GRB.TIME_LIMIT: "time limit reached before finding a solution",
            GRB.INTERRUPTED: "optimization interrupted before finding a solution",
        }
        status_text = status_names.get(
            model.Status,
            f"solver status {model.Status}",
        )
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