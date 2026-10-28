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
    start_time = time.monotonic()
    args = parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n_items"])
    T = int(data["n_periods"])

    capacity = [float(v) for v in data["capacity"]]
    initial_inventory = [float(v) for v in data["initial_inventory"]]
    demands = [[float(v) for v in row] for row in data["demands"]]
    setup_costs = [[float(v) for v in row] for row in data["setup_costs"]]
    holding_costs = [[float(v) for v in row] for row in data["holding_costs"]]
    variable_costs = [[float(v) for v in row] for row in data["variable_costs"]]
    setup_times = [[float(v) for v in row] for row in data["setup_times"]]
    variable_times = [[float(v) for v in row] for row in data["variable_times"]]
    input_big_m = [[float(v) for v in row] for row in data["big_M"]]

    # Tighten the supplied production bounds using total remaining demand and
    # the capacity available to an item if it were the only item produced.
    big_m = [[0.0] * T for _ in range(n)]
    for i in range(n):
        total_required_production = max(
            0.0, sum(demands[i]) - initial_inventory[i]
        )
        remaining_demand = 0.0
        remaining_by_period = [0.0] * T
        for t in range(T - 1, -1, -1):
            remaining_demand += demands[i][t]
            remaining_by_period[t] = remaining_demand

        for t in range(T):
            bound = max(0.0, input_big_m[i][t])
            bound = min(bound, total_required_production, remaining_by_period[t])

            if setup_times[i][t] > capacity[t] + 1e-9:
                bound = 0.0
            elif variable_times[i][t] > 0.0:
                isolated_bound = (
                    capacity[t] - setup_times[i][t]
                ) / variable_times[i][t]
                bound = min(bound, max(0.0, isolated_bound))

            big_m[i][t] = max(0.0, bound)

    def objective_from_arrays(x_values, s_values, y_values):
        value = 0.0
        for i in range(n):
            for t in range(T):
                value += setup_costs[i][t] * y_values[i][t]
                value += variable_costs[i][t] * x_values[i][t]
                # s[i][t+1] is inventory held after period t.
                value += holding_costs[i][t] * s_values[i][t + 1]
        return float(value)

    def make_solution(x_values, s_values, y_values):
        clean_x = [[0.0] * T for _ in range(n)]
        clean_s = [[0.0] * (T + 1) for _ in range(n)]
        clean_y = [[0] * T for _ in range(n)]

        for i in range(n):
            for t in range(T):
                yv = 1 if float(y_values[i][t]) >= 0.5 else 0
                xv = float(x_values[i][t])
                if xv < 0.0 and xv > -1e-7:
                    xv = 0.0
                if yv == 0 and abs(xv) <= 1e-7:
                    xv = 0.0
                clean_x[i][t] = max(0.0, xv)
                clean_y[i][t] = yv

            clean_s[i][0] = initial_inventory[i]
            for k in range(1, T):
                sv = float(s_values[i][k])
                if sv < 0.0 and sv > -1e-7:
                    sv = 0.0
                clean_s[i][k] = max(0.0, sv)
            clean_s[i][T] = 0.0

        obj = objective_from_arrays(clean_x, clean_s, clean_y)
        return {
            "objective_value": obj,
            "x": clean_x,
            "s": clean_s,
            "y": clean_y,
        }

    # Construct a just-in-time incumbent when period capacities permit it.
    fallback_solution = None
    jit_x = [[0.0] * T for _ in range(n)]
    jit_s = [[0.0] * (T + 1) for _ in range(n)]
    jit_y = [[0] * T for _ in range(n)]

    jit_feasible = True
    for i in range(n):
        inventory = initial_inventory[i]
        jit_s[i][0] = inventory

        for t in range(T):
            if inventory >= demands[i][t]:
                production = 0.0
                inventory -= demands[i][t]
            else:
                production = demands[i][t] - inventory
                inventory = 0.0

            jit_x[i][t] = production
            jit_y[i][t] = 1 if production > 1e-9 else 0
            jit_s[i][t + 1] = inventory

            if production > big_m[i][t] + 1e-7:
                jit_feasible = False

        if abs(jit_s[i][T]) > 1e-7:
            jit_feasible = False

    if jit_feasible:
        for t in range(T):
            usage = 0.0
            for i in range(n):
                usage += setup_times[i][t] * jit_y[i][t]
                usage += variable_times[i][t] * jit_x[i][t]
            if usage > capacity[t] + 1e-7:
                jit_feasible = False
                break

    logged_state = {"best": math.inf}

    if jit_feasible:
        fallback_solution = make_solution(jit_x, jit_s, jit_y)
        logged_state["best"] = fallback_solution["objective_value"]
        if logger:
            logger.log_solution(
                fallback_solution["objective_value"], fallback_solution
            )

    model = gp.Model("capacitated_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = {}
    y = {}
    s = {}

    for i in range(n):
        for t in range(T):
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

        for k in range(T + 1):
            if k == 0:
                lb = initial_inventory[i]
                ub = initial_inventory[i]
            elif k == T:
                lb = 0.0
                ub = 0.0
            else:
                # Inventory at the beginning of period k can only be used
                # against demands in periods k,...,T-1.
                lb = 0.0
                ub = sum(demands[i][k:])

            s[i, k] = model.addVar(
                lb=lb,
                ub=ub,
                vtype=GRB.CONTINUOUS,
                name=f"s_{i}_{k}",
            )

    model.update()

    for i in range(n):
        for t in range(T):
            model.addConstr(
                s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1]
            )
            model.addConstr(x[i, t] <= big_m[i][t] * y[i, t])

    for t in range(T):
        model.addConstr(
            gp.quicksum(
                setup_times[i][t] * y[i, t]
                + variable_times[i][t] * x[i, t]
                for i in range(n)
            )
            <= capacity[t]
        )

    model.setObjective(
        gp.quicksum(
            setup_costs[i][t] * y[i, t]
            + variable_costs[i][t] * x[i, t]
            + holding_costs[i][t] * s[i, t + 1]
            for i in range(n)
            for t in range(T)
        ),
        GRB.MINIMIZE,
    )

    if fallback_solution is not None:
        for i in range(n):
            for t in range(T):
                x[i, t].Start = fallback_solution["x"][i][t]
                y[i, t].Start = fallback_solution["y"][i][t]
            for k in range(T + 1):
                s[i, k].Start = fallback_solution["s"][i][k]

    x_flat = [x[i, t] for i in range(n) for t in range(T)]
    y_flat = [y[i, t] for i in range(n) for t in range(T)]
    s_flat = [s[i, k] for i in range(n) for k in range(T + 1)]

    def values_to_solution(x_flat_values, s_flat_values, y_flat_values):
        xv = [[0.0] * T for _ in range(n)]
        yv = [[0.0] * T for _ in range(n)]
        sv = [[0.0] * (T + 1) for _ in range(n)]

        p = 0
        for i in range(n):
            for t in range(T):
                xv[i][t] = x_flat_values[p]
                p += 1

        p = 0
        for i in range(n):
            for k in range(T + 1):
                sv[i][k] = s_flat_values[p]
                p += 1

        p = 0
        for i in range(n):
            for t in range(T):
                yv[i][t] = y_flat_values[p]
                p += 1

        return make_solution(xv, sv, yv)

    def incumbent_callback(cb_model, where):
        if logger is None or where != GRB.Callback.MIPSOL:
            return

        incumbent_obj = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
        tolerance = 1e-9 * max(1.0, abs(incumbent_obj))
        if incumbent_obj >= logged_state["best"] - tolerance:
            return

        try:
            x_values = cb_model.cbGetSolution(x_flat)
            s_values = cb_model.cbGetSolution(s_flat)
            y_values = cb_model.cbGetSolution(y_flat)
            solution = values_to_solution(x_values, s_values, y_values)
            logger.log_solution(solution["objective_value"], solution)
            logged_state["best"] = incumbent_obj
        except Exception:
            # Logging must not terminate the optimization search.
            pass

    elapsed = time.monotonic() - start_time
    remaining_time = max(0.0, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    model.optimize(incumbent_callback if logger else None)

    candidates = []
    if fallback_solution is not None:
        candidates.append(fallback_solution)

    if model.SolCount > 0:
        model_x = [[float(x[i, t].X) for t in range(T)] for i in range(n)]
        model_y = [[float(y[i, t].X) for t in range(T)] for i in range(n)]
        model_s = [
            [float(s[i, k].X) for k in range(T + 1)]
            for i in range(n)
        ]
        candidates.append(make_solution(model_x, model_s, model_y))

    if not candidates:
        raise RuntimeError(
            "No feasible solution was found within the specified time limit."
        )

    final_solution = min(candidates, key=lambda sol: sol["objective_value"])

    if logger:
        tolerance = 1e-9 * max(1.0, abs(final_solution["objective_value"]))
        if final_solution["objective_value"] < logged_state["best"] - tolerance:
            logger.log_solution(
                final_solution["objective_value"], final_solution
            )

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()