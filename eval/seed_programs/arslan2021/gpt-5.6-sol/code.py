import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(objective_value, selected, item_ids):
    return {
        "objective_value": float(objective_value),
        "x": {
            str(item_ids[i]): int(selected[i])
            for i in range(len(item_ids))
        },
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    items = instance["items"]
    n = len(items)
    capacity = float(instance["problem_parameters"]["knapsack_capacity"])
    gamma = float(instance["problem_parameters"]["uncertainty_budget"])

    item_ids = [item["id"] for item in items]
    weights = [float(item["weight"]) for item in items]
    profits = [float(item["nominal_profit"]) for item in items]
    degradations = [float(item["max_degradation"]) for item in items]
    penalties = [float(item["outsource_penalty"]) for item in items]
    repair_capacities = [float(item["repair_capacity"]) for item in items]

    # Selecting nothing is always a feasible solution with objective zero.
    best_selected = [0] * n
    best_objective = 0.0
    initial_solution = make_solution(best_objective, best_selected, item_ids)
    if logger:
        logger.log_solution(best_objective, initial_solution)

    if args.time_limit <= 0:
        write_solution(args.solution_path, initial_solution)
        return

    try:
        model = gp.Model("robust_knapsack_lp_recourse")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(
            0.0, float(args.time_limit) - (time.monotonic() - start_time)
        )

        # First-stage selection.
        x = model.addVars(n, vtype=GRB.BINARY, name="x")

        # LP-relaxed recourse fractions.
        y = model.addVars(n, lb=0.0, ub=1.0, name="produced")
        z = model.addVars(n, lb=0.0, ub=1.0, name="repaired")

        # Dual variables for the budgeted uncertainty support function:
        # max sum_i a_i d_i, 0 <= d_i <= 1, sum_i d_i <= gamma.
        pi = model.addVar(lb=0.0, name="budget_dual")
        rho = model.addVars(n, lb=0.0, name="item_dual")

        for i in range(n):
            model.addConstr(y[i] <= x[i], name=f"selection_{i}")
            model.addConstr(z[i] <= y[i], name=f"repair_fraction_{i}")
            model.addConstr(
                pi + rho[i] >= degradations[i] * (y[i] - z[i]),
                name=f"uncertainty_dual_{i}",
            )

        model.addConstr(
            gp.quicksum(
                weights[i] * y[i] + repair_capacities[i] * z[i]
                for i in range(n)
            )
            <= capacity,
            name="capacity",
        )

        model.setObjective(
            gp.quicksum(
                (penalties[i] - profits[i]) * x[i] - penalties[i] * y[i]
                for i in range(n)
            )
            + gamma * pi
            + gp.quicksum(rho[i] for i in range(n)),
            GRB.MINIMIZE,
        )

        # Supply the guaranteed feasible all-zero solution.
        for i in range(n):
            x[i].Start = 0.0
            y[i].Start = 0.0
            z[i].Start = 0.0
            rho[i].Start = 0.0
        pi.Start = 0.0

        callback_state = {"best_logged": 0.0}

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
                if objective < callback_state["best_logged"] - 1e-8:
                    x_values = cb_model.cbGetSolution([x[i] for i in range(n)])
                    selected = [1 if value >= 0.5 else 0 for value in x_values]
                    solution = make_solution(objective, selected, item_ids)
                    if logger:
                        logger.log_solution(objective, solution)
                    callback_state["best_logged"] = objective
            except (gp.GurobiError, ValueError, TypeError):
                pass

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            best_selected = [
                1 if float(x[i].X) >= 0.5 else 0
                for i in range(n)
            ]
            best_objective = float(model.ObjVal)

            # Re-optimize the continuous recourse for the returned selection.
            # This is useful when the main MIP stops on the time limit with a
            # heuristic incumbent whose continuous variables are not fully
            # polished.
            remaining = float(args.time_limit) - (time.monotonic() - start_time)
            if remaining > 0.01:
                for i in range(n):
                    x[i].LB = best_selected[i]
                    x[i].UB = best_selected[i]

                model.Params.TimeLimit = remaining
                model.Params.MIPGap = 1e-4
                model.optimize(incumbent_callback)

                if model.SolCount > 0:
                    best_objective = float(model.ObjVal)

        final_solution = make_solution(
            best_objective, best_selected, item_ids
        )

        if logger and best_objective < callback_state["best_logged"] - 1e-8:
            logger.log_solution(best_objective, final_solution)

        write_solution(args.solution_path, final_solution)

    except gp.GurobiError:
        # Preserve a valid output even if solver initialization or optimization
        # fails unexpectedly.
        fallback = make_solution(0.0, [0] * n, item_ids)
        write_solution(args.solution_path, fallback)


if __name__ == "__main__":
    main()