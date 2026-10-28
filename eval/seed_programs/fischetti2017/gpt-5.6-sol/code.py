import argparse
import json
import math
import os
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the quadratic uncapacitated facility location problem.")
    parser.add_argument("--instance_path", required=True, help="Path to the input instance JSON file.")
    parser.add_argument("--solution_path", required=True, help="Path to the output solution JSON file.")
    parser.add_argument("--time_limit", required=True, type=int, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None, help="Optional JSONL convergence log path.")
    return parser.parse_args()


def load_instance(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    n = int(data["num_facilities"])
    m = int(data["num_customers"])
    opening_costs = np.asarray(data["opening_costs"], dtype=np.float64)
    allocation_costs = np.asarray(data["allocation_costs"], dtype=np.float64)

    if n <= 0 or m <= 0:
        raise ValueError("The instance must contain at least one facility and one customer.")
    if opening_costs.shape != (n,):
        raise ValueError("opening_costs has an invalid shape.")
    if allocation_costs.shape != (n, m):
        raise ValueError("allocation_costs has an invalid shape.")
    if np.any(~np.isfinite(opening_costs)) or np.any(opening_costs < 0.0):
        raise ValueError("Opening costs must be finite and nonnegative.")
    if np.any(~np.isfinite(allocation_costs)) or np.any(allocation_costs <= 0.0):
        raise ValueError("Quadratic allocation cost coefficients must be finite and strictly positive.")

    return n, m, opening_costs, allocation_costs


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, m, opening_costs, allocation_costs = load_instance(args.instance_path)

    # For a fixed set S of open facilities, customer j's allocation problem is:
    #
    #   min sum_i c_ij x_ij^2
    #   s.t. sum_i x_ij = 1, x_ij >= 0, i in S.
    #
    # Its exact solution is x_ij = (1/c_ij) / sum_{k in S}(1/c_kj),
    # with objective 1 / sum_{k in S}(1/c_kj).
    #
    # Scale each customer column to improve numerical conditioning:
    #   k_j = min_i c_ij
    #   w_ij = k_j / c_ij
    # Then the allocation objective is k_j / sum_{i in S} w_ij.
    column_scale = np.min(allocation_costs, axis=0)
    weights = column_scale[np.newaxis, :] / allocation_costs

    best_subset = None
    best_objective = math.inf

    def subset_objective(subset):
        selected = np.flatnonzero(subset)
        if selected.size == 0:
            return math.inf
        denominators = np.sum(weights[selected, :], axis=0)
        if np.any(denominators <= 0.0):
            return math.inf
        return float(
            np.sum(opening_costs[selected])
            + np.sum(column_scale / denominators)
        )

    def build_solution(subset):
        selected = np.flatnonzero(subset)
        if selected.size == 0:
            raise ValueError("Cannot build a solution without an open facility.")

        denominators = np.sum(weights[selected, :], axis=0)
        fractions = weights[selected, :] / denominators[np.newaxis, :]

        # Normalize once more and correct the largest entry in each column so
        # that each reported customer column sums as closely as possible to 1.
        fraction_sums = np.sum(fractions, axis=0)
        fractions /= fraction_sums[np.newaxis, :]
        largest_rows = np.argmax(fractions, axis=0)
        corrections = 1.0 - np.sum(fractions, axis=0)
        fractions[largest_rows, np.arange(m)] += corrections

        x = np.zeros((n, m), dtype=np.float64)
        x[selected, :] = fractions

        allocation_part = float(
            np.sum(allocation_costs[selected, :] * fractions * fractions)
        )
        objective = float(np.sum(opening_costs[selected]) + allocation_part)

        return {
            "objective_value": objective,
            "variant": "quadratic",
            "y": subset.astype(np.int64).tolist(),
            "x": x.tolist(),
        }

    def consider_subset(subset):
        nonlocal best_subset, best_objective

        subset = np.asarray(subset, dtype=bool)
        if subset.shape != (n,) or not np.any(subset):
            return False

        objective = subset_objective(subset)
        tolerance = 1e-10 * max(1.0, abs(best_objective)) if math.isfinite(best_objective) else 0.0
        if objective >= best_objective - tolerance:
            return False

        best_subset = subset.copy()
        best_objective = objective

        if logger:
            solution = build_solution(best_subset)
            logger.log_solution(solution["objective_value"], solution)
        return True

    # Always create an immediate feasible solution: the best one-facility set.
    singleton_objectives = opening_costs + np.sum(allocation_costs, axis=1)
    initial_facility = int(np.argmin(singleton_objectives))
    current = np.zeros(n, dtype=bool)
    current[initial_facility] = True
    consider_subset(current)

    # Spend a small part of the budget on a one-flip local search. It supplies
    # a useful MIP start and guarantees a feasible answer even if model
    # construction or optimization consumes the remaining time.
    heuristic_budget = min(5.0, max(0.0, 0.08 * max(0, args.time_limit)))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    current_sum = weights[initial_facility, :].copy()
    current_value = subset_objective(current)

    while time.monotonic() < heuristic_deadline:
        best_delta = 0.0
        best_index = -1
        best_is_addition = True

        inv_current = 1.0 / current_sum

        for i in range(n):
            if (i & 15) == 0 and time.monotonic() >= heuristic_deadline:
                break

            if not current[i]:
                new_sum = current_sum + weights[i, :]
                delta = float(
                    opening_costs[i]
                    + np.sum(column_scale * (1.0 / new_sum - inv_current))
                )
                if delta < best_delta:
                    best_delta = delta
                    best_index = i
                    best_is_addition = True
            elif np.count_nonzero(current) > 1:
                new_sum = current_sum - weights[i, :]
                if np.all(new_sum > 0.0):
                    delta = float(
                        -opening_costs[i]
                        + np.sum(column_scale * (1.0 / new_sum - inv_current))
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_index = i
                        best_is_addition = False

        improvement_tolerance = 1e-10 * max(1.0, abs(current_value))
        if best_index < 0 or best_delta >= -improvement_tolerance:
            break

        if best_is_addition:
            current[best_index] = True
            current_sum += weights[best_index, :]
        else:
            current[best_index] = False
            current_sum -= weights[best_index, :]

        current_value = subset_objective(current)
        consider_subset(current)

    model = None

    # Exact reduced mixed-integer convex formulation:
    #
    #   s_j = sum_i w_ij y_i
    #   t_j s_j >= 1
    #   min sum_i f_i y_i + sum_j k_j t_j
    #
    # Since s_j,t_j >= 0, t_j s_j >= 1 is a rotated second-order-cone
    # representable convex quadratic constraint. At optimum t_j = 1/s_j.
    if time.monotonic() < deadline:
        try:
            model = gp.Model("quadratic_ufl")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            y_vars = list(model.addVars(n, vtype=GRB.BINARY, name="y").values())
            s_vars = list(model.addVars(m, lb=0.0, name="s").values())
            t_vars = list(model.addVars(m, lb=0.0, name="t").values())

            model.addConstr(gp.quicksum(y_vars) >= 1.0, name="open_at_least_one")

            for j in range(m):
                column = weights[:, j]
                upper_bound = float(np.sum(column))
                s_vars[j].UB = upper_bound
                if upper_bound > 0.0:
                    t_vars[j].LB = 1.0 / upper_bound

                expression = gp.LinExpr(column.tolist(), y_vars)
                model.addConstr(s_vars[j] == expression, name=f"service_{j}")
                model.addQConstr(t_vars[j] * s_vars[j] >= 1.0, name=f"reciprocal_{j}")

            objective = gp.LinExpr(opening_costs.tolist(), y_vars)
            objective += gp.LinExpr(column_scale.tolist(), t_vars)
            model.setObjective(objective, GRB.MINIMIZE)

            # Warm start from the best heuristic subset.
            warm_selected = np.flatnonzero(best_subset)
            warm_sums = np.sum(weights[warm_selected, :], axis=0)
            for i in range(n):
                y_vars[i].Start = 1.0 if best_subset[i] else 0.0
            for j in range(m):
                s_vars[j].Start = float(warm_sums[j])
                t_vars[j].Start = float(1.0 / warm_sums[j])

            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                model.Params.TimeLimit = max(0.001, remaining)

                def incumbent_callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL:
                        return
                    y_values = cb_model.cbGetSolution(y_vars)
                    candidate = np.fromiter(
                        (value > 0.5 for value in y_values),
                        dtype=bool,
                        count=n,
                    )
                    if np.any(candidate):
                        consider_subset(candidate)

                model.optimize(incumbent_callback)

                if model.SolCount > 0:
                    candidate = np.fromiter(
                        (var.X > 0.5 for var in y_vars),
                        dtype=bool,
                        count=n,
                    )
                    if np.any(candidate):
                        consider_subset(candidate)

        except gp.GurobiError:
            # The heuristic incumbent remains a valid solution if Gurobi
            # cannot initialize or solve the model.
            pass

    final_solution = build_solution(best_subset)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as fh:
        json.dump(final_solution, fh, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()