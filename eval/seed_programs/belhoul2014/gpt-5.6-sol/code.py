import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


class HeuristicTimeout(Exception):
    pass


def hungarian(cost, deadline):
    """Minimum-cost assignment for a square floating-point cost matrix."""
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    matched_row = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        if time.monotonic() >= deadline:
            raise HeuristicTimeout

        matched_row[0] = i
        min_value = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        col0 = 0

        while True:
            used[col0] = True
            row0 = matched_row[col0]
            delta = math.inf
            col1 = 0
            row_cost = cost[row0 - 1]

            for j in range(1, n + 1):
                if not used[j]:
                    cur = float(row_cost[j - 1]) - u[row0] - v[j]
                    if cur < min_value[j]:
                        min_value[j] = cur
                        way[j] = col0
                    if min_value[j] < delta:
                        delta = min_value[j]
                        col1 = j

            for j in range(n + 1):
                if used[j]:
                    u[matched_row[j]] += delta
                    v[j] -= delta
                else:
                    min_value[j] -= delta

            col0 = col1
            if matched_row[col0] == 0:
                break

        while True:
            col1 = way[col0]
            matched_row[col0] = matched_row[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        assignment[matched_row[j] - 1] = j - 1
    return assignment


def evaluate_assignment(assignment, cost_matrices, reference, weights):
    p_count = len(cost_matrices)
    totals = [0] * p_count

    for i, j in enumerate(assignment):
        for k in range(p_count):
            totals[k] += cost_matrices[k][i][j]

    objective = max(
        weights[k] * (totals[k] - reference[k])
        for k in range(p_count)
    )
    return float(objective), totals


def make_solution(assignment, objective):
    n = len(assignment)
    matrix = [[0] * n for _ in range(n)]
    for i, j in enumerate(assignment):
        matrix[i][j] = 1

    return {
        "objective_value": float(objective),
        "assignment": [int(j) for j in assignment],
        "assignment_matrix": matrix,
    }


def valid_assignment(assignment, n):
    return (
        len(assignment) == n
        and all(isinstance(j, int) and 0 <= j < n for j in assignment)
        and len(set(assignment)) == n
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    p_count = int(data["p"])
    costs = data["cost_matrices"]
    reference = [float(x) for x in data["reference_point"]]
    weights = [float(x) for x in data["search_direction_lambda"]]

    if n <= 0 or p_count <= 0:
        raise ValueError("The instance must have positive n and p.")

    # Identity is an immediate feasible fallback, including for tiny limits.
    best_assignment = list(range(n))
    best_objective, best_totals = evaluate_assignment(
        best_assignment, costs, reference, weights
    )
    best_solution = make_solution(best_assignment, best_objective)

    if logger:
        logger.log_solution(best_objective, best_solution)

    def accept_candidate(assignment, totals=None):
        nonlocal best_assignment, best_objective, best_totals, best_solution

        if not valid_assignment(assignment, n):
            return False

        if totals is None:
            objective, totals = evaluate_assignment(
                assignment, costs, reference, weights
            )
        else:
            objective = max(
                weights[k] * (totals[k] - reference[k])
                for k in range(p_count)
            )
            objective = float(objective)

        tolerance = 1e-10 * max(1.0, abs(best_objective))
        if objective < best_objective - tolerance:
            best_assignment = assignment[:]
            best_objective = objective
            best_totals = totals[:]
            best_solution = make_solution(best_assignment, best_objective)
            if logger:
                logger.log_solution(best_objective, best_solution)
            return True
        return False

    # Spend only a small fraction of the limit on deterministic warm starts.
    heuristic_budget = min(
        1.5,
        max(0.0, 0.15 * max(0, args.time_limit))
    )
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    if n > 1 and time.monotonic() < heuristic_deadline:
        # A scalarized assignment is usually a strong minimax warm start.
        try:
            aggregate = []
            for i in range(n):
                if time.monotonic() >= heuristic_deadline:
                    raise HeuristicTimeout
                aggregate.append([
                    sum(weights[k] * costs[k][i][j] for k in range(p_count))
                    for j in range(n)
                ])

            candidate = hungarian(aggregate, heuristic_deadline)
            objective, totals = evaluate_assignment(
                candidate, costs, reference, weights
            )
            accept_candidate(candidate, totals)
        except HeuristicTimeout:
            pass

        # Add assignments optimized for individual criteria.
        for k in range(p_count):
            if time.monotonic() >= heuristic_deadline:
                break
            try:
                candidate = hungarian(costs[k], heuristic_deadline)
                objective, totals = evaluate_assignment(
                    candidate, costs, reference, weights
                )
                accept_candidate(candidate, totals)
            except HeuristicTimeout:
                break

        # Pair-swap local search on the best warm start.
        current = best_assignment[:]
        current_totals = best_totals[:]

        while time.monotonic() < heuristic_deadline:
            current_obj = max(
                weights[k] * (current_totals[k] - reference[k])
                for k in range(p_count)
            )
            selected = None
            selected_obj = current_obj
            selected_totals = None

            timed_out = False
            for i in range(n):
                if time.monotonic() >= heuristic_deadline:
                    timed_out = True
                    break
                ai = current[i]

                for h in range(i + 1, n):
                    ah = current[h]
                    trial_totals = [0] * p_count
                    trial_obj = -math.inf

                    for k in range(p_count):
                        delta = (
                            costs[k][i][ah]
                            + costs[k][h][ai]
                            - costs[k][i][ai]
                            - costs[k][h][ah]
                        )
                        total = current_totals[k] + delta
                        trial_totals[k] = total
                        value = weights[k] * (total - reference[k])
                        if value > trial_obj:
                            trial_obj = value

                    if trial_obj < selected_obj - 1e-10 * max(
                        1.0, abs(selected_obj)
                    ):
                        selected = (i, h)
                        selected_obj = trial_obj
                        selected_totals = trial_totals

            if timed_out or selected is None:
                break

            i, h = selected
            current[i], current[h] = current[h], current[i]
            current_totals = selected_totals
            accept_candidate(current, current_totals)

    model = None

    try:
        remaining = deadline - time.monotonic()

        if remaining > 0.02:
            model = gp.Model("multi_criteria_assignment")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
            theta = model.addVar(
                lb=-GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name="worst_deviation"
            )

            model.setObjective(theta, GRB.MINIMIZE)

            for i in range(n):
                model.addConstr(
                    gp.quicksum(x[i, j] for j in range(n)) == 1,
                    name=f"task_{i}"
                )

            for j in range(n):
                model.addConstr(
                    gp.quicksum(x[i, j] for i in range(n)) == 1,
                    name=f"agent_{j}"
                )

            flat_vars = [x[i, j] for i in range(n) for j in range(n)]

            for k in range(p_count):
                coeffs = [
                    weights[k] * costs[k][i][j]
                    for i in range(n)
                    for j in range(n)
                ]
                expression = gp.LinExpr(coeffs, flat_vars)
                model.addConstr(
                    expression - theta
                    <= weights[k] * reference[k],
                    name=f"criterion_{k}"
                )

            # Warm start from the best heuristic assignment.
            for i in range(n):
                assigned = best_assignment[i]
                for j in range(n):
                    x[i, j].Start = 1.0 if j == assigned else 0.0
            theta.Start = best_objective

            model.update()

            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                model.Params.TimeLimit = max(0.001, remaining)

                def incumbent_callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL:
                        return

                    try:
                        values = cb_model.cbGetSolution(flat_vars)
                        assignment = [-1] * n
                        used = set()

                        for i in range(n):
                            row_offset = i * n
                            j = max(
                                range(n),
                                key=lambda col: values[row_offset + col]
                            )
                            if values[row_offset + j] < 0.5 or j in used:
                                return
                            assignment[i] = j
                            used.add(j)

                        accept_candidate(assignment)
                    except gp.GurobiError:
                        return

                model.optimize(incumbent_callback)

                if model.SolCount > 0:
                    assignment = [-1] * n
                    used = set()
                    feasible = True

                    for i in range(n):
                        j = max(range(n), key=lambda col: x[i, col].X)
                        if x[i, j].X < 0.5 or j in used:
                            feasible = False
                            break
                        assignment[i] = j
                        used.add(j)

                    if feasible:
                        accept_candidate(assignment)

    except gp.GurobiError:
        # The heuristic solution remains a valid fallback.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except gp.GurobiError:
                pass

    # Recompute to ensure the reported objective exactly matches the output.
    final_objective, _ = evaluate_assignment(
        best_assignment, costs, reference, weights
    )
    final_solution = make_solution(best_assignment, final_objective)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()