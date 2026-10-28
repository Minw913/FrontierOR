import argparse
import heapq
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def objective_value(assignment, profits, pairwise):
    total = 0
    bags = {}
    for i, k in enumerate(assignment):
        if k >= 0:
            total += profits[i]
            bags.setdefault(k, []).append(i)

    for items in bags.values():
        for a in range(len(items)):
            i = items[a]
            for b in range(a + 1, len(items)):
                total += pairwise[i][items[b]]
    return total


def make_solution(assignment, value):
    return {
        "objective_value": float(value),
        "assignment": [[i, int(k)] for i, k in enumerate(assignment) if k >= 0],
    }


def greedy_completion(seed_assignment, profits, pairwise, weights, capacities,
                      stop_time):
    n = len(profits)
    m = len(capacities)
    assignment = list(seed_assignment)
    loads = [0] * m

    for i, k in enumerate(assignment):
        if k >= 0:
            loads[k] += weights[i]

    gains = [[profits[i] for _ in range(m)] for i in range(n)]
    for j, assigned_k in enumerate(assignment):
        if assigned_k >= 0:
            for i in range(n):
                if assignment[i] < 0 and i != j:
                    gains[i][assigned_k] += pairwise[i][j]

    while time.monotonic() < stop_time:
        best_gain = 0
        best_i = -1
        best_k = -1

        for i in range(n):
            if assignment[i] >= 0:
                continue
            wi = weights[i]
            for k in range(m):
                if loads[k] + wi <= capacities[k]:
                    gain = gains[i][k]
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_k = k

        if best_i < 0:
            break

        assignment[best_i] = best_k
        loads[best_k] += weights[best_i]

        for i in range(n):
            if assignment[i] < 0:
                gains[i][best_k] += pairwise[i][best_i]

    return assignment


def find_pair_seeds(profits, pairwise, weights, capacities, count, stop_time):
    n = len(profits)
    capacity_order = sorted(range(len(capacities)), key=lambda k: capacities[k])
    max_capacity = max(capacities, default=-1)
    heap = []

    for i in range(n):
        if time.monotonic() >= stop_time:
            break
        for j in range(i + 1, n):
            combined_weight = weights[i] + weights[j]
            if combined_weight > max_capacity:
                continue

            value = profits[i] + profits[j] + pairwise[i][j]
            if value <= 0:
                continue

            chosen_k = -1
            for k in capacity_order:
                if combined_weight <= capacities[k]:
                    chosen_k = k
                    break
            if chosen_k < 0:
                continue

            entry = (value, i, j, chosen_k)
            if len(heap) < count:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    return sorted(heap, reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    m = int(data["m"])
    profits = [int(v) for v in data["profits"]]
    pairwise = [[int(v) for v in row] for row in data["pairwise_profits"]]
    weights = [int(v) for v in data["weights"]]
    capacities = [int(v) for v in data["capacities"]]

    best_assignment = [-1] * n
    best_value = 0
    initial_solution = make_solution(best_assignment, best_value)
    if logger:
        logger.log_solution(float(best_value), initial_solution)

    # Spend only a modest part of the budget on deterministic primal heuristics.
    heuristic_budget = min(
        1.5,
        max(0.0, args.time_limit * 0.08),
        max(0.0, deadline - time.monotonic() - 0.05),
    )
    heuristic_deadline = time.monotonic() + heuristic_budget

    if heuristic_budget > 0:
        candidate = greedy_completion(
            [-1] * n, profits, pairwise, weights, capacities, heuristic_deadline
        )
        value = objective_value(candidate, profits, pairwise)
        if value > best_value:
            best_assignment = candidate
            best_value = value
            if logger:
                logger.log_solution(
                    float(best_value), make_solution(best_assignment, best_value)
                )

        if time.monotonic() < heuristic_deadline:
            seeds = find_pair_seeds(
                profits, pairwise, weights, capacities, 4, heuristic_deadline
            )
            for _, i, j, k in seeds:
                if time.monotonic() >= heuristic_deadline:
                    break
                seed = [-1] * n
                seed[i] = k
                seed[j] = k
                candidate = greedy_completion(
                    seed, profits, pairwise, weights, capacities, heuristic_deadline
                )
                value = objective_value(candidate, profits, pairwise)
                if value > best_value:
                    best_assignment = candidate
                    best_value = value
                    if logger:
                        logger.log_solution(
                            float(best_value),
                            make_solution(best_assignment, best_value),
                        )

    # Leave a small amount of time for extracting and writing the final solution.
    remaining = deadline - time.monotonic() - 0.05

    if remaining > 0.01 and n > 0 and m > 0:
        try:
            model = gp.Model("multiple_quadratic_knapsack")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining)

            feasible_keys = [
                (i, k)
                for i in range(n)
                for k in range(m)
                if weights[i] <= capacities[k]
            ]
            x = model.addVars(feasible_keys, vtype=GRB.BINARY, name="x")

            for i in range(n):
                item_vars = [x[i, k] for k in range(m) if (i, k) in x]
                if item_vars:
                    model.addConstr(gp.quicksum(item_vars) <= 1)

            for k in range(m):
                model.addConstr(
                    gp.quicksum(
                        weights[i] * x[i, k]
                        for i in range(n)
                        if (i, k) in x
                    )
                    <= capacities[k]
                )

            nonzero_pairs = [
                (i, j)
                for i in range(n)
                for j in range(i + 1, n)
                if pairwise[i][j] != 0
            ]

            # A linearized model is generally stronger for moderate instances.
            # For very dense large instances, the compact quadratic formulation
            # avoids creating hundreds of thousands of auxiliary variables.
            estimated_pair_assignments = len(nonzero_pairs) * m
            use_linearization = estimated_pair_assignments <= 300000

            if use_linearization:
                y_keys = []
                for i, j in nonzero_pairs:
                    combined_weight = weights[i] + weights[j]
                    for k in range(m):
                        if (
                            (i, k) in x
                            and (j, k) in x
                            and combined_weight <= capacities[k]
                        ):
                            y_keys.append((i, j, k))

                y = model.addVars(
                    y_keys, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y"
                )

                for i, j, k in y_keys:
                    q = pairwise[i][j]
                    if q > 0:
                        model.addConstr(y[i, j, k] <= x[i, k])
                        model.addConstr(y[i, j, k] <= x[j, k])
                    else:
                        model.addConstr(
                            y[i, j, k] >= x[i, k] + x[j, k] - 1
                        )

                obj = gp.LinExpr()
                for i, k in feasible_keys:
                    if profits[i] != 0:
                        obj.add(x[i, k], profits[i])
                for i, j, k in y_keys:
                    obj.add(y[i, j, k], pairwise[i][j])
                model.setObjective(obj, GRB.MAXIMIZE)

                for i, k in feasible_keys:
                    x[i, k].Start = 1.0 if best_assignment[i] == k else 0.0
                for i, j, k in y_keys:
                    y[i, j, k].Start = (
                        1.0
                        if best_assignment[i] == k and best_assignment[j] == k
                        else 0.0
                    )
            else:
                model.Params.NonConvex = 2
                obj = gp.QuadExpr()

                for i, k in feasible_keys:
                    if profits[i] != 0:
                        obj.add(x[i, k], profits[i])

                for i, j in nonzero_pairs:
                    q = pairwise[i][j]
                    combined_weight = weights[i] + weights[j]
                    for k in range(m):
                        if (
                            (i, k) in x
                            and (j, k) in x
                            and combined_weight <= capacities[k]
                        ):
                            obj.add(x[i, k] * x[j, k], q)

                model.setObjective(obj, GRB.MAXIMIZE)
                for i, k in feasible_keys:
                    x[i, k].Start = 1.0 if best_assignment[i] == k else 0.0

            x_items = list(x.items())
            incumbent = {
                "value": best_value,
                "assignment": list(best_assignment),
            }

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = cb_model.cbGetSolution([var for _, var in x_items])
                    assignment = [-1] * n
                    for ((i, k), _), value in zip(x_items, values):
                        if value > 0.5:
                            assignment[i] = k

                    true_value = objective_value(
                        assignment, profits, pairwise
                    )
                    if true_value > incumbent["value"]:
                        incumbent["value"] = true_value
                        incumbent["assignment"] = assignment
                        if logger:
                            logger.log_solution(
                                float(true_value),
                                make_solution(assignment, true_value),
                            )
                except Exception:
                    # Logging or incumbent extraction must not terminate search.
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                assignment = [-1] * n
                for (i, k), var in x_items:
                    if var.X > 0.5:
                        assignment[i] = k
                value = objective_value(assignment, profits, pairwise)
                if value > incumbent["value"]:
                    incumbent["value"] = value
                    incumbent["assignment"] = assignment
                    if logger:
                        logger.log_solution(
                            float(value), make_solution(assignment, value)
                        )

            if incumbent["value"] > best_value:
                best_value = incumbent["value"]
                best_assignment = incumbent["assignment"]

        except gp.GurobiError:
            # Return the best heuristic solution if model construction or
            # optimization cannot be completed.
            pass

    final_solution = make_solution(best_assignment, best_value)
    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()