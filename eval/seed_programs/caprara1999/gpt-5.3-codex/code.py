import argparse
import json
import math
from typing import List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_objective(selected: List[int], profit_matrix: List[List[int]]) -> float:
    n = len(selected)
    obj = 0.0
    for i in range(n):
        if selected[i]:
            obj += profit_matrix[i][i]
    for i in range(n):
        if not selected[i]:
            continue
        pi = profit_matrix[i]
        for j in range(i + 1, n):
            if selected[j]:
                obj += 2.0 * pi[j]
    return obj


def greedy_dynamic(weights: List[int], profit_matrix: List[List[int]], capacity: int, mode: str) -> List[int]:
    """
    mode in {"ratio", "gain"}:
      gain(i | S) = p_ii + 2*sum_{j in S} p_ij
      score = gain / w_i  (ratio) or gain (gain)
    """
    n = len(weights)
    selected = [0] * n
    chosen = []
    rem = capacity

    while True:
        best_i = -1
        best_score = -1.0
        best_gain = -1.0

        for i in range(n):
            if selected[i] or weights[i] > rem:
                continue

            gain = float(profit_matrix[i][i])
            row = profit_matrix[i]
            for j in chosen:
                gain += 2.0 * row[j]

            score = gain / weights[i] if mode == "ratio" else gain

            if (score > best_score + 1e-12) or (
                abs(score - best_score) <= 1e-12 and gain > best_gain + 1e-12
            ):
                best_score = score
                best_gain = gain
                best_i = i

        if best_i == -1:
            break

        selected[best_i] = 1
        chosen.append(best_i)
        rem -= weights[best_i]

    return selected


def greedy_static(weights: List[int], profit_matrix: List[List[int]], capacity: int, mode: str) -> List[int]:
    """
    Static surrogate:
      s_i = p_ii + sum_{j != i} p_ij
    Sort by s_i/weight or s_i.
    """
    n = len(weights)
    surrogate = []
    for i in range(n):
        s = float(profit_matrix[i][i])
        row = profit_matrix[i]
        for j in range(n):
            if j != i:
                s += row[j]
        surrogate.append(s)

    if mode == "ratio":
        order = sorted(range(n), key=lambda i: (surrogate[i] / weights[i], surrogate[i]), reverse=True)
    else:
        order = sorted(range(n), key=lambda i: surrogate[i], reverse=True)

    selected = [0] * n
    rem = capacity
    for i in order:
        if weights[i] <= rem:
            selected[i] = 1
            rem -= weights[i]
    return selected


def build_initial_solution(
    weights: List[int],
    profit_matrix: List[List[int]],
    capacity: int,
    logger: Optional[SolutionLogger] = None,
) -> Tuple[List[int], float]:
    candidates = []

    for mode in ("ratio", "gain"):
        s = greedy_dynamic(weights, profit_matrix, capacity, mode)
        candidates.append(s)

    for mode in ("ratio", "gain"):
        s = greedy_static(weights, profit_matrix, capacity, mode)
        candidates.append(s)

    # Always include empty feasible solution
    n = len(weights)
    candidates.append([0] * n)

    best_sel = None
    best_obj = -math.inf
    for s in candidates:
        obj = compute_objective(s, profit_matrix)
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_sel = s

    if logger is not None:
        logger.log(float(best_obj))

    return best_sel, float(best_obj)


def solve_instance(instance: dict, time_limit: int, logger: Optional[SolutionLogger] = None) -> dict:
    n = int(instance["n"])
    capacity = int(instance["capacity"])
    weights = list(map(int, instance["weights"]))
    profit_matrix = instance["profit_matrix"]

    # Basic validation
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} does not match n={n}")
    if len(profit_matrix) != n or any(len(row) != n for row in profit_matrix):
        raise ValueError("profit_matrix must have shape [n][n]")

    init_sel, init_obj = build_initial_solution(weights, profit_matrix, capacity, logger=logger)

    model = gp.Model("quadratic_knapsack")
    model.Params.TimeLimit = max(0, int(time_limit))
    model.Params.Threads = 1
    model.Params.NonConvex = 2
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

    # Objective: sum_i p_ii x_i + sum_{i<j} 2 p_ij x_i x_j
    obj = gp.QuadExpr()
    for i in range(n):
        pii = profit_matrix[i][i]
        if pii != 0:
            obj.add(pii * x[i])

    for i in range(n):
        row = profit_matrix[i]
        xi = x[i]
        for j in range(i + 1, n):
            pij = row[j]
            if pij != 0:
                obj.add((2.0 * pij) * xi * x[j])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Warm start from heuristic
    for i in range(n):
        x[i].Start = int(init_sel[i])

    # Logging incumbents from MIP
    model._logger = logger
    model._best_logged = float(init_obj)

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if incumbent_obj > cb_model._best_logged + 1e-9:
                cb_model._best_logged = float(incumbent_obj)
                if cb_model._logger is not None:
                    cb_model._logger.log(float(incumbent_obj))

    model.optimize(callback)

    if model.SolCount > 0:
        selected = [int(round(x[i].X)) for i in range(n)]
    else:
        selected = init_sel

    objective_value = compute_objective(selected, profit_matrix)

    return {
        "objective_value": float(objective_value),
        "selected_items": selected,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve quadratic knapsack with pairwise profits.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Max runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log.")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()