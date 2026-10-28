import argparse
import json
import time
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_objective(solution: List[int], profit_matrix: List[List[int]]) -> float:
    selected = [i for i, v in enumerate(solution) if v == 1]
    obj = 0.0
    for idx_i, i in enumerate(selected):
        obj += profit_matrix[i][i]
        row_i = profit_matrix[i]
        for j in selected[idx_i + 1 :]:
            obj += row_i[j]
    return float(obj)


def greedy_construct(
    weights: List[int],
    capacity: int,
    profit_matrix: List[List[int]],
    initial_solution: List[int],
    strategy: str = "ratio",  # "ratio" or "gain"
) -> Tuple[List[int], float]:
    n = len(weights)
    sol = initial_solution[:]
    remaining = capacity - sum(weights[i] * sol[i] for i in range(n))

    # Marginal gain of adding each item given current selected set
    marginal = [profit_matrix[i][i] for i in range(n)]
    selected_indices = [i for i, v in enumerate(sol) if v == 1]
    for j in selected_indices:
        row_j = profit_matrix[j]
        for i in range(n):
            if sol[i] == 0:
                marginal[i] += row_j[i]

    current_obj = compute_objective(sol, profit_matrix)

    while True:
        best_i = -1
        best_score = -1.0
        best_gain = -1.0

        for i in range(n):
            if sol[i] == 1:
                continue
            w = weights[i]
            if w > remaining:
                continue

            gain = marginal[i]
            if gain <= 0:
                continue

            if strategy == "gain":
                score = float(gain)
            else:  # ratio
                score = float(gain) / w if w > 0 else float(gain) * 1e12

            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and gain > best_gain
            ):
                best_score = score
                best_gain = gain
                best_i = i

        if best_i == -1:
            break

        sol[best_i] = 1
        remaining -= weights[best_i]
        current_obj += marginal[best_i]

        row = profit_matrix[best_i]
        for k in range(n):
            if sol[k] == 0:
                marginal[k] += row[k]

    return sol, float(current_obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["n"])
    capacity = int(instance["capacity"])
    weights = list(map(int, instance["weights"]))
    profit_matrix = instance["profit_matrix"]

    # Incumbent bookkeeping
    best_sol = None
    best_obj = float("-inf")
    best_logged = [float("-inf")]

    def register_incumbent(obj_val: float, sol_vec: List[int]):
        nonlocal best_obj, best_sol
        if obj_val > best_obj + 1e-9:
            best_obj = obj_val
            best_sol = sol_vec[:]
        if logger and obj_val > best_logged[0] + 1e-9:
            logger.log(float(obj_val))
            best_logged[0] = float(obj_val)

    # Baseline: always include zero-weight items (profits are non-negative)
    baseline = [1 if weights[i] == 0 else 0 for i in range(n)]
    baseline_obj = compute_objective(baseline, profit_matrix)
    register_incumbent(baseline_obj, baseline)

    # Quick exact shortcut if all items fit
    if sum(weights) <= capacity:
        all_sol = [1] * n
        all_obj = compute_objective(all_sol, profit_matrix)
        register_incumbent(all_obj, all_sol)
        output = {"objective_value": float(best_obj), "solution": best_sol}
        with open(args.solution_path, "w") as f:
            json.dump(output, f)
        return

    # Heuristic starts
    sol_ratio, obj_ratio = greedy_construct(weights, capacity, profit_matrix, baseline, "ratio")
    register_incumbent(obj_ratio, sol_ratio)

    sol_gain, obj_gain = greedy_construct(weights, capacity, profit_matrix, baseline, "gain")
    register_incumbent(obj_gain, sol_gain)

    elapsed = time.time() - start_time
    remaining_time = max(0.0, float(args.time_limit) - elapsed)

    if remaining_time > 1e-6:
        model = gp.Model("quadratic_knapsack")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining_time
        model.Params.NonConvex = 2
        model.Params.MIPFocus = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

        # Fix zero-weight items to 1 (always beneficial/non-harmful under non-negative profits)
        for i in range(n):
            if weights[i] == 0:
                x[i].LB = 1
                x[i].UB = 1

        obj = gp.QuadExpr()
        for i in range(n):
            pii = profit_matrix[i][i]
            if pii != 0:
                obj += pii * x[i]
            row_i = profit_matrix[i]
            for j in range(i + 1, n):
                pij = row_i[j]
                if pij != 0:
                    obj += pij * x[i] * x[j]
        model.setObjective(obj, GRB.MAXIMIZE)

        # Warm start
        if best_sol is not None:
            for i in range(n):
                x[i].Start = int(best_sol[i])

        def cb(m, where):
            if where == GRB.Callback.MIPSOL and logger:
                obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val > best_logged[0] + 1e-9:
                    logger.log(float(obj_val))
                    best_logged[0] = float(obj_val)

        model.optimize(cb if logger else None)

        if model.SolCount > 0:
            mip_sol = [int(round(x[i].X)) for i in range(n)]
            mip_obj = compute_objective(mip_sol, profit_matrix)
            register_incumbent(mip_obj, mip_sol)

    # Safety fallback
    if best_sol is None:
        best_sol = [0] * n
        best_obj = 0.0

    output = {
        "objective_value": float(best_obj),
        "solution": [int(v) for v in best_sol],
    }

    with open(args.solution_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()