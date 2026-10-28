import argparse
import json
from typing import List, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_objective_value(
    assignment: List[int],
    cost_matrices: List[List[List[int]]],
    reference_point: List[float],
    search_direction_lambda: List[float],
) -> float:
    """Compute max_k lambda_k * (cost_k(assignment) - reference_k)."""
    n = len(assignment)
    p = len(cost_matrices)
    worst = -float("inf")
    for k in range(p):
        total = 0.0
        cm = cost_matrices[k]
        for i in range(n):
            total += cm[i][assignment[i]]
        dev = search_direction_lambda[k] * (total - reference_point[k])
        if dev > worst:
            worst = dev
    return float(worst)


def greedy_fallback_assignment(
    n: int,
    cost_matrices: List[List[List[int]]],
    search_direction_lambda: List[float],
) -> List[int]:
    """
    Fast feasible fallback if no MIP incumbent is available.
    Greedy over weighted sum of criteria.
    """
    weighted = [[0.0] * n for _ in range(n)]
    p = len(cost_matrices)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += search_direction_lambda[k] * cost_matrices[k][i][j]
            weighted[i][j] = s

    available = set(range(n))
    assignment = [-1] * n
    for i in range(n):
        best_j = min(available, key=lambda j: weighted[i][j])
        assignment[i] = best_j
        available.remove(best_j)
    return assignment


def build_assignment_matrix(assignment: List[int]) -> List[List[int]]:
    n = len(assignment)
    mat = [[0] * n for _ in range(n)]
    for i, j in enumerate(assignment):
        if 0 <= j < n:
            mat[i][j] = 1
    return mat


def solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger]) -> dict:
    n = int(instance["n"])
    p = int(instance["p"])
    cost_matrices = instance["cost_matrices"]
    reference_point = [float(v) for v in instance["reference_point"]]
    search_direction_lambda = [float(v) for v in instance["search_direction_lambda"]]

    model = gp.Model("minmax_multiobjective_assignment")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0, int(time_limit))
    model.Params.Threads = 1

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    z = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")

    # One-to-one assignment constraints
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"task_{i}")
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"agent_{j}")

    # Min-max epigraph constraints
    for k in range(p):
        expr = gp.quicksum(cost_matrices[k][i][j] * x[i, j] for i in range(n) for j in range(n))
        model.addConstr(
            z >= search_direction_lambda[k] * (expr - reference_point[k]),
            name=f"crit_{k}",
        )

    model.setObjective(z, GRB.MINIMIZE)

    # Incumbent logging callback
    model._logger = logger
    model._best_logged = float("inf")

    def cb(m: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
            if obj + 1e-12 < m._best_logged:
                m._best_logged = obj
                m._logger.log(obj)

    model.optimize(cb)

    assignment = None
    objective_value = None

    if model.SolCount > 0:
        assignment = [-1] * n
        for i in range(n):
            chosen_j = -1
            for j in range(n):
                if x[i, j].X > 0.5:
                    chosen_j = j
                    break
            if chosen_j == -1:
                # Numerical fallback: pick max-valued variable in row
                chosen_j = max(range(n), key=lambda jj: x[i, jj].X)
            assignment[i] = int(chosen_j)

        objective_value = float(model.ObjVal)
    else:
        # Fallback feasible solution in extreme cases (e.g., very tiny time limit)
        assignment = greedy_fallback_assignment(n, cost_matrices, search_direction_lambda)
        objective_value = compute_objective_value(
            assignment, cost_matrices, reference_point, search_direction_lambda
        )
        if logger is not None:
            logger.log(objective_value)

    assignment_matrix = build_assignment_matrix(assignment)

    return {
        "objective_value": float(objective_value),
        "assignment": assignment,
        "assignment_matrix": assignment_matrix,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve min-max weighted-deviation assignment problem.")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, ensure_ascii=False)


if __name__ == "__main__":
    main()