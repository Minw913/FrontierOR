import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_components(assignment, A, B, C):
    sA = 0.0
    sB = 0.0
    sC = 0.0
    for i, p in enumerate(assignment):
        sA += A[i][p]
        sB += B[i][p]
        sC += C[i][p]
    return sA, sB, sC


def compute_objective(assignment, A, B, C):
    sA, sB, sC = compute_components(assignment, A, B, C)
    return float(sA * sB + sC)


def initial_heuristic(n, A, B, C, deadline):
    """Produce a decent starting assignment quickly, then refine with pairwise swaps."""
    assignment = None
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        # Proxy pair-cost resembling the true objective structure.
        W = [[A[i][j] * B[i][j] + C[i][j] for j in range(n)] for i in range(n)]
        rows, cols = linear_sum_assignment(W)
        assignment = [0] * n
        for r, c in zip(rows, cols):
            assignment[int(r)] = int(c)
    except Exception:
        assignment = None
    if assignment is None:
        assignment = list(range(n))

    # Pairwise-swap local search on the true objective.
    sA, sB, sC = compute_components(assignment, A, B, C)
    current = sA * sB + sC
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            pi = assignment[i]
            for k in range(i + 1, n):
                pk = assignment[k]
                dA = A[i][pk] + A[k][pi] - A[i][pi] - A[k][pk]
                dB = B[i][pk] + B[k][pi] - B[i][pi] - B[k][pk]
                dC = C[i][pk] + C[k][pi] - C[i][pi] - C[k][pk]
                new_val = (sA + dA) * (sB + dB) + (sC + dC)
                if new_val < current - 1e-9:
                    assignment[i], assignment[k] = pk, pi
                    sA += dA
                    sB += dB
                    sC += dC
                    current = new_val
                    pi = pk
                    improved = True
    return assignment, float(current)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    A = [[float(v) for v in row] for row in inst["A"]]
    B = [[float(v) for v in row] for row in inst["B"]]
    C = [[float(v) for v in row] for row in inst["C"]]

    # Quick heuristic to guarantee a feasible solution and a first incumbent.
    heuristic_deadline = min(time.time() + max(1.0, min(5.0, args.time_limit * 0.1)),
                             start_time + max(1, args.time_limit))
    best_assignment, best_obj = initial_heuristic(n, A, B, C, heuristic_deadline)
    if logger:
        logger.log_solution(best_obj, {
            "objective_value": best_obj,
            "assignment": [int(p) for p in best_assignment],
        })

    remaining = max(1, int(args.time_limit - (time.time() - start_time)))

    # Build the bilinear MIP. The product (sum A) * (sum B) is a quadratic
    # objective; Gurobi handles nonconvex quadratic objectives with NonConvex=2.
    try:
        model = gp.Model("bilinear_assignment")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.NonConvex = 2
        model.Params.TimeLimit = remaining

        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        sA = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sA")
        sB = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sB")
        sC = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sC")

        model.addConstr(sA == gp quicksum(A[i][j] * x[i, j] for i in range(n) for j in range(n)))
        model.addConstr(sB == gp.quicksum(B[i][j] * x[i, j] for i in range(n) for j in range(n)))
        model.addConstr(sC == gp.quicksum(C[i][j] * x[i, j] for i in range(n) for j in range(n)))

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
        for j in range(n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)

        model.setObjective(sA * sB + sC, GRB.MINIMIZE)

        state = {"best": best_obj}

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    vals = m.cbGetSolution(x)
                    assignment = [0] * n
                    for i in range(n):
                        chosen = 0
                        bestv = -1.0
                        for j in range(n):
                            v = vals.get((i, j), 0.0)
                            if v > bestv:
                                bestv = v
                                chosen = j
                        assignment[i] = chosen
                    obj = compute_objective(assignment, A, B, C)
                    if obj < state["best"] - 1e-9:
                        state["best"] = obj
                        if logger:
                            logger.log_solution(obj, {
                                "objective_value": float(obj),
                                "assignment": [int(p) for p in assignment],
                            })
                except Exception:
                    pass

        model.optimize(callback)

        if model.SolCount > 0:
            vals = model.getAttr("X", x)
            assignment = [0] * n
            for i in range(n):
                chosen = 0
                bestv = -1.0
                for j in range(n):
                    v = vals.get((i, j), 0.0)
                    if v > bestv:
                        bestv = v
                        chosen = j
                assignment[i] = chosen
            obj = compute_objective(assignment, A, B, C)
            if obj < best_obj - 1e-12:
                best_obj = obj
                best_assignment = assignment
    except Exception:
        # Fall back to the heuristic incumbent if the MIP fails for any reason.
        pass

    if logger:
        logger.log_solution(best_obj, {
            "objective_value": float(best_obj),
            "assignment": [int(p) for p in best_assignment],
        })

    solution = {
        "objective_value": float(best_obj),
        "assignment": [int(p) for p in best_assignment],
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()