import argparse
import json
import sys

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def compute_objective(assignment, cost_matrices, ref, lam, n, p):
    vals = []
    for k in range(p):
        tot = sum(cost_matrices[k][i][assignment[i]] for i in range(n))
        vals.append(lam[k] * (tot - ref[k]))
    return max(vals)


def build_solution_dict(assignment, obj, n):
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        mat[i][assignment[i]] = 1
    return {
        "objective_value": float(obj),
        "assignment": [int(a) for a in assignment],
        "assignment_matrix": mat,
    }


def initial_heuristic(cost_matrices, ref, lam, n, p):
    """Hungarian on aggregated cost to get a decent starting assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        agg = np.zeros((n, n))
        for k in range(p):
            agg += lam[k] * np.array(cost_matrices[k], dtype=float)
        r, c = linear_sum_assignment(agg)
        assignment = [0] * n
        for i, j in zip(r, c):
            assignment[i] = int(j)
        return assignment
    except Exception:
        return list(range(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["n"]
    p = data["p"]
    C = data["cost_matrices"]
    ref = data["reference_point"]
    lam = data["search_direction_lambda"]

    # Heuristic warm start
    heur_assign = initial_heuristic(C, ref, lam, n, p)
    heur_obj = compute_objective(heur_assign, C, ref, lam, n, p)
    best = {"assignment": heur_assign, "obj": heur_obj}
    if logger:
        logger.log_solution(heur_obj, build_solution_dict(heur_assign, heur_obj, n))

    try:
        model = gp.Model("minmax_assignment")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1, args.time_limit - 2)

        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
        for j in range(n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)

        for k in range(p):
            expr = gp.quicksum(C[k][i][j] * x[i, j] for i in range(n) for j in range(n))
            model.addConstr(z >= lam[k] * (expr - ref[k]))

        model.setObjective(z, GRB.MINIMIZE)

        # Warm start
        for i in range(n):
            for j in range(n):
                x[i, j].Start = 1 if heur_assign[i] == j else 0
        z.Start = heur_obj

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = m.cbGetSolution(x)
                    assignment = [0] * n
                    for i in range(n):
                        for j in range(n):
                            if xv[i, j] > 0.5:
                                assignment[i] = j
                                break
                    obj = compute_objective(assignment, C, ref, lam, n, p)
                    if obj < best["obj"] - 1e-12:
                        best["obj"] = obj
                        best["assignment"] = assignment
                        if logger:
                            logger.log_solution(obj, build_solution_dict(assignment, obj, n))
                except Exception:
                    pass

        model.optimize(callback)

        if model.SolCount > 0:
            assignment = [0] * n
            for i in range(n):
                for j in range(n):
                    if x[i, j].X > 0.5:
                        assignment[i] = j
                        break
            obj = compute_objective(assignment, C, ref, lam, n, p)
            if obj < best["obj"] - 1e-12:
                best["obj"] = obj
                best["assignment"] = assignment
                if logger:
                    logger.log_solution(obj, build_solution_dict(assignment, obj, n))
    except Exception as e:
        print(f"Solver error: {e}", file=sys.stderr)

    sol = build_solution_dict(best["assignment"], best["obj"], n)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()