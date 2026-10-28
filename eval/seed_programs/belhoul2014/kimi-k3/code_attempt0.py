import argparse
import json

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None, type=str)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = inst["n"]
    p = inst["p"]
    costs = inst["cost_matrices"]
    ref = inst["reference_point"]
    lam = inst["search_direction_lambda"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def build_solution(assignment):
        totals = [0.0] * p
        for i, j in enumerate(assignment):
            for k in range(p):
                totals[k] += costs[k][i][j]
        obj = 0.0
        if p > 0:
            obj = max(lam[k] * (totals[k] - ref[k]) for k in range(p))
        amat = [[0] * n for _ in range(n)]
        for i, j in enumerate(assignment):
            amat[i][int(j)] = 1
        return {
            "objective_value": float(obj),
            "assignment": [int(j) for j in assignment],
            "assignment_matrix": amat,
        }

    def extract_assignment(flat_vals, n):
        # flat_vals is row-major [i * n + j]
        pairs = [(flat_vals[i * n + j], i, j) for i in range(n) for j in range(n)]
        pairs.sort(key=lambda t: -t[0])
        assignment = [-1] * n
        used = [False] * n
        remaining = n
        for v, i, j in pairs:
            if assignment[i] == -1 and not used[j]:
                assignment[i] = j
                used[j] = True
                remaining -= 1
                if remaining == 0:
                    break
        return assignment

    # Fallback identity assignment (feasible) in case the solver cannot run.
    best_assignment = list(range(n))
    best_solution = build_solution(best_assignment)

    if n > 0 and p > 0:
        model = gp.Model("multi_criteria_assignment")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(args.time_limit, 0)

        x = [[model.addVar(vtype=GRB.BINARY, name="x_%d_%d" % (i, j))
              for j in range(n)] for i in range(n)]
        t = model.addVar(lb=-GRB.INFINITY, name="t")
        flat_vars = [x[i][j] for i in range(n) for j in range(n)]

        for i in range(n):
            model.addConstr(gp.quicksum(x[i][j] for j in range(n)) == 1)
        for j in range(n):
            model.addConstr(gp.quicksum(x[i][j] for i in range(n)) == 1)

        for k in range(p):
            ck = costs[k]
            expr = gp.quicksum(ck[i][j] * x[i][j] for i in range(n) for j in range(n))
            model.addConstr(lam[k] * (expr - ref[k]) <= t)

        model.setObjective(t, GRB.MINIMIZE)

        def callback(m, where):
            if where == GRB.Callback.MIPSOL and logger is not None:
                try:
                    vals = m.cbGetSolution(flat_vars)
                    assignment = extract_assignment(vals, n)
                    sol = build_solution(assignment)
                    logger.log_solution(sol["objective_value"], sol)
                except Exception:
                    pass

        model.optimize(callback)

        if model.SolCount > 0:
            vals = [v.X for v in flat_vars]
            best_assignment = extract_assignment(vals, n)
            best_solution = build_solution(best_assignment)

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    main()