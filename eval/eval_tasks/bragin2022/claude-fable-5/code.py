import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    M = data["num_machines"]
    J = data["num_jobs"]
    cost = data["cost_matrix"]
    res = data["resource_matrix"]
    cap = data["capacities"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("gap")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining = max(1.0, args.time_limit - (time.time() - start_time) - 2.0)
    model.Params.TimeLimit = remaining

    x = model.addVars(M, J, vtype=GRB.BINARY)

    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for i in range(M) for j in range(J)),
        GRB.MINIMIZE,
    )

    # each job assigned to exactly one machine
    for j in range(J):
        model.addConstr(gp.quicksum(x[i, j] for i in range(M)) == 1)

    # capacity constraints
    for i in range(M):
        model.addConstr(
            gp.quicksum(res[i][j] * x[i, j] for j in range(J)) <= cap[i]
        )

    def extract_assignments(vals):
        assignments = {}
        for j in range(J):
            best_i = 0
            best_v = -1.0
            for i in range(M):
                v = vals[i, j]
                if v > best_v:
                    best_v = v
                    best_i = i
            assignments[str(j)] = best_i
        return assignments

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                vals = m.cbGetSolution(x)
                assignments = extract_assignments(vals)
                logger.log_solution(obj, {
                    "objective_value": obj,
                    "assignments": assignments,
                })

    model.optimize(callback)

    solution = None
    if model.SolCount > 0:
        vals = {(i, j): x[i, j].X for i in range(M) for j in range(J)}
        assignments = extract_assignments(vals)
        obj = float(model.ObjVal)
        solution = {"objective_value": obj, "assignments": assignments}
        if logger:
            logger.log_solution(obj, solution)
    else:
        # Fallback: greedy assignment ignoring optimality (best effort feasibility)
        remaining_cap = list(cap)
        assignments = {}
        total = 0.0
        # Sort jobs by fewest feasible options first
        order = sorted(
            range(J),
            key=lambda j: sum(1 for i in range(M) if res[i][j] <= remaining_cap[i]),
        )
        for j in order:
            candidates = [i for i in range(M) if res[i][j] <= remaining_cap[i]]
            if candidates:
                i = min(candidates, key=lambda i: cost[i][j])
            else:
                i = min(range(M), key=lambda i: cost[i][j])
            assignments[str(j)] = i
            remaining_cap[i] -= res[i][j]
            total += cost[i][j]
        solution = {"objective_value": total, "assignments": assignments}
        if logger:
            logger.log_solution(total, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()