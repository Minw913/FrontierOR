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

    start = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    m = data["num_agents"]
    n = data["num_jobs"]
    profits = data["profits"]
    weights = data["weights"]
    caps = data["capacities"]

    model = gp.Model("gap")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining = max(1.0, args.time_limit - (time.time() - start) - 2.0)
    model.Params.TimeLimit = remaining

    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")

    # Each job assigned to exactly one agent
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == 1)

    # Capacity constraints
    for i in range(m):
        model.addConstr(
            gp.quicksum(weights[i][j] * x[i, j] for j in range(n)) <= caps[i]
        )

    model.setObjective(
        gp.quicksum(profits[i][j] * x[i, j] for i in range(m) for j in range(n)),
        GRB.MAXIMIZE,
    )

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGetSolution(mdl.getObjective())
            vals = mdl.cbGetSolution(list(x.values()))
            assignment = {}
            idx = 0
            for i in range(m):
                for j in range(n):
                    if vals[idx] > 0.5:
                        assignment[str(j)] = i
                    idx += 1
            sol = {"objective_value": float(obj), "assignment": assignment}
            if logger:
                logger.log_solution(float(obj), sol)

    model.optimize(callback)

    assignment = {}
    obj_val = 0.0
    if model.SolCount > 0:
        obj_val = float(model.ObjVal)
        for j in range(n):
            for i in range(m):
                if x[i, j].X > 0.5:
                    assignment[str(j)] = i
                    break
    else:
        # Fallback greedy: assign each job to best feasible agent, else best profit
        rem = list(caps)
        for j in range(n):
            best_i, best_p = -1, -1
            for i in range(m):
                if weights[i][j] <= rem[i] and profits[i][j] > best_p:
                    best_i, best_p = i, profits[i][j]
            if best_i < 0:
                best_i = max(range(m), key=lambda i: profits[i][j])
                best_p = profits[best_i][j]
            rem[best_i] -= weights[best_i][j]
            assignment[str(j)] = best_i
            obj_val += best_p
        if logger:
            logger.log_solution(float(obj_val),
                                {"objective_value": float(obj_val),
                                 "assignment": assignment})

    solution = {"objective_value": float(obj_val), "assignment": assignment}
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()