import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    capacity = instance['capacity']
    weights = instance['weights']
    values = instance['values']
    pairwise = instance['pairwise_values']

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model = gp.Model("QuadraticMultipleKnapsack")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)
    model.setParam('Heuristics', 0.8)
    model.setParam('MIPFocus', 1)

    x = {}
    for i in range(n):
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

    y = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")

    # Objective
    obj = gp.LinExpr()
    for i in range(n):
        for k in range(m):
            obj += values[i] * x[i, k]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                obj += pairwise[i][j] * y[i, j, k]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraints
    for k in range(m):
        model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity)

    # Linking constraints for y
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                model.addConstr(y[i, j, k] <= x[i, k])
                model.addConstr(y[i, j, k] <= x[j, k])
                model.addConstr(y[i, j, k] >= x[i, k] + x[j, k] - 1)

    # Each item in at most one knapsack
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1)

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj_val))

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    assignment = {}
    if model.SolCount > 0:
        sol_obj = model.objVal
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment[f"item_{i}"] = k
                    break
    else:
        sol_obj = 0.0

    solution = {
        "objective_value": float(sol_obj),
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and model.SolCount > 0:
        logger.log(float(sol_obj))

if __name__ == "__main__":
    main()