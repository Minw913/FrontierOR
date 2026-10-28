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
    capacity = instance['capacity']
    weights = instance['weights']
    profit_matrix = instance['profit_matrix']

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()
    best_obj = -float('inf')
    best_sol = None

    # Model
    model = gp.Model("QuadraticKnapsack")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    x = [model.addVar(vtype=GRB.BINARY, name=f'x_{i}') for i in range(n)]

    # Objective: maximize sum_{i} p_ii * x_i + sum_{i<j} p_ij * x_i * x_j
    obj = gp.QuadExpr()
    for i in range(n):
        obj.addTerms(profit_matrix[i][i], x[i])
        for j in range(i+1, n):
            if profit_matrix[i][j] != 0:
                obj.addTerms(profit_matrix[i][j], x[i], x[j])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Weight constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)

    # Callback to log improved solutions
    def callback(model, where):
        nonlocal best_obj, best_sol
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                sol = [int(model.cbGetSolution(x[i]) > 0.5) for i in range(n)]
                best_sol = sol
                if logger:
                    logger.log(float(obj_val))
                # Early termination check
                elapsed = time.time() - start_time
                if elapsed > args.time_limit:
                    model.terminate()

    try:
        model.optimize(callback)
    except:
        pass

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        solution = [int(x[i].X > 0.5) for i in range(n)]
    else:
        # Fallback: greedy by profit/weight if no solution found
        profit_density = []
        for i in range(n):
            p = profit_matrix[i][i]
            w = weights[i]
            if w > 0:
                profit_density.append((p / w, i))
            else:
                profit_density.append((float('inf'), i))
        profit_density.sort(reverse=True)
        
        current_weight = 0
        selected = [0] * n
        for _, i in profit_density:
            if current_weight + weights[i] <= capacity:
                selected[i] = 1
                current_weight += weights[i]
        
        # Compute objective
        objective_value = 0.0
        for i in range(n):
            if selected[i]:
                objective_value += profit_matrix[i][i]
        for i in range(n):
            for j in range(i+1, n):
                if selected[i] and selected[j]:
                    objective_value += profit_matrix[i][j]
        
        solution = selected
        if logger:
            logger.log(float(objective_value))

    result = {
        "objective_value": float(objective_value),
        "solution": solution
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()