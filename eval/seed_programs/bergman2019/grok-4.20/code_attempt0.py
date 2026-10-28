import json
import argparse
import time
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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
    profits = instance['profits']
    pairwise_profits = instance['pairwise_profits']
    weights = instance['weights']
    capacities = instance['capacities']

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model = gp.Model("QuadraticKnapsack")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)
    model.setParam('Heuristics', 0.8)
    model.setParam('MIPFocus', 1)

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    z = model.addVars(n, m, vtype=GRB.BINARY, name="z")

    # Objective: sum individual profits + pairwise when both in same knapsack
    obj = gp.LinExpr()
    for i in range(n):
        for k in range(m):
            obj.add(x[i, k] * profits[i])

    for i in range(n):
        for j in range(i + 1, n):
            pij = pairwise_profits[i][j]
            if pij != 0:
                for k in range(m):
                    obj.add(pij * z[i, k] * z[j, k])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Each item in at most one knapsack
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1)

    # Capacity constraints
    for k in range(m):
        model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k])

    # Linearization constraints for z_ik = x_ik (since only one k per i if assigned)
    for i in range(n):
        for k in range(m):
            model.addConstr(z[i, k] <= x[i, k])
            model.addConstr(z[i, k] <= 1 - (gp.quicksum(x[i, kk] for kk in range(m) if kk != k)))

    # Warm start with greedy
    try:
        items = list(range(n))
        items.sort(key=lambda i: profits[i] + sum(pairwise_profits[i]), reverse=True)
        
        knapsacks = [[] for _ in range(m)]
        current_weights = [0] * m
        current_profit = 0.0
        assigned = set()

        for i in items:
            if i in assigned:
                continue
            best_k = -1
            best_delta = -float('inf')
            for k in range(m):
                w_sum = current_weights[k] + weights[i]
                if w_sum > capacities[k]:
                    continue
                delta = profits[i]
                for j in knapsacks[k]:
                    delta += pairwise_profits[i][j]
                if delta > best_delta:
                    best_delta = delta
                    best_k = k
            if best_k != -1 and best_delta > -100000:  # arbitrary threshold to avoid very negative
                knapsacks[best_k].append(i)
                current_weights[best_k] += weights[i]
                current_profit += best_delta
                assigned.add(i)
                for other in knapsacks[best_k]:
                    if other != i:
                        assigned.add(other)  # not really, just marking for greedy

        # Apply warm start
        for k in range(m):
            for i in knapsacks[k]:
                x[i, k].start = 1.0
                z[i, k].start = 1.0
        if current_profit > -1e8:
            model.setParam('StartNumber', 1)
    except:
        pass

    start_time = time.time()
    best_obj = -float('inf')
    best_assignment = []

    def callback(model, where):
        nonlocal best_obj, best_assignment
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                sol_x = model.cbGetSolution(x)
                assignment = []
                for i in range(n):
                    for k in range(m):
                        if sol_x[i, k] > 0.5:
                            assignment.append([i, k])
                best_assignment = assignment
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SOLUTION_LIMIT):
        if model.SolCount > 0:
            objective_value = model.ObjVal
            assignment = []
            for i in range(n):
                for k in range(m):
                    if x[i, k].X > 0.5:
                        assignment.append([i, k])
        else:
            objective_value = 0.0
            assignment = []
    else:
        objective_value = 0.0
        assignment = []

    # Final logging
    if logger and objective_value > -1e10:
        logger.log(float(objective_value))

    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()