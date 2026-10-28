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
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Heuristics', 0.8)
    model.setParam('MIPFocus', 1)
    model.setParam('LazyConstraints', 0)

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # Linearization variables for each pair per knapsack
    y = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                y[i,j,k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")

    # Objective: sum individual profits + pairwise when both in same knapsack
    obj = gp.LinExpr()
    for i in range(n):
        for k in range(m):
            obj.addTerms(profits[i], x[i, k])

    for i in range(n):
        for j in range(i + 1, n):
            pij = pairwise_profits[i][j]
            if pij != 0:
                for k in range(m):
                    obj.addTerms(pij, y[i,j,k])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Each item in at most one knapsack
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"assign_{i}")

    # Capacity constraints
    for k in range(m):
        model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k], name=f"cap_{k}")

    # Linearization constraints for y_ijk
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                yvar = y[i,j,k]
                model.addConstr(yvar <= x[i,k], name=f"y1_{i}_{j}_{k}")
                model.addConstr(yvar <= x[j,k], name=f"y2_{i}_{j}_{k}")
                model.addConstr(yvar >= x[i,k] + x[j,k] - 1, name=f"y3_{i}_{j}_{k}")

    # Warm start with a simple greedy heuristic
    try:
        # Score items by individual profit plus average pairwise
        scores = []
        for i in range(n):
            total_pair = sum(pairwise_profits[i])
            scores.append((profits[i] + total_pair / n, i))
        scores.sort(reverse=True)
        
        knapsacks = [[] for _ in range(m)]
        current_weights = [0] * m
        assigned = [False] * n

        for _, i in scores:
            if assigned[i]:
                continue
            best_k = -1
            best_inc = -1e9
            for k_idx in range(m):
                if current_weights[k_idx] + weights[i] > capacities[k_idx]:
                    continue
                inc = profits[i]
                for j in knapsacks[k_idx]:
                    inc += pairwise_profits[i][j]
                if inc > best_inc:
                    best_inc = inc
                    best_k = k_idx
            if best_k >= 0:
                knapsacks[best_k].append(i)
                current_weights[best_k] += weights[i]
                assigned[i] = True
                # Also "assign" very negative partners to avoid them (heuristic)
                for j_idx in range(n):
                    if not assigned[j_idx] and pairwise_profits[i][j_idx] < -1000:
                        assigned[j_idx] = True

        # Set MIP start
        for k in range(m):
            for i in knapsacks[k]:
                x[i, k].start = 1.0
                for j_idx in knapsacks[k]:
                    if i < j_idx:
                        y[i, j_idx, k].start = 1.0
                    elif j_idx < i:
                        y[j_idx, i, k].start = 1.0
    except Exception:
        pass  # warm start is optional

    start_time = time.time()
    best_obj = -float('inf')
    best_assignment: List[List[int]] = []

    def callback(model, where):
        nonlocal best_obj, best_assignment
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                sol_x = model.cbGetSolution(x)
                assignment = []
                for i_val in range(n):
                    for k_val in range(m):
                        if sol_x[i_val, k_val] > 0.5:
                            assignment.append([i_val, k_val])
                best_assignment = assignment
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    # Extract final solution
    if model.SolCount >= 1:
        objective_value = model.ObjVal
        assignment = []
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment.append([i, k])
    else:
        objective_value = 0.0
        assignment = []

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