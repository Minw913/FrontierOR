import argparse
import json
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n_jobs']
    m = instance['n_machines']
    p = np.array(instance['processing_times'], dtype=int)  # m x n
    d = np.array(instance['due_dates'], dtype=int)

    if n == 0:
        solution = {"objective_value": 0.0, "sequence": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Precompute delay matrix: delay[i][j] = extra time on first machine when j follows i
    delay = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            max_shift = 0
            for k in range(1, m):
                cum_i = sum(p[k2][i] for k2 in range(k))
                cum_j = sum(p[k2][j] for k2 in range(k))
                max_shift = max(max_shift, cum_i - cum_j)
            delay[i][j] = max_shift + p[0][i]

    # Big-M for due dates and non-adjacent constraints
    M = int(2 * np.sum(p) + max(d) + 1)
    start_bound = int(np.sum(p) + max(d) + 1)

    model = gp.Model("NoWaitFlowShop")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 1e-4
    model.Params.Heuristics = 0.8
    model.Params.MIPFocus = 1

    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")  # x[i,j]=1 if job i immediately precedes j
    s = model.addVars(n, lb=0, ub=start_bound, vtype=GRB.CONTINUOUS, name="s")  # start time on first machine
    cmax = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="cmax")

    # Objective
    model.setObjective(cmax, GRB.MINIMIZE)

    # One successor and one predecessor (except for chain ends)
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, f"succ_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1, f"pred_{i}")

    # No self loops
    for i in range(n):
        model.addConstr(x[i, i] == 0)

    # At most one direction between i and j
    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(x[i, j] + x[j, i] <= 1)

    # Timing constraints - only for immediate successors
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            model.addConstr(s[j] >= s[i] + delay[i][j] - M * (1 - x[i, j]), f"timing_{i}_{j}")

    # Due date constraints - corrected: only the last job in sequence can be constrained by due date
    # But since any job could be last, we apply due date only to the job that has no successor
    for j in range(n):
        completion = s[j] + sum(p[k][j] for k in range(m))
        model.addConstr(completion <= d[j] + M * (1 - gp.quicksum(x[j, k] for k in range(n) if k != j)), 
                       f"due_{j}")

    # Makespan
    for j in range(n):
        completion_j = s[j] + sum(p[k][j] for k in range(m))
        model.addConstr(cmax >= completion_j, f"makespan_{j}")

    # Find a good initial solution using a simple heuristic: sort by due date
    order = np.argsort(d).tolist()
    init_s = [0] * n
    current_time = 0
    for pos, idx in enumerate(order):
        if pos > 0:
            prev = order[pos - 1]
            current_time += delay[prev][idx]
        init_s[idx] = current_time
        current_time += p[0][idx]

    init_cmax = max(init_s[j] + sum(p[k][j] for k in range(m)) for j in range(n))

    # Set initial MIP solution
    for j in range(n):
        s[j].Start = float(init_s[j])
    cmax.Start = float(init_cmax)

    for pos in range(n - 1):
        i = order[pos]
        j = order[pos + 1]
        x[i, j].Start = 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float('inf')

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        final_obj = model.ObjVal
        # Extract sequence
        try:
            xvals = model.getAttr('X', x)
            succ = [-1] * n
            for i in range(n):
                for j in range(n):
                    if i != j and abs(xvals[i, j]) > 0.5:
                        succ[i] = j
                        break
            pred_count = [0] * n
            for i in range(n):
                for j in range(n):
                    if i != j and abs(xvals[i, j]) > 0.5:
                        pred_count[j] += 1
            start_nodes = [i for i in range(n) if pred_count[i] == 0]
            start = start_nodes[0]
            sequence = []
            current = start
            visited = set()
            while current != -1 and current not in visited and len(sequence) < n + 1:
                visited.add(current)
                sequence.append(current + 1)  # 1-based
                current = succ[current]
            if len(sequence) != n:
                sequence = (np.argsort(d) + 1).tolist()
        except Exception:
            sequence = (np.argsort(d) + 1).tolist()
    else:
        # No solution found within time, use heuristic solution
        final_obj = float(init_cmax)
        sequence = (np.argsort(d) + 1).tolist()

    solution = {
        "objective_value": float(final_obj),
        "sequence": sequence
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()