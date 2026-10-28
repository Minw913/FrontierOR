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
    model.Params.MIPGap = 0.01
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

    # Due date constraints
    for j in range(n):
        completion = s[j] + sum(p[k][j] for k in range(m))
        model.addConstr(completion <= d[j] + M * (1 - gp.quicksum(x[i, j] for i in range(n) if i != j)), 
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
            current_time = max(current_time, init_s[prev] + delay[prev][idx])
        init_s[idx] = current_time
        current_time += p[0][idx]

    init_cmax = max(init_s[j] + sum(p[k][j] for k in range(m)) for j in range(n))

    # Set initial solution
    for j in range(n):
        model.getVarByName(f"s[{j}]").Start = float(init_s[j])
    model.getVarByName("cmax").Start = float(init_cmax)

    for pos in range(n - 1):
        i = order[pos]
        j = order[pos + 1]
        model.getVarByName(f"x[{i},{j}]").Start = 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float('inf')
    best_sequence = None
    incumbent_count = 0

    def callback(model, where):
        nonlocal best_obj, best_sequence, incumbent_count
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if obj < best_obj - 1e-6:
                best_obj = obj
                incumbent_count += 1
                if logger and incumbent_count % 3 == 0:  # limit logging frequency
                    logger.log(float(obj))
                
                # Extract sequence
                try:
                    xvals = model.cbGetSolution(model.getAttr('X', x))
                    succ = [-1] * n
                    for i in range(n):
                        for j in range(n):
                            if i != j and xvals[i, j] > 0.5:
                                succ[i] = j
                                break
                    # Build sequence starting from node with no predecessor
                    pred_count = [0] * n
                    for i in range(n):
                        for j in range(n):
                            if i != j and xvals[i, j] > 0.5:
                                pred_count[j] += 1
                    start = next(i for i in range(n) if pred_count[i] == 0)
                    seq = []
                    current = start
                    while current != -1:
                        seq.append(current + 1)  # 1-based
                        current = succ[current]
                    if len(seq) == n:
                        best_sequence = seq
                except:
                    pass

    model.optimize(callback)

    # Final solution extraction
    if model.SolCount > 0:
        final_obj = model.ObjVal
        # Extract sequence from final solution
        sequence = []
        try:
            xvals = model.getAttr('X', x)
            succ = [-1] * n
            for i_idx in range(n):
                for j_idx in range(n):
                    if i_idx != j_idx and xvals[i_idx, j_idx] > 0.5:
                        succ[i_idx] = j_idx
                        break
            pred_count = [0] * n
            for i in range(n):
                for j in range(n):
                    if i != j and xvals[i, j] > 0.5:
                        pred_count[j] += 1
            start = next(i for i in range(n) if pred_count[i] == 0)
            current = start
            while current != -1 and len(sequence) < n:
                sequence.append(current + 1)
                current = succ[current]
        except:
            # Fallback: use sorted by due date
            sequence = (np.argsort(d) + 1).tolist()
            final_obj = init_cmax

        if not sequence or len(sequence) != n:
            sequence = (np.argsort(d) + 1).tolist()
            final_obj = init_cmax
    else:
        # No solution found, use heuristic
        sequence = (np.argsort(d) + 1).tolist()
        final_obj = float(init_cmax)

    solution = {
        "objective_value": float(final_obj),
        "sequence": sequence
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and best_obj < float('inf'):
        logger.log(float(best_obj))


if __name__ == "__main__":
    main()