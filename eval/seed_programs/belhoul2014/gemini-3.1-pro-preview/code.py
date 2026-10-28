import argparse
import json
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def evaluate_assignment(c_ind, p, c_matrices, q, lam):
    max_dev = -float('inf')
    row_ind = np.arange(len(c_ind))
    for k in range(p):
        cost = c_matrices[k, row_ind, c_ind].sum()
        dev = float(lam[k]) * (cost - float(q[k]))
        if dev > max_dev:
            max_dev = dev
    return float(max_dev)

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = int(data["n"])
    p = int(data["p"])
    c_matrices = np.array(data["cost_matrices"], dtype=np.float64)
    q = np.array(data["reference_point"], dtype=np.float64)
    lam = np.array(data["search_direction_lambda"], dtype=np.float64)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    try:
        from scipy.optimize import linear_sum_assignment
        has_scipy = True
    except ImportError:
        has_scipy = False

    unique_starts = set()
    
    if has_scipy:
        # Optimize each criterion independently
        for k in range(p):
            r, c_ind = linear_sum_assignment(c_matrices[k])
            unique_starts.add(tuple(c_ind))

        # Optimize the base weighted sum
        avg_matrix = np.zeros((n, n), dtype=np.float64)
        for k in range(p):
            avg_matrix += lam[k] * c_matrices[k]
        r, c_ind = linear_sum_assignment(avg_matrix)
        unique_starts.add(tuple(c_ind))
        
        # Optimize over random convex combinations of weights
        rng = np.random.RandomState(42)
        for _ in range(20):
            # Guard to leave sufficient time for exact solving
            if args.time_limit and (time.time() - start_time) > args.time_limit * 0.2:
                break
            w = rng.dirichlet(np.ones(p))
            avg_matrix = np.zeros((n, n), dtype=np.float64)
            for k in range(p):
                avg_matrix += w[k] * lam[k] * c_matrices[k]
            r, c_ind = linear_sum_assignment(avg_matrix)
            unique_starts.add(tuple(c_ind))
    else:
        # Fallback greedy heuristic if SciPy is not available
        def greedy_assignment(matrix):
            assigned_cols = set()
            c_ind = [-1] * n
            for i in range(n):
                best_j = -1
                best_val = float('inf')
                for j in range(n):
                    if j not in assigned_cols and matrix[i, j] < best_val:
                        best_val = matrix[i, j]
                        best_j = j
                c_ind[i] = best_j
                assigned_cols.add(best_j)
            return tuple(c_ind)

        for k in range(p):
            unique_starts.add(greedy_assignment(c_matrices[k]))
        
        avg_matrix = np.zeros((n, n), dtype=np.float64)
        for k in range(p):
            avg_matrix += lam[k] * c_matrices[k]
        unique_starts.add(greedy_assignment(avg_matrix))

    env = gp.Env(empty=True)
    env.start()
    model = gp.Model("MinimaxAssignment", env=env)
    
    if args.time_limit:
        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            remaining_time = 0.5
        model.setParam('TimeLimit', remaining_time)
    
    # Restrict to a single CPU core as per requirements
    model.setParam('Threads', 1)

    # Variables
    x_vars = [[model.addVar(vtype=GRB.BINARY) for j in range(n)] for i in range(n)]
    Z = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Z")

    # Assignment constraints: one-to-one strictly
    for i in range(n):
        model.addConstr(gp.quicksum(x_vars[i][j] for j in range(n)) == 1)
    for j in range(n):
        model.addConstr(gp.quicksum(x_vars[i][j] for i in range(n)) == 1)

    flat_x = [x_vars[i][j] for i in range(n) for j in range(n)]

    # Bound Z against the scaled evaluation of each criteria's assignment cost
    for k in range(p):
        flat_c = [float(val) for val in c_matrices[k].ravel()]
        lam_k = float(lam[k])
        q_k = float(q[k])
        
        cost_k = gp.LinExpr()
        cost_k.addTerms(flat_c, flat_x)
        
        model.addConstr(Z >= lam_k * cost_k - lam_k * q_k)

    model.setObjective(Z, GRB.MINIMIZE)

    model._best_obj = float('inf')
    model._logger = logger
    best_st = None
    
    # Evaluate heuristics locally prior to MIP
    for st in unique_starts:
        obj = evaluate_assignment(st, p, c_matrices, q, lam)
        if obj < model._best_obj - 1e-6:
            model._best_obj = obj
            best_st = st
            if model._logger:
                model._logger.log(obj)

    # Establish Warm Starts into Gurobi
    if unique_starts:
        model.NumStart = len(unique_starts)
        model.update()
        for start_idx, st in enumerate(unique_starts):
            model.params.StartNumber = start_idx
            for i in range(n):
                for j in range(n):
                    if st[i] == j:
                        x_vars[i][j].Start = 1.0
                    else:
                        x_vars[i][j].Start = 0.0

    # Hook callback for subsequent solving iterations
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < model._best_obj - 1e-6:
                model._best_obj = obj
                if model._logger:
                    model._logger.log(obj)

    model.optimize(cb)

    # Parse best solution (or fallback to explicitly registered heuristic)
    if model.SolCount > 0:
        assignment = [-1] * n
        assignment_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if x_vars[i][j].X > 0.5:
                    assignment[i] = j
                    assignment_matrix[i][j] = 1
        
        # Use discrete assignment to generate clean precise float output without rounding errors
        obj_val = evaluate_assignment(assignment, p, c_matrices, q, lam)
        if obj_val < model._best_obj:  # Guarantee recording final valid best
            if model._logger:
                model._logger.log(obj_val)
    else:
        obj_val = float(model._best_obj)
        assignment = list(best_st)
        assignment_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            assignment_matrix[i][assignment[i]] = 1

    ans = {
        "objective_value": float(obj_val),
        "assignment": assignment,
        "assignment_matrix": assignment_matrix
    }

    with open(args.solution_path, 'w') as f:
        json.dump(ans, f, indent=2)

if __name__ == "__main__":
    main()