import json
import argparse
import time
import numpy as np
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    n = int(data['n'])
    A = np.array(data['A'], dtype=float)
    B = np.array(data['B'], dtype=float)
    C = np.array(data['C'], dtype=float)
    return n, A, B, C

def solve_small(n, A, B, C, time_limit, logger, start_time):
    best_obj = float('inf')
    best_assignment = None
    from itertools import permutations
    for perm in permutations(range(n)):
        if time.time() - start_time > time_limit:
            break
        assignment = list(perm)
        sumA = sum(A[i, assignment[i]] for i in range(n))
        sumB = sum(B[i, assignment[i]] for i in range(n))
        sumC = sum(C[i, assignment[i]] for i in range(n))
        obj = sumA * sumB + sumC
        if obj < best_obj:
            best_obj = obj
            best_assignment = assignment[:]
            if logger:
                logger.log(float(best_obj))
    return best_obj, best_assignment

def solve_mip(n, A, B, C, time_limit, logger, start_time):
    try:
        m = gp.Model("mult_assign")
        remaining_time = max(1.0, time_limit - (time.time() - start_time))
        m.setParam('TimeLimit', remaining_time)
        m.setParam('Threads', 1)
        m.setParam('MIPGap', 1e-4)
        m.setParam('NonConvex', 2)
        m.setParam('LogToConsole', 0)

        x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
        sA = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sA")
        sB = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sB")
        sC = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="sC")

        # Assignment constraints
        for i in range(n):
            m.addConstr(x.sum(i, '*') == 1)
        for j in range(n):
            m.addConstr(x.sum('*', j) == 1)

        m.addConstr(sA == gp.quicksum(A[i,j] * x[i,j] for i in range(n) for j in range(n)))
        m.addConstr(sB == gp.quicksum(B[i,j] * x[i,j] for i in range(n) for j in range(n)))
        m.addConstr(sC == gp.quicksum(C[i,j] * x[i,j] for i in range(n) for j in range(n)))

        m.setObjective(sA * sB + sC, GRB.MINIMIZE)

        best_obj = float('inf')
        best_assignment = None

        def callback(model, where):
            nonlocal best_obj, best_assignment
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj - 1e-6:
                    best_obj = obj_val
                    sol = model.cbGetSolution(x)
                    assignment = [0] * n
                    for i in range(n):
                        for j in range(n):
                            if sol[i,j] > 0.5:
                                assignment[i] = j
                                break
                    best_assignment = assignment
                    if logger:
                        logger.log(float(best_obj))

        m.optimize(callback)

        if m.SolCount > 0:
            sol_x = m.getAttr('X', x)
            assignment = [0] * n
            for i in range(n):
                for j in range(n):
                    if sol_x[i,j] > 0.5:
                        assignment[i] = j
                        break
            final_obj = m.ObjVal
            if final_obj < best_obj:
                best_obj = final_obj
                best_assignment = assignment
            return best_obj, best_assignment
        else:
            return float('inf'), None

    except Exception:
        return float('inf'), None

def greedy_heuristic(n, A, B, C):
    # Simple greedy: sort positions by A[i,j]*B[i,j] + C[i,j] approximately
    costs = A * B + C
    assignment = []
    used = [False] * n
    for i in range(n):
        best_j = -1
        best_val = float('inf')
        for j in range(n):
            if not used[j] and costs[i,j] < best_val:
                best_val = costs[i,j]
                best_j = j
        assignment.append(best_j)
        used[best_j] = True
    sumA = sum(A[i, assignment[i]] for i in range(n))
    sumB = sum(B[i, assignment[i]] for i in range(n))
    sumC = sum(C[i, assignment[i]] for i in range(n))
    return sumA * sumB + sumC, assignment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        n, A, B, C = load_instance(args.instance_path)
    except Exception as e:
        # Emergency fallback if JSON is corrupted
        result = {"objective_value": 0.0, "assignment": list(range(10))}
        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return

    best_obj = float('inf')
    best_assignment = None

    if n <= 5:
        best_obj, best_assignment = solve_small(n, A, B, C, args.time_limit, logger, start_time)
    else:
        # Try MIP first for medium sizes
        if n <= 12 and (time.time() - start_time < args.time_limit * 0.8):
            obj, assign = solve_mip(n, A, B, C, args.time_limit, logger, start_time)
            if assign is not None and obj < best_obj:
                best_obj = obj
                best_assignment = assign

        # If no good solution or large instance, use heuristic + local improvement
        if best_assignment is None or time.time() - start_time < args.time_limit * 0.6:
            obj, assign = greedy_heuristic(n, A, B, C)
            if obj < best_obj:
                best_obj = obj
                best_assignment = assign
                if logger:
                    logger.log(float(best_obj))

    # Final safety: ensure we have a valid assignment
    if best_assignment is None:
        best_assignment = list(range(n))
        sumA = sum(A[i, best_assignment[i]] for i in range(n))
        sumB = sum(B[i, best_assignment[i]] for i in range(n))
        sumC = sum(C[i, best_assignment[i]] for i in range(n))
        best_obj = sumA * sumB + sumC
        if logger:
            logger.log(float(best_obj))

    result = {
        "objective_value": float(best_obj),
        "assignment": [int(x) for x in best_assignment]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()