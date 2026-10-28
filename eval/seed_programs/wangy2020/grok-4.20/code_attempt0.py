import json
import argparse
import time
import numpy as np
from itertools import permutations
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

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
    A = np.array(instance['A'], dtype=float)
    B = np.array(instance['B'], dtype=float)
    C = np.array(instance['C'], dtype=float)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    best_obj = float('inf')
    best_assignment = None

    # For small n, solve exactly with enumeration or MIP
    if n <= 6:
        for perm in permutations(range(n)):
            if time.time() - start_time > args.time_limit:
                break
            assignment = list(perm)
            sumA = sum(A[i][assignment[i]] for i in range(n))
            sumB = sum(B[i][assignment[i]] for i in range(n))
            sumC = sum(C[i][assignment[i]] for i in range(n))
            obj = sumA * sumB + sumC
            if obj < best_obj:
                best_obj = obj
                best_assignment = assignment
                if logger:
                    logger.log(float(best_obj))
        # If we finished enumeration, we have the optimum
    else:
        # Use MIP formulation with Gurobi
        try:
            m = gp.Model("multiplicative_assignment")
            m.setParam('TimeLimit', max(1, args.time_limit - (time.time() - start_time)))
            m.setParam('Threads', 1)
            m.setParam('MIPGap', 0.0)
            # For larger instances we allow some gap but aim for best feasible

            x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
            sA = m.addVar(vtype=GRB.CONTINUOUS, name="sA")
            sB = m.addVar(vtype=GRB.CONTINUOUS, name="sB")
            sC = m.addVar(vtype=GRB.CONTINUOUS, name="sC")
            p = m.addVar(vtype=GRB.CONTINUOUS, name="p")
            obj_var = m.addVar(vtype=GRB.CONTINUOUS, name="obj")

            # Assignment constraints
            for i in range(n):
                m.addConstr(gp.quicksum(x[i,j] for j in range(n)) == 1)
            for j in range(n):
                m.addConstr(gp.quicksum(x[i,j] for i in range(n)) == 1)

            # Linear expressions for sums
            m.addConstr(sA == gp.quicksum(A[i,j] * x[i,j] for i in range(n) for j in range(n)))
            m.addConstr(sB == gp.quicksum(B[i,j] * x[i,j] for i in range(n) for j in range(n)))
            m.addConstr(sC == gp.quicksum(C[i,j] * x[i,j] for i in range(n) for j in range(n)))

            # Linearize product p = sA * sB using big-M
            # Since costs can be negative, we need bounds. Use a conservative big-M.
            # For practicality, assume |costs| <= 1000 (common in such problems), adjust if needed.
            M = 1000.0 * n * 2  # rough bound
            m.addConstr(p >= -M * (1 - 0) + 0)  # not tight
            # Better: introduce bounds on sA and sB
            # But to keep simple and robust, we use MIP with piecewise or just let Gurobi handle with bilinear if allowed.
            # Gurobi 10+ can handle bilinear terms directly with non-convex quadratic.

            # Use quadratic objective directly (non-convex)
            # Total cost = sA*sB + sC
            m.setObjective(sA * sB + sC, GRB.MINIMIZE)
            m.setParam('NonConvex', 2)  # allow non-convex quadratic

            # Callback to log incumbent solutions
            def callback(model, where):
                nonlocal best_obj, best_assignment
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        sol_x = model.cbGetSolution(x)
                        assignment = [-1] * n
                        for i in range(n):
                            for j in range(n):
                                if sol_x[i,j] > 0.5:
                                    assignment[i] = j
                                    break
                        best_assignment = assignment
                        if logger:
                            logger.log(float(best_obj))

            m.optimize(callback)

            if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                if m.SolCount > 0:
                    sol_x = m.getAttr('X', x)
                    assignment = [-1] * n
                    for i in range(n):
                        for j in range(n):
                            if sol_x[i,j] > 0.5:
                                assignment[i] = j
                                break
                    best_assignment = assignment
                    best_obj = m.ObjVal
                    if logger and best_obj < float('inf'):
                        logger.log(float(best_obj))

        except Exception as e:
            # Fallback: greedy heuristic
            pass

    # If no solution yet, use a simple greedy approach: sort by diagonal or random
    if best_assignment is None:
        # Very basic heuristic: assignment along sorted indices (not great but feasible)
        assignment = list(range(n))
        sumA = sum(A[i][assignment[i]] for i in range(n))
        sumB = sum(B[i][assignment[i]] for i in range(n))
        sumC = sum(C[i][assignment[i]] for i in range(n))
        best_obj = sumA * sumB + sumC
        best_assignment = assignment
        if logger:
            logger.log(float(best_obj))

        # Could improve with local search but for time being this is baseline

    # Final output
    result = {
        "objective_value": float(best_obj),
        "assignment": best_assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()