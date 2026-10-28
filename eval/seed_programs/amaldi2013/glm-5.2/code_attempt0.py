import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json
import argparse
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    d = instance['d']
    epsilon = instance['epsilon']
    big_M = instance['big_M']
    upper_bound_K = instance['upper_bound_K']
    points = np.array(instance['points'])

    # Initialize Gurobi environment
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('MIPGap', 1e-4)
    env.setParam('NumericFocus', 0)
    env.setParam('Threads', 1)
    env.setParam('NonConvex', 2)
    env.start()
    
    model = gp.Model(env=env)
    model.params.OutputFlag = 0
    # Subtract 1 second to allow time for file I/O and termination
    model.params.TimeLimit = max(1, args.time_limit - 1)
    
    # Decision variables
    y = model.addVars(upper_bound_K, vtype=GRB.BINARY, name="y")
    z = model.addVars(n, upper_bound_K, vtype=GRB.BINARY, name="z")
    w = model.addVars(upper_bound_K, d, lb=-1, ub=1, name="w")
    w0 = model.addVars(upper_bound_K, lb=-big_M, ub=big_M, name="w0")
    
    # Objective: Minimize the number of hyperplanes used
    model.setObjective(gp.quicksum(y[j] for j in range(upper_bound_K)), GRB.MINIMIZE)
    
    # Constraints: Each point must be covered by at least one hyperplane
    for i in range(n):
        model.addConstr(gp.quicksum(z[i, j] for j in range(upper_bound_K)) >= 1)
    
    # Symmetry breaking constraints: hyperplanes are used in order
    for j in range(upper_bound_K - 1):
        model.addConstr(y[j] >= y[j+1])
        
    # Assignment and distance constraints
    for i in range(n):
        for j in range(upper_bound_K):
            model.addConstr(z[i, j] <= y[j])
            wx = gp.quicksum(w[j, k] * points[i, k] for k in range(d))
            # |w^T * x_i - w0| <= epsilon + big_M * (1 - z[i, j])
            model.addConstr(wx - w0[j] <= epsilon + big_M * (1 - z[i, j]))
            model.addConstr(w0[j] - wx <= epsilon + big_M * (1 - z[i, j]))
            
    # Unit-norm constraint for hyperplanes: ||w_j||_2 = 1 if y_j = 1, 0 otherwise
    for j in range(upper_bound_K):
        model.addConstr(gp.quicksum(w[j, k] * w[j, k] for k in range(d)) == y[j])
        
    model.optimize()
    
    solution = {
        "objective_value": upper_bound_K,
        "hyperplanes": [],
        "point_assignments": [[] for _ in range(n)]
    }
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        hyperplanes = []
        for j in range(upper_bound_K):
            if y[j].X > 0.5:
                w_vec = [float(w[j, k].X) for k in range(d)]
                w0_val = float(w0[j].X)
                assigned = []
                for i in range(n):
                    if z[i, j].X > 0.5:
                        assigned.append(i)
                        solution["point_assignments"][i].append(len(hyperplanes))
                hyperplanes.append({
                    "w": w_vec,
                    "w0": w0_val,
                    "assigned_points": assigned
                })
        solution["objective_value"] = float(obj_val)
        solution["hyperplanes"] = hyperplanes
    else:
        # Fallback if Gurobi does not find a solution in time
        hyperplanes = []
        K = min(n, upper_bound_K)
        for i in range(K):
            w_vec = [0.0] * d
            if d > 0:
                w_vec[0] = 1.0
            w0_val = float(points[i, 0]) if d > 0 else 0.0
            hyperplanes.append({
                "w": w_vec,
                "w0": w0_val,
                "assigned_points": [i]
            })
            solution["point_assignments"][i].append(i)
        for i in range(K, n):
            hyperplanes[0]["assigned_points"].append(i)
            solution["point_assignments"][i].append(0)
        solution["objective_value"] = float(len(hyperplanes))
        solution["hyperplanes"] = hyperplanes
        
    if logger:
        logger.log_solution(solution["objective_value"], solution)
        
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()