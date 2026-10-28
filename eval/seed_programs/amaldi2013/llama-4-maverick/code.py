import json
import argparse
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path):
    n = instance['n']
    d = instance['d']
    epsilon = instance['epsilon']
    big_M = instance['big_M']
    upper_bound_K = instance['upper_bound_K']
    points = np.array(instance['points'])

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    # Variables
    z = model.addVars(upper_bound_K, vtype=GRB.BINARY, name='z')
    w = model.addVars(upper_bound_K, d, name='w')
    w0 = model.addVars(upper_bound_K, name='w0')
    x = model.addVars(n, upper_bound_K, vtype=GRB.BINARY, name='x')
    y = model.addVars(n, upper_bound_K, vtype=GRB.BINARY, name='y')

    # Objective: minimize the number of hyperplanes used
    model.setObjective(gp.quicksum(z[k] for k in range(upper_bound_K)), GRB.MINIMIZE)

    # Constraints
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(upper_bound_K)) >= 1)

    for i in range(n):
        for k in range(upper_bound_K):
            model.addConstr(x[i, k] <= z[k])

    for k in range(upper_bound_K):
        model.addConstr(gp.quicksum(w[k, j] * w[k, j] for j in range(d)) == 1)

    for i in range(n):
        for k in range(upper_bound_K):
            model.addConstr(y[i, k] * big_M >= x[i, k])
            point = points[i]
            hyperplane_normal = [w[k, j] for j in range(d)]
            hyperplane_offset = w0[k]
            distance = gp.quicksum(point[j] * hyperplane_normal[j] for j in range(d)) + hyperplane_offset
            model.addConstr(y[i, k] >= distance - epsilon)
            model.addConstr(y[i, k] >= -distance - epsilon)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        hyperplanes = []
        point_assignments = [[] for _ in range(n)]

        for k in range(upper_bound_K):
            if z[k].x > 0.5:
                w_vec = [w[k, j].x for j in range(d)]
                w0_val = w0[k].x
                assigned_points = []
                for i in range(n):
                    if x[i, k].x > 0.5:
                        assigned_points.append(i)
                        point_assignments[i].append(len(hyperplanes))
                hyperplanes.append({
                    'w': w_vec,
                    'w0': w0_val,
                    'assigned_points': assigned_points
                })

        solution = {
            'objective_value': objective_value,
            'hyperplanes': hyperplanes,
            'point_assignments': point_assignments
        }
        return solution
    else:
        raise Exception("Failed to find a feasible solution")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()