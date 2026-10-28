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
    p = instance['p']
    cost_matrices = instance['cost_matrices']
    reference_point = instance['reference_point']
    search_direction_lambda = instance['search_direction_lambda']

    # Precompute total cost coefficients
    # cost[k][i][j] already given

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        model = gp.Model("MultiCriteriaAssignment")
        model.setParam('TimeLimit', args.time_limit)
        model.setParam('Threads', 1)
        model.setParam('LogToConsole', 0)

        # Variables
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
        z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z")

        # Objective
        model.setObjective(z, GRB.MINIMIZE)

        # Assignment constraints
        for i in range(n):
            model.addConstr(gp.quicksum(x[i,j] for j in range(n)) == 1, f"task_{i}")
        for j in range(n):
            model.addConstr(gp.quicksum(x[i,j] for i in range(n)) == 1, f"agent_{j}")

        # z constraints for each criterion
        for k in range(p):
            total_cost = gp.quicksum(cost_matrices[k][i][j] * x[i,j] for i in range(n) for j in range(n))
            weighted_dev = search_direction_lambda[k] * (total_cost - reference_point[k])
            model.addConstr(z >= weighted_dev, f"z_{k}")

        # Callback for logging incumbents
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if logger:
                    logger.log(float(obj))

        # Optimize
        model.optimize(callback)

        # Extract solution
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            objective_value = model.objVal
            assignment = [-1] * n
            assignment_matrix = [[0] * n for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if x[i,j].X > 0.5:
                        assignment[i] = j
                        assignment_matrix[i][j] = 1

            solution = {
                "objective_value": float(objective_value),
                "assignment": assignment,
                "assignment_matrix": assignment_matrix
            }

            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
        else:
            # Fallback: if no solution, create a random feasible assignment
            import numpy as np
            perm = np.random.permutation(n).tolist()
            assignment_matrix = [[0] * n for _ in range(n)]
            for i in range(n):
                assignment_matrix[i][perm[i]] = 1
            
            # Compute objective
            totals = [0] * p
            for k in range(p):
                for i in range(n):
                    totals[k] += cost_matrices[k][i][perm[i]]
            weighted_devs = [search_direction_lambda[k] * (totals[k] - reference_point[k]) for k in range(p)]
            obj_val = max(weighted_devs)
            
            solution = {
                "objective_value": float(obj_val),
                "assignment": perm,
                "assignment_matrix": assignment_matrix
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)

    except Exception as e:
        # Emergency fallback
        import numpy as np
        perm = np.random.permutation(n).tolist()
        assignment_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            assignment_matrix[i][perm[i]] = 1
        totals = [0] * p
        for k in range(p):
            for i in range(n):
                totals[k] += cost_matrices[k][i][perm[i]]
        weighted_devs = [search_direction_lambda[k] * (totals[k] - reference_point[k]) for k in range(p)]
        obj_val = max(weighted_devs) if p > 0 else 0.0
        
        solution = {
            "objective_value": float(obj_val),
            "assignment": perm,
            "assignment_matrix": assignment_matrix
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()