import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_assignment_problem(instance_data, time_limit):
    n = instance_data["n"]
    p = instance_data["p"]
    cost_matrices = instance_data["cost_matrices"]
    reference_point = instance_data["reference_point"]
    search_direction_lambda = instance_data["search_direction_lambda"]

    # Create a new Gurobi model
    model = gp.Model("multi_criteria_assignment")
    
    # Set Gurobi parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('LogToConsole', 0)  # Disable console logging
    
    # Decision variables: x[i][j] = 1 if task i is assigned to agent j, 0 otherwise
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    
    # Additional variable for the worst-case weighted deviation
    w = model.addVar(vtype=GRB.CONTINUOUS, name="worst_case_deviation")
    
    # Constraints: each task assigned to exactly one agent
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"task_{i}")
        
    # Constraints: each agent assigned to exactly one task
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"agent_{j}")
    
    # Constraints: w >= lambda_k * (total_cost_k - ref_k) for each criterion k
    for k in range(p):
        total_cost_k = gp.quicksum(
            cost_matrices[k][i][j] * x[i, j] 
            for i in range(n) 
            for j in range(n)
        )
        model.addConstr(w >= search_direction_lambda[k] * (total_cost_k - reference_point[k]), 
                        name=f"deviation_constraint_{k}")
    
    # Objective: minimize w (the worst-case weighted deviation)
    model.setObjective(w, GRB.MINIMIZE)
    
    # Optimize the model
    model.optimize()
    
    # Extract solution if found
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Get the assignment matrix and assignment list
        assignment_matrix = [[0 for _ in range(n)] for _ in range(n)]
        assignment = [-1] * n
        
        for i in range(n):
            for j in range(n):
                if x[i, j].X > 0.5:  # Threshold to determine if the binary variable is 1
                    assignment_matrix[i][j] = 1
                    assignment[i] = j
        
        objective_value = w.X
        return {
            "objective_value": objective_value,
            "assignment": assignment,
            "assignment_matrix": assignment_matrix
        }
    else:
        # Return empty solution if no feasible solution found
        raise ValueError(f"No feasible solution found. Model status: {model.status}")


def main():
    parser = argparse.ArgumentParser(description='Solve multi-criteria assignment problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, required=False, help='Path to logging file')
    
    args = parser.parse_args()
    
    # Initialize logger if path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_assignment_problem(instance_data, args.time_limit)
    
    # Log final objective value if logging is enabled
    if logger:
        logger.log(solution["objective_value"])
    
    # Write solution to output file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()