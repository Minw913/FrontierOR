import argparse
import json
import math
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve_p_median(instance_path, solution_path, time_limit, log_path):
    # Load instance
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    
    n = instance_data['n']
    p = instance_data['p']
    cost_matrix = instance_data['cost_matrix']
    
    # Initialize the logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Create Gurobi model
    model = gp.Model("p_median")
    model.Params.LogToConsole = 0  # Suppress console output
    model.Params.TimeLimit = time_limit
    
    # Add decision variables
    # x[i][j] = 1 if customer i is assigned to median j
    x = []
    for i in range(n):
        x_row = []
        for j in range(n):
            var_name = f"x_{i}_{j}"
            x_ij = model.addVar(vtype=GRB.BINARY, name=var_name)
            x_row.append(x_ij)
        x.append(x_row)

    # y[j] = 1 if node j is selected as a median
    y = []
    for j in range(n):
        var_name = f"y_{j}"
        y_j = model.addVar(vtype=GRB.BINARY, name=var_name)
        y.append(y_j)
    
    # Set objective function
    obj_terms = []
    for i in range(n):
        for j in range(n):
            cost_ij = cost_matrix[i][j]
            obj_terms.append(cost_ij * x[i][j])
    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    
    # Add constraints
    # Each customer must be assigned to exactly one median
    for i in range(n):
        model.addConstr(gp.quicksum(x[i][j] for j in range(n)) == 1)
    
    # Customer can only be assigned to nodes selected as medians
    for i in range(n):
        for j in range(n):
            model.addConstr(x[i][j] <= y[j])
    
    # Exactly p facilities must be opened
    model.addConstr(gp.quicksum(y[j] for j in range(n)) == p)
    
    # Solve the model
    start_time = time.time()
    model.optimize()
    
    # Prepare solution based on current best solution available
    if model.SolCount > 0:
        # Get values of the binary variables
        y_vals = [round(y[j].x) for j in range(n)]
        opened_facilities = [j for j in range(n) if y_vals[j] == 1]
        
        # Extract assignments
        assignments = []
        for i in range(n):
            facility_idx = -1
            for j in range(n):
                if round(x[i][j].x) == 1:
                    facility_idx = j
                    break
            assert facility_idx != -1, f"Customer {i} not assigned to any facility!"
            assignment_dict = {
                "customer": i,
                "facility": facility_idx,
                "cost": cost_matrix[i][facility_idx]
            }
            assignments.append(assignment_dict)
        
        # Compute objective value
        objective_value = int(model.ObjVal)
        
        # Log the solution if logger is enabled
        if logger:
            logger.log(objective_value)
        
        # Prepare solution dictionary
        solution = {
            "objective_value": objective_value,
            "opened_facilities": opened_facilities,
            "assignments": assignments
        }
    else:
        # If no solution was found within time limit, we return empty result (this shouldn't happen unless infeasible)
        solution = {
            "objective_value": float('inf'),
            "opened_facilities": [],
            "assignments": []
        }
    
    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    
    args = parser.parse_args()
    
    solve_p_median(args.instance_path, args.solution_path, args.time_limit, args.log_path)