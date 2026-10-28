import argparse
import json
import math
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve_quadratic_ufl(instance_data, time_limit, log_path):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_facilities = instance_data["num_facilities"]
    num_customers = instance_data["num_customers"]
    opening_costs = instance_data["opening_costs"]
    allocation_costs = instance_data["allocation_costs"]
    
    # Create model
    model = gp.Model("quadratic_ufl")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")  # Facility open/close
    x = model.addVars(num_facilities, num_customers, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")  # Assignment fractions
    
    # Constraints
    # Each customer's demand must be fully satisfied
    for j in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_facilities)) == 1, f"demand_satisfaction_{j}")
    
    # A customer can only be served by an open facility
    for i in range(num_facilities):
        for j in range(num_customers):
            model.addConstr(x[i, j] <= y[i], f"service_constraint_{i}_{j}")
    
    # Objective function: minimize total cost
    # Total cost = opening costs + quadratic allocation costs
    obj_expr = gp.quicksum(opening_costs[i] * y[i] for i in range(num_facilities))
    
    # Add quadratic allocation costs
    for i in range(num_facilities):
        for j in range(num_customers):
            obj_expr += allocation_costs[i][j] * x[i, j] * x[i, j]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Callback to handle intermediate solutions
    def solution_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get current solution values
            current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            
            # Only log if we have a better solution
            if logger:
                # Extract current solution
                y_vals = [int(model.cbGetSolution(y[i])) for i in range(num_facilities)]
                x_vals = [[model.cbGetSolution(x[i, j]) for j in range(num_customers)] for i in range(num_facilities)]
                
                solution_dict = {
                    "objective_value": current_obj,
                    "variant": "quadratic",
                    "y": y_vals,
                    "x": x_vals
                }
                
                logger.log_solution(current_obj, solution_dict)
    
    # Set callback
    model.optimize(solution_callback)
    
    # Prepare final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        # Extract solution values
        y_vals = [int(y[i].X) for i in range(num_facilities)]
        x_vals = [[x[i, j].X for j in range(num_customers)] for i in range(num_facilities)]
        
        solution_dict = {
            "objective_value": obj_val,
            "variant": "quadratic",
            "y": y_vals,
            "x": x_vals
        }
        
        return solution_dict
    else:
        # No solution found - return empty solution
        y_vals = [0] * num_facilities
        x_vals = [[0.0 for _ in range(num_customers)] for _ in range(num_facilities)]
        
        solution_dict = {
            "objective_value": float('inf'),
            "variant": "quadratic",
            "y": y_vals,
            "x": x_vals
        }
        
        return solution_dict


def main():
    parser = argparse.ArgumentParser(description='Solve Quadratic Uncapacitated Facility Location Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, required=False, help='Path to log intermediate solutions')
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_quadratic_ufl(instance_data, args.time_limit, args.log_path)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()