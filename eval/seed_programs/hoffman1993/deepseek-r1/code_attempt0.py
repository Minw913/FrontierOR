import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import os
import time

def main():
    parser = argparse.ArgumentParser(description='Crew Rotation Assignment Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output JSON solution file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to JSONL log file for intermediate solutions')
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    dimensions = instance_data['dimensions']
    num_rows = dimensions['num_rows']
    num_cols = dimensions['num_cols']
    cost_vector = instance_data['cost_vector']
    constraint_matrix_A = instance_data['constraint_matrix_A']['columns']
    has_base_constraints = instance_data['has_base_constraints']
    
    # Initialize solution logger if log_path is provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    # Precompute leg_cover: for each flight leg, list of rotations that cover it
    leg_cover = [[] for _ in range(num_rows)]
    for j in range(num_cols):
        for leg_index in constraint_matrix_A[j]:
            leg_cover[leg_index].append(j)
    
    # Create Gurobi model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Create variables
    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x")
    
    # Set objective: minimize total cost
    model.setObjective(gp.quicksum(cost_vector[j] * x[j] for j in range(num_cols)), GRB.MINIMIZE)
    
    # Add set partitioning constraints
    for i in range(num_rows):
        if leg_cover[i]:
            model.addConstr(gp.quicksum(x[j] for j in leg_cover[i]) == 1, name=f"leg_{i}")
    
    # Add base constraints if present
    if has_base_constraints:
        base_data = instance_data['base_constraints']
        num_bases = base_data['num_bases']
        D_matrix = base_data['D_matrix']
        lower_bounds = base_data['lower_bounds_d1']
        upper_bounds = base_data['upper_bounds_d2']
        
        for k in range(num_bases):
            expr = gp.quicksum(D_matrix[k][j] * x[j] for j in range(num_cols))
            model.addConstr(expr >= lower_bounds[k], name=f"base_{k}_lower")
            model.addConstr(expr <= upper_bounds[k], name=f"base_{k}_upper")
    
    # Callback function for logging incumbents
    def gurobi_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
    
    # Optimize model
    model.optimize(gurobi_callback if logger else None)
    
    # Extract solution
    if model.SolCount > 0:
        objective_value = model.objVal
        selected_rotations = [j for j in range(num_cols) if x[j].X > 0.5]
        variable_values = {str(j): float(x[j].X) for j in range(num_cols)}
    else:
        objective_value = 0.0
        selected_rotations = []
        variable_values = {str(j): 0.0 for j in range(num_cols)}
    
    # Write solution to file
    solution = {
        "objective_value": objective_value,
        "selected_rotations": selected_rotations,
        "variable_values": variable_values
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()