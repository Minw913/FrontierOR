import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_crew_scheduling(instance_data):
    num_rows = instance_data['dimensions']['num_rows']
    num_cols = instance_data['dimensions']['num_cols']
    cost_vector = instance_data['cost_vector']
    constraint_matrix_A = instance_data['constraint_matrix_A']
    has_base_constraints = instance_data['has_base_constraints']
    
    # Create model
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)  # Silent mode
    
    # Decision variables: binary variables indicating whether each rotation is selected
    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x")
    
    # Objective function: minimize total cost
    obj_expr = gp.quicksum(cost_vector[j] * x[j] for j in range(num_cols))
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Coverage constraints: every flight leg must be covered exactly once
    for i in range(num_rows):
        # Find all rotations that cover flight leg i
        covering_rotations = []
        for j in range(num_cols):
            if i in constraint_matrix_A['columns'][j]:
                covering_rotations.append(j)
        
        # Add constraint: sum of selected rotations covering this flight leg = 1
        model.addConstr(gp.quicksum(x[j] for j in covering_rotations) == 1, 
                      name=f"coverage_leg_{i}")
    
    # Base constraints: enforce labor utilization rules
    if has_base_constraints:
        base_constraints_data = instance_data['base_constraints']
        D_matrix = base_constraints_data['D_matrix']['rows']
        lower_bounds_d1 = base_constraints_data['lower_bounds_d1']
        upper_bounds_d2 = base_constraints_data['upper_bounds_d2']
        num_bases = base_constraints_data['num_bases']
        
        for b in range(num_bases):
            # Weighted sum of hours away from base for base b
            base_expr = gp.quicksum(D_matrix[b][j] * x[j] for j in range(num_cols))
            model.addConstr(base_expr >= lower_bounds_d1[b], 
                          name=f"base_lower_{b}")
            model.addConstr(base_expr <= upper_bounds_d2[b], 
                          name=f"base_upper_{b}")

    return model, x


def main():
    parser = argparse.ArgumentParser(description='Solve crew scheduling problem.')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the input instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Time limit for solving in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path for logging intermediate solutions (JSONL)')
    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Build and solve the model
    model, variables = solve_crew_scheduling(instance_data)
    
    # Register callback for logging if available
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Set time limit and optimize
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    if logger:
        model.optimize(log_callback)
    else:
        model.optimize()

    # Extract solution
    selected_rotations = []
    variable_values = {}
    for j in range(len(variables)):
        val = variables[j].X
        variable_values[f"{j}"] = val
        if abs(val - 1.0) < 1e-6:  # Selected rotation
            selected_rotations.append(j)
    
    # Prepare solution dictionary
    solution = {
        "objective_value": model.ObjVal if model.SolCount > 0 else None,
        "selected_rotations": selected_rotations,
        "variable_values": variable_values
    }

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()