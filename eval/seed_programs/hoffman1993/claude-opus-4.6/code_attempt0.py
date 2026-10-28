import argparse
import json
import time
import sys

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    cost_vector = instance['cost_vector']
    columns = instance['constraint_matrix_A']['columns']
    has_base_constraints = instance.get('has_base_constraints', False)
    
    # Build the model
    model = gp.Model("crew_scheduling")
    model.setParam('TimeLimit', max(1, time_limit - int(time.time() - start_time) - 2))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
    
    # Decision variables: binary, one per rotation
    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x")
    
    # Set objective: minimize total cost
    model.setObjective(
        gp.quicksum(cost_vector[j] * x[j] for j in range(num_cols)),
        GRB.MINIMIZE
    )
    
    # Build row-to-columns mapping for coverage constraints
    row_to_cols = [[] for _ in range(num_rows)]
    for j in range(num_cols):
        for row_idx in columns[j]:
            row_to_cols[row_idx].append(j)
    
    # Exact coverage constraints: each flight leg covered exactly once
    for i in range(num_rows):
        cols_covering_i = row_to_cols[i]
        if len(cols_covering_i) == 0:
            # Infeasible - no rotation covers this flight leg
            # Add impossible constraint to signal infeasibility
            model.addConstr(gp.LinExpr(0) == 1, name=f"cover_{i}")
        else:
            model.addConstr(
                gp.quicksum(x[j] for j in cols_covering_i) == 1,
                name=f"cover_{i}"
            )
    
    # Base constraints
    if has_base_constraints and instance.get('base_constraints'):
        base_data = instance['base_constraints']
        num_bases = base_data['num_bases']
        D_matrix = base_data['D_matrix']['rows']
        lower_bounds = base_data['lower_bounds_d1']
        upper_bounds = base_data['upper_bounds_d2']
        
        for b in range(num_bases):
            d_row = D_matrix[b]
            # Find non-zero entries for efficiency
            nonzero_cols = [(j, d_row[j]) for j in range(num_cols) if d_row[j] != 0.0]
            
            if nonzero_cols:
                expr = gp.quicksum(coef * x[j] for j, coef in nonzero_cols)
                model.addConstr(expr >= lower_bounds[b], name=f"base_lb_{b}")
                model.addConstr(expr <= upper_bounds[b], name=f"base_ub_{b}")
    
    # Track incumbent solutions via callback
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-6:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        selected = []
        var_values = {}
        for j in range(num_cols):
            val = x[j].X
            if val > 0.5:
                selected.append(j)
                var_values[str(j)] = 1.0
            else:
                var_values[str(j)] = 0.0
        
        solution = {
            "objective_value": obj_val,
            "selected_rotations": selected,
            "variable_values": var_values
        }
    else:
        # No feasible solution found
        solution = {
            "objective_value": float('inf'),
            "selected_rotations": [],
            "variable_values": {str(j): 0.0 for j in range(num_cols)}
        }
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description="Crew Scheduling Optimization")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()