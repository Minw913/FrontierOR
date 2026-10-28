import argparse
import json
import time
from gurobipy import *
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Crew Pairing Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    cost_vector = instance['cost_vector']
    columns = instance['constraint_matrix_A']['columns']
    has_base_constraints = instance['has_base_constraints']

    # Build row-to-columns mapping for efficient constraint construction
    row_to_cols = [[] for _ in range(num_rows)]
    for j in range(num_cols):
        for i in columns[j]:
            row_to_cols[i].append(j)

    # Build Gurobi model
    model = Model("crew_pairing")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x", lb=0.0, ub=1.0)

    # Objective: minimize total cost
    model.setObjective(quicksum(cost_vector[j] * x[j] for j in range(num_cols)), GRB.MINIMIZE)

    # Coverage constraints (set partitioning): each flight leg covered exactly once
    for i in range(num_rows):
        model.addConstr(
            quicksum(x[j] for j in row_to_cols[i]) == 1,
            name=f"cover_{i}"
        )

    # Base constraints
    if has_base_constraints:
        base_data = instance['base_constraints']
        num_bases = base_data['num_bases']
        D_matrix = base_data['D_matrix']['rows']
        lower_bounds = base_data['lower_bounds_d1']
        upper_bounds = base_data['upper_bounds_d2']

        for k in range(num_bases):
            # Only include variables with non-zero coefficients for efficiency
            nonzero_js = [j for j in range(num_cols) if D_matrix[k][j] != 0.0]
            if nonzero_js:
                expr = quicksum(D_matrix[k][j] * x[j] for j in nonzero_js)
                model.addConstr(expr >= lower_bounds[k], name=f"base_lb_{k}")
                model.addConstr(expr <= upper_bounds[k], name=f"base_ub_{k}")

    model.update()

    # Callback to log incumbent solutions
    best_obj = float('inf')

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj:
                best_obj = obj
                sol_vals = model.cbGetSolution(x)
                selected = [j for j in range(num_cols) if sol_vals[j] > 0.5]
                variable_values = {str(j): (1.0 if j in selected else 0.0) for j in range(num_cols)}
                solution_dict = {
                    "objective_value": obj,
                    "selected_rotations": selected,
                    "variable_values": variable_values
                }
                if logger:
                    logger.log_solution(obj, solution_dict)

    # Optimize
    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        selected = [j for j in range(num_cols) if x[j].X > 0.5]
        obj_val = sum(cost_vector[j] for j in selected)
    else:
        # Fallback: try to construct a feasible solution greedily
        selected = []
        obj_val = 0.0

    variable_values = {str(j): (1.0 if j in selected else 0.0) for j in range(num_cols)}
    solution = {
        "objective_value": obj_val,
        "selected_rotations": selected,
        "variable_values": variable_values
    }

    # Log final solution
    if logger:
        logger.log_solution(obj_val, solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()