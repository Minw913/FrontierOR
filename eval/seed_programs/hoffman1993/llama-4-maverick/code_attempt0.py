import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    cost_vector = instance['cost_vector']
    constraint_matrix_A = instance['constraint_matrix_A']['columns']
    has_base_constraints = instance['has_base_constraints']

    model = gp.Model('crew_rotation')
    model.Params.LogToConsole = 0  # Silence Gurobi output

    rotation_vars = model.addVars(num_cols, vtype=GRB.BINARY, name='rotation')

    # Objective function: minimize total cost
    model.setObjective(gp.quicksum(cost_vector[j] * rotation_vars[j] for j in range(num_cols)), GRB.MINIMIZE)

    # Exact coverage constraints
    for i in range(num_rows):
        covering_rotations = [j for j in range(num_cols) if i in constraint_matrix_A[j]]
        model.addConstr(gp.quicksum(rotation_vars[j] for j in covering_rotations) == 1, name=f'coverage_{i}')

    if has_base_constraints:
        num_bases = instance['base_constraints']['num_bases']
        D_matrix = instance['base_constraints']['D_matrix']['rows']
        lower_bounds_d1 = instance['base_constraints']['lower_bounds_d1']
        upper_bounds_d2 = instance['base_constraints']['upper_bounds_d2']

        for b in range(num_bases):
            weighted_sum = gp.quicksum(D_matrix[b][j] * rotation_vars[j] for j in range(num_cols))
            model.addConstr(weighted_sum >= lower_bounds_d1[b], name=f'base_lower_{b}')
            model.addConstr(weighted_sum <= upper_bounds_d2[b], name=f'base_upper_{b}')

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected_rotations = [j for j in range(num_cols) if rotation_vars[j].X > 0.5]
        objective_value = model.getObjective().getValue()
        variable_values = {str(j): rotation_vars[j].X for j in range(num_cols)}
        return objective_value, selected_rotations, variable_values
    else:
        raise RuntimeError(f'Gurobi model failed to find a feasible solution within the time limit. Status: {model.status}')

def write_solution(solution_path, objective_value, selected_rotations, variable_values):
    solution = {
        'objective_value': objective_value,
        'selected_rotations': selected_rotations,
        'variable_values': variable_values
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    global args
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    model = gp.Model('crew_rotation')
    model.Params.LogToConsole = 0  # Silence Gurobi output

    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    cost_vector = instance['cost_vector']
    constraint_matrix_A = instance['constraint_matrix_A']['columns']
    has_base_constraints = instance['has_base_constraints']

    rotation_vars = model.addVars(num_cols, vtype=GRB.BINARY, name='rotation')

    # Objective function: minimize total cost
    model.setObjective(gp.quicksum(cost_vector[j] * rotation_vars[j] for j in range(num_cols)), GRB.MINIMIZE)

    # Exact coverage constraints
    for i in range(num_rows):
        covering_rotations = [j for j in range(num_cols) if i in constraint_matrix_A[j]]
        model.addConstr(gp.quicksum(rotation_vars[j] for j in covering_rotations) == 1, name=f'coverage_{i}')

    if has_base_constraints:
        num_bases = instance['base_constraints']['num_bases']
        D_matrix = instance['base_constraints']['D_matrix']['rows']
        lower_bounds_d1 = instance['base_constraints']['lower_bounds_d1']
        upper_bounds_d2 = instance['base_constraints']['upper_bounds_d2']

        for b in range(num_bases):
            weighted_sum = gp.quicksum(D_matrix[b][j] * rotation_vars[j] for j in range(num_cols))
            model.addConstr(weighted_sum >= lower_bounds_d1[b], name=f'base_lower_{b}')
            model.addConstr(weighted_sum <= upper_bounds_d2[b], name=f'base_upper_{b}')

    model.Params.TimeLimit = args.time_limit

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    model.optimize(callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected_rotations = [j for j in range(num_cols) if rotation_vars[j].X > 0.5]
        objective_value = model.getObjective().getValue()
        variable_values = {str(j): rotation_vars[j].X for j in range(num_cols)}
        write_solution(args.solution_path, objective_value, selected_rotations, variable_values)
    else:
        raise RuntimeError(f'Gurobi model failed to find a feasible solution within the time limit. Status: {model.status}')

if __name__ == '__main__':
    main()