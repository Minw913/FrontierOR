import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve_instance(instance_data, time_limit, logger):
    num_slabs = instance_data['num_slabs']
    num_orders = instance_data['num_orders']
    num_colors = instance_data['num_colors']
    max_colors_per_knapsack = instance_data['max_colors_per_knapsack']
    slab_weights = instance_data['slab_weights']
    order_weights = instance_data['order_weights']
    order_colors = instance_data['order_colors']
    eligible_slabs_per_order = instance_data['eligible_slabs_per_order']

    model = gp.Model("SlabAssignment")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Decision Variables
    assign_vars = model.addVars([(i, j) for i in range(num_orders) for j in eligible_slabs_per_order[i]], vtype=GRB.BINARY, name='assign')
    slab_in_service = model.addVars(range(num_slabs), vtype=GRB.BINARY, name='slab_in_service')
    color_used = model.addVars([(j, c) for j in range(num_slabs) for c in range(num_colors)], vtype=GRB.BINARY, name='color_used')

    # Objective Function
    model.setObjective(
        gp.quicksum(2 * order_weights[i] * assign_vars[i, j] for i in range(num_orders) for j in eligible_slabs_per_order[i]) -
        gp.quicksum(slab_weights[j] * slab_in_service[j] for j in range(num_slabs)),
        GRB.MAXIMIZE
    )

    # Constraints
    for i in range(num_orders):
        model.addConstr(gp.quicksum(assign_vars[i, j] for j in eligible_slabs_per_order[i]) <= 1, name=f'order_assigned_once_{i}')

    for j in range(num_slabs):
        model.addConstr(gp.quicksum(order_weights[i] * assign_vars[i, j] for i in range(num_orders) if j in eligible_slabs_per_order[i]) <= slab_weights[j] * slab_in_service[j], name=f'slab_capacity_{j}')

    for j in range(num_slabs):
        for i in range(num_orders):
            if j in eligible_slabs_per_order[i]:
                model.addConstr(assign_vars[i, j] <= slab_in_service[j], name=f'assign_implies_slab_in_service_{i}_{j}')

    for j in range(num_slabs):
        for c in range(num_colors):
            model.addConstr(color_used[j, c] >= gp.quicksum(assign_vars[i, j] for i in range(num_orders) if order_colors[i] == c and j in eligible_slabs_per_order[i]), name=f'color_used_{j}_{c}')

    for j in range(num_slabs):
        model.addConstr(gp.quicksum(color_used[j, c] for c in range(num_colors)) <= max_colors_per_knapsack, name=f'max_colors_{j}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        assignments = []
        slabs_used = []

        for i in range(num_orders):
            for j in eligible_slabs_per_order[i]:
                if j in range(num_slabs) and assign_vars[i, j].X > 0.5:
                    assignments.append({'order': i, 'slab': j})

        for j in range(num_slabs):
            if slab_in_service[j].X > 0.5:
                slabs_used.append(j)

        if logger:
            logger.log(objective_value)

        return objective_value, assignments, slabs_used
    else:
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Solve Slab Assignment Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    objective_value, assignments, slabs_used = solve_instance(instance_data, args.time_limit, logger)

    if objective_value is not None:
        solution = {
            'objective_value': objective_value,
            'assignments': assignments,
            'slabs_used': slabs_used
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()