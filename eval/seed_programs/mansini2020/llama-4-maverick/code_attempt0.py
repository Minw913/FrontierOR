import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance_data = json.load(file)
    return instance_data

def solve_instance(instance_data, time_limit, log_path):
    n_groups = instance_data['parameters']['n_groups']
    m_resources = instance_data['parameters']['m_resources']
    n_items_per_group = instance_data['parameters']['n_items_per_group']
    capacities = instance_data['capacities']
    groups = instance_data['groups']

    model = gp.Model("ItemSelection")
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Create variables
    vars_x = {}
    for group in groups:
        group_id = group['group_id']
        vars_x[group_id] = {}
        for item in group['items']:
            item_id = item['item_id']
            vars_x[group_id][item_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{group_id}_{item_id}")

    # Objective function: Maximize total profit
    objective_expr = gp.quicksum(
        vars_x[group['group_id']][item['item_id']] * item['profit']
        for group in groups
        for item in group['items']
    )
    model.setObjective(objective_expr, GRB.MAXIMIZE)

    # Constraints: Select exactly one item per group
    for group in groups:
        group_id = group['group_id']
        constraint_expr = gp.quicksum(vars_x[group_id][item['item_id']] for item in group['items'])
        model.addConstr(constraint_expr == 1, name=f"one_item_per_group_{group_id}")

    # Constraints: Respect resource capacities
    for res_id in range(m_resources):
        resource_constraint_expr = gp.quicksum(
            vars_x[group['group_id']][item['item_id']] * item['weights'][res_id]
            for group in groups
            for item in group['items']
        )
        model.addConstr(resource_constraint_expr <= capacities[res_id], name=f"resource_capacity_{res_id}")

    # Optimize the model
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    model.optimize(lambda model, where: callback(model, where, logger))

    # Extract and return the solution
    if model.SolCount > 0:
        objective_value = model.getObjective().getValue()
        selected_items = extract_solution(vars_x, groups)
        return objective_value, selected_items
    else:
        return None, None

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

def extract_solution(vars_x, groups):
    selected_items = []
    for group in groups:
        group_id = group['group_id']
        for item in group['items']:
            item_id = item['item_id']
            if vars_x[group_id][item_id].X > 0.5:  # Considered selected if value is > 0.5
                selected_item = {
                    "group_id": group_id,
                    "item_id": item_id,
                    "profit": item['profit'],
                    "weights": item['weights']
                }
                selected_items.append(selected_item)
    return selected_items

def write_solution(solution_path, objective_value, selected_items):
    solution_data = {
        "objective_value": objective_value,
        "selected_items": selected_items
    }
    with open(solution_path, 'w') as file:
        json.dump(solution_data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Solve the item selection problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    instance_data = read_instance(args.instance_path)
    objective_value, selected_items = solve_instance(instance_data, args.time_limit, args.log_path)

    if objective_value is not None and selected_items is not None:
        write_solution(args.solution_path, objective_value, selected_items)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()