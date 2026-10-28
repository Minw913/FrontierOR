import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_gfsk_with_gurobi(instance_data, time_limit, log_path):
    """
    Solves the Generalized Family Selection Knapsack problem using Gurobi.
    """
    N = instance_data['N']
    knapsack_capacity = instance_data['knapsack_capacity']

    # Preprocess data: collect all items and their attributes
    family_setup_costs = {}
    family_setup_capacities = {}
    n_items_per_family = {}

    all_items_data = []  # List of (family_id, item_index, weight, profit)
    
    for family in instance_data['families']:
        fid = family['family_id']
        family_setup_costs[fid] = family['setup_cost']
        family_setup_capacities[fid] = family['setup_capacity']
        n_items_per_family[fid] = family['n_i']
        
        for idx, item in enumerate(family['items']):
            all_items_data.append((fid, idx, item['weight'], item['profit']))

    # Create model
    model = gp.Model('Generalized_Family_Selection_Knapsack')
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)

    # Decision variables:
    # 1. Binary variable whether a family is selected 
    family_vars = {}  # f_vars[family_id] indicates if family is selected
    for fid in family_setup_costs:
        family_vars[fid] = model.addVar(vtype=GRB.BINARY, name=f'y_{fid}')

    # 2. Binary variable whether an item is selected (implies family selection)
    item_vars = {}  # i_vars[(family_id, item_id)] indicates if item is selected
    for fid, idx, weight, profit in all_items_data:
        item_vars[(fid, idx)] = model.addVar(vtype=GRB.BINARY, name=f'x_{fid}_{idx}')
    
    # Add constraint: an item can only be selected if its family is selected
    for fid, idx, weight, profit in all_items_data:
        model.addConstr(item_vars[(fid, idx)] <= family_vars[fid])

    # Capacity constraint: sum of selected item weights + family setup capacities <= knapsack capacity
    capacity_expr = gp.LinExpr()
    for fid, s_cap in family_setup_capacities.items():
        capacity_expr += s_cap * family_vars[fid]
        
    for fid, idx, weight, profit in all_items_data:
        capacity_expr += weight * item_vars[(fid, idx)]
    
    model.addConstr(capacity_expr <= knapsack_capacity, name="Capacity_constraint")

    # Objective: max sum of item profits - sum of selected family setup costs
    obj_expr = gp.LinExpr()
    # Add profit terms for selected items
    for fid, idx, weight, profit in all_items_data:
        obj_expr += profit * item_vars[(fid, idx)]
    # Subtract setup cost terms for active families
    for fid, s_cost in family_setup_costs.items():
        obj_expr -= s_cost * family_vars[fid]

    model.setObjective(obj_expr, sense=GRB.MAXIMIZE)

    # Set up logger if provided
    logger = None
    if log_path:
        logger = SolutionLogger(log_path, sense="maximize")

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            # New integer feasible solution found
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(mycallback)

    # Extract solution
    status = model.Status

    if status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE:
        raise Exception("Model is infeasible or unbounded.")
    elif status == GRB.UNBOUNDED:
        raise Exception("Model is unbounded.")

    # Now extract the result whether optimal or not (e.g. due to timeout)
    objective_value = model.ObjVal if model.SolCount > 0 else 0.0
    
    # Get selected families
    families_selected = []
    for fid, var in family_vars.items():
        if var.X > 0.5:  # if binary variable is set to 1
            families_selected.append(fid)

    # Get selected items
    items_selected = []
    for (fid, idx), var in item_vars.items():
        if var.X > 0.5:  # if binary variable is set to 1
            items_selected.append({
                "family": fid,
                "item": idx
            })

    solution_output = {
        "objective_value": float(objective_value),
        "families_selected": families_selected,
        "items_selected": items_selected
    }

    return solution_output


def main():
    parser = argparse.ArgumentParser(description='Solve GFSK problem with Gurobi.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the output JSON file.')
    parser.add_argument('--time_limit', type=int, default=300, help='Maximum allowed runtime in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to save logs in JSONL format. Optional.')

    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_gfsk_with_gurobi(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()