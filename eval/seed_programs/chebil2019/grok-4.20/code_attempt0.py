import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['N']
    C = instance['knapsack_capacity']
    families = instance['families']

    # Prepare data
    family_list = []
    for fam in families:
        family_list.append({
            'id': fam['family_id'],
            'setup_cost': fam['setup_cost'],
            'setup_cap': fam['setup_capacity'],
            'items': fam['items']
        })

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Create Gurobi model
    model = gp.Model("FamilyKnapsack")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.0)
    model.setParam('LogToConsole', 0)

    # Variables
    y = {}  # y[f] = 1 if family f activated
    x = {}  # x[f][i] = 1 if item i in family f selected

    for f_idx, fam in enumerate(family_list):
        f_id = fam['id']
        y[f_id] = model.addVar(vtype=GRB.BINARY, name=f"y_{f_id}")
        x[f_id] = {}
        for item in fam['items']:
            i_id = item['item_id']
            x[f_id][i_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{f_id}_{i_id}")

    # Objective: maximize sum profits - sum setup_costs
    profit_expr = gp.quicksum(
        item['profit'] * x[fam['id']][item['item_id']]
        for fam in family_list
        for item in fam['items']
    )
    setup_expr = gp.quicksum(
        fam['setup_cost'] * y[fam['id']]
        for fam in family_list
    )
    model.setObjective(profit_expr - setup_expr, GRB.MAXIMIZE)

    # Capacity constraint
    weight_expr = gp.quicksum(
        item['weight'] * x[fam['id']][item['item_id']]
        for fam in family_list
        for item in fam['items']
    )
    setup_cap_expr = gp.quicksum(
        fam['setup_cap'] * y[fam['id']]
        for fam in family_list
    )
    model.addConstr(weight_expr + setup_cap_expr <= C, name="capacity")

    # Linking constraints: items only if family activated
    for fam in family_list:
        f_id = fam['id']
        for item in fam['items']:
            i_id = item['item_id']
            model.addConstr(x[f_id][i_id] <= y[f_id], name=f"link_{f_id}_{i_id}")

    # Callback to log incumbent solutions
    best_obj = -float('inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj > best_obj + 1e-6:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        objective_value = model.objVal if model.objVal is not None else 0.0
        families_selected = []
        items_selected = []

        for fam in family_list:
            f_id = fam['id']
            if y[f_id].X > 0.5:
                families_selected.append(f_id)
                for item in fam['items']:
                    i_id = item['item_id']
                    if x[f_id][i_id].X > 0.5:
                        items_selected.append({
                            "family": f_id,
                            "item": i_id
                        })

        solution = {
            "objective_value": float(objective_value),
            "families_selected": families_selected,
            "items_selected": items_selected
        }
    else:
        # Infeasible or no solution
        solution = {
            "objective_value": 0.0,
            "families_selected": [],
            "items_selected": []
        }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()