#!/usr/bin/env python3
import argparse
import json
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    knapsack_capacity = instance['knapsack_capacity']
    families_data = instance['families']
    
    logger = SolutionLogger(args.log_path, "maximize") if args.log_path else None

    try:
        model = gp.Model()
        model.setParam('TimeLimit', args.time_limit)
        model.setParam('Threads', 1)
        model.setParam('LogToConsole', 0)
        
        num_families = len(families_data)
        y = model.addVars(num_families, vtype=GRB.BINARY, name="y")
        x = []
        for idx, family in enumerate(families_data):
            n_items = len(family['items'])
            x_var = model.addVars(n_items, vtype=GRB.BINARY, name=f"x_F{family['family_id']}")
            x.append(x_var)

        for fam_idx, family in enumerate(families_data):
            for item_idx in range(len(family['items'])):
                model.addConstr(x[fam_idx][item_idx] <= y[fam_idx], 
                                name=f"link_fam{fam_idx}_item{item_idx}")

        total_capacity = gp.LinExpr(0)
        for fam_idx, family in enumerate(families_data):
            total_capacity += y[fam_idx] * family['setup_capacity']
            for item_idx, item in enumerate(family['items']):
                total_capacity += x[fam_idx][item_idx] * item['weight']
        model.addConstr(total_capacity <= knapsack_capacity, "capacity")

        total_profit = gp.LinExpr(0)
        for fam_idx, family in enumerate(families_data):
            total_profit -= y[fam_idx] * family['setup_cost']
            for item_idx, item in enumerate(family['items']):
                total_profit += x[fam_idx][item_idx] * item['profit']
        model.setObjective(total_profit, GRB.MAXIMIZE)

        if logger is not None:
            def callback(ml, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = ml.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
            model.optimize(callback)
        else:
            model.optimize()

        solution = {
            "objective_value": 0.0,
            "families_selected": [],
            "items_selected": []
        }

        if model.SolCount > 0:
            solution['objective_value'] = model.ObjVal
            for fam_idx, family in enumerate(families_data):
                if y[fam_idx].X > 0.5:
                    solution['families_selected'].append(family['family_id'])
            for fam_idx, family in enumerate(families_data):
                family_id = family['family_id']
                for item_idx, item in enumerate(family['items']):
                    if x[fam_idx][item_idx].X > 0.5:
                        solution['items_selected'].append({"family": family_id, "item": item['item_id']})

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

    except Exception as e:
        solution = {"objective_value": 0.0, "families_selected": [], "items_selected": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        raise

if __name__ == "__main__":
    main()