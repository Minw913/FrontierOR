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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_groups = instance['parameters']['n_groups']
    m_resources = instance['parameters']['m_resources']
    capacities = instance['capacities']
    groups = instance['groups']

    # Preprocess: group items
    group_items = []
    for g in groups:
        items = []
        for item in g['items']:
            items.append({
                'item_id': item['item_id'],
                'profit': item['profit'],
                'weights': item['weights']
            })
        group_items.append(items)

    # Build model
    model = gp.Model("MultipleChoiceKnapsack")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    # Binary variables: x[g][i] = 1 if item i from group g is selected
    x = []
    for g in range(n_groups):
        x_g = []
        for idx in range(len(group_items[g])):
            x_g.append(model.addVar(vtype=GRB.BINARY, name=f"x_{g}_{idx}"))
        x.append(x_g)

    # Exactly one per group
    for g in range(n_groups):
        model.addConstr(gp.quicksum(x[g]) == 1, name=f"one_per_group_{g}")

    # Resource constraints
    for r in range(m_resources):
        total_use = gp.quicksum(
            x[g][idx] * group_items[g][idx]['weights'][r]
            for g in range(n_groups)
            for idx in range(len(group_items[g]))
        )
        model.addConstr(total_use <= capacities[r], name=f"resource_{r}")

    # Objective: maximize profit
    total_profit = gp.quicksum(
        x[g][idx] * group_items[g][idx]['profit']
        for g in range(n_groups)
        for idx in range(len(group_items[g]))
    )
    model.setObjective(total_profit, GRB.MAXIMIZE)

    # Callback for logging incumbents
    best_obj = -float('inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    selected = []
    objective_value = 0.0
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        if model.SolCount > 0:
            objective_value = model.ObjVal
            for g in range(n_groups):
                for idx, var in enumerate(x[g]):
                    if abs(var.X) > 0.5:
                        item = group_items[g][idx]
                        selected.append({
                            "group_id": groups[g]['group_id'],
                            "item_id": item['item_id'],
                            "profit": item['profit'],
                            "weights": item['weights']
                        })
                        break

    # Write solution
    solution = {
        "objective_value": float(objective_value),
        "selected_items": selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Final log if not already logged
    if logger and objective_value > -float('inf'):
        logger.log(float(objective_value))

if __name__ == "__main__":
    main()