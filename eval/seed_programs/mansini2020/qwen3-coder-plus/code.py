import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_mckp(instance_data, time_limit, log_path=None):
    # Initialize the logger if provided
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    n_groups = instance_data['parameters']['n_groups']
    m_resources = instance_data['parameters']['m_resources']
    capacities = instance_data['capacities']
    
    # Create model
    model = gp.Model("Multiple_Choice_Knapsack_Problem")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    
    # Variables: x[g][i] binary indicating whether item i in group g is selected
    x = []
    for g_idx in range(n_groups):
        group_vars = []
        group_items = instance_data['groups'][g_idx]['items']
        
        for i_idx in range(len(group_items)):
            var = model.addVar(vtype=GRB.BINARY, name=f"x_{g_idx}_{i_idx}")
            group_vars.append(var)
        x.append(group_vars)
    
    # Constraint 1: Exactly one item must be selected from each group
    for g_idx in range(n_groups):
        model.addConstr(gp.quicksum(x[g_idx]) == 1, f"group_constraint_{g_idx}")
    
    # Constraint 2: Respect resource limits
    for r_idx in range(m_resources):
        resource_usage = gp.LinExpr()
        for g_idx in range(n_groups):
            group_items = instance_data['groups'][g_idx]['items']
            for i_idx in range(len(group_items)):
                weight = group_items[i_idx]['weights'][r_idx]
                resource_usage += weight * x[g_idx][i_idx]
        
        model.addConstr(resource_usage <= capacities[r_idx], f"resource_constraint_{r_idx}")
    
    # Objective: Maximize total profit
    total_profit = gp.LinExpr()
    for g_idx in range(n_groups):
        group_items = instance_data['groups'][g_idx]['items']
        for i_idx in range(len(group_items)):
            profit = group_items[i_idx]['profit']
            total_profit += profit * x[g_idx][i_idx]
    
    model.setObjective(total_profit, GRB.MAXIMIZE)
    
    # Callback to log solutions as they're found
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    # Optimize
    model.optimize(callback)
    
    # Prepare solution
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        selected_items = []
        objective_value = 0
        
        for g_idx in range(n_groups):
            group_items = instance_data['groups'][g_idx]['items']
            for i_idx in range(len(group_items)):
                if x[g_idx][i_idx].X > 0.5:  # Binary choice is selected
                    item = group_items[i_idx]
                    selected_items.append({
                        "group_id": g_idx,
                        "item_id": i_idx,
                        "profit": item['profit'],
                        "weights": item['weights']
                    })
                    objective_value += item['profit']
                    break  # Only one item per group
    
        result = {
            "objective_value": objective_value,
            "selected_items": selected_items
        }
    else:
        # Return empty solution if infeasible or other status
        result = {
            "objective_value": 0,
            "selected_items": []
        }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_mckp(instance_data, args.time_limit, args.log_path)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()