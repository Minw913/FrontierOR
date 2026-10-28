import argparse
import json
import time
import numpy as np

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    start_time = time.time()
    
    n_groups = instance['parameters']['n_groups']
    m_resources = instance['parameters']['m_resources']
    n_items_per_group = instance['parameters']['n_items_per_group']
    capacities = instance['capacities']
    groups = instance['groups']
    
    # Pre-process data into numpy arrays for efficiency
    profits = np.zeros((n_groups, n_items_per_group), dtype=np.int64)
    weights = np.zeros((n_groups, n_items_per_group, m_resources), dtype=np.int64)
    
    for g in groups:
        gid = g['group_id']
        for item in g['items']:
            iid = item['item_id']
            profits[gid][iid] = item['profit']
            weights[gid][iid] = item['weights']
    
    caps = np.array(capacities, dtype=np.int64)
    
    # Try Gurobi for exact/near-optimal solution
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining > 1:
            model = gp.Model("MMKP")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = remaining - 0.5
            model.Params.Threads = 1
            
            # Create binary variables
            x = {}
            for g in range(n_groups):
                for i in range(n_items_per_group):
                    x[g, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{g}_{i}")
            
            model.update()
            
            # Objective: maximize total profit
            model.setObjective(
                gp.quicksum(profits[g][i] * x[g, i] 
                           for g in range(n_groups) 
                           for i in range(n_items_per_group)),
                GRB.MAXIMIZE
            )
            
            # Exactly one item per group
            for g in range(n_groups):
                model.addConstr(
                    gp.quicksum(x[g, i] for i in range(n_items_per_group)) == 1
                )
            
            # Resource constraints
            for r in range(m_resources):
                model.addConstr(
                    gp.quicksum(weights[g][i][r] * x[g, i]
                               for g in range(n_groups)
                               for i in range(n_items_per_group)) <= caps[r]
                )
            
            # Callback to log incumbents
            best_obj = [float('-inf')]
            best_sol = [None]
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj > best_obj[0]:
                        best_obj[0] = obj
                        sol = {}
                        for g in range(n_groups):
                            for i in range(n_items_per_group):
                                val = model.cbGetSolution(x[g, i])
                                if val > 0.5:
                                    sol[g] = i
                                    break
                        best_sol[0] = sol
                        if logger:
                            logger.log(obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                selection = {}
                for g in range(n_groups):
                    for i in range(n_items_per_group):
                        if x[g, i].X > 0.5:
                            selection[g] = i
                            break
                
                obj_val = model.ObjVal
                
                selected_items = []
                for g in range(n_groups):
                    i = selection[g]
                    grp_data = groups[g]
                    item_data = grp_data['items'][i]
                    selected_items.append({
                        "group_id": g,
                        "item_id": i,
                        "profit": int(item_data['profit']),
                        "weights": [int(w) for w in item_data['weights']]
                    })
                
                solution = {
                    "objective_value": float(obj_val),
                    "selected_items": selected_items
                }
                
                with open(solution_path, 'w') as f:
                    json.dump(solution, f)
                return
    except Exception as e:
        pass
    
    # Fallback: greedy heuristic
    # For each group, pick the item with highest profit-to-weight ratio that fits
    remaining_caps = caps.copy()
    selection = {}
    total_profit = 0.0
    
    # First pass: greedy by profit density
    for g in range(n_groups):
        best_item = None
        best_score = -1
        for i in range(n_items_per_group):
            w = weights[g][i]
            if np.all(w <= remaining_caps):
                total_weight = np.sum(w)
                score = profits[g][i] / (total_weight + 1)
                if best_item is None or profits[g][i] > profits[g][best_item]:
                    best_item = i
                    best_score = profits[g][i]
        
        if best_item is None:
            # Pick item with minimum resource usage
            best_item = 0
            best_excess = float('inf')
            for i in range(n_items_per_group):
                w = weights[g][i]
                excess = np.sum(np.maximum(w - remaining_caps, 0))
                if excess < best_excess:
                    best_excess = excess
                    best_item = i
        
        selection[g] = best_item
        remaining_caps -= weights[g][best_item]
        total_profit += profits[g][best_item]
    
    if logger:
        logger.log(total_profit)
    
    selected_items = []
    for g in range(n_groups):
        i = selection[g]
        grp_data = groups[g]
        item_data = grp_data['items'][i]
        selected_items.append({
            "group_id": g,
            "item_id": i,
            "profit": int(item_data['profit']),
            "weights": [int(w) for w in item_data['weights']]
        })
    
    solution = {
        "objective_value": float(total_profit),
        "selected_items": selected_items
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)