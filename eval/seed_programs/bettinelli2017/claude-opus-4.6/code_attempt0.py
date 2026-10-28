import argparse
import json
import time
import sys

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_items = instance['num_items']
    capacity = instance['capacity']
    profits = instance['items']['profits']
    weights = instance['items']['weights']
    edges = instance['conflict_graph']['edges']
    num_edges = instance['conflict_graph']['num_edges']

    start_time = time.time()

    # Use Gurobi MIP solver
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("knapsack_conflict")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', max(1, args.time_limit - 2))
    model.setParam('Threads', 1)
    
    # Create binary variables
    x = model.addVars(num_items, vtype=GRB.BINARY, name="x")
    
    # Set objective: maximize total profit
    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(num_items)), GRB.MAXIMIZE)
    
    # Knapsack capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(num_items)) <= capacity, "capacity")
    
    # Conflict constraints
    for edge in edges:
        i, j = edge[0], edge[1]
        model.addConstr(x[i] + x[j] <= 1)
    
    # Callback to log incumbent solutions
    best_obj = [float('-inf')]
    best_solution = [None]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > best_obj[0] + 1e-6:
                best_obj[0] = obj
                vals = model.cbGetSolution([x[i] for i in range(num_items)])
                selected = [i for i in range(num_items) if vals[i] > 0.5]
                best_solution[0] = selected
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        selected = [i for i in range(num_items) if x[i].X > 0.5]
        obj_val = sum(profits[i] for i in selected)
    elif best_solution[0] is not None:
        selected = best_solution[0]
        obj_val = sum(profits[i] for i in selected)
    else:
        # Fallback: greedy solution
        # Build adjacency list
        adj = [set() for _ in range(num_items)]
        for edge in edges:
            i, j = edge[0], edge[1]
            adj[i].add(j)
            adj[j].add(i)
        
        # Sort by profit/weight ratio
        order = sorted(range(num_items), key=lambda i: profits[i] / weights[i], reverse=True)
        selected = []
        total_weight = 0
        in_knapsack = set()
        blocked = set()
        
        for i in order:
            if i in blocked:
                continue
            if total_weight + weights[i] <= capacity:
                selected.append(i)
                total_weight += weights[i]
                in_knapsack.add(i)
                for j in adj[i]:
                    blocked.add(j)
        
        obj_val = sum(profits[i] for i in selected)
        if logger:
            logger.log(obj_val)
    
    solution = {
        "objective_value": obj_val,
        "selected_items": selected
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()