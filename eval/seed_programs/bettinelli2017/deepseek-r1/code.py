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
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['num_items']
    capacity = instance['capacity']
    profits = instance['items']['profits']
    weights = instance['items']['weights']
    edges = instance['conflict_graph']['edges']
    
    # Initialize logger if specified
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    
    # Create Gurobi model
    model = gp.Model("KnapsackWithConflicts")
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Decision variables: binary indicators for item selection
    x = model.addVars(n, vtype=GRB.BINARY, name="item")
    
    # Objective: maximize total profit
    total_profit = gp.quicksum(profits[i] * x[i] for i in range(n))
    model.setObjective(total_profit, GRB.MAXIMIZE)
    
    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")
    
    # Conflict constraints
    for u, v in edges:
        model.addConstr(x[u] + x[v] <= 1, name=f"conflict_{u}_{v}")
    
    # Define callback for logging
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(objval)
        model.optimize(callback)
    else:
        model.optimize()
    
    # Process solution
    if model.SolCount > 0:
        selected = [i for i in range(n) if x[i].X > 0.5]
        objval = model.ObjVal
    else:
        selected = []
        objval = 0.0
    
    # Write solution
    solution = {
        "objective_value": float(objval),
        "selected_items": selected
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()