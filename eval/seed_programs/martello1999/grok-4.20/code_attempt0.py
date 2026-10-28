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
    
    n = instance['n']
    capacity = instance['capacity']
    items = instance['items']
    profits = [item['profit'] for item in items]
    weights = [item['weight'] for item in items]
    ids = [item['id'] for item in items]
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Create Gurobi model
    model = gp.Model("Knapsack")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)
    
    # Decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Objective
    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(int(round(obj)))
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        objective_value = int(round(model.objVal))
        selected = []
        for i in range(n):
            if x[i].X > 0.5:
                selected.append(ids[i])
        selected.sort()
    else:
        # If no solution found (unlikely), return empty
        objective_value = 0
        selected = []
    
    # Write solution
    solution = {
        "objective_value": objective_value,
        "selected_items": selected
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()