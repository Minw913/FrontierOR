import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve 0-1 Knapsack Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (JSONL format)')
    
    args = parser.parse_args()
    
    # Read the instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    capacity = data['capacity']
    items = data['items']
    
    # Extract item details
    ids = [item['id'] for item in items]
    profits = [item['profit'] for item in items]
    weights = [item['weight'] for item in items]
    
    # Initialize solution logger if log_path is provided
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None
    
    # Create Gurobi model
    model = Model()
    model.setParam('OutputFlag', 0)  # Disable solver output
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)  # Use only one CPU core
    
    # Decision variables: binary for each item
    x = model.addVars(len(items), vtype=GRB.BINARY, name="x")
    
    # Objective: maximize total profit
    model.setObjective(quicksum(profits[i] * x[i] for i in range(len(items))), GRB.MAXIMIZE)
    
    # Constraint: total weight <= capacity
    model.addConstr(quicksum(weights[i] * x[i] for i in range(len(items))) <= capacity, name="capacity_constraint")
    
    # Optimize with callback for logging incumbents
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(int(obj_val))
        model.optimize(callback)
    else:
        model.optimize()
    
    # Extract the best solution found
    if model.SolCount > 0:
        total_profit = 0
        selected_ids = []
        for i in range(len(items)):
            if x[i].X > 0.5:  # Item is selected
                total_profit += profits[i]
                selected_ids.append(ids[i])
    else:
        total_profit = 0
        selected_ids = []
    
    # Write solution to JSON file
    solution = {
        "objective_value": total_profit,
        "selected_items": selected_ids
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()