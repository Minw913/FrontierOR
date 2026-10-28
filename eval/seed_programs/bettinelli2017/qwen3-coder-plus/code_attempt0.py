import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_conflict_aware_knapsack(instance_data, time_limit, log_path=None):
    # Initialize the logger if provided
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    n = instance_data["num_items"]
    capacity = instance_data["capacity"]
    profits = instance_data["items"]["profits"]
    weights = instance_data["items"]["weights"]
    
    # Process conflict graph
    conflict_edges = instance_data["conflict_graph"]["edges"]
    
    # Create model
    model = gp.Model("conflict_aware_knapsack")
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Objective function: Maximize total profit
    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Knapsack capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "knapsack_capacity")
    
    # Conflict constraints: for each edge (i, j) in conflict graph, x[i] + x[j] <= 1
    for edge in conflict_edges:
        i, j = edge[0], edge[1]
        model.addConstr(x[i] + x[j] <= 1, f"conflict_{i}_{j}")
    
    # Track the best solution found
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    # Optimize model
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        selected_items = []
        for i in range(n):
            if x[i].X > 0.5:
                selected_items.append(i)
        
        total_profit = sum(profits[i] for i in selected_items)
        
        solution = {
            "objective_value": total_profit,
            "selected_items": selected_items
        }
    else:
        # No solution found within time limit
        solution = {
            "objective_value": 0,
            "selected_items": []
        }
    
    return solution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_conflict_aware_knapsack(
        instance_data=instance_data,
        time_limit=args.time_limit,
        log_path=args.log_path
    )
    
    # Write the solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()