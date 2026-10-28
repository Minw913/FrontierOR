import argparse
import json
import sys
from solution_logger import SolutionLogger


def solve_knapsack_dynamic_programming(n, capacity, items, time_limit):
    # Create arrays for profit and weight for easier access
    profits = [item['profit'] for item in items]
    weights = [item['weight'] for item in items]

    # Initialize a table for dynamic programming
    # dp[i][w] represents the maximum profit achievable with the first i items and weight limit w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Fill the dp table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # If current item's weight is more than the current weight limit,
            # we can't include it
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                # Max of including and excluding the current item
                include_current = profits[i - 1] + dp[i - 1][w - weights[i - 1]]
                exclude_current = dp[i - 1][w]
                dp[i][w] = max(include_current, exclude_current)

    # Find the selected items
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        # If the value differs from the row above, then this item was included
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(items[i - 1]['id'])
            w -= weights[i - 1]

    # Reverse the list since we built it backwards
    selected_items.reverse()

    # The objective value is at dp[n][capacity]
    objective_value = dp[n][capacity]

    return {
        'objective_value': objective_value,
        'selected_items': selected_items
    }


def solve_knapsack_gurobi(n, capacity, items, time_limit):
    import gurobipy as gp
    from gurobipy import GRB
    
    # Create a model
    model = gp.Model("knapsack")
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: total weight must not exceed capacity
    model.addConstr(
        gp.quicksum(items[i]['weight'] * x[i] for i in range(n)) <= capacity,
        name="capacity"
    )
    
    # Objective: maximize total profit
    model.setObjective(
        gp.quicksum(items[i]['profit'] * x[i] for i in range(n)),
        GRB.MAXIMIZE
    )
    
    def callback(model, where):
        """
        Callback function to log the objective value each time a better solution is found
        """
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            print(f"Found solution with objective {obj}")
            
    # Solve the model
    model.optimize(callback)

    # Prepare the solution
    selected_items = []
    if model.SolCount > 0:  # Check if there is a feasible solution
        for i in range(n):
            if x[i].X > 0.5:  # If variable is 1 (or close enough due to numerical accuracy)
                selected_items.append(items[i]['id'])
                
        objective_value = int(model.ObjVal) if model.ObjVal is not None else 0
    else:
        objective_value = 0
    
    return {
        'objective_value': objective_value,
        'selected_items': selected_items
    }


def solve_knapsack(instance_data, time_limit, log_path=None):
    n = instance_data['n']
    capacity = instance_data['capacity']
    items = instance_data['items']

    # Initialize logger if provided
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # If the number of items and capacity are small, use the Dynamic Programming approach
    # which guarantees an optimal solution
    if n * capacity <= 10e6:  # Arbitrary threshold - adjust as needed
        solution = solve_knapsack_dynamic_programming(n, capacity, items, time_limit)
        if logger:
            logger.log(solution['objective_value'])
    else:
        # Otherwise, use a solver that implements an exact method or approximation
        solution = solve_knapsack_gurobi(n, capacity, items, time_limit)
        
        # When using Gurobi, update logger with intermediate solutions in the callback function
        
    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve the Knapsack problem.")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to a JSONL file for logging intermediate solutions (optional).")

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_knapsack(instance_data, args.time_limit, args.log_path)

    # Write the solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()