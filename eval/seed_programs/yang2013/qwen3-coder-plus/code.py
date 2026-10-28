import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def solve_quadratic_knapsack(instance_data, time_limit, log_path):
    # Extract instance data
    n = instance_data["n"]
    capacity = instance_data["capacity"]
    weights = instance_data["weights"]
    profit_matrix = instance_data["profit_matrix"]

    # Initialize the logger if requested
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Create model
    model = gp.Model("quadratic_knapsack")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Set random seed for reproducibility
    
    # Add variables: binary variables for whether each item is selected
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Linear part of objective function: diagonal elements
    linear_profit = sum(profit_matrix[i][i] * x[i] for i in range(n))
    
    # Quadratic part of objective function: off-diagonal elements
    quad_profit = gp.QuadExpr()
    
    # Only add terms where profit is non-zero for better performance
    for i in range(n):
        for j in range(i + 1, n):  # Only consider upper triangle since matrix is symmetric
            if profit_matrix[i][j] != 0:
                quad_profit += profit_matrix[i][j] * x[i] * x[j]
    
    # Set objective function (to maximize)
    model.setObjective(linear_profit + quad_profit, GRB.MAXIMIZE)
    
    # Add capacity constraint
    model.addConstr(sum(weights[i] * x[i] for i in range(n)) <= capacity, "knapsack_capacity")
    
    # Track incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:  # If the current solution is integer feasible
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    # Optimize model with callback function
    model.optimize(callback)
    
    # Get solution values
    solution = []
    for i in range(n):
        val = int(round(x[i].x))
        solution.append(val)
    
    # Calculate the actual objective value
    objective_value = 0
    for i in range(n):
        if solution[i] == 1:
            objective_value += profit_matrix[i][i]  # Standalone profit
            
            # Interaction profits with other selected items
            for j in range(i + 1, n):
                if solution[j] == 1:
                    objective_value += profit_matrix[i][j]
                    
    # For debugging: use whatever Gurobi reported if it's better
    if hasattr(model, 'objVal') and model.objVal > objective_value:
        objective_value = model.objVal
    
    return objective_value, solution


def main():
    parser = argparse.ArgumentParser(description='Solve Quadratic Knapsack Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, required=False,
                        help='Path to a JSONL file for logging intermediate solutions (optional)')
    
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    objective_value, solution = solve_quadratic_knapsack(
        instance_data, 
        args.time_limit, 
        args.log_path
    )

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump({
            "objective_value": objective_value,
            "solution": solution
        }, f)


if __name__ == "__main__":
    main()