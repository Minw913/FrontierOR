import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import time


def solve_knapsack_problem(instance_data, time_limit, logger=None):
    n = instance_data['n']
    m = instance_data['m']
    capacity = instance_data['capacity']
    weights = instance_data['weights']
    values = instance_data['values']
    pairwise_values = np.array(instance_data['pairwise_values'])
    
    # Create model
    model = gp.Model("knapsack_with_pairing")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    # Variables:
    # x[i, k] = 1 if item i is placed in knapsack k
    x = {}
    for i in range(n):
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
    
    # z[i, j, k] = 1 if both items i and j are placed in knapsack k
    z = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                z[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}_{k}")
    
    # Constraints:
    # Each item is assigned to at most one knapsack
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"item_assignment_{i}")
    
    # Capacity constraints
    for k in range(m):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity,
            name=f"capacity_{k}"
        )
    
    # If both items i and j are in knapsack k, then z[i,j,k] must be 1
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                model.addConstr(z[i, j, k] <= x[i, k], name=f"pair_1_{i}_{j}_{k}")
                model.addConstr(z[i, j, k] <= x[j, k], name=f"pair_2_{i}_{j}_{k}")
                model.addConstr(z[i, j, k] >= x[i, k] + x[j, k] - 1, name=f"pair_3_{i}_{j}_{k}")
    
    # Objective function
    obj_expr = gp.LinExpr()
    # Add individual values
    for i in range(n):
        for k in range(m):
            obj_expr += values[i] * x[i, k]
    # Add pairwise values
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                obj_expr += pairwise_values[i][j] * z[i, j, k]
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.update()
    
    start_time = time.time()
    best_solution = {"objective_value": float("-inf"), "assignment": {}}
    
    def update_best_solution():
        nonlocal best_solution
        current_obj = model.ObjVal
        current_assignment = {
            str(i): k 
            for i in range(n) 
            for k in range(m) 
            if round(x[i, k].X) == 1
        }
        current_solution = {
            "objective_value": current_obj,
            "assignment": current_assignment
        }
        best_solution = current_solution
        if logger:
            logger.log(-current_obj if model.ModelSense == GRB.MAXIMIZE else current_obj)
    
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            # Update best known solution if current is better
            if current_obj > best_solution["objective_value"]:
                # Temporarily save current solution
                current_assignment = {}
                for i in range(n):
                    for k in range(m):
                        if model.cbGetSolution(x[i, k]) > 0.5:
                            current_assignment[str(i)] = k
                            break  # Item i was found
                
                temp_sol = {"objective_value": current_obj, "assignment": current_assignment.copy()}
                
                if temp_sol["objective_value"] > best_solution["objective_value"]:
                    best_solution.update(temp_sol)
                    if logger:
                        logger.log(-temp_sol["objective_value"] if model.ModelSense == GRB.MAXIMIZE else temp_sol["objective_value"])
    
    model.optimize(my_callback)
    
    # Final check in case the solver stopped without finding a solution or before callback worked
    if model.SolCount > 0:
        # Make sure we have the latest solution stored properly
        update_best_solution()
    
    if model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        # Return best feasible solution even if time limit is reached
        return best_solution
    
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        if model.SolCount > 0:
            return best_solution
        else:
            # Handle case where no solution is found
            return {"objective_value": float("-inf"), "assignment": {}}
    else:
        print(f"Model status: {model.status}")
        # If no solution found, return empty assignment with inf
        return {"objective_value": float("-inf"), "assignment": {}}


def main():
    parser = argparse.ArgumentParser(description='Solve knapsack problem with pairing values.')
    parser.add_argument('--instance_path', type=str, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file (JSONL)')
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Solve the problem
    solution = solve_knapsack_problem(instance_data, args.time_limit, logger)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    print(f"Optimization completed. Solution written to {args.solution_path}")


if __name__ == '__main__':
    main()