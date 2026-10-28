import argparse
import json
import math
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import time
from solution_logger import SolutionLogger


def solve_cutting_stock(instance_data, time_limit, log_path):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    bin_capacity = instance_data["bin_capacity"]
    items = instance_data["items"]
    
    # Sort items by width in descending order (first fit decreasing heuristic)
    sorted_items = sorted(items, key=lambda x: x["width"], reverse=True)
    
    # Calculate lower bound using linear relaxation
    total_demand = sum(item["demand"] for item in items)
    max_width = max(item["width"] for item in items)
    
    # Lower bound calculation: total area / bin capacity
    total_area = sum(item["width"] * item["demand"] for item in items)
    lower_bound = math.ceil(total_area / bin_capacity)
    
    # Generate initial patterns using first-fit-decreasing heuristic
    initial_patterns = []
    
    remaining_demands = {item["type_id"]: item["demand"] for item in items}
    
    # Create patterns using FFD
    while any(remaining_demands[tid] > 0 for tid in remaining_demands):
        current_pattern = [0] * len(items)  # Count of each item type in pattern
        current_load = 0
        
        for i, item in enumerate(sorted_items):
            item_type_idx = next(j for j, it in enumerate(items) if it["type_id"] == item["type_id"])
            
            while remaining_demands[item["type_id"]] > 0 and \
                  current_load + item["width"] <= bin_capacity:
                current_pattern[item_type_idx] += 1
                current_load += item["width"]
                remaining_demands[item["type_id"]] -= 1
                
                if remaining_demands[item["type_id"]] == 0:
                    break
        
        if any(count > 0 for count in current_pattern):
            initial_patterns.append(current_pattern)
    
    # Create master problem
    model = gp.Model("cutting_stock")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Variables: number of times each pattern is used
    pattern_vars = []
    
    for idx, pattern in enumerate(initial_patterns):
        var = model.addVar(vtype=GRB.INTEGER, name=f"pattern_{idx}", lb=0)
        pattern_vars.append(var)
    
    # Constraints: meet demand for each item type
    constraints = []
    for i, item in enumerate(items):
        constr = model.addConstr(
            gp.quicksum(pattern[i] * pattern_vars[j] 
                       for j, pattern in enumerate(initial_patterns)),
            GRB.GREATER_EQUAL,
            item["demand"],
            name=f"demand_{i}"
        )
        constraints.append(constr)
    
    # Objective: minimize number of bins used
    model.setObjective(gp.quicksum(var for var in pattern_vars), GRB.MINIMIZE)
    
    # Set initial solution
    for var in pattern_vars:
        var.Start = 1.0  # Start with 1 for each pattern
    
    model.optimize()
    
    # Column generation implementation
    if model.Status != GRB.INFEASIBLE:
        # Continue with column generation until no more improvement
        iteration_count = 0
        max_iterations = 100  # Prevent infinite loops
        while iteration_count < max_iterations and time.time() - start_time < time_limit - 0.1:
            # Get dual values
            duals = [constr.Pi for constr in constraints]
            
            # Solve subproblem: find new pattern with negative reduced cost
            sub_model = gp.Model("subproblem")
            sub_model.setParam('OutputFlag', 0)
            sub_model.setParam('Seed', 0)
            sub_model.setParam('MIPGap', 1e-6)
            sub_model.setParam('NumericFocus', 0)
            sub_model.setParam('Threads', 1)
            
            # Variables for subproblem: how many of each item type in new pattern
            x_vars = []
            for i, item in enumerate(items):
                var = sub_model.addVar(vtype=GRB.INTEGER, lb=0, 
                                      ub=item["demand"],  # Upper bound to avoid unnecessary large values
                                      name=f"x_{i}")
                x_vars.append(var)
            
            # Constraint: don't exceed bin capacity
            sub_model.addConstr(
                gp.quicksum(items[i]["width"] * x_vars[i] for i in range(len(items))),
                GRB.LESS_EQUAL,
                bin_capacity,
                name="capacity"
            )
            
            # Objective: minimize reduced cost (maximize profit of new pattern)
            sub_model.setObjective(
                1 - gp.quicksum(duals[i] * x_vars[i] for i in range(len(items))),
                GRB.MINIMIZE
            )
            
            sub_model.optimize()
            
            if sub_model.Status == GRB.OPTIMAL and sub_model.ObjVal < -1e-6:
                # Add new pattern
                new_pattern = [int(x_var.X + 0.5) for x_var in x_vars]
                
                new_var = model.addVar(vtype=GRB.INTEGER, name=f"pattern_{len(pattern_vars)}", lb=0)
                pattern_vars.append(new_var)
                
                # Update constraints with new pattern
                for i in range(len(items)):
                    if new_pattern[i] > 0:
                        constraints[i].addTerms(new_pattern[i], new_var)
                
                model.update()
                model.optimize()
                
                # Log solution if improved
                if logger and model.SolCount > 0:
                    current_obj = int(model.ObjVal + 0.5)
                    # Prepare solution dict
                    sol_dict = {}
                    for i, var in enumerate(pattern_vars):
                        val = var.X
                        if abs(val) > 1e-6:  # Only include non-zero values
                            sol_dict[f"pattern_{i}"] = val
                    logger.log_solution(current_obj, {"model_variables": sol_dict})
            else:
                # No more columns to add
                break
            
            iteration_count += 1
    
    # Final solution processing
    if model.SolCount > 0:
        solution_vars = {}
        total_bins = 0
        for i, var in enumerate(pattern_vars):
            val = var.X
            if abs(val) > 1e-9:
                solution_vars[f"pattern_{i}"] = round(val)
                total_bins += round(val)
        
        result = {
            "objective_value": total_bins,
            "model_variables": solution_vars
        }
    else:
        # Fallback: return simple solution if optimization failed
        result = {
            "objective_value": len(initial_patterns),
            "model_variables": {f"pattern_{i}": 1 for i in range(len(initial_patterns))}
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Solve cutting stock problem')
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Path to log file for intermediate solutions')
    
    args = parser.parse_args()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve
    solution = solve_cutting_stock(instance_data, args.time_limit, args.log_path)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()