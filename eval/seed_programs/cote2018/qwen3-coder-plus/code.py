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
    
    # Calculate lower bound using linear relaxation
    total_area = sum(item["width"] * item["demand"] for item in items)
    lower_bound = math.ceil(total_area / bin_capacity)
    
    # Generate initial patterns using greedy heuristic
    initial_patterns = []
    
    # Create a list of items with their demands
    item_list = []
    for item in items:
        for _ in range(item["demand"]):
            item_list.append(item["width"])
    
    # Sort items in descending order for first-fit-decreasing heuristic
    item_list.sort(reverse=True)
    
    # Apply first-fit-decreasing to generate initial patterns
    bins = []
    
    for item_width in item_list:
        # Try to place item in existing bin
        placed = False
        for bin_contents in bins:
            if sum(bin_contents) + item_width <= bin_capacity:
                bin_contents.append(item_width)
                placed = True
                break
        
        # If item doesn't fit in any existing bin, create new bin
        if not placed:
            bins.append([item_width])
    
    # Convert bins to patterns (count of each item type per bin)
    for bin_contents in bins:
        pattern = [0] * len(items)
        # Temporarily track what we've used for this pattern
        temp_item_counts = [0] * len(items)
        
        # Process each width in the bin contents
        for width in bin_contents:
            # Find which item type matches this width
            for i, item in enumerate(items):
                if item["width"] == width and temp_item_counts[i] < item["demand"]:
                    pattern[i] += 1
                    temp_item_counts[i] += 1
                    break
        
        initial_patterns.append(pattern)
    
    # Remove duplicate patterns to avoid redundancy
    unique_patterns = []
    seen_patterns = set()
    for pattern in initial_patterns:
        pattern_tuple = tuple(pattern)
        if pattern_tuple not in seen_patterns:
            unique_patterns.append(pattern)
            seen_patterns.add(pattern_tuple)
    
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
    
    for idx, pattern in enumerate(unique_patterns):
        var = model.addVar(vtype=GRB.INTEGER, name=f"pattern_{idx}", lb=0)
        pattern_vars.append(var)
    
    # Constraints: meet demand for each item type
    constraints = []
    for i, item in enumerate(items):
        expr = gp.LinExpr()
        for j, pattern in enumerate(unique_patterns):
            if pattern[i] > 0:
                expr.addTerms(pattern[i], pattern_vars[j])
        
        constr = model.addConstr(expr, GRB.GREATER_EQUAL, item["demand"], name=f"demand_{item['type_id']}")
        constraints.append(constr)
    
    # Objective: minimize number of bins used
    model.setObjective(gp.quicksum(var for var in pattern_vars), GRB.MINIMIZE)
    
    # Set initial solution
    for var in pattern_vars:
        var.Start = 1.0  # Start with 1 for each pattern
    
    model.optimize()
    
    # Column generation implementation
    if model.Status != GRB.INFEASIBLE and model.Status != GRB.INF_OR_UNBD:
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
            sub_model.setParam('MIPGap', 1e-7)
            sub_model.setParam('NumericFocus', 0)
            sub_model.setParam('Threads', 1)
            
            # Variables for subproblem: how many of each item type in new pattern
            x_vars = []
            for i, item in enumerate(items):
                max_count = bin_capacity // item["width"] if item["width"] > 0 else 0
                var = sub_model.addVar(vtype=GRB.INTEGER, lb=0, 
                                      ub=max_count,
                                      name=f"x_{item['type_id']}")
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
                new_pattern = [int(round(x_var.X)) for x_var in x_vars]
                
                new_var = model.addVar(vtype=GRB.INTEGER, name=f"pattern_{len(pattern_vars)}", lb=0)
                pattern_vars.append(new_var)
                
                # Update constraints with new pattern
                for i in range(len(items)):
                    if new_pattern[i] > 0:
                        # Add coefficient to the corresponding constraint
                        model.addConstr(
                            new_pattern[i] * new_var >= 0,  # Dummy constraint to reference new var
                        )
                        
                        # Update the actual demand constraints
                        model.chgCoeff(constraints[i], new_var, new_pattern[i])
                
                # Add new pattern to list
                unique_patterns.append(new_pattern)
                
                model.update()
                model.optimize()
                
                # Log solution if improved
                if logger and model.SolCount > 0:
                    current_obj = int(round(model.ObjVal))
                    # Prepare solution dict
                    sol_dict = {}
                    for i, var in enumerate(pattern_vars):
                        val = var.X
                        if abs(val) > 1e-6:  # Only include non-zero values
                            sol_dict[f"pattern_{i}"] = round(val)
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
                rounded_val = round(val)
                solution_vars[f"pattern_{i}"] = rounded_val
                total_bins += rounded_val
        
        result = {
            "objective_value": total_bins,
            "model_variables": solution_vars
        }
    else:
        # Fallback: return simple solution if optimization failed
        result = {
            "objective_value": len(unique_patterns),
            "model_variables": {f"pattern_{i}": 1 for i in range(len(unique_patterns))}
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