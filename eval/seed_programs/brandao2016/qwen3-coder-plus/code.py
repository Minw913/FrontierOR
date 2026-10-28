import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_bin_packing(instance_data, time_limit):
    n = instance_data['parameters']['n']
    m = instance_data['parameters']['m']
    W = instance_data['parameters']['W']
    
    items = []
    item_weights_dict = {}
    item_demands_dict = {}
    
    for item_info in instance_data['items']:
        item_type = item_info['item_type']
        weight = item_info['weight']
        demand = item_info['demand']
        items.append(item_type)
        item_weights_dict[item_type] = weight
        item_demands_dict[item_type] = demand
    
    # Calculate lower bound for number of bins
    total_weight = sum(item_weights_dict[it] * item_demands_dict[it] for it in items)
    min_bins_needed = max(int(total_weight / W),  
                          max(math.ceil(item_weights_dict[it] * item_demands_dict[it] / W) for it in items))
    # We can add more bins than the lower bound if needed
    max_bins_needed = len([it for it in items for _ in range(item_demands_dict[it])])
    
    # Create model
    model = gp.Model("bin_packing")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    model.setParam('Heuristics', 0.15)  # Increase heuristic effort
    
    # We'll try a reasonable upper bound for the number of bins
    B = max(max_bins_needed, min_bins_needed + 10)  # Set a large but reasonable upper bound
    
    # Variables
    x = {}  # x[i, b] = number of copies of item type i placed in bin b
    y = {}  # y[b] = 1 if bin b is used, 0 otherwise

    for i in items:
        for b in range(B):
            x[i, b] = model.addVar(vtype=GRB.INTEGER, name=f"x_{i}_{b}", lb=0, 
                                   ub=min(item_demands_dict[i], W // item_weights_dict[i]) if item_weights_dict[i] > 0 else 0)
    
    for b in range(B):
        y[b] = model.addVar(vtype=GRB.BINARY, name=f"y_{b}")

    # Constraints
    # Each item's demand must be satisfied
    for i in items:
        model.addConstr(
            gp.quicksum(x[i, b] for b in range(B)) >= item_demands_dict[i],
            name=f"demand_{i}"
        )
        
    # Capacity constraint of each bin
    for b in range(B):
        model.addConstr(
            gp.quicksum(x[i, b] * item_weights_dict[i] for i in items) <= W * y[b],
            name=f"capacity_{b}"
        )

    # Symmetry breaking: bins are filled in order
    for b in range(B - 1):
        model.addConstr(y[b] >= y[b+1], name=f"sym_break_{b}")
    
    # Objective: minimize number of bins used
    model.setObjective(gp.quicksum(y[b] for b in range(B)), GRB.MINIMIZE)

    # Optimize
    model.optimize()
    
    # Process result
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return None
        
    solution_bins = []
    num_used_bins = 0

    # Collect which bins are actually used
    bin_contents = [[] for _ in range(B)]
    for i in items:
        for b in range(B):
            if x[i, b].x > 0:
                count = int(round(x[i, b].x))
                bin_contents[b].extend([i] * count)

    # Create solution bins for used ones
    for b in range(B):
        if y[b].x > 0.5 and bin_contents[b]:
            num_used_bins += 1
            solution_bins.append(bin_contents[b])

    num_bins = num_used_bins
    objective_value = num_bins

    return {
        "objective_value": objective_value,
        "num_bins": num_bins,
        "bins": solution_bins
    }


def solve_with_column_generation_heuristic(instance_data, time_limit):
    """
    A heuristic that generates patterns column by column and tries to find a good solution quickly
    """
    # Parse instance data
    m = instance_data['parameters']['m']
    W = instance_data['parameters']['W']
    
    item_types = []
    item_weights = []
    item_demands = []
    
    weight_map = {}
    demand_map = {}
    
    for item_info in instance_data['items']:
        item_type = item_info['item_type']
        weight = item_info['weight']
        demand = item_info['demand']
        item_types.append(item_type)
        item_weights.append(weight)
        item_demands.append(demand)
        weight_map[item_type] = weight
        demand_map[item_type] = demand
    
    # First-fit decreasing heuristic implementation
    # Sort items by non-increasing weight
    sorted_items = [(weight_map[i], i, demand_map[i]) for i in item_types]
    sorted_items.sort(reverse=True)
    
    # Duplicate each item according to its demand
    all_items_to_place = []
    for weight, item_type, demand in sorted_items:
        all_items_to_place.extend([(weight, item_type)] * demand)
    
    bins = []  # List of bins, each bin is a list of item types
    bin_loads = []  # Current load of each bin
    
    for weight, item_type in all_items_to_place:
        placed = False
        # Try to place this item in an existing bin
        for i, current_load in enumerate(bin_loads):
            if current_load + weight <= W:
                bins[i].append(item_type)
                bin_loads[i] += weight
                placed = True
                break
        
        # If could not place in existing bins, create a new one
        if not placed:
            bins.append([item_type])
            bin_loads.append(weight)
    
    result = {
        "objective_value": len(bins),
        "num_bins": len(bins),
        "bins": bins
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Solve Bin Packing Problem')
    parser.add_argument('--instance_path', required=True, help='Path to input JSON instance file')
    parser.add_argument('--solution_path', required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Path to log file (optional)')
    
    args = parser.parse_args()

    # Initialize logger after argument parsing
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve using heuristic first (fast) to get an initial solution
    try:
        heuristic_solution = solve_with_column_generation_heuristic(instance_data, args.time_limit)
        
        if logger:
            logger.log(heuristic_solution["objective_value"])
        
        # If there's more time, try the exact MIP model with some timeout for just the solving part
        remaining_time = max(1, args.time_limit - 10)  # Reserve 10 seconds for overhead
        
        if remaining_time > 10:  # Only try exact IP if significant time remains
            mIP_solution = solve_bin_packing(instance_data, remaining_time)
            
            if mIP_solution and mIP_solution["objective_value"] < heuristic_solution["objective_value"]:
                result = mIP_solution
            else:
                result = heuristic_solution
        else:
            result = heuristic_solution
            
    except Exception as e:
        print(f"Error occurred: {e}. Falling back to heuristic.")
        result = solve_with_column_generation_heuristic(instance_data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()