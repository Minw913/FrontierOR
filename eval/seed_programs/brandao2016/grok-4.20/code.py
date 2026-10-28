import json
import argparse
import time
import sys
from typing import List, Dict, Any
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['parameters']['n']
    m = instance['parameters']['m']
    W = instance['parameters']['W']
    items = instance['items']
    
    # Prepare data
    weights = [0] * (m + 1)
    demands = [0] * (m + 1)
    for item in items:
        t = item['item_type']
        weights[t] = item['weight']
        demands[t] = item['demand']
    
    # Lower bound
    total_weight = sum(weights[t] * demands[t] for t in range(1, m+1))
    lb = (total_weight + W - 1) // W
    max_demand = max(demands[1:])
    lb = max(lb, max_demand)
    
    # Upper bound: naive
    ub = sum(demands[1:])
    
    start_time = time.time()
    
    # Model
    model = gp.Model("BinPacking")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0
    
    # Variables
    max_bins = ub
    x = model.addVars(range(1, m+1), range(max_bins), vtype=GRB.INTEGER, lb=0, name="x")
    y = model.addVars(range(max_bins), vtype=GRB.BINARY, name="y")
    
    # Objective
    model.setObjective(gp.quicksum(y[b] for b in range(max_bins)), GRB.MINIMIZE)
    
    # Demand constraints
    for t in range(1, m+1):
        model.addConstr(gp.quicksum(x[t, b] for b in range(max_bins)) >= demands[t])
    
    # Capacity constraints
    for b in range(max_bins):
        model.addConstr(gp.quicksum(weights[t] * x[t, b] for t in range(1, m+1)) <= W * y[b])
    
    # Symmetry breaking: bins are ordered
    for b in range(1, max_bins):
        model.addConstr(y[b] <= y[b-1])
    
    # Callback for incumbent logging
    best_obj = float('inf')
    incumbent_solutions = []
    
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj - 1e-6:
                best_obj = obj
                if logger:
                    logger.log(float(obj))
    
    model.optimize(callback)
    
    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        num_bins_used = int(model.objVal + 0.5)
        bins = []
        used_bins = []
        
        for b in range(max_bins):
            if y[b].X > 0.5:
                bin_items = []
                for t in range(1, m+1):
                    cnt = int(x[t, b].X + 0.5)
                    bin_items.extend([t] * cnt)
                if bin_items:
                    bins.append(bin_items)
                    used_bins.append(b)
        
        # If we have fewer bins than expected due to rounding, fill with empty if needed (should not happen)
        while len(bins) < num_bins_used:
            bins.append([])
        
        objective_value = float(num_bins_used)
        
        # Validate total packed
        total_packed = [0] * (m + 1)
        for bin_content in bins:
            for t in bin_content:
                total_packed[t] += 1
        for t in range(1, m+1):
            if total_packed[t] < demands[t]:
                # Try to fix by adding missing items to last bin if capacity allows
                for t_missing in range(1, m+1):
                    while total_packed[t_missing] < demands[t_missing]:
                        added = False
                        for bin_content in reversed(bins):
                            current_weight = sum(weights[it] for it in bin_content)
                            if current_weight + weights[t_missing] <= W:
                                bin_content.append(t_missing)
                                total_packed[t_missing] += 1
                                added = True
                                break
                        if not added:
                            # Add new bin
                            bins.append([t_missing])
                            total_packed[t_missing] += 1
                            num_bins_used += 1
                            objective_value = float(num_bins_used)
        
        # Final output
        solution = {
            "objective_value": objective_value,
            "num_bins": int(num_bins_used),
            "bins": bins
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Fallback: greedy first-fit decreasing
        item_list = []
        for t in range(1, m+1):
            item_list.extend([t] * demands[t])
        # Sort by weight descending
        item_list.sort(key=lambda t: -weights[t])
        
        bins = []
        for item in item_list:
            placed = False
            for b in bins:
                if sum(weights[i] for i in b) + weights[item] <= W:
                    b.append(item)
                    placed = True
                    break
            if not placed:
                bins.append([item])
        
        num_bins_used = len(bins)
        objective_value = float(num_bins_used)
        
        solution = {
            "objective_value": objective_value,
            "num_bins": num_bins_used,
            "bins": bins
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        
        if logger:
            logger.log(objective_value)

if __name__ == "__main__":
    main()