import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    c = instance['c']
    weights = instance['weights']
    incompatibilities = instance.get('incompatibilities', [])
    
    # Build incompatibility adjacency sets
    incompat = [set() for _ in range(n)]
    for pair in incompatibilities:
        i, j = pair[0], pair[1]
        incompat[i].add(j)
        incompat[j].add(i)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Handle edge cases
    if n == 0:
        solution = {"objective_value": 0, "bins": []}
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # First Fit Decreasing heuristic
    def first_fit_decreasing(items_order):
        bins = []  # list of (current_weight, set_of_items)
        assignment = {}
        
        for idx in items_order:
            w = weights[idx]
            placed = False
            for b_idx, (bw, bitems) in enumerate(bins):
                if bw + w <= c:
                    # Check incompatibility
                    compatible = True
                    for item_in_bin in bitems:
                        if item_in_bin in incompat[idx]:
                            compatible = False
                            break
                    if compatible:
                        bins[b_idx] = (bw + w, bitems | {idx})
                        assignment[idx] = b_idx
                        placed = True
                        break
            if not placed:
                bins.append((w, {idx}))
                assignment[idx] = len(bins) - 1
        
        return bins, assignment
    
    # Sort items by decreasing weight
    sorted_items = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    best_bins, best_assignment = first_fit_decreasing(sorted_items)
    best_obj = len(best_bins)
    
    if logger:
        logger.log(best_obj)
    
    # Best Fit Decreasing heuristic
    def best_fit_decreasing(items_order):
        bins = []  # list of (current_weight, set_of_items)
        assignment = {}
        
        for idx in items_order:
            w = weights[idx]
            best_b = -1
            best_remaining = c + 1
            
            for b_idx, (bw, bitems) in enumerate(bins):
                remaining = c - bw - w
                if remaining >= 0 and remaining < best_remaining:
                    # Check incompatibility
                    compatible = True
                    for item_in_bin in bitems:
                        if item_in_bin in incompat[idx]:
                            compatible = False
                            break
                    if compatible:
                        best_b = b_idx
                        best_remaining = remaining
            
            if best_b >= 0:
                bw, bitems = bins[best_b]
                bins[best_b] = (bw + w, bitems | {idx})
                assignment[idx] = best_b
            else:
                bins.append((w, {idx}))
                assignment[idx] = len(bins) - 1
        
        return bins, assignment
    
    bins2, assignment2 = best_fit_decreasing(sorted_items)
    if len(bins2) < best_obj:
        best_bins = bins2
        best_assignment = assignment2
        best_obj = len(bins2)
        if logger:
            logger.log(best_obj)
    
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    
    # Lower bound
    total_weight = sum(weights)
    lb = max(1, (total_weight + c - 1) // c)
    
    # If we have incompatibilities, the chromatic number of the incompatibility graph
    # is also a lower bound, but computing it is expensive. We can use max clique size.
    # For simplicity, use max degree + 1 as an upper bound on chromatic number... 
    # Actually let's just check if any item has incompatibilities
    if incompatibilities:
        # A simple lower bound: for each item, it needs its own bin if it's incompatible
        # with items that fill other bins. Max clique is a better LB.
        # Simple: max over items of (1 + number of incompatible items that together with it exceed capacity)
        # Actually just use the weight-based LB for now
        pass
    
    if best_obj <= lb:
        # Already optimal
        result_bins = []
        bin_contents = {}
        for item, b in best_assignment.items():
            if b not in bin_contents:
                bin_contents[b] = []
            bin_contents[b].append(item)
        for b in sorted(bin_contents.keys()):
            result_bins.append(bin_contents[b])
        
        solution = {"objective_value": best_obj, "bins": result_bins}
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Try to improve with Gurobi if problem is not too large and we have time
    if remaining_time > 2 and n <= 500:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Upper bound on number of bins
            num_bins = best_obj
            
            model = gp.Model("binpacking")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, remaining_time - 1))
            model.setParam('Threads', 1)
            model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
            
            # Binary variables: x[i,j] = 1 if item i is in bin j
            x = {}
            for i in range(n):
                for j in range(num_bins):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
            # Binary variables: y[j] = 1 if bin j is used
            y = {}
            for j in range(num_bins):
                y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
            
            model.update()
            
            # Each item in exactly one bin
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(num_bins)) == 1)
            
            # Capacity constraints
            for j in range(num_bins):
                model.addConstr(
                    gp.quicksum(weights[i] * x[i, j] for i in range(n)) <= c * y[j]
                )
            
            # Incompatibility constraints
            for pair in incompatibilities:
                i1, i2 = pair[0], pair[1]
                for j in range(num_bins):
                    model.addConstr(x[i1, j] + x[i2, j] <= 1)
            
            # Symmetry breaking: bin j can only be used if bin j-1 is used
            for j in range(1, num_bins):
                model.addConstr(y[j] <= y[j - 1])
            
            # Objective
            model.setObjective(gp.quicksum(y[j] for j in range(num_bins)), GRB.MINIMIZE)
            
            # Provide initial solution as warm start
            for i in range(n):
                for j in range(num_bins):
                    x[i, j].Start = 0
            for j in range(num_bins):
                y[j].Start = 0
            
            for item, b in best_assignment.items():
                if b < num_bins:
                    x[item, b].Start = 1
                    y[b].Start = 1
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJBST)))
                    if logger:
                        logger.log(obj_val)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                obj_val = int(round(model.ObjVal))
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_assignment = {}
                    for i in range(n):
                        for j in range(num_bins):
                            if x[i, j].X > 0.5:
                                best_assignment[i] = j
                                break
                    if logger:
                        logger.log(best_obj)
        except Exception as e:
            pass
    
    # Build output
    bin_contents = {}
    for item, b in best_assignment.items():
        if b not in bin_contents:
            bin_contents[b] = []
        bin_contents[b].append(item)
    
    result_bins = []
    for b in sorted(bin_contents.keys()):
        result_bins.append(bin_contents[b])
    
    solution = {"objective_value": best_obj, "bins": result_bins}
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()