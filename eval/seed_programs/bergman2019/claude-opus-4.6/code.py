import argparse
import json
import time
import numpy as np
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    profits = data['profits']
    pairwise_profits = data['pairwise_profits']
    weights = data['weights']
    capacities = data['capacities']
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    best_obj = 0  # Empty assignment is always feasible with objective 0
    best_assignment = []
    
    if logger:
        logger.log(best_obj)
    
    # For small instances, use Gurobi MIP
    # For larger instances, use greedy + local search
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining <= 1:
        # No time left, return empty
        solution = {
            "objective_value": best_obj,
            "assignment": best_assignment
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Try Gurobi for moderate-sized instances
    use_gurobi = (n <= 200 and m <= 20)
    
    if use_gurobi:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("qkp")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, remaining - 2))
            model.setParam('Threads', 1)
            
            # x[i][k] = 1 if item i assigned to knapsack k
            x = {}
            for i in range(n):
                for k in range(m):
                    x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
            
            model.update()
            
            # Each item assigned to at most one knapsack
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1)
            
            # Capacity constraints
            for k in range(m):
                model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k])
            
            # Objective: individual profits + pairwise profits
            obj = gp.LinExpr()
            for i in range(n):
                for k in range(m):
                    obj += profits[i] * x[i, k]
            
            # For pairwise profits, we need y[i,j,k] = x[i,k] * x[j,k]
            # Use linearization or quadratic objective
            # For moderate n, use auxiliary variables
            
            if n <= 80:
                # Use auxiliary binary variables for pairs in same knapsack
                y = {}
                for i in range(n):
                    for j in range(i+1, n):
                        if pairwise_profits[i][j] != 0:
                            for k in range(m):
                                y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")
                
                model.update()
                
                for i in range(n):
                    for j in range(i+1, n):
                        if pairwise_profits[i][j] != 0:
                            for k in range(m):
                                model.addConstr(y[i, j, k] <= x[i, k])
                                model.addConstr(y[i, j, k] <= x[j, k])
                                if pairwise_profits[i][j] > 0:
                                    model.addConstr(y[i, j, k] >= x[i, k] + x[j, k] - 1)
                            
                            obj += pairwise_profits[i][j] * gp.quicksum(y[i, j, k] for k in range(m))
                
            else:
                # For larger instances, use z[i,j] = sum_k y[i,j,k] (items in same knapsack)
                # z[i,j] <= sum_k x[i,k], z[i,j] <= sum_k x[j,k]
                # And for each k: z contributes... this is trickier
                # Use a different formulation: z[i,j,k] approach but aggregate
                
                # Alternative: use quadratic constraints or lazy constraints
                # Let's try with z[i,j] = 1 if i and j are in the same knapsack
                z = {}
                for i in range(n):
                    for j in range(i+1, n):
                        if pairwise_profits[i][j] != 0:
                            z[i, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
                
                model.update()
                
                # z[i,j] = 1 iff i and j are in the same knapsack
                # z[i,j] <= sum_k x[i,k]
                # z[i,j] <= sum_k x[j,k]
                # z[i,j] >= x[i,k] + x[j,k] - 1 for each k (if profit positive)
                
                for i in range(n):
                    for j in range(i+1, n):
                        if pairwise_profits[i][j] != 0:
                            model.addConstr(z[i, j] <= gp.quicksum(x[i, k] for k in range(m)))
                            model.addConstr(z[i, j] <= gp.quicksum(x[j, k] for k in range(m)))
                            
                            if pairwise_profits[i][j] > 0:
                                for k in range(m):
                                    model.addConstr(z[i, j] >= x[i, k] + x[j, k] - 1)
                            
                            # But z[i,j] can be 1 even if i and j are in different knapsacks
                            # We need: z[i,j] <= sum_k (x[i,k] * x[j,k])
                            # This requires: for negative pairwise profits, we need tighter constraints
                            if pairwise_profits[i][j] < 0:
                                for k in range(m):
                                    model.addConstr(z[i, j] >= x[i, k] + x[j, k] - 1)
                            
                            obj += pairwise_profits[i][j] * z[i, j]
            
            model.setObjective(obj, GRB.MAXIMIZE)
            
            # Add callback for logging
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    nonlocal best_obj, best_assignment
                    if obj_val > best_obj:
                        best_obj = obj_val
                        # Extract solution
                        vals = model.cbGetSolution([x[i, k] for i in range(n) for k in range(m)])
                        assignment = []
                        idx = 0
                        for i in range(n):
                            for k in range(m):
                                if vals[idx] > 0.5:
                                    assignment.append([i, k])
                                idx += 1
                        best_assignment = assignment
                        if logger:
                            logger.log(best_obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val > best_obj:
                    best_obj = obj_val
                    best_assignment = []
                    for i in range(n):
                        for k in range(m):
                            if x[i, k].X > 0.5:
                                best_assignment.append([i, k])
                    if logger:
                        logger.log(best_obj)
            
        except Exception as e:
            # Fall through to greedy
            pass
    
    else:
        # Greedy + local search for large instances
        # Sort knapsacks by capacity descending
        knapsack_order = sorted(range(m), key=lambda k: capacities[k], reverse=True)
        
        # assignment_of[i] = k if item i is in knapsack k, -1 if unassigned
        assignment_of = [-1] * n
        knapsack_weight = [0] * m
        knapsack_items = [[] for _ in range(m)]
        current_obj = 0
        
        # Greedy: compute marginal value of adding each item to each knapsack
        # For each unassigned item, compute: profit[i] + sum of pairwise_profits[i][j] for j in knapsack k
        # Pick the best (item, knapsack) pair
        
        def marginal_value(item, knap):
            val = profits[item]
            for j in knapsack_items[knap]:
                val += pairwise_profits[item][j]
            return val
        
        # Greedy construction
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_item = -1
            best_knap = -1
            
            elapsed = time.time() - start_time
            if elapsed > time_limit - 2:
                break
            
            for i in range(n):
                if assignment_of[i] != -1:
                    continue
                for k in range(m):
                    if knapsack_weight[k] + weights[i] > capacities[k]:
                        continue
                    gain = marginal_value(i, k)
                    if gain > best_gain:
                        best_gain = gain
                        best_item = i
                        best_knap = k
            
            if best_gain > 0 and best_item >= 0:
                assignment_of[best_item] = best_knap
                knapsack_weight[best_knap] += weights[best_item]
                knapsack_items[best_knap].append(best_item)
                current_obj += best_gain
                improved = True
                
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_assignment = [[i, assignment_of[i]] for i in range(n) if assignment_of[i] != -1]
                    if logger:
                        logger.log(best_obj)
        
        # Local search: try moves, swaps, removals
        def compute_removal_gain(item):
            """Gain from removing item (negative means removing is costly)"""
            k = assignment_of[item]
            if k == -1:
                return 0
            val = profits[item]
            for j in knapsack_items[k]:
                if j != item:
                    val += pairwise_profits[item][j]
            return -val  # gain from removing = -contribution
        
        def compute_move_gain(item, new_knap):
            """Net gain from moving item from current location to new_knap"""
            old_knap = assignment_of[item]
            # Loss from removing from old
            loss = profits[item]
            if old_knap != -1:
                for j in knapsack_items[old_knap]:
                    if j != item:
                        loss += pairwise_profits[item][j]
            
            # Gain from adding to new
            gain = profits[item]
            for j in knapsack_items[new_knap]:
                if j != item:
                    gain += pairwise_profits[item][j]
            
            return gain - loss
        
        # Local search iterations
        for iteration in range(1000):
            elapsed = time.time() - start_time
            if elapsed > time_limit - 2:
                break
            
            improved_local = False
            
            # Try moving each item to a different or new knapsack
            for i in range(n):
                old_k = assignment_of[i]
                for k in range(m):
                    if k == old_k:
                        continue
                    
                    # Check capacity
                    add_weight = weights[i]
                    if old_k != -1:
                        # Moving from old_k to k
                        if knapsack_weight[k] + add_weight > capacities[k]:
                            continue
                    else:
                        if knapsack_weight[k] + add_weight > capacities[k]:
                            continue
                    
                    gain = compute_move_gain(i, k)
                    if gain > 1e-9:
                        # Execute move
                        if old_k != -1:
                            knapsack_items[old_k].remove(i)
                            knapsack_weight[old_k] -= weights[i]
                        
                        assignment_of[i] = k
                        knapsack_items[k].append(i)
                        knapsack_weight[k] += weights[i]
                        current_obj += gain
                        improved_local = True
                        
                        if current_obj > best_obj:
                            best_obj = current_obj
                            best_assignment = [[ii, assignment_of[ii]] for ii in range(n) if assignment_of[ii] != -1]
                            if logger:
                                logger.log(best_obj)
                        break
                
                if improved_local:
                    break
            
            if not improved_local:
                # Try removing items with negative contribution
                for i in range(n):
                    if assignment_of[i] == -1:
                        continue
                    removal_gain = compute_removal_gain(i)
                    if removal_gain > 1e-9:
                        k = assignment_of[i]
                        knapsack_items[k].remove(i)
                        knapsack_weight[k] -= weights[i]
                        assignment_of[i] = -1
                        current_obj += removal_gain
                        improved_local = True
                        
                        if current_obj > best_obj:
                            best_obj = current_obj
                            best_assignment = [[ii, assignment_of[ii]] for ii in range(n) if assignment_of[ii] != -1]
                            if logger:
                                logger.log(best_obj)
                        break
            
            if not improved_local:
                break
    
    solution = {
        "objective_value": best_obj,
        "assignment": best_assignment
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)