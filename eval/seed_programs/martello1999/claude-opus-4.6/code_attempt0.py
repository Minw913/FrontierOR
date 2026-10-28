import argparse
import json
import time
import sys

def solve_knapsack(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    capacity = instance['capacity']
    items = instance['items']
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    start_time = time.time()
    
    # For small instances, use dynamic programming
    # For large instances, use Gurobi or branch and bound
    
    # Extract item data
    ids = [item['id'] for item in items]
    profits = [item['profit'] for item in items]
    weights = [item['weight'] for item in items]
    
    # Try DP if capacity and n are manageable
    # DP complexity: O(n * capacity)
    # If capacity is too large for DP, use Gurobi
    
    dp_complexity = n * capacity
    
    if dp_complexity <= 500_000_000 and capacity <= 100_000_000:
        # Use DP approach with optimized memory (1D array)
        # But even 100M array might be too much memory. Let's be more conservative.
        if capacity <= 10_000_000:
            # 1D DP
            dp = [0] * (capacity + 1)
            # Track which items are selected using backtracking
            # We'll store decisions in a 2D bit array - but that's too much memory for large cases
            # Instead, use the DP and then backtrack
            
            # For backtracking, we need to store the DP table or use a different approach
            # Let's use a compact approach: store dp values and backtrack
            
            # Actually for large n and capacity, storing the full table is expensive
            # Let's just do 1D DP and backtrack by recomputing
            
            # Standard approach: compute dp, then backtrack
            # We need to know for each (i, w) whether item i was included
            # Memory efficient: use 2 rows approach and divide-and-conquer for reconstruction
            # Or just store keep array if memory allows
            
            if n * (capacity + 1) <= 200_000_000:
                # Can afford keep array (as bytes)
                keep = bytearray(n * (capacity + 1))
                
                for i in range(n):
                    if time.time() - start_time > time_limit - 1:
                        break
                    w_i = weights[i]
                    p_i = profits[i]
                    # Process in reverse to use 1D DP
                    row_offset = i * (capacity + 1)
                    for w in range(capacity, w_i - 1, -1):
                        if dp[w - w_i] + p_i > dp[w]:
                            dp[w] = dp[w - w_i] + p_i
                            keep[row_offset + w] = 1
                
                # Backtrack
                selected = []
                w = capacity
                for i in range(n - 1, -1, -1):
                    if keep[i * (capacity + 1) + w]:
                        selected.append(ids[i])
                        w -= weights[i]
                
                obj = dp[capacity]
                if logger:
                    logger.log(obj)
                
                solution = {
                    "objective_value": obj,
                    "selected_items": selected
                }
                with open(solution_path, 'w') as f:
                    json.dump(solution, f)
                return
            else:
                # Use Gurobi instead
                pass
        # else use Gurobi
    
    # Use Gurobi for larger instances
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 1:
            remaining_time = 1
        
        model = gp.Model("knapsack")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', remaining_time - 0.5)
        model.setParam('Threads', 1)
        
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        
        model.setObjective(
            gp.quicksum(profits[i] * x[i] for i in range(n)),
            GRB.MAXIMIZE
        )
        
        model.addConstr(
            gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity
        )
        
        # Callback for logging
        best_obj = [0]
        best_solution = [None]
        
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                obj_int = int(round(obj))
                if obj_int > best_obj[0]:
                    best_obj[0] = obj_int
                    vals = model.cbGetSolution(x)
                    best_solution[0] = [ids[i] for i in range(n) if vals[i] > 0.5]
                    if logger:
                        logger.log(obj_int)
        
        model.optimize(callback)
        
        if model.SolCount > 0:
            obj = int(round(model.ObjVal))
            selected = [ids[i] for i in range(n) if x[i].X > 0.5]
        elif best_solution[0] is not None:
            obj = best_obj[0]
            selected = best_solution[0]
        else:
            # Greedy fallback
            ratio = [(profits[i] / weights[i], i) for i in range(n)]
            ratio.sort(reverse=True)
            selected = []
            obj = 0
            remaining_cap = capacity
            for _, i in ratio:
                if weights[i] <= remaining_cap:
                    selected.append(ids[i])
                    obj += profits[i]
                    remaining_cap -= weights[i]
            if logger:
                logger.log(obj)
        
        solution = {
            "objective_value": obj,
            "selected_items": selected
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        
    except ImportError:
        # Fallback greedy
        ratio = [(profits[i] / weights[i], i) for i in range(n)]
        ratio.sort(reverse=True)
        selected = []
        obj = 0
        remaining_cap = capacity
        for _, i in ratio:
            if weights[i] <= remaining_cap:
                selected.append(ids[i])
                obj += profits[i]
                remaining_cap -= weights[i]
        if logger:
            logger.log(obj)
        
        solution = {
            "objective_value": obj,
            "selected_items": selected
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_knapsack(args.instance_path, args.solution_path, args.time_limit, args.log_path)