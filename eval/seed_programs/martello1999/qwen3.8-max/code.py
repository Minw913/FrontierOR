import json
import argparse
import time
from solution_logger import SolutionLogger


def greedy_solution(items, capacity):
    """Greedy solution based on profit/weight ratio."""
    sorted_items = sorted(items, key=lambda x: x['profit'] / x['weight'], reverse=True)
    
    selected = []
    total_weight = 0
    total_profit = 0
    
    for item in sorted_items:
        if total_weight + item['weight'] <= capacity:
            selected.append(item['id'])
            total_weight += item['weight']
            total_profit += item['profit']
    
    return total_profit, selected


def dp_solution(items, capacity, time_limit, logger):
    """Dynamic programming solution for knapsack (exact)."""
    n = len(items)
    start_time = time.time()
    
    # 1D DP array
    dp = [0] * (capacity + 1)
    # keep[i][w] = True if item i is included in optimal solution for capacity w
    # Using a list of bytearrays for memory efficiency
    keep = [bytearray(capacity + 1) for _ in range(n)]
    
    for i in range(n):
        if time.time() - start_time > time_limit * 0.9:
            return None, None  # Timeout, return None to signal fallback
        
        w_i = items[i]['weight']
        p_i = items[i]['profit']
        for w in range(capacity, w_i - 1, -1):
            if dp[w - w_i] + p_i > dp[w]:
                dp[w] = dp[w - w_i] + p_i
                keep[i][w] = 1
    
    # Reconstruct solution
    selected = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            selected.append(items[i]['id'])
            w -= items[i]['weight']
    
    selected.reverse()
    obj_val = dp[capacity]
    
    return obj_val, selected


def solve_with_gurobi(items, capacity, time_limit, logger, initial_profit, initial_selected):
    """Solve knapsack using Gurobi MIP solver."""
    import gurobipy as gp
    from gurobipy import GRB
    
    n = len(items)
    
    best_profit = initial_profit
    best_selected = initial_selected
    
    model = gp.Model("knapsack")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    
    # Decision variables
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
    
    # Set initial solution from greedy
    greedy_set = set(initial_selected)
    for i in range(n):
        if items[i]['id'] in greedy_set:
            x[i].Start = 1.0
        else:
            x[i].Start = 0.0
    
    # Objective: maximize profit
    model.setObjective(
        gp.quicksum(items[i]['profit'] * x[i] for i in range(n)),
        GRB.MAXIMIZE
    )
    
    # Constraint: total weight <= capacity
    model.addConstr(
        gp.quicksum(items[i]['weight'] * x[i] for i in range(n)) <= capacity
    )
    
    # Callback to log intermediate solutions
    def callback(model, where):
        nonlocal best_profit, best_selected
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_int = int(round(obj_val))
            if obj_int > best_profit:
                sol = model.cbGetSolution(x)
                selected = [items[i]['id'] for i in range(n) if sol[i] > 0.5]
                best_profit = obj_int
                best_selected = selected
                if logger:
                    solution_dict = {
                        "objective_value": best_profit,
                        "selected_items": best_selected
                    }
                    logger.log_solution(best_profit, solution_dict)
    
    model.optimize(callback)
    
    # Get final solution from model
    if model.SolCount > 0:
        obj_val = model.ObjVal
        obj_int = int(round(obj_val))
        if obj_int > best_profit:
            selected = [items[i]['id'] for i in range(n) if x[i].X > 0.5]
            best_profit = obj_int
            best_selected = selected
            if logger:
                solution_dict = {
                    "objective_value": best_profit,
                    "selected_items": best_selected
                }
                logger.log_solution(best_profit, solution_dict)
    
    return best_profit, best_selected


def main():
    parser = argparse.ArgumentParser(description="0/1 Knapsack Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    capacity = instance['capacity']
    items = instance['items']
    n = len(items)
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    start_time = time.time()
    
    # Step 1: Greedy solution as initial incumbent
    greedy_profit, greedy_selected = greedy_solution(items, capacity)
    
    best_profit = greedy_profit
    best_selected = greedy_selected
    
    if logger:
        solution_dict = {
            "objective_value": best_profit,
            "selected_items": best_selected
        }
        logger.log_solution(best_profit, solution_dict)
    
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    if remaining <= 0:
        # Write solution and exit
        solution = {
            "objective_value": best_profit,
            "selected_items": best_selected
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return
    
    # Step 2: Try DP if feasible (exact solution)
    # Threshold for DP: n * capacity should be manageable
    dp_threshold = 8_000_000
    if n * capacity <= dp_threshold and remaining > 1:
        dp_result = dp_solution(items, capacity, remaining, logger)
        if dp_result[0] is not None:
            dp_profit, dp_selected = dp_result
            if dp_profit > best_profit:
                best_profit = dp_profit
                best_selected = dp_selected
                if logger:
                    solution_dict = {
                        "objective_value": best_profit,
                        "selected_items": best_selected
                    }
                    logger.log_solution(best_profit, solution_dict)
            # DP gives exact solution, we're done
            solution = {
                "objective_value": best_profit,
                "selected_items": best_selected
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f)
            return
    
    # Step 3: Use Gurobi for larger instances
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    if remaining > 0.5:
        best_profit, best_selected = solve_with_gurobi(
            items, capacity, remaining, logger, best_profit, best_selected
        )
    
    # Write final solution
    solution = {
        "objective_value": best_profit,
        "selected_items": best_selected
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()