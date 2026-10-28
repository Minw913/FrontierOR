import argparse
import json
import time
import math

def compute_customer_cost(allocation_costs, open_facilities, j):
    """Compute optimal allocation cost for customer j given set of open facilities."""
    inv_sum = 0.0
    for i in open_facilities:
        if allocation_costs[i][j] > 0:
            inv_sum += 1.0 / allocation_costs[i][j]
    if inv_sum == 0:
        return float('inf')
    return 1.0 / inv_sum

def compute_total_cost(opening_costs, allocation_costs, open_set, num_customers):
    """Compute total cost for a given set of open facilities."""
    total = sum(opening_costs[i] for i in open_set)
    for j in range(num_customers):
        total += compute_customer_cost(allocation_costs, open_set, j)
    return total

def greedy_add(opening_costs, allocation_costs, num_facilities, num_customers):
    """Greedy heuristic: iteratively add the facility that gives best improvement."""
    open_set = set()
    
    # Start by opening the facility with lowest opening cost
    # Actually, let's compute the cost with no facilities (infinite) and add greedily
    
    # Compute cost with all facilities open as baseline for comparison
    all_set = set(range(num_facilities))
    
    # Greedy: add facilities one by one
    best_cost = float('inf')
    
    while True:
        best_facility = -1
        best_improvement = 0
        
        for i in range(num_facilities):
            if i in open_set:
                continue
            # Compute cost if we add facility i
            trial_set = open_set | {i}
            cost = compute_total_cost(opening_costs, allocation_costs, trial_set, num_customers)
            if cost < best_cost:
                improvement = best_cost - cost
                if improvement > best_improvement or best_facility == -1:
                    best_improvement = improvement
                    best_facility = i
        
        if best_facility == -1:
            break
            
        open_set.add(best_facility)
        best_cost = compute_total_cost(opening_costs, allocation_costs, open_set, num_customers)
        
        # Check if adding more helps
        can_improve = False
        for i in range(num_facilities):
            if i not in open_set:
                trial_set = open_set | {i}
                trial_cost = compute_total_cost(opening_costs, allocation_costs, trial_set, num_customers)
                if trial_cost < best_cost:
                    can_improve = True
                    break
        
        if not can_improve:
            break
    
    return open_set, best_cost

def greedy_remove(opening_costs, allocation_costs, num_facilities, num_customers):
    """Greedy heuristic: start with all open, iteratively remove worst."""
    open_set = set(range(num_facilities))
    current_cost = compute_total_cost(opening_costs, allocation_costs, open_set, num_customers)
    
    while len(open_set) > 1:
        best_facility = -1
        best_cost_after = float('inf')
        
        for i in open_set:
            trial_set = open_set - {i}
            cost = compute_total_cost(opening_costs, allocation_costs, trial_set, num_customers)
            if cost < best_cost_after:
                best_cost_after = cost
                best_facility = i
        
        if best_cost_after >= current_cost:
            break
        
        open_set.remove(best_facility)
        current_cost = best_cost_after
    
    return open_set, current_cost

def local_search(opening_costs, allocation_costs, num_facilities, num_customers, open_set, time_limit, start_time):
    """Local search: try adding/removing/swapping facilities."""
    current_cost = compute_total_cost(opening_costs, allocation_costs, open_set, num_customers)
    improved = True
    
    while improved and (time.time() - start_time) < time_limit * 0.8:
        improved = False
        
        # Try removing each open facility
        for i in list(open_set):
            if len(open_set) <= 1:
                break
            trial_set = open_set - {i}
            cost = compute_total_cost(opening_costs, allocation_costs, trial_set, num_customers)
            if cost < current_cost:
                open_set = trial_set
                current_cost = cost
                improved = True
        
        # Try adding each closed facility
        for i in range(num_facilities):
            if i not in open_set:
                trial_set = open_set | {i}
                cost = compute_total_cost(opening_costs, allocation_costs, trial_set, num_customers)
                if cost < current_cost:
                    open_set = trial_set
                    current_cost = cost
                    improved = True
        
        # Try swaps (remove one, add another)
        if (time.time() - start_time) < time_limit * 0.6:
            for i in list(open_set):
                for k in range(num_facilities):
                    if k not in open_set and (time.time() - start_time) < time_limit * 0.6:
                        trial_set = (open_set - {i}) | {k}
                        cost = compute_total_cost(opening_costs, allocation_costs, trial_set, num_customers)
                        if cost < current_cost:
                            open_set = trial_set
                            current_cost = cost
                            improved = True
                            break
                if improved:
                    break
    
    return open_set, current_cost

def compute_optimal_allocation(allocation_costs, open_set, num_customers):
    """Compute optimal allocation fractions for given open facilities."""
    num_facilities = len(allocation_costs)
    x_vals = [[0.0] * num_customers for _ in range(num_facilities)]
    
    for j in range(num_customers):
        inv_sum = 0.0
        for i in open_set:
            if allocation_costs[i][j] > 0:
                inv_sum += 1.0 / allocation_costs[i][j]
        
        if inv_sum > 0:
            for i in open_set:
                if allocation_costs[i][j] > 0:
                    x_vals[i][j] = (1.0 / allocation_costs[i][j]) / inv_sum
    
    return x_vals

def solve_with_gurobi(instance, args, logger, heuristic_solution=None):
    """Solve using Gurobi MIQP."""
    import gurobipy as gp
    from gurobipy import GRB
    
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    opening_costs = instance['opening_costs']
    allocation_costs = instance['allocation_costs']
    
    remaining_time = args.time_limit - (time.time() - args._start_time)
    if remaining_time <= 0:
        return None
    
    try:
        model = gp.Model("qUFL")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, remaining_time))
        model.setParam('OutputFlag', 0)
        
        # Decision variables
        y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")
        x = model.addVars(num_facilities, num_customers, lb=0.0, ub=1.0, name="x")
        
        # Objective
        obj = gp.LinExpr()
        for i in range(num_facilities):
            obj += opening_costs[i] * y[i]
        
        qobj = gp.QuadExpr()
        for i in range(num_facilities):
            for j in range(num_customers):
                if allocation_costs[i][j] > 0:
                    qobj += allocation_costs[i][j] * x[i, j] * x[i, j]
        
        model.setObjective(obj + qobj, GRB.MINIMIZE)
        
        # Demand constraints
        for j in range(num_customers):
            model.addConstr(gp.quicksum(x[i, j] for i in range(num_facilities)) == 1.0, name=f"demand_{j}")
        
        # Linking constraints
        for i in range(num_facilities):
            for j in range(num_customers):
                model.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")
        
        # Set MIP start from heuristic if available
        if heuristic_solution is not None:
            y_vals, x_vals = heuristic_solution
            for i in range(num_facilities):
                y[i].Start = y_vals[i]
            for i in range(num_facilities):
                for j in range(num_customers):
                    x[i, j].Start = x_vals[i][j]
        
        model.optimize()
        
        if model.SolCount > 0:
            y_sol = [int(round(y[i].X)) for i in range(num_facilities)]
            x_sol = [[max(0.0, min(1.0, x[i, j].X)) for j in range(num_customers)] for i in range(num_facilities)]
            
            # Recompute objective from solution values for accuracy
            obj_val = sum(opening_costs[i] * y_sol[i] for i in range(num_facilities))
            for i in range(num_facilities):
                for j in range(num_customers):
                    obj_val += allocation_costs[i][j] * x_sol[i][j] ** 2
            
            return obj_val, y_sol, x_sol
        
        return None
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    args._start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    opening_costs = instance['opening_costs']
    allocation_costs = instance['allocation_costs']

    # Initialize logger
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Phase 1: Greedy heuristics
    open_set_add, cost_add = greedy_add(opening_costs, allocation_costs, num_facilities, num_customers)
    
    elapsed = time.time() - args._start_time
    if elapsed < args.time_limit * 0.5:
        open_set_rem, cost_rem = greedy_remove(opening_costs, allocation_costs, num_facilities, num_customers)
    else:
        open_set_rem, cost_rem = open_set_add, cost_add

    # Pick best greedy solution
    if cost_add <= cost_rem:
        open_set = open_set_add
        best_cost = cost_add
    else:
        open_set = open_set_rem
        best_cost = cost_rem

    # Phase 2: Local search improvement
    elapsed = time.time() - args._start_time
    if elapsed < args.time_limit * 0.6 and num_facilities <= 200:
        open_set, best_cost = local_search(opening_costs, allocation_costs, num_facilities, num_customers, open_set, args.time_limit, args._start_time)

    # Compute allocation for best heuristic solution
    x_vals = compute_optimal_allocation(allocation_costs, open_set, num_customers)
    y_vals = [1 if i in open_set else 0 for i in range(num_facilities)]
    
    # Recompute objective precisely
    obj_val = sum(opening_costs[i] * y_vals[i] for i in range(num_facilities))
    for i in range(num_facilities):
        for j in range(num_customers):
            obj_val += allocation_costs[i][j] * x_vals[i][j] ** 2

    solution = {
        "objective_value": obj_val,
        "variant": "quadratic",
        "y": y_vals,
        "x": x_vals
    }

    if logger:
        logger.log_solution(obj_val, solution)

    # Phase 3: Try Gurobi MIQP if time permits and problem is manageable
    elapsed = time.time() - args._start_time
    remaining = args.time_limit - elapsed
    
    # Only use Gurobi if problem size is reasonable and we have time
    if remaining > 5 and num_facilities * num_customers <= 500000:
        gurobi_result = solve_with_gurobi(instance, args, logger, (y_vals, x_vals))
        
        if gurobi_result is not None:
            g_obj, g_y, g_x = gurobi_result
            if g_obj < obj_val:
                obj_val = g_obj
                y_vals = g_y
                x_vals = g_x
                
                solution = {
                    "objective_value": obj_val,
                    "variant": "quadratic",
                    "y": y_vals,
                    "x": x_vals
                }
                
                if logger:
                    logger.log_solution(obj_val, solution)

    # Write final solution
    solution = {
        "objective_value": obj_val,
        "variant": "quadratic",
        "y": y_vals,
        "x": x_vals
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()