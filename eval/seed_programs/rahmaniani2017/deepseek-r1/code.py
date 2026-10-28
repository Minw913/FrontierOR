import argparse
import json
import time
import math
from solution_logger import SolutionLogger

# Attempt to import Gurobi for small instances
try:
    import gurobipy as grb
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

def build_trivial_solution(instance):
    """Fallback solution using cheapest single hub"""
    nodes = instance['nodes']
    commodities = instance['commodities']
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    
    n = len(nodes)
    fixed_costs = [node['fixed_cost'] for node in nodes]
    min_fixed_cost = min(fixed_costs)
    min_index = fixed_costs.index(min_fixed_cost)
    open_hubs = [min_index]
    
    assignments = []
    total_cost = min_fixed_cost
    for idx_k, com in enumerate(commodities):
        o = com['origin']
        d = com['destination']
        flow = com['flow']
        i = min_index
        j = min_index
        cost_val = flow * (chi * distances[o][i] + tau * distances[i][j] + delta * distances[j][d])
        total_cost += cost_val
        hub_edge = [i] if i == j else sorted([i, j])
        assignments.append({
            "commodity": idx_k,
            "origin": o,
            "destination": d,
            "hub_edge": hub_edge,
            "cost": cost_val
        })
    
    return {
        "objective_value": total_cost,
        "open_hubs": open_hubs,
        "assignments": assignments
    }

def compute_total_cost(S, instance, deadline=None):
    """Compute total cost for hub set S with early termination"""
    if deadline and time.time() > deadline:
        return float('inf')
        
    nodes = instance['nodes']
    commodities = instance['commodities']
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    
    total_fixed = sum(nodes[i]['fixed_cost'] for i in S)
    total_transport = 0
    S_list = list(S)
    
    for com in commodities:
        if deadline and time.time() > deadline:
            return float('inf')
            
        o = com['origin']
        d = com['destination']
        flow = com['flow']
        best_cost_val = float('inf')
        
        # Find best hub pair
        for i in S_list:
            cost_i = chi * distances[o][i]
            best_transfer = float('inf')
            for j in S_list:
                transfer_cost = tau * distances[i][j] + delta * distances[j][d]
                if transfer_cost < best_transfer:
                    best_transfer = transfer_cost
            total_leg_cost = cost_i + best_transfer
            if total_leg_cost < best_cost_val:
                best_cost_val = total_leg_cost
                
        total_transport += flow * best_cost_val
        
    return total_fixed + total_transport

def greedy_heuristic(instance, time_limit, logger=None):
    """Efficient greedy heuristic for large instances"""
    start_time = time.time()
    deadline = start_time + time_limit
    
    nodes = instance['nodes']
    n = len(nodes)
    commodities = instance['commodities']
    fixed_costs = [node['fixed_cost'] for node in nodes]
    
    # Step 1: Find best single hub
    best_S = set()
    best_cost = float('inf')
    for i in range(n):
        if time.time() > deadline:
            break
        cost_i = compute_total_cost({i}, instance, deadline)
        if cost_i < best_cost:
            best_cost = cost_i
            best_S = {i}
            if logger:
                logger.log(best_cost)
    
    if not best_S:  # Timeout during initialization
        return build_trivial_solution(instance)
    
    current_S = best_S.copy()
    current_cost = best_cost
    candidates = set(range(n)) - current_S
    
    # Step 2: Greedy adding
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = float('inf')
        best_j = None
        
        for j in candidates:
            if time.time() > deadline:
                break
            new_S = current_S | {j}
            new_cost = compute_total_cost(new_S, instance, deadline)
            delta_cost = new_cost - current_cost
            if delta_cost < best_delta:
                best_delta = delta_cost
                best_j = j
                
        if best_j is not None and best_delta < 0:
            current_S.add(best_j)
            candidates.remove(best_j)
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = current_S.copy()
                if logger:
                    logger.log(best_cost)
            improved = True
    
    # Step 3: Greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = float('inf')
        best_j = None
        
        for j in list(current_S):
            if time.time() > deadline:
                break
            new_S = current_S - {j}
            new_cost = compute_total_cost(new_S, instance, deadline)
            delta_cost = new_cost - current_cost
            if delta_cost < best_delta:
                best_delta = delta_cost
                best_j = j
                
        if best_j is not None and best_delta < 0:
            current_S.remove(best_j)
            candidates.add(best_j)
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = current_S.copy()
                if logger:
                    logger.log(best_cost)
            improved = True
    
    # Step 4: Greedy swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = float('inf')
        best_remove = None
        best_add = None
        
        for j in list(current_S):
            if time.time() > deadline:
                break
            for k in candidates:
                if time.time() > deadline:
                    break
                new_S = current_S - {j} | {k}
                new_cost = compute_total_cost(new_S, instance, deadline)
                delta_cost = new_cost - current_cost
                if delta_cost < best_delta:
                    best_delta = delta_cost
                    best_remove = j
                    best_add = k
                    
        if best_remove is not None and best_delta < 0:
            current_S.remove(best_remove)
            current_S.add(best_add)
            candidates.remove(best_add)
            candidates.add(best_remove)
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = current_S.copy()
                if logger:
                    logger.log(best_cost)
            improved = True
    
    # Build final solution
    return build_solution(best_S, instance, deadline)

def build_solution(S, instance, deadline=None):
    """Construct solution dictionary from hub set S"""
    if not S:  # Fallback if no hubs selected
        return build_trivial_solution(instance)
        
    nodes = instance['nodes']
    commodities = instance['commodities']
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    
    open_hubs = sorted(S)
    total_fixed = sum(nodes[i]['fixed_cost'] for i in S)
    assignments = []
    total_transport = 0
    S_list = list(S)
    
    for idx_k, com in enumerate(commodities):
        if deadline and time.time() > deadline:
            break
            
        o = com['origin']
        d = com['destination']
        flow = com['flow']
        best_cost_val = float('inf')
        best_i = None
        best_j = None
        
        # Find best hub pair
        for i in S_list:
            cost_i = chi * distances[o][i]
            best_transfer = float('inf')
            best_j_candidate = None
            for j in S_list:
                transfer_cost = tau * distances[i][j] + delta * distances[j][d]
                if transfer_cost < best_transfer:
                    best_transfer = transfer_cost
                    best_j_candidate = j
            total_leg_cost = cost_i + best_transfer
            if total_leg_cost < best_cost_val:
                best_cost_val = total_leg_cost
                best_i = i
                best_j = best_j_candidate
                
        transport_cost = flow * best_cost_val
        total_transport += transport_cost
        hub_edge = [best_i] if best_i == best_j else sorted([best_i, best_j])
        assignments.append({
            "commodity": idx_k,
            "origin": o,
            "destination": d,
            "hub_edge": hub_edge,
            "cost": transport_cost
        })
    
    return {
        "objective_value": total_fixed + total_transport,
        "open_hubs": open_hubs,
        "assignments": assignments
    }

def run_MIP(instance, time_limit, logger=None):
    """MIP solver for small instances"""
    if not GUROBI_AVAILABLE:
        return None
        
    nodes = instance['nodes']
    n = len(nodes)
    commodities = instance['commodities']
    num_commodities = len(commodities)
    distances = instance['distances']
    params = instance['parameters']
    chi = params['chi']
    tau = params['tau']
    delta = params['delta']
    fixed_costs = [node['fixed_cost'] for node in nodes]
    
    try:
        model = grb.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', time_limit)
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        
        # Decision variables
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        z = model.addVars(num_commodities, n, n, vtype=GRB.BINARY, name="z")
        
        # Objective function
        fixed_cost_expr = grb.quicksum(fixed_costs[i] * y[i] for i in range(n))
        transport_cost_expr = 0
        for idx_k, com in enumerate(commodities):
            o = com['origin']
            d = com['destination']
            flow = com['flow']
            for i in range(n):
                for j in range(n):
                    cost_val = flow * (chi * distances[o][i] + 
                                      tau * distances[i][j] + 
                                      delta * distances[j][d])
                    transport_cost_expr += z[idx_k, i, j] * cost_val
                    
        model.setObjective(fixed_cost_expr + transport_cost_expr, GRB.MINIMIZE)
        
        # Constraints
        for idx_k in range(num_commodities):
            model.addConstr(grb.quicksum(z[idx_k, i, j] for i in range(n) for j in range(n)) == 1)
            
        for idx_k in range(num_commodities):
            for i in range(n):
                for j in range(n):
                    model.addConstr(z[idx_k, i, j] <= y[i])
                    model.addConstr(z[idx_k, i, j] <= y[j])
        
        # Solve with logging
        if logger:
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj)
            model.optimize(callback)
        else:
            model.optimize()
            
        # Extract solution
        if model.SolCount > 0:
            open_hubs = [i for i in range(n) if y[i].x > 0.5]
            assignments = []
            for idx_k, com in enumerate(commodities):
                o = com['origin']
                d = com['destination']
                flow = com['flow']
                for i in range(n):
                    for j in range(n):
                        if z[idx_k, i, j].x > 0.5:
                            cost_val = flow * (chi * distances[o][i] + 
                                              tau * distances[i][j] + 
                                              delta * distances[j][d])
                            hub_edge = [i] if i == j else sorted([i, j])
                            assignments.append({
                                "commodity": idx_k,
                                "origin": o,
                                "destination": d,
                                "hub_edge": hub_edge,
                                "cost": cost_val
                            })
                            break
                    else:
                        continue
                    break
            return {
                "objective_value": model.objVal,
                "open_hubs": open_hubs,
                "assignments": assignments
            }
    except Exception as e:
        print(f"MIP solver error: {str(e)}")
        
    return None

def main(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Load instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Calculate remaining time
    elapsed = time.time() - start_time
    time_left = max(1, time_limit - elapsed)
    
    # Check instance size
    n = instance['num_nodes']
    num_commodities = instance['num_commodities']
    size_factor = n * n * num_commodities
    
    # Solve with appropriate method
    solution = None
    if GUROBI_AVAILABLE and size_factor <= 1000000 and time_left > 5:
        solution = run_MIP(instance, time_left, logger)
    
    if solution is None and time_left > 1:
        solution = greedy_heuristic(instance, time_left, logger)
    
    if solution is None:
        solution = build_trivial_solution(instance)
        if logger:
            logger.log(solution['objective_value'])
    
    # Write final solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)