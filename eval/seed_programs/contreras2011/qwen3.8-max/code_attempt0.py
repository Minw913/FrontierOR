import json
import argparse
import time
import numpy as np
import sys

def compute_objective(hubs_list, assignment, n, W, D, O, I, setup_costs, chi, alpha, delta):
    """Compute total objective value."""
    cost = 0.0
    hub_set = set(hubs_list)
    # Setup costs
    for k in hub_set:
        cost += setup_costs[k]
    # Collection + Distribution costs
    for i in range(n):
        k = assignment[i]
        cost += O[i] * chi * D[i][k] + I[i] * delta * D[k][i]
    # Transfer cost
    for i in range(n):
        ki = assignment[i]
        row_W = W[i]
        for j in range(n):
            if row_W[j] > 0:
                cost += alpha * row_W[j] * D[ki][assignment[j]]
    return cost


def compute_objective_fast(hubs_list, assignment, n, W, D, O, I, setup_costs, chi, alpha, delta):
    """Faster objective computation using numpy."""
    cost = 0.0
    hub_set = set(hubs_list)
    # Setup costs
    for k in hub_set:
        cost += setup_costs[k]
    # Collection + Distribution costs
    for i in range(n):
        k = assignment[i]
        cost += O[i] * chi * D[i][k] + I[i] * delta * D[k][i]
    # Transfer cost using matrix approach
    hubs_sorted = sorted(hub_set)
    hub_to_idx = {h: idx for idx, h in enumerate(hubs_sorted)}
    num_hubs = len(hubs_sorted)
    A = np.zeros((n, num_hubs))
    for i in range(n):
        A[i, hub_to_idx[assignment[i]]] = 1.0
    T = A.T @ W @ A
    D_sub = D[np.ix_(hubs_sorted, hubs_sorted)]
    cost += alpha * np.sum(D_sub * T)
    return cost


def get_hub_loads(assignment, hubs_list, O, n):
    """Compute load on each hub."""
    loads = {k: 0.0 for k in hubs_list}
    for i in range(n):
        loads[assignment[i]] += O[i]
    return loads


def reassignment_cost_change(i, k_old, k_new, assignment, W, D, O, I, chi, alpha, delta, n):
    """Compute cost change of reassigning node i from k_old to k_new."""
    change = O[i] * chi * (D[i][k_new] - D[i][k_old])
    change += I[i] * delta * (D[k_new][i] - D[k_old][i])
    # Transfer cost change
    Wi = W[i]
    Wcol_i = W[:, i]
    for j in range(n):
        if j == i:
            continue
        h_j = assignment[j]
        flow_sum = Wi[j] + Wcol_i[j]
        if flow_sum > 0:
            change += alpha * flow_sum * (D[k_new][h_j] - D[k_old][h_j])
    return change


def optimize_assignment(hubs_list, n, W, D, O, I, chi, alpha, delta, capacities, max_iter=50):
    """Find best assignment of nodes to given hubs."""
    hub_set = set(hubs_list)
    
    # Initial assignment: minimize individual cost (collection + distribution)
    assignment = np.zeros(n, dtype=int)
    for i in range(n):
        if i in hub_set:
            assignment[i] = i
            continue
        best_cost = float('inf')
        best_hub = hubs_list[0]
        for k in hubs_list:
            cost = O[i] * chi * D[i][k] + I[i] * delta * D[k][i]
            if cost < best_cost:
                best_cost = cost
                best_hub = k
        assignment[i] = best_hub
    
    # Fix capacity violations
    loads = get_hub_loads(assignment, hubs_list, O, n)
    for k in hubs_list:
        while loads[k] > capacities[k] + 1e-10:
            # Find node assigned to k with smallest O[i] that can be moved
            nodes_at_k = [i for i in range(n) if assignment[i] == k and i != k]
            if not nodes_at_k:
                break
            # Try to move the node that causes least cost increase
            best_node = None
            best_cost_inc = float('inf')
            best_new_hub = None
            for i in nodes_at_k:
                for h in hubs_list:
                    if h == k:
                        continue
                    if loads[h] + O[i] <= capacities[h] + 1e-10:
                        cost_inc = reassignment_cost_change(i, k, h, assignment, W, D, O, I, chi, alpha, delta, n)
                        if cost_inc < best_cost_inc:
                            best_cost_inc = cost_inc
                            best_node = i
                            best_new_hub = h
            if best_node is None:
                break
            loads[k] -= O[best_node]
            loads[best_new_hub] += O[best_node]
            assignment[best_node] = best_new_hub
    
    # Iterative improvement considering transfer costs
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            if i in hub_set:
                continue  # Hub must be assigned to itself
            current_hub = assignment[i]
            best_change = -1e-12
            best_new_hub = current_hub
            
            for k in hubs_list:
                if k == current_hub:
                    continue
                # Check capacity
                if loads[k] + O[i] > capacities[k] + 1e-10:
                    continue
                change = reassignment_cost_change(i, current_hub, k, assignment, W, D, O, I, chi, alpha, delta, n)
                if change < best_change:
                    best_change = change
                    best_new_hub = k
            
            if best_new_hub != current_hub:
                loads[current_hub] -= O[i]
                loads[best_new_hub] += O[i]
                assignment[i] = best_new_hub
                improved = True
        
        if not improved:
            break
    
    return assignment


def greedy_hub_selection(n, W, D, O, I, setup_costs, capacities, chi, alpha, delta, p):
    """Select p hubs using a greedy scoring approach."""
    scores = np.zeros(n)
    for k in range(n):
        # Score: lower is better
        # Consider setup cost, centrality, and capacity
        dist_cost = 0.0
        for i in range(n):
            dist_cost += (O[i] * chi + I[i] * delta) * D[i][k]
        # Flow through node k
        flow_through = O[k] + I[k]
        scores[k] = setup_costs[k] + dist_cost * 0.1 - capacities[k] * 0.001
    
    selected = np.argsort(scores)[:p].tolist()
    return selected


def flow_based_hub_selection(n, W, D, O, I, setup_costs, capacities, p):
    """Select hubs based on flow volume."""
    flow_score = O + I  # Total flow through each node
    # Normalize by setup cost
    score = flow_score / (setup_costs + 1.0)
    selected = np.argsort(-score)[:p].tolist()
    return selected


def central_hub_selection(n, W, D, O, I, setup_costs, capacities, p):
    """Select hubs that are most central (minimize weighted distance)."""
    weighted_dist = np.zeros(n)
    for k in range(n):
        for i in range(n):
            weighted_dist[k] += (O[i] + I[i]) * D[i][k]
    score = weighted_dist + setup_costs * 10
    selected = np.argsort(score)[:p].tolist()
    return selected


def local_search(hubs_list, assignment, n, W, D, O, I, setup_costs, capacities, chi, alpha, delta, time_limit, start_time):
    """Perform local search to improve solution."""
    best_obj = compute_objective_fast(hubs_list, assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
    best_hubs = hubs_list[:]
    best_assignment = assignment.copy()
    
    no_improve_count = 0
    max_no_improve = 3
    
    while no_improve_count < max_no_improve:
        if time.time() - start_time > time_limit * 0.85:
            break
        
        improved = False
        
        # Operator 1: Reassign nodes
        loads = get_hub_loads(assignment, hubs_list, O, n)
        for i in range(n):
            if i in set(hubs_list):
                continue
            current_hub = assignment[i]
            for k in hubs_list:
                if k == current_hub:
                    continue
                if loads[k] + O[i] > capacities[k] + 1e-10:
                    continue
                change = reassignment_cost_change(i, current_hub, k, assignment, W, D, O, I, chi, alpha, delta, n)
                if change < -1e-10:
                    loads[current_hub] -= O[i]
                    loads[k] += O[i]
                    assignment[i] = k
                    improved = True
                    break
        
        if improved:
            obj = compute_objective_fast(hubs_list, assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
            if obj < best_obj - 1e-10:
                best_obj = obj
                best_hubs = hubs_list[:]
                best_assignment = assignment.copy()
                no_improve_count = 0
                continue
        
        # Operator 2: Swap hub with non-hub
        if time.time() - start_time > time_limit * 0.7:
            break
        
        hub_set = set(hubs_list)
        non_hubs = [i for i in range(n) if i not in hub_set]
        
        swap_improved = False
        for h in hubs_list[:]:
            for v in non_hubs:
                if time.time() - start_time > time_limit * 0.8:
                    break
                # Try swapping h and v
                new_hubs = [x if x != h else v for x in hubs_list]
                new_hub_set = set(new_hubs)
                
                # Check if capacity is sufficient
                if capacities[v] < O[v]:
                    continue
                
                # Quick check: is total capacity sufficient?
                total_cap = sum(capacities[k] for k in new_hubs)
                if total_cap < sum(O):
                    continue
                
                # Reassign
                new_assignment = optimize_assignment(new_hubs, n, W, D, O, I, chi, alpha, delta, capacities, max_iter=20)
                
                # Verify feasibility
                loads = get_hub_loads(new_assignment, new_hubs, O, n)
                feasible = all(loads.get(k, 0) <= capacities[k] + 1e-10 for k in new_hubs)
                if not feasible:
                    continue
                
                obj = compute_objective_fast(new_hubs, new_assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
                if obj < best_obj - 1e-10:
                    best_obj = obj
                    best_hubs = new_hubs
                    best_assignment = new_assignment.copy()
                    hubs_list = new_hubs
                    assignment = new_assignment
                    swap_improved = True
                    improved = True
                    break
            if swap_improved:
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Operator 3: Add a hub
        if time.time() - start_time > time_limit * 0.7:
            break
        
        add_improved = False
        for v in non_hubs:
            if capacities[v] < O[v]:
                continue
            new_hubs = hubs_list + [v]
            new_assignment = optimize_assignment(new_hubs, n, W, D, O, I, chi, alpha, delta, capacities, max_iter=15)
            loads = get_hub_loads(new_assignment, new_hubs, O, n)
            feasible = all(loads.get(k, 0) <= capacities[k] + 1e-10 for k in new_hubs)
            if not feasible:
                continue
            obj = compute_objective_fast(new_hubs, new_assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
            if obj < best_obj - 1e-10:
                best_obj = obj
                best_hubs = new_hubs
                best_assignment = new_assignment.copy()
                hubs_list = new_hubs
                assignment = new_assignment
                add_improved = True
                improved = True
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Operator 4: Drop a hub (if more than 1 hub)
        if len(hubs_list) > 1 and time.time() - start_time > time_limit * 0.7:
            drop_improved = False
            for h in hubs_list[:]:
                if len(hubs_list) <= 1:
                    break
                new_hubs = [x for x in hubs_list if x != h]
                total_cap = sum(capacities[k] for k in new_hubs)
                if total_cap < sum(O) - 1e-10:
                    continue
                new_assignment = optimize_assignment(new_hubs, n, W, D, O, I, chi, alpha, delta, capacities, max_iter=15)
                loads = get_hub_loads(new_assignment, new_hubs, O, n)
                feasible = all(loads.get(k, 0) <= capacities[k] + 1e-10 for k in new_hubs)
                if not feasible:
                    continue
                obj = compute_objective_fast(new_hubs, new_assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
                if obj < best_obj - 1e-10:
                    best_obj = obj
                    best_hubs = new_hubs
                    best_assignment = new_assignment.copy()
                    hubs_list = new_hubs
                    assignment = new_assignment
                    drop_improved = True
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
        
        no_improve_count += 1
    
    return best_hubs, best_assignment, best_obj


def solve_mip(n, W, D, O, I, setup_costs, capacities, chi, alpha, delta, total_flow, time_remaining):
    """Solve using Gurobi MIP for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None, None
    
    model = gp.Model("hub_location")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_remaining))
    model.setParam('OutputFlag', 0)
    
    # Decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    z = model.addVars(n, n, vtype=GRB.BINARY, name="z")
    
    # Objective
    obj = gp.LinExpr()
    
    # Setup costs
    for k in range(n):
        obj += setup_costs[k] * x[k]
    
    # Collection costs: sum_i O[i] * chi * D[i][k] * z[i,k]
    for i in range(n):
        for k in range(n):
            coeff = O[i] * chi * D[i][k]
            if coeff > 0:
                obj += coeff * z[i, k]
    
    # Distribution costs: sum_j I[j] * delta * D[m][j] * z[j,m]
    for j in range(n):
        for m in range(n):
            coeff = I[j] * delta * D[m][j]
            if coeff > 0:
                obj += coeff * z[j, m]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Add quadratic transfer cost terms
    # alpha * sum_i sum_j sum_k sum_m W[i][j] * D[k][m] * z[i,k] * z[j,m]
    # Only add non-trivial terms (i != j or k != m with D[k][m] > 0)
    qterms = []
    for i in range(n):
        for j in range(n):
            if W[i][j] <= 0:
                continue
            for k in range(n):
                for m in range(n):
                    if k == m:
                        continue  # D[k][k] = 0
                    coeff = alpha * W[i][j] * D[k][m]
                    if abs(coeff) > 1e-15:
                        qterms.append((coeff, z[i, k], z[j, m]))
    
    if qterms:
        model.addQTerms(*[(t[0], t[1], t[2]) for t in qterms])
    
    # Constraints
    # Each node assigned to exactly one hub
    for i in range(n):
        model.addConstr(gp.quicksum(z[i, k] for k in range(n)) == 1, name=f"assign_{i}")
    
    # Can only assign to open hubs
    for i in range(n):
        for k in range(n):
            model.addConstr(z[i, k] <= x[k], name=f"link_{i}_{k}")
    
    # Hub assigned to itself
    for k in range(n):
        model.addConstr(z[k, k] == x[k], name=f"self_{k}")
    
    # Capacity constraints
    for k in range(n):
        model.addConstr(
            gp.quicksum(O[i] * z[i, k] for i in range(n)) <= capacities[k] * x[k],
            name=f"cap_{k}"
        )
    
    # Combined capacity constraint
    model.addConstr(
        gp.quicksum(capacities[k] * x[k] for k in range(n)) >= total_flow,
        name="total_cap"
    )
    
    # Must have at least one hub
    model.addConstr(gp.quicksum(x[k] for k in range(n)) >= 1)
    
    model.optimize()
    
    if model.SolCount > 0:
        hubs_list = [k for k in range(n) if x[k].X > 0.5]
        assignment = np.zeros(n, dtype=int)
        for i in range(n):
            for k in range(n):
                if z[i, k].X > 0.5:
                    assignment[i] = k
                    break
        obj_val = model.ObjVal
        return hubs_list, assignment, obj_val
    
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Capacitated Hub Location Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        inst = json.load(f)
    
    n = inst['n']
    chi = inst['cost_parameters']['collection_cost_chi']
    alpha = inst['cost_parameters']['transfer_cost_alpha']
    delta = inst['cost_parameters']['distribution_cost_delta']
    W = np.array(inst['flow_matrix'], dtype=np.float64)
    D = np.array(inst['distance_matrix'], dtype=np.float64)
    O = np.array(inst['outgoing_flow'], dtype=np.float64)
    setup_costs = np.array(inst['setup_costs'], dtype=np.float64)
    capacities = np.array(inst['capacities'], dtype=np.float64)
    total_flow = inst['total_flow_D']
    I = W.sum(axis=0)  # incoming flow for each node
    
    start_time = time.time()
    time_limit = args.time_limit
    
    best_obj = float('inf')
    best_hubs = None
    best_assignment = None
    
    def log_solution(hubs_list, assignment_arr, obj_val):
        nonlocal best_obj, best_hubs, best_assignment
        if obj_val < best_obj:
            best_obj = obj_val
            best_hubs = hubs_list[:]
            best_assignment = assignment_arr.copy()
            if logger:
                sol = {
                    "objective_value": float(obj_val),
                    "hubs": sorted([int(h) for h in hubs_list]),
                    "assignment": [int(a) for a in assignment_arr]
                }
                logger.log_solution(float(obj_val), sol)
    
    # Determine max number of hubs to try
    max_p = min(n, max(2, int(np.sqrt(n)) + 3))
    
    # Phase 1: Heuristic construction and local search
    strategies = [
        ('greedy', greedy_hub_selection),
        ('flow', flow_based_hub_selection),
        ('central', central_hub_selection),
    ]
    
    for p in range(1, max_p + 1):
        if time.time() - start_time > time_limit * 0.5:
            break
        
        for strat_name, strat_func in strategies:
            if time.time() - start_time > time_limit * 0.5:
                break
            
            try:
                hubs_list = strat_func(n, W, D, O, I, setup_costs, capacities, chi, alpha, delta, p)
            except:
                continue
            
            # Ensure capacity feasibility
            total_cap = sum(capacities[k] for k in hubs_list)
            if total_cap < total_flow - 1e-10:
                continue
            
            # Optimize assignment
            assignment = optimize_assignment(hubs_list, n, W, D, O, I, chi, alpha, delta, capacities, max_iter=30)
            
            # Verify feasibility
            loads = get_hub_loads(assignment, hubs_list, O, n)
            feasible = all(loads.get(k, 0) <= capacities[k] + 1e-10 for k in hubs_list)
            if not feasible:
                continue
            
            obj = compute_objective_fast(hubs_list, assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
            log_solution(hubs_list, assignment, obj)
            
            # Local search
            if time.time() - start_time > time_limit * 0.6:
                continue
            
            new_hubs, new_assignment, new_obj = local_search(
                hubs_list, assignment, n, W, D, O, I, setup_costs, capacities,
                chi, alpha, delta, time_limit, start_time
            )
            
            if new_obj < obj:
                log_solution(new_hubs, new_assignment, new_obj)
    
    # Phase 2: Additional random restarts if time permits
    np.random.seed(42)
    while time.time() - start_time < time_limit * 0.6:
        p = np.random.randint(1, max_p + 1)
        # Random hub selection weighted by flow
        probs = (O + I + 1.0)
        probs = probs / probs.sum()
        try:
            hubs_list = list(np.random.choice(n, size=min(p, n), replace=False, p=probs))
        except:
            hubs_list = list(np.random.choice(n, size=min(p, n), replace=False))
        
        total_cap = sum(capacities[k] for k in hubs_list)
        if total_cap < total_flow - 1e-10:
            continue
        
        assignment = optimize_assignment(hubs_list, n, W, D, O, I, chi, alpha, delta, capacities, max_iter=20)
        loads = get_hub_loads(assignment, hubs_list, O, n)
        feasible = all(loads.get(k, 0) <= capacities[k] + 1e-10 for k in hubs_list)
        if not feasible:
            continue
        
        obj = compute_objective_fast(hubs_list, assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
        log_solution(hubs_list, assignment, obj)
    
    # Phase 3: MIP for small instances
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if n <= 30 and remaining > 10 and best_obj < float('inf'):
        try:
            mip_hubs, mip_assignment, mip_obj = solve_mip(
                n, W, D, O, I, setup_costs, capacities, chi, alpha, delta, total_flow,
                remaining - 2
            )
            if mip_hubs is not None and mip_obj is not None:
                # Verify feasibility
                loads = get_hub_loads(mip_assignment, mip_hubs, O, n)
                feasible = all(loads.get(k, 0) <= capacities[k] + 1e-10 for k in mip_hubs)
                if feasible:
                    # Recompute objective to ensure consistency
                    actual_obj = compute_objective_fast(mip_hubs, mip_assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
                    log_solution(mip_hubs, mip_assignment, actual_obj)
        except Exception as e:
            pass
    
    # Phase 4: If no solution found yet, create a basic feasible solution
    if best_obj == float('inf'):
        # Open all nodes as hubs (always feasible if capacities allow)
        hubs_list = list(range(n))
        assignment = np.arange(n)
        obj = compute_objective_fast(hubs_list, assignment, n, W, D, O, I, setup_costs, chi, alpha, delta)
        log_solution(hubs_list, assignment, obj)
    
    # Write final solution
    solution = {
        "objective_value": float(best_obj),
        "hubs": sorted([int(h) for h in best_hubs]),
        "assignment": [int(a) for a in best_assignment]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()