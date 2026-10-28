import argparse
import json
import time
import numpy as np
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    chi = data['cost_parameters']['collection_cost_chi']
    alpha = data['cost_parameters']['transfer_cost_alpha']
    delta = data['cost_parameters']['distribution_cost_delta']
    
    flow = np.array(data['flow_matrix'])
    dist = np.array(data['distance_matrix'])
    outgoing = np.array(data['outgoing_flow'])
    setup_costs = np.array(data['setup_costs'])
    capacities = np.array(data['capacities'])
    total_flow_D = data['total_flow_D']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Precompute: for each node i, the cost of assigning i to hub k (collection component)
    # Collection cost for node i assigned to hub k: chi * dist[i][k] * outgoing[i]
    # But we also need distribution cost: delta * dist[k][j] * (incoming flow to j through hub k)
    # Actually, the routing cost for flow from i to j via hubs k (i's hub) and m (j's hub) is:
    # flow[i][j] * (chi * dist[i][k] + alpha * dist[k][m] + delta * dist[m][j])
    
    # For a given assignment (each node to a hub), the total routing cost is:
    # sum_i sum_j flow[i][j] * (chi * dist[i][hub[i]] + alpha * dist[hub[i]][hub[j]] + delta * dist[hub[j]][j])
    
    # = sum_i chi * dist[i][hub[i]] * sum_j flow[i][j]   (= outgoing[i])
    #   + sum_j delta * dist[hub[j]][j] * sum_i flow[i][j]  (= incoming[j])
    #   + sum_i sum_j flow[i][j] * alpha * dist[hub[i]][hub[j]]
    
    incoming = flow.sum(axis=0)  # incoming[j] = sum_i flow[i][j]
    
    # Precompute collection cost: for assigning node i to hub k
    collection_cost = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            collection_cost[i][k] = chi * dist[i][k] * outgoing[i]
    
    # Distribution cost: for assigning node j to hub m
    distribution_cost = np.zeros((n, n))
    for j in range(n):
        for m in range(n):
            distribution_cost[j][m] = delta * dist[m][j] * incoming[j]
    
    # Node cost of assigning i to hub k (collection + distribution)
    node_hub_cost = collection_cost + distribution_cost  # shape (n, n)
    
    # Inter-hub transfer: sum_i sum_j flow[i][j] * alpha * dist[hub[i]][hub[j]]
    # = sum over hub pairs (k, m): alpha * dist[k][m] * (sum of flow[i][j] for i assigned to k, j assigned to m)
    # This is alpha * dist[k][m] * total_flow_from_k_to_m
    # where total_flow_from_k_to_m = sum_{i in cluster_k} sum_{j in cluster_m} flow[i][j]
    
    # For a given assignment, let's define cluster_flow[k][m] = sum of flow[i][j] for i->k, j->m
    # Transfer cost = sum_{k,m} alpha * dist[k][m] * cluster_flow[k][m]
    
    # Precompute flow aggregated by (origin_hub, dest_hub) for any assignment
    # This depends on the assignment, so we compute it on the fly
    
    def compute_objective(hubs_set, assignment):
        """Compute total objective for a given set of hubs and assignment."""
        # Setup costs
        total = sum(setup_costs[h] for h in hubs_set)
        
        # Node-hub costs (collection + distribution)
        for i in range(n):
            total += node_hub_cost[i][assignment[i]]
        
        # Inter-hub transfer costs
        # Build cluster flow matrix
        hub_list = sorted(hubs_set)
        hub_idx = {h: idx for idx, h in enumerate(hub_list)}
        nh = len(hub_list)
        cluster_flow = np.zeros((nh, nh))
        for i in range(n):
            ki = hub_idx[assignment[i]]
            for j in range(n):
                mj = hub_idx[assignment[j]]
                cluster_flow[ki][mj] += flow[i][j]
        
        for ki, k in enumerate(hub_list):
            for mi, m in enumerate(hub_list):
                total += alpha * dist[k][m] * cluster_flow[ki][mi]
        
        return total
    
    def compute_objective_fast(assignment):
        """Faster objective computation."""
        total = 0.0
        
        hubs_set = set(assignment)
        total += sum(setup_costs[h] for h in hubs_set)
        
        for i in range(n):
            total += node_hub_cost[i][assignment[i]]
        
        # Transfer cost
        for i in range(n):
            hi = assignment[i]
            for j in range(n):
                hj = assignment[j]
                total += alpha * dist[hi][hj] * flow[i][j]
        
        return total
    
    def compute_objective_vectorized(assignment):
        """Vectorized objective computation."""
        hubs_set = set(assignment)
        total = sum(setup_costs[h] for h in hubs_set)
        
        # Node-hub costs
        total += sum(node_hub_cost[i][assignment[i]] for i in range(n))
        
        # Transfer cost - vectorized
        hub_i = np.array([assignment[i] for i in range(n)])
        hub_j = np.array([assignment[j] for j in range(n)])
        # dist[hub_i[i], hub_j[j]] for all i,j
        hub_dist = dist[np.ix_(hub_i, hub_j)]  # n x n
        total += alpha * np.sum(flow * hub_dist)
        
        return total
    
    # Strategy: Use Gurobi for small instances, heuristic for larger ones
    
    best_obj = float('inf')
    best_hubs = None
    best_assignment = None
    
    def try_solution(hubs_list, assignment):
        nonlocal best_obj, best_hubs, best_assignment
        obj = compute_objective_vectorized(assignment)
        if obj < best_obj:
            best_obj = obj
            best_hubs = list(hubs_list)
            best_assignment = list(assignment)
            if logger:
                logger.log(obj)
        return obj
    
    # Greedy heuristic to find a good initial solution
    def greedy_assignment(hubs_set):
        """Assign each node to the best hub, respecting capacity."""
        hubs_list = sorted(hubs_set)
        nh = len(hubs_list)
        
        # For each node, compute the "individual" cost of assigning to each hub
        # (ignoring transfer costs for now, or using a simple approximation)
        # Simple: assign based on node_hub_cost + approximate transfer
        
        assignment = [-1] * n
        remaining_cap = {h: capacities[h] for h in hubs_list}
        
        # Sort nodes by outgoing flow (descending) to assign high-flow nodes first
        node_order = sorted(range(n), key=lambda i: outgoing[i], reverse=True)
        
        for i in node_order:
            best_cost = float('inf')
            best_hub = -1
            for h in hubs_list:
                if remaining_cap[h] >= outgoing[i]:
                    cost = node_hub_cost[i][h]
                    if cost < best_cost:
                        best_cost = cost
                        best_hub = h
            if best_hub == -1:
                # Try to find any hub with remaining capacity
                for h in hubs_list:
                    if remaining_cap[h] >= outgoing[i]:
                        best_hub = h
                        break
                if best_hub == -1:
                    return None  # Infeasible
            assignment[i] = best_hub
            remaining_cap[best_hub] -= outgoing[i]
        
        return assignment
    
    def greedy_assignment_with_transfer(hubs_set, current_assignment=None):
        """Better assignment considering transfer costs."""
        hubs_list = sorted(hubs_set)
        
        # If we have a current assignment, compute transfer cost contribution
        # For node i assigned to hub k, total cost includes:
        # node_hub_cost[i][k] + sum_j flow[i][j] * alpha * dist[k][hub[j]] + sum_j flow[j][i] * alpha * dist[hub[j]][k]
        
        # Iterative refinement
        if current_assignment is None:
            assignment = greedy_assignment(hubs_set)
            if assignment is None:
                return None
        else:
            assignment = list(current_assignment)
        
        # Iterative improvement
        improved = True
        max_iter = 50
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            hub_loads = {h: 0.0 for h in hubs_list}
            for i in range(n):
                hub_loads[assignment[i]] += outgoing[i]
            
            for i in range(n):
                current_hub = assignment[i]
                best_hub = current_hub
                
                # Compute current cost for node i
                best_cost = node_hub_cost[i][current_hub]
                for j in range(n):
                    if i != j:
                        best_cost += alpha * dist[current_hub][assignment[j]] * flow[i][j]
                        best_cost += alpha * dist[assignment[j]][current_hub] * flow[j][i]
                
                for h in hubs_list:
                    if h == current_hub:
                        continue
                    # Check capacity
                    if hub_loads[h] + outgoing[i] > capacities[h]:
                        continue
                    
                    cost = node_hub_cost[i][h]
                    for j in range(n):
                        if i != j:
                            cost += alpha * dist[h][assignment[j]] * flow[i][j]
                            cost += alpha * dist[assignment[j]][h] * flow[j][i]
                    
                    if cost < best_cost - 1e-6:
                        best_cost = cost
                        best_hub = h
                
                if best_hub != current_hub:
                    hub_loads[current_hub] -= outgoing[i]
                    hub_loads[best_hub] += outgoing[i]
                    assignment[i] = best_hub
                    improved = True
            
            if time.time() - start_time > time_limit * 0.9:
                break
        
        return assignment
    
    # Check feasibility of a hub set
    def is_feasible_hub_set(hubs_set):
        total_cap = sum(capacities[h] for h in hubs_set)
        if total_cap < total_flow_D - 1e-6:
            return False
        return True
    
    # For small n, try Gurobi MIP
    use_gurobi = (n <= 50)
    
    if use_gurobi:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5:
                use_gurobi = False
            else:
                m = gp.Model("hub_location")
                m.setParam('TimeLimit', remaining_time * 0.85)
                m.setParam('Threads', 1)
                m.setParam('MIPGap', 1e-6)
                
                # Variables
                # z[k] = 1 if hub k is open
                z = m.addVars(n, vtype=GRB.BINARY, name="z")
                # x[i][k] = 1 if node i is assigned to hub k
                x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
                
                # Constraints
                # Each node assigned to exactly one hub
                for i in range(n):
                    m.addConstr(gp.quicksum(x[i, k] for k in range(n)) == 1)
                
                # Can only assign to open hubs
                for i in range(n):
                    for k in range(n):
                        m.addConstr(x[i, k] <= z[k])
                
                # Hub assigned to itself
                for k in range(n):
                    m.addConstr(x[k, k] >= z[k])
                
                # Capacity constraints
                for k in range(n):
                    m.addConstr(gp.quicksum(outgoing[i] * x[i, k] for i in range(n)) <= capacities[k] * z[k])
                
                # Total capacity
                m.addConstr(gp.quicksum(capacities[k] * z[k] for k in range(n)) >= total_flow_D)
                
                # Objective
                # Setup costs
                obj = gp.quicksum(setup_costs[k] * z[k] for k in range(n))
                
                # Node-hub costs (collection + distribution)
                for i in range(n):
                    for k in range(n):
                        obj += node_hub_cost[i][k] * x[i, k]
                
                # Transfer costs: sum_i sum_j flow[i][j] * alpha * dist[hub_i][hub_j]
                # = sum_i sum_j sum_k sum_m flow[i][j] * alpha * dist[k][m] * x[i,k] * x[j,m]
                # This is quadratic. Linearize using y[i,j,k,m] = x[i,k] * x[j,m]
                # But that's O(n^4) variables - too many for n > ~20
                
                # Alternative linearization using aggregated flow
                # Let F_km = sum_{i assigned to k, j assigned to m} flow[i][j]
                # Transfer cost = sum_{k,m} alpha * dist[k][m] * F_km
                
                # F_km = sum_i sum_j flow[i][j] * x[i,k] * x[j,m]
                # = sum_i x[i,k] * (sum_j flow[i][j] * x[j,m])
                
                # Define w[i,m] = sum_j flow[i][j] * x[j,m] = flow from i to cluster m
                # Then F_km = sum_i x[i,k] * w[i,m]
                
                # But this is still nonlinear. Let's use the y variables approach but more carefully.
                
                # For n <= 25, we can use O(n^4) approach
                # For 25 < n <= 50, we need a smarter formulation
                
                if n <= 30:
                    # Use y[i,j,k,m] but only for pairs with non-zero flow
                    # Actually, let's reformulate differently
                    # 
                    # sum_i sum_j flow[i][j] * alpha * sum_k sum_m dist[k][m] * x[i,k] * x[j,m]
                    # = alpha * sum_k sum_m dist[k][m] * (sum_i flow_out_to_m[i] * x[i,k])
                    # where flow_out_to_m[i] = sum_j flow[i][j] * x[j,m]
                    
                    # Let's define: y[i,k,m] = x[i,k] * x[j,m] ... no
                    
                    # Alternative: define flow variables
                    # Let q[k,m] = total flow from hub k to hub m
                    # q[k,m] = sum_i sum_j flow[i][j] * x[i,k] * x[j,m]
                    
                    # We can linearize: define for each node i:
                    # contribution of i to flow from k to m = x[i,k] * sum_j flow[i][j] * x[j,m]
                    
                    # Let's define r[i,m] = sum_j flow[i][j] * x[j,m] (flow from i destined to cluster m)
                    # r[i,m] is linear
                    # q[k,m] = sum_i x[i,k] * r[i,m] - this is bilinear
                    
                    # Instead, use standard linearization with:
                    # For each (i,j), the transfer cost is flow[i][j] * alpha * sum_k sum_m dist[k][m] * x[i,k] * x[j,m]
                    # = flow[i][j] * alpha * (sum_k dist[k, hub_j] * x[i,k])  -- but hub_j is also a variable
                    
                    # Let's try a different formulation with fewer variables
                    # Define t[i,m] = x[i,k] for node i's hub choice, 
                    # and the transfer cost for flow from i is:
                    # sum_j flow[i][j] * alpha * dist[hub_i][hub_j]
                    # = alpha * sum_k x[i,k] * sum_j flow[i][j] * sum_m dist[k][m] * x[j,m]
                    # = alpha * sum_k x[i,k] * sum_m dist[k][m] * (sum_j flow[i][j] * x[j,m])
                    
                    # Define u[i,m] = sum_j flow[i][j] * x[j,m]  -- this is linear, O(n^2)
                    # Transfer cost = alpha * sum_i sum_k x[i,k] * sum_m dist[k][m] * u[i,m]
                    # = alpha * sum_i sum_k sum_m dist[k][m] * x[i,k] * u[i,m]
                    
                    # Need to linearize x[i,k] * u[i,m]. Since u[i,m] is continuous and bounded,
                    # we need McCormick or define v[i,k,m] = x[i,k] * u[i,m]
                    # Since x[i,k] is binary: v[i,k,m] = u[i,m] if x[i,k]=1, else 0
                    # v[i,k,m] <= M * x[i,k]
                    # v[i,k,m] <= u[i,m]
                    # v[i,k,m] >= u[i,m] - M*(1-x[i,k])
                    # v[i,k,m] >= 0
                    
                    # This adds O(n^3) variables and O(n^3) constraints - feasible for n<=30
                    
                    # Actually, let's use a simpler approach.
                    # We can precompute for each pair (i, k, m):
                    # cost_transfer_ikm = alpha * sum_j flow[i][j] * dist[k][m] for assigning i->k and j->m
                    # But j varies...
                    
                    # Let me try yet another approach: 
                    # For each pair of hubs (k,m), the flow between them is:
                    # q[k,m] = sum_i sum_j flow[i][j] * x[i,k] * x[j,m]
                    # 
                    # We can write this as matrix product of assignment vectors:
                    # Let a_k = vector of x[i,k] for all i
                    # q[k,m] = a_k^T * flow * a_m
                    
                    # For the MIP, let's use the standard CSAHLP formulation.
                    # The standard approach uses variables y[i,k,j,m] = fraction of flow from i to j
                    # routed via hubs k and m. But with the single-assignment constraint,
                    # y[i,k,j,m] = x[i,k] * x[j,m]
                    
                    # A common efficient formulation (Ernst & Krishnamoorthy):
                    # Uses aggregate flow variables.
                    
                    # Let me use the formulation from the literature:
                    # Variables: z[k] (hub open), x[i,k] (assignment), 
                    # and f[k,m,i] = flow originating at i that goes from hub k to hub m
                    # f[k,m,i] = x[i,k] * sum_j flow[i][j] * x[j,m]
                    
                    # Actually for small n, let's just add the quadratic terms via linearization.
                    # For each pair (i,j) with flow[i][j] > 0:
                    #   transfer_cost += flow[i][j] * alpha * sum_k sum_m dist[k][m] * x[i,k] * x[j,m]
                    
                    # With the single assignment constraint, for a given (i,j):
                    # sum_k sum_m dist[k][m] * x[i,k] * x[j,m] 
                    # = (sum_k dist[k, hub_j] * x[i,k]) where hub_j is j's hub
                    
                    # Actually, let me use a clean approach. Define:
                    # For each i, j: the product x[i,k]*x[j,m] can be linearized with 
                    # a single variable p[i,j,k,m] but that's O(n^4).
                    
                    # For n<=30, O(n^4) is 810,000 which might be manageable.
                    # But actually we can be smarter.
                    
                    # Note that the transfer cost depends only on hub assignments, not individual (i,j) pairs:
                    # transfer = alpha * sum_k sum_m dist[k][m] * Q[k][m]
                    # where Q[k][m] = sum_i sum_j flow[i][j] * x[i,k] * x[j,m]
                    #                = (sum_i outflow_to_m[i] * x[i,k])
                    # where outflow_to_m[i] = sum_j flow[i][j] * x[j,m]
                    
                    # Q[k][m] = sum_i x[i,k] * (sum_j flow[i][j] * x[j,m])
                    # If we define: g[i][m] = sum_j flow[i][j] * x[j,m]  -- linear in x[j,m]
                    # Then Q[k][m] = sum_i x[i,k] * g[i][m]  -- product of binary and continuous
                    
                    # We can linearize with v[i,k,m] = x[i,k] * g[i][m]:
                    # v[i,k,m] <= g[i][m]
                    # v[i,k,m] <= M * x[i,k]
                    # v[i,k,m] >= g[i][m] - M*(1-x[i,k])
                    # v[i,k,m] >= 0
                    
                    # Q[k][m] = sum_i v[i,k,m]
                    # transfer = alpha * sum_k sum_m dist[k][m] * sum_i v[i,k,m]
                    #          = alpha * sum_i sum_k sum_m dist[k][m] * v[i,k,m]
                    
                    # Number of v variables: O(n^3), constraints: O(n^3)
                    # For n=30: 27000 variables, 4*27000 = 108000 constraints - feasible
                    
                    # But we can simplify further. Since x[i,k] is binary and sum_k x[i,k] = 1:
                    # When x[i,k]=1, g[i][m] = sum_j flow[i][j] * x[j,m], and v[i,k,m] = g[i][m]
                    # When x[i,k]=0, v[i,k,m] = 0
                    
                    # So sum_k v[i,k,m] = g[i][m] (since exactly one k has x[i,k]=1)
                    # And the transfer cost becomes:
                    # alpha * sum_i sum_k sum_m dist[k][m] * v[i,k,m]
                    
                    # This is still O(n^3) auxiliary variables. Let me try it.
                    
                    # Actually, let me think of an even simpler formulation.
                    # Define for each (k, m): q[k,m] as a continuous variable for aggregate flow.
                    # We need: q[k,m] >= sum_i sum_j flow[i][j] * x[i,k] * x[j,m]
                    # But that's an equality for minimization since dist[k][m] >= 0.
                    # Actually with alpha > 0 and dist >= 0, minimization will push q down.
                    # But equality is needed.
                    
                    # It's hard to write q[k,m] = sum_i sum_j flow[i][j] * x[i,k] * x[j,m] as linear.
                    
                    # Let me just use the v[i,k,m] formulation.
                    
                    # Precompute M[i] = sum_j flow[i][j] = outgoing[i]
                    # g[i][m] = sum_j flow[i][j] * x[j,m] <= outgoing[i]
                    
                    g = m.addVars(n, n, lb=0, name="g")
                    for i in range(n):
                        for mm in range(n):
                            m.addConstr(g[i, mm] == gp.quicksum(flow[i][j] * x[j, mm] for j in range(n)))
                    
                    v = m.addVars(n, n, n, lb=0, name="v")
                    M_val = float(max(outgoing))
                    
                    for i in range(n):
                        for k in range(n):
                            for mm in range(n):
                                m.addConstr(v[i, k, mm] <= g[i, mm])
                                m.addConstr(v[i, k, mm] <= outgoing[i] * x[i, k])
                                m.addConstr(v[i, k, mm] >= g[i, mm] - outgoing[i] * (1 - x[i, k]))
                    
                    # Transfer cost
                    transfer_cost = gp.quicksum(
                        alpha * dist[k][mm] * v[i, k, mm]
                        for i in range(n) for k in range(n) for mm in range(n)
                    )
                    obj += transfer_cost
                    
                else:
                    # For larger n (30-50), use a different approach
                    # Use the Ernst-Krishnamoorthy formulation with flow variables
                    
                    # Define Y[i][k][l] as the total flow from node i that is routed on inter-hub link (k,l)
                    # But this is still O(n^3)
                    
                    # Let's try: for each i, define the hub assignment cost including an approximation
                    # of transfer costs, then solve with fewer constraints
                    
                    # Actually, let me just use the same v formulation but be smarter about it
                    # Only add variables where flow[i][j] > 0 or where dist[k][m] > 0
                    
                    g = m.addVars(n, n, lb=0, name="g")
                    for i in range(n):
                        for mm in range(n):
                            nonzero_js = [j for j in range(n) if flow[i][j] > 0]
                            if nonzero_js:
                                m.addConstr(g[i, mm] == gp.quicksum(flow[i][j] * x[j, mm] for j in nonzero_js))
                            else:
                                m.addConstr(g[i, mm] == 0)
                    
                    v = m.addVars(n, n, n, lb=0, name="v")
                    
                    for i in range(n):
                        oi = outgoing[i]
                        if oi < 1e-10:
                            for k in range(n):
                                for mm in range(n):
                                    m.addConstr(v[i, k, mm] == 0)
                            continue
                        for k in range(n):
                            for mm in range(n):
                                m.addConstr(v[i, k, mm] <= g[i, mm])
                                m.addConstr(v[i, k, mm] <= oi * x[i, k])
                                m.addConstr(v[i, k, mm] >= g[i, mm] - oi * (1 - x[i, k]))
                    
                    transfer_cost = gp.quicksum(
                        alpha * dist[k][mm] * v[i, k, mm]
                        for i in range(n) for k in range(n) for mm in range(n)
                        if outgoing[i] > 0
                    )
                    obj += transfer_cost
                
                m.setObjective(obj, GRB.MINIMIZE)
                
                # Provide initial solution from heuristic
                # First run a quick heuristic
                def quick_heuristic():
                    # Try opening all nodes as hubs
                    all_hubs = set(range(n))
                    if is_feasible_hub_set(all_hubs):
                        asgn = greedy_assignment(all_hubs)
                        if asgn is not None:
                            return all_hubs, asgn
                    
                    # Try nodes with largest capacities
                    sorted_by_cap = sorted(range(n), key=lambda k: capacities[k], reverse=True)
                    hubs = set()
                    total_cap = 0
                    for k in sorted_by_cap:
                        hubs.add(k)
                        total_cap += capacities[k]
                        if total_cap >= total_flow_D:
                            break
                    
                    if is_feasible_hub_set(hubs):
                        asgn = greedy_assignment(hubs)
                        if asgn is not None:
                            return hubs, asgn
                    
                    return None, None
                
                qh_hubs, qh_asgn = quick_heuristic()
                if qh_hubs is not None and qh_asgn is not None:
                    # Set as MIP start
                    for k in range(n):
                        z[k].Start = 1 if k in qh_hubs else 0
                    for i in range(n):
                        for k in range(n):
                            x[i, k].Start = 1 if qh_asgn[i] == k else 0
                    
                    # Record heuristic solution
                    try_solution(qh_hubs, qh_asgn)
                
                # Solve
                m.optimize()
                
                if m.SolCount > 0:
                    # Extract solution
                    hubs_sol = [k for k in range(n) if z[k].X > 0.5]
                    assignment_sol = [0] * n
                    for i in range(n):
                        for k in range(n):
                            if x[i, k].X > 0.5:
                                assignment_sol[i] = k
                                break
                    
                    try_solution(set(hubs_sol), assignment_sol)
                
        except Exception as e:
            print(f"Gurobi failed: {e}")
            use_gurobi = False
    
    if not use_gurobi or best_obj == float('inf'):
        # Pure heuristic approach
        
        # Sort potential hubs by setup_cost / capacity ratio
        hub_candidates = sorted(range(n), key=lambda k: setup_costs[k] / max(capacities[k], 1e-10))
        
        # Find minimum number of hubs needed
        def find_min_hubs():
            sorted_by_cap = sorted(range(n), key=lambda k: capacities[k], reverse=True)
            total_cap = 0
            count = 0
            for k in sorted_by_cap:
                total_cap += capacities[k]
                count += 1
                if total_cap >= total_flow_D:
                    return count
            return n
        
        min_hubs = find_min_hubs()
        
        # Try different numbers of hubs
        def evaluate_hub_set(hubs_set):
            if not is_feasible_hub_set(hubs_set):
                return float('inf'), None
            
            asgn = greedy_assignment(hubs_set)
            if asgn is None:
                return float('inf'), None
            
            # Refine
            if time.time() - start_time < time_limit * 0.8:
                asgn = greedy_assignment_with_transfer(hubs_set, asgn)
                if asgn is None:
                    return float('inf'), None
            
            obj = compute_objective_vectorized(asgn)
            return obj, asgn
        
        # Start with a greedy construction
        # Open hubs greedily
        best_local_hubs = set()
        remaining_flow = total_flow_D
        
        for k in hub_candidates:
            if remaining_flow <= 0:
                break
            best_local_hubs.add(k)
            remaining_flow -= capacities[k]
        
        obj, asgn = evaluate_hub_set(best_local_hubs)
        if asgn is not None:
            try_solution(best_local_hubs, asgn)
        
        # Try adding/removing hubs
        for iteration in range(100):
            if time.time() - start_time > time_limit * 0.7:
                break
            
            improved = False
            
            # Try removing each hub
            if best_hubs is not None:
                current_hubs = set(best_hubs)
                for h in list(current_hubs):
                    if len(current_hubs) <= min_hubs:
                        break
                    new_hubs = current_hubs - {h}
                    if is_feasible_hub_set(new_hubs):
                        obj, asgn = evaluate_hub_set(new_hubs)
                        if asgn is not None and obj < best_obj:
                            try_solution(new_hubs, asgn)
                            current_hubs = set(best_hubs)
                            improved = True
            
            # Try swapping hubs
            if best_hubs is not None:
                current_hubs = set(best_hubs)
                for h_out in list(current_hubs):
                    if time.time() - start_time > time_limit * 0.7:
                        break
                    for h_in in range(n):
                        if h_in in current_hubs:
                            continue
                        new_hubs = (current_hubs - {h_out}) | {h_in}
                        if is_feasible_hub_set(new_hubs):
                            obj, asgn = evaluate_hub_set(new_hubs)
                            if asgn is not None and obj < best_obj:
                                try_solution(new_hubs, asgn)
                                current_hubs = set(best_hubs)
                                improved = True
                                break
            
            # Try adding a hub
            if best_hubs is not None:
                current_hubs = set(best_hubs)
                for h in range(n):
                    if h in current_hubs:
                        continue
                    if time.time() - start_time > time_limit * 0.7:
                        break
                    new_hubs = current_hubs | {h}
                    obj, asgn = evaluate_hub_set(new_hubs)
                    if asgn is not None and obj < best_obj:
                        try_solution(new_hubs, asgn)
                        improved = True
                        break
            
            if not improved:
                break
        
        # Final refinement of best solution
        if best_hubs is not None and time.time() - start_time < time_limit * 0.9:
            asgn = greedy_assignment_with_transfer(set(best_hubs), best_assignment)
            if asgn is not None:
                try_solution(set(best_hubs), asgn)
    
    # Also try to refine the Gurobi solution with local search if time remains
    if best_hubs is not None and best_assignment is not None and time.time() - start_time < time_limit * 0.95:
        # Local search: try reassigning nodes
        current_assignment = list(best_assignment)
        current_hubs = set(best_hubs)
        
        improved = True
        while improved and time.time() - start_time < time_limit * 0.95:
            improved = False
            hub_loads = {}
            for h in current_hubs:
                hub_loads[h] = sum(outgoing[i] for i in range(n) if current_assignment[i] == h)
            
            for i in range(n):
                if time.time() - start_time > time_limit * 0.95:
                    break
                
                current_hub = current_assignment[i]
                current_cost = node_hub_cost[i][current_hub]
                for j in range(n):
                    if i != j:
                        current_cost += alpha * dist[current_hub][current_assignment[j]] * flow[i][j]
                        current_cost += alpha * dist[current_assignment[j]][current_hub] * flow[j][i]
                
                best_hub_local = current_hub
                best_cost_local = current_cost
                
                for h in current_hubs:
                    if h == current_hub:
                        continue
                    if hub_loads[h] + outgoing[i] > capacities[h]:
                        continue
                    
                    cost = node_hub_cost[i][h]
                    for j in range(n):
                        if i != j:
                            cost += alpha * dist[h][current_assignment[j]] * flow[i][j]
                            cost += alpha * dist[current_assignment[j]][h] * flow[j][i]
                    
                    if cost < best_cost_local - 1e-6:
                        best_cost_local = cost
                        best_hub_local = h
                
                if best_hub_local != current_hub:
                    hub_loads[current_hub] -= outgoing[i]
                    hub_loads[best_hub_local] += outgoing[i]
                    current_assignment[i] = best_hub_local
                    improved = True
            
            if improved:
                try_solution(current_hubs, current_assignment)
    
    # Write solution
    solution = {
        "objective_value": best_obj,
        "hubs": sorted(best_hubs) if best_hubs else [],
        "assignment": best_assignment if best_assignment else []
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