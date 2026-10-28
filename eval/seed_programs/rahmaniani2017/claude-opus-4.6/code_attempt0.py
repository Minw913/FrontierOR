import argparse
import json
import time
import math
import numpy as np
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_nodes = instance['num_nodes']
    num_commodities = instance['num_commodities']
    params = instance['parameters']
    tau = params['tau']
    chi = params['chi']
    delta = params['delta']
    
    nodes = instance['nodes']
    distances = instance['distances']
    commodities = instance['commodities']
    
    # Convert distances to numpy array for faster access
    dist = np.array(distances, dtype=np.float64)
    
    # Precompute flow array
    flows = np.array([c['flow'] for c in commodities], dtype=np.float64)
    origins = np.array([c['origin'] for c in commodities], dtype=np.int32)
    destinations = np.array([c['destination'] for c in commodities], dtype=np.int32)
    
    fixed_costs = np.array([n['fixed_cost'] for n in nodes], dtype=np.float64)
    
    # For each commodity c and hub pair (h1, h2), the routing cost is:
    # flow[c] * (chi * dist[o][h1] + tau * dist[h1][h2] + delta * dist[h2][d])
    # Due to symmetry (chi == delta, symmetric distances), cost(h1,h2) == cost(h2,h1)
    # So we can consider unordered pairs.
    
    # For single hub h: flow[c] * (chi * dist[o][h] + delta * dist[h][d])
    # For hub pair (h1, h2): min over ordering
    
    # Precompute for each commodity, the cost of using each single hub
    # cost_single[c][h] = flow[c] * (chi * dist[origin[c]][h] + delta * dist[h][dest[c]])
    
    # For small instances, we can use Gurobi MIP
    # For larger instances, we need heuristics
    
    n = num_nodes
    m = num_commodities
    
    # Precompute cost_single[c, h] for all commodities and hubs
    # cost_single shape: (m, n)
    origin_dist = dist[origins]  # shape (m, n) - dist from each origin to each node
    dest_dist = dist[destinations]  # shape (m, n) - dist from each node to each destination
    
    # For single hub h: flow * (chi * dist[o,h] + delta * dist[h,d])
    cost_single = flows[:, None] * (chi * origin_dist + delta * dest_dist)  # (m, n)
    
    # For hub pair (h1, h2) with h1 != h2, best cost is:
    # flow * min(chi*dist[o,h1] + tau*dist[h1,h2] + delta*dist[h2,d],
    #            chi*dist[o,h2] + tau*dist[h1,h2] + delta*dist[h2,d]... wait no)
    # Actually: flow * (chi*dist[o,h1] + tau*dist[h1,h2] + delta*dist[h2,d])
    # or flow * (chi*dist[o,h2] + tau*dist[h2,h1] + delta*dist[h1,d])
    # Due to symmetry these are:
    # flow * (chi*dist[o,h1] + tau*dist[h1,h2] + delta*dist[h2,d])
    # flow * (chi*dist[o,h2] + tau*dist[h1,h2] + delta*dist[h1,d])
    # We take the min.
    
    def compute_commodity_cost(c, h1, h2):
        """Compute the minimum routing cost for commodity c through hubs h1 and h2."""
        if h1 == h2:
            return cost_single[c, h1]
        f = flows[c]
        o = origins[c]
        d = destinations[c]
        c1 = f * (chi * dist[o][h1] + tau * dist[h1][h2] + delta * dist[h2][d])
        c2 = f * (chi * dist[o][h2] + tau * dist[h2][h1] + delta * dist[h1][d])
        return min(c1, c2)
    
    def evaluate_solution(open_hubs_set):
        """Given a set of open hubs, find optimal assignment and total cost."""
        hubs = list(open_hubs_set)
        nh = len(hubs)
        if nh == 0:
            return float('inf'), None, None
        
        total_transport = 0.0
        assignments = []
        
        # For each commodity, find the best hub or hub pair
        # Precompute cost_single for these hubs
        hub_arr = np.array(hubs)
        
        for c in range(m):
            best_cost = float('inf')
            best_h1 = -1
            best_h2 = -1
            
            o = origins[c]
            d = destinations[c]
            f = flows[c]
            
            # Single hub costs
            for h in hubs:
                cc = cost_single[c, h]
                if cc < best_cost:
                    best_cost = cc
                    best_h1 = h
                    best_h2 = h
            
            # Hub pair costs
            if nh > 1:
                for i in range(nh):
                    h1 = hubs[i]
                    collection1 = chi * dist[o][h1]
                    distribution1 = delta * dist[h1][d]
                    for j in range(i + 1, nh):
                        h2 = hubs[j]
                        transfer = tau * dist[h1][h2]
                        # Route 1: o -> h1 -> h2 -> d
                        c1 = f * (collection1 + transfer + delta * dist[h2][d])
                        # Route 2: o -> h2 -> h1 -> d
                        c2 = f * (chi * dist[o][h2] + transfer + distribution1)
                        cc = min(c1, c2)
                        if cc < best_cost:
                            best_cost = cc
                            if c1 <= c2:
                                best_h1 = h1
                                best_h2 = h2
                            else:
                                best_h1 = h2
                                best_h2 = h1
            
            total_transport += best_cost
            assignments.append((best_h1, best_h2, best_cost))
        
        total_fixed = sum(fixed_costs[h] for h in hubs)
        return total_fixed + total_transport, assignments, hubs
    
    # Strategy: For small n, use Gurobi MIP. For larger n, use greedy/local search.
    
    best_obj = float('inf')
    best_assignments = None
    best_hubs = None
    
    def update_best(obj, assignments, hubs):
        nonlocal best_obj, best_assignments, best_hubs
        if obj < best_obj:
            best_obj = obj
            best_assignments = assignments
            best_hubs = hubs
            if logger:
                logger.log(obj)
    
    # Try Gurobi for small/medium instances
    use_gurobi = (n <= 50 and m <= 2000) or (n <= 30)
    
    remaining_time = time_limit - (time.time() - start_time)
    
    if use_gurobi and remaining_time > 5:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model("hub_location")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', max(1, remaining_time - 2))
            model.setParam('Threads', 1)
            
            # Decision variables
            # y[h] = 1 if hub h is open
            y = model.addVars(n, vtype=GRB.BINARY, name="y")
            
            # For each commodity c, x[c, h1, h2] = 1 if commodity c is routed through (h1, h2)
            # Due to symmetry, we only need unordered pairs plus single hubs
            # But to keep it simpler with Gurobi, let's use a different formulation
            
            # Actually, let's use the classic formulation:
            # For each commodity c, and each unordered hub pair {h1, h2} (including h1=h2),
            # z[c, h1, h2] = 1 if commodity c uses this pair
            
            # For efficiency, precompute cost for each commodity and hub pair
            # This can be huge: m * n * (n+1) / 2
            
            total_pairs = n * (n + 1) // 2
            
            if m * total_pairs > 500000:
                # Too many variables, use a different approach
                # Use the allocation formulation with hub pairs
                # Or decompose
                
                # Alternative: use the classic O(m*n^2) formulation
                # x[c,k,l] where k is first hub, l is second hub
                # But with symmetry we can reduce
                
                # Let's try a compact formulation
                # For each commodity c, define:
                # a[c,h] = fraction of commodity c collected at hub h (from origin)
                # b[c,h] = fraction of commodity c distributed from hub h (to destination)
                # sum_h a[c,h] = 1, sum_h b[c,h] = 1
                # a[c,h] <= y[h], b[c,h] <= y[h]
                # Transfer: for each pair, flow * tau * dist[h1][h2] * a[c,h1] * b[c,h2]
                # This is quadratic... not directly useful for MIP
                
                # Standard linearization: z[c,h1,h2] = a[c,h1] * b[c,h2]
                # This gives O(m*n^2) variables
                
                # Let's just go with the standard formulation
                # z[c, k, l] for all c, k, l where k <= l (unordered pairs)
                
                # For large instances, skip Gurobi
                if m * n * n > 2000000:
                    raise Exception("Too large for MIP")
                
            # Standard formulation with ordered pairs but using symmetry
            # z[c, k, l] = 1 if commodity c is routed o->k->l->d
            # Due to symmetry, cost(k,l) = cost(l,k), so we use unordered pairs
            
            # Precompute costs
            # For unordered pair (k, l) with k <= l:
            # cost[c, k, l] = flow[c] * min(chi*d[o,k] + tau*d[k,l] + delta*d[l,d],
            #                                chi*d[o,l] + tau*d[l,k] + delta*d[k,d])
            # For k = l: cost[c, k, k] = flow[c] * (chi*d[o,k] + delta*d[k,d])
            
            # Create variables and constraints
            z = {}
            obj_expr = gp.LinExpr()
            
            # Add fixed costs
            for h in range(n):
                obj_expr += fixed_costs[h] * y[h]
            
            # For each commodity, create assignment variables
            commodity_constrs = []
            
            for c in range(m):
                o = origins[c]
                d_node = destinations[c]
                f = flows[c]
                
                assign_expr = gp.LinExpr()
                
                for k in range(n):
                    for l in range(k, n):
                        if k == l:
                            cc = f * (chi * dist[o][k] + delta * dist[k][d_node])
                        else:
                            c1 = f * (chi * dist[o][k] + tau * dist[k][l] + delta * dist[l][d_node])
                            c2 = f * (chi * dist[o][l] + tau * dist[l][k] + delta * dist[k][d_node])
                            cc = min(c1, c2)
                        
                        z[c, k, l] = model.addVar(vtype=GRB.BINARY, name=f"z_{c}_{k}_{l}")
                        obj_expr += cc * z[c, k, l]
                        assign_expr += z[c, k, l]
                        
                        # Link to hub opening
                        model.addConstr(z[c, k, l] <= y[k])
                        if k != l:
                            model.addConstr(z[c, k, l] <= y[l])
                
                model.addConstr(assign_expr == 1)
            
            model.setObjective(obj_expr, GRB.MINIMIZE)
            
            # Use LP relaxation first with aggressive settings
            model.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
            
            # Callback to log incumbent solutions
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                obj_val = model.ObjVal
                open_hubs = [h for h in range(n) if y[h].X > 0.5]
                
                assignments = []
                for c in range(m):
                    o = origins[c]
                    d_node = destinations[c]
                    f = flows[c]
                    
                    for k in range(n):
                        for l in range(k, n):
                            if z[c, k, l].X > 0.5:
                                if k == l:
                                    cc = f * (chi * dist[o][k] + delta * dist[k][d_node])
                                    assignments.append((k, k, cc))
                                else:
                                    c1 = f * (chi * dist[o][k] + tau * dist[k][l] + delta * dist[l][d_node])
                                    c2 = f * (chi * dist[o][l] + tau * dist[l][k] + delta * dist[k][d_node])
                                    if c1 <= c2:
                                        assignments.append((k, l, min(c1, c2)))
                                    else:
                                        assignments.append((l, k, min(c1, c2)))
                                break
                        else:
                            continue
                        break
                
                update_best(obj_val, assignments, open_hubs)
                
                # If optimal, write and return
                if model.Status == GRB.OPTIMAL:
                    write_solution(solution_path, best_obj, best_hubs, best_assignments, commodities)
                    return
                    
        except Exception as e:
            pass  # Fall through to heuristic
    
    remaining_time = time_limit - (time.time() - start_time)
    
    if remaining_time > 1:
        # Greedy heuristic + local search
        
        # First, find a good initial solution using greedy approach
        # Start with all hubs open, then try removing hubs
        
        # Evaluate all-hubs-open solution
        all_hubs = set(range(n))
        obj_all, assign_all, hubs_all = evaluate_solution(all_hubs)
        update_best(obj_all, assign_all, hubs_all)
        
        # Try each single hub
        for h in range(n):
            if time.time() - start_time > time_limit - 1:
                break
            obj_h, assign_h, hubs_h = evaluate_solution({h})
            update_best(obj_h, assign_h, hubs_h)
        
        # Greedy add: start from best single hub, add hubs that improve objective most
        # Find best single hub
        best_single_hub = -1
        best_single_obj = float('inf')
        for h in range(n):
            obj_h = fixed_costs[h]
            for c in range(m):
                obj_h += cost_single[c, h]
            if obj_h < best_single_obj:
                best_single_obj = obj_h
                best_single_hub = h
        
        current_hubs = {best_single_hub}
        current_obj, current_assign, _ = evaluate_solution(current_hubs)
        update_best(current_obj, current_assign, list(current_hubs))
        
        # Greedy add phase
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            best_add = -1
            best_add_obj = current_obj
            
            for h in range(n):
                if h in current_hubs:
                    continue
                if time.time() - start_time > time_limit - 1:
                    break
                    
                test_hubs = current_hubs | {h}
                obj_test, assign_test, hubs_test = evaluate_solution(test_hubs)
                if obj_test < best_add_obj:
                    best_add_obj = obj_test
                    best_add = h
                    best_add_assign = assign_test
                    best_add_hubs = hubs_test
            
            if best_add >= 0:
                current_hubs.add(best_add)
                current_obj = best_add_obj
                current_assign = best_add_assign
                update_best(current_obj, current_assign, list(current_hubs))
                improved = True
        
        # Local search: try removing each hub
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            for h in list(current_hubs):
                if len(current_hubs) <= 1:
                    break
                if time.time() - start_time > time_limit - 1:
                    break
                    
                test_hubs = current_hubs - {h}
                obj_test, assign_test, hubs_test = evaluate_solution(test_hubs)
                if obj_test < current_obj:
                    current_hubs = test_hubs
                    current_obj = obj_test
                    current_assign = assign_test
                    update_best(current_obj, current_assign, list(current_hubs))
                    improved = True
                    break  # restart removal loop
        
        # Swap: try replacing one hub with another
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 2:
            improved = True
            while improved and time.time() - start_time < time_limit - 2:
                improved = False
                for h_out in list(current_hubs):
                    if time.time() - start_time > time_limit - 1:
                        break
                    for h_in in range(n):
                        if h_in in current_hubs:
                            continue
                        if time.time() - start_time > time_limit - 1:
                            break
                        
                        test_hubs = (current_hubs - {h_out}) | {h_in}
                        obj_test, assign_test, hubs_test = evaluate_solution(test_hubs)
                        if obj_test < current_obj:
                            current_hubs = test_hubs
                            current_obj = obj_test
                            current_assign = assign_test
                            update_best(current_obj, current_assign, list(current_hubs))
                            improved = True
                            break
                    if improved:
                        break
        
        # Additional: try add again after removal/swap
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 2:
            improved = True
            while improved and time.time() - start_time < time_limit - 2:
                improved = False
                best_add = -1
                best_add_obj = current_obj
                
                for h in range(n):
                    if h in current_hubs:
                        continue
                    if time.time() - start_time > time_limit - 1:
                        break
                    
                    test_hubs = current_hubs | {h}
                    obj_test, assign_test, hubs_test = evaluate_solution(test_hubs)
                    if obj_test < best_add_obj:
                        best_add_obj = obj_test
                        best_add = h
                        best_add_assign = assign_test
                
                if best_add >= 0:
                    current_hubs.add(best_add)
                    current_obj = best_add_obj
                    current_assign = best_add_assign
                    update_best(current_obj, current_assign, list(current_hubs))
                    improved = True
        
        # Try removal again
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 2:
            improved = True
            while improved and time.time() - start_time < time_limit - 2:
                improved = False
                for h in list(current_hubs):
                    if len(current_hubs) <= 1:
                        break
                    if time.time() - start_time > time_limit - 1:
                        break
                    
                    test_hubs = current_hubs - {h}
                    obj_test, assign_test, hubs_test = evaluate_solution(test_hubs)
                    if obj_test < current_obj:
                        current_hubs = test_hubs
                        current_obj = obj_test
                        current_assign = assign_test
                        update_best(current_obj, current_assign, list(current_hubs))
                        improved = True
                        break
    
    write_solution(solution_path, best_obj, best_hubs, best_assignments, commodities)


def write_solution(solution_path, best_obj, best_hubs, best_assignments, commodities):
    # Build output
    output_assignments = []
    for c in range(len(commodities)):
        h1, h2, cost = best_assignments[c]
        if h1 == h2:
            hub_edge = [h1]
        else:
            # Unordered pair
            hub_edge = sorted([h1, h2])
        
        output_assignments.append({
            "commodity": c,
            "origin": commodities[c]['origin'],
            "destination": commodities[c]['destination'],
            "hub_edge": hub_edge,
            "cost": cost
        })
    
    solution = {
        "objective_value": best_obj,
        "open_hubs": sorted(best_hubs),
        "assignments": output_assignments
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


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