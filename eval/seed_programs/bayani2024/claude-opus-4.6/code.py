import argparse
import json
import time
import numpy as np
from collections import defaultdict

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n_clients = instance['n_clients']
    n_servers = instance['n_servers']
    demands = instance['demands']
    processing_rates = instance['processing_rates']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Build linear cost matrix
    lc = np.array(linear_costs)  # shape [n_clients][n_servers]
    
    # Build quadratic cost data structure
    # For each pair of (client, server) assignments, store the quadratic cost
    # quad_dict[(c1, s1, c2, s2)] = cost
    # We need to efficiently compute total quadratic cost for a given assignment
    
    # For efficiency, precompute:
    # 1. Adjacent costs: when two clients share the same server
    #    For adjacent pairs, cost = lc[c1][s] + lc[c2][s]
    # 2. Out-of-star costs: specified in quadratic_costs for different servers
    
    # Group quadratic costs by type for efficient evaluation
    # For a given assignment, we need sum of all active quadratic costs
    
    # Build index: for each client c, for each server s it could be assigned to,
    # what quadratic interactions does it have?
    # quad_by_client_server[(c, s)] = list of (c2, s2, cost)
    
    # But we also need to handle adjacent costs efficiently
    # Adjacent costs: for clients c1, c2 on same server s, cost = lc[c1][s] + lc[c2][s]
    # This means for a server s with clients C_s, adjacent cost = sum over pairs (c1,c2) in C_s of (lc[c1][s] + lc[c2][s])
    # = sum over pairs of lc[c1][s] + sum over pairs of lc[c2][s]
    # = (|C_s| - 1) * sum_{c in C_s} lc[c][s]
    
    # So adjacent cost for server s with client set C_s:
    # = (|C_s| - 1) * sum_{c in C_s} lc[c][s]
    
    # Out-of-star costs: need to track separately
    # Build dict for out-of-star costs
    oos_costs = defaultdict(list)  # oos_costs[c1] = list of (s1, c2, s2, cost)
    
    # Also build a structure for quick delta computation
    # When we change client c from server s_old to s_new, we need to compute the change in cost
    
    # Parse quadratic costs
    # Separate adjacent from out-of-star
    oos_by_pair = {}  # (c1, s1, c2, s2) -> cost, only out-of-star
    
    for entry in quadratic_costs:
        c1, s1, c2, s2, cost = int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3]), entry[4]
        if s1 == s2:
            # Adjacent - we handle these analytically
            pass
        else:
            oos_by_pair[(c1, s1, c2, s2)] = cost
            # Also store reverse for convenience (if not already symmetric)
    
    # Build index: for each client c, list of out-of-star interactions
    # oos_by_client[c] = list of (s_c, c2, s_c2, cost) meaning if c->s_c and c2->s_c2, add cost
    oos_by_client = defaultdict(list)
    for (c1, s1, c2, s2), cost in oos_by_pair.items():
        oos_by_client[c1].append((s1, c2, s2, cost))
        oos_by_client[c2].append((s2, c1, s1, cost))
    
    # Remove duplicates in oos_by_client by being careful:
    # Actually, the quadratic_costs list each pair once. When we index by both c1 and c2,
    # we'll double-count. We need to be careful.
    # Let's rebuild: only store each interaction once but index from both sides
    
    # Actually, let's use a different approach for delta computation
    # For out-of-star, store for each client c and server s:
    # oos_index[(c, s)] = dict mapping c2 -> (s2, cost) 
    # meaning if c is assigned to s, and c2 is assigned to s2, add cost
    
    # But a client c2 has a specific assignment, so we need:
    # For client c assigned to server s, the oos contribution involving c is:
    # sum over all (c,s,c2,s2,cost) in oos where c2 is currently assigned to s2
    # + sum over all (c2,s2,c,s,cost) in oos where c2 is currently assigned to s2
    
    # Let's build: oos_from[(c, s)] = list of (c2, s2, cost)
    # meaning if c->s and c2->s2, then add cost
    oos_from = defaultdict(list)
    for (c1, s1, c2, s2), cost in oos_by_pair.items():
        oos_from[(c1, s1)].append((c2, s2, cost))
        oos_from[(c2, s2)].append((c1, s1, cost))
    
    def compute_objective(assignment):
        """Compute total objective for a given assignment."""
        total = 0.0
        
        # Linear costs
        for c in range(n_clients):
            total += lc[c][assignment[c]]
        
        # Adjacent quadratic costs
        # Group clients by server
        server_clients = defaultdict(list)
        for c in range(n_clients):
            server_clients[assignment[c]].append(c)
        
        for s, clients in server_clients.items():
            if len(clients) > 1:
                sum_lc = sum(lc[c][s] for c in clients)
                total += (len(clients) - 1) * sum_lc
        
        # Out-of-star quadratic costs
        for (c1, s1, c2, s2), cost in oos_by_pair.items():
            if assignment[c1] == s1 and assignment[c2] == s2:
                total += cost
        
        return total
    
    def compute_objective_fast(assignment, server_clients, server_sum_lc, server_count):
        """Compute total objective using precomputed server data."""
        total = 0.0
        
        # Linear costs
        for c in range(n_clients):
            total += lc[c][assignment[c]]
        
        # Adjacent quadratic costs
        for s in range(n_servers):
            if server_count[s] > 1:
                total += (server_count[s] - 1) * server_sum_lc[s]
        
        # Out-of-star quadratic costs
        for (c1, s1, c2, s2), cost in oos_by_pair.items():
            if assignment[c1] == s1 and assignment[c2] == s2:
                total += cost
        
        return total
    
    # For small instances, try Gurobi MIQP/linearization
    # For larger instances, use local search
    
    use_gurobi = (n_clients <= 50 and n_servers <= 20) or (n_clients * n_servers <= 500)
    
    # Try Gurobi for small-medium instances
    best_assignment = None
    best_obj = float('inf')
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if use_gurobi and remaining > 2:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', remaining - 1)
            model.setParam('Threads', 1)
            
            # Binary variables x[c][s] = 1 if client c assigned to server s
            x = {}
            for c in range(n_clients):
                for s in range(n_servers):
                    x[c, s] = model.addVar(vtype=GRB.BINARY, name=f'x_{c}_{s}')
            
            # Each client assigned to exactly one server
            for c in range(n_clients):
                model.addConstr(gp.quicksum(x[c, s] for s in range(n_servers)) == 1)
            
            # Objective: linear + quadratic
            obj = gp.QuadExpr()
            
            # Linear costs
            for c in range(n_clients):
                for s in range(n_servers):
                    obj += lc[c][s] * x[c, s]
            
            # Adjacent quadratic costs
            # For each server s and pair of clients (c1, c2), if both assigned to s:
            # cost = lc[c1][s] + lc[c2][s]
            # This equals (lc[c1][s] + lc[c2][s]) * x[c1,s] * x[c2,s]
            for s in range(n_servers):
                for c1 in range(n_clients):
                    for c2 in range(c1 + 1, n_clients):
                        adj_cost = lc[c1][s] + lc[c2][s]
                        if adj_cost != 0:
                            obj += adj_cost * x[c1, s] * x[c2, s]
            
            # Out-of-star quadratic costs
            for (c1, s1, c2, s2), cost in oos_by_pair.items():
                if cost != 0:
                    obj += cost * x[c1, s1] * x[c2, s2]
            
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Callback to log solutions
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        logger.log(obj_val)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                assignment = {}
                for c in range(n_clients):
                    for s in range(n_servers):
                        if x[c, s].X > 0.5:
                            assignment[c] = s
                            break
                
                obj_val = model.ObjVal
                best_assignment = assignment
                best_obj = obj_val
                
        except Exception as e:
            print(f"Gurobi failed: {e}")
            use_gurobi = False
    
    # If Gurobi didn't work or for larger instances, use local search
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if best_assignment is None or remaining > 2:
        # Greedy initialization if no solution yet
        if best_assignment is None:
            # Simple greedy: assign each client to cheapest server (linear only)
            assignment = np.zeros(n_clients, dtype=int)
            for c in range(n_clients):
                assignment[c] = np.argmin(lc[c])
            
            # Convert to dict-like
            assign_dict = {c: int(assignment[c]) for c in range(n_clients)}
            obj_val = compute_objective(assign_dict)
            
            if obj_val < best_obj:
                best_obj = obj_val
                best_assignment = assign_dict.copy()
                if logger:
                    logger.log(best_obj)
        
        # Local search: try moving each client to each server
        # Use efficient delta computation
        
        assignment = np.array([best_assignment[c] for c in range(n_clients)], dtype=int)
        
        # Precompute server data
        server_clients_list = [[] for _ in range(n_servers)]
        server_sum_lc = np.zeros(n_servers)
        server_count = np.zeros(n_servers, dtype=int)
        
        for c in range(n_clients):
            s = assignment[c]
            server_clients_list[s].append(c)
            server_sum_lc[s] += lc[c][s]
            server_count[s] += 1
        
        # Precompute oos contributions for current assignment
        # For each client c, oos_contribution[c] = sum of oos costs involving c that are active
        # Total oos cost = sum of oos_contribution[c] / 2 (each pair counted twice)
        # Actually, let's just track total cost
        
        current_obj = best_obj
        
        # Build oos index more efficiently
        # For each client c, for each possible server s_c:
        # oos_effect[(c, s_c)] = list of (c2, s2, cost)
        # meaning: if c is assigned to s_c and c2 is assigned to s2, cost is added
        
        # Delta computation for moving client c from s_old to s_new:
        # Linear change: lc[c][s_new] - lc[c][s_old]
        # Adjacent change:
        #   Remove c from s_old: we lose interactions with all other clients on s_old
        #     Lost adjacent = sum over c2 in s_old\{c} of (lc[c][s_old] + lc[c2][s_old])
        #     = (server_count[s_old]-1) * lc[c][s_old] + (server_sum_lc[s_old] - lc[c][s_old])*(1 if server_count[s_old]>1 else 0)
        #     Wait, more carefully:
        #     = sum_{c2 in s_old, c2 != c} (lc[c][s_old] + lc[c2][s_old])
        #     = (count_old - 1) * lc[c][s_old] + (sum_lc_old - lc[c][s_old])  [if count_old > 1]
        #   Add c to s_new: gain interactions with all clients on s_new
        #     Gained adjacent = sum_{c2 in s_new} (lc[c][s_new] + lc[c2][s_new])
        #     = count_new * lc[c][s_new] + sum_lc_new
        # OOS change:
        #   For each oos entry involving c:
        #     If entry is (c, s_old, c2, s2, cost) and assignment[c2]==s2: lose cost
        #     If entry is (c, s_new, c2, s2, cost) and assignment[c2]==s2: gain cost
        
        def compute_delta(c, s_old, s_new):
            """Compute change in objective when moving client c from s_old to s_new."""
            delta = 0.0
            
            # Linear change
            delta += lc[c][s_new] - lc[c][s_old]
            
            # Adjacent change - remove from s_old
            count_old = server_count[s_old]
            if count_old > 1:
                # Lost adjacent costs
                lost = (count_old - 1) * lc[c][s_old] + (server_sum_lc[s_old] - lc[c][s_old])
                delta -= lost
            
            # Adjacent change - add to s_new
            count_new = server_count[s_new]
            if count_new > 0:
                # Gained adjacent costs
                gained = count_new * lc[c][s_new] + server_sum_lc[s_new]
                delta += gained
            
            # OOS change
            # Check all oos entries involving client c
            for (s_c, c2, s_c2, cost) in oos_by_client[c]:
                if assignment[c2] == s_c2:
                    if s_c == s_old:
                        delta -= cost
                    if s_c == s_new:
                        delta += cost
            
            return delta
        
        # Local search iterations
        improved = True
        iteration = 0
        
        while improved and (time.time() - start_time) < time_limit - 0.5:
            improved = False
            iteration += 1
            
            for c in range(n_clients):
                if (time.time() - start_time) >= time_limit - 0.5:
                    break
                
                s_old = assignment[c]
                best_delta = 0.0
                best_s_new = -1
                
                for s_new in range(n_servers):
                    if s_new == s_old:
                        continue
                    
                    delta = compute_delta(c, s_old, s_new)
                    
                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_s_new = s_new
                
                if best_s_new >= 0:
                    # Apply move
                    s_new = best_s_new
                    
                    # Update server data
                    server_clients_list[s_old].remove(c)
                    server_sum_lc[s_old] -= lc[c][s_old]
                    server_count[s_old] -= 1
                    
                    server_clients_list[s_new].append(c)
                    server_sum_lc[s_new] += lc[c][s_new]
                    server_count[s_new] += 1
                    
                    assignment[c] = s_new
                    current_obj += best_delta
                    
                    if current_obj < best_obj - 1e-10:
                        best_obj = current_obj
                        best_assignment = {i: int(assignment[i]) for i in range(n_clients)}
                        if logger:
                            logger.log(best_obj)
                    
                    improved = True
        
        # Try swap-based improvements (swap assignments of two clients)
        elapsed = time.time() - start_time
        if elapsed < time_limit - 1 and n_clients <= 500:
            swap_improved = True
            while swap_improved and (time.time() - start_time) < time_limit - 0.5:
                swap_improved = False
                for c1 in range(n_clients):
                    if (time.time() - start_time) >= time_limit - 0.5:
                        break
                    for c2 in range(c1 + 1, n_clients):
                        if (time.time() - start_time) >= time_limit - 0.5:
                            break
                        
                        s1 = assignment[c1]
                        s2 = assignment[c2]
                        if s1 == s2:
                            continue
                        
                        # Compute delta for swapping c1 and c2
                        # Move c1 from s1 to s2, then move c2 from s2 to s1
                        # But we need to account for interactions between c1 and c2
                        
                        # First, compute delta for moving c1 to s2
                        delta1 = compute_delta(c1, s1, s2)
                        
                        # Temporarily apply move
                        server_clients_list[s1].remove(c1)
                        server_sum_lc[s1] -= lc[c1][s1]
                        server_count[s1] -= 1
                        server_clients_list[s2].append(c1)
                        server_sum_lc[s2] += lc[c1][s2]
                        server_count[s2] += 1
                        assignment[c1] = s2
                        
                        # Compute delta for moving c2 to s1
                        delta2 = compute_delta(c2, s2, s1)
                        
                        total_delta = delta1 + delta2
                        
                        # Undo temporary move
                        server_clients_list[s2].remove(c1)
                        server_sum_lc[s2] -= lc[c1][s2]
                        server_count[s2] -= 1
                        server_clients_list[s1].append(c1)
                        server_sum_lc[s1] += lc[c1][s1]
                        server_count[s1] += 1
                        assignment[c1] = s1
                        
                        if total_delta < -1e-10:
                            # Apply swap
                            # Move c1 to s2
                            server_clients_list[s1].remove(c1)
                            server_sum_lc[s1] -= lc[c1][s1]
                            server_count[s1] -= 1
                            server_clients_list[s2].append(c1)
                            server_sum_lc[s2] += lc[c1][s2]
                            server_count[s2] += 1
                            assignment[c1] = s2
                            
                            # Move c2 to s1
                            server_clients_list[s2].remove(c2)
                            server_sum_lc[s2] -= lc[c2][s2]
                            server_count[s2] -= 1
                            server_clients_list[s1].append(c2)
                            server_sum_lc[s1] += lc[c2][s1]
                            server_count[s1] += 1
                            assignment[c2] = s1
                            
                            current_obj += total_delta
                            
                            if current_obj < best_obj - 1e-10:
                                best_obj = current_obj
                                best_assignment = {i: int(assignment[i]) for i in range(n_clients)}
                                if logger:
                                    logger.log(best_obj)
                            
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                
                # After swaps, do another round of single moves
                if swap_improved:
                    single_improved = True
                    while single_improved and (time.time() - start_time) < time_limit - 0.5:
                        single_improved = False
                        for c in range(n_clients):
                            if (time.time() - start_time) >= time_limit - 0.5:
                                break
                            s_old = assignment[c]
                            best_delta_val = 0.0
                            best_s_new_val = -1
                            for s_new in range(n_servers):
                                if s_new == s_old:
                                    continue
                                delta = compute_delta(c, s_old, s_new)
                                if delta < best_delta_val - 1e-10:
                                    best_delta_val = delta
                                    best_s_new_val = s_new
                            if best_s_new_val >= 0:
                                s_new = best_s_new_val
                                server_clients_list[s_old].remove(c)
                                server_sum_lc[s_old] -= lc[c][s_old]
                                server_count[s_old] -= 1
                                server_clients_list[s_new].append(c)
                                server_sum_lc[s_new] += lc[c][s_new]
                                server_count[s_new] += 1
                                assignment[c] = s_new
                                current_obj += best_delta_val
                                if current_obj < best_obj - 1e-10:
                                    best_obj = current_obj
                                    best_assignment = {i: int(assignment[i]) for i in range(n_clients)}
                                    if logger:
                                        logger.log(best_obj)
                                single_improved = True
    
    # Recompute objective to be safe
    final_obj = compute_objective(best_assignment)
    
    # Write solution
    solution = {
        "objective_value": final_obj,
        "assignment": {str(c): best_assignment[c] for c in range(n_clients)}
    }
    
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