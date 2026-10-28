import argparse
import json
import time
import math

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
    delta_param = params['delta']
    
    nodes = instance['nodes']
    dist = instance['distances']
    commodities_data = instance['commodities']
    
    n = num_nodes
    m = num_commodities
    
    fixed_costs = [nd['fixed_cost'] for nd in nodes]
    flows = [c['flow'] for c in commodities_data]
    origins = [c['origin'] for c in commodities_data]
    destinations = [c['destination'] for c in commodities_data]
    
    # Aggregate commodities by (origin, destination) pair
    od_flow = {}
    od_commodities = {}
    for c in range(m):
        o = origins[c]
        d = destinations[c]
        key = (o, d)
        if key not in od_flow:
            od_flow[key] = 0.0
            od_commodities[key] = []
        od_flow[key] += flows[c]
        od_commodities[key].append(c)
    
    od_list = list(od_flow.keys())
    num_od = len(od_list)
    od_flows = [od_flow[k] for k in od_list]
    od_origins = [k[0] for k in od_list]
    od_dests = [k[1] for k in od_list]
    
    def compute_od_cost_single(oi, k):
        """Cost for OD pair oi through single hub k (per unit flow already factored)."""
        o = od_origins[oi]
        d = od_dests[oi]
        return od_flows[oi] * (chi * dist[o][k] + delta_param * dist[k][d])
    
    def compute_od_cost_pair(oi, k, l):
        """Cost for OD pair oi through hub pair (k, l), choosing best direction."""
        o = od_origins[oi]
        d = od_dests[oi]
        if k == l:
            return od_flows[oi] * (chi * dist[o][k] + delta_param * dist[k][d])
        c1 = chi * dist[o][k] + tau * dist[k][l] + delta_param * dist[l][d]
        c2 = chi * dist[o][l] + tau * dist[l][k] + delta_param * dist[k][d]
        return od_flows[oi] * min(c1, c2)
    
    def evaluate_hubs_fast(hub_list):
        """Evaluate total cost for a given list of open hubs. Returns (total_obj, best_hub_assignments)."""
        nh = len(hub_list)
        total_fixed = sum(fixed_costs[h] for h in hub_list)
        total_transport = 0.0
        # For each OD pair, find best hub pair
        od_best_k = [0] * num_od
        od_best_l = [0] * num_od
        od_best_cost = [0.0] * num_od
        
        for oi in range(num_od):
            o = od_origins[oi]
            d = od_dests[oi]
            f = od_flows[oi]
            
            best_c = float('inf')
            bk = hub_list[0]
            bl = hub_list[0]
            
            # Precompute collection and distribution costs for each hub
            coll = [chi * dist[o][hub_list[i]] for i in range(nh)]
            distr = [delta_param * dist[hub_list[i]][d] for i in range(nh)]
            
            # Single hubs
            for i in range(nh):
                cc = coll[i] + distr[i]
                if cc < best_c:
                    best_c = cc
                    bk = hub_list[i]
                    bl = hub_list[i]
            
            # Hub pairs
            for i in range(nh):
                for j in range(i + 1, nh):
                    transfer = tau * dist[hub_list[i]][hub_list[j]]
                    c1 = coll[i] + transfer + distr[j]
                    c2 = coll[j] + transfer + distr[i]
                    cc = min(c1, c2)
                    if cc < best_c:
                        best_c = cc
                        if c1 <= c2:
                            bk = hub_list[i]
                            bl = hub_list[j]
                        else:
                            bk = hub_list[j]
                            bl = hub_list[i]
            
            od_best_k[oi] = bk
            od_best_l[oi] = bl
            od_best_cost[oi] = f * best_c
            total_transport += f * best_c
        
        return total_fixed + total_transport, od_best_k, od_best_l, od_best_cost
    
    best_obj = float('inf')
    best_od_k = None
    best_od_l = None
    best_od_cost = None
    best_hubs = None
    
    def update_best(obj, od_k, od_l, od_c, hubs):
        nonlocal best_obj, best_od_k, best_od_l, best_od_cost, best_hubs
        if obj < best_obj:
            best_obj = obj
            best_od_k = list(od_k)
            best_od_l = list(od_l)
            best_od_cost = list(od_c)
            best_hubs = sorted(hubs)
            if logger:
                logger.log(best_obj)
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # --- Phase 1: Find best single hub ---
    best_single_obj = float('inf')
    best_single_hub = 0
    for k in range(n):
        total = fixed_costs[k]
        for oi in range(num_od):
            o = od_origins[oi]
            d = od_dests[oi]
            total += od_flows[oi] * (chi * dist[o][k] + delta_param * dist[k][d])
        if total < best_single_obj:
            best_single_obj = total
            best_single_hub = k
    
    current_hubs = {best_single_hub}
    obj, od_k, od_l, od_c = evaluate_hubs_fast(sorted(current_hubs))
    update_best(obj, od_k, od_l, od_c, current_hubs)
    current_obj = obj
    
    # --- Phase 2: Greedy add hubs ---
    improved = True
    while improved and time_remaining() > 5:
        improved = False
        best_add = -1
        best_add_obj = current_obj
        best_add_data = None
        
        for h in range(n):
            if h in current_hubs:
                continue
            if time_remaining() < 3:
                break
            test_hubs = sorted(current_hubs | {h})
            obj_test, ok, ol, oc = evaluate_hubs_fast(test_hubs)
            if obj_test < best_add_obj:
                best_add_obj = obj_test
                best_add = h
                best_add_data = (ok, ol, oc)
        
        if best_add >= 0:
            current_hubs.add(best_add)
            current_obj = best_add_obj
            update_best(current_obj, best_add_data[0], best_add_data[1], best_add_data[2], current_hubs)
            improved = True
    
    # --- Phase 3: Greedy drop hubs ---
    if len(current_hubs) > 1 and time_remaining() > 5:
        improved = True
        while improved and time_remaining() > 5:
            improved = False
            best_drop = -1
            best_drop_obj = current_obj
            best_drop_data = None
            
            for h in list(current_hubs):
                if len(current_hubs) <= 1:
                    break
                if time_remaining() < 3:
                    break
                test_hubs = sorted(current_hubs - {h})
                obj_test, ok, ol, oc = evaluate_hubs_fast(test_hubs)
                if obj_test < best_drop_obj:
                    best_drop_obj = obj_test
                    best_drop = h
                    best_drop_data = (ok, ol, oc)
            
            if best_drop >= 0:
                current_hubs.discard(best_drop)
                current_obj = best_drop_obj
                update_best(current_obj, best_drop_data[0], best_drop_data[1], best_drop_data[2], current_hubs)
                improved = True
    
    # --- Phase 4: Swap hubs (remove one, add another) ---
    if time_remaining() > 10:
        improved = True
        while improved and time_remaining() > 5:
            improved = False
            best_swap_obj = current_obj
            best_swap_remove = -1
            best_swap_add = -1
            best_swap_data = None
            
            for h_remove in list(current_hubs):
                if time_remaining() < 3:
                    break
                for h_add in range(n):
                    if h_add in current_hubs:
                        continue
                    if time_remaining() < 2:
                        break
                    test_hubs = sorted((current_hubs - {h_remove}) | {h_add})
                    obj_test, ok, ol, oc = evaluate_hubs_fast(test_hubs)
                    if obj_test < best_swap_obj:
                        best_swap_obj = obj_test
                        best_swap_remove = h_remove
                        best_swap_add = h_add
                        best_swap_data = (ok, ol, oc)
            
            if best_swap_remove >= 0:
                current_hubs.discard(best_swap_remove)
                current_hubs.add(best_swap_add)
                current_obj = best_swap_obj
                update_best(current_obj, best_swap_data[0], best_swap_data[1], best_swap_data[2], current_hubs)
                improved = True
    
    # --- Phase 5: Try Gurobi for small instances ---
    if n <= 50 and time_remaining() > 10:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining = time_remaining() - 2
            if remaining < 5:
                remaining = 5
            
            hub_pairs = []
            for k in range(n):
                for l in range(k, n):
                    hub_pairs.append((k, l))
            
            num_pairs = len(hub_pairs)
            
            # unit routing cost per OD pair per hub pair
            unit_cost = []
            for oi in range(num_od):
                o = od_origins[oi]
                d = od_dests[oi]
                costs = []
                for k, l in hub_pairs:
                    if k == l:
                        cc = chi * dist[o][k] + delta_param * dist[k][d]
                    else:
                        c1 = chi * dist[o][k] + tau * dist[k][l] + delta_param * dist[l][d]
                        c2 = chi * dist[o][l] + tau * dist[l][k] + delta_param * dist[k][d]
                        cc = min(c1, c2)
                    costs.append(cc)
                unit_cost.append(costs)
            
            model = gp.Model("hub_location")
            model.setParam('TimeLimit', remaining)
            model.setParam('Threads', 1)
            model.setParam('MIPGap', 1e-4)
            model.setParam('OutputFlag', 0)
            
            y = {}
            for k in range(n):
                y[k] = model.addVar(vtype=GRB.BINARY, obj=fixed_costs[k], name=f"y_{k}")
            
            x = {}
            for oi in range(num_od):
                f_od = od_flows[oi]
                for pi in range(num_pairs):
                    x[oi, pi] = model.addVar(vtype=GRB.BINARY,
                                              obj=f_od * unit_cost[oi][pi],
                                              name=f"x_{oi}_{pi}")
            
            model.update()
            
            for oi in range(num_od):
                model.addConstr(
                    gp.quicksum(x[oi, pi] for pi in range(num_pairs)) == 1,
                    name=f"assign_{oi}"
                )
            
            # Linking: aggregate by hub
            hub_to_pairs = [[] for _ in range(n)]
            for pi, (k, l) in enumerate(hub_pairs):
                hub_to_pairs[k].append(pi)
                if l != k:
                    hub_to_pairs[l].append(pi)
            
            for k in range(n):
                if not hub_to_pairs[k]:
                    continue
                for oi in range(num_od):
                    model.addConstr(
                        gp.quicksum(x[oi, pi] for pi in hub_to_pairs[k]) <= y[k],
                        name=f"link_{oi}_{k}"
                    )
            
            # Warm start from best known solution
            if best_hubs is not None:
                for k in range(n):
                    y[k].Start = 1.0 if k in set(best_hubs) else 0.0
                
                for oi in range(num_od):
                    hk = best_od_k[oi]
                    hl = best_od_l[oi]
                    pair_key = (min(hk, hl), max(hk, hl))
                    for pi in range(num_pairs):
                        if hub_pairs[pi] == pair_key:
                            x[oi, pi].Start = 1.0
                        else:
                            x[oi, pi].Start = 0.0
            
            model.optimize()
            
            if model.SolCount > 0:
                open_hubs_set = set()
                for k in range(n):
                    if y[k].X > 0.5:
                        open_hubs_set.add(k)
                
                od_assignment = {}
                for oi in range(num_od):
                    for pi in range(num_pairs):
                        if x[oi, pi].X > 0.5:
                            od_assignment[oi] = hub_pairs[pi]
                            break
                
                gurobi_od_k = [0] * num_od
                gurobi_od_l = [0] * num_od
                gurobi_od_c = [0.0] * num_od
                
                for oi in range(num_od):
                    o = od_origins[oi]
                    d = od_dests[oi]
                    k, l = od_assignment.get(oi, (list(open_hubs_set)[0], list(open_hubs_set)[0]))
                    if k == l:
                        first_h, second_h = k, l
                        cc = chi * dist[o][k] + delta_param * dist[k][d]
                    else:
                        c1 = chi * dist[o][k] + tau * dist[k][l] + delta_param * dist[l][d]
                        c2 = chi * dist[o][l] + tau * dist[l][k] + delta_param * dist[k][d]
                        if c1 <= c2:
                            first_h, second_h = k, l
                            cc = c1
                        else:
                            first_h, second_h = l, k
                            cc = c2
                    gurobi_od_k[oi] = first_h
                    gurobi_od_l[oi] = second_h
                    gurobi_od_c[oi] = od_flows[oi] * cc
                
                total_obj = sum(fixed_costs[h] for h in open_hubs_set) + sum(gurobi_od_c)
                update_best(total_obj, gurobi_od_k, gurobi_od_l, gurobi_od_c, open_hubs_set)
        except Exception:
            pass
    
    # Build per-commodity output
    h1_arr = [0] * m
    h2_arr = [0] * m
    cost_arr = [0.0] * m
    
    for oi, key in enumerate(od_list):
        first_h = best_od_k[oi]
        second_h = best_od_l[oi]
        o, d = key
        
        for c in od_commodities[key]:
            h1_arr[c] = first_h
            h2_arr[c] = second_h
            if first_h == second_h:
                cost_arr[c] = flows[c] * (chi * dist[o][first_h] + delta_param * dist[first_h][d])
            else:
                cost_arr[c] = flows[c] * (chi * dist[o][first_h] + tau * dist[first_h][second_h] + delta_param * dist[second_h][d])
    
    output_assignments = []
    for c in range(m):
        h1_val = h1_arr[c]
        h2_val = h2_arr[c]
        hub_edge = [h1_val] if h1_val == h2_val else sorted([h1_val, h2_val])
        output_assignments.append({
            "commodity": c,
            "origin": origins[c],
            "destination": destinations[c],
            "hub_edge": hub_edge,
            "cost": cost_arr[c]
        })
    
    # Recompute objective for safety
    final_obj = sum(fixed_costs[h] for h in best_hubs) + sum(cost_arr)
    
    solution = {
        "objective_value": final_obj,
        "open_hubs": best_hubs,
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