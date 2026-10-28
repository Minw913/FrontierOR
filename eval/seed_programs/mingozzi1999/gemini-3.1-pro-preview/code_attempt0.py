#!/usr/bin/env python3
import json
import argparse
import time
import random
import math
import copy

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="VRPB Optimization")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    
    start_time = time.time()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    num_vehicles = instance["num_vehicles"]
    vehicle_capacity = instance["vehicle_capacity"]
    dist_matrix = instance["distance_matrix"]
    
    L_customers = instance.get("linehaul_customers", [])
    B_customers = instance.get("backhaul_customers", [])
    
    N = instance["num_total_customers"]
    
    demands = [0] * (N + 1)
    L_set = []
    B_set = []
    for c in L_customers:
        idx = c["index"]
        L_set.append(idx)
        demands[idx] = c["demand"]
        
    for c in B_customers:
        idx = c["index"]
        B_set.append(idx)
        demands[idx] = c["demand"]

    # Initial FFD Bin Packing
    routes = [{'lp': [], 'bp': [], 'lp_load': 0, 'bp_load': 0} for _ in range(num_vehicles)]
    sorted_L = sorted(L_set, key=lambda i: demands[i], reverse=True)
    for node in sorted_L:
        r_min = min(routes, key=lambda x: x['lp_load'])
        for r in routes:
            if r['lp_load'] + demands[node] <= vehicle_capacity:
                r_min = r
                break
        r_min['lp'].append(node)
        r_min['lp_load'] += demands[node]

    sorted_B = sorted(B_set, key=lambda i: demands[i], reverse=True)
    for node in sorted_B:
        valid_routes = [r for r in routes if len(r['lp']) > 0]
        if not valid_routes:
            # Fallback if somehow no route has linehauls (should not happen if sorted_B exists)
            valid_routes = routes
        r_min = min(valid_routes, key=lambda x: x['bp_load'])
        for r in valid_routes:
            if r['bp_load'] + demands[node] <= vehicle_capacity:
                r_min = r
                break
        r_min['bp'].append(node)
        r_min['bp_load'] += demands[node]

    def calc_total_cost(r_list):
        cost = 0
        for r in r_list:
            c = 0
            prev = 0
            for node in r['lp']:
                c += dist_matrix[prev][node]
                prev = node
            for node in r['bp']:
                c += dist_matrix[prev][node]
                prev = node
            c += dist_matrix[prev][0]
            cost += c
            
            # Penalties
            if r['lp_load'] > vehicle_capacity:
                cost += (r['lp_load'] - vehicle_capacity) * 100000
            if r['bp_load'] > vehicle_capacity:
                cost += (r['bp_load'] - vehicle_capacity) * 100000
        return cost

    def is_feasible(r_list):
        for r in r_list:
            if r['lp_load'] > vehicle_capacity or r['bp_load'] > vehicle_capacity:
                return False
            if len(r['lp']) == 0 and len(r['bp']) > 0:
                return False
        return True

    def best_insertion_lp(lp, bp_head, u):
        best_delta = float('inf')
        best_idx = -1
        n = len(lp)
        for i in range(n + 1):
            prev = 0 if i == 0 else lp[i-1]
            nxt = bp_head if i == n else lp[i]
            delta = dist_matrix[prev][u] + dist_matrix[u][nxt] - dist_matrix[prev][nxt]
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        return best_idx, best_delta

    def best_insertion_bp(lp_tail, bp, u):
        best_delta = float('inf')
        best_idx = -1
        n = len(bp)
        for i in range(n + 1):
            prev = lp_tail if i == 0 else bp[i-1]
            nxt = 0 if i == n else bp[i]
            delta = dist_matrix[prev][u] + dist_matrix[u][nxt] - dist_matrix[prev][nxt]
            if delta < best_delta:
                best_delta = delta
                best_idx = i
        return best_idx, best_delta

    best_routes = copy.deepcopy(routes)
    best_cost = calc_total_cost(best_routes)
    
    best_feasible_routes = None
    best_feasible_cost = float('inf')
    
    if is_feasible(best_routes):
        best_feasible_routes = copy.deepcopy(best_routes)
        best_feasible_cost = best_cost
        if logger:
            logger.log(best_feasible_cost)
            
    curr_routes = copy.deepcopy(best_routes)
    curr_cost = best_cost

    alns_time_limit = min(args.time_limit * 0.5, 30.0)
    
    T_start = max(10.0, best_cost * 0.1)
    
    # ALNS Loop
    while True:
        elapsed = time.time() - start_time
        if elapsed > alns_time_limit:
            break
        
        time_ratio = elapsed / alns_time_limit
        T = T_start * (0.001 ** time_ratio)
        
        new_routes = copy.deepcopy(curr_routes)
        removed_L = []
        removed_B = []
        
        q = random.randint(2, max(2, int(N * 0.2)))
        
        all_L = []
        all_B = []
        for r_idx, r in enumerate(new_routes):
            for node in r['lp']: all_L.append((r_idx, node))
            for node in r['bp']: all_B.append((r_idx, node))
            
        all_nodes = [('L', r, n) for r, n in all_L] + [('B', r, n) for r, n in all_B]
        random.shuffle(all_nodes)
        to_remove = all_nodes[:q]
        
        for typ, r_idx, node in to_remove:
            if typ == 'L':
                new_routes[r_idx]['lp'].remove(node)
                new_routes[r_idx]['lp_load'] -= demands[node]
                removed_L.append(node)
            else:
                new_routes[r_idx]['bp'].remove(node)
                new_routes[r_idx]['bp_load'] -= demands[node]
                removed_B.append(node)
                
        for r_idx, r in enumerate(new_routes):
            if len(r['lp']) == 0 and len(r['bp']) > 0:
                for node in r['bp']:
                    removed_B.append(node)
                r['bp'] = []
                r['bp_load'] = 0
                
        removed_L.sort(key=lambda x: demands[x], reverse=True)
        removed_B.sort(key=lambda x: demands[x], reverse=True)
        
        for u in removed_L:
            b_val = float('inf')
            b_r = -1
            b_pos = -1
            for r_idx, r in enumerate(new_routes):
                old_pen = max(0, r['lp_load'] - vehicle_capacity) * 100000
                new_pen = max(0, r['lp_load'] + demands[u] - vehicle_capacity) * 100000
                
                bp_head = r['bp'][0] if r['bp'] else 0
                pos, delta = best_insertion_lp(r['lp'], bp_head, u)
                delta += (new_pen - old_pen)
                
                if delta < b_val:
                    b_val = delta
                    b_r = r_idx
                    b_pos = pos
            if b_r != -1:
                new_routes[b_r]['lp'].insert(b_pos, u)
                new_routes[b_r]['lp_load'] += demands[u]
                
        for u in removed_B:
            b_val = float('inf')
            b_r = -1
            b_pos = -1
            for r_idx, r in enumerate(new_routes):
                if len(r['lp']) == 0: continue
                old_pen = max(0, r['bp_load'] - vehicle_capacity) * 100000
                new_pen = max(0, r['bp_load'] + demands[u] - vehicle_capacity) * 100000
                
                lp_tail = r['lp'][-1]
                pos, delta = best_insertion_bp(lp_tail, r['bp'], u)
                delta += (new_pen - old_pen)
                
                if delta < b_val:
                    b_val = delta
                    b_r = r_idx
                    b_pos = pos
            if b_r != -1:
                new_routes[b_r]['bp'].insert(b_pos, u)
                new_routes[b_r]['bp_load'] += demands[u]
                
        new_cost = calc_total_cost(new_routes)
        
        if new_cost < curr_cost or random.random() < math.exp((curr_cost - new_cost) / T):
            curr_routes = new_routes
            curr_cost = new_cost
            if new_cost < best_cost:
                best_routes = copy.deepcopy(curr_routes)
                best_cost = new_cost
                
            if is_feasible(new_routes) and new_cost < best_feasible_cost:
                best_feasible_routes = copy.deepcopy(new_routes)
                best_feasible_cost = new_cost
                if logger:
                    logger.log(best_feasible_cost)

    solution_to_warm_start = best_feasible_routes if best_feasible_routes else best_routes

    # Gurobi MTZ exact model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(env=env)
    
    remaining_time = max(1.0, args.time_limit - (time.time() - start_time))
    model.setParam('TimeLimit', remaining_time)
    model.setParam('Threads', 1)

    x = {}
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j and i != 0: continue
            if i == 0 and j in B_set: continue
            if i in B_set and j in L_set: continue
            if i == 0 and j == 0:
                x[i, j] = model.addVar(vtype=GRB.INTEGER, ub=num_vehicles)
            else:
                x[i, j] = model.addVar(vtype=GRB.BINARY)

    for i in L_set + B_set:
        model.addConstr(gp.quicksum(x[i, j] for j in range(N+1) if (i,j) in x) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(N+1) if (j,i) in x) == 1)

    model.addConstr(gp.quicksum(x[0, j] for j in range(N+1) if (0,j) in x) == num_vehicles)
    model.addConstr(gp.quicksum(x[j, 0] for j in range(N+1) if (j,0) in x) == num_vehicles)

    u = {i: model.addVar(lb=1, ub=N, vtype=GRB.CONTINUOUS) for i in L_set + B_set}
    for i in L_set + B_set:
        for j in L_set + B_set:
            if i != j and (i,j) in x:
                model.addConstr(u[j] >= u[i] + 1 - N * (1 - x[i, j]))

    q = {i: model.addVar(lb=demands[i], ub=vehicle_capacity, vtype=GRB.CONTINUOUS) for i in L_set}
    for i in [0] + L_set:
        for j in L_set:
            if i != j and (i,j) in x:
                if i != 0:
                    model.addConstr(q[j] >= q[i] + demands[j] - vehicle_capacity * (1 - x[i, j]))

    p = {i: model.addVar(lb=demands[i], ub=vehicle_capacity, vtype=GRB.CONTINUOUS) for i in B_set}
    for i in L_set + B_set:
        for j in B_set:
            if i != j and (i,j) in x:
                if i in B_set:
                    model.addConstr(p[j] >= p[i] + demands[j] - vehicle_capacity * (1 - x[i, j]))

    model.setObjective(gp.quicksum(dist_matrix[i][j] * x[i, j] for (i,j) in x), GRB.MINIMIZE)

    # Warm Start with ALNS
    start_counts = {(i,j): 0 for (i,j) in x}
    for r in solution_to_warm_start:
        prev = 0
        for node in r['lp']:
            start_counts[prev, node] += 1
            prev = node
        for node in r['bp']:
            start_counts[prev, node] += 1
            prev = node
        start_counts[prev, 0] += 1

    for (i, j), count in start_counts.items():
        if count > 0:
            x[i, j].Start = count
        else:
            x[i, j].Start = 0

    def my_callback(mod, where):
        if where == GRB.Callback.MIPSOL:
            obj = mod.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    try:
        model.optimize(my_callback)
    except gp.GurobiError:
        pass

    final_cost = float('inf')
    final_routes = []

    if model.SolCount > 0:
        final_cost = model.ObjVal
        adj = {i: [] for i in range(N+1)}
        for (i, j), var in x.items():
            val = int(round(var.X))
            if i == 0 and j == 0:
                for _ in range(val):
                    final_routes.append([0, 0])
            elif val > 0:
                for _ in range(val):
                    adj[i].append(j)
                
        for nxt in adj[0]:
            route = [0, nxt]
            curr = nxt
            max_len = N + 2
            while curr != 0 and len(route) < max_len:
                if not adj[curr]:
                    route.append(0)
                    break
                nxt_curr = adj[curr].pop(0)
                route.append(nxt_curr)
                curr = nxt_curr
            final_routes.append(route)
            
        while len(final_routes) < num_vehicles:
            final_routes.append([0, 0])

    if not final_routes or final_cost >= best_feasible_cost:
        final_cost = best_feasible_cost
        final_routes = []
        for r in solution_to_warm_start:
            seq = [0] + r['lp'] + r['bp'] + [0]
            final_routes.append(seq)
            
    # Compute Exact Objective based on standard sequences
    exact_cost = 0.0
    for r in final_routes:
        for i in range(len(r) - 1):
            exact_cost += dist_matrix[r[i]][r[i+1]]

    output_data = {
        "objective_value": float(exact_cost),
        "routes": final_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()