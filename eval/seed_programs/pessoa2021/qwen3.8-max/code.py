#!/usr/bin/env python3
"""
Robust Vehicle Routing Problem Solver.
Combines Clarke-Wright savings heuristic, local search, and optional MIP for small instances.
"""

import json
import argparse
import time
import random
from copy import deepcopy

def compute_worst_case_demand(customer_list, nominal, deviation, Gamma):
    """Compute worst-case demand for a set of customers under budgeted uncertainty."""
    if not customer_list:
        return 0
    nom_sum = sum(nominal[c] for c in customer_list)
    devs = sorted([deviation[c] for c in customer_list], reverse=True)
    additional = sum(devs[:min(Gamma, len(devs))])
    return nom_sum + additional

def route_cost(route, dist, depot):
    """Compute cost of a single route (depot -> customers -> depot)."""
    if not route:
        return 0
    cost = dist[depot][route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i]][route[i + 1]]
    cost += dist[route[-1]][depot]
    return cost

def total_cost(routes, dist, depot):
    """Compute total cost across all routes."""
    return sum(route_cost(r, dist, depot) for r in routes)

def format_solution(routes, dist, depot, nominal, deviation, Gamma):
    """Format solution according to output schema."""
    obj = total_cost(routes, dist, depot)
    solution = {
        "objective_value": float(obj),
        "routes": []
    }
    for r in routes:
        wc = compute_worst_case_demand(r, nominal, deviation, Gamma)
        c = route_cost(r, dist, depot)
        solution["routes"].append({
            "customers": list(r),
            "cost": int(c),
            "worst_case_demand": int(wc)
        })
    return solution

def clarke_wright_savings(customer_ids, dist, depot, nominal, deviation, Gamma, capacity, num_vehicles):
    """Clarke-Wright savings heuristic with robust feasibility checks."""
    routes = [[c] for c in customer_ids]
    
    # Check individual feasibility
    for c in customer_ids:
        if compute_worst_case_demand([c], nominal, deviation, Gamma) > capacity:
            return None
    
    # Compute all savings
    savings = []
    for i in range(len(customer_ids)):
        for j in range(i + 1, len(customer_ids)):
            ci, cj = customer_ids[i], customer_ids[j]
            # Savings for different merge orientations
            s = dist[depot][ci] + dist[cj][depot] - dist[ci][cj]
            savings.append((s, ci, cj))
            s2 = dist[depot][cj] + dist[ci][depot] - dist[cj][ci]
            savings.append((s2, cj, ci))
    savings.sort(reverse=True, key=lambda x: x[0])
    
    def find_route_idx(cust, rts):
        for idx, r in enumerate(rts):
            if cust in r:
                return idx
        return -1
    
    for s, ci, cj in savings:
        if len(routes) <= num_vehicles:
            break
        ri = find_route_idx(ci, routes)
        rj = find_route_idx(cj, routes)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Try all merge orientations
        merged = None
        if route_i[-1] == ci and route_j[0] == cj:
            merged = route_i + route_j
        elif route_i[0] == ci and route_j[-1] == cj:
            merged = route_j + route_i
        elif route_i[-1] == ci and route_j[-1] == cj:
            merged = route_i + route_j[::-1]
        elif route_i[0] == ci and route_j[0] == cj:
            merged = route_i[::-1] + route_j
        
        if merged is None:
            continue
        
        if compute_worst_case_demand(merged, nominal, deviation, Gamma) <= capacity:
            routes[ri] = merged
            routes.pop(rj)
    
    # Force merge remaining routes if needed
    while len(routes) > num_vehicles:
        best = None
        best_cost = float('inf')
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                for ri_rev in [False, True]:
                    for rj_rev in [False, True]:
                        r_i = routes[i][::-1] if ri_rev else routes[i]
                        r_j = routes[j][::-1] if rj_rev else routes[j]
                        for merged in [r_i + r_j, r_j + r_i]:
                            if compute_worst_case_demand(merged, nominal, deviation, Gamma) <= capacity:
                                c = route_cost(merged, dist, depot)
                                if c < best_cost:
                                    best_cost = c
                                    best = (i, j, merged)
        if best is None:
            return None
        i, j, merged = best
        routes[i] = merged
        routes.pop(j)
    
    return routes if len(routes) == num_vehicles else None

def nearest_neighbor_construct(customer_ids, dist, depot, nominal, deviation, Gamma, capacity, num_vehicles, seed=42):
    """Nearest neighbor construction with randomized customer ordering."""
    rng = random.Random(seed)
    unvisited = list(customer_ids)
    rng.shuffle(unvisited)
    routes = []
    
    for _ in range(num_vehicles):
        if not unvisited:
            routes.append([])
            continue
        route = []
        current = depot
        while unvisited:
            candidates = sorted(unvisited, key=lambda c: dist[current][c])
            added = False
            for c in candidates:
                test_route = route + [c]
                if compute_worst_case_demand(test_route, nominal, deviation, Gamma) <= capacity:
                    route.append(c)
                    current = c
                    unvisited.remove(c)
                    added = True
                    break
            if not added:
                break
        routes.append(route)
    
    if unvisited:
        return None
    if any(len(r) == 0 for r in routes):
        return None
    return routes

def two_opt(route, dist, depot):
    """Apply 2-opt improvement to a single route."""
    if len(route) <= 2:
        return route
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_i, best_j = -1, -1
        current_cost = route_cost(route, dist, depot)
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                gain = current_cost - route_cost(new_route, dist, depot)
                if gain > best_gain + 1e-10:
                    best_gain = gain
                    best_i, best_j = i, j
        if best_i >= 0:
            route = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
            improved = True
    return route

def local_search(routes, dist, depot, nominal, deviation, Gamma, capacity, time_limit, start_time):
    """Apply local search operators: 2-opt, relocate, swap, or-opt."""
    def time_left():
        return time_limit - (time.time() - start_time)
    
    # 2-opt on each route
    for i in range(len(routes)):
        if time_left() < 0.5:
            return routes
        routes[i] = two_opt(routes[i], dist, depot)
    
    # Iterative improvement with relocate, swap, and or-opt
    for _ in range(50):  # max iterations
        if time_left() < 1.0:
            break
        improved = False
        
        # Relocate: move one customer from one route to another
        for i in range(len(routes)):
            if time_left() < 0.5:
                return routes
            ci = 0
            while ci < len(routes[i]):
                customer = routes[i][ci]
                old_cost_i = route_cost(routes[i], dist, depot)
                best_gain = 0
                best_move = None
                
                for j in range(len(routes)):
                    if i == j:
                        continue
                    for pos in range(len(routes[j]) + 1):
                        new_route_j = routes[j][:pos] + [customer] + routes[j][pos:]
                        if compute_worst_case_demand(new_route_j, nominal, deviation, Gamma) > capacity:
                            continue
                        new_route_i = routes[i][:ci] + routes[i][ci+1:]
                        old_cost = old_cost_i + route_cost(routes[j], dist, depot)
                        new_cost = route_cost(new_route_i, dist, depot) + route_cost(new_route_j, dist, depot)
                        gain = old_cost - new_cost
                        if gain > best_gain + 1e-10:
                            best_gain = gain
                            best_move = (j, pos)
                
                if best_move:
                    j, pos = best_move
                    customer = routes[i].pop(ci)
                    routes[j] = routes[j][:pos] + [customer] + routes[j][pos:]
                    improved = True
                else:
                    ci += 1
        
        # Swap between routes
        for i in range(len(routes)):
            if time_left() < 0.5:
                return routes
            found = False
            for j in range(i + 1, len(routes)):
                for ci in range(len(routes[i])):
                    for cj in range(len(routes[j])):
                        cust_i = routes[i][ci]
                        cust_j = routes[j][cj]
                        new_ri = routes[i][:ci] + [cust_j] + routes[i][ci+1:]
                        new_rj = routes[j][:cj] + [cust_i] + routes[j][cj+1:]
                        if compute_worst_case_demand(new_ri, nominal, deviation, Gamma) > capacity:
                            continue
                        if compute_worst_case_demand(new_rj, nominal, deviation, Gamma) > capacity:
                            continue
                        old_cost = route_cost(routes[i], dist, depot) + route_cost(routes[j], dist, depot)
                        new_cost = route_cost(new_ri, dist, depot) + route_cost(new_rj, dist, depot)
                        if new_cost < old_cost - 1e-10:
                            routes[i] = new_ri
                            routes[j] = new_rj
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        # Or-opt: move segments of 2 or 3
        for seg_len in [2, 3]:
            if time_left() < 0.5:
                return routes
            found = False
            for i in range(len(routes)):
                if len(routes[i]) < seg_len:
                    continue
                for ci in range(len(routes[i]) - seg_len + 1):
                    segment = routes[i][ci:ci+seg_len]
                    remaining = routes[i][:ci] + routes[i][ci+seg_len:]
                    
                    best_gain = 0
                    best_move = None
                    
                    # Try inserting in same route
                    for pos in range(len(remaining) + 1):
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        if new_route == routes[i]:
                            continue
                        gain = route_cost(routes[i], dist, depot) - route_cost(new_route, dist, depot)
                        if gain > best_gain + 1e-10:
                            best_gain = gain
                            best_move = (i, pos)
                    
                    # Try inserting in other routes
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        for pos in range(len(routes[j]) + 1):
                            new_route_j = routes[j][:pos] + segment + routes[j][pos:]
                            if compute_worst_case_demand(new_route_j, nominal, deviation, Gamma) > capacity:
                                continue
                            old_cost = route_cost(routes[i], dist, depot) + route_cost(routes[j], dist, depot)
                            new_cost = route_cost(remaining, dist, depot) + route_cost(new_route_j, dist, depot)
                            gain = old_cost - new_cost
                            if gain > best_gain + 1e-10:
                                best_gain = gain
                                best_move = (j, pos)
                    
                    if best_move:
                        j, pos = best_move
                        segment = routes[i][ci:ci+seg_len]
                        remaining = routes[i][:ci] + routes[i][ci+seg_len:]
                        if j == i:
                            routes[i] = remaining[:pos] + segment + remaining[pos:]
                        else:
                            routes[i] = remaining
                            routes[j] = routes[j][:pos] + segment + routes[j][pos:]
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if not improved:
            break
    
    return routes

def solve_with_gurobi(data, time_remaining, initial_routes=None):
    """Solve using Gurobi MIP for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    num_customers = data['num_customers']
    num_vehicles = data['num_vehicles']
    capacity = data['vehicle_capacity']
    Gamma = data['uncertainty_set']['Gamma']
    depot_id = data['depot']['id']
    customers = data['customers']
    customer_ids = [c['id'] for c in customers]
    nominal = {c['id']: c['nominal_demand'] for c in customers}
    deviation = {c['id']: c['demand_deviation'] for c in customers}
    dist = data['distance_matrix']
    
    n = num_customers
    K = num_vehicles
    nodes = [depot_id] + customer_ids
    
    try:
        model = gp.Model("RVRP")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, time_remaining))
        model.setParam('OutputFlag', 0)
        
        # Decision variables
        x = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    for k in range(K):
                        x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
        
        y = {}
        for i in customer_ids:
            for k in range(K):
                y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
        
        u = {}
        for i in customer_ids:
            u[i] = model.addVar(lb=1, ub=n, vtype=GRB.CONTINUOUS, name=f"u_{i}")
        
        # Robust capacity variables
        lam = {}
        for k in range(K):
            max_dev = max(deviation.values()) if deviation else 0
            lam[k] = model.addVar(lb=0, ub=max_dev, vtype=GRB.CONTINUOUS, name=f"lam_{k}")
        
        s_var = {}
        for i in customer_ids:
            for k in range(K):
                s_var[i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"s_{i}_{k}")
        
        t_var = {}
        for i in customer_ids:
            for k in range(K):
                t_var[i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")
        
        model.update()
        
        # Objective
        obj = gp.quicksum(dist[i][j] * x[i, j, k]
                         for i in nodes for j in nodes if i != j for k in range(K))
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Each customer assigned to exactly one vehicle
        for i in customer_ids:
            model.addConstr(gp.quicksum(y[i, k] for k in range(K)) == 1)
        
        # Link y to incoming arcs
        for i in customer_ids:
            for k in range(K):
                model.addConstr(y[i, k] == gp.quicksum(x[j, i, k] for j in nodes if j != i))
        
        # Flow conservation
        for k in range(K):
            for i in nodes:
                model.addConstr(
                    gp.quicksum(x[i, j, k] for j in nodes if j != i) ==
                    gp.quicksum(x[j, i, k] for j in nodes if j != i))
        
        # Vehicle depot constraints
        for k in range(K):
            model.addConstr(gp.quicksum(x[depot_id, j, k] for j in customer_ids) <= 1)
            model.addConstr(gp.quicksum(x[j, depot_id, k] for j in customer_ids) <= 1)
            model.addConstr(gp.quicksum(y[i, k] for i in customer_ids) >= 1)
        
        # MTZ subtour elimination
        for i in customer_ids:
            for j in customer_ids:
                if i != j:
                    for k in range(K):
                        model.addConstr(u[i] - u[j] + n * x[i, j, k] <= n - 1)
        
        # Robust capacity constraints
        for k in range(K):
            model.addConstr(
                gp.quicksum(t_var[i, k] for i in customer_ids) + lam[k] * Gamma <= capacity)
            for i in customer_ids:
                model.addConstr(s_var[i, k] >= deviation[i] - lam[k])
                big_M = nominal[i] + deviation[i]
                model.addConstr(t_var[i, k] <= big_M * y[i, k])
                model.addConstr(t_var[i, k] <= nominal[i] + s_var[i, k])
                model.addConstr(t_var[i, k] >= nominal[i] + s_var[i, k] - big_M * (1 - y[i, k]))
        
        # Warm start
        if initial_routes is not None:
            try:
                for k, route in enumerate(initial_routes):
                    if k >= K:
                        break
                    for i in customer_ids:
                        y[i, k].Start = 1 if i in route else 0
                    if route:
                        x[depot_id, route[0], k].Start = 1
                        for idx in range(len(route) - 1):
                            x[route[idx], route[idx+1], k].Start = 1
                        x[route[-1], depot_id, k].Start = 1
                        for idx, cust in enumerate(route):
                            u[cust].Start = idx + 1
            except:
                pass
        
        model.optimize()
        
        if model.SolCount > 0:
            routes = []
            for k in range(K):
                route = []
                current = depot_id
                visited = set()
                while len(visited) <= n:
                    next_node = None
                    for j in nodes:
                        if j != current and (current, j, k) in x and x[current, j, k].X > 0.5:
                            next_node = j
                            break
                    if next_node is None or next_node == depot_id:
                        break
                    if next_node in visited:
                        break
                    route.append(next_node)
                    visited.add(next_node)
                    current = next_node
                routes.append(route)
            
            # Verify
            all_custs = set()
            for r in routes:
                all_custs.update(r)
            if all_custs == set(customer_ids) and len(routes) == K:
                feasible = all(
                    compute_worst_case_demand(r, nominal, deviation, Gamma) <= capacity
                    for r in routes)
                if feasible:
                    return routes
    except Exception:
        pass
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Robust VRP Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    time_limit = args.time_limit
    
    num_customers = data['num_customers']
    num_vehicles = data['num_vehicles']
    capacity = data['vehicle_capacity']
    Gamma = data['uncertainty_set']['Gamma']
    depot_id = data['depot']['id']
    customers = data['customers']
    customer_ids = [c['id'] for c in customers]
    nominal = {c['id']: c['nominal_demand'] for c in customers}
    deviation = {c['id']: c['demand_deviation'] for c in customers}
    dist = data['distance_matrix']
    
    best_routes = None
    best_cost = float('inf')
    
    def update_best(routes):
        nonlocal best_routes, best_cost
        if routes is None:
            return
        if len(routes) != num_vehicles:
            return
        all_custs = set()
        for r in routes:
            if not r:
                return
            all_custs.update(r)
            if compute_worst_case_demand(r, nominal, deviation, Gamma) > capacity:
                return
        if all_custs != set(customer_ids):
            return
        cost = total_cost(routes, dist, depot_id)
        if cost < best_cost:
            best_cost = cost
            best_routes = deepcopy(routes)
            solution = format_solution(best_routes, dist, depot_id, nominal, deviation, Gamma)
            if logger:
                logger.log_solution(best_cost, solution)
    
    # Phase 1: Clarke-Wright savings + local search
    routes = clarke_wright_savings(customer_ids, dist, depot_id, nominal, deviation, Gamma, capacity, num_vehicles)
    if routes:
        routes = local_search(routes, dist, depot_id, nominal, deviation, Gamma, capacity,
                             time_limit * 0.3, start_time)
        update_best(routes)
    
    # Phase 2: Multiple nearest neighbor constructions + local search
    for seed in range(20):
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.6:
            break
        routes = nearest_neighbor_construct(customer_ids, dist, depot_id, nominal, deviation,
                                           Gamma, capacity, num_vehicles, seed=seed)
        if routes:
            routes = local_search(routes, dist, depot_id, nominal, deviation, Gamma, capacity,
                                 time_limit * 0.7, start_time)
            update_best(routes)
    
    # Phase 3: MIP for small instances
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if num_customers <= 25 and remaining > 10:
        mip_routes = solve_with_gurobi(data, remaining - 3, best_routes)
        if mip_routes:
            update_best(mip_routes)
            if time.time() - start_time < time_limit - 3:
                mip_routes = local_search(mip_routes, dist, depot_id, nominal, deviation,
                                         Gamma, capacity, time_limit, start_time)
                update_best(mip_routes)
    
    # Write final solution
    if best_routes is not None:
        solution = format_solution(best_routes, dist, depot_id, nominal, deviation, Gamma)
    else:
        solution = {"objective_value": 0.0, "routes": []}
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()