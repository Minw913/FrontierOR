import argparse
import json
import math
import time
import random
from copy import deepcopy

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_customers = data['num_customers']
    vehicle_capacity = data['vehicle_capacity']
    num_vehicles_ub = data['num_vehicles_upper_bound']
    depot = data['depot']
    customers = data['customers']
    dist_matrix = data['distance_matrix']
    split_allowed = data.get('split_deliveries_allowed', True)
    
    cust_by_id = {}
    for c in customers:
        cust_by_id[c['id']] = c
    
    depot_tw = depot['time_window']
    
    def get_service_time(node):
        if node == 0:
            return depot['service_time']
        return cust_by_id[node]['service_time']
    
    def get_tw(node):
        if node == 0:
            return depot['time_window']
        return cust_by_id[node]['time_window']
    
    def arc_travel_time(i, j):
        return dist_matrix[i][j] + get_service_time(i)
    
    def compute_route_cost(route):
        if not route:
            return 0.0
        cost = dist_matrix[0][route[0]]
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i]][route[i+1]]
        cost += dist_matrix[route[-1]][0]
        return cost
    
    def check_route_time_feasibility(route):
        if not route:
            return True, []
        
        current_time = depot_tw[0]
        schedule = []
        prev = 0
        for node in route:
            arrival = current_time + arc_travel_time(prev, node)
            tw = get_tw(node)
            service_start = max(arrival, tw[0])
            if service_start > tw[1]:
                return False, []
            schedule.append(service_start)
            current_time = service_start
            prev = node
        
        arrival_depot = current_time + arc_travel_time(prev, 0)
        if arrival_depot > depot_tw[1]:
            return False, []
        
        return True, schedule
    
    customer_ids = [c['id'] for c in customers]
    demands = {c['id']: c['demand'] for c in customers}
    
    sorted_customers = sorted(customer_ids, key=lambda c: (get_tw(c)[0], get_tw(c)[1]))
    
    def greedy_construct(customer_order, remaining_demand=None):
        if remaining_demand is None:
            remaining_demand = {c: demands[c] for c in customer_ids}
        else:
            remaining_demand = dict(remaining_demand)
        
        routes = []
        route_deliveries = []
        
        unserved = [c for c in customer_order if remaining_demand.get(c, 0) > 0]
        max_iterations = num_customers * 3
        iteration = 0
        
        while unserved and iteration < max_iterations:
            iteration += 1
            route = []
            deliv = {}
            cap_left = vehicle_capacity
            current_time = depot_tw[0]
            prev = 0
            
            candidates = list(unserved)
            
            while candidates:
                best_cust = None
                best_cost = float('inf')
                
                for c in candidates:
                    if remaining_demand.get(c, 0) <= 0:
                        continue
                    
                    arrival = current_time + arc_travel_time(prev, c)
                    tw = get_tw(c)
                    service_start = max(arrival, tw[0])
                    if service_start > tw[1]:
                        continue
                    
                    return_time = service_start + arc_travel_time(c, 0)
                    if return_time > depot_tw[1]:
                        continue
                    
                    if cap_left <= 0:
                        continue
                    
                    cost = dist_matrix[prev][c]
                    if cost < best_cost:
                        best_cost = cost
                        best_cust = c
                
                if best_cust is None:
                    break
                
                c = best_cust
                arrival = current_time + arc_travel_time(prev, c)
                tw = get_tw(c)
                service_start = max(arrival, tw[0])
                
                qty = min(remaining_demand[c], cap_left)
                if qty <= 0:
                    candidates.remove(c)
                    continue
                
                route.append(c)
                deliv[c] = qty
                remaining_demand[c] -= qty
                cap_left -= qty
                current_time = service_start
                prev = c
                
                if remaining_demand[c] <= 0:
                    candidates.remove(c)
                if cap_left <= 0:
                    break
            
            if route:
                routes.append(route)
                route_deliveries.append(deliv)
            
            new_unserved = [c for c in customer_order if remaining_demand.get(c, 0) > 0]
            if not route and new_unserved:
                for c in list(new_unserved):
                    if remaining_demand[c] > 0:
                        feasible, _ = check_route_time_feasibility([c])
                        if feasible:
                            qty = min(remaining_demand[c], vehicle_capacity)
                            routes.append([c])
                            route_deliveries.append({c: qty})
                            remaining_demand[c] -= qty
                
                new_unserved = [c for c in customer_order if remaining_demand.get(c, 0) > 0]
                if new_unserved == unserved:
                    break
            unserved = new_unserved
        
        return routes, route_deliveries, remaining_demand
    
    def total_cost(routes):
        return sum(compute_route_cost(r) for r in routes)
    
    best_routes, best_deliveries, remaining = greedy_construct(sorted_customers)
    best_cost = total_cost(best_routes)
    
    if logger:
        logger.log(best_cost)
    
    def try_construction(order):
        nonlocal best_routes, best_deliveries, best_cost
        routes, deliveries, rem = greedy_construct(order)
        if all(rem.get(c, 0) <= 0 for c in customer_ids):
            cost = total_cost(routes)
            if cost < best_cost:
                best_routes = routes
                best_deliveries = deliveries
                best_cost = cost
                if logger:
                    logger.log(best_cost)
                return True
        return False
    
    orderings = [
        sorted(customer_ids, key=lambda c: get_tw(c)[1]),
        sorted(customer_ids, key=lambda c: get_tw(c)[0]),
        sorted(customer_ids, key=lambda c: dist_matrix[0][c]),
        sorted(customer_ids, key=lambda c: -dist_matrix[0][c]),
        sorted(customer_ids, key=lambda c: -demands[c]),
        sorted(customer_ids, key=lambda c: get_tw(c)[1] - get_tw(c)[0]),
    ]
    
    for order in orderings:
        if time.time() - start_time > time_limit * 0.1:
            break
        try_construction(order)
    
    def get_route_load(route, deliv):
        return sum(deliv.get(c, 0) for c in route)
    
    def remove_empty_routes(routes, route_deliveries):
        new_routes = []
        new_deliveries = []
        for i in range(len(routes)):
            if routes[i]:
                new_routes.append(routes[i])
                new_deliveries.append(route_deliveries[i])
        return new_routes, new_deliveries
    
    def try_relocate(routes, route_deliveries):
        for r1_idx in range(len(routes)):
            for pos1 in range(len(routes[r1_idx])):
                cust = routes[r1_idx][pos1]
                qty1 = route_deliveries[r1_idx].get(cust, 0)
                
                r1 = routes[r1_idx]
                old_cost_r1 = compute_route_cost(r1)
                
                if len(r1) == 1:
                    new_r1 = []
                    new_cost_r1 = 0
                else:
                    new_r1 = r1[:pos1] + r1[pos1+1:]
                    feasible, _ = check_route_time_feasibility(new_r1)
                    if not feasible:
                        continue
                    new_cost_r1 = compute_route_cost(new_r1)
                
                saving_r1 = old_cost_r1 - new_cost_r1
                
                for r2_idx in range(len(routes)):
                    if r2_idx == r1_idx:
                        continue
                    
                    r2 = routes[r2_idx]
                    current_load_r2 = get_route_load(r2, route_deliveries[r2_idx])
                    if current_load_r2 + qty1 > vehicle_capacity:
                        continue
                    
                    old_cost_r2 = compute_route_cost(r2)
                    
                    best_insert_pos = -1
                    best_insert_cost = float('inf')
                    
                    for pos2 in range(len(r2) + 1):
                        new_r2 = r2[:pos2] + [cust] + r2[pos2:]
                        feasible, _ = check_route_time_feasibility(new_r2)
                        if not feasible:
                            continue
                        new_cost_r2 = compute_route_cost(new_r2)
                        added_cost = new_cost_r2 - old_cost_r2
                        
                        if saving_r1 - added_cost > 0.01:
                            if added_cost < best_insert_cost:
                                best_insert_cost = added_cost
                                best_insert_pos = pos2
                    
                    if best_insert_pos >= 0:
                        new_r2 = r2[:best_insert_pos] + [cust] + r2[best_insert_pos:]
                        routes[r1_idx] = new_r1
                        routes[r2_idx] = new_r2
                        
                        if cust in route_deliveries[r1_idx]:
                            del route_deliveries[r1_idx][cust]
                        route_deliveries[r2_idx][cust] = route_deliveries[r2_idx].get(cust, 0) + qty1
                        
                        return True
        
        return False
    
    def try_2opt_within(routes, route_deliveries):
        improved = False
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 3:
                continue
            
            best_improvement = 0
            best_i = -1
            best_j = -1
            
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _ = check_route_time_feasibility(new_route)
                    if not feasible:
                        continue
                    old_cost = compute_route_cost(route)
                    new_cost = compute_route_cost(new_route)
                    improvement = old_cost - new_cost
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i = i
                        best_j = j
            
            if best_i >= 0:
                routes[r_idx] = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                improved = True
        
        return improved
    
    def try_or_opt(routes, route_deliveries):
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 2:
                continue
            
            for seg_len in [1, 2, 3]:
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i+seg_len]
                    remaining_route = route[:i] + route[i+seg_len:]
                    
                    old_cost = compute_route_cost(route)
                    
                    for j in range(len(remaining_route) + 1):
                        if j == i:
                            continue
                        new_route = remaining_route[:j] + segment + remaining_route[j:]
                        feasible, _ = check_route_time_feasibility(new_route)
                        if not feasible:
                            continue
                        new_cost = compute_route_cost(new_route)
                        if new_cost < old_cost - 0.01:
                            routes[r_idx] = new_route
                            return True
        
        return False
    
    def try_swap(routes, route_deliveries):
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)):
                r1 = routes[r1_idx]
                r2 = routes[r2_idx]
                
                for i in range(len(r1)):
                    for j in range(len(r2)):
                        c1 = r1[i]
                        c2 = r2[j]
                        
                        load1 = get_route_load(r1, route_deliveries[r1_idx])
                        load2 = get_route_load(r2, route_deliveries[r2_idx])
                        
                        q1 = route_deliveries[r1_idx].get(c1, 0)
                        q2 = route_deliveries[r2_idx].get(c2, 0)
                        
                        new_load1 = load1 - q1 + q2
                        new_load2 = load2 - q2 + q1
                        
                        if new_load1 > vehicle_capacity or new_load2 > vehicle_capacity:
                            continue
                        
                        new_r1 = r1[:i] + [c2] + r1[i+1:]
                        new_r2 = r2[:j] + [c1] + r2[j+1:]
                        
                        f1, _ = check_route_time_feasibility(new_r1)
                        if not f1:
                            continue
                        f2, _ = check_route_time_feasibility(new_r2)
                        if not f2:
                            continue
                        
                        old_cost = compute_route_cost(r1) + compute_route_cost(r2)
                        new_cost = compute_route_cost(new_r1) + compute_route_cost(new_r2)
                        
                        if new_cost < old_cost - 0.01:
                            routes[r1_idx] = new_r1
                            routes[r2_idx] = new_r2
                            
                            # Update deliveries
                            new_d1 = dict(route_deliveries[r1_idx])
                            new_d2 = dict(route_deliveries[r2_idx])
                            
                            if c1 in new_d1:
                                del new_d1[c1]
                            if c2 in new_d2:
                                del new_d2[c2]
                            
                            new_d1[c2] = q2
                            new_d2[c1] = q1
                            
                            route_deliveries[r1_idx] = new_d1
                            route_deliveries[r2_idx] = new_d2
                            
                            return True
        
        return False
    
    def local_search(routes, route_deliveries, time_budget):
        ls_start = time.time()
        # Deep copy to avoid side effects
        routes = [list(r) for r in routes]
        route_deliveries = [dict(d) for d in route_deliveries]
        
        improved = True
        while improved and (time.time() - ls_start) < time_budget:
            improved = False
            
            try:
                if try_2opt_within(routes, route_deliveries):
                    improved = True
                    routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                    continue
            except Exception:
                pass
            
            if (time.time() - ls_start) >= time_budget:
                break
            
            try:
                if try_or_opt(routes, route_deliveries):
                    improved = True
                    routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                    continue
            except Exception:
                pass
            
            if (time.time() - ls_start) >= time_budget:
                break
            
            try:
                if try_relocate(routes, route_deliveries):
                    improved = True
                    routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                    continue
            except Exception:
                pass
            
            if (time.time() - ls_start) >= time_budget:
                break
            
            try:
                if try_swap(routes, route_deliveries):
                    improved = True
                    routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                    continue
            except Exception:
                pass
        
        return routes, route_deliveries
    
    time_for_ls = min(time_limit * 0.4, time_limit - (time.time() - start_time) - 2)
    if time_for_ls > 0:
        best_routes, best_deliveries = local_search(best_routes, best_deliveries, time_for_ls)
        best_routes, best_deliveries = remove_empty_routes(best_routes, best_deliveries)
        new_cost = total_cost(best_routes)
        if new_cost < best_cost:
            best_cost = new_cost
            if logger:
                logger.log(best_cost)
    
    def perturb(routes, route_deliveries):
        routes = [list(r) for r in routes]
        route_deliveries = [dict(d) for d in route_deliveries]
        
        if not routes:
            return routes, route_deliveries
        
        all_custs = []
        for r_idx, r in enumerate(routes):
            for c in r:
                all_custs.append((r_idx, c))
        
        if not all_custs:
            return routes, route_deliveries
        
        num_remove = min(random.randint(2, max(2, min(5, len(all_custs) // 3))), len(all_custs))
        to_remove = random.sample(all_custs, num_remove)
        
        removed_custs = []
        for r_idx, c in to_remove:
            if r_idx < len(routes) and c in routes[r_idx]:
                qty = route_deliveries[r_idx].pop(c, 0)
                routes[r_idx].remove(c)
                removed_custs.append((c, qty))
        
        routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
        
        for c, qty in removed_custs:
            best_insert_route = -1
            best_insert_pos = -1
            best_insert_cost_increase = float('inf')
            
            for r_idx in range(len(routes)):
                r = routes[r_idx]
                current_load = get_route_load(r, route_deliveries[r_idx])
                if current_load + qty > vehicle_capacity:
                    continue
                
                old_cost = compute_route_cost(r)
                for pos in range(len(r) + 1):
                    new_r = r[:pos] + [c] + r[pos:]
                    feasible, _ = check_route_time_feasibility(new_r)
                    if not feasible:
                        continue
                    new_cost = compute_route_cost(new_r)
                    increase = new_cost - old_cost
                    if increase < best_insert_cost_increase:
                        best_insert_cost_increase = increase
                        best_insert_route = r_idx
                        best_insert_pos = pos
            
            new_route_cost = dist_matrix[0][c] + dist_matrix[c][0]
            feasible_new, _ = check_route_time_feasibility([c])
            
            if feasible_new and new_route_cost < best_insert_cost_increase:
                routes.append([c])
                route_deliveries.append({c: qty})
            elif best_insert_route >= 0:
                r = routes[best_insert_route]
                routes[best_insert_route] = r[:best_insert_pos] + [c] + r[best_insert_pos:]
                route_deliveries[best_insert_route][c] = route_deliveries[best_insert_route].get(c, 0) + qty
            elif feasible_new:
                routes.append([c])
                route_deliveries.append({c: qty})
        
        return routes, route_deliveries
    
    # Iterated local search
    current_routes = [list(r) for r in best_routes]
    current_deliveries = [dict(d) for d in best_deliveries]
    current_cost = best_cost
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit - 3:
        try:
            p_routes, p_deliveries = perturb(current_routes, current_deliveries)
            
            ls_time = min(2.0, (time_limit - (time.time() - start_time) - 2) * 0.3)
            if ls_time <= 0:
                break
            
            p_routes, p_deliveries = local_search(p_routes, p_deliveries, ls_time)
            p_routes, p_deliveries = remove_empty_routes(p_routes, p_deliveries)
            p_cost = total_cost(p_routes)
            
            if p_cost < current_cost - 0.01:
                current_routes = p_routes
                current_deliveries = p_deliveries
                current_cost = p_cost
                no_improve_count = 0
                
                if p_cost < best_cost - 0.01:
                    best_routes = [list(r) for r in p_routes]
                    best_deliveries = [dict(d) for d in p_deliveries]
                    best_cost = p_cost
                    if logger:
                        logger.log(best_cost)
            else:
                no_improve_count += 1
                if random.random() < 0.3:
                    current_routes = p_routes
                    current_deliveries = p_deliveries
                    current_cost = p_cost
            
            if no_improve_count >= max_no_improve:
                current_routes = [list(r) for r in best_routes]
                current_deliveries = [dict(d) for d in best_deliveries]
                current_cost = best_cost
                no_improve_count = 0
        except Exception:
            # If anything goes wrong during ILS iteration, continue
            continue
    
    # Verify all demands are met and fix if needed
    delivered = {}
    for r_idx, route in enumerate(best_routes):
        for c in route:
            delivered[c] = delivered.get(c, 0) + best_deliveries[r_idx].get(c, 0)
    
    for c_id in customer_ids:
        if delivered.get(c_id, 0) < demands[c_id]:
            shortfall = demands[c_id] - delivered.get(c_id, 0)
            # Try to add to existing routes that visit this customer
            for r_idx, route in enumerate(best_routes):
                if c_id in route:
                    current_load = get_route_load(route, best_deliveries[r_idx])
                    can_add = min(shortfall, vehicle_capacity - current_load)
                    if can_add > 0:
                        best_deliveries[r_idx][c_id] = best_deliveries[r_idx].get(c_id, 0) + can_add
                        shortfall -= can_add
                        if shortfall <= 0:
                            break
            
            if shortfall > 0:
                while shortfall > 0:
                    qty = min(shortfall, vehicle_capacity)
                    feasible, _ = check_route_time_feasibility([c_id])
                    if feasible:
                        best_routes.append([c_id])
                        best_deliveries.append({c_id: qty})
                        shortfall -= qty
                    else:
                        break
    
    best_cost = total_cost(best_routes)
    
    # Build output
    routes_output = []
    deliveries_output = {}
    
    for c in customers:
        deliveries_output[str(c['id'])] = {
            "demand": c['demand'],
            "total_delivered": 0.0
        }
    
    num_vehicles_used = 0
    for r_idx, route in enumerate(best_routes):
        if not route:
            continue
        num_vehicles_used += 1
        
        route_with_depot = [0] + route + [num_customers + 1]
        
        deliv_dict = {}
        for c in route:
            qty = best_deliveries[r_idx].get(c, 0)
            deliv_dict[str(c)] = qty
            deliveries_output[str(c)]["total_delivered"] += qty
        
        routes_output.append({
            "vehicle": r_idx + 1,
            "route": route_with_depot,
            "deliveries": deliv_dict
        })
    
    solution = {
        "objective_value": round(best_cost, 1),
        "num_vehicles_used": num_vehicles_used,
        "routes": routes_output,
        "deliveries": deliveries_output
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)