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
    
    # Build customer info indexed by id
    cust_by_id = {}
    for c in customers:
        cust_by_id[c['id']] = c
    
    depot_tw = depot['time_window']
    
    # Precompute feasible arcs considering time windows
    # Nodes: 0 = depot, 1..num_customers = customers
    # For travel time: includes service time at origin
    # travel_time(i, j) = dist(i, j) + service_time(i)
    # But problem says "The travel time along an arc from one location to another includes the service time at the origin location."
    # So effective_travel_time(i, j) = dist(i, j) + service_time(i) ... wait, let me re-read.
    # "Travel times equal distances" but "The travel time along an arc from one location to another includes the service time at the origin location."
    # This means: when computing arrival time, time(j) >= time(i) + service_time(i) + dist(i, j)
    # But the "travel time" on an arc already includes the service time at origin.
    # So travel_time_arc(i,j) = service_time(i) + dist(i,j)
    # And the constraint is: time(j) >= time(i) + travel_time_arc(i,j)
    
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
    
    def is_arc_feasible(i, j):
        """Check if arc from i to j is feasible: starting service at i at earliest time of i,
        arriving at j no later than end of j's time window."""
        tw_i = get_tw(i)
        tw_j = get_tw(j)
        earliest_arrival = tw_i[0] + arc_travel_time(i, j)
        return earliest_arrival <= tw_j[1]
    
    # For the return to depot, we consider depot's time window end
    # Node num_customers+1 is the return depot (same location as 0)
    
    # Build a nearest neighbor / savings based heuristic
    
    def compute_route_cost(route):
        """Route is list of customer ids (not including depot)"""
        if not route:
            return 0.0
        cost = dist_matrix[0][route[0]]
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i]][route[i+1]]
        cost += dist_matrix[route[-1]][0]
        return cost
    
    def check_route_time_feasibility(route):
        """Check if route is time-window feasible. Returns (feasible, schedule)"""
        if not route:
            return True, []
        
        current_time = depot_tw[0]
        schedule = []
        
        # Depot to first customer
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
        
        # Return to depot
        arrival_depot = current_time + arc_travel_time(prev, 0)
        if arrival_depot > depot_tw[1]:
            return False, []
        
        return True, schedule
    
    def check_route_capacity(route, deliveries_dict):
        """Check if total deliveries on route <= vehicle_capacity"""
        total = sum(deliveries_dict.get(c, 0) for c in route)
        return total <= vehicle_capacity
    
    # Strategy: Use a greedy construction followed by local search
    # Since split deliveries are allowed, we can be more flexible
    
    # First, try a simple approach: treat each customer as needing full demand delivery
    # Build routes using nearest neighbor heuristic, then improve
    
    customer_ids = [c['id'] for c in customers]
    demands = {c['id']: c['demand'] for c in customers}
    
    # Sort customers by time window start
    sorted_customers = sorted(customer_ids, key=lambda c: (get_tw(c)[0], get_tw(c)[1]))
    
    def greedy_construct(customer_order, remaining_demand=None):
        """Construct routes greedily"""
        if remaining_demand is None:
            remaining_demand = {c: demands[c] for c in customer_ids}
        else:
            remaining_demand = dict(remaining_demand)
        
        routes = []
        route_deliveries = []
        
        unserved = [c for c in customer_order if remaining_demand.get(c, 0) > 0]
        
        while unserved:
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
                    if remaining_demand[c] <= 0:
                        continue
                    
                    # Check time feasibility
                    arrival = current_time + arc_travel_time(prev, c)
                    tw = get_tw(c)
                    service_start = max(arrival, tw[0])
                    if service_start > tw[1]:
                        continue
                    
                    # Check can return to depot after
                    return_time = service_start + arc_travel_time(c, 0)
                    if return_time > depot_tw[1]:
                        continue
                    
                    # Check capacity
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
            
            unserved = [c for c in customer_order if remaining_demand.get(c, 0) > 0]
            if not route and unserved:
                # Try individual routes for remaining
                for c in unserved:
                    if remaining_demand[c] > 0:
                        # Check direct feasibility
                        arrival = depot_tw[0] + arc_travel_time(0, c)
                        tw = get_tw(c)
                        ss = max(arrival, tw[0])
                        if ss <= tw[1]:
                            ret = ss + arc_travel_time(c, 0)
                            if ret <= depot_tw[1]:
                                qty = min(remaining_demand[c], vehicle_capacity)
                                routes.append([c])
                                route_deliveries.append({c: qty})
                                remaining_demand[c] -= qty
                
                unserved = [c for c in customer_order if remaining_demand.get(c, 0) > 0]
                if unserved:
                    # Force break to avoid infinite loop
                    break
        
        return routes, route_deliveries, remaining_demand
    
    def total_cost(routes):
        return sum(compute_route_cost(r) for r in routes)
    
    # Initial construction
    best_routes, best_deliveries, remaining = greedy_construct(sorted_customers)
    best_cost = total_cost(best_routes)
    
    if logger:
        logger.log(best_cost)
    
    # Try different orderings for construction
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
    
    # Try sorting by different criteria
    orderings = [
        sorted(customer_ids, key=lambda c: get_tw(c)[1]),  # earliest deadline
        sorted(customer_ids, key=lambda c: get_tw(c)[0]),  # earliest ready
        sorted(customer_ids, key=lambda c: dist_matrix[0][c]),  # closest to depot
        sorted(customer_ids, key=lambda c: -dist_matrix[0][c]),  # farthest from depot
        sorted(customer_ids, key=lambda c: -demands[c]),  # largest demand first
        sorted(customer_ids, key=lambda c: get_tw(c)[1] - get_tw(c)[0]),  # tightest window
    ]
    
    for order in orderings:
        if time.time() - start_time > time_limit * 0.1:
            break
        try_construction(order)
    
    # Local search improvement
    def try_relocate(routes, route_deliveries):
        """Try moving a customer from one route to another"""
        improved = False
        for r1_idx in range(len(routes)):
            for pos1 in range(len(routes[r1_idx])):
                cust = routes[r1_idx][pos1]
                qty1 = route_deliveries[r1_idx].get(cust, 0)
                
                # Cost of removing cust from route r1
                r1 = routes[r1_idx]
                if len(r1) == 1:
                    old_cost_r1 = compute_route_cost(r1)
                    new_cost_r1 = 0
                    new_r1 = []
                else:
                    old_cost_r1 = compute_route_cost(r1)
                    new_r1 = r1[:pos1] + r1[pos1+1:]
                    feasible, _ = check_route_time_feasibility(new_r1)
                    if not feasible:
                        continue
                    new_cost_r1 = compute_route_cost(new_r1)
                
                saving_r1 = old_cost_r1 - new_cost_r1
                
                # Try inserting into another route
                for r2_idx in range(len(routes)):
                    if r2_idx == r1_idx:
                        continue
                    
                    r2 = routes[r2_idx]
                    # Check capacity
                    current_load_r2 = sum(route_deliveries[r2_idx].get(c, 0) for c in r2)
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
                        # Perform the move
                        new_r2 = r2[:best_insert_pos] + [cust] + r2[best_insert_pos:]
                        routes[r1_idx] = new_r1
                        routes[r2_idx] = new_r2
                        
                        del route_deliveries[r1_idx][cust]
                        route_deliveries[r2_idx][cust] = route_deliveries[r2_idx].get(cust, 0) + qty1
                        
                        improved = True
                        return improved
        
        return improved
    
    def try_2opt_within(routes, route_deliveries):
        """2-opt within a single route"""
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
        """Or-opt: move a segment of 1, 2, or 3 customers within/between routes"""
        improved = False
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 2:
                continue
            
            for seg_len in [1, 2, 3]:
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i+seg_len]
                    remaining = route[:i] + route[i+seg_len:]
                    
                    old_cost = compute_route_cost(route)
                    
                    # Try reinserting segment at different positions in same route
                    for j in range(len(remaining) + 1):
                        if j == i:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        feasible, _ = check_route_time_feasibility(new_route)
                        if not feasible:
                            continue
                        new_cost = compute_route_cost(new_route)
                        if new_cost < old_cost - 0.01:
                            routes[r_idx] = new_route
                            improved = True
                            return improved
        
        return improved
    
    def try_swap(routes, route_deliveries):
        """Swap customers between two routes"""
        improved = False
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)):
                r1 = routes[r1_idx]
                r2 = routes[r2_idx]
                
                for i in range(len(r1)):
                    for j in range(len(r2)):
                        c1 = r1[i]
                        c2 = r2[j]
                        
                        # Check capacity feasibility
                        load1 = sum(route_deliveries[r1_idx].get(c, 0) for c in r1)
                        load2 = sum(route_deliveries[r2_idx].get(c, 0) for c in r2)
                        
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
                            
                            route_deliveries[r1_idx][c2] = q2
                            del route_deliveries[r1_idx][c1]
                            route_deliveries[r2_idx][c1] = q1
                            del route_deliveries[r2_idx][c2]
                            
                            return True
        
        return False
    
    def remove_empty_routes(routes, route_deliveries):
        new_routes = []
        new_deliveries = []
        for i in range(len(routes)):
            if routes[i]:
                new_routes.append(routes[i])
                new_deliveries.append(route_deliveries[i])
        return new_routes, new_deliveries
    
    # Run local search
    def local_search(routes, route_deliveries, time_budget):
        ls_start = time.time()
        improved = True
        while improved and time.time() - ls_start < time_budget:
            improved = False
            
            if try_2opt_within(routes, route_deliveries):
                improved = True
                routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                continue
            
            if try_or_opt(routes, route_deliveries):
                improved = True
                routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                continue
            
            if try_relocate(routes, route_deliveries):
                improved = True
                routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                continue
            
            if try_swap(routes, route_deliveries):
                improved = True
                routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
                continue
        
        return routes, route_deliveries
    
    time_for_ls = min(time_limit * 0.7, time_limit - (time.time() - start_time) - 1)
    if time_for_ls > 0:
        best_routes, best_deliveries = local_search(best_routes, best_deliveries, time_for_ls)
        best_routes, best_deliveries = remove_empty_routes(best_routes, best_deliveries)
        new_cost = total_cost(best_routes)
        if new_cost < best_cost:
            best_cost = new_cost
            if logger:
                logger.log(best_cost)
    
    # Perturbation + local search (iterated local search)
    def perturb(routes, route_deliveries):
        """Random perturbation: remove random customers and reinsert"""
        routes = [list(r) for r in routes]
        route_deliveries = [dict(d) for d in route_deliveries]
        
        if not routes:
            return routes, route_deliveries
        
        # Remove 2-4 random customers and reinsert
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
            if c in routes[r_idx]:
                qty = route_deliveries[r_idx].pop(c, 0)
                routes[r_idx].remove(c)
                removed_custs.append((c, qty))
        
        routes, route_deliveries = remove_empty_routes(routes, route_deliveries)
        
        # Reinsert removed customers
        for c, qty in removed_custs:
            best_insert_route = -1
            best_insert_pos = -1
            best_insert_cost_increase = float('inf')
            
            for r_idx in range(len(routes)):
                r = routes[r_idx]
                current_load = sum(route_deliveries[r_idx].get(cc, 0) for cc in r)
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
            
            # Try new route
            new_route_cost = dist_matrix[0][c] + dist_matrix[c][0]
            feasible, _ = check_route_time_feasibility([c])
            if feasible and new_route_cost < best_insert_cost_increase:
                routes.append([c])
                route_deliveries.append({c: qty})
            elif best_insert_route >= 0:
                r = routes[best_insert_route]
                routes[best_insert_route] = r[:best_insert_pos] + [c] + r[best_insert_pos:]
                route_deliveries[best_insert_route][c] = qty
            else:
                # Force new route
                if feasible:
                    routes.append([c])
                    route_deliveries.append({c: qty})
        
        return routes, route_deliveries
    
    # Iterated local search
    ils_start = time.time()
    remaining_time = time_limit - (time.time() - start_time) - 1
    
    current_routes = [list(r) for r in best_routes]
    current_deliveries = [dict(d) for d in best_deliveries]
    current_cost = best_cost
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit - 2:
        # Perturb
        p_routes, p_deliveries = perturb(current_routes, current_deliveries)
        
        # Local search with limited time
        ls_time = min(2.0, (time_limit - (time.time() - start_time) - 1) * 0.3)
        if ls_time <= 0:
            break
        
        p_routes, p_deliveries = local_search(p_routes, p_deliveries, ls_time)
        p_routes, p_deliveries = remove_empty_routes(p_routes, p_deliveries)
        p_cost = total_cost(p_routes)
        
        # Accept if better
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
            # Accept worse with some probability (simulated annealing style)
            if random.random() < 0.3:
                current_routes = p_routes
                current_deliveries = p_deliveries
                current_cost = p_cost
        
        if no_improve_count >= max_no_improve:
            # Reset to best
            current_routes = [list(r) for r in best_routes]
            current_deliveries = [dict(d) for d in best_deliveries]
            current_cost = best_cost
            no_improve_count = 0
    
    # Verify all demands are met and fix if needed
    delivered = {}
    for r_idx, route in enumerate(best_routes):
        for c in route:
            delivered[c] = delivered.get(c, 0) + best_deliveries[r_idx].get(c, 0)
    
    # Check and fix unmet demands
    for c_id in customer_ids:
        if delivered.get(c_id, 0) < demands[c_id]:
            shortfall = demands[c_id] - delivered.get(c_id, 0)
            # Try to add to existing routes that visit this customer
            added = False
            for r_idx, route in enumerate(best_routes):
                if c_id in route:
                    current_load = sum(best_deliveries[r_idx].get(cc, 0) for cc in route)
                    can_add = min(shortfall, vehicle_capacity - current_load)
                    if can_add > 0:
                        best_deliveries[r_idx][c_id] = best_deliveries[r_idx].get(c_id, 0) + can_add
                        shortfall -= can_add
                        if shortfall <= 0:
                            added = True
                            break
            
            if not added and shortfall > 0:
                # Create new route(s)
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
    
    # Initialize deliveries tracking
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