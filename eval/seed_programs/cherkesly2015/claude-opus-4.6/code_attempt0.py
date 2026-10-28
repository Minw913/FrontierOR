import argparse
import json
import math
import time
import random
from copy import deepcopy
from solution_logger import SolutionLogger


def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    n_requests = instance['n_requests']
    n_nodes = instance['n_nodes']
    vehicle_capacity = instance['vehicle_capacity']
    vehicle_fixed_cost = instance['vehicle_fixed_cost']
    
    nodes = {}
    for n in instance['nodes']:
        nodes[n['node_id']] = n
    
    # Build arc data structures
    arc_cost = {}
    arc_time = {}
    for a in instance['arcs']:
        arc_cost[(a['from'], a['to'])] = a['cost']
        arc_time[(a['from'], a['to'])] = a['travel_time']
    
    origin_depot = 0
    dest_depot = 2 * n_requests + 1
    
    pickup_nodes = list(range(1, n_requests + 1))
    delivery_nodes = list(range(n_requests + 1, 2 * n_requests + 1))
    
    def get_delivery(pickup):
        return pickup + n_requests
    
    def get_pickup(delivery):
        return delivery - n_requests
    
    def get_cost(i, j):
        if (i, j) in arc_cost:
            return arc_cost[(i, j)]
        return float('inf')
    
    def get_time(i, j):
        if (i, j) in arc_time:
            return arc_time[(i, j)]
        return float('inf')
    
    def evaluate_route(route):
        """Evaluate a route (list of nodes starting with origin depot and ending with dest depot).
        Returns (feasible, cost, arrival_times, loads) or (False, inf, None, None)."""
        if len(route) < 2:
            return False, float('inf'), None, None
        if route[0] != origin_depot or route[-1] != dest_depot:
            return False, float('inf'), None, None
        
        cost = 0.0
        current_time = nodes[origin_depot]['tw_early']
        current_load = 0
        arrival_times = [current_time]
        loads = [0]
        
        # Check LIFO: maintain a stack of request IDs
        stack = []
        
        for idx in range(1, len(route)):
            prev = route[idx - 1]
            curr = route[idx]
            
            if (prev, curr) not in arc_cost:
                return False, float('inf'), None, None
            
            c = arc_cost[(prev, curr)]
            t = arc_time[(prev, curr)]
            
            cost += c
            
            earliest_arrival = current_time + nodes[prev]['service_time'] + t
            tw_early = nodes[curr]['tw_early']
            tw_late = nodes[curr]['tw_late']
            
            service_start = max(earliest_arrival, tw_early)
            if service_start > tw_late:
                return False, float('inf'), None, None
            
            current_time = service_start
            current_load += nodes[curr]['load']
            
            if current_load < 0 or current_load > vehicle_capacity:
                return False, float('inf'), None, None
            
            # LIFO check
            node_info = nodes[curr]
            if node_info['node_type'] == 'pickup':
                stack.append(node_info['request_id'])
            elif node_info['node_type'] == 'delivery':
                req_id = node_info['request_id']
                if not stack or stack[-1] != req_id:
                    return False, float('inf'), None, None
                stack.pop()
            
            arrival_times.append(current_time)
            loads.append(current_load)
        
        # Check precedence constraints
        for i in range(1, len(route) - 1):
            ni = nodes[route[i]]
            if ni['node_type'] == 'pickup':
                req_id = ni['request_id']
                del_node = get_delivery(route[i])
                # Find delivery in route
                found = False
                for j in range(i + 1, len(route) - 1):
                    if route[j] == del_node:
                        # Check: arrival_times[i] + service_time + travel_time <= arrival_times[j]
                        direct_time = get_time(route[i], del_node)
                        if arrival_times[i] + ni['service_time'] + direct_time > arrival_times[j] + 1e-6:
                            return False, float('inf'), None, None
                        found = True
                        break
                if not found:
                    return False, float('inf'), None, None
        
        return True, cost, arrival_times, loads
    
    def evaluate_route_fast(route):
        """Fast feasibility and cost check without detailed precedence (LIFO implies precedence)."""
        if len(route) < 2:
            return False, float('inf'), None, None
        
        cost = 0.0
        current_time = float(nodes[route[0]]['tw_early'])
        current_load = 0
        arrival_times = [current_time]
        loads = [0]
        
        stack = []
        
        for idx in range(1, len(route)):
            prev = route[idx - 1]
            curr = route[idx]
            
            key = (prev, curr)
            if key not in arc_cost:
                return False, float('inf'), None, None
            
            cost += arc_cost[key]
            t = arc_time[key]
            
            earliest_arrival = current_time + nodes[prev]['service_time'] + t
            tw_early = nodes[curr]['tw_early']
            tw_late = nodes[curr]['tw_late']
            
            service_start = max(earliest_arrival, tw_early)
            if service_start > tw_late + 1e-6:
                return False, float('inf'), None, None
            
            current_time = service_start
            current_load += nodes[curr]['load']
            
            if current_load < -1e-6 or current_load > vehicle_capacity + 1e-6:
                return False, float('inf'), None, None
            
            node_info = nodes[curr]
            if node_info['node_type'] == 'pickup':
                stack.append(node_info['request_id'])
            elif node_info['node_type'] == 'delivery':
                req_id = node_info['request_id']
                if not stack or stack[-1] != req_id:
                    return False, float('inf'), None, None
                stack.pop()
            
            arrival_times.append(current_time)
            loads.append(current_load)
        
        return True, cost, arrival_times, loads
    
    # Build initial solution: each request gets its own vehicle
    def build_single_request_routes():
        routes = []
        for req in range(1, n_requests + 1):
            p = req
            d = get_delivery(req)
            route = [origin_depot, p, d, dest_depot]
            feasible, cost, at, ld = evaluate_route_fast(route)
            if feasible:
                routes.append(route)
            else:
                # Try anyway - this should always work for a single request
                routes.append(route)
        return routes
    
    def total_cost(routes):
        tc = 0.0
        for route in routes:
            _, cost, _, _ = evaluate_route_fast(route)
            tc += cost
        return tc
    
    # Nearest neighbor insertion heuristic
    def build_initial_solution():
        routes = []
        unassigned = list(range(1, n_requests + 1))
        
        # Sort by earliest pickup time
        unassigned.sort(key=lambda r: nodes[r]['tw_early'])
        
        while unassigned:
            # Start a new route
            route = [origin_depot, dest_depot]
            
            # Try to insert requests greedily
            remaining = list(unassigned)
            for req in remaining:
                p = req
                d = get_delivery(req)
                
                best_cost_increase = float('inf')
                best_route = None
                
                # Try all valid insertion positions respecting LIFO
                # For LIFO, pickup and delivery of the same request must be nested
                # This means we insert pickup at position i and delivery at position i+1 (adjacent)
                # or with other nested pairs in between
                
                # Simple approach: try inserting p at position i, d at position j > i
                route_len = len(route)
                for i in range(1, route_len):
                    for j in range(i + 1, route_len + 1):
                        new_route = route[:i] + [p] + route[i:j] + [d] + route[j:]
                        feasible, cost, _, _ = evaluate_route_fast(new_route)
                        if feasible:
                            # Calculate cost increase
                            old_feasible, old_cost, _, _ = evaluate_route_fast(route)
                            if not old_feasible:
                                old_cost = 0
                            increase = cost - old_cost
                            if increase < best_cost_increase:
                                best_cost_increase = increase
                                best_route = new_route
                
                if best_route is not None and best_cost_increase < vehicle_fixed_cost * 1.5:
                    route = best_route
                    unassigned.remove(req)
            
            if len(route) > 2:
                routes.append(route)
            else:
                break
        
        # Assign remaining unassigned requests to individual routes
        for req in unassigned:
            p = req
            d = get_delivery(req)
            route = [origin_depot, p, d, dest_depot]
            routes.append(route)
        
        return routes
    
    def build_initial_solution_v2():
        """Build initial solution using a savings-based approach."""
        # Start with individual routes
        routes = build_single_request_routes()
        return routes
    
    def try_insert_request_in_route(route, req):
        """Try to insert a request (pickup + delivery) into an existing route.
        Returns (best_route, best_cost) or (None, inf) if infeasible."""
        p = req
        d = get_delivery(req)
        
        best_cost = float('inf')
        best_route = None
        
        route_len = len(route)
        
        # For LIFO, we need to ensure the stack discipline
        # Try all valid positions
        for i in range(1, route_len):
            for j in range(i + 1, route_len + 1):
                new_route = route[:i] + [p] + route[i:j] + [d] + route[j:]
                feasible, cost, _, _ = evaluate_route_fast(new_route)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_route = new_route
        
        return best_route, best_cost
    
    # Build solution with greedy insertion
    def build_greedy_solution():
        unassigned = list(range(1, n_requests + 1))
        random.shuffle(unassigned)
        
        routes = []
        
        for req in unassigned:
            p = req
            d = get_delivery(req)
            
            best_route_idx = -1
            best_new_route = None
            best_increase = float('inf')
            
            for ridx, route in enumerate(routes):
                new_route, new_cost = try_insert_request_in_route(route, req)
                if new_route is not None:
                    _, old_cost, _, _ = evaluate_route_fast(route)
                    increase = new_cost - old_cost
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = ridx
                        best_new_route = new_route
            
            # Cost of new vehicle
            single_route = [origin_depot, p, d, dest_depot]
            _, single_cost, _, _ = evaluate_route_fast(single_route)
            
            if best_route_idx >= 0 and best_increase < single_cost:
                routes[best_route_idx] = best_new_route
            else:
                routes.append(single_route)
        
        return routes
    
    # Improvement operators
    def relocate_request(routes):
        """Try to move a request from one route to another."""
        improved = False
        best_routes = routes
        best_total = total_cost(routes)
        
        for r1_idx in range(len(routes)):
            route1 = routes[r1_idx]
            # Find requests in route1
            requests_in_r1 = []
            for node_id in route1:
                if nodes[node_id]['node_type'] == 'pickup':
                    requests_in_r1.append(node_id)
            
            for req_pickup in requests_in_r1:
                req_delivery = get_delivery(req_pickup)
                
                # Remove request from route1
                new_r1 = [n for n in route1 if n != req_pickup and n != req_delivery]
                
                if len(new_r1) <= 2:
                    # Route becomes empty
                    r1_feasible = True
                    r1_cost = 0
                    if len(new_r1) == 2:
                        r1_feasible, r1_cost, _, _ = evaluate_route_fast(new_r1)
                else:
                    r1_feasible, r1_cost, _, _ = evaluate_route_fast(new_r1)
                
                if not r1_feasible:
                    continue
                
                _, old_r1_cost, _, _ = evaluate_route_fast(route1)
                
                for r2_idx in range(len(routes)):
                    if r2_idx == r1_idx:
                        continue
                    
                    route2 = routes[r2_idx]
                    _, old_r2_cost, _, _ = evaluate_route_fast(route2)
                    
                    new_r2, new_r2_cost = try_insert_request_in_route(route2, req_pickup)
                    
                    if new_r2 is not None:
                        # Calculate improvement
                        old_total = old_r1_cost + old_r2_cost
                        new_total = r1_cost + new_r2_cost
                        
                        if len(new_r1) <= 2:
                            # We save a vehicle
                            new_total = new_r2_cost
                        
                        if new_total < old_total - 1e-6:
                            new_routes = list(routes)
                            new_routes[r1_idx] = new_r1
                            new_routes[r2_idx] = new_r2
                            # Remove empty routes
                            new_routes = [r for r in new_routes if len(r) > 2]
                            tc = total_cost(new_routes)
                            if tc < best_total - 1e-6:
                                best_total = tc
                                best_routes = new_routes
                                improved = True
                
                # Also try creating a new route (shouldn't help but for completeness)
        
        return improved, best_routes, best_total
    
    def or_opt_within_route(routes):
        """Try to reorder within a route by moving a segment."""
        improved = False
        best_routes = list(routes)
        best_total = total_cost(routes)
        
        for ridx in range(len(routes)):
            route = routes[ridx]
            if len(route) <= 4:  # depot + 1 pair + depot
                continue
            
            _, old_cost, _, _ = evaluate_route_fast(route)
            
            # Try swapping pairs of pickup-delivery
            # Get requests in order
            req_pairs = []
            for i, node_id in enumerate(route):
                if nodes[node_id]['node_type'] == 'pickup':
                    # Find corresponding delivery
                    del_id = get_delivery(node_id)
                    del_pos = route.index(del_id)
                    req_pairs.append((i, del_pos, node_id, del_id))
            
            # Try moving a request pair to different positions
            for pair_idx, (pi, di, p, d) in enumerate(req_pairs):
                # Remove this pair
                new_route = [n for n in route if n != p and n != d]
                
                # Try reinserting at best position
                best_r, best_c = try_insert_request_in_route(new_route, p)
                if best_r is not None and best_c < old_cost - 1e-6:
                    new_routes = list(routes)
                    new_routes[ridx] = best_r
                    tc = total_cost(new_routes)
                    if tc < best_total - 1e-6:
                        best_total = tc
                        best_routes = new_routes
                        improved = True
        
        return improved, best_routes, best_total
    
    def swap_requests_between_routes(routes):
        """Try swapping one request between two routes."""
        improved = False
        best_routes = list(routes)
        best_total = total_cost(routes)
        
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)):
                route1 = routes[r1_idx]
                route2 = routes[r2_idx]
                
                _, cost1, _, _ = evaluate_route_fast(route1)
                _, cost2, _, _ = evaluate_route_fast(route2)
                old_pair_cost = cost1 + cost2
                
                reqs1 = [n for n in route1 if nodes[n]['node_type'] == 'pickup']
                reqs2 = [n for n in route2 if nodes[n]['node_type'] == 'pickup']
                
                for p1 in reqs1:
                    d1 = get_delivery(p1)
                    for p2 in reqs2:
                        d2 = get_delivery(p2)
                        
                        # Remove p1,d1 from route1 and p2,d2 from route2
                        nr1 = [n for n in route1 if n != p1 and n != d1]
                        nr2 = [n for n in route2 if n != p2 and n != d2]
                        
                        # Insert p2,d2 into nr1 and p1,d1 into nr2
                        new_r1, new_c1 = try_insert_request_in_route(nr1, p2)
                        new_r2, new_c2 = try_insert_request_in_route(nr2, p1)
                        
                        if new_r1 is not None and new_r2 is not None:
                            new_pair_cost = new_c1 + new_c2
                            if new_pair_cost < old_pair_cost - 1e-6:
                                new_routes = list(routes)
                                new_routes[r1_idx] = new_r1
                                new_routes[r2_idx] = new_r2
                                tc = total_cost(new_routes)
                                if tc < best_total - 1e-6:
                                    best_total = tc
                                    best_routes = new_routes
                                    improved = True
        
        return improved, best_routes, best_total
    
    def merge_routes(routes):
        """Try to merge two routes into one."""
        improved = False
        best_routes = list(routes)
        best_total = total_cost(routes)
        
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)):
                if time.time() - start_time > time_limit * 0.9:
                    return improved, best_routes, best_total
                
                route1 = routes[r1_idx]
                route2 = routes[r2_idx]
                
                _, cost1, _, _ = evaluate_route_fast(route1)
                _, cost2, _, _ = evaluate_route_fast(route2)
                old_pair_cost = cost1 + cost2
                
                # Get requests from route2
                reqs2 = [n for n in route2 if nodes[n]['node_type'] == 'pickup']
                
                # Try inserting all requests from route2 into route1
                merged = list(route1)
                all_inserted = True
                
                for p2 in reqs2:
                    new_merged, new_cost = try_insert_request_in_route(merged, p2)
                    if new_merged is not None:
                        merged = new_merged
                    else:
                        all_inserted = False
                        break
                
                if all_inserted:
                    _, merged_cost, _, _ = evaluate_route_fast(merged)
                    if merged_cost < old_pair_cost - 1e-6:
                        new_routes = list(routes)
                        new_routes[r1_idx] = merged
                        new_routes.pop(r2_idx)
                        tc = total_cost(new_routes)
                        if tc < best_total - 1e-6:
                            best_total = tc
                            best_routes = new_routes
                            improved = True
        
        return improved, best_routes, best_total
    
    # Build initial solution
    if n_requests <= 30:
        routes = build_greedy_solution()
    else:
        routes = build_single_request_routes()
    
    current_total = total_cost(routes)
    best_solution = deepcopy(routes)
    best_obj = current_total
    
    if logger:
        logger.log(best_obj)
    
    # Try greedy construction multiple times with different orderings
    n_tries = min(20, max(1, int((time_limit * 0.3) / max(0.01, (time.time() - start_time + 0.001)))))
    for _ in range(n_tries):
        if time.time() - start_time > time_limit * 0.3:
            break
        candidate = build_greedy_solution()
        c_cost = total_cost(candidate)
        if c_cost < best_obj - 1e-6:
            best_obj = c_cost
            best_solution = deepcopy(candidate)
            if logger:
                logger.log(best_obj)
    
    routes = deepcopy(best_solution)
    current_total = best_obj
    
    # Local search improvement loop
    max_iterations = 1000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit * 0.85:
            break
        
        improved = False
        
        # Try merge routes first (reduce vehicles)
        if time.time() - start_time < time_limit * 0.7:
            imp, routes, current_total = merge_routes(routes)
            if imp:
                improved = True
                if current_total < best_obj - 1e-6:
                    best_obj = current_total
                    best_solution = deepcopy(routes)
                    if logger:
                        logger.log(best_obj)
        
        # Relocate requests
        if time.time() - start_time < time_limit * 0.8:
            imp, routes, current_total = relocate_request(routes)
            if imp:
                improved = True
                if current_total < best_obj - 1e-6:
                    best_obj = current_total
                    best_solution = deepcopy(routes)
                    if logger:
                        logger.log(best_obj)
        
        # Or-opt within routes
        if time.time() - start_time < time_limit * 0.85:
            imp, routes, current_total = or_opt_within_route(routes)
            if imp:
                improved = True
                if current_total < best_obj - 1e-6:
                    best_obj = current_total
                    best_solution = deepcopy(routes)
                    if logger:
                        logger.log(best_obj)
        
        # Swap between routes
        if time.time() - start_time < time_limit * 0.85 and len(routes) > 1:
            imp, routes, current_total = swap_requests_between_routes(routes)
            if imp:
                improved = True
                if current_total < best_obj - 1e-6:
                    best_obj = current_total
                    best_solution = deepcopy(routes)
                    if logger:
                        logger.log(best_obj)
        
        if not improved:
            break
    
    # Perturbation + local search (simple restart approach)
    while time.time() - start_time < time_limit * 0.95:
        # Perturb current solution
        perturbed = deepcopy(best_solution)
        
        if len(perturbed) > 1 and random.random() < 0.5:
            # Move a random request to a different route
            r_idx = random.randint(0, len(perturbed) - 1)
            route = perturbed[r_idx]
            reqs = [n for n in route if nodes[n]['node_type'] == 'pickup']
            if reqs:
                req = random.choice(reqs)
                d = get_delivery(req)
                new_route = [n for n in route if n != req and n != d]
                
                other_indices = [i for i in range(len(perturbed)) if i != r_idx]
                if other_indices:
                    target_idx = random.choice(other_indices)
                    target_route = perturbed[target_idx]
                    new_target, _ = try_insert_request_in_route(target_route, req)
                    if new_target is not None:
                        perturbed[r_idx] = new_route
                        perturbed[target_idx] = new_target
                        perturbed = [r for r in perturbed if len(r) > 2]
        else:
            # Swap two random requests between routes
            if len(perturbed) > 1:
                r1_idx, r2_idx = random.sample(range(len(perturbed)), 2)
                reqs1 = [n for n in perturbed[r1_idx] if nodes[n]['node_type'] == 'pickup']
                reqs2 = [n for n in perturbed[r2_idx] if nodes[n]['node_type'] == 'pickup']
                if reqs1 and reqs2:
                    p1 = random.choice(reqs1)
                    p2 = random.choice(reqs2)
                    d1 = get_delivery(p1)
                    d2 = get_delivery(p2)
                    
                    nr1 = [n for n in perturbed[r1_idx] if n != p1 and n != d1]
                    nr2 = [n for n in perturbed[r2_idx] if n != p2 and n != d2]
                    
                    new_r1, _ = try_insert_request_in_route(nr1, p2)
                    new_r2, _ = try_insert_request_in_route(nr2, p1)
                    
                    if new_r1 is not None and new_r2 is not None:
                        perturbed[r1_idx] = new_r1
                        perturbed[r2_idx] = new_r2
        
        # Local search on perturbed
        routes = perturbed
        current_total = total_cost(routes)
        
        for _ in range(50):
            if time.time() - start_time > time_limit * 0.93:
                break
            
            improved_any = False
            
            imp, routes, current_total = merge_routes(routes)
            if imp:
                improved_any = True
            
            if time.time() - start_time > time_limit * 0.93:
                break
            
            imp, routes, current_total = relocate_request(routes)
            if imp:
                improved_any = True
            
            if time.time() - start_time > time_limit * 0.93:
                break
            
            imp, routes, current_total = or_opt_within_route(routes)
            if imp:
                improved_any = True
            
            if not improved_any:
                break
        
        if current_total < best_obj - 1e-6:
            best_obj = current_total
            best_solution = deepcopy(routes)
            if logger:
                logger.log(best_obj)
    
    return best_solution, best_obj


def format_solution(routes, instance):
    nodes_dict = {}
    for n in instance['nodes']:
        nodes_dict[n['node_id']] = n
    
    arc_cost_map = {}
    arc_time_map = {}
    for a in instance['arcs']:
        arc_cost_map[(a['from'], a['to'])] = a['cost']
        arc_time_map[(a['from'], a['to'])] = a['travel_time']
    
    origin_depot = 0
    
    solution_routes = []
    total_obj = 0.0
    
    for vid, route in enumerate(routes):
        if len(route) <= 2:
            continue
        
        route_details = []
        current_time = float(nodes_dict[route[0]]['tw_early'])
        current_load = 0
        
        route_details.append({
            "node_id": route[0],
            "arrival_time": current_time,
            "load_after": 0.0
        })
        
        route_cost = 0.0
        
        for idx in range(1, len(route)):
            prev = route[idx - 1]
            curr = route[idx]
            
            c = arc_cost_map.get((prev, curr), 0)
            t = arc_time_map.get((prev, curr), 0)
            route_cost += c
            
            earliest = current_time + nodes_dict[prev]['service_time'] + t
            service_start = max(earliest, nodes_dict[curr]['tw_early'])
            current_time = service_start
            current_load += nodes_dict[curr]['load']
            
            route_details.append({
                "node_id": curr,
                "arrival_time": current_time,
                "load_after": float(current_load)
            })
        
        total_obj += route_cost
        
        solution_routes.append({
            "vehicle_id": vid,
            "route_sequence": route,
            "route_details": route_details
        })
    
    return {
        "objective_value": total_obj,
        "routes": solution_routes
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    routes, obj = solve(instance, args.time_limit, logger)
    
    solution = format_solution(routes, instance)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()