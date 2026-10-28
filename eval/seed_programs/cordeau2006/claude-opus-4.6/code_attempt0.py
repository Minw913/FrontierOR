import argparse
import json
import math
import time
import random
import copy
from solution_logger import SolutionLogger


def euclidean_distance(n1, n2):
    return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)


def solve_darp(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_vehicles = instance['num_vehicles']
    num_users = instance['num_users']
    vehicle_capacity = instance['vehicle_capacity']
    maximum_ride_time = instance['maximum_ride_time']
    num_nodes = instance['num_nodes']
    maximum_route_duration = instance['maximum_route_duration']
    nodes = instance['nodes']
    
    # Build node lookup
    node_by_id = {}
    for n in nodes:
        node_by_id[n['node_id']] = n
    
    origin_depot = 0
    dest_depot = num_nodes - 1
    
    # Precompute distance matrix
    dist = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist[(i, j)] = euclidean_distance(node_by_id[i], node_by_id[j])
    
    # Build user info
    # Users are numbered 0..num_users-1
    # Pickup nodes: 1..num_users
    # Dropoff nodes: num_users+1..2*num_users
    pickup_node = {}  # user_id -> pickup node id
    dropoff_node = {}  # user_id -> dropoff node id
    
    for n in nodes:
        if n['node_type'] == 'pickup':
            pickup_node[n['user_id']] = n['node_id']
        elif n['node_type'] == 'dropoff':
            dropoff_node[n['user_id']] = n['node_id']
    
    users = list(range(num_users))
    
    def evaluate_route(route):
        """Evaluate a route (list of node ids starting/ending at depots).
        Returns (feasible, cost, service_times, loads, ride_times) or (False, ...) if infeasible.
        """
        if len(route) < 2:
            return False, float('inf'), {}, {}, {}
        
        if route[0] != origin_depot or route[-1] != dest_depot:
            return False, float('inf'), {}, {}, {}
        
        cost = 0.0
        service_times = {}
        loads = {}
        
        # Start at depot
        n0 = node_by_id[route[0]]
        current_time = n0['earliest_time']
        service_times[route[0]] = current_time
        current_load = 0
        loads[route[0]] = 0
        
        for idx in range(1, len(route)):
            prev = route[idx - 1]
            curr = route[idx]
            prev_node = node_by_id[prev]
            curr_node = node_by_id[curr]
            
            travel_time = dist[(prev, curr)]
            cost += travel_time
            
            arrival_time = service_times[prev] + prev_node['service_duration'] + travel_time
            
            # Must respect earliest time
            begin_service = max(arrival_time, curr_node['earliest_time'])
            
            # Check latest time
            if begin_service > curr_node['latest_time'] + 1e-6:
                return False, float('inf'), {}, {}, {}
            
            service_times[curr] = begin_service
            
            # Update load
            current_load += curr_node['load']
            if current_load < 0 - 1e-6 or current_load > vehicle_capacity + 1e-6:
                return False, float('inf'), {}, {}, {}
            
            loads[curr] = current_load
        
        # Check route duration
        route_duration = service_times[route[-1]] - service_times[route[0]]
        if route_duration > maximum_route_duration + 1e-6:
            return False, float('inf'), {}, {}, {}
        
        # Check ride times
        ride_times = {}
        for u in users:
            p = pickup_node[u]
            d = dropoff_node[u]
            if p in service_times and d in service_times:
                p_node = node_by_id[p]
                rt = service_times[d] - (service_times[p] + p_node['service_duration'])
                if rt < dist[(p, d)] - 1e-6:
                    return False, float('inf'), {}, {}, {}
                if rt > maximum_ride_time + 1e-6:
                    return False, float('inf'), {}, {}, {}
                ride_times[u] = rt
        
        return True, cost, service_times, loads, ride_times
    
    def forward_time_slack(route, service_times):
        """Compute forward time slack to allow shifting service times later."""
        # This helps with optimization by allowing us to push times forward
        n = len(route)
        if n <= 2:
            return {}
        
        slack = {}
        for i in range(n):
            nid = route[i]
            nd = node_by_id[nid]
            slack[i] = nd['latest_time'] - service_times[nid]
        
        # Propagate backwards
        for i in range(n - 2, -1, -1):
            nid = route[i]
            nd = node_by_id[nid]
            next_nid = route[i + 1]
            
            travel = dist[(nid, next_nid)]
            earliest_next = service_times[nid] + nd['service_duration'] + travel
            wait = max(0, node_by_id[next_nid]['earliest_time'] - earliest_next)
            
            slack[i] = min(slack[i], wait + slack[i + 1])
        
        return slack
    
    def try_schedule_route(route):
        """Try to schedule a route respecting all constraints, using forward scheduling
        with potential optimization."""
        if len(route) < 2:
            return False, float('inf'), {}, {}, {}
        
        return evaluate_route(route)
    
    def eight_step_scheduling(route):
        """
        Cordeau & Laporte style scheduling to minimize ride times.
        For each pickup node, try to push its service time as late as possible
        while still respecting all constraints.
        """
        if len(route) <= 2:
            return evaluate_route(route)
        
        # First do forward scheduling
        feasible, cost, service_times, loads, ride_times = evaluate_route(route)
        if not feasible:
            return False, float('inf'), {}, {}, {}
        
        # Now try to optimize by pushing service times later to reduce ride times
        # For each pickup in route order, try to delay it
        pickup_indices = []
        for idx, nid in enumerate(route):
            nd = node_by_id[nid]
            if nd['node_type'] == 'pickup':
                pickup_indices.append(idx)
        
        for p_idx in pickup_indices:
            p_nid = route[p_idx]
            p_nd = node_by_id[p_nid]
            uid = p_nd['user_id']
            d_nid = dropoff_node[uid]
            
            # Find dropoff index
            d_idx = None
            for idx in range(p_idx + 1, len(route)):
                if route[idx] == d_nid:
                    d_idx = idx
                    break
            
            if d_idx is None:
                continue
            
            # Compute how much we can shift the pickup later
            # without violating any constraints between p_idx and the end
            max_shift = float('inf')
            
            for idx in range(p_idx, len(route)):
                nid = route[idx]
                nd = node_by_id[nid]
                available_shift = nd['latest_time'] - service_times[nid]
                
                if idx > p_idx:
                    prev_nid = route[idx - 1]
                    prev_nd = node_by_id[prev_nid]
                    travel = dist[(prev_nid, nid)]
                    earliest_arrival = service_times[prev_nid] + prev_nd['service_duration'] + travel
                    wait_time = service_times[nid] - earliest_arrival
                    if wait_time > 1e-9:
                        # This wait absorbs some shift
                        available_shift += wait_time
                
                max_shift = min(max_shift, available_shift)
                if max_shift <= 1e-9:
                    break
            
            # Also limited by ride time constraint for user uid
            current_ride_time = service_times[d_nid] - (service_times[p_nid] + p_nd['service_duration'])
            
            # Also limited by maximum route duration
            route_dur = service_times[route[-1]] - service_times[route[0]]
            duration_slack = maximum_route_duration - route_dur
            
            if max_shift > 1e-9:
                # Apply shift
                shift = max_shift
                # Recompute service times from p_idx onwards
                new_service_times = dict(service_times)
                new_service_times[route[p_idx]] = service_times[route[p_idx]] + shift
                
                valid = True
                for idx in range(p_idx + 1, len(route)):
                    prev_nid = route[idx - 1]
                    prev_nd = node_by_id[prev_nid]
                    curr_nid = route[idx]
                    curr_nd = node_by_id[curr_nid]
                    travel = dist[(prev_nid, curr_nid)]
                    earliest = new_service_times[prev_nid] + prev_nd['service_duration'] + travel
                    begin = max(earliest, curr_nd['earliest_time'])
                    if begin > curr_nd['latest_time'] + 1e-6:
                        valid = False
                        break
                    new_service_times[curr_nid] = begin
                
                if valid:
                    # Check route duration
                    new_dur = new_service_times[route[-1]] - new_service_times[route[0]]
                    if new_dur <= maximum_route_duration + 1e-6:
                        # Check all ride times
                        all_ok = True
                        new_ride_times = {}
                        for u in users:
                            p = pickup_node[u]
                            d = dropoff_node[u]
                            if p in new_service_times and d in new_service_times:
                                pn = node_by_id[p]
                                rt = new_service_times[d] - (new_service_times[p] + pn['service_duration'])
                                if rt < dist[(p, d)] - 1e-6 or rt > maximum_ride_time + 1e-6:
                                    all_ok = False
                                    break
                                new_ride_times[u] = rt
                        
                        if all_ok:
                            service_times = new_service_times
                            ride_times = new_ride_times
        
        return True, cost, service_times, loads, ride_times
    
    # Greedy construction heuristic
    def greedy_construct():
        """Build initial solution using greedy insertion."""
        routes = [[] for _ in range(num_vehicles)]
        # Initialize routes as depot-to-depot
        for v in range(num_vehicles):
            routes[v] = [origin_depot, dest_depot]
        
        unassigned = list(range(num_users))
        random.shuffle(unassigned)
        
        for u in unassigned:
            p = pickup_node[u]
            d = dropoff_node[u]
            
            best_cost = float('inf')
            best_vehicle = -1
            best_route = None
            
            for v in range(num_vehicles):
                route = routes[v]
                n = len(route)
                
                # Try inserting pickup at position i and dropoff at position j > i
                for i in range(1, n):
                    for j in range(i + 1, n + 1):
                        new_route = route[:i] + [p] + route[i:j] + [d] + route[j:]
                        
                        feasible, cost, st, ld, rt = evaluate_route(new_route)
                        if feasible and cost < best_cost:
                            best_cost = cost
                            best_vehicle = v
                            best_route = new_route
            
            if best_vehicle >= 0:
                routes[best_vehicle] = best_route
            else:
                # Force insert into least loaded vehicle
                # Try relaxing - just find any vehicle that can take it
                inserted = False
                for v in range(num_vehicles):
                    route = routes[v]
                    n = len(route)
                    for i in range(1, n):
                        new_route = route[:i] + [p] + route[i:i] + [d] + route[i:]
                        feasible, cost, st, ld, rt = evaluate_route(new_route)
                        if feasible:
                            routes[v] = new_route
                            inserted = True
                            break
                    if inserted:
                        break
                
                if not inserted:
                    # Last resort: append to first vehicle
                    route = routes[0]
                    new_route = route[:-1] + [p, d] + [route[-1]]
                    routes[0] = new_route
        
        return routes
    
    def compute_solution(routes):
        """Compute full solution metrics."""
        total_cost = 0.0
        all_service_times = {}
        all_ride_times = {}
        all_feasible = True
        
        for v in range(num_vehicles):
            route = routes[v]
            feasible, cost, st, ld, rt = eight_step_scheduling(route)
            if not feasible:
                all_feasible = False
                # Try basic evaluation
                feasible, cost, st, ld, rt = evaluate_route(route)
                if not feasible:
                    return False, float('inf'), {}, {}, {}
            
            total_cost += cost
            all_service_times.update(st)
            all_ride_times.update(rt)
            
            # Store depot times per vehicle
            all_service_times[f'depot_start_{v}'] = st.get(route[0], 0)
            all_service_times[f'depot_end_{v}'] = st.get(route[-1], 0)
        
        return all_feasible, total_cost, all_service_times, all_ride_times, routes
    
    def get_user_vehicle(routes):
        """Get mapping from user to vehicle."""
        user_to_vehicle = {}
        for v in range(num_vehicles):
            for nid in routes[v]:
                nd = node_by_id[nid]
                if nd['node_type'] == 'pickup':
                    user_to_vehicle[nd['user_id']] = v
        return user_to_vehicle
    
    # Local search operators
    def relocate_user(routes, best_cost):
        """Try moving a user from one vehicle to another."""
        improved = True
        while improved:
            improved = False
            user_to_vehicle = get_user_vehicle(routes)
            
            for u in users:
                if time.time() - start_time > time_limit - 2:
                    return routes, best_cost
                
                v_from = user_to_vehicle.get(u)
                if v_from is None:
                    continue
                
                p = pickup_node[u]
                d = dropoff_node[u]
                
                # Remove user from current route
                route_from = routes[v_from]
                new_route_from = [nid for nid in route_from if nid != p and nid != d]
                
                feas_from, cost_from, _, _, _ = evaluate_route(new_route_from)
                if not feas_from:
                    feas_from, cost_from = True, 0
                    if len(new_route_from) == 2:
                        cost_from = dist[(new_route_from[0], new_route_from[1])]
                    else:
                        # Evaluate anyway
                        feas_from, cost_from, _, _, _ = evaluate_route(new_route_from)
                
                old_cost_from_route, _ = 0, None
                feas_old, cost_old_from, _, _, _ = evaluate_route(route_from)
                if feas_old:
                    old_cost_from_route = cost_old_from
                
                for v_to in range(num_vehicles):
                    if v_to == v_from:
                        continue
                    
                    if time.time() - start_time > time_limit - 2:
                        return routes, best_cost
                    
                    route_to = routes[v_to]
                    old_cost_to_route = 0
                    feas_old_to, cost_old_to, _, _, _ = evaluate_route(route_to)
                    if feas_old_to:
                        old_cost_to_route = cost_old_to
                    
                    n = len(route_to)
                    best_insert_cost = float('inf')
                    best_insert_route = None
                    
                    for i in range(1, n):
                        for j in range(i + 1, n + 1):
                            new_route_to = route_to[:i] + [p] + route_to[i:j] + [d] + route_to[j:]
                            feas_to, cost_to, _, _, _ = evaluate_route(new_route_to)
                            if feas_to and cost_to < best_insert_cost:
                                best_insert_cost = cost_to
                                best_insert_route = new_route_to
                    
                    if best_insert_route is not None:
                        new_total = best_cost - old_cost_from_route - old_cost_to_route + cost_from + best_insert_cost
                        if new_total < best_cost - 1e-6:
                            routes[v_from] = new_route_from
                            routes[v_to] = best_insert_route
                            best_cost = new_total
                            improved = True
                            user_to_vehicle[u] = v_to
                            
                            if logger:
                                logger.log(best_cost)
                            break
                
                if improved:
                    break
        
        return routes, best_cost
    
    def intra_route_reorder(routes, best_cost):
        """Try reordering nodes within a route using Or-opt style moves."""
        improved = True
        while improved:
            improved = False
            for v in range(num_vehicles):
                if time.time() - start_time > time_limit - 2:
                    return routes, best_cost
                
                route = routes[v]
                if len(route) <= 4:  # Just depots + maybe one user
                    continue
                
                feas_old, cost_old, _, _, _ = evaluate_route(route)
                if not feas_old:
                    continue
                
                # Try removing a user and reinserting at best position
                users_in_route = []
                for nid in route:
                    nd = node_by_id[nid]
                    if nd['node_type'] == 'pickup':
                        users_in_route.append(nd['user_id'])
                
                for u in users_in_route:
                    p = pickup_node[u]
                    d = dropoff_node[u]
                    
                    stripped = [nid for nid in route if nid != p and nid != d]
                    
                    n = len(stripped)
                    best_new_cost = cost_old
                    best_new_route = None
                    
                    for i in range(1, n):
                        for j in range(i + 1, n + 1):
                            new_route = stripped[:i] + [p] + stripped[i:j] + [d] + stripped[j:]
                            feas, cost, _, _, _ = evaluate_route(new_route)
                            if feas and cost < best_new_cost - 1e-6:
                                best_new_cost = cost
                                best_new_route = new_route
                    
                    if best_new_route is not None:
                        routes[v] = best_new_route
                        best_cost = best_cost - cost_old + best_new_cost
                        cost_old = best_new_cost
                        route = best_new_route
                        improved = True
                        
                        if logger:
                            logger.log(best_cost)
        
        return routes, best_cost
    
    def swap_users(routes, best_cost):
        """Try swapping users between vehicles."""
        improved = True
        while improved:
            improved = False
            user_to_vehicle = get_user_vehicle(routes)
            
            for u1 in users:
                if time.time() - start_time > time_limit - 2:
                    return routes, best_cost
                
                for u2 in users:
                    if u2 <= u1:
                        continue
                    
                    v1 = user_to_vehicle.get(u1)
                    v2 = user_to_vehicle.get(u2)
                    
                    if v1 is None or v2 is None or v1 == v2:
                        continue
                    
                    if time.time() - start_time > time_limit - 2:
                        return routes, best_cost
                    
                    p1, d1 = pickup_node[u1], dropoff_node[u1]
                    p2, d2 = pickup_node[u2], dropoff_node[u2]
                    
                    feas1_old, cost1_old, _, _, _ = evaluate_route(routes[v1])
                    feas2_old, cost2_old, _, _, _ = evaluate_route(routes[v2])
                    
                    if not feas1_old or not feas2_old:
                        continue
                    
                    # Remove u1 from v1, add u2
                    route1_stripped = [nid for nid in routes[v1] if nid != p1 and nid != d1]
                    route2_stripped = [nid for nid in routes[v2] if nid != p2 and nid != d2]
                    
                    # Try best insertion of u2 into route1_stripped
                    best_r1_cost = float('inf')
                    best_r1 = None
                    n1 = len(route1_stripped)
                    for i in range(1, n1):
                        for j in range(i + 1, n1 + 1):
                            nr = route1_stripped[:i] + [p2] + route1_stripped[i:j] + [d2] + route1_stripped[j:]
                            f, c, _, _, _ = evaluate_route(nr)
                            if f and c < best_r1_cost:
                                best_r1_cost = c
                                best_r1 = nr
                    
                    if best_r1 is None:
                        continue
                    
                    # Try best insertion of u1 into route2_stripped
                    best_r2_cost = float('inf')
                    best_r2 = None
                    n2 = len(route2_stripped)
                    for i in range(1, n2):
                        for j in range(i + 1, n2 + 1):
                            nr = route2_stripped[:i] + [p1] + route2_stripped[i:j] + [d1] + route2_stripped[j:]
                            f, c, _, _, _ = evaluate_route(nr)
                            if f and c < best_r2_cost:
                                best_r2_cost = c
                                best_r2 = nr
                    
                    if best_r2 is None:
                        continue
                    
                    new_total = best_cost - cost1_old - cost2_old + best_r1_cost + best_r2_cost
                    if new_total < best_cost - 1e-6:
                        routes[v1] = best_r1
                        routes[v2] = best_r2
                        best_cost = new_total
                        improved = True
                        user_to_vehicle[u1] = v2
                        user_to_vehicle[u2] = v1
                        
                        if logger:
                            logger.log(best_cost)
                        break
                
                if improved:
                    break
        
        return routes, best_cost
    
    # --- Main solving logic ---
    
    # Try multiple random constructions and keep best
    best_solution = None
    best_obj = float('inf')
    
    num_constructions = max(1, min(50, int((time_limit - 1) / 2)))
    
    for attempt in range(num_constructions):
        if time.time() - start_time > time_limit * 0.3:
            break
        
        routes = greedy_construct()
        
        # Verify feasibility
        feasible, total_cost, all_st, all_rt, _ = compute_solution(routes)
        
        if feasible and total_cost < best_obj:
            best_obj = total_cost
            best_solution = [list(r) for r in routes]
            if logger:
                logger.log(best_obj)
    
    if best_solution is None:
        # Fallback: create trivial routes
        routes = [[] for _ in range(num_vehicles)]
        for v in range(num_vehicles):
            routes[v] = [origin_depot, dest_depot]
        
        # Assign all users to vehicle 0
        for u in range(num_users):
            p = pickup_node[u]
            d = dropoff_node[u]
            route = routes[0]
            route = route[:-1] + [p, d] + [route[-1]]
            routes[0] = route
        
        feasible, total_cost, _, _, _ = compute_solution(routes)
        if feasible:
            best_obj = total_cost
            best_solution = routes
            if logger:
                logger.log(best_obj)
    
    # Local search improvement
    if best_solution is not None:
        routes = [list(r) for r in best_solution]
        current_cost = best_obj
        
        # Intra-route optimization
        if time.time() - start_time < time_limit - 5:
            routes, current_cost = intra_route_reorder(routes, current_cost)
            if current_cost < best_obj:
                best_obj = current_cost
                best_solution = [list(r) for r in routes]
        
        # Inter-route relocate
        if time.time() - start_time < time_limit - 5:
            routes, current_cost = relocate_user(routes, current_cost)
            if current_cost < best_obj:
                best_obj = current_cost
                best_solution = [list(r) for r in routes]
        
        # Intra-route again after relocate
        if time.time() - start_time < time_limit - 5:
            routes, current_cost = intra_route_reorder(routes, current_cost)
            if current_cost < best_obj:
                best_obj = current_cost
                best_solution = [list(r) for r in routes]
        
        # Swap users
        if time.time() - start_time < time_limit - 5:
            routes, current_cost = swap_users(routes, current_cost)
            if current_cost < best_obj:
                best_obj = current_cost
                best_solution = [list(r) for r in routes]
        
        # Another round of intra-route + relocate
        if time.time() - start_time < time_limit - 5:
            routes, current_cost = intra_route_reorder(routes, current_cost)
            if current_cost < best_obj:
                best_obj = current_cost
                best_solution = [list(r) for r in routes]
        
        if time.time() - start_time < time_limit - 5:
            routes, current_cost = relocate_user(routes, current_cost)
            if current_cost < best_obj:
                best_obj = current_cost
                best_solution = [list(r) for r in routes]
        
        # Perturbation + re-optimization loop
        perturbation_rounds = 0
        while time.time() - start_time < time_limit - 10:
            perturbation_rounds += 1
            
            # Perturb: randomly remove some users and reinsert
            perturbed_routes = [list(r) for r in best_solution]
            user_to_vehicle = get_user_vehicle(perturbed_routes)
            
            # Remove random subset of users
            num_remove = max(1, min(num_users // 3, random.randint(1, max(1, num_users // 2))))
            to_remove = random.sample(users, min(num_remove, len(users)))
            
            removed_users = []
            for u in to_remove:
                v = user_to_vehicle.get(u)
                if v is not None:
                    p = pickup_node[u]
                    d = dropoff_node[u]
                    perturbed_routes[v] = [nid for nid in perturbed_routes[v] if nid != p and nid != d]
                    removed_users.append(u)
            
            # Reinsert in random order using best insertion
            random.shuffle(removed_users)
            all_inserted = True
            for u in removed_users:
                if time.time() - start_time > time_limit - 5:
                    break
                
                p = pickup_node[u]
                d = dropoff_node[u]
                
                best_insert_cost_diff = float('inf')
                best_v = -1
                best_route = None
                
                for v in range(num_vehicles):
                    route = perturbed_routes[v]
                    feas_old, cost_old, _, _, _ = evaluate_route(route)
                    if not feas_old:
                        cost_old = 0
                    
                    n = len(route)
                    for i in range(1, n):
                        for j in range(i + 1, n + 1):
                            new_route = route[:i] + [p] + route[i:j] + [d] + route[j:]
                            feas, cost, _, _, _ = evaluate_route(new_route)
                            if feas:
                                diff = cost - cost_old
                                # Add some randomness for diversification
                                noise = random.uniform(0, 0.1) * diff if diff > 0 else 0
                                if diff + noise < best_insert_cost_diff:
                                    best_insert_cost_diff = diff
                                    best_v = v
                                    best_route = new_route
                
                if best_v >= 0:
                    perturbed_routes[best_v] = best_route
                else:
                    all_inserted = False
                    # Force insert
                    for v in range(num_vehicles):
                        route = perturbed_routes[v]
                        new_route = route[:-1] + [p, d] + [route[-1]]
                        feas, _, _, _, _ = evaluate_route(new_route)
                        if feas:
                            perturbed_routes[v] = new_route
                            all_inserted = True
                            break
                    
                    if not all_inserted:
                        break
            
            if not all_inserted:
                continue
            
            # Evaluate perturbed solution
            feasible, perturbed_cost, _, _, _ = compute_solution(perturbed_routes)
            if not feasible:
                continue
            
            # Local search on perturbed solution
            if time.time() - start_time < time_limit - 5:
                perturbed_routes, perturbed_cost = intra_route_reorder(perturbed_routes, perturbed_cost)
            
            if time.time() - start_time < time_limit - 5:
                perturbed_routes, perturbed_cost = relocate_user(perturbed_routes, perturbed_cost)
            
            if time.time() - start_time < time_limit - 5:
                perturbed_routes, perturbed_cost = intra_route_reorder(perturbed_routes, perturbed_cost)
            
            if perturbed_cost < best_obj - 1e-6:
                best_obj = perturbed_cost
                best_solution = [list(r) for r in perturbed_routes]
                if logger:
                    logger.log(best_obj)
    
    # Final solution computation
    routes = best_solution if best_solution else [[origin_depot, dest_depot] for _ in range(num_vehicles)]
    
    # Recompute with scheduling optimization
    total_cost = 0.0
    all_service_times = {}
    all_ride_times = {}
    route_costs = []
    
    for v in range(num_vehicles):
        route = routes[v]
        feasible, cost, st, ld, rt = eight_step_scheduling(route)
        if not feasible:
            feasible, cost, st, ld, rt = evaluate_route(route)
        
        total_cost += cost
        route_costs.append(cost)
        
        for nid, t in st.items():
            all_service_times[str(nid)] = t
        
        all_service_times[f'depot_start_{v}'] = st.get(route[0], 0)
        all_service_times[f'depot_end_{v}'] = st.get(route[-1], 0)
        
        for uid, t in rt.items():
            all_ride_times[str(uid)] = t
    
    # Build output
    routes_output = {}
    for v in range(num_vehicles):
        routes_output[str(v)] = routes[v]
    
    solution = {
        "objective_value": total_cost,
        "routes": routes_output,
        "service_times": {str(k): v for k, v in all_service_times.items()},
        "ride_times": {str(k): v for k, v in all_ride_times.items()}
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_darp(args.instance_path, args.solution_path, args.time_limit, args.log_path)