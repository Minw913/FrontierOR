import argparse
import json
import math
import time
import random
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist_matrix = instance['distance_matrix']
    scheduling_horizon = instance['scheduling_horizon']
    
    # Build customer info indexed by id
    cust_info = {}
    cust_info[0] = {
        'id': 0, 'x': depot['x'], 'y': depot['y'],
        'demand': 0, 'ready_time': depot['ready_time'],
        'due_date': depot['due_date'], 'service_time': 0
    }
    for c in customers:
        cust_info[c['id']] = c
    
    # Compute effective travel times
    # From problem: effective travel time from customer i to location j:
    # base = dist_matrix[i][j] + service_time[i] (if i is a customer)
    # But "travel_time_includes_service_time" might already be true
    travel_time_includes = instance.get('travel_time_includes_service_time', False)
    
    n = num_customers + 1  # 0 = depot, 1..num_customers = customers
    
    # Build effective travel time matrix
    eff_time = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            base = dist_matrix[i][j]
            if not travel_time_includes and i > 0:
                base += cust_info[i]['service_time']
            # If difference between destination's ready_time and origin's due_date > adjusted travel time
            if i > 0 and j > 0:
                diff = cust_info[j]['ready_time'] - cust_info[i]['due_date']
                if diff > base:
                    base = diff
            elif i > 0 and j == 0:
                diff = cust_info[0]['ready_time'] - cust_info[i]['due_date']
                if diff > base:
                    base = diff
            elif i == 0 and j > 0:
                # depot to customer: no service time at depot
                diff = cust_info[j]['ready_time'] - cust_info[0]['due_date']
                if diff > base:
                    base = diff
            eff_time[i][j] = base
    
    # Customer IDs: 1..num_customers
    cust_ids = list(range(1, num_customers + 1))
    
    def check_route_feasibility(route):
        """Check if a route (list of customer ids, without depot) is feasible.
        Returns (feasible, total_distance, departure_times, loads) or (False, ...)"""
        if not route:
            return True, 0.0, {}, {}
        
        load = 0
        dep_times = {}
        loads = {}
        total_dist = 0.0
        
        # Start from depot
        prev = 0
        prev_dep_time = cust_info[0]['ready_time']
        
        for cid in route:
            c = cust_info[cid]
            travel = eff_time[prev][cid]
            arrival = prev_dep_time + travel
            
            # Departure time must be >= ready_time and <= due_date
            dep_time = max(arrival, c['ready_time'])
            if dep_time > c['due_date']:
                return False, 0, {}, {}
            
            load += c['demand']
            if load > vehicle_capacity:
                return False, 0, {}, {}
            
            total_dist += dist_matrix[prev][cid]
            dep_times[cid] = dep_time
            loads[cid] = load
            
            prev = cid
            prev_dep_time = dep_time
        
        # Return to depot
        travel_back = eff_time[prev][0]
        arrival_depot = prev_dep_time + travel_back
        if arrival_depot > cust_info[0]['due_date']:
            return False, 0, {}, {}
        
        total_dist += dist_matrix[prev][0]
        
        return True, total_dist, dep_times, loads
    
    def compute_route_details(route):
        """Compute details for a feasible route."""
        if not route:
            return 0.0, {}, {}
        
        load = 0
        dep_times = {}
        loads_dict = {}
        total_dist = 0.0
        
        prev = 0
        prev_dep_time = cust_info[0]['ready_time']
        
        for cid in route:
            c = cust_info[cid]
            travel = eff_time[prev][cid]
            arrival = prev_dep_time + travel
            dep_time = max(arrival, c['ready_time'])
            
            load += c['demand']
            total_dist += dist_matrix[prev][cid]
            dep_times[cid] = dep_time
            loads_dict[cid] = load
            
            prev = cid
            prev_dep_time = dep_time
        
        total_dist += dist_matrix[prev][0]
        return total_dist, dep_times, loads_dict
    
    def try_insert(route, cust_id, position=None):
        """Try to insert cust_id into route at best position. Returns (best_pos, cost_increase) or None."""
        best_pos = None
        best_cost = float('inf')
        
        positions = [position] if position is not None else range(len(route) + 1)
        
        for pos in positions:
            new_route = route[:pos] + [cust_id] + route[pos:]
            feasible, dist, _, _ = check_route_feasibility(new_route)
            if feasible:
                if dist < best_cost:
                    best_cost = dist
                    best_pos = pos
        
        if best_pos is not None:
            return best_pos, best_cost
        return None
    
    # Solomon I1 insertion heuristic
    def solomon_insertion(customer_order=None):
        if customer_order is None:
            customer_order = sorted(cust_ids, key=lambda c: cust_info[c]['ready_time'])
        
        unrouted = set(customer_order)
        routes = []
        
        while unrouted:
            # Start new route with the earliest-deadline unrouted customer
            seed = min(unrouted, key=lambda c: cust_info[c]['ready_time'])
            
            feasible, _, _, _ = check_route_feasibility([seed])
            if not feasible:
                # Try each unrouted customer as seed
                found_seed = False
                for s in sorted(unrouted, key=lambda c: cust_info[c]['ready_time']):
                    feasible, _, _, _ = check_route_feasibility([s])
                    if feasible:
                        seed = s
                        found_seed = True
                        break
                if not found_seed:
                    break
            
            route = [seed]
            unrouted.remove(seed)
            
            improved = True
            while improved:
                improved = False
                best_cust = None
                best_pos = None
                best_cost = float('inf')
                
                for cid in list(unrouted):
                    result = try_insert(route, cid)
                    if result is not None:
                        pos, cost = result
                        # Use regret-like: just pick cheapest insertion
                        if cost < best_cost:
                            best_cost = cost
                            best_cust = cid
                            best_pos = pos
                
                if best_cust is not None:
                    route.insert(best_pos, best_cust)
                    unrouted.remove(best_cust)
                    improved = True
            
            routes.append(route)
        
        return routes
    
    # Regret-2 insertion heuristic
    def regret_insertion(customer_order=None):
        if customer_order is None:
            customer_order = sorted(cust_ids, key=lambda c: cust_info[c]['ready_time'])
        
        unrouted = set(customer_order)
        routes = []
        
        while unrouted:
            seed = min(unrouted, key=lambda c: cust_info[c]['ready_time'])
            feasible, _, _, _ = check_route_feasibility([seed])
            if not feasible:
                found_seed = False
                for s in sorted(unrouted, key=lambda c: cust_info[c]['ready_time']):
                    feasible, _, _, _ = check_route_feasibility([s])
                    if feasible:
                        seed = s
                        found_seed = True
                        break
                if not found_seed:
                    break
            
            route = [seed]
            unrouted.remove(seed)
            
            improved = True
            while improved:
                improved = False
                best_cust = None
                best_pos = None
                best_regret = -float('inf')
                best_cost_val = float('inf')
                
                for cid in list(unrouted):
                    result = try_insert(route, cid)
                    if result is not None:
                        pos, cost = result
                        # Simple: just use cost as negative regret (greedy)
                        # For regret: we'd need cost in other routes too, but we only have one open route
                        regret = -cost  # minimize cost => maximize -cost
                        if regret > best_regret or (regret == best_regret and cost < best_cost_val):
                            best_regret = regret
                            best_cost_val = cost
                            best_cust = cid
                            best_pos = pos
                
                if best_cust is not None:
                    route.insert(best_pos, best_cust)
                    unrouted.remove(best_cust)
                    improved = True
            
            routes.append(route)
        
        return routes
    
    # Build initial solution
    routes = solomon_insertion()
    
    if logger:
        logger.log(len(routes))
    
    best_routes = [r[:] for r in routes]
    best_num_vehicles = len(routes)
    
    # Try regret insertion too
    routes2 = regret_insertion()
    if len(routes2) < best_num_vehicles:
        best_routes = [r[:] for r in routes2]
        best_num_vehicles = len(routes2)
        if logger:
            logger.log(best_num_vehicles)
    
    # Try with different seeds
    for _ in range(min(20, num_customers)):
        if time.time() - start_time > time_limit * 0.1:
            break
        order = list(cust_ids)
        random.shuffle(order)
        r = solomon_insertion(order)
        if len(r) < best_num_vehicles:
            best_routes = [ri[:] for ri in r]
            best_num_vehicles = len(r)
            if logger:
                logger.log(best_num_vehicles)
    
    routes = [r[:] for r in best_routes]
    
    # Local search improvements to reduce number of vehicles
    def try_merge_routes(routes):
        """Try to merge routes to reduce vehicle count."""
        improved = True
        while improved:
            improved = False
            n_routes = len(routes)
            for i in range(n_routes):
                if time.time() - start_time > time_limit * 0.9:
                    return routes
                for j in range(n_routes):
                    if i == j:
                        continue
                    # Try to insert all customers of route j into route i
                    route_i = routes[i][:]
                    all_inserted = True
                    remaining = list(routes[j])
                    
                    for cid in remaining:
                        result = try_insert(route_i, cid)
                        if result is not None:
                            pos, cost = result
                            route_i.insert(pos, cid)
                        else:
                            all_inserted = False
                            break
                    
                    if all_inserted:
                        routes[i] = route_i
                        routes.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return routes
    
    routes = try_merge_routes(routes)
    
    if len(routes) < best_num_vehicles:
        best_routes = [r[:] for r in routes]
        best_num_vehicles = len(routes)
        if logger:
            logger.log(best_num_vehicles)
    
    # Ejection chain / route elimination: try to eliminate the smallest route
    def try_eliminate_route(routes, route_idx):
        """Try to redistribute customers from route_idx to other routes."""
        target_custs = routes[route_idx][:]
        other_routes = [r[:] for i, r in enumerate(routes) if i != route_idx]
        
        # Try all permutations of insertion order for small routes
        # For larger, try greedy
        for cid in target_custs:
            inserted = False
            best_route_idx = None
            best_pos = None
            best_increase = float('inf')
            
            for ri in range(len(other_routes)):
                result = try_insert(other_routes[ri], cid)
                if result is not None:
                    pos, cost = result
                    old_dist, _, _ = compute_route_details(other_routes[ri])
                    increase = cost - old_dist
                    if increase < best_increase:
                        best_increase = increase
                        best_route_idx = ri
                        best_pos = pos
                        inserted = True
            
            if inserted:
                other_routes[best_route_idx].insert(best_pos, cid)
            else:
                return None
        
        return other_routes
    
    def elimination_search(routes):
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit * 0.8:
                break
            # Try to eliminate smallest route first
            route_sizes = [(len(r), i) for i, r in enumerate(routes)]
            route_sizes.sort()
            
            for _, idx in route_sizes:
                result = try_eliminate_route(routes, idx)
                if result is not None:
                    routes = result
                    improved = True
                    break
        return routes
    
    routes = elimination_search(routes)
    
    if len(routes) < best_num_vehicles:
        best_routes = [r[:] for r in routes]
        best_num_vehicles = len(routes)
        if logger:
            logger.log(best_num_vehicles)
    
    # More aggressive: try inter-route relocate and swap to make room for elimination
    def inter_route_relocate(routes):
        """Move a customer from one route to another if feasible and improves flexibility."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit * 0.85:
                return routes
            for i in range(len(routes)):
                for pos_i in range(len(routes[i])):
                    cid = routes[i][pos_i]
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        new_route_i = routes[i][:pos_i] + routes[i][pos_i+1:]
                        result = try_insert(routes[j], cid)
                        if result is not None:
                            pos_j, _ = result
                            new_route_j = routes[j][:pos_j] + [cid] + routes[j][pos_j:]
                            
                            # Check if this helps (e.g., empties route i)
                            if len(new_route_i) == 0:
                                routes[i] = new_route_i
                                routes[j] = new_route_j
                                routes = [r for r in routes if len(r) > 0]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes
    
    routes = inter_route_relocate(best_routes[:])
    
    if len(routes) < best_num_vehicles:
        best_routes = [r[:] for r in routes]
        best_num_vehicles = len(routes)
        if logger:
            logger.log(best_num_vehicles)
    
    # ALNS-style destroy and repair
    def destroy_and_repair(routes, num_iterations=1000):
        nonlocal best_routes, best_num_vehicles
        
        current_routes = [r[:] for r in routes]
        current_nv = len(current_routes)
        
        for iteration in range(num_iterations):
            if time.time() - start_time > time_limit * 0.95:
                break
            
            # Destroy: remove random customers
            all_customers_in_routes = []
            for ri, route in enumerate(current_routes):
                for pos, cid in enumerate(route):
                    all_customers_in_routes.append((ri, pos, cid))
            
            if not all_customers_in_routes:
                break
            
            # Remove a fraction of customers
            num_remove = max(1, min(len(all_customers_in_routes) // 5, 15))
            removed = random.sample(all_customers_in_routes, min(num_remove, len(all_customers_in_routes)))
            
            removed_custs = [cid for _, _, cid in removed]
            
            new_routes = []
            removed_set = set(removed_custs)
            for route in current_routes:
                new_route = [c for c in route if c not in removed_set]
                if new_route:
                    new_routes.append(new_route)
            
            # Verify feasibility of remaining routes
            all_feasible = True
            for route in new_routes:
                f, _, _, _ = check_route_feasibility(route)
                if not f:
                    all_feasible = False
                    break
            
            if not all_feasible:
                continue
            
            # Repair: insert removed customers back
            random.shuffle(removed_custs)
            failed = False
            for cid in removed_custs:
                best_ri = None
                best_pos = None
                best_cost = float('inf')
                
                for ri in range(len(new_routes)):
                    result = try_insert(new_routes[ri], cid)
                    if result is not None:
                        pos, cost = result
                        if cost < best_cost:
                            best_cost = cost
                            best_ri = ri
                            best_pos = pos
                
                if best_ri is not None:
                    new_routes[best_ri].insert(best_pos, cid)
                else:
                    # Need new route
                    f, _, _, _ = check_route_feasibility([cid])
                    if f:
                        new_routes.append([cid])
                    else:
                        failed = True
                        break
            
            if failed:
                continue
            
            new_nv = len(new_routes)
            if new_nv < best_num_vehicles:
                best_routes = [r[:] for r in new_routes]
                best_num_vehicles = new_nv
                current_routes = [r[:] for r in new_routes]
                current_nv = new_nv
                if logger:
                    logger.log(best_num_vehicles)
            elif new_nv <= current_nv:
                current_routes = new_routes
                current_nv = new_nv
            elif random.random() < 0.1:  # Accept worse with small probability
                current_routes = new_routes
                current_nv = new_nv
        
        return best_routes
    
    best_routes = destroy_and_repair(best_routes)
    
    # Try one more round of elimination
    routes = elimination_search([r[:] for r in best_routes])
    if len(routes) < best_num_vehicles:
        best_routes = [r[:] for r in routes]
        best_num_vehicles = len(routes)
        if logger:
            logger.log(best_num_vehicles)
    
    # Focused elimination: try harder to eliminate each route
    def focused_elimination(routes):
        nonlocal best_routes, best_num_vehicles
        
        for target_idx in range(len(routes)):
            if time.time() - start_time > time_limit * 0.95:
                break
            
            target_custs = routes[target_idx][:]
            if len(target_custs) == 0:
                continue
            
            # Try all permutations for very small routes
            from itertools import permutations
            
            if len(target_custs) <= 6:
                perms = list(permutations(target_custs))
            else:
                perms = [target_custs]
                for _ in range(50):
                    p = list(target_custs)
                    random.shuffle(p)
                    perms.append(p)
            
            for perm in perms:
                if time.time() - start_time > time_limit * 0.95:
                    break
                other_routes = [r[:] for i, r in enumerate(routes) if i != target_idx]
                success = True
                
                for cid in perm:
                    inserted = False
                    best_ri = None
                    best_pos = None
                    best_increase = float('inf')
                    
                    for ri in range(len(other_routes)):
                        result = try_insert(other_routes[ri], cid)
                        if result is not None:
                            pos, cost = result
                            old_dist, _, _ = compute_route_details(other_routes[ri])
                            increase = cost - old_dist
                            if increase < best_increase:
                                best_increase = increase
                                best_ri = ri
                                best_pos = pos
                                inserted = True
                    
                    if inserted:
                        other_routes[best_ri].insert(best_pos, cid)
                    else:
                        success = False
                        break
                
                if success:
                    routes = other_routes
                    if len(routes) < best_num_vehicles:
                        best_routes = [r[:] for r in routes]
                        best_num_vehicles = len(routes)
                        if logger:
                            logger.log(best_num_vehicles)
                    return focused_elimination(routes)  # Recurse
        
        return routes
    
    focused_elimination([r[:] for r in best_routes])
    
    # Build final solution
    routes = best_routes
    all_dep_times = {}
    all_loads = {}
    total_distance = 0.0
    
    final_routes = []
    for route in routes:
        full_route = [0] + route + [0]
        final_routes.append(full_route)
        dist, dep_times, loads_dict = compute_route_details(route)
        total_distance += dist
        all_dep_times.update(dep_times)
        all_loads.update(loads_dict)
    
    # Convert keys to strings
    dep_times_str = {str(k): v for k, v in all_dep_times.items()}
    loads_str = {str(k): v for k, v in all_loads.items()}
    
    solution = {
        "objective_value": float(len(routes)),
        "num_vehicles": len(routes),
        "routes": final_routes,
        "total_distance": round(total_distance, 1),
        "departure_times": dep_times_str,
        "loads": loads_str
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