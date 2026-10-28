import argparse
import json
import math
import time
import itertools
from solution_logger import SolutionLogger

def solve(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_customers = instance['num_customers']
    num_nodes = instance['num_nodes']
    alpha = instance['alpha']
    nodes = instance['nodes']
    truck_tt = instance['truck_travel_times']
    drone_tt = instance['drone_travel_times']
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Depot is node 0, customers are 1..num_customers
    # Depot destination is node num_customers+1 (virtual)
    customers = list(range(1, num_customers + 1))
    depot = 0
    depot_dest = num_customers + 1
    
    best_solution = None
    best_obj = float('inf')
    
    def compute_route_time(route, tt):
        """Compute total travel time along a route using given travel time matrix."""
        total = 0
        for i in range(len(route) - 1):
            total += tt[route[i]][route[i+1]]
        return total
    
    def evaluate_solution(truck_customers, drone_customers, combined_customers):
        """Given partition of customers, find best routes and return objective."""
        # Truck visits: truck_customers + combined_customers
        # Drone visits: drone_customers + combined_customers
        truck_visits = sorted(truck_customers + combined_customers)
        drone_visits = sorted(drone_customers + combined_customers)
        
        if not truck_visits:
            return float('inf'), None, None, None
        if not drone_visits:
            return float('inf'), None, None, None
        
        # Check drone constraint: drone can't just go depot -> customer -> depot
        if len(drone_visits) == 1 and len(combined_customers) == 0:
            # drone only has one drone-only customer, which would violate the constraint
            return float('inf'), None, None, None
        
        # For small number of visits, try all permutations; otherwise use nearest neighbor
        def best_route(visits, tt, max_perm=10):
            if len(visits) == 0:
                return [0, 0], 0  # shouldn't happen
            if len(visits) <= max_perm:
                best_r = None
                best_t = float('inf')
                for perm in itertools.permutations(visits):
                    route = [0] + list(perm) + [0]
                    t = compute_route_time(route, tt)
                    if t < best_t:
                        best_t = t
                        best_r = route
                return best_r, best_t
            else:
                # Nearest neighbor heuristic with multiple starts
                best_r = None
                best_t = float('inf')
                for start_idx in range(min(len(visits), 5)):
                    route = [0]
                    remaining = set(visits)
                    # Start with different first customers
                    if start_idx == 0:
                        current = 0
                    else:
                        # Pick start_idx-th nearest to depot
                        sorted_by_dist = sorted(visits, key=lambda c: tt[0][c])
                        first = sorted_by_dist[min(start_idx - 1, len(sorted_by_dist) - 1)]
                        route.append(first)
                        remaining.remove(first)
                        current = first
                    
                    if current == 0:
                        # pick nearest
                        if remaining:
                            nxt = min(remaining, key=lambda c: tt[0][c])
                            route.append(nxt)
                            remaining.remove(nxt)
                            current = nxt
                    
                    while remaining:
                        nxt = min(remaining, key=lambda c: tt[current][c])
                        route.append(nxt)
                        remaining.remove(nxt)
                        current = nxt
                    route.append(0)
                    t = compute_route_time(route, tt)
                    if t < best_t:
                        best_t = t
                        best_r = route
                
                # Try 2-opt improvement
                best_r, best_t = two_opt(best_r, tt)
                return best_r, best_t
        
        truck_route, truck_time = best_route(truck_visits, truck_tt)
        drone_route, drone_time = best_route(drone_visits, drone_tt)
        
        # Check drone constraint: can't go depot -> single customer -> depot
        if len(drone_route) == 3:  # [0, c, 0]
            return float('inf'), None, None, None
        
        # Now compute arrival times considering combined customers
        # For combined customers, both truck and drone must arrive
        # Simple approach: compute arrival times independently, then adjust for combined
        
        # Actually we need to compute proper arrival times
        # For simplicity, the objective is max(truck_time, drone_time) if no combined customers
        # With combined customers, it's more complex due to synchronization
        
        if not combined_customers:
            obj = max(truck_time, drone_time)
            return obj, truck_route, drone_route, {}
        
        # With combined customers, we need to compute synchronized arrival times
        arrival = compute_synchronized_arrivals(truck_route, drone_route, combined_customers, truck_tt, drone_tt)
        if arrival is None:
            return float('inf'), None, None, None
        obj = arrival[depot_dest]
        return obj, truck_route, drone_route, arrival
    
    def two_opt(route, tt):
        """Apply 2-opt improvement to a route."""
        if len(route) <= 4:
            return route, compute_route_time(route, tt)
        improved = True
        best_t = compute_route_time(route, tt)
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Reverse segment from i to j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_t = compute_route_time(new_route, tt)
                    if new_t < best_t - 1e-9:
                        route = new_route
                        best_t = new_t
                        improved = True
                        break
                if improved:
                    break
        return route, best_t
    
    def compute_synchronized_arrivals(truck_route, drone_route, combined_customers, truck_tt, drone_tt):
        """Compute arrival times with synchronization at combined customers."""
        combined_set = set(combined_customers)
        
        # Build arrival time arrays for truck and drone independently first
        truck_arrival = {}
        t = 0
        for i in range(len(truck_route)):
            if i == 0:
                truck_arrival[truck_route[0]] = 0
            else:
                t += truck_tt[truck_route[i-1]][truck_route[i]]
                truck_arrival[truck_route[i]] = t
        
        drone_arrival = {}
        t = 0
        for i in range(len(drone_route)):
            if i == 0:
                drone_arrival[drone_route[0]] = 0
            else:
                t += drone_tt[drone_route[i-1]][drone_route[i]]
                drone_arrival[drone_route[i]] = t
        
        if not combined_customers:
            # Simple case
            arrival = {0: 0}
            for node in truck_route[1:-1]:
                arrival[node] = truck_arrival[node]
            for node in drone_route[1:-1]:
                if node in arrival:
                    arrival[node] = max(arrival[node], drone_arrival[node])
                else:
                    arrival[node] = drone_arrival[node]
            arrival[depot_dest] = max(truck_arrival[0], drone_arrival[0])
            return arrival
        
        # With synchronization, we need to propagate delays
        # This is iterative: at combined nodes, both must wait for the later one
        # Then subsequent nodes get delayed
        
        # We'll iterate until convergence
        arrival = {0: 0}
        
        # Initialize truck arrivals
        truck_arrivals_list = [0] * len(truck_route)
        for i in range(1, len(truck_route)):
            truck_arrivals_list[i] = truck_arrivals_list[i-1] + truck_tt[truck_route[i-1]][truck_route[i]]
        
        drone_arrivals_list = [0] * len(drone_route)
        for i in range(1, len(drone_route)):
            drone_arrivals_list[i] = drone_arrivals_list[i-1] + drone_tt[drone_route[i-1]][drone_route[i]]
        
        # Map node to index in each route
        truck_node_idx = {}
        for i, n in enumerate(truck_route):
            if i == 0:
                truck_node_idx[('depot_start', n)] = i
            elif i == len(truck_route) - 1:
                truck_node_idx[('depot_end', n)] = i
            else:
                truck_node_idx[n] = i
        
        drone_node_idx = {}
        for i, n in enumerate(drone_route):
            if i == 0:
                drone_node_idx[('depot_start', n)] = i
            elif i == len(drone_route) - 1:
                drone_node_idx[('depot_end', n)] = i
            else:
                drone_node_idx[n] = i
        
        # Iterate to propagate synchronization delays
        for _ in range(len(combined_customers) + 2):
            # Forward pass on truck
            for i in range(1, len(truck_route)):
                truck_arrivals_list[i] = truck_arrivals_list[i-1] + truck_tt[truck_route[i-1]][truck_route[i]]
                node = truck_route[i]
                if node in combined_set and node in drone_node_idx:
                    didx = drone_node_idx[node]
                    truck_arrivals_list[i] = max(truck_arrivals_list[i], drone_arrivals_list[didx])
            
            # Forward pass on drone
            for i in range(1, len(drone_route)):
                drone_arrivals_list[i] = drone_arrivals_list[i-1] + drone_tt[drone_route[i-1]][drone_route[i]]
                node = drone_route[i]
                if node in combined_set and node in truck_node_idx:
                    tidx = truck_node_idx[node]
                    drone_arrivals_list[i] = max(drone_arrivals_list[i], truck_arrivals_list[tidx])
        
        # Build arrival dict
        for i, n in enumerate(truck_route):
            if i == 0:
                continue
            elif i == len(truck_route) - 1:
                arrival[depot_dest] = truck_arrivals_list[i]
            else:
                arrival[n] = truck_arrivals_list[i]
        
        for i, n in enumerate(drone_route):
            if i == 0:
                continue
            elif i == len(drone_route) - 1:
                if depot_dest in arrival:
                    arrival[depot_dest] = max(arrival[depot_dest], drone_arrivals_list[i])
                else:
                    arrival[depot_dest] = drone_arrivals_list[i]
            else:
                if n in arrival:
                    arrival[n] = max(arrival[n], drone_arrivals_list[i])
                else:
                    arrival[n] = drone_arrivals_list[i]
        
        return arrival
    
    def verify_drone_adjacency_constraint(drone_route, combined_set):
        """Check that for any pair of distinct customers connected by drone arcs,
        at least one must be combined. Also check no depot->customer->depot for single customer."""
        if len(drone_route) <= 2:
            return True
        if len(drone_route) == 3:
            # depot -> customer -> depot: not allowed
            return False
        for i in range(len(drone_route) - 1):
            u, v = drone_route[i], drone_route[i+1]
            if u != 0 and v != 0:  # both are customers
                if u not in combined_set and v not in combined_set:
                    return False
        return True
    
    # Strategy: enumerate possible partitions using heuristics
    # For small instances, try more combinations
    
    def try_partition(truck_only, drone_only, combined):
        combined_set = set(combined)
        
        # Get visits
        truck_visits = list(truck_only) + list(combined)
        drone_visits = list(drone_only) + list(combined)
        
        if not truck_visits or not drone_visits:
            return float('inf'), None
        
        # Check: drone can't just have one customer and no combined
        if len(drone_visits) == 1:
            return float('inf'), None
        
        max_perm = 10
        
        def get_best_route(visits, tt, max_p=max_perm):
            if len(visits) <= max_p:
                best_r = None
                best_t = float('inf')
                for perm in itertools.permutations(visits):
                    route = [0] + list(perm) + [0]
                    t = compute_route_time(route, tt)
                    if t < best_t:
                        best_t = t
                        best_r = route
                return best_r, best_t
            else:
                # Multiple nearest neighbor starts + 2-opt
                best_r = None
                best_t = float('inf')
                for start_node in [None] + visits[:min(5, len(visits))]:
                    route = [0]
                    remaining = set(visits)
                    if start_node is not None and start_node in remaining:
                        route.append(start_node)
                        remaining.remove(start_node)
                    current = route[-1]
                    while remaining:
                        nxt = min(remaining, key=lambda c: tt[current][c])
                        route.append(nxt)
                        remaining.remove(nxt)
                        current = nxt
                    route.append(0)
                    route, t = two_opt(route, tt)
                    if t < best_t:
                        best_t = t
                        best_r = route
                return best_r, best_t
        
        # For drone route, we need to respect adjacency constraint
        # If no combined customers, consecutive drone customers must not happen
        # Actually the constraint says: for any pair of distinct customers with drone arc,
        # at least one must be combined
        
        # First get optimal truck route
        truck_route, truck_time = get_best_route(truck_visits, truck_tt)
        
        # For drone, need to ensure adjacency constraint
        if len(drone_visits) <= max_perm:
            best_dr = None
            best_dt = float('inf')
            for perm in itertools.permutations(drone_visits):
                route = [0] + list(perm) + [0]
                if not verify_drone_adjacency_constraint(route, combined_set):
                    continue
                t = compute_route_time(route, drone_tt)
                if t < best_dt:
                    best_dt = t
                    best_dr = route
            if best_dr is None:
                return float('inf'), None
            drone_route = best_dr
        else:
            drone_route, _ = get_best_route(drone_visits, drone_tt)
            if not verify_drone_adjacency_constraint(drone_route, combined_set):
                # Try to fix by reordering
                # Ensure combined customers separate drone-only customers
                drone_only_list = [c for c in drone_visits if c not in combined_set]
                combined_list = [c for c in drone_visits if c in combined_set]
                
                if len(combined_list) == 0 and len(drone_only_list) > 1:
                    return float('inf'), None
                
                # Interleave: combined, drone_only, combined, drone_only, ...
                # This ensures each drone-only is adjacent to a combined
                interleaved = []
                di = 0
                ci = 0
                while di < len(drone_only_list) or ci < len(combined_list):
                    if ci < len(combined_list):
                        interleaved.append(combined_list[ci])
                        ci += 1
                    if di < len(drone_only_list):
                        interleaved.append(drone_only_list[di])
                        di += 1
                
                drone_route = [0] + interleaved + [0]
                if not verify_drone_adjacency_constraint(drone_route, combined_set):
                    return float('inf'), None
                drone_route, _ = two_opt_constrained(drone_route, drone_tt, combined_set)
        
        # Compute synchronized arrival times
        arrival = compute_synchronized_arrivals(truck_route, drone_route, list(combined), truck_tt, drone_tt)
        if arrival is None:
            return float('inf'), None
        
        obj = arrival[depot_dest]
        
        # Build solution
        customer_types = {}
        for c in truck_only:
            customer_types[str(c)] = 'truck'
        for c in drone_only:
            customer_types[str(c)] = 'drone'
        for c in combined:
            customer_types[str(c)] = 'combined'
        
        arrival_times = {str(0): 0}
        for c in customers:
            if c in arrival:
                arrival_times[str(c)] = arrival[c]
        arrival_times[str(depot_dest)] = obj
        
        sol = {
            'objective_value': obj,
            'truck_route': truck_route,
            'drone_route': drone_route,
            'customer_types': customer_types,
            'arrival_times': arrival_times
        }
        
        return obj, sol
    
    def two_opt_constrained(route, tt, combined_set):
        """2-opt that respects drone adjacency constraint."""
        best_t = compute_route_time(route, tt)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if not verify_drone_adjacency_constraint(new_route, combined_set):
                        continue
                    new_t = compute_route_time(new_route, tt)
                    if new_t < best_t - 1e-9:
                        route = new_route
                        best_t = new_t
                        improved = True
                        break
                if improved:
                    break
        return route, best_t
    
    # === Main solving logic ===
    
    # For small instances, try enumeration of partitions
    # Each customer can be: truck, drone, or combined
    # But we need constraints satisfied
    
    # First, simple baseline: all truck
    truck_route_all = None
    if num_customers <= 12:
        best_t = float('inf')
        for perm in itertools.permutations(customers):
            route = [0] + list(perm) + [0]
            t = compute_route_time(route, truck_tt)
            if t < best_t:
                best_t = t
                truck_route_all = route
    else:
        # Nearest neighbor + 2-opt
        route = [0]
        remaining = set(customers)
        current = 0
        while remaining:
            nxt = min(remaining, key=lambda c: truck_tt[current][c])
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt
        route.append(0)
        truck_route_all, best_t = two_opt(route, truck_tt)
    
    # All truck solution
    arrival_times = {str(0): 0}
    t = 0
    for i in range(1, len(truck_route_all)):
        t += truck_tt[truck_route_all[i-1]][truck_route_all[i]]
        if i < len(truck_route_all) - 1:
            arrival_times[str(truck_route_all[i])] = t
    arrival_times[str(depot_dest)] = t
    
    # Drone route: needs at least one customer. For all-truck, drone goes nowhere useful.
    # Actually every customer must be served, and drone must also go depot->...->depot
    # But if all customers are truck-only, drone still needs a route
    # Wait, re-reading: "The drone must likewise depart from the depot origin exactly once 
    # and return to the depot destination exactly once, traversing its own sequence of arcs."
    # But drone arcs leaving a customer = 1 only if drone-only or combined. So if no 
    # drone-only or combined customers, the drone goes directly depot->depot? But that's not allowed.
    # Actually: "No arc connects the depot origin directly to the depot destination"
    # So we need at least one combined or drone-only customer.
    
    # Hmm, so we must have at least one customer that is combined or drone-only.
    # Let's reconsider. The drone must traverse from depot origin to depot destination.
    # It needs at least one intermediate node. So at least one customer must be combined or drone-only.
    
    # Also: drone can't go depot -> single_customer -> depot (constraint at end of problem description)
    # So drone needs at least 2 customers in its route.
    
    # So we need at least 2 customers as combined or drone-only.
    # Actually re-reading: "the drone cannot simultaneously use the arc from the depot origin 
    # to that customer and the arc from that customer to the depot destination" for ANY customer.
    # This means if drone visits customer c, it can't be that c is both the first and last customer
    # in the drone's route. So drone needs at least 2 customers.
    
    # And the adjacency constraint: consecutive drone-only customers can't be adjacent.
    # Between any pair of customers with a drone arc, at least one must be combined.
    # So drone-only customers must always be adjacent to combined customers in the drone route.
    
    # Simplest valid solution: pick 2 customers as combined, rest as truck-only
    # Both truck and drone visit the 2 combined customers
    
    # Let's enumerate strategies
    
    # Strategy 1: All customers are truck + pick some as combined for drone
    # Strategy 2: Some drone-only, some truck-only, some combined
    
    def generate_partitions():
        """Generate candidate partitions to evaluate."""
        partitions = []
        
        if num_customers <= 15:
            # For small instances, try many partitions
            # Enumerate subsets for drone service
            
            # Key insight: drone needs ≥2 customers, and adjacency constraint must hold
            # Combined customers can be adjacent to anything in drone route
            # Drone-only must be adjacent to combined in drone route
            
            # Strategy: try all pairs of combined customers (drone visits them + truck visits them)
            # And optionally add drone-only customers between combined ones
            
            for size_combined in range(2, min(num_customers + 1, 8)):
                if time.time() - start_time > time_limit * 0.7:
                    break
                for combined in itertools.combinations(customers, size_combined):
                    if time.time() - start_time > time_limit * 0.6:
                        break
                    combined_set = set(combined)
                    remaining = [c for c in customers if c not in combined_set]
                    
                    # All remaining as truck-only
                    partitions.append((remaining, [], list(combined)))
                    
                    # Try moving some remaining to drone-only
                    if len(combined) >= 1 and len(remaining) > 0:
                        for num_drone in range(1, min(len(remaining) + 1, 5)):
                            if time.time() - start_time > time_limit * 0.5:
                                break
                            for drone_only in itertools.combinations(remaining, num_drone):
                                truck_only = [c for c in remaining if c not in set(drone_only)]
                                partitions.append((truck_only, list(drone_only), list(combined)))
            
            # Also try: 2 combined, rest split between truck and drone
            for c1, c2 in itertools.combinations(customers, 2):
                combined = [c1, c2]
                remaining = [c for c in customers if c != c1 and c != c2]
                # Try all ways to split remaining
                if len(remaining) <= 10:
                    for mask in range(1 << len(remaining)):
                        truck_only = []
                        drone_only = []
                        for i, c in enumerate(remaining):
                            if mask & (1 << i):
                                drone_only.append(c)
                            else:
                                truck_only.append(c)
                        if not truck_only and not drone_only:
                            partitions.append(([], [], combined))
                        else:
                            partitions.append((truck_only, drone_only, combined))
        else:
            # Larger instances: use heuristic partitions
            # Try different numbers of drone customers
            
            # Sort customers by distance from depot
            drone_savings = []
            for c in customers:
                truck_dist = truck_tt[0][c] + truck_tt[c][0]
                drone_dist = drone_tt[0][c] + drone_tt[c][0]
                saving = truck_dist - drone_dist
                drone_savings.append((saving, c))
            drone_savings.sort(reverse=True)
            
            # Try giving top-k customers to drone as combined
            for k in range(2, min(num_customers, 10)):
                combined = [c for _, c in drone_savings[:k]]
                truck_only = [c for c in customers if c not in set(combined)]
                partitions.append((truck_only, [], combined))
                
                # Also try some as drone-only
                if k >= 3:
                    # Make the ones with highest savings drone-only, keep 2 as combined
                    for num_comb in range(2, k):
                        comb = [c for _, c in drone_savings[:num_comb]]
                        drone_only = [c for _, c in drone_savings[num_comb:k]]
                        truck_only2 = [c for c in customers if c not in set(comb) and c not in set(drone_only)]
                        partitions.append((truck_only2, drone_only, comb))
        
        return partitions
    
    # First, establish baseline with simple strategy: 2 combined customers (nearest to depot by drone)
    drone_dists = [(drone_tt[0][c] + drone_tt[c][0], c) for c in customers]
    drone_dists.sort()
    
    if num_customers >= 2:
        # Pick 2 nearest customers as combined
        combined_init = [drone_dists[0][1], drone_dists[1][1]]
        truck_only_init = [c for c in customers if c not in set(combined_init)]
        
        obj, sol = try_partition(truck_only_init, [], combined_init)
        if sol is not None and obj < best_obj:
            best_obj = obj
            best_solution = sol
            if logger:
                logger.log(best_obj)
    
    # Try all-combined (every customer visited by both)
    if num_customers <= 12:
        obj, sol = try_partition([], [], customers)
        if sol is not None and obj < best_obj:
            best_obj = obj
            best_solution = sol
            if logger:
                logger.log(best_obj)
    
    # Now try systematic partitions
    if num_customers <= 8:
        # Enumerate all possible 3-partitions
        # Each customer: truck(0), drone(1), combined(2)
        for code in range(3 ** num_customers):
            if time.time() - start_time > time_limit * 0.85:
                break
            assignment = []
            c = code
            for _ in range(num_customers):
                assignment.append(c % 3)
                c //= 3
            
            truck_only = [customers[i] for i in range(num_customers) if assignment[i] == 0]
            drone_only = [customers[i] for i in range(num_customers) if assignment[i] == 1]
            combined = [customers[i] for i in range(num_customers) if assignment[i] == 2]
            
            # Basic feasibility checks
            drone_visits = drone_only + combined
            truck_visits = truck_only + combined
            
            if len(drone_visits) < 2:
                continue
            if len(truck_visits) < 1:
                continue
            
            obj, sol = try_partition(truck_only, drone_only, combined)
            if sol is not None and obj < best_obj:
                best_obj = obj
                best_solution = sol
                if logger:
                    logger.log(best_obj)
    else:
        # Use heuristic partitions
        partitions = generate_partitions()
        
        for truck_only, drone_only, combined in partitions:
            if time.time() - start_time > time_limit * 0.85:
                break
            
            drone_visits = drone_only + combined
            truck_visits = truck_only + combined
            
            if len(drone_visits) < 2:
                continue
            if len(truck_visits) < 1:
                continue
            
            obj, sol = try_partition(truck_only, drone_only, combined)
            if sol is not None and obj < best_obj:
                best_obj = obj
                best_solution = sol
                if logger:
                    logger.log(best_obj)
    
    # Local search improvement if we have time
    if best_solution is not None and time.time() - start_time < time_limit * 0.9:
        # Try swapping customer types
        current_types = best_solution['customer_types']
        improved = True
        while improved and time.time() - start_time < time_limit * 0.9:
            improved = False
            for c in customers:
                if time.time() - start_time > time_limit * 0.9:
                    break
                current_type = current_types[str(c)]
                for new_type in ['truck', 'drone', 'combined']:
                    if new_type == current_type:
                        continue
                    
                    # Create new partition
                    truck_only = []
                    drone_only = []
                    combined = []
                    for c2 in customers:
                        if c2 == c:
                            t = new_type
                        else:
                            t = current_types[str(c2)]
                        if t == 'truck':
                            truck_only.append(c2)
                        elif t == 'drone':
                            drone_only.append(c2)
                        else:
                            combined.append(c2)
                    
                    drone_visits = drone_only + combined
                    truck_visits = truck_only + combined
                    
                    if len(drone_visits) < 2 or len(truck_visits) < 1:
                        continue
                    
                    obj, sol = try_partition(truck_only, drone_only, combined)
                    if sol is not None and obj < best_obj - 1e-9:
                        best_obj = obj
                        best_solution = sol
                        current_types = sol['customer_types']
                        improved = True
                        if logger:
                            logger.log(best_obj)
                        break
                if improved:
                    break
    
    # Try pairwise swaps
    if best_solution is not None and time.time() - start_time < time_limit * 0.9 and num_customers <= 20:
        current_types = best_solution['customer_types']
        improved = True
        while improved and time.time() - start_time < time_limit * 0.9:
            improved = False
            for c1, c2 in itertools.combinations(customers, 2):
                if time.time() - start_time > time_limit * 0.9:
                    break
                t1 = current_types[str(c1)]
                t2 = current_types[str(c2)]
                if t1 == t2:
                    continue
                
                # Swap types
                truck_only = []
                drone_only = []
                combined = []
                for c in customers:
                    if c == c1:
                        t = t2
                    elif c == c2:
                        t = t1
                    else:
                        t = current_types[str(c)]
                    if t == 'truck':
                        truck_only.append(c)
                    elif t == 'drone':
                        drone_only.append(c)
                    else:
                        combined.append(c)
                
                drone_visits = drone_only + combined
                truck_visits = truck_only + combined
                
                if len(drone_visits) < 2 or len(truck_visits) < 1:
                    continue
                
                obj, sol = try_partition(truck_only, drone_only, combined)
                if sol is not None and obj < best_obj - 1e-9:
                    best_obj = obj
                    best_solution = sol
                    current_types = sol['customer_types']
                    improved = True
                    if logger:
                        logger.log(best_obj)
                    break
    
    # If we still don't have a solution, create a simple valid one
    if best_solution is None:
        # Make first 2 customers combined, rest truck
        if num_customers >= 2:
            combined = [customers[0], customers[1]]
            truck_only = customers[2:]
            
            truck_visits = truck_only + combined
            drone_visits = combined
            
            truck_route = [0] + truck_visits + [0]
            drone_route = [0] + drone_visits + [0]
            
            arrival = compute_synchronized_arrivals(truck_route, drone_route, combined, truck_tt, drone_tt)
            obj = arrival[depot_dest]
            
            customer_types = {}
            for c in truck_only:
                customer_types[str(c)] = 'truck'
            for c in combined:
                customer_types[str(c)] = 'combined'
            
            arrival_times = {str(0): 0}
            for c in customers:
                if c in arrival:
                    arrival_times[str(c)] = arrival[c]
            arrival_times[str(depot_dest)] = obj
            
            best_solution = {
                'objective_value': obj,
                'truck_route': truck_route,
                'drone_route': drone_route,
                'customer_types': customer_types,
                'arrival_times': arrival_times
            }
            best_obj = obj
            if logger:
                logger.log(best_obj)
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)