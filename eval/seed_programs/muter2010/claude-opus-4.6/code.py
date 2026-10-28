import argparse
import json
import math
import time
import random
import sys
from solution_logger import SolutionLogger


def euclidean_distance(a, b):
    return math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2)


def compute_dist_matrix(nodes):
    n = len(nodes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(nodes[i], nodes[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def check_route_feasibility(route, nodes, dist, capacity):
    """Check if a route is feasible (capacity and time windows)."""
    if not route:
        return True, 0.0
    
    total_demand = sum(nodes[c]['demand'] for c in route)
    if total_demand > capacity:
        return False, float('inf')
    
    total_dist = 0.0
    current_time = nodes[0]['ready_time']
    prev = 0  # depot
    
    for c in route:
        travel = dist[prev][c]
        total_dist += travel
        arrival = current_time + travel
        if arrival > nodes[c]['due_date']:
            return False, float('inf')
        start_service = max(arrival, nodes[c]['ready_time'])
        current_time = start_service + nodes[c]['service_time']
        prev = c
    
    # Return to depot
    travel = dist[prev][0]
    total_dist += travel
    arrival = current_time + travel
    if arrival > nodes[0]['due_date']:
        return False, float('inf')
    
    return True, total_dist


def route_distance(route, dist):
    """Calculate total distance of a route."""
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total


def route_time_check(route, nodes, dist):
    """Check time window feasibility and return arrival times."""
    if not route:
        return True, []
    
    current_time = nodes[0]['ready_time']
    prev = 0
    arrivals = []
    
    for c in route:
        travel = dist[prev][c]
        arrival = current_time + travel
        if arrival > nodes[c]['due_date']:
            return False, []
        start_service = max(arrival, nodes[c]['ready_time'])
        arrivals.append(start_service)
        current_time = start_service + nodes[c]['service_time']
        prev = c
    
    # Check return to depot
    travel = dist[prev][0]
    arrival = current_time + travel
    if arrival > nodes[0]['due_date']:
        return False, []
    
    return True, arrivals


def solomon_insertion(nodes, dist, capacity, time_limit_abs):
    """Solomon's I1 insertion heuristic with multiple seed strategies."""
    n = len(nodes)  # includes depot at index 0
    num_customers = n - 1
    
    best_routes = None
    best_total_dist = float('inf')
    
    # Try different seed selection strategies
    for seed_strategy in range(3):
        if time.time() > time_limit_abs:
            break
            
        unrouted = set(range(1, n))
        routes = []
        
        while unrouted:
            if time.time() > time_limit_abs:
                break
            
            # Select seed customer
            if seed_strategy == 0:
                # Farthest from depot
                seed = max(unrouted, key=lambda c: dist[0][c])
            elif seed_strategy == 1:
                # Earliest deadline
                seed = min(unrouted, key=lambda c: nodes[c]['due_date'])
            else:
                # Earliest ready time
                seed = min(unrouted, key=lambda c: nodes[c]['ready_time'])
            
            route = [seed]
            unrouted.remove(seed)
            route_demand = nodes[seed]['demand']
            
            improved = True
            while improved and unrouted:
                improved = False
                best_customer = None
                best_position = None
                best_cost = float('inf')
                
                for c in unrouted:
                    if route_demand + nodes[c]['demand'] > capacity:
                        continue
                    
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        
                        # Quick distance delta
                        if pos == 0:
                            prev_node = 0
                            next_node = route[0]
                        elif pos == len(route):
                            prev_node = route[-1]
                            next_node = 0
                        else:
                            prev_node = route[pos - 1]
                            next_node = route[pos]
                        
                        if pos == 0:
                            old_cost = dist[0][route[0]]
                            new_cost = dist[0][c] + dist[c][route[0]]
                        elif pos == len(route):
                            old_cost = dist[route[-1]][0]
                            new_cost = dist[route[-1]][c] + dist[c][0]
                        else:
                            old_cost = dist[route[pos-1]][route[pos]]
                            new_cost = dist[route[pos-1]][c] + dist[c][route[pos]]
                        
                        delta = new_cost - old_cost
                        
                        if delta < best_cost:
                            feasible, _ = route_time_check(new_route, nodes, dist)
                            if feasible:
                                best_cost = delta
                                best_customer = c
                                best_position = pos
                
                if best_customer is not None:
                    route.insert(best_position, best_customer)
                    unrouted.remove(best_customer)
                    route_demand += nodes[best_customer]['demand']
                    improved = True
            
            routes.append(route)
        
        total_dist = sum(route_distance(r, dist) for r in routes)
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_routes = [r[:] for r in routes]
    
    return best_routes, best_total_dist


def nearest_neighbor_construction(nodes, dist, capacity):
    """Construct initial solution using nearest neighbor heuristic."""
    n = len(nodes)
    unrouted = set(range(1, n))
    routes = []
    
    while unrouted:
        route = []
        route_demand = 0
        current_time = nodes[0]['ready_time']
        prev = 0
        
        improved = True
        while improved:
            improved = False
            best_next = None
            best_dist = float('inf')
            
            for c in unrouted:
                if route_demand + nodes[c]['demand'] > capacity:
                    continue
                
                travel = dist[prev][c]
                arrival = current_time + travel
                
                if arrival > nodes[c]['due_date']:
                    continue
                
                start_service = max(arrival, nodes[c]['ready_time'])
                departure = start_service + nodes[c]['service_time']
                
                # Check if we can return to depot
                return_time = departure + dist[c][0]
                if return_time > nodes[0]['due_date']:
                    continue
                
                if travel < best_dist:
                    best_dist = travel
                    best_next = c
            
            if best_next is not None:
                route.append(best_next)
                unrouted.remove(best_next)
                route_demand += nodes[best_next]['demand']
                travel = dist[prev][best_next]
                arrival = current_time + travel
                start_service = max(arrival, nodes[best_next]['ready_time'])
                current_time = start_service + nodes[best_next]['service_time']
                prev = best_next
                improved = True
        
        if not route:
            # Force insert the nearest unrouted customer
            if unrouted:
                c = min(unrouted, key=lambda x: dist[0][x])
                route.append(c)
                unrouted.remove(c)
        
        routes.append(route)
    
    return routes


def or_opt_intra(route, nodes, dist, capacity):
    """Or-opt moves within a single route (move segments of 1, 2, or 3 customers)."""
    improved = True
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            if improved:
                break
            for i in range(len(route)):
                if i + seg_len > len(route):
                    break
                if improved:
                    break
                segment = route[i:i + seg_len]
                remaining = route[:i] + route[i + seg_len:]
                
                for j in range(len(remaining) + 1):
                    if j == i:
                        continue
                    new_route = remaining[:j] + segment + remaining[j:]
                    feasible, _ = check_route_feasibility(new_route, nodes, dist, capacity)
                    if feasible:
                        new_dist = route_distance(new_route, dist)
                        old_dist = route_distance(route, dist)
                        if new_dist < old_dist - 1e-10:
                            route = new_route
                            improved = True
                            break
    return route


def two_opt_intra(route, nodes, dist, capacity):
    """2-opt improvement within a single route."""
    improved = True
    while improved:
        improved = False
        n = len(route)
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                feasible, _ = check_route_feasibility(new_route, nodes, dist, capacity)
                if feasible:
                    new_dist = route_distance(new_route, dist)
                    old_dist = route_distance(route, dist)
                    if new_dist < old_dist - 1e-10:
                        route = new_route
                        improved = True
                        break
    return route


def relocate_inter(routes, nodes, dist, capacity):
    """Relocate a customer from one route to another."""
    improved = True
    while improved:
        improved = False
        for r1_idx in range(len(routes)):
            if improved:
                break
            for pos1 in range(len(routes[r1_idx])):
                if improved:
                    break
                customer = routes[r1_idx][pos1]
                
                # Cost of removing customer from route r1
                old_r1 = routes[r1_idx][:]
                new_r1 = old_r1[:pos1] + old_r1[pos1 + 1:]
                
                old_r1_dist = route_distance(old_r1, dist)
                new_r1_dist = route_distance(new_r1, dist) if new_r1 else 0.0
                removal_saving = old_r1_dist - new_r1_dist
                
                for r2_idx in range(len(routes)):
                    if r2_idx == r1_idx:
                        continue
                    if improved:
                        break
                    
                    # Check capacity
                    r2_demand = sum(nodes[c]['demand'] for c in routes[r2_idx])
                    if r2_demand + nodes[customer]['demand'] > capacity:
                        continue
                    
                    old_r2_dist = route_distance(routes[r2_idx], dist)
                    
                    best_insert_pos = None
                    best_insert_cost = float('inf')
                    
                    for pos2 in range(len(routes[r2_idx]) + 1):
                        new_r2 = routes[r2_idx][:pos2] + [customer] + routes[r2_idx][pos2:]
                        feasible, _ = check_route_feasibility(new_r2, nodes, dist, capacity)
                        if feasible:
                            new_r2_dist = route_distance(new_r2, dist)
                            insertion_cost = new_r2_dist - old_r2_dist
                            if insertion_cost < best_insert_cost:
                                best_insert_cost = insertion_cost
                                best_insert_pos = pos2
                    
                    if best_insert_pos is not None:
                        total_change = best_insert_cost - removal_saving
                        if total_change < -1e-10:
                            # Check new_r1 feasibility
                            if new_r1:
                                feasible_r1, _ = check_route_feasibility(new_r1, nodes, dist, capacity)
                            else:
                                feasible_r1 = True
                            
                            if feasible_r1:
                                new_r2 = routes[r2_idx][:best_insert_pos] + [customer] + routes[r2_idx][best_insert_pos:]
                                routes[r1_idx] = new_r1
                                routes[r2_idx] = new_r2
                                improved = True
        
        # Remove empty routes
        routes = [r for r in routes if r]
    
    return routes


def exchange_inter(routes, nodes, dist, capacity):
    """Exchange customers between two routes (swap)."""
    improved = True
    while improved:
        improved = False
        for r1_idx in range(len(routes)):
            if improved:
                break
            for r2_idx in range(r1_idx + 1, len(routes)):
                if improved:
                    break
                for pos1 in range(len(routes[r1_idx])):
                    if improved:
                        break
                    for pos2 in range(len(routes[r2_idx])):
                        c1 = routes[r1_idx][pos1]
                        c2 = routes[r2_idx][pos2]
                        
                        # Check capacity
                        r1_demand = sum(nodes[c]['demand'] for c in routes[r1_idx])
                        r2_demand = sum(nodes[c]['demand'] for c in routes[r2_idx])
                        
                        new_r1_demand = r1_demand - nodes[c1]['demand'] + nodes[c2]['demand']
                        new_r2_demand = r2_demand - nodes[c2]['demand'] + nodes[c1]['demand']
                        
                        if new_r1_demand > capacity or new_r2_demand > capacity:
                            continue
                        
                        new_r1 = routes[r1_idx][:]
                        new_r1[pos1] = c2
                        new_r2 = routes[r2_idx][:]
                        new_r2[pos2] = c1
                        
                        f1, _ = check_route_feasibility(new_r1, nodes, dist, capacity)
                        if not f1:
                            continue
                        f2, _ = check_route_feasibility(new_r2, nodes, dist, capacity)
                        if not f2:
                            continue
                        
                        old_dist = route_distance(routes[r1_idx], dist) + route_distance(routes[r2_idx], dist)
                        new_dist = route_distance(new_r1, dist) + route_distance(new_r2, dist)
                        
                        if new_dist < old_dist - 1e-10:
                            routes[r1_idx] = new_r1
                            routes[r2_idx] = new_r2
                            improved = True
                            break
    
    return routes


def cross_exchange(routes, nodes, dist, capacity):
    """Cross exchange: swap tails of two routes."""
    improved = True
    while improved:
        improved = False
        for r1_idx in range(len(routes)):
            if improved:
                break
            for r2_idx in range(r1_idx + 1, len(routes)):
                if improved:
                    break
                r1 = routes[r1_idx]
                r2 = routes[r2_idx]
                for i in range(len(r1) + 1):
                    if improved:
                        break
                    for j in range(len(r2) + 1):
                        new_r1 = r1[:i] + r2[j:]
                        new_r2 = r2[:j] + r1[i:]
                        
                        # Check capacity
                        d1 = sum(nodes[c]['demand'] for c in new_r1)
                        d2 = sum(nodes[c]['demand'] for c in new_r2)
                        if d1 > capacity or d2 > capacity:
                            continue
                        
                        f1, _ = check_route_feasibility(new_r1, nodes, dist, capacity)
                        if not f1:
                            continue
                        f2, _ = check_route_feasibility(new_r2, nodes, dist, capacity)
                        if not f2:
                            continue
                        
                        old_dist = route_distance(r1, dist) + route_distance(r2, dist)
                        new_dist = route_distance(new_r1, dist) + route_distance(new_r2, dist)
                        
                        if new_dist < old_dist - 1e-10:
                            routes[r1_idx] = new_r1
                            routes[r2_idx] = new_r2
                            improved = True
                            break
        
        routes = [r for r in routes if r]
    
    return routes


def local_search(routes, nodes, dist, capacity, time_limit_abs):
    """Apply local search operators."""
    improved = True
    iteration = 0
    while improved:
        if time.time() > time_limit_abs:
            break
        improved = False
        iteration += 1
        
        # Intra-route improvements
        for i in range(len(routes)):
            if time.time() > time_limit_abs:
                break
            old_dist = route_distance(routes[i], dist)
            new_route = two_opt_intra(routes[i], nodes, dist, capacity)
            new_dist = route_distance(new_route, dist)
            if new_dist < old_dist - 1e-10:
                routes[i] = new_route
                improved = True
        
        for i in range(len(routes)):
            if time.time() > time_limit_abs:
                break
            old_dist = route_distance(routes[i], dist)
            new_route = or_opt_intra(routes[i], nodes, dist, capacity)
            new_dist = route_distance(new_route, dist)
            if new_dist < old_dist - 1e-10:
                routes[i] = new_route
                improved = True
        
        if time.time() > time_limit_abs:
            break
        
        # Inter-route improvements
        old_total = sum(route_distance(r, dist) for r in routes)
        routes = relocate_inter(routes, nodes, dist, capacity)
        routes = [r for r in routes if r]
        new_total = sum(route_distance(r, dist) for r in routes)
        if new_total < old_total - 1e-10:
            improved = True
        
        if time.time() > time_limit_abs:
            break
        
        old_total = sum(route_distance(r, dist) for r in routes)
        routes = exchange_inter(routes, nodes, dist, capacity)
        new_total = sum(route_distance(r, dist) for r in routes)
        if new_total < old_total - 1e-10:
            improved = True
        
        if time.time() > time_limit_abs:
            break
        
        old_total = sum(route_distance(r, dist) for r in routes)
        routes = cross_exchange(routes, nodes, dist, capacity)
        routes = [r for r in routes if r]
        new_total = sum(route_distance(r, dist) for r in routes)
        if new_total < old_total - 1e-10:
            improved = True
    
    return routes


def perturbation(routes, nodes, dist, capacity, rng):
    """Random perturbation for ILS: remove some customers and reinsert."""
    if not routes:
        return routes
    
    all_customers = []
    for r in routes:
        all_customers.extend(r)
    
    if len(all_customers) <= 1:
        return routes
    
    # Remove random customers (between 1 and min(5, n/3))
    num_remove = min(max(1, len(all_customers) // 5), 8)
    num_remove = min(num_remove, len(all_customers))
    
    to_remove = set(rng.sample(all_customers, num_remove))
    
    new_routes = []
    for r in routes:
        new_r = [c for c in r if c not in to_remove]
        if new_r:
            # Check feasibility (should be fine since we only removed)
            feasible, _ = check_route_feasibility(new_r, nodes, dist, capacity)
            if feasible:
                new_routes.append(new_r)
            else:
                # Shouldn't happen but handle gracefully
                for c in new_r:
                    to_remove.add(c)
    
    # Reinsert removed customers
    for c in to_remove:
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1
        
        for r_idx in range(len(new_routes)):
            r = new_routes[r_idx]
            r_demand = sum(nodes[cc]['demand'] for cc in r)
            if r_demand + nodes[c]['demand'] > capacity:
                continue
            
            for pos in range(len(r) + 1):
                new_r = r[:pos] + [c] + r[pos:]
                feasible, _ = check_route_feasibility(new_r, nodes, dist, capacity)
                if feasible:
                    cost = route_distance(new_r, dist) - route_distance(r, dist)
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
        
        if best_route_idx >= 0:
            new_routes[best_route_idx].insert(best_pos, c)
        else:
            # Create new route
            new_r = [c]
            feasible, _ = check_route_feasibility(new_r, nodes, dist, capacity)
            if feasible:
                new_routes.append(new_r)
            else:
                # Should not happen for valid instances
                new_routes.append(new_r)
    
    return new_routes


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    time_limit_abs = start_time + time_limit
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    depot = instance['depot']
    customers = instance['customers']
    capacity = instance['vehicle_capacity']
    num_customers = instance['num_customers']
    
    # Build nodes array: index 0 = depot, index 1..n = customers
    nodes = [depot] + customers
    n = len(nodes)
    
    dist = compute_dist_matrix(nodes)
    
    rng = random.Random(42)
    
    # Construction phase: try multiple heuristics
    # 1. Solomon's insertion
    best_routes, best_dist = solomon_insertion(nodes, dist, capacity, time_limit_abs)
    
    if logger and best_routes:
        logger.log(best_dist)
    
    # 2. Nearest neighbor
    if time.time() < time_limit_abs:
        nn_routes = nearest_neighbor_construction(nodes, dist, capacity)
        nn_dist = sum(route_distance(r, dist) for r in nn_routes)
        if nn_dist < best_dist:
            best_routes = nn_routes
            best_dist = nn_dist
            if logger:
                logger.log(best_dist)
    
    # Local search on best initial solution
    if time.time() < time_limit_abs:
        ls_time = min(time_limit_abs, start_time + time_limit * 0.4)
        best_routes = local_search(best_routes, nodes, dist, capacity, ls_time)
        best_routes = [r for r in best_routes if r]
        best_dist = sum(route_distance(r, dist) for r in best_routes)
        if logger:
            logger.log(best_dist)
    
    # ILS: Iterated Local Search
    current_routes = [r[:] for r in best_routes]
    current_dist = best_dist
    
    ils_iterations = 0
    no_improve_count = 0
    
    while time.time() < time_limit_abs - 1.0:  # Leave 1 second buffer
        ils_iterations += 1
        
        # Perturb
        perturbed = perturbation([r[:] for r in current_routes], nodes, dist, capacity, rng)
        
        # Local search on perturbed solution
        remaining_time = time_limit_abs - time.time() - 1.0
        if remaining_time <= 0:
            break
        ls_time_limit = min(time.time() + remaining_time * 0.3, time_limit_abs - 1.0)
        perturbed = local_search(perturbed, nodes, dist, capacity, ls_time_limit)
        perturbed = [r for r in perturbed if r]
        perturbed_dist = sum(route_distance(r, dist) for r in perturbed)
        
        # Acceptance criterion
        if perturbed_dist < current_dist - 1e-10:
            current_routes = perturbed
            current_dist = perturbed_dist
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Accept with some probability for diversification
            if no_improve_count > 5 and rng.random() < 0.1:
                current_routes = perturbed
                current_dist = perturbed_dist
        
        if current_dist < best_dist - 1e-10:
            best_routes = [r[:] for r in current_routes]
            best_dist = current_dist
            if logger:
                logger.log(best_dist)
        
        # If stuck for too long, restart from a different seed
        if no_improve_count > 20:
            # Larger perturbation
            large_perturbed = perturbation(
                [r[:] for r in best_routes], nodes, dist, capacity, rng
            )
            large_perturbed = perturbation(
                large_perturbed, nodes, dist, capacity, rng
            )
            current_routes = large_perturbed
            current_dist = sum(route_distance(r, dist) for r in current_routes)
            no_improve_count = 0
    
    # Verify all customers are served
    served = set()
    for r in best_routes:
        for c in r:
            served.add(c)
    
    all_customers_ids = set(range(1, n))
    assert served == all_customers_ids, f"Not all customers served: missing {all_customers_ids - served}"
    
    # Build output
    total_distance = sum(route_distance(r, dist) for r in best_routes)
    
    output_routes = []
    for r in best_routes:
        customer_ids = [nodes[c]['id'] for c in r]
        path = [nodes[0]['id']] + customer_ids + [nodes[0]['id']]
        output_routes.append({
            'customer_ids': customer_ids,
            'path': path
        })
    
    solution = {
        'objective_value': total_distance,
        'routes': output_routes
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    return solution


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