import argparse
import json
import math
import time
import random
from solution_logger import SolutionLogger


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def solve_vrptw(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    params = instance["problem_parameters"]
    n_nodes = params["n_nodes"]
    n_customers = params["n_customers"]
    vehicle_capacity = params["vehicle_capacity"]
    planning_horizon = params["planning_horizon"]
    
    nodes = instance["nodes"]
    
    # Identify depot source, depot sink, and customers
    depot_source = None
    depot_sink = None
    customers = []
    
    for node in nodes:
        if node["type"] == "depot_source":
            depot_source = node
        elif node["type"] == "depot_sink":
            depot_sink = node
        elif node["type"] == "customer":
            customers.append(node)
    
    depot_source_id = depot_source["id"]
    depot_sink_id = depot_sink["id"]
    customer_ids = [c["id"] for c in customers]
    
    # Precompute distance matrix
    n = len(nodes)
    node_by_id = {node["id"]: node for node in nodes}
    
    all_ids = [depot_source_id] + customer_ids + [depot_sink_id]
    id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
    
    dist = {}
    for i in all_ids:
        for j in all_ids:
            ni = node_by_id[i]
            nj = node_by_id[j]
            dist[(i, j)] = euclidean_distance(ni["x"], ni["y"], nj["x"], nj["y"])
    
    # Helper functions
    def route_cost(route):
        """Calculate total travel cost for a route (list of customer ids, without depot)."""
        if not route:
            return 0
        cost = dist[(depot_source_id, route[0])]
        for i in range(len(route) - 1):
            cost += dist[(route[i], route[i + 1])]
        cost += dist[(route[-1], depot_sink_id)]
        return cost
    
    def route_feasible(route):
        """Check if a route is feasible (capacity, time windows)."""
        if not route:
            return True
        
        # Check capacity
        total_demand = sum(node_by_id[c]["demand"] for c in route)
        if total_demand > vehicle_capacity:
            return False
        
        # Check time windows
        current_time = 0.0
        prev = depot_source_id
        for c in route:
            cn = node_by_id[c]
            travel = dist[(prev, c)]
            arrival = current_time + travel
            service_start = max(arrival, cn["time_window_open"])
            if service_start > cn["time_window_close"]:
                return False
            current_time = service_start + cn["service_time"]
            prev = c
        
        # Check return to depot
        travel_back = dist[(prev, depot_sink_id)]
        if current_time + travel_back > planning_horizon:
            return False
        
        return True
    
    def compute_route_times(route):
        """Compute service start times for each customer in route."""
        times = []
        current_time = 0.0
        prev = depot_source_id
        for c in route:
            cn = node_by_id[c]
            travel = dist[(prev, c)]
            arrival = current_time + travel
            service_start = max(arrival, cn["time_window_open"])
            times.append(service_start)
            current_time = service_start + cn["service_time"]
            prev = c
        return times
    
    def route_demand(route):
        return sum(node_by_id[c]["demand"] for c in route)
    
    # Build initial solution using Solomon's I1 insertion heuristic
    def build_initial_solution():
        unrouted = set(customer_ids)
        routes = []
        
        while unrouted:
            # Start a new route with the unrouted customer having earliest deadline
            # (or furthest from depot as seed)
            seed = None
            best_val = -1
            for c in unrouted:
                cn = node_by_id[c]
                d = dist[(depot_source_id, c)]
                if d > best_val:
                    best_val = d
                    seed = c
            
            route = [seed]
            unrouted.remove(seed)
            
            # Iteratively insert best customer
            improved = True
            while improved:
                improved = False
                best_customer = None
                best_position = None
                best_cost_increase = float('inf')
                
                for c in unrouted:
                    cn = node_by_id[c]
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        
                        # Quick capacity check
                        if route_demand(new_route) > vehicle_capacity:
                            continue
                        
                        if route_feasible(new_route):
                            old_cost = route_cost(route)
                            new_cost = route_cost(new_route)
                            cost_increase = new_cost - old_cost
                            
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_customer = c
                                best_position = pos
                
                if best_customer is not None:
                    route = route[:best_position] + [best_customer] + route[best_position:]
                    unrouted.remove(best_customer)
                    improved = True
            
            routes.append(route)
        
        return routes
    
    # Also try nearest neighbor heuristic
    def build_nearest_neighbor_solution():
        unrouted = set(customer_ids)
        routes = []
        
        while unrouted:
            route = []
            current_time = 0.0
            current_loc = depot_source_id
            current_demand = 0
            
            while True:
                best_c = None
                best_dist = float('inf')
                
                for c in unrouted:
                    cn = node_by_id[c]
                    if current_demand + cn["demand"] > vehicle_capacity:
                        continue
                    
                    travel = dist[(current_loc, c)]
                    arrival = current_time + travel
                    service_start = max(arrival, cn["time_window_open"])
                    
                    if service_start > cn["time_window_close"]:
                        continue
                    
                    departure = service_start + cn["service_time"]
                    travel_back = dist[(c, depot_sink_id)]
                    if departure + travel_back > planning_horizon:
                        continue
                    
                    # Check we can still return with remaining route
                    if travel < best_dist:
                        best_dist = travel
                        best_c = c
                
                if best_c is None:
                    break
                
                route.append(best_c)
                cn = node_by_id[best_c]
                travel = dist[(current_loc, best_c)]
                arrival = current_time + travel
                service_start = max(arrival, cn["time_window_open"])
                current_time = service_start + cn["service_time"]
                current_loc = best_c
                current_demand += cn["demand"]
                unrouted.remove(best_c)
            
            if route:
                routes.append(route)
            else:
                # If we can't route any customer, force single-customer routes
                c = next(iter(unrouted))
                routes.append([c])
                unrouted.remove(c)
        
        return routes
    
    def total_cost(routes):
        return sum(route_cost(r) for r in routes)
    
    # Build initial solutions and pick best
    sol1 = build_initial_solution()
    sol2 = build_nearest_neighbor_solution()
    
    cost1 = total_cost(sol1)
    cost2 = total_cost(sol2)
    
    if cost1 <= cost2:
        best_routes = [r[:] for r in sol1]
        best_cost = cost1
    else:
        best_routes = [r[:] for r in sol2]
        best_cost = cost2
    
    current_routes = [r[:] for r in best_routes]
    current_cost = best_cost
    
    if logger:
        logger.log(best_cost)
    
    # Local search improvements
    def try_relocate(routes):
        """Try to relocate a customer from one route to another."""
        best_improvement = 0
        best_move = None
        
        for r1_idx in range(len(routes)):
            if not routes[r1_idx]:
                continue
            for pos1 in range(len(routes[r1_idx])):
                c = routes[r1_idx][pos1]
                cn = node_by_id[c]
                
                # Cost of removing c from r1
                old_r1 = routes[r1_idx]
                new_r1 = old_r1[:pos1] + old_r1[pos1 + 1:]
                
                if not new_r1 and len(routes) <= 1:
                    continue
                
                cost_r1_old = route_cost(old_r1)
                cost_r1_new = route_cost(new_r1) if new_r1 else 0
                saving_r1 = cost_r1_old - cost_r1_new
                
                for r2_idx in range(len(routes)):
                    if r1_idx == r2_idx:
                        continue
                    
                    old_r2 = routes[r2_idx]
                    
                    # Check capacity
                    if route_demand(old_r2) + cn["demand"] > vehicle_capacity:
                        continue
                    
                    cost_r2_old = route_cost(old_r2)
                    
                    for pos2 in range(len(old_r2) + 1):
                        new_r2 = old_r2[:pos2] + [c] + old_r2[pos2:]
                        
                        if not route_feasible(new_r2):
                            continue
                        
                        cost_r2_new = route_cost(new_r2)
                        extra_r2 = cost_r2_new - cost_r2_old
                        
                        improvement = saving_r1 - extra_r2
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_move = (r1_idx, pos1, r2_idx, pos2)
                
                # Also try relocate within same route
                for pos2 in range(len(old_r1)):
                    if pos2 == pos1 or pos2 == pos1:
                        continue
                    new_r = old_r1[:pos1] + old_r1[pos1 + 1:]
                    actual_pos2 = pos2 if pos2 < pos1 else pos2 - 1
                    if actual_pos2 < 0:
                        actual_pos2 = 0
                    new_r = new_r[:actual_pos2] + [c] + new_r[actual_pos2:]
                    
                    if not route_feasible(new_r):
                        continue
                    
                    improvement = cost_r1_old - route_cost(new_r)
                    if improvement > best_improvement + 1e-10:
                        best_improvement = improvement
                        best_move = (r1_idx, pos1, r1_idx, actual_pos2, True)
        
        return best_improvement, best_move
    
    def apply_relocate(routes, move):
        if len(move) == 5:  # intra-route
            r_idx, pos1, _, pos2, _ = move
            c = routes[r_idx][pos1]
            new_r = routes[r_idx][:pos1] + routes[r_idx][pos1 + 1:]
            new_r = new_r[:pos2] + [c] + new_r[pos2:]
            routes[r_idx] = new_r
        else:
            r1_idx, pos1, r2_idx, pos2 = move
            c = routes[r1_idx][pos1]
            routes[r1_idx] = routes[r1_idx][:pos1] + routes[r1_idx][pos1 + 1:]
            routes[r2_idx] = routes[r2_idx][:pos2] + [c] + routes[r2_idx][pos2:]
            # Remove empty routes
            routes[:] = [r for r in routes if r]
    
    def try_swap(routes):
        """Try swapping two customers between routes."""
        best_improvement = 0
        best_move = None
        
        for r1_idx in range(len(routes)):
            for pos1 in range(len(routes[r1_idx])):
                c1 = routes[r1_idx][pos1]
                cn1 = node_by_id[c1]
                
                for r2_idx in range(r1_idx + 1, len(routes)):
                    for pos2 in range(len(routes[r2_idx])):
                        c2 = routes[r2_idx][pos2]
                        cn2 = node_by_id[c2]
                        
                        # Check capacity
                        r1_demand = route_demand(routes[r1_idx])
                        r2_demand = route_demand(routes[r2_idx])
                        
                        new_r1_demand = r1_demand - cn1["demand"] + cn2["demand"]
                        new_r2_demand = r2_demand - cn2["demand"] + cn1["demand"]
                        
                        if new_r1_demand > vehicle_capacity or new_r2_demand > vehicle_capacity:
                            continue
                        
                        new_r1 = routes[r1_idx][:]
                        new_r1[pos1] = c2
                        new_r2 = routes[r2_idx][:]
                        new_r2[pos2] = c1
                        
                        if not route_feasible(new_r1) or not route_feasible(new_r2):
                            continue
                        
                        old_cost = route_cost(routes[r1_idx]) + route_cost(routes[r2_idx])
                        new_cost = route_cost(new_r1) + route_cost(new_r2)
                        improvement = old_cost - new_cost
                        
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_move = (r1_idx, pos1, r2_idx, pos2)
        
        return best_improvement, best_move
    
    def apply_swap(routes, move):
        r1_idx, pos1, r2_idx, pos2 = move
        routes[r1_idx][pos1], routes[r2_idx][pos2] = routes[r2_idx][pos2], routes[r1_idx][pos1]
    
    def try_2opt_intra(routes):
        """Try 2-opt within each route."""
        best_improvement = 0
        best_move = None
        
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 2:
                continue
            
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    
                    if not route_feasible(new_route):
                        continue
                    
                    improvement = route_cost(route) - route_cost(new_route)
                    if improvement > best_improvement + 1e-10:
                        best_improvement = improvement
                        best_move = (r_idx, i, j)
        
        return best_improvement, best_move
    
    def apply_2opt_intra(routes, move):
        r_idx, i, j = move
        routes[r_idx] = routes[r_idx][:i] + routes[r_idx][i:j + 1][::-1] + routes[r_idx][j + 1:]
    
    def try_or_opt(routes):
        """Try moving a segment of 1, 2, or 3 consecutive customers."""
        best_improvement = 0
        best_move = None
        
        for r1_idx in range(len(routes)):
            route1 = routes[r1_idx]
            for seg_len in [1, 2, 3]:
                for start in range(len(route1) - seg_len + 1):
                    segment = route1[start:start + seg_len]
                    remaining = route1[:start] + route1[start + seg_len:]
                    
                    seg_demand = sum(node_by_id[c]["demand"] for c in segment)
                    
                    old_r1_cost = route_cost(route1)
                    remaining_cost = route_cost(remaining) if remaining else 0
                    saving = old_r1_cost - remaining_cost
                    
                    # Try inserting segment into other routes
                    for r2_idx in range(len(routes)):
                        if r1_idx == r2_idx:
                            route2 = remaining
                        else:
                            route2 = routes[r2_idx]
                        
                        if r1_idx != r2_idx:
                            if route_demand(route2) + seg_demand > vehicle_capacity:
                                continue
                        
                        old_r2_cost = route_cost(route2) if r1_idx != r2_idx else remaining_cost
                        
                        for pos in range(len(route2) + 1):
                            new_r2 = route2[:pos] + segment + route2[pos:]
                            
                            if not route_feasible(new_r2):
                                continue
                            
                            new_r2_cost = route_cost(new_r2)
                            
                            if r1_idx == r2_idx:
                                improvement = old_r1_cost - new_r2_cost
                            else:
                                improvement = saving - (new_r2_cost - old_r2_cost)
                            
                            if improvement > best_improvement + 1e-10:
                                best_improvement = improvement
                                best_move = (r1_idx, start, seg_len, r2_idx, pos)
        
        return best_improvement, best_move
    
    def apply_or_opt(routes, move):
        r1_idx, start, seg_len, r2_idx, pos = move
        segment = routes[r1_idx][start:start + seg_len]
        
        if r1_idx == r2_idx:
            remaining = routes[r1_idx][:start] + routes[r1_idx][start + seg_len:]
            new_route = remaining[:pos] + segment + remaining[pos:]
            routes[r1_idx] = new_route
        else:
            routes[r1_idx] = routes[r1_idx][:start] + routes[r1_idx][start + seg_len:]
            routes[r2_idx] = routes[r2_idx][:pos] + segment + routes[r2_idx][pos:]
            routes[:] = [r for r in routes if r]
    
    def try_cross_exchange(routes):
        """Try exchanging tails of two routes."""
        best_improvement = 0
        best_move = None
        
        for r1_idx in range(len(routes)):
            for r2_idx in range(r1_idx + 1, len(routes)):
                route1 = routes[r1_idx]
                route2 = routes[r2_idx]
                
                for i in range(len(route1) + 1):
                    for j in range(len(route2) + 1):
                        new_r1 = route1[:i] + route2[j:]
                        new_r2 = route2[:j] + route1[i:]
                        
                        if not new_r1 and not new_r2:
                            continue
                        
                        if route_demand(new_r1) > vehicle_capacity or route_demand(new_r2) > vehicle_capacity:
                            continue
                        
                        if (new_r1 and not route_feasible(new_r1)) or (new_r2 and not route_feasible(new_r2)):
                            continue
                        
                        old_cost = route_cost(route1) + route_cost(route2)
                        new_cost = (route_cost(new_r1) if new_r1 else 0) + (route_cost(new_r2) if new_r2 else 0)
                        improvement = old_cost - new_cost
                        
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_move = (r1_idx, i, r2_idx, j)
        
        return best_improvement, best_move
    
    def apply_cross_exchange(routes, move):
        r1_idx, i, r2_idx, j = move
        route1 = routes[r1_idx]
        route2 = routes[r2_idx]
        new_r1 = route1[:i] + route2[j:]
        new_r2 = route2[:j] + route1[i:]
        routes[r1_idx] = new_r1
        routes[r2_idx] = new_r2
        routes[:] = [r for r in routes if r]
    
    # Iterative local search
    def local_search(routes):
        nonlocal best_routes, best_cost
        current_cost = total_cost(routes)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit - 1:
            improved = False
            
            # Try relocate
            imp, move = try_relocate(routes)
            if imp > 1e-10:
                if move and len(move) == 5:
                    apply_relocate(routes, move)
                elif move:
                    apply_relocate(routes, move)
                current_cost -= imp
                improved = True
                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_routes = [r[:] for r in routes]
                    if logger:
                        logger.log(best_cost)
                continue
            
            if (time.time() - start_time) >= time_limit - 1:
                break
            
            # Try swap
            imp, move = try_swap(routes)
            if imp > 1e-10:
                apply_swap(routes, move)
                current_cost -= imp
                improved = True
                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_routes = [r[:] for r in routes]
                    if logger:
                        logger.log(best_cost)
                continue
            
            if (time.time() - start_time) >= time_limit - 1:
                break
            
            # Try 2-opt
            imp, move = try_2opt_intra(routes)
            if imp > 1e-10:
                apply_2opt_intra(routes, move)
                current_cost -= imp
                improved = True
                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_routes = [r[:] for r in routes]
                    if logger:
                        logger.log(best_cost)
                continue
            
            if (time.time() - start_time) >= time_limit - 1:
                break
            
            # Try or-opt
            imp, move = try_or_opt(routes)
            if imp > 1e-10:
                apply_or_opt(routes, move)
                current_cost -= imp
                improved = True
                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_routes = [r[:] for r in routes]
                    if logger:
                        logger.log(best_cost)
                continue
            
            if (time.time() - start_time) >= time_limit - 1:
                break
            
            # Try cross exchange
            imp, move = try_cross_exchange(routes)
            if imp > 1e-10:
                apply_cross_exchange(routes, move)
                current_cost -= imp
                improved = True
                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_routes = [r[:] for r in routes]
                    if logger:
                        logger.log(best_cost)
                continue
        
        return routes, current_cost
    
    # Run local search on initial solution
    current_routes, current_cost = local_search(current_routes)
    
    if current_cost < best_cost - 1e-10:
        best_cost = current_cost
        best_routes = [r[:] for r in current_routes]
        if logger:
            logger.log(best_cost)
    
    # Perturbation + local search (iterated local search)
    def perturb(routes):
        """Random perturbation: remove random customers and reinsert."""
        routes = [r[:] for r in routes]
        all_customers = []
        for r in routes:
            all_customers.extend(r)
        
        if len(all_customers) < 2:
            return routes
        
        # Remove a few random customers
        n_remove = max(1, min(len(all_customers) // 5, 8))
        to_remove = set(random.sample(all_customers, n_remove))
        
        for i in range(len(routes)):
            routes[i] = [c for c in routes[i] if c not in to_remove]
        routes = [r for r in routes if r]
        
        # Reinsert customers greedily
        for c in to_remove:
            cn = node_by_id[c]
            best_pos = None
            best_r_idx = None
            best_inc = float('inf')
            
            for r_idx in range(len(routes)):
                if route_demand(routes[r_idx]) + cn["demand"] > vehicle_capacity:
                    continue
                for pos in range(len(routes[r_idx]) + 1):
                    new_r = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
                    if route_feasible(new_r):
                        inc = route_cost(new_r) - route_cost(routes[r_idx])
                        if inc < best_inc:
                            best_inc = inc
                            best_pos = pos
                            best_r_idx = r_idx
            
            if best_r_idx is not None:
                routes[best_r_idx] = routes[best_r_idx][:best_pos] + [c] + routes[best_r_idx][best_pos:]
            else:
                # Create new route
                new_route = [c]
                if route_feasible(new_route):
                    routes.append(new_route)
                else:
                    # Force it (shouldn't happen for valid instances)
                    routes.append(new_route)
        
        return routes
    
    # ILS loop
    iteration = 0
    while (time.time() - start_time) < time_limit - 2:
        iteration += 1
        
        # Perturb from best solution
        perturbed = perturb(best_routes)
        perturbed, perturbed_cost = local_search(perturbed)
        
        if perturbed_cost < best_cost - 1e-10:
            best_cost = perturbed_cost
            best_routes = [r[:] for r in perturbed]
            if logger:
                logger.log(best_cost)
    
    # Format output
    output_routes = []
    for route in best_routes:
        if route:
            full_route = [depot_source_id] + route + [depot_sink_id]
            output_routes.append(full_route)
    
    # Recalculate exact cost
    final_cost = 0
    for route in best_routes:
        final_cost += route_cost(route)
    
    solution = {
        "objective_value": final_cost,
        "routes": output_routes
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve_vrptw(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()