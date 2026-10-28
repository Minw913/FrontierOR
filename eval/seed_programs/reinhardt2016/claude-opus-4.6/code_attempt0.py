import argparse
import json
import math
import time
import random
import sys
from solution_logger import SolutionLogger


def euclidean_distance(n1, n2):
    return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)


def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    num_edge_sets = instance['num_edge_sets']
    schedule_horizon = instance['schedule_horizon']
    nodes = instance['nodes']
    edge_sets = instance['edge_sets']
    free_edges_list = instance['free_edges']
    dist = instance['distance_matrix']
    
    # Build node lookup
    node_by_id = {}
    for n in nodes:
        node_by_id[n['id']] = n
    
    depot = node_by_id[0]
    customers = [node_by_id[i] for i in range(1, num_customers + 1)]
    
    # Build edge-to-edge_set mapping
    edge_to_set = {}
    for es in edge_sets:
        es_id = es['id']
        for e in es['edges']:
            u, v = e[0], e[1]
            edge_to_set[(min(u,v), max(u,v))] = es_id
    
    free_edges = set()
    for e in free_edges_list:
        u, v = e[0], e[1]
        free_edges.add((min(u,v), max(u,v)))
    
    def get_edge_set_id(u, v):
        key = (min(u,v), max(u,v))
        if key in free_edges:
            return None  # free edge
        return edge_to_set.get(key, None)
    
    # Compute travel time: distance + service_time at origin
    def travel_time(i, j):
        return dist[i][j] + node_by_id[i]['service_time']
    
    def compute_route_cost_and_feasibility(route):
        """Route is a list of customer ids (not including depot)."""
        if not route:
            return 0, True, {}, set()
        
        # Check capacity
        total_demand = sum(node_by_id[c]['demand'] for c in route)
        if total_demand > vehicle_capacity:
            return float('inf'), False, {}, set()
        
        # Compute visit times (greedy earliest)
        visit_times = {}
        visit_times[0] = node_by_id[0]['time_window_start']
        
        current_time = visit_times[0]
        prev = 0
        for c in route:
            arrival = current_time + travel_time(prev, c)
            tw_start = node_by_id[c]['time_window_start']
            tw_end = node_by_id[c]['time_window_end']
            visit_time = max(arrival, tw_start)
            if visit_time > tw_end:
                return float('inf'), False, {}, set()
            visit_times[c] = visit_time
            current_time = visit_time
            prev = c
        
        # Return to depot
        arrival_depot = current_time + travel_time(prev, 0)
        if arrival_depot > node_by_id[0]['time_window_end']:
            return float('inf'), False, {}, set()
        visit_times[0] = max(visit_times.get(0, 0), 0)  # depot visit time at start
        
        # Traversal cost
        traversal_cost = 0
        edge_sets_used = set()
        full_route = [0] + route + [0]
        for i in range(len(full_route) - 1):
            u, v = full_route[i], full_route[i+1]
            traversal_cost += dist[u][v]
            es_id = get_edge_set_id(u, v)
            if es_id is not None:
                edge_sets_used.add(es_id)
        
        return traversal_cost, True, visit_times, edge_sets_used
    
    def compute_solution_cost(routes):
        total_traversal = 0
        all_edge_sets = set()
        for route in routes:
            cost, feasible, _, es_used = compute_route_cost_and_feasibility(route)
            if not feasible:
                return float('inf'), set()
            total_traversal += cost
            all_edge_sets |= es_used
        
        access_cost = sum(next(es['access_cost'] for es in edge_sets if es['id'] == es_id) for es_id in all_edge_sets)
        return total_traversal + access_cost, all_edge_sets
    
    def build_solution(routes):
        total_cost, all_edge_sets = compute_solution_cost(routes)
        if total_cost == float('inf'):
            return None
        
        edge_set_costs = {}
        for es in edge_sets:
            edge_set_costs[es['id']] = es['access_cost']
        
        sol_routes = []
        vehicle_id = 0
        for route in routes:
            if not route:
                continue
            cost, feasible, visit_times, es_used = compute_route_cost_and_feasibility(route)
            if not feasible:
                return None
            
            full_route = [0] + route + [0]
            arcs = [[full_route[i], full_route[i+1]] for i in range(len(full_route)-1)]
            
            # Build visit_times dict with string keys
            vt_dict = {}
            # Depot start time
            current_time = node_by_id[0]['time_window_start']
            vt_dict[str(0)] = current_time
            prev = 0
            for c in route:
                arrival = current_time + travel_time(prev, c)
                visit_time = max(arrival, node_by_id[c]['time_window_start'])
                vt_dict[str(c)] = visit_time
                current_time = visit_time
                prev = c
            # Final depot return
            arrival_depot = current_time + travel_time(prev, 0)
            # The visit time at depot for return - use the start time or the arrival?
            # The problem says visit time must be within time window. Depot TW is [0, schedule_horizon]
            # We already set depot visit time at start
            
            sol_routes.append({
                "vehicle": vehicle_id,
                "sequence": full_route,
                "arcs": arcs,
                "visit_times": vt_dict
            })
            vehicle_id += 1
        
        return {
            "objective_value": total_cost,
            "routes": sol_routes,
            "edge_sets_used": list(all_edge_sets)
        }
    
    # ---- Nearest neighbor heuristic with time windows ----
    def nearest_neighbor_construction():
        unvisited = set(range(1, num_customers + 1))
        routes = []
        
        while unvisited:
            route = []
            current = 0
            current_time = node_by_id[0]['time_window_start']
            current_load = 0
            
            while True:
                best_next = None
                best_cost = float('inf')
                
                for c in unvisited:
                    if current_load + node_by_id[c]['demand'] > vehicle_capacity:
                        continue
                    
                    arrival = current_time + travel_time(current, c)
                    visit_time = max(arrival, node_by_id[c]['time_window_start'])
                    
                    if visit_time > node_by_id[c]['time_window_end']:
                        continue
                    
                    # Check can return to depot
                    return_time = visit_time + travel_time(c, 0)
                    if return_time > node_by_id[0]['time_window_end']:
                        continue
                    
                    cost = dist[current][c]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = c
                
                if best_next is None:
                    break
                
                route.append(best_next)
                unvisited.remove(best_next)
                arrival = current_time + travel_time(current, best_next)
                current_time = max(arrival, node_by_id[best_next]['time_window_start'])
                current_load += node_by_id[best_next]['demand']
                current = best_next
            
            if route:
                routes.append(route)
            elif unvisited:
                # Force add single customer
                c = min(unvisited)
                routes.append([c])
                unvisited.remove(c)
        
        return routes
    
    # ---- Time window aware insertion heuristic ----
    def insertion_heuristic():
        # Sort customers by time window start
        sorted_customers = sorted(range(1, num_customers + 1), key=lambda c: node_by_id[c]['time_window_start'])
        
        routes = []
        unvisited = list(sorted_customers)
        
        while unvisited:
            # Start a new route with the earliest unvisited customer
            seed = unvisited[0]
            route = [seed]
            unvisited.remove(seed)
            
            improved = True
            while improved and unvisited:
                improved = False
                best_insertion = None
                best_cost_increase = float('inf')
                best_customer = None
                
                for c in unvisited:
                    if sum(node_by_id[r]['demand'] for r in route) + node_by_id[c]['demand'] > vehicle_capacity:
                        continue
                    
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        cost, feasible, _, _ = compute_route_cost_and_feasibility(new_route)
                        if feasible:
                            old_cost, _, _, _ = compute_route_cost_and_feasibility(route)
                            increase = cost - old_cost
                            if increase < best_cost_increase:
                                best_cost_increase = increase
                                best_insertion = pos
                                best_customer = c
                                improved = True
                
                if best_customer is not None:
                    route = route[:best_insertion] + [best_customer] + route[best_insertion:]
                    unvisited.remove(best_customer)
            
            routes.append(route)
        
        return routes
    
    # ---- Savings-based construction heuristic (Clarke-Wright) ----
    def savings_construction():
        # Start with each customer in its own route
        routes = {c: [c] for c in range(1, num_customers + 1)}
        customer_route = {c: c for c in range(1, num_customers + 1)}
        
        # Compute savings
        savings = []
        for i in range(1, num_customers + 1):
            for j in range(i + 1, num_customers + 1):
                s = dist[0][i] + dist[0][j] - dist[i][j]
                savings.append((s, i, j))
        
        savings.sort(reverse=True)
        
        for s, i, j in savings:
            if time.time() - start_time > time_limit * 0.3:
                break
            
            ri = customer_route[i]
            rj = customer_route[j]
            
            if ri == rj:
                continue
            
            route_i = routes.get(ri)
            route_j = routes.get(rj)
            
            if route_i is None or route_j is None:
                continue
            
            # Try merging: i must be at end of route_i, j must be at start of route_j (or vice versa)
            merged = None
            
            if route_i[-1] == i and route_j[0] == j:
                merged = route_i + route_j
            elif route_i[-1] == i and route_j[-1] == j:
                merged = route_i + route_j[::-1]
            elif route_i[0] == i and route_j[0] == j:
                merged = route_i[::-1] + route_j
            elif route_i[0] == i and route_j[-1] == j:
                merged = route_j + route_i
            
            if merged is None:
                continue
            
            # Check feasibility
            total_demand = sum(node_by_id[c]['demand'] for c in merged)
            if total_demand > vehicle_capacity:
                continue
            
            cost, feasible, _, _ = compute_route_cost_and_feasibility(merged)
            if not feasible:
                continue
            
            # Merge
            new_key = ri
            routes[new_key] = merged
            if rj in routes:
                del routes[rj]
            for c in merged:
                customer_route[c] = new_key
        
        return list(routes.values())
    
    # ---- Local search operators ----
    def route_is_feasible(route):
        if not route:
            return True
        cost, feasible, _, _ = compute_route_cost_and_feasibility(route)
        return feasible
    
    def or_opt(routes, max_time):
        """Or-opt: move a segment of 1, 2, or 3 consecutive customers from one route to another."""
        improved = True
        while improved:
            if time.time() - start_time > max_time:
                break
            improved = False
            for ri in range(len(routes)):
                if time.time() - start_time > max_time:
                    break
                for seg_len in [1, 2, 3]:
                    if time.time() - start_time > max_time:
                        break
                    for pos in range(len(routes[ri]) - seg_len + 1):
                        if time.time() - start_time > max_time:
                            break
                        segment = routes[ri][pos:pos+seg_len]
                        remaining = routes[ri][:pos] + routes[ri][pos+seg_len:]
                        
                        old_cost_ri, _, _, _ = compute_route_cost_and_feasibility(routes[ri])
                        new_cost_remaining, feas_remaining, _, _ = compute_route_cost_and_feasibility(remaining)
                        if not feas_remaining and remaining:
                            continue
                        
                        for rj in range(len(routes)):
                            if ri == rj:
                                continue
                            
                            seg_demand = sum(node_by_id[c]['demand'] for c in segment)
                            route_j_demand = sum(node_by_id[c]['demand'] for c in routes[rj])
                            if seg_demand + route_j_demand > vehicle_capacity:
                                continue
                            
                            old_cost_rj, _, _, _ = compute_route_cost_and_feasibility(routes[rj])
                            
                            best_insert = None
                            best_insert_cost = float('inf')
                            
                            for insert_pos in range(len(routes[rj]) + 1):
                                new_route_j = routes[rj][:insert_pos] + segment + routes[rj][insert_pos:]
                                cost_j, feas_j, _, _ = compute_route_cost_and_feasibility(new_route_j)
                                if feas_j and cost_j < best_insert_cost:
                                    best_insert_cost = cost_j
                                    best_insert = insert_pos
                            
                            if best_insert is not None:
                                improvement = (old_cost_ri + old_cost_rj) - (new_cost_remaining + best_insert_cost)
                                if improvement > 1e-6:
                                    new_route_j = routes[rj][:best_insert] + segment + routes[rj][best_insert:]
                                    routes[ri] = remaining
                                    routes[rj] = new_route_j
                                    improved = True
                                    break
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Remove empty routes
            routes = [r for r in routes if r]
        
        return routes
    
    def two_opt_within(route):
        """2-opt within a single route."""
        if len(route) < 2:
            return route
        
        best_route = route[:]
        best_cost, _, _, _ = compute_route_cost_and_feasibility(best_route)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    cost, feasible, _, _ = compute_route_cost_and_feasibility(new_route)
                    if feasible and cost < best_cost - 1e-6:
                        best_route = new_route
                        best_cost = cost
                        improved = True
                        break
                if improved:
                    break
        
        return best_route
    
    def relocate(routes, max_time):
        """Try to move single customers between routes or within a route."""
        improved = True
        while improved:
            if time.time() - start_time > max_time:
                break
            improved = False
            for ri in range(len(routes)):
                if time.time() - start_time > max_time:
                    break
                for ci in range(len(routes[ri])):
                    if time.time() - start_time > max_time:
                        break
                    customer = routes[ri][ci]
                    remaining = routes[ri][:ci] + routes[ri][ci+1:]
                    
                    old_cost_ri, _, _, _ = compute_route_cost_and_feasibility(routes[ri])
                    new_cost_remaining, feas_remaining, _, _ = compute_route_cost_and_feasibility(remaining)
                    if not feas_remaining and remaining:
                        continue
                    if not remaining:
                        new_cost_remaining = 0
                    
                    best_improvement = 0
                    best_rj = None
                    best_pos = None
                    
                    for rj in range(len(routes)):
                        if ri == rj:
                            # Try within same route
                            for pos in range(len(remaining) + 1):
                                new_route = remaining[:pos] + [customer] + remaining[pos:]
                                if new_route == routes[ri]:
                                    continue
                                cost, feasible, _, _ = compute_route_cost_and_feasibility(new_route)
                                if feasible:
                                    improvement = old_cost_ri - cost
                                    if improvement > best_improvement + 1e-6:
                                        best_improvement = improvement
                                        best_rj = rj
                                        best_pos = pos
                        else:
                            demand_check = sum(node_by_id[c]['demand'] for c in routes[rj]) + node_by_id[customer]['demand']
                            if demand_check > vehicle_capacity:
                                continue
                            
                            old_cost_rj, _, _, _ = compute_route_cost_and_feasibility(routes[rj])
                            
                            for pos in range(len(routes[rj]) + 1):
                                new_route_j = routes[rj][:pos] + [customer] + routes[rj][pos:]
                                cost_j, feas_j, _, _ = compute_route_cost_and_feasibility(new_route_j)
                                if feas_j:
                                    improvement = (old_cost_ri + old_cost_rj) - (new_cost_remaining + cost_j)
                                    if improvement > best_improvement + 1e-6:
                                        best_improvement = improvement
                                        best_rj = rj
                                        best_pos = pos
                    
                    if best_rj is not None and best_improvement > 1e-6:
                        if best_rj == ri:
                            routes[ri] = remaining[:best_pos] + [customer] + remaining[best_pos:]
                        else:
                            routes[ri] = remaining
                            routes[best_rj] = routes[best_rj][:best_pos] + [customer] + routes[best_rj][best_pos:]
                        improved = True
                        break
                if improved:
                    break
            
            routes = [r for r in routes if r]
        
        return routes
    
    def exchange(routes, max_time):
        """Swap single customers between two routes."""
        improved = True
        while improved:
            if time.time() - start_time > max_time:
                break
            improved = False
            for ri in range(len(routes)):
                if time.time() - start_time > max_time:
                    break
                for rj in range(ri + 1, len(routes)):
                    if time.time() - start_time > max_time:
                        break
                    for ci in range(len(routes[ri])):
                        if time.time() - start_time > max_time:
                            break
                        for cj in range(len(routes[rj])):
                            # Swap customer ci from ri with cj from rj
                            cust_i = routes[ri][ci]
                            cust_j = routes[rj][cj]
                            
                            new_ri = routes[ri][:ci] + [cust_j] + routes[ri][ci+1:]
                            new_rj = routes[rj][:cj] + [cust_i] + routes[rj][cj+1:]
                            
                            # Demand check
                            demand_ri = sum(node_by_id[c]['demand'] for c in new_ri)
                            demand_rj = sum(node_by_id[c]['demand'] for c in new_rj)
                            if demand_ri > vehicle_capacity or demand_rj > vehicle_capacity:
                                continue
                            
                            old_cost_ri, _, _, _ = compute_route_cost_and_feasibility(routes[ri])
                            old_cost_rj, _, _, _ = compute_route_cost_and_feasibility(routes[rj])
                            
                            cost_ri, feas_ri, _, _ = compute_route_cost_and_feasibility(new_ri)
                            cost_rj, feas_rj, _, _ = compute_route_cost_and_feasibility(new_rj)
                            
                            if feas_ri and feas_rj:
                                improvement = (old_cost_ri + old_cost_rj) - (cost_ri + cost_rj)
                                if improvement > 1e-6:
                                    routes[ri] = new_ri
                                    routes[rj] = new_rj
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return routes
    
    def local_search(routes, max_time):
        """Apply multiple local search operators."""
        # 2-opt within each route
        for i in range(len(routes)):
            if time.time() - start_time > max_time:
                break
            routes[i] = two_opt_within(routes[i])
        
        # Relocate
        if time.time() - start_time < max_time:
            routes = relocate(routes, max_time)
        
        # Exchange
        if time.time() - start_time < max_time:
            routes = exchange(routes, max_time)
        
        # Or-opt
        if time.time() - start_time < max_time:
            routes = or_opt(routes, max_time)
        
        # 2-opt again
        for i in range(len(routes)):
            if time.time() - start_time > max_time:
                break
            routes[i] = two_opt_within(routes[i])
        
        routes = [r for r in routes if r]
        return routes
    
    # ---- Main solution process ----
    best_solution = None
    best_cost = float('inf')
    
    # Try multiple construction heuristics
    construction_methods = [nearest_neighbor_construction, savings_construction]
    
    all_routes = []
    for method in construction_methods:
        if time.time() - start_time > time_limit * 0.1:
            break
        try:
            routes = method()
            all_routes.append(routes)
        except:
            pass
    
    # Also try randomized nearest neighbor
    def randomized_nearest_neighbor():
        unvisited = set(range(1, num_customers + 1))
        routes = []
        
        while unvisited:
            route = []
            current = 0
            current_time = node_by_id[0]['time_window_start']
            current_load = 0
            
            while True:
                candidates = []
                
                for c in unvisited:
                    if current_load + node_by_id[c]['demand'] > vehicle_capacity:
                        continue
                    
                    arrival = current_time + travel_time(current, c)
                    visit_time = max(arrival, node_by_id[c]['time_window_start'])
                    
                    if visit_time > node_by_id[c]['time_window_end']:
                        continue
                    
                    return_time = visit_time + travel_time(c, 0)
                    if return_time > node_by_id[0]['time_window_end']:
                        continue
                    
                    cost = dist[current][c]
                    candidates.append((cost, c))
                
                if not candidates:
                    break
                
                candidates.sort()
                # Pick from top-k with randomization
                k = min(3, len(candidates))
                chosen = candidates[random.randint(0, k-1)]
                c = chosen[1]
                
                route.append(c)
                unvisited.remove(c)
                arrival = current_time + travel_time(current, c)
                current_time = max(arrival, node_by_id[c]['time_window_start'])
                current_load += node_by_id[c]['demand']
                current = c
            
            if route:
                routes.append(route)
            elif unvisited:
                c = min(unvisited)
                routes.append([c])
                unvisited.remove(c)
        
        return routes
    
    for _ in range(5):
        if time.time() - start_time > time_limit * 0.15:
            break
        try:
            routes = randomized_nearest_neighbor()
            all_routes.append(routes)
        except:
            pass
    
    # Evaluate and improve each set of routes
    for routes in all_routes:
        if time.time() - start_time > time_limit * 0.8:
            break
        
        remaining_time = start_time + time_limit * 0.85
        routes = local_search(routes, remaining_time)
        
        cost, es_used = compute_solution_cost(routes)
        if cost < best_cost:
            best_cost = cost
            best_solution = [r[:] for r in routes]
            if logger:
                logger.log(best_cost)
    
    # Perturbation-based improvement (simplified LNS)
    if best_solution is not None:
        current_routes = [r[:] for r in best_solution]
        current_cost = best_cost
        
        iterations = 0
        while time.time() - start_time < time_limit * 0.92:
            iterations += 1
            
            # Perturb: randomly remove some customers and reinsert
            routes_copy = [r[:] for r in current_routes]
            
            # Determine how many to remove
            all_customers_in_routes = []
            for r in routes_copy:
                all_customers_in_routes.extend(r)
            
            n_remove = max(1, min(len(all_customers_in_routes) // 4, 8))
            removed = random.sample(all_customers_in_routes, min(n_remove, len(all_customers_in_routes)))
            
            for c in removed:
                for r in routes_copy:
                    if c in r:
                        r.remove(c)
                        break
            
            routes_copy = [r for r in routes_copy if r]
            
            # Reinsert removed customers
            random.shuffle(removed)
            for c in removed:
                best_insert_cost = float('inf')
                best_route_idx = None
                best_insert_pos = None
                
                for ri in range(len(routes_copy)):
                    demand_check = sum(node_by_id[x]['demand'] for x in routes_copy[ri]) + node_by_id[c]['demand']
                    if demand_check > vehicle_capacity:
                        continue
                    
                    for pos in range(len(routes_copy[ri]) + 1):
                        new_route = routes_copy[ri][:pos] + [c] + routes_copy[ri][pos:]
                        cost, feasible, _, _ = compute_route_cost_and_feasibility(new_route)
                        if feasible and cost < best_insert_cost:
                            best_insert_cost = cost
                            best_route_idx = ri
                            best_insert_pos = pos
                
                # Also try new route
                new_route = [c]
                cost, feasible, _, _ = compute_route_cost_and_feasibility(new_route)
                if feasible and cost < best_insert_cost:
                    best_insert_cost = cost
                    best_route_idx = -1
                    best_insert_pos = 0
                
                if best_route_idx == -1:
                    routes_copy.append([c])
                elif best_route_idx is not None:
                    routes_copy[best_route_idx] = routes_copy[best_route_idx][:best_insert_pos] + [c] + routes_copy[best_route_idx][best_insert_pos:]
                else:
                    # Fallback: just add as new route
                    routes_copy.append([c])
            
            # Local search on perturbed solution
            remaining = min(start_time + time_limit * 0.92, time.time() + 2.0)
            routes_copy = local_search(routes_copy, remaining)
            
            cost, es_used = compute_solution_cost(routes_copy)
            
            if cost < best_cost:
                best_cost = cost
                best_solution = [r[:] for r in routes_copy]
                current_routes = [r[:] for r in routes_copy]
                current_cost = cost
                if logger:
                    logger.log(best_cost)
            elif cost < current_cost * 1.02:  # Accept slightly worse for diversification
                current_routes = [r[:] for r in routes_copy]
                current_cost = cost
    
    # Build final solution
    if best_solution is None:
        # Emergency: create individual routes for each customer
        best_solution = [[c] for c in range(1, num_customers + 1)]
    
    solution = build_solution(best_solution)
    return solution


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
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()