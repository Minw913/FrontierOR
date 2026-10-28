import argparse
import json
import math
import time
import random
import copy
from solution_logger import SolutionLogger


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_distance_matrix(depot, customers):
    n = len(customers) + 1  # 0 is depot
    dist = [[0.0] * n for _ in range(n)]
    locations = [(depot['x'], depot['y'])] + [(c['x'], c['y']) for c in customers]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = euclidean_distance(locations[i][0], locations[i][1],
                                                  locations[j][0], locations[j][1])
    return dist


def solve(instance, time_limit, logger):
    start_time = time.time()
    
    depot = instance['depot']
    customers = instance['customers']
    num_customers = instance['num_customers']
    num_days = instance['num_days']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    max_route_duration = instance['max_route_duration']
    
    # Map customer id to index (0-based)
    cust_id_to_idx = {}
    for i, c in enumerate(customers):
        cust_id_to_idx[c['id']] = i
    
    dist = compute_distance_matrix(depot, customers)
    
    # Service times: index 0 is depot
    service_times = [depot['service_time']] + [c['service_time'] for c in customers]
    
    # Time windows: index 0 is depot
    time_windows = [depot['time_window']] + [c['time_window'] for c in customers]
    
    # Travel time between i and j = dist[i][j] + service_time[i]
    def travel_time(i, j):
        return dist[i][j] + service_times[i]
    
    # For each customer, pick the best schedule (greedy: pick the one that seems most routing-friendly)
    # Initially just pick the first valid schedule for each customer
    
    def evaluate_schedule_selection(schedule_indices):
        """Given schedule selection for each customer, return day-customer assignments."""
        day_visits = {d: [] for d in range(num_days)}
        for ci, si in enumerate(schedule_indices):
            c = customers[ci]
            sched = c['schedules'][si]
            for part in sched['parts']:
                day = part['visit_day']
                day_visits[day].append({
                    'customer_idx': ci,
                    'customer_id': c['id'],
                    'demand': part['demand_per_visit'],
                    'days_covered': part['days_covered'],
                    'visit_day': part['visit_day']
                })
        return day_visits
    
    def check_time_feasibility(route, demands_map):
        """Check if a route is time-feasible. Route is list of customer indices (1-based in dist matrix)."""
        if not route:
            return True, 0.0
        current_time = time_windows[0][0]  # depot earliest
        # depot to first customer
        prev = 0
        total_duration = 0.0
        for node in route:
            tt = travel_time(prev, node)
            arrival = current_time + tt
            if arrival > time_windows[node][1]:
                return False, float('inf')
            current_time = max(arrival, time_windows[node][0])
            prev = node
        # return to depot
        tt = travel_time(prev, 0)
        arrival = current_time + tt
        total_duration = arrival - time_windows[0][0]
        if arrival > time_windows[0][1]:
            return False, float('inf')
        if total_duration > max_route_duration:
            return False, float('inf')
        return True, total_duration
    
    def route_cost(route):
        """Compute cost of a route (list of node indices, 1-based for customers)."""
        if not route:
            return 0.0
        cost = dist[0][route[0]]
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i + 1]]
        cost += dist[route[-1]][0]
        return cost
    
    def route_duration(route):
        """Compute duration of a route considering time windows and service times."""
        if not route:
            return 0.0
        current_time = time_windows[0][0]
        prev = 0
        for node in route:
            tt = travel_time(prev, node)
            arrival = current_time + tt
            current_time = max(arrival, time_windows[node][0])
            prev = node
        tt = travel_time(prev, 0)
        arrival = current_time + tt
        return arrival - time_windows[0][0]
    
    def nearest_neighbor_routes(day_visits_list):
        """Build routes for a single day using nearest neighbor heuristic."""
        if not day_visits_list:
            return []
        
        # Sort by time window end (earliest deadline first)
        visits = sorted(day_visits_list, key=lambda v: time_windows[v['customer_idx'] + 1][1])
        
        routes = []
        unassigned = list(range(len(visits)))
        
        while unassigned:
            route = []
            route_demand = 0
            current_node = 0
            current_time = time_windows[0][0]
            remaining = list(unassigned)
            route_visit_indices = []
            
            while remaining:
                best_next = None
                best_cost = float('inf')
                best_idx_in_remaining = -1
                
                for ri, vi in enumerate(remaining):
                    v = visits[vi]
                    node = v['customer_idx'] + 1
                    demand = v['demand']
                    
                    if route_demand + demand > vehicle_capacity:
                        continue
                    
                    tt = travel_time(current_node, node)
                    arrival = current_time + tt
                    
                    if arrival > time_windows[node][1]:
                        continue
                    
                    service_start = max(arrival, time_windows[node][0])
                    
                    # Check if we can return to depot after
                    return_time = service_start + travel_time(node, 0)
                    if return_time > time_windows[0][1]:
                        continue
                    
                    # Check total duration
                    total_dur = return_time - time_windows[0][0]
                    if total_dur > max_route_duration:
                        continue
                    
                    cost = dist[current_node][node]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = vi
                        best_idx_in_remaining = ri
                
                if best_next is None:
                    break
                
                v = visits[best_next]
                node = v['customer_idx'] + 1
                tt = travel_time(current_node, node)
                arrival = current_time + tt
                current_time = max(arrival, time_windows[node][0])
                current_node = node
                route_demand += v['demand']
                route.append(best_next)
                route_visit_indices.append(best_next)
                remaining.pop(best_idx_in_remaining)
            
            for vi in route_visit_indices:
                unassigned.remove(vi)
            
            if route:
                route_nodes = [visits[vi]['customer_idx'] + 1 for vi in route]
                routes.append({
                    'nodes': route_nodes,
                    'visits': [visits[vi] for vi in route],
                    'cost': route_cost(route_nodes),
                    'demand': sum(visits[vi]['demand'] for vi in route)
                })
            
            if len(routes) >= num_vehicles and unassigned:
                # Force remaining into existing routes if possible, or create overflow
                # For now, just keep creating routes (might be infeasible)
                pass
        
        return routes
    
    def build_solution(schedule_indices):
        """Build complete solution given schedule selection."""
        day_visits = evaluate_schedule_selection(schedule_indices)
        all_routes = {}
        total_cost = 0.0
        
        for day in range(num_days):
            routes = nearest_neighbor_routes(day_visits[day])
            all_routes[day] = routes
            for r in routes:
                total_cost += r['cost']
        
        return all_routes, total_cost
    
    # Try different schedule selections
    # Start with first schedule for each customer
    best_schedule_indices = [0] * num_customers
    best_routes, best_cost = build_solution(best_schedule_indices)
    
    if logger:
        logger.log(best_cost)
    
    # Try to improve by trying different schedules for each customer
    def try_improve_schedules():
        nonlocal best_schedule_indices, best_routes, best_cost
        
        improved = True
        while improved and time.time() - start_time < time_limit * 0.3:
            improved = False
            for ci in range(num_customers):
                if time.time() - start_time >= time_limit * 0.3:
                    break
                c = customers[ci]
                if len(c['schedules']) <= 1:
                    continue
                
                current_si = best_schedule_indices[ci]
                best_local_cost = best_cost
                best_local_si = current_si
                
                for si in range(len(c['schedules'])):
                    if si == current_si:
                        continue
                    
                    trial = list(best_schedule_indices)
                    trial[ci] = si
                    _, trial_cost = build_solution(trial)
                    
                    if trial_cost < best_local_cost:
                        best_local_cost = trial_cost
                        best_local_si = si
                
                if best_local_si != current_si:
                    best_schedule_indices[ci] = best_local_si
                    best_routes, best_cost = build_solution(best_schedule_indices)
                    improved = True
                    if logger:
                        logger.log(best_cost)
    
    try_improve_schedules()
    
    # Now apply local search on routes (2-opt within routes, relocate between routes)
    def two_opt_route(route_nodes):
        """Apply 2-opt to improve a single route."""
        if len(route_nodes) <= 2:
            return route_nodes, route_cost(route_nodes)
        
        improved = True
        best_nodes = list(route_nodes)
        best_c = route_cost(best_nodes)
        
        while improved:
            improved = False
            for i in range(len(best_nodes) - 1):
                for j in range(i + 1, len(best_nodes)):
                    new_nodes = best_nodes[:i] + best_nodes[i:j + 1][::-1] + best_nodes[j + 1:]
                    feasible, _ = check_time_feasibility(new_nodes, None)
                    if feasible:
                        new_c = route_cost(new_nodes)
                        if new_c < best_c - 1e-10:
                            best_nodes = new_nodes
                            best_c = new_c
                            improved = True
                            break
                if improved:
                    break
        
        return best_nodes, best_c
    
    def optimize_routes():
        nonlocal best_routes, best_cost
        
        total_cost = 0.0
        for day in range(num_days):
            if time.time() - start_time >= time_limit * 0.7:
                break
            for ri, r in enumerate(best_routes[day]):
                if time.time() - start_time >= time_limit * 0.7:
                    break
                new_nodes, new_cost = two_opt_route(r['nodes'])
                if new_cost < r['cost'] - 1e-10:
                    r['nodes'] = new_nodes
                    r['cost'] = new_cost
        
        # Recalculate total cost
        total_cost = sum(r['cost'] for day in range(num_days) for r in best_routes[day])
        if total_cost < best_cost - 1e-10:
            best_cost = total_cost
            if logger:
                logger.log(best_cost)
    
    optimize_routes()
    
    # Or-opt: move sequences of 1, 2, or 3 customers between positions within a route
    def or_opt_route(route_nodes):
        if len(route_nodes) <= 1:
            return route_nodes, route_cost(route_nodes)
        
        best_nodes = list(route_nodes)
        best_c = route_cost(best_nodes)
        improved = True
        
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                if improved:
                    break
                for i in range(len(best_nodes) - seg_len + 1):
                    if improved:
                        break
                    segment = best_nodes[i:i + seg_len]
                    remaining = best_nodes[:i] + best_nodes[i + seg_len:]
                    
                    for j in range(len(remaining) + 1):
                        if j == i:
                            continue
                        new_nodes = remaining[:j] + segment + remaining[j:]
                        feasible, _ = check_time_feasibility(new_nodes, None)
                        if feasible:
                            new_c = route_cost(new_nodes)
                            if new_c < best_c - 1e-10:
                                best_nodes = new_nodes
                                best_c = new_c
                                improved = True
                                break
        
        return best_nodes, best_c
    
    def optimize_routes_or_opt():
        nonlocal best_routes, best_cost
        
        for day in range(num_days):
            if time.time() - start_time >= time_limit * 0.85:
                break
            for ri, r in enumerate(best_routes[day]):
                if time.time() - start_time >= time_limit * 0.85:
                    break
                new_nodes, new_cost = or_opt_route(r['nodes'])
                if new_cost < r['cost'] - 1e-10:
                    r['nodes'] = new_nodes
                    r['cost'] = new_cost
        
        total_cost = sum(r['cost'] for day in range(num_days) for r in best_routes[day])
        if total_cost < best_cost - 1e-10:
            best_cost = total_cost
            if logger:
                logger.log(best_cost)
    
    optimize_routes_or_opt()
    
    # Inter-route relocate: move a customer from one route to another on the same day
    def inter_route_relocate():
        nonlocal best_routes, best_cost
        
        improved_any = True
        while improved_any and time.time() - start_time < time_limit * 0.95:
            improved_any = False
            for day in range(num_days):
                if time.time() - start_time >= time_limit * 0.95:
                    break
                routes = best_routes[day]
                if len(routes) < 2:
                    continue
                
                for r1_idx in range(len(routes)):
                    if time.time() - start_time >= time_limit * 0.95:
                        break
                    r1 = routes[r1_idx]
                    if not r1['nodes']:
                        continue
                    
                    for pos1 in range(len(r1['nodes'])):
                        if time.time() - start_time >= time_limit * 0.95:
                            break
                        
                        node = r1['nodes'][pos1]
                        visit = r1['visits'][pos1]
                        
                        # Cost savings from removing node from r1
                        new_r1_nodes = r1['nodes'][:pos1] + r1['nodes'][pos1 + 1:]
                        feasible1, _ = check_time_feasibility(new_r1_nodes, None)
                        if not feasible1:
                            continue
                        new_r1_cost = route_cost(new_r1_nodes)
                        savings = r1['cost'] - new_r1_cost
                        
                        best_insert_cost = float('inf')
                        best_r2_idx = -1
                        best_insert_pos = -1
                        
                        for r2_idx in range(len(routes)):
                            if r2_idx == r1_idx:
                                continue
                            r2 = routes[r2_idx]
                            
                            # Check capacity
                            if r2['demand'] + visit['demand'] > vehicle_capacity:
                                continue
                            
                            for pos2 in range(len(r2['nodes']) + 1):
                                new_r2_nodes = r2['nodes'][:pos2] + [node] + r2['nodes'][pos2:]
                                feasible2, _ = check_time_feasibility(new_r2_nodes, None)
                                if not feasible2:
                                    continue
                                new_r2_cost = route_cost(new_r2_nodes)
                                insert_cost = new_r2_cost - r2['cost']
                                
                                if insert_cost < best_insert_cost:
                                    best_insert_cost = insert_cost
                                    best_r2_idx = r2_idx
                                    best_insert_pos = pos2
                        
                        if best_r2_idx >= 0 and savings - best_insert_cost > 1e-10:
                            # Perform the move
                            r2 = routes[best_r2_idx]
                            
                            # Remove from r1
                            r1['nodes'] = new_r1_nodes
                            r1['visits'] = r1['visits'][:pos1] + r1['visits'][pos1 + 1:]
                            r1['cost'] = new_r1_cost
                            r1['demand'] -= visit['demand']
                            
                            # Insert into r2
                            r2['nodes'] = r2['nodes'][:best_insert_pos] + [node] + r2['nodes'][best_insert_pos:]
                            r2['visits'] = r2['visits'][:best_insert_pos] + [visit] + r2['visits'][best_insert_pos:]
                            r2['cost'] = route_cost(r2['nodes'])
                            r2['demand'] += visit['demand']
                            
                            improved_any = True
                            break
            
            # Remove empty routes
            for day in range(num_days):
                best_routes[day] = [r for r in best_routes[day] if r['nodes']]
            
            total_cost = sum(r['cost'] for day in range(num_days) for r in best_routes[day])
            if total_cost < best_cost - 1e-10:
                best_cost = total_cost
                if logger:
                    logger.log(best_cost)
    
    inter_route_relocate()
    
    # Final 2-opt pass
    if time.time() - start_time < time_limit * 0.98:
        optimize_routes()
    
    # Format output
    output_routes = {}
    for day in range(num_days):
        day_routes = []
        for r in best_routes[day]:
            if not r['nodes']:
                continue
            customer_ids = [customers[n - 1]['id'] for n in r['nodes']]
            parts = {}
            for v in r['visits']:
                parts[str(v['customer_id'])] = {
                    'visit_day': v['visit_day'],
                    'days_covered': v['days_covered'],
                    'demand_per_visit': v['demand']
                }
            day_routes.append({
                'customers': customer_ids,
                'cost': r['cost'],
                'demand': r['demand'],
                'parts': parts
            })
        output_routes[str(day)] = day_routes
    
    output_schedules = {}
    for ci in range(num_customers):
        c = customers[ci]
        si = best_schedule_indices[ci]
        sched = c['schedules'][si]
        output_schedules[str(c['id'])] = {
            'schedule_index': si,
            'days': sched['days'],
            'visit_frequency': sched['visit_frequency']
        }
    
    solution = {
        'objective_value': best_cost,
        'routes': output_routes,
        'schedules': output_schedules
    }
    
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