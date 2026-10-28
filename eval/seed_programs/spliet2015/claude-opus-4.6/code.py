import argparse
import json
import time
import math
import random
from copy import deepcopy
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance['num_customers']
    depot = instance['depot']
    vehicle_capacity = instance['vehicle_capacity']
    etw_width = instance.get('endogenous_time_window_width', 0)
    customers = instance['customers']
    scenarios = instance['scenarios']
    dist_matrix = instance['distance_matrix']

    # Node indices in distance matrix:
    # 0 = start depot
    # 1..num_customers = customers (customer id corresponds to index)
    # num_customers+1 = end depot
    start_depot_idx = 0
    end_depot_idx = num_customers + 1

    depot_etw = depot['exogenous_time_window']

    # Customer info indexed by customer id
    cust_info = {}
    for c in customers:
        cid = c['id']
        cust_info[cid] = {
            'x': c['x'],
            'y': c['y'],
            'exo_tw': c['exogenous_time_window'],
            'etw_width': c.get('endogenous_time_window_width', etw_width)
        }

    # Helper: travel cost/time from node i to node j (using distance matrix indices)
    def travel(i, j):
        return dist_matrix[i][j]

    # Assign endogenous time windows
    # Strategy: place the endogenous TW to maximize flexibility
    # We'll try centering within the exogenous TW initially, then optimize later
    def assign_centered_tw():
        tw = {}
        for c in customers:
            cid = c['id']
            info = cust_info[cid]
            exo_start, exo_end = info['exo_tw']
            width = info['etw_width']
            # Center the endogenous TW within the exogenous TW
            center = (exo_start + exo_end) / 2.0
            endo_start = center - width / 2.0
            # Clamp
            endo_start = max(exo_start, min(exo_end - width, endo_start))
            tw[cid] = [endo_start, endo_start + width]
        return tw

    def assign_early_tw():
        tw = {}
        for c in customers:
            cid = c['id']
            info = cust_info[cid]
            exo_start, exo_end = info['exo_tw']
            width = info['etw_width']
            tw[cid] = [exo_start, exo_start + width]
        return tw

    def assign_late_tw():
        tw = {}
        for c in customers:
            cid = c['id']
            info = cust_info[cid]
            exo_start, exo_end = info['exo_tw']
            width = info['etw_width']
            endo_start = exo_end - width
            tw[cid] = [endo_start, exo_end]
        return tw

    # Build routes for a scenario given time windows
    # Using nearest-neighbor heuristic with capacity and time window constraints
    def build_routes_for_scenario(scenario, time_windows):
        demands = scenario['demands']  # indexed by customer position (0-based matching customer ids 1..n)
        # demands[i] corresponds to customer with id (i+1) based on schema: "ordered by customer id"
        # Actually, let's check: customers are 1..num_customers, demands is length num_customers
        # demands[0] = demand of customer 1, demands[1] = demand of customer 2, etc.

        customer_ids = [c['id'] for c in customers]
        cust_demand = {}
        for i, cid in enumerate(customer_ids):
            cust_demand[cid] = demands[i]

        unvisited = set(customer_ids)
        routes = []

        while unvisited:
            route = [start_depot_idx]
            current_node = start_depot_idx
            current_time = depot_etw[0]
            current_load = 0
            route_customers = []

            while True:
                best_next = None
                best_cost = float('inf')
                best_arrival = None

                for cid in unvisited:
                    d = cust_demand[cid]
                    if current_load + d > vehicle_capacity:
                        continue

                    t = travel(current_node, cid)
                    arrival = current_time + t
                    tw_start, tw_end = time_windows[cid]
                    exo_start, exo_end = cust_info[cid]['exo_tw']

                    # Must arrive such that service can start within both exogenous and endogenous TW
                    service_start = max(arrival, tw_start, exo_start)
                    if service_start > tw_end or service_start > exo_end:
                        continue

                    # Check if we can return to depot after serving this customer
                    return_time = service_start + travel(cid, end_depot_idx)
                    if return_time > depot_etw[1]:
                        continue

                    cost = t
                    # Prioritize by cost (distance), break ties by earliest feasible service
                    if cost < best_cost or (cost == best_cost and (best_arrival is None or service_start < best_arrival)):
                        best_cost = cost
                        best_next = cid
                        best_arrival = service_start

                if best_next is None:
                    break

                route.append(best_next)
                route_customers.append(best_next)
                unvisited.remove(best_next)
                current_node = best_next
                current_time = best_arrival
                current_load += cust_demand[best_next]

            route.append(end_depot_idx)
            if route_customers:
                routes.append(route)
            else:
                # No customer could be added - shouldn't happen if instance is feasible
                # Force add the nearest unvisited customer
                if unvisited:
                    forced = min(unvisited, key=lambda cid: travel(start_depot_idx, cid))
                    routes.append([start_depot_idx, forced, end_depot_idx])
                    unvisited.remove(forced)

        return routes

    def compute_arrival_times(route, time_windows):
        """Compute arrival/service times for each node in a route."""
        arrivals = {}
        current_time = depot_etw[0]
        arrivals[route[0]] = current_time  # start depot

        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            t = travel(prev, curr)
            arrival = current_time + t

            if curr == end_depot_idx:
                service_start = arrival
            elif curr == start_depot_idx:
                service_start = max(arrival, depot_etw[0])
            else:
                tw_start, tw_end = time_windows[curr]
                exo_start = cust_info[curr]['exo_tw'][0]
                service_start = max(arrival, tw_start, exo_start)

            arrivals[curr] = service_start
            current_time = service_start

        return arrivals

    def compute_route_cost(route):
        cost = 0
        for i in range(len(route) - 1):
            cost += travel(route[i], route[i + 1])
        return cost

    def validate_route(route, time_windows, scenario):
        """Check if route is feasible."""
        demands = scenario['demands']
        customer_ids = [c['id'] for c in customers]
        cust_demand = {}
        for i, cid in enumerate(customer_ids):
            cust_demand[cid] = demands[i]

        # Check capacity
        total_demand = sum(cust_demand.get(n, 0) for n in route if n != start_depot_idx and n != end_depot_idx)
        if total_demand > vehicle_capacity:
            return False

        # Check time windows
        current_time = depot_etw[0]
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            t = travel(prev, curr)
            arrival = current_time + t

            if curr == end_depot_idx:
                if arrival > depot_etw[1]:
                    return False
                current_time = arrival
            elif curr == start_depot_idx:
                current_time = max(arrival, depot_etw[0])
            else:
                tw_start, tw_end = time_windows[curr]
                exo_start, exo_end = cust_info[curr]['exo_tw']
                service_start = max(arrival, tw_start, exo_start)
                if service_start > tw_end or service_start > exo_end:
                    return False
                current_time = service_start

        return True

    def evaluate_solution(time_windows, scenarios_list):
        """Evaluate total expected cost for given time windows."""
        total_cost = 0
        all_routes = {}
        for scenario in scenarios_list:
            sid = scenario['scenario_id']
            prob = scenario['probability']
            routes = build_routes_for_scenario(scenario, time_windows)
            scenario_cost = sum(compute_route_cost(r) for r in routes)
            total_cost += prob * scenario_cost
            all_routes[sid] = routes
        return total_cost, all_routes

    # 2-opt improvement for a single route
    def two_opt_route(route, time_windows, scenario):
        """Apply 2-opt to improve a single route."""
        if len(route) <= 4:  # start_depot, at most 2 customers, end_depot
            return route

        improved = True
        best_route = route[:]
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    if validate_route(new_route, time_windows, scenario):
                        new_cost = compute_route_cost(new_route)
                        old_cost = compute_route_cost(best_route)
                        if new_cost < old_cost - 1e-10:
                            best_route = new_route
                            improved = True
                            break
                if improved:
                    break
        return best_route

    # Or-opt: move a segment of 1, 2, or 3 customers to another position
    def or_opt_route(route, time_windows, scenario):
        if len(route) <= 4:
            return route

        best_route = route[:]
        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(1, len(best_route) - seg_len - 1):
                    segment = best_route[i:i + seg_len]
                    remaining = best_route[:i] + best_route[i + seg_len:]
                    for j in range(1, len(remaining)):
                        new_route = remaining[:j] + segment + remaining[j:]
                        if validate_route(new_route, time_windows, scenario):
                            new_cost = compute_route_cost(new_route)
                            old_cost = compute_route_cost(best_route)
                            if new_cost < old_cost - 1e-10:
                                best_route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return best_route

    # Inter-route operators
    def try_relocate(routes, time_windows, scenario):
        """Try moving a customer from one route to another."""
        demands = scenario['demands']
        customer_ids = [c['id'] for c in customers]
        cust_demand = {}
        for i, cid in enumerate(customer_ids):
            cust_demand[cid] = demands[i]

        improved = True
        while improved:
            improved = False
            for ri in range(len(routes)):
                for ci in range(1, len(routes[ri]) - 1):
                    cust = routes[ri][ci]
                    # Try inserting into another route
                    for rj in range(len(routes)):
                        if ri == rj:
                            continue
                        # Check capacity
                        rj_demand = sum(cust_demand.get(n, 0) for n in routes[rj]
                                        if n != start_depot_idx and n != end_depot_idx)
                        if rj_demand + cust_demand[cust] > vehicle_capacity:
                            continue

                        # Try all positions in rj
                        best_pos = None
                        best_delta = 0

                        # Cost of removing cust from ri
                        prev_i = routes[ri][ci - 1]
                        next_i = routes[ri][ci + 1]
                        remove_delta = (travel(prev_i, next_i)
                                        - travel(prev_i, cust) - travel(cust, next_i))

                        for pos in range(1, len(routes[rj])):
                            prev_j = routes[rj][pos - 1]
                            next_j = routes[rj][pos]
                            insert_delta = (travel(prev_j, cust) + travel(cust, next_j)
                                            - travel(prev_j, next_j))
                            total_delta = remove_delta + insert_delta
                            if total_delta < best_delta - 1e-10:
                                # Check feasibility
                                new_rj = routes[rj][:pos] + [cust] + routes[rj][pos:]
                                new_ri = routes[ri][:ci] + routes[ri][ci + 1:]
                                if (validate_route(new_rj, time_windows, scenario) and
                                        (len(new_ri) <= 2 or validate_route(new_ri, time_windows, scenario))):
                                    best_delta = total_delta
                                    best_pos = pos

                        if best_pos is not None:
                            new_rj = routes[rj][:best_pos] + [cust] + routes[rj][best_pos:]
                            new_ri = routes[ri][:ci] + routes[ri][ci + 1:]
                            routes[rj] = new_rj
                            routes[ri] = new_ri
                            # Remove empty routes
                            routes = [r for r in routes if len(r) > 2]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def try_swap(routes, time_windows, scenario):
        """Try swapping customers between routes."""
        improved = True
        while improved:
            improved = False
            for ri in range(len(routes)):
                for ci in range(1, len(routes[ri]) - 1):
                    for rj in range(ri + 1, len(routes)):
                        for cj in range(1, len(routes[rj]) - 1):
                            # Try swapping routes[ri][ci] and routes[rj][cj]
                            demands_list = scenario['demands']
                            customer_ids = [c['id'] for c in customers]
                            cust_demand = {}
                            for k, cid in enumerate(customer_ids):
                                cust_demand[cid] = demands_list[k]

                            c1 = routes[ri][ci]
                            c2 = routes[rj][cj]

                            # Check capacity
                            ri_demand = sum(cust_demand.get(n, 0) for n in routes[ri]
                                            if n != start_depot_idx and n != end_depot_idx)
                            rj_demand = sum(cust_demand.get(n, 0) for n in routes[rj]
                                            if n != start_depot_idx and n != end_depot_idx)

                            new_ri_demand = ri_demand - cust_demand[c1] + cust_demand[c2]
                            new_rj_demand = rj_demand - cust_demand[c2] + cust_demand[c1]

                            if new_ri_demand > vehicle_capacity or new_rj_demand > vehicle_capacity:
                                continue

                            # Cost change
                            old_cost = compute_route_cost(routes[ri]) + compute_route_cost(routes[rj])

                            new_ri = routes[ri][:]
                            new_ri[ci] = c2
                            new_rj = routes[rj][:]
                            new_rj[cj] = c1

                            new_cost = compute_route_cost(new_ri) + compute_route_cost(new_rj)

                            if new_cost < old_cost - 1e-10:
                                if (validate_route(new_ri, time_windows, scenario) and
                                        validate_route(new_rj, time_windows, scenario)):
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

    def improve_routes(routes, time_windows, scenario):
        """Apply local search to improve routes for a scenario."""
        # Intra-route improvements
        for i in range(len(routes)):
            routes[i] = two_opt_route(routes[i], time_windows, scenario)
            routes[i] = or_opt_route(routes[i], time_windows, scenario)

        # Inter-route improvements
        routes = try_relocate(routes, time_windows, scenario)
        routes = try_swap(routes, time_windows, scenario)

        # Another round of intra-route
        for i in range(len(routes)):
            routes[i] = two_opt_route(routes[i], time_windows, scenario)

        return routes

    def build_and_improve(time_windows, scenarios_list, time_budget=None):
        """Build and improve routes for all scenarios."""
        total_cost = 0
        all_routes = {}
        for scenario in scenarios_list:
            if time_budget is not None and time.time() - start_time > time_budget:
                # Just build without improvement
                routes = build_routes_for_scenario(scenario, time_windows)
            else:
                routes = build_routes_for_scenario(scenario, time_windows)
                routes = improve_routes(routes, time_windows, scenario)

            sid = scenario['scenario_id']
            prob = scenario['probability']
            scenario_cost = sum(compute_route_cost(r) for r in routes)
            total_cost += prob * scenario_cost
            all_routes[sid] = routes
        return total_cost, all_routes

    # Build initial solution with different TW strategies and pick the best
    best_obj = float('inf')
    best_tw = None
    best_routes = None

    # Try different TW assignment strategies
    tw_strategies = [assign_centered_tw, assign_early_tw, assign_late_tw]

    for strategy in tw_strategies:
        if time.time() - start_time > time_limit * 0.8:
            break
        tw = strategy()
        obj, routes = build_and_improve(tw, scenarios, time_limit * 0.6)
        if obj < best_obj:
            best_obj = obj
            best_tw = tw
            best_routes = routes
            if logger:
                logger.log(best_obj)

    # Local search on time windows
    # Try shifting each customer's TW start and see if it improves expected cost
    def try_tw_perturbation(time_windows, best_obj, best_routes):
        customer_ids = [c['id'] for c in customers]
        improved = True
        iteration = 0
        while improved and time.time() - start_time < time_limit * 0.9:
            improved = False
            iteration += 1
            random.shuffle(customer_ids)
            for cid in customer_ids:
                if time.time() - start_time > time_limit * 0.9:
                    break
                info = cust_info[cid]
                exo_start, exo_end = info['exo_tw']
                width = info['etw_width']
                current_start = time_windows[cid][0]

                # Try several positions
                min_start = exo_start
                max_start = exo_end - width
                if max_start <= min_start:
                    continue

                # Try a grid of positions
                num_steps = 10
                step = (max_start - min_start) / num_steps
                candidates = [min_start + i * step for i in range(num_steps + 1)]
                candidates.append(current_start)

                best_local_obj = best_obj
                best_local_start = current_start

                for cand_start in candidates:
                    if abs(cand_start - current_start) < 1e-6:
                        continue
                    old_tw = time_windows[cid]
                    time_windows[cid] = [cand_start, cand_start + width]

                    obj, routes = build_and_improve(time_windows, scenarios,
                                                     time_limit * 0.85)
                    if obj < best_local_obj - 1e-6:
                        best_local_obj = obj
                        best_local_start = cand_start
                        best_local_routes = routes

                    time_windows[cid] = old_tw

                if best_local_start != current_start:
                    time_windows[cid] = [best_local_start, best_local_start + width]
                    best_obj = best_local_obj
                    best_routes = best_local_routes
                    improved = True
                    if logger:
                        logger.log(best_obj)

        return time_windows, best_obj, best_routes

    if time.time() - start_time < time_limit * 0.7:
        best_tw, best_obj, best_routes = try_tw_perturbation(
            deepcopy(best_tw), best_obj, best_routes)

    # Additional improvement: re-run route improvement with more time
    if time.time() - start_time < time_limit * 0.95:
        obj2, routes2 = build_and_improve(best_tw, scenarios, time_limit * 0.95)
        if obj2 < best_obj:
            best_obj = obj2
            best_routes = routes2
            if logger:
                logger.log(best_obj)

    # Format output
    time_windows_output = {}
    for cid, tw in best_tw.items():
        time_windows_output[str(cid)] = tw

    routes_output = {}
    for sid, routes in best_routes.items():
        scenario_routes = []
        for route in routes:
            arrivals = compute_arrival_times(route, best_tw)
            cost = compute_route_cost(route)
            arrival_dict = {}
            for node in route:
                arrival_dict[str(node)] = arrivals[node]
            scenario_routes.append({
                'route': route,
                'arrival_times': arrival_dict,
                'cost': cost
            })
        routes_output[str(sid)] = scenario_routes

    solution = {
        'objective_value': best_obj,
        'time_windows': time_windows_output,
        'routes': routes_output
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

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)