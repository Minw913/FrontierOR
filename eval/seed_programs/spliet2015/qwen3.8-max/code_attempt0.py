import json
import argparse
import time
import random
import math
from copy import deepcopy
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    start_time = time.time()
    time_limit = args.time_limit
    random.seed(42)

    # Parse instance
    num_customers = instance["num_customers"]
    depot = instance["depot"]
    capacity = instance["vehicle_capacity"]
    customers = instance["customers"]
    scenarios = instance["scenarios"]
    dist_matrix = instance["distance_matrix"]

    depot_exo = depot["exogenous_time_window"]
    start_depot = 0
    end_depot = num_customers + 1

    # Customer info indexed by customer id (= matrix index)
    cust_info = {}
    cust_indices = []
    for c in customers:
        cid = c["id"]
        cust_indices.append(cid)
        cust_info[cid] = {
            "exo_tw": c["exogenous_time_window"],
            "endo_width": c["endogenous_time_window_width"]
        }
    cust_indices.sort()

    # Scenario demands: dict scenario_id -> {cust_id: demand}
    scenario_demands = {}
    for s in scenarios:
        sid = s["scenario_id"]
        scenario_demands[sid] = {}
        for i, cid in enumerate(cust_indices):
            scenario_demands[sid][cid] = s["demands"][i]

    def time_elapsed():
        return time.time() - start_time

    def remaining():
        return max(0, time_limit - time_elapsed())

    def get_route_cost(route):
        cost = 0.0
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i]][route[i + 1]]
        return cost

    def compute_service_times(route, endo_windows):
        """Compute service times for a feasible route."""
        times = {}
        current_time = float(depot_exo[0])
        times[route[0]] = current_time
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            arrival = current_time + dist_matrix[prev][curr]
            if curr == end_depot:
                times[curr] = arrival
                current_time = arrival
            else:
                tw_start = endo_windows[curr][0]
                service = max(arrival, tw_start)
                times[curr] = service
                current_time = service
        return times

    def is_route_feasible(route, demands, endo_windows):
        """Check capacity and time window feasibility."""
        total_dem = 0
        for i in range(1, len(route) - 1):
            total_dem += demands[route[i]]
        if total_dem > capacity:
            return False

        current_time = float(depot_exo[0])
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            arrival = current_time + dist_matrix[prev][curr]
            if curr == end_depot:
                if arrival > depot_exo[1]:
                    return False
                current_time = arrival
            else:
                tw = endo_windows[curr]
                service = max(arrival, tw[0])
                if service > tw[1]:
                    return False
                current_time = service
        return True

    def cheapest_insertion(demands, endo_windows):
        """Construct routes using cheapest insertion heuristic."""
        unserved = set(cust_indices)
        routes = []

        # Sort customers by endogenous window start for initial seeding
        sorted_custs = sorted(unserved, key=lambda c: endo_windows[c][0])

        while unserved:
            if remaining() < 1:
                break
            # Pick the unserved customer with earliest window end as seed
            seed = min(unserved, key=lambda c: endo_windows[c][1])
            route = [start_depot, seed, end_depot]
            unserved.remove(seed)
            route_demand = demands[seed]

            # Try inserting customers
            changed = True
            while changed and unserved:
                changed = False
                best_cost = float('inf')
                best_cust = None
                best_pos = None

                for c in unserved:
                    if route_demand + demands[c] > capacity:
                        continue
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [c] + route[pos:]
                        if is_route_feasible(new_route, demands, endo_windows):
                            ins_cost = (dist_matrix[route[pos - 1]][c] +
                                        dist_matrix[c][route[pos]] -
                                        dist_matrix[route[pos - 1]][route[pos]])
                            if ins_cost < best_cost:
                                best_cost = ins_cost
                                best_cust = c
                                best_pos = pos

                if best_cust is not None:
                    route = route[:best_pos] + [best_cust] + route[best_pos:]
                    route_demand += demands[best_cust]
                    unserved.remove(best_cust)
                    changed = True

            routes.append(route)

        # If time ran out, create individual routes for remaining
        for c in unserved:
            routes.append([start_depot, c, end_depot])

        return routes

    def or_opt_improve(route, demands, endo_windows):
        """Try or-opt moves (relocate segments of 1, 2, 3 customers)."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            for seg_len in [1, 2, 3]:
                for i in range(1, n - seg_len):
                    segment = route[i:i + seg_len]
                    remaining_route = route[:i] + route[i + seg_len:]
                    for j in range(1, len(remaining_route)):
                        new_route = remaining_route[:j] + segment + remaining_route[j:]
                        if is_route_feasible(new_route, demands, endo_windows):
                            if get_route_cost(new_route) < get_route_cost(route) - 1e-9:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route

    def two_opt_improve(route, demands, endo_windows):
        """2-opt improvement within a route."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if is_route_feasible(new_route, demands, endo_windows):
                        if get_route_cost(new_route) < get_route_cost(route) - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def relocate_between_routes(routes, demands, endo_windows):
        """Try relocating a customer from one route to another."""
        improved = True
        while improved:
            improved = False
            for r1 in range(len(routes)):
                for i in range(1, len(routes[r1]) - 1):
                    cust = routes[r1][i]
                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        # Check capacity
                        dem_r2 = sum(demands[routes[r2][k]] for k in range(1, len(routes[r2]) - 1))
                        if dem_r2 + demands[cust] > capacity:
                            continue
                        for j in range(1, len(routes[r2])):
                            new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                            new_r2 = routes[r2][:j] + [cust] + routes[r2][j:]
                            if is_route_feasible(new_r1, demands, endo_windows) and \
                               is_route_feasible(new_r2, demands, endo_windows):
                                old_cost = get_route_cost(routes[r1]) + get_route_cost(routes[r2])
                                new_cost = get_route_cost(new_r1) + get_route_cost(new_r2)
                                if new_cost < old_cost - 1e-9:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            # Remove empty routes
            routes = [r for r in routes if len(r) > 2]
        return routes

    def solve_scenario_full(demands, endo_windows, do_improve=True):
        """Solve VRPTW for one scenario with construction + improvement."""
        routes = cheapest_insertion(demands, endo_windows)

        if do_improve and remaining() > 2:
            # Intra-route improvements
            for idx in range(len(routes)):
                routes[idx] = or_opt_improve(routes[idx], demands, endo_windows)
                if remaining() < 2:
                    break

            # Inter-route improvements
            if remaining() > 3:
                routes = relocate_between_routes(routes, demands, endo_windows)

            # Another round of intra-route
            if remaining() > 2:
                for idx in range(len(routes)):
                    routes[idx] = two_opt_improve(routes[idx], demands, endo_windows)
                    if remaining() < 2:
                        break

        return routes

    def evaluate_all(endo_windows, do_improve=True):
        """Evaluate solution quality across all scenarios."""
        total_cost = 0.0
        all_routes = {}
        for s in scenarios:
            sid = s["scenario_id"]
            demands = scenario_demands[sid]
            routes = solve_scenario_full(demands, endo_windows, do_improve)
            sc_cost = sum(get_route_cost(r) for r in routes)
            total_cost += s["probability"] * sc_cost
            all_routes[sid] = routes
        return total_cost, all_routes

    def build_solution(endo_windows, all_routes, obj):
        """Build output solution dictionary."""
        solution = {
            "objective_value": round(obj, 6),
            "time_windows": {},
            "routes": {}
        }
        for c in cust_indices:
            solution["time_windows"][str(c)] = [round(endo_windows[c][0], 6), round(endo_windows[c][1], 6)]

        for s in scenarios:
            sid = s["scenario_id"]
            solution["routes"][str(sid)] = []
            for route in all_routes[sid]:
                times = compute_service_times(route, endo_windows)
                route_entry = {
                    "route": route,
                    "arrival_times": {str(k): round(v, 6) for k, v in times.items()},
                    "cost": round(get_route_cost(route), 6)
                }
                solution["routes"][str(sid)].append(route_entry)
        return solution

    # Initialize endogenous windows at center of exogenous
    endo_windows = {}
    for c in cust_indices:
        exo = cust_info[c]["exo_tw"]
        width = cust_info[c]["endo_width"]
        start = exo[0] + (exo[1] - exo[0] - width) / 2.0
        endo_windows[c] = [start, start + width]

    # Get initial solution
    best_obj, best_routes = evaluate_all(endo_windows, do_improve=True)
    best_endo = deepcopy(endo_windows)

    solution = build_solution(best_endo, best_routes, best_obj)
    if logger:
        logger.log_solution(best_obj, solution)

    # Local search on time window placement
    iteration = 0
    while remaining() > 5:
        iteration += 1
        improved = False

        # Strategy 1: Try shifting each customer's window
        for c in cust_indices:
            if remaining() < 5:
                break
            exo = cust_info[c]["exo_tw"]
            width = cust_info[c]["endo_width"]
            max_start = exo[1] - width
            min_start = exo[0]

            current_start = endo_windows[c][0]

            # Try several alternative positions
            candidates = []
            # Shift left and right
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                new_start = current_start + delta * (max_start - min_start)
                if min_start <= new_start <= max_start:
                    candidates.append(new_start)

            # Also try extremes and center
            candidates.append(min_start)
            candidates.append(max_start)
            candidates.append(min_start + (max_start - min_start) / 2.0)

            for new_start in candidates:
                if abs(new_start - current_start) < 1e-9:
                    continue
                # Temporarily change
                old_window = endo_windows[c]
                endo_windows[c] = [new_start, new_start + width]

                # Quick evaluation (no improvement to save time)
                new_obj, new_routes = evaluate_all(endo_windows, do_improve=False)

                if new_obj < best_obj - 1e-9:
                    # Verify with full improvement
                    new_obj_full, new_routes_full = evaluate_all(endo_windows, do_improve=True)
                    if new_obj_full < best_obj - 1e-9:
                        best_obj = new_obj_full
                        best_routes = new_routes_full
                        best_endo = deepcopy(endo_windows)
                        improved = True
                        solution = build_solution(best_endo, best_routes, best_obj)
                        if logger:
                            logger.log_solution(best_obj, solution)
                        break
                    else:
                        endo_windows[c] = old_window
                else:
                    endo_windows[c] = old_window

            if improved:
                break

        if not improved:
            # Strategy 2: Random perturbation
            if remaining() > 10:
                c = random.choice(cust_indices)
                exo = cust_info[c]["exo_tw"]
                width = cust_info[c]["endo_width"]
                max_start = exo[1] - width
                min_start = exo[0]
                new_start = random.uniform(min_start, max_start)

                old_window = endo_windows[c]
                endo_windows[c] = [new_start, new_start + width]

                new_obj, new_routes = evaluate_all(endo_windows, do_improve=True)
                if new_obj < best_obj - 1e-9:
                    best_obj = new_obj
                    best_routes = new_routes
                    best_endo = deepcopy(endo_windows)
                    solution = build_solution(best_endo, best_routes, best_obj)
                    if logger:
                        logger.log_solution(best_obj, solution)
                else:
                    endo_windows[c] = old_window

        if iteration > 100:
            break

    # Final: try to improve routes with current best time windows
    if remaining() > 3:
        final_obj, final_routes = evaluate_all(best_endo, do_improve=True)
        if final_obj < best_obj - 1e-9:
            best_obj = final_obj
            best_routes = final_routes
            solution = build_solution(best_endo, best_routes, best_obj)
            if logger:
                logger.log_solution(best_obj, solution)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()