import json
import argparse
import time
import math
import random
import copy
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Parse instance
    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']

    depot = instance['depot']
    depot_end = instance['depot_end']
    customers = instance['customers']
    dist = instance['distance_matrix']
    rho = instance['inconvenience_cost_function']['rho']

    DEPOT = 0
    DEPOT_END = num_customers + 1

    # Customer data arrays
    demand = [0] * (num_customers + 2)
    tw_early = [0.0] * (num_customers + 2)
    tw_late = [float('inf')] * (num_customers + 2)
    serv_time = [0.0] * (num_customers + 2)
    preferred = [0.0] * (num_customers + 2)

    tw_early[DEPOT] = float(depot['time_window'][0])
    tw_late[DEPOT] = float(depot['time_window'][1])
    tw_early[DEPOT_END] = float(depot_end['time_window'][0])
    tw_late[DEPOT_END] = float(depot_end['time_window'][1])

    for c in customers:
        cid = c['id']
        demand[cid] = c['demand']
        tw_early[cid] = float(c['time_window'][0])
        tw_late[cid] = float(c['time_window'][1])
        serv_time[cid] = float(c['service_time'])
        preferred[cid] = float(c['preferred_time'])

    def evaluate_route(route):
        """
        Evaluate a route with optimized service start times.
        Returns (feasible, routing_cost, inconvenience_cost, schedule_dict)
        """
        n_stops = len(route)
        if n_stops <= 2:
            return True, 0.0, 0.0, {}

        # Forward pass: earliest feasible times
        times = [0.0] * n_stops
        times[0] = tw_early[DEPOT]

        for i in range(1, n_stops):
            node = route[i]
            prev_node = route[i - 1]
            travel = dist[prev_node][node]

            if i == 1:
                arrival = times[0] + travel
            else:
                arrival = times[i - 1] + serv_time[prev_node] + travel

            if node == DEPOT_END:
                if arrival > tw_late[DEPOT_END] + 1e-9:
                    return False, float('inf'), float('inf'), {}
                times[i] = arrival
            else:
                service_start = max(arrival, tw_early[node])
                if service_start > tw_late[node] + 1e-9:
                    return False, float('inf'), float('inf'), {}
                times[i] = service_start

        # Backward pass: latest feasible times
        latest = [0.0] * n_stops
        latest[n_stops - 1] = tw_late[DEPOT_END]

        for i in range(n_stops - 2, 0, -1):
            node = route[i]
            next_node = route[i + 1]
            travel = dist[node][next_node]
            max_start = latest[i + 1] - serv_time[node] - travel
            latest[i] = min(tw_late[node], max_start)

        # Optimize service times: iterative forward-backward sweeps
        for _ in range(10):
            changed = False
            # Forward sweep
            for i in range(1, n_stops - 1):
                node = route[i]
                if i == 1:
                    lb = times[0] + dist[route[0]][node]
                else:
                    lb = times[i - 1] + serv_time[route[i - 1]] + dist[route[i - 1]][node]
                lb = max(lb, tw_early[node])

                ub = latest[i]
                if i < n_stops - 2:
                    next_node = route[i + 1]
                    ub2 = times[i + 1] - serv_time[node] - dist[node][next_node]
                    ub = min(ub, ub2)

                if lb <= ub + 1e-9:
                    target = max(lb, min(ub, preferred[node]))
                    if abs(target - times[i]) > 1e-10:
                        times[i] = target
                        changed = True

            # Backward sweep
            for i in range(n_stops - 2, 0, -1):
                node = route[i]
                if i == 1:
                    lb = times[0] + dist[route[0]][node]
                else:
                    lb = times[i - 1] + serv_time[route[i - 1]] + dist[route[i - 1]][node]
                lb = max(lb, tw_early[node])

                if i < n_stops - 2:
                    next_node = route[i + 1]
                    ub = times[i + 1] - serv_time[node] - dist[node][next_node]
                else:
                    ub = tw_late[DEPOT_END] - serv_time[node] - dist[node][DEPOT_END]
                ub = min(ub, tw_late[node])

                if lb <= ub + 1e-9:
                    target = max(lb, min(ub, preferred[node]))
                    if abs(target - times[i]) > 1e-10:
                        times[i] = target
                        changed = True

            if not changed:
                break

        # Final feasibility check
        for i in range(1, n_stops - 1):
            node = route[i]
            if times[i] < tw_early[node] - 1e-6 or times[i] > tw_late[node] + 1e-6:
                return False, float('inf'), float('inf'), {}
            if i > 1:
                prev_node = route[i - 1]
                min_arrival = times[i - 1] + serv_time[prev_node] + dist[prev_node][node]
                if times[i] < min_arrival - 1e-6:
                    return False, float('inf'), float('inf'), {}

        if n_stops > 2:
            last_cust = route[n_stops - 2]
            arrival_end = times[n_stops - 2] + serv_time[last_cust] + dist[last_cust][DEPOT_END]
            if arrival_end > tw_late[DEPOT_END] + 1e-6:
                return False, float('inf'), float('inf'), {}

        # Compute costs
        routing_cost = sum(dist[route[i]][route[i + 1]] for i in range(n_stops - 1))
        inconvenience = 0.0
        schedule = {}
        for i in range(1, n_stops - 1):
            node = route[i]
            schedule[node] = times[i]
            inconvenience += rho * (times[i] - preferred[node]) ** 2

        return True, routing_cost, inconvenience, schedule

    def evaluate_solution(routes):
        total_routing = 0.0
        total_inconvenience = 0.0
        full_schedule = {}

        for route in routes:
            feasible, rc, ic, sched = evaluate_route(route)
            if not feasible:
                return False, float('inf'), {}
            total_routing += rc
            total_inconvenience += ic
            full_schedule.update(sched)

        return True, total_routing + total_inconvenience, full_schedule

    def get_route_demand(route):
        return sum(demand[c] for c in route if 1 <= c <= num_customers)

    def make_solution_dict(routes, obj, schedule):
        return {
            "objective_value": obj,
            "routes": routes,
            "schedule": {str(k): v for k, v in schedule.items()}
        }

    # Constructive heuristic: cheapest insertion
    def construct_solution():
        routes = [[DEPOT, DEPOT_END] for _ in range(num_vehicles)]
        unassigned = list(range(1, num_customers + 1))
        random.shuffle(unassigned)

        for cust in unassigned:
            best_cost = float('inf')
            best_r = -1
            best_p = -1

            for r_idx in range(len(routes)):
                if get_route_demand(routes[r_idx]) + demand[cust] > vehicle_capacity:
                    continue
                for pos in range(1, len(routes[r_idx])):
                    new_route = routes[r_idx][:pos] + [cust] + routes[r_idx][pos:]
                    feasible, rc, ic, _ = evaluate_route(new_route)
                    if feasible:
                        cost = rc + ic
                        if cost < best_cost:
                            best_cost = cost
                            best_r = r_idx
                            best_p = pos

            if best_r >= 0:
                routes[best_r] = routes[best_r][:best_p] + [cust] + routes[best_r][best_p:]
            else:
                # Try without capacity check as fallback
                for r_idx in range(len(routes)):
                    for pos in range(1, len(routes[r_idx])):
                        new_route = routes[r_idx][:pos] + [cust] + routes[r_idx][pos:]
                        feasible, rc, ic, _ = evaluate_route(new_route)
                        if feasible:
                            routes[r_idx] = new_route
                            break
                    else:
                        continue
                    break

        return routes

    # Local search: intra-route 2-opt
    def intra_route_2opt(routes):
        improved = True
        while improved and time.time() - start_time < args.time_limit - 1:
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        feasible_new, cost_new, _, _ = evaluate_route(new_route)
                        if feasible_new:
                            feasible_old, cost_old, _, _ = evaluate_route(route)
                            if feasible_old and cost_new < cost_old - 1e-9:
                                route = new_route
                                improved = True
                routes[r_idx] = route
        return routes

    # Local search: inter-route relocate
    def inter_route_relocate(routes):
        improved = True
        while improved and time.time() - start_time < args.time_limit - 1:
            improved = False
            for r1 in range(len(routes)):
                if len(routes[r1]) <= 2:
                    continue
                for pos1 in range(1, len(routes[r1]) - 1):
                    cust = routes[r1][pos1]
                    new_r1 = routes[r1][:pos1] + routes[r1][pos1 + 1:]

                    for r2 in range(len(routes)):
                        if r2 == r1:
                            continue
                        if get_route_demand(routes[r2]) + demand[cust] > vehicle_capacity:
                            continue
                        for pos2 in range(1, len(routes[r2])):
                            new_r2 = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                            f1, c1, _, _ = evaluate_route(new_r1)
                            f2, c2, _, _ = evaluate_route(new_r2)
                            if f1 and f2:
                                fo1, oc1, _, _ = evaluate_route(routes[r1])
                                fo2, oc2, _, _ = evaluate_route(routes[r2])
                                if c1 + c2 < oc1 + oc2 - 1e-9:
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
        return routes

    # Local search: swap between routes
    def inter_route_swap(routes):
        improved = True
        while improved and time.time() - start_time < args.time_limit - 1:
            improved = False
            for r1 in range(len(routes)):
                for r2 in range(r1 + 1, len(routes)):
                    for p1 in range(1, len(routes[r1]) - 1):
                        for p2 in range(1, len(routes[r2]) - 1):
                            c1 = routes[r1][p1]
                            c2 = routes[r2][p2]
                            new_r1 = routes[r1][:p1] + [c2] + routes[r1][p1 + 1:]
                            new_r2 = routes[r2][:p2] + [c1] + routes[r2][p2 + 1:]

                            if get_route_demand(new_r1) > vehicle_capacity:
                                continue
                            if get_route_demand(new_r2) > vehicle_capacity:
                                continue

                            f1, nc1, _, _ = evaluate_route(new_r1)
                            f2, nc2, _, _ = evaluate_route(new_r2)
                            if f1 and f2:
                                _, oc1, _, _ = evaluate_route(routes[r1])
                                _, oc2, _, _ = evaluate_route(routes[r2])
                                if nc1 + nc2 < oc1 + oc2 - 1e-9:
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
        return routes

    # ALNS
    def alns(routes, time_limit_val):
        best_routes = copy.deepcopy(routes)
        _, best_cost, _ = evaluate_solution(best_routes)

        current_routes = copy.deepcopy(routes)
        _, current_cost, _ = evaluate_solution(current_routes)

        temp = max(best_cost * 0.05, 1.0)
        cooling = 0.9998
        no_improve = 0

        while time.time() - start_time < time_limit_val - 0.5:
            # Get all customers
            all_custs = []
            for r in current_routes:
                for c in r:
                    if 1 <= c <= num_customers:
                        all_custs.append(c)

            if not all_custs:
                break

            # Choose destroy
            destroy_type = random.choice(['random', 'worst', 'related', 'route'])
            num_remove = max(1, min(int(random.uniform(0.1, 0.4) * len(all_custs)), len(all_custs)))

            if destroy_type == 'random':
                removed = set(random.sample(all_custs, num_remove))
            elif destroy_type == 'worst':
                _, _, sched = evaluate_solution(current_routes)
                scored = []
                for c in all_custs:
                    if c in sched:
                        ic = rho * (sched[c] - preferred[c]) ** 2
                        scored.append((ic + random.random() * 0.01, c))
                scored.sort(reverse=True)
                removed = set(c for _, c in scored[:num_remove])
            elif destroy_type == 'related':
                seed_c = random.choice(all_custs)
                others = [(dist[seed_c][c] + random.random() * 0.01, c) for c in all_custs if c != seed_c]
                others.sort()
                removed = {seed_c} | set(c for _, c in others[:num_remove - 1])
            else:
                non_empty = [i for i, r in enumerate(current_routes) if len(r) > 2]
                if non_empty:
                    r_idx = random.choice(non_empty)
                    removed = set(c for c in current_routes[r_idx] if 1 <= c <= num_customers)
                else:
                    removed = set(random.sample(all_custs, min(num_remove, len(all_custs))))

            # Destroy
            new_routes = []
            for route in current_routes:
                new_route = [c for c in route if c not in removed]
                if len(new_route) < 2:
                    new_route = [DEPOT, DEPOT_END]
                elif new_route[-1] != DEPOT_END:
                    new_route.append(DEPOT_END)
                new_routes.append(new_route)

            # Repair: cheapest insertion
            removed_list = list(removed)
            random.shuffle(removed_list)
            all_inserted = True

            for cust in removed_list:
                best_cost_ins = float('inf')
                best_r_ins = -1
                best_p_ins = -1

                for r_idx in range(len(new_routes)):
                    if get_route_demand(new_routes[r_idx]) + demand[cust] > vehicle_capacity:
                        continue
                    for pos in range(1, len(new_routes[r_idx])):
                        new_route = new_routes[r_idx][:pos] + [cust] + new_routes[r_idx][pos:]
                        feasible, rc, ic, _ = evaluate_route(new_route)
                        if feasible:
                            cost = rc + ic
                            if cost < best_cost_ins:
                                best_cost_ins = cost
                                best_r_ins = r_idx
                                best_p_ins = pos

                if best_r_ins >= 0:
                    new_routes[best_r_ins] = new_routes[best_r_ins][:best_p_ins] + [cust] + new_routes[best_r_ins][best_p_ins:]
                else:
                    all_inserted = False
                    break

            if not all_inserted:
                temp *= cooling
                no_improve += 1
                if no_improve > 300:
                    temp = max(best_cost * 0.05, 1.0)
                    no_improve = 0
                continue

            # Evaluate
            feasible, new_cost, _ = evaluate_solution(new_routes)

            if feasible:
                if new_cost < current_cost:
                    current_routes = new_routes
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_routes = copy.deepcopy(new_routes)
                        best_cost = new_cost
                        no_improve = 0

                        if logger:
                            _, _, sched = evaluate_solution(best_routes)
                            sol_dict = make_solution_dict(best_routes, best_cost, sched)
                            logger.log_solution(best_cost, sol_dict)
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    delta = new_cost - current_cost
                    if temp > 0.001 and random.random() < math.exp(-delta / max(temp, 1e-10)):
                        current_routes = new_routes
                        current_cost = new_cost

                temp *= cooling

                if no_improve > 300:
                    temp = max(best_cost * 0.05, 1.0)
                    no_improve = 0

        return best_routes, best_cost

    # Main execution
    random.seed(42)

    # Build initial solution
    routes = construct_solution()
    feasible, obj, schedule = evaluate_solution(routes)

    if feasible and logger:
        sol_dict = make_solution_dict(routes, obj, schedule)
        logger.log_solution(obj, sol_dict)

    # Local search phase
    if time.time() - start_time < args.time_limit - 2:
        routes = intra_route_2opt(routes)
        feasible, obj, schedule = evaluate_solution(routes)
        if feasible and logger:
            sol_dict = make_solution_dict(routes, obj, schedule)
            logger.log_solution(obj, sol_dict)

    if time.time() - start_time < args.time_limit - 2:
        routes = inter_route_relocate(routes)
        feasible, obj, schedule = evaluate_solution(routes)
        if feasible and logger:
            sol_dict = make_solution_dict(routes, obj, schedule)
            logger.log_solution(obj, sol_dict)

    if time.time() - start_time < args.time_limit - 2:
        routes = inter_route_swap(routes)
        feasible, obj, schedule = evaluate_solution(routes)
        if feasible and logger:
            sol_dict = make_solution_dict(routes, obj, schedule)
            logger.log_solution(obj, sol_dict)

    # ALNS phase
    if time.time() - start_time < args.time_limit - 2:
        routes, obj = alns(routes, args.time_limit)
        feasible, obj, schedule = evaluate_solution(routes)
        if feasible and logger:
            sol_dict = make_solution_dict(routes, obj, schedule)
            logger.log_solution(obj, sol_dict)

    # Final local search polish
    if time.time() - start_time < args.time_limit - 1:
        routes = intra_route_2opt(routes)
        feasible, obj, schedule = evaluate_solution(routes)
        if feasible and logger:
            sol_dict = make_solution_dict(routes, obj, schedule)
            logger.log_solution(obj, sol_dict)

    # Final output
    feasible, obj, schedule = evaluate_solution(routes)

    if not feasible:
        # Fallback: reconstruct
        random.seed(0)
        routes = construct_solution()
        feasible, obj, schedule = evaluate_solution(routes)

    sol_dict = make_solution_dict(routes, obj, schedule)

    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f, indent=2)


if __name__ == '__main__':
    main()