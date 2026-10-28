import json
import argparse
import time
import sys
from typing import Dict, List, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']                    # customers 0..n-1
    d = instance['d']                    # depots 0..d-1
    t = instance['t']                    # periods (1-based in patterns)
    m = instance['m']                    # vehicles per depot
    Q = instance['vehicle_capacity']
    Tmax = instance['route_duration_limit']

    depots = instance['depots']
    customers = instance['customers']

    # Precompute distance matrix (symmetric, Euclidean)
    N_total = d + n
    dist = [[0.0 for _ in range(N_total)] for _ in range(N_total)]
    coords = [(depots[i]['x'], depots[i]['y']) for i in range(d)]
    for cust in customers:
        coords.append((cust['x'], cust['y']))

    for i in range(N_total):
        for j in range(N_total):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist[i][j] = (dx*dx + dy*dy)**0.5

    # Customer data
    demand = [c['demand'] for c in customers]
    service = [c['service_duration'] for c in customers]
    patterns = [c['pattern_list'] for c in customers]  # list of lists (1-based days)

    start_time = time.time()
    best_obj = float('inf')
    best_solution = None

    # Simple greedy heuristic: assign each customer to closest depot with feasible pattern
    # Then solve daily VRP approximately using cheapest insertion for each day

    assignment = {}
    customer_to_depot = [-1] * n
    customer_pattern = [None] * n
    customer_days = [set() for _ in range(n)]

    for i in range(n):
        best_cost = float('inf')
        best_dep = -1
        best_pat = None
        for pat_idx, pat in enumerate(patterns[i]):
            for dep in range(d):
                # cost proxy = distance to depot * frequency
                freq = len(pat)
                cost_proxy = dist[dep][d + i] * freq
                if cost_proxy < best_cost:
                    best_cost = cost_proxy
                    best_dep = dep
                    best_pat = pat[:]
        assignment[str(i)] = {"depot": best_dep, "pattern": best_pat}
        customer_to_depot[i] = best_dep
        customer_pattern[i] = best_pat
        customer_days[i] = set(best_pat)

    # For each depot and each day, collect customers that must be served
    depot_day_customers = [[[] for _ in range(t)] for _ in range(d)]

    for i in range(n):
        dep = customer_to_depot[i]
        for day1 in customer_days[i]:  # 1-based
            day0 = day1 - 1
            depot_day_customers[dep][day0].append(i)

    # Now for each depot-day, solve a simple VRP using cheapest insertion + multiple vehicles
    total_cost = 0.0
    all_routes = []

    for dep in range(d):
        for day0 in range(t):
            customers_today = depot_day_customers[dep][day0]
            if not customers_today:
                continue

            # Sort customers by distance from depot (farthest first for insertion)
            cust_with_dist = [(c, dist[dep][d + c]) for c in customers_today]
            cust_with_dist.sort(key=lambda x: -x[1])
            remaining = [c for c, _ in cust_with_dist]

            vehicle_routes = []
            vehicle_loads = []
            vehicle_durations = []

            while remaining:
                # Start a new route
                route = []
                load = 0
                duration = 0
                last = dep  # start at depot (index in full graph)

                inserted = True
                while inserted and remaining:
                    inserted = False
                    best_c = None
                    best_pos = None
                    best_inc = float('inf')

                    for idx, c in enumerate(remaining):
                        c_idx = d + c
                        for pos in range(len(route) + 1):
                            # insert between pos-1 and pos
                            prev = last if pos == 0 else (d + route[pos-1])
                            nxt = dep if pos == len(route) else (d + route[pos])
                            inc = dist[prev][c_idx] + dist[c_idx][nxt] - dist[prev][nxt]
                            inc += service[c]
                            new_load = load + demand[c]
                            new_dur = duration + inc
                            if new_load <= Q and new_dur <= Tmax:
                                if inc < best_inc:
                                    best_inc = inc
                                    best_c = c
                                    best_pos = pos

                    if best_c is not None:
                        route.insert(best_pos, best_c)
                        load += demand[best_c]
                        # update duration accurately
                        if best_pos == 0:
                            duration += dist[dep][d + best_c] + service[best_c]
                        else:
                            prev_c = route[best_pos-1]
                            duration += dist[d + prev_c][d + best_c] + service[best_c]
                        if best_pos == len(route) - 1:
                            duration += dist[d + best_c][dep]
                        else:
                            next_c = route[best_pos+1]
                            duration -= dist[d + best_c][d + next_c]  # already added in inc
                        # better to rebuild duration
                        # rebuild to avoid bugs
                        duration = 0.0
                        prev = dep
                        for cc in route:
                            duration += dist[prev][d + cc] + service[cc]
                            prev = d + cc
                        duration += dist[prev][dep]
                        load = sum(demand[c] for c in route)
                        remaining.remove(best_c)
                        inserted = True
                        last = d + route[-1] if route else dep

                # If no customer could be inserted, we have a problem (but with enough vehicles should be ok)
                if not route:
                    # Fallback: one customer per vehicle if capacity violated
                    if remaining:
                        c = remaining.pop(0)
                        route = [c]
                        duration = dist[dep][d+c] + service[c] + dist[d+c][dep]
                        load = demand[c]
                vehicle_routes.append(route)
                vehicle_loads.append(load)
                vehicle_durations.append(duration)
                total_cost += duration  # since cost = time here

            # Record routes
            for v_idx, route in enumerate(vehicle_routes):
                if route:
                    all_routes.append({
                        "depot": dep,
                        "period": day0 + 1,
                        "vehicle": v_idx + 1,
                        "customers": route
                    })

    # Final solution
    solution = {
        "objective_value": total_cost,
        "routes": all_routes,
        "assignments": assignment
    }

    # Log the solution
    if logger:
        logger.log_solution(total_cost, solution)

    # Write output
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Final objective: {total_cost}", file=sys.stderr)


if __name__ == "__main__":
    main()