import json
import random
import math
import time
import argparse

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    random.seed(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    N = instance['num_customers']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    Gamma = instance['uncertainty_set']['Gamma']

    customers = instance['customers']
    d = {c['id']: c['nominal_demand'] for c in customers}
    dev = {c['id']: c['demand_deviation'] for c in customers}
    customer_ids = [c['id'] for c in customers]
    dist = instance['distance_matrix']

    depot = 0  # row/column 0 is the depot

    max_dist = max(max(row) for row in dist) if dist and dist[0] else 1
    PENALTY = max(1000, max_dist * 1000)

    def robust_demand(route):
        if not route:
            return 0
        nominal = sum(d[i] for i in route)
        if Gamma == 0:
            return nominal
        devs = [dev[i] for i in route]
        if len(devs) <= Gamma:
            return nominal + sum(devs)
        else:
            devs.sort(reverse=True)
            return nominal + sum(devs[:Gamma])

    def route_cost(route):
        if not route:
            return 0
        cost = dist[depot][route[0]]
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i + 1]]
        cost += dist[route[-1]][depot]
        return cost

    def evaluate(routes):
        travel = 0
        viol = 0
        for r in routes:
            travel += route_cost(r)
            rd = robust_demand(r)
            if rd > Q:
                viol += rd - Q
        return travel, viol

    def build_solution_dict(routes):
        sol_routes = []
        total_cost = 0
        for r in routes:
            cost = route_cost(r)
            wd = robust_demand(r)
            total_cost += cost
            sol_routes.append({
                "customers": list(r),
                "cost": int(cost),
                "worst_case_demand": int(wd)
            })
        return {
            "objective_value": float(total_cost),
            "routes": sol_routes
        }

    def order_route(custs):
        if not custs:
            return []
        route = []
        current = depot
        remaining = set(custs)
        while remaining:
            nxt = min(remaining, key=lambda c: dist[current][c])
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt
        return route

    def fix_feasibility(routes):
        improved = True
        while improved:
            improved = False
            for r in range(K):
                if robust_demand(routes[r]) > Q:
                    for c in list(routes[r]):
                        old_viol = max(0, robust_demand(routes[r]) - Q)
                        for r2 in range(K):
                            if r2 == r:
                                continue
                            new_r = [x for x in routes[r] if x != c]
                            new_r2 = routes[r2] + [c]
                            new_viol_r = max(0, robust_demand(new_r) - Q)
                            new_viol_r2 = max(0, robust_demand(new_r2) - Q)
                            if new_viol_r + new_viol_r2 < old_viol + max(0, robust_demand(routes[r2]) - Q):
                                routes[r] = new_r
                                routes[r2] = new_r2
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        return routes

    # Initial solution
    routes = [[] for _ in range(K)]
    if N == 0:
        pass
    elif K >= N:
        for i, c in enumerate(customer_ids):
            routes[i] = [c]
    else:
        sorted_customers = sorted(customer_ids, key=lambda c: -(d[c] + (dev[c] if Gamma > 0 else 0)))
        for c in sorted_customers:
            placed = False
            for r in range(K):
                if robust_demand(routes[r] + [c]) <= Q:
                    routes[r].append(c)
                    placed = True
                    break
            if not placed:
                best_r = min(range(K), key=lambda r: robust_demand(routes[r]))
                routes[best_r].append(c)
        for r in range(K):
            routes[r] = order_route(routes[r])

    current_travel, current_viol = evaluate(routes)
    current_obj = current_travel + PENALTY * current_viol
    best_feasible = None
    best_feasible_cost = float('inf')

    if current_viol > 0:
        routes = fix_feasibility(routes)
        current_travel, current_viol = evaluate(routes)
        current_obj = current_travel + PENALTY * current_viol

    if current_viol == 0:
        best_feasible = [list(r) for r in routes]
        best_feasible_cost = current_travel
        if logger:
            logger.log_solution(float(best_feasible_cost), build_solution_dict(best_feasible))

    T0 = max_dist * 10 if max_dist > 0 else 1000.0
    T = T0
    cooling = 0.9995
    min_T = 1e-6
    max_q = min(N, max(2, int(0.2 * N))) if N > 0 else 1

    start_time = time.time()

    while time.time() - start_time < args.time_limit:
        op = random.random()
        q = random.randint(1, max_q)
        new_routes = [list(r) for r in routes]
        removed = []

        if op < 0.33:
            all_c = [c for r in new_routes for c in r]
            if all_c:
                removed = random.sample(all_c, min(q, len(all_c)))
                new_routes = [[c for c in r if c not in removed] for r in new_routes]
        elif op < 0.66:
            savings = []
            for r_idx, r in enumerate(new_routes):
                for pos, c in enumerate(r):
                    prev = depot if pos == 0 else r[pos - 1]
                    nxt = depot if pos == len(r) - 1 else r[pos + 1]
                    saving = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                    savings.append((saving, c))
            savings.sort(key=lambda x: x[0], reverse=True)
            removed = [c for _, c in savings[:q]]
            new_routes = [[c for c in r if c not in removed] for r in new_routes]
        else:
            all_c = [c for r in new_routes for c in r]
            if all_c:
                seed = random.choice(all_c)
                dists = [(dist[seed][c], c) for c in all_c]
                dists.sort(key=lambda x: x[0])
                removed = [c for _, c in dists[:q]]
                new_routes = [[c for c in r if c not in removed] for r in new_routes]

        removed.sort(key=lambda c: -(d[c] + (dev[c] if Gamma > 0 else 0)))
        for c in removed:
            best_r = -1
            best_pos = -1
            best_score = float('inf')
            for r_idx, r in enumerate(new_routes):
                base_rd = robust_demand(r)
                base_viol = max(0, base_rd - Q)
                new_rd = robust_demand(r + [c])
                new_viol = max(0, new_rd - Q)
                viol_delta = new_viol - base_viol
                for pos in range(len(r) + 1):
                    prev = depot if pos == 0 else r[pos - 1]
                    nxt = depot if pos == len(r) else r[pos]
                    cost_delta = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                    score = cost_delta + PENALTY * viol_delta
                    if score < best_score:
                        best_score = score
                        best_r = r_idx
                        best_pos = pos
            new_routes[best_r].insert(best_pos, c)

        new_travel, new_viol = evaluate(new_routes)
        new_obj = new_travel + PENALTY * new_viol

        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / T):
            routes = new_routes
            current_obj = new_obj
            current_travel, current_viol = new_travel, new_viol
            if current_viol == 0 and current_travel < best_feasible_cost:
                best_feasible = [list(r) for r in routes]
                best_feasible_cost = current_travel
                if logger:
                    logger.log_solution(float(best_feasible_cost), build_solution_dict(best_feasible))

        T *= cooling
        if T < min_T:
            T = T0 * 0.5

    if best_feasible is not None:
        final_routes = best_feasible
    else:
        final_routes = routes

    sol_dict = build_solution_dict(final_routes)
    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f, indent=2)


if __name__ == '__main__':
    main()