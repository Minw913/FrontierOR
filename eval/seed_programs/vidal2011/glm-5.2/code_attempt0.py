import argparse
import json
import math
import time
import sys

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    start_time = time.time()
    time_limit = args.time_limit

    # ------------------------------------------------------------------
    # Read instance
    # ------------------------------------------------------------------
    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['n']
    d = inst['d']
    t = inst['t']
    m = inst['m']
    Q = inst['vehicle_capacity']
    D = inst['route_duration_limit']
    depots = inst['depots']
    customers = inst['customers']

    depot_coords = [(dep['x'], dep['y']) for dep in depots]
    cust_coords = [(cust['x'], cust['y']) for cust in customers]
    demands = [cust['demand'] for cust in customers]
    service_durations = [cust['service_duration'] for cust in customers]
    pattern_lists = [cust['pattern_list'] for cust in customers]

    # Distance matrix (nodes 0..d-1 = depots, d..d+n-1 = customers)
    num_nodes = d + n
    dist = [[0.0] * num_nodes for _ in range(num_nodes)]
    coords = depot_coords + cust_coords
    for i in range(num_nodes):
        xi, yi = coords[i]
        for j in range(i + 1, num_nodes):
            xj, yj = coords[j]
            dij = math.hypot(xi - xj, yi - yj)
            dist[i][j] = dij
            dist[j][i] = dij

    def dist_dc(dep, cust):
        return dist[dep][d + cust]

    def dist_cc(c1, c2):
        return dist[d + c1][d + c2]

    def route_duration(dep, route):
        if not route:
            return 0.0
        dur = dist_dc(dep, route[0])
        for i in range(len(route) - 1):
            dur += dist_cc(route[i], route[i + 1])
        dur += dist_dc(dep, route[-1])
        for c in route:
            dur += service_durations[c]
        return dur

    def route_demand(route):
        return sum(demands[c] for c in route)

    def route_cost(dep, route):
        if not route:
            return 0.0
        cost = dist_dc(dep, route[0])
        for i in range(len(route) - 1):
            cost += dist_cc(route[i], route[i + 1])
        cost += dist_dc(dep, route[-1])
        return cost

    def route_feasible(dep, route):
        return route_demand(route) <= Q and route_duration(dep, route) <= D

    # ------------------------------------------------------------------
    # Assignment MIP (joint depot & pattern selection)
    # ------------------------------------------------------------------
    def solve_assignment_mip(time_limit_mip):
        model = gp.Model("assignment")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit_mip

        x = {}
        for c in range(n):
            for dep in range(d):
                for p_idx, pat in enumerate(pattern_lists[c]):
                    x[c, dep, p_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{c}_{dep}_{p_idx}")

        obj = gp.LinExpr()
        for c in range(n):
            for dep in range(d):
                for p_idx, pat in enumerate(pattern_lists[c]):
                    obj += 2 * dist_dc(dep, c) * len(pat) * x[c, dep, p_idx]
        model.setObjective(obj, GRB.MINIMIZE)

        for c in range(n):
            model.addConstr(
                gp.quicksum(x[c, dep, p_idx]
                            for dep in range(d)
                            for p_idx in range(len(pattern_lists[c]))) == 1
            )

        for dep in range(d):
            for day in range(1, t + 1):
                model.addConstr(
                    gp.quicksum(demands[c] * x[c, dep, p_idx]
                                for c in range(n)
                                for p_idx, pat in enumerate(pattern_lists[c])
                                if day in pat) <= m * Q
                )

        model.optimize()
        if model.SolCount == 0:
            return {}
        assignments = {}
        for c in range(n):
            for dep in range(d):
                for p_idx, pat in enumerate(pattern_lists[c]):
                    if x[c, dep, p_idx].X > 0.5:
                        assignments[c] = (dep, pat)
                        break
            if c not in assignments:
                assignments[c] = (0, pattern_lists[c][0])
        return assignments

    def greedy_assignment():
        assignments = {}
        cust_order = sorted(range(n), key=lambda c: demands[c], reverse=True)
        load = {}
        for c in cust_order:
            best_dep = -1
            best_pat = None
            best_cost = float('inf')
            for dep in range(d):
                for pat in pattern_lists[c]:
                    feasible = True
                    for day in pat:
                        if load.get((dep, day), 0) + demands[c] > m * Q:
                            feasible = False
                            break
                    if not feasible:
                        continue
                    cost = 2 * dist_dc(dep, c) * len(pat)
                    if cost < best_cost:
                        best_cost = cost
                        best_dep = dep
                        best_pat = pat
            if best_dep == -1:  # no capacity-feasible choice
                for dep in range(d):
                    for pat in pattern_lists[c]:
                        cost = 2 * dist_dc(dep, c) * len(pat)
                        if cost < best_cost:
                            best_cost = cost
                            best_dep = dep
                            best_pat = pat
            assignments[c] = (best_dep, best_pat)
            for day in best_pat:
                load[(best_dep, day)] = load.get((best_dep, day), 0) + demands[c]
        return assignments

    assignments = {}
    if GUROBI_AVAILABLE:
        assignments = solve_assignment_mip(max(5.0, time_limit * 0.2))
    if not assignments:
        assignments = greedy_assignment()

    # ------------------------------------------------------------------
    # Build day-depot customer lists
    # ------------------------------------------------------------------
    day_customers = {}
    for c in range(n):
        dep, pat = assignments[c]
        for day in pat:
            key = (dep, day)
            if key not in day_customers:
                day_customers[key] = []
            day_customers[key].append(c)

    routes = []

    # ------------------------------------------------------------------
    # Routing heuristics
    # ------------------------------------------------------------------
    def tsp_heuristic(dep, customers):
        if not customers:
            return []
        unvisited = set(customers)
        seq = []
        while unvisited:
            if not seq:
                nearest = min(unvisited, key=lambda c: dist_dc(dep, c))
            else:
                nearest = min(unvisited, key=lambda c: dist_cc(seq[-1], c))
            seq.append(nearest)
            unvisited.remove(nearest)

        def seq_cost(s):
            if not s:
                return 0.0
            cost = dist_dc(dep, s[0])
            for i in range(len(s) - 1):
                cost += dist_cc(s[i], s[i + 1])
            cost += dist_dc(dep, s[-1])
            return cost

        best_seq = seq[:]
        best_cost = seq_cost(best_seq)
        improved = True
        while improved:
            improved = False
            for i in range(len(seq) - 1):
                for j in range(i + 1, len(seq)):
                    new_seq = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    new_cost = seq_cost(new_seq)
                    if new_cost < best_cost - 1e-6:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
            seq = best_seq
        return best_seq

    def dp_split(dep, seq):
        n_seq = len(seq)
        if n_seq == 0:
            return []
        pref_demand = [0] * (n_seq + 1)
        pref_service = [0] * (n_seq + 1)
        for i in range(n_seq):
            pref_demand[i + 1] = pref_demand[i] + demands[seq[i]]
            pref_service[i + 1] = pref_service[i] + service_durations[seq[i]]
        pref_dist = [0.0] * (n_seq + 1)
        for i in range(1, n_seq):
            pref_dist[i] = pref_dist[i - 1] + dist_cc(seq[i - 1], seq[i])

        def seg_travel(i, j):
            travel = dist_dc(dep, seq[i])
            if j > i:
                travel += pref_dist[j] - pref_dist[i]
            travel += dist_dc(dep, seq[j])
            return travel

        def seg_demand(i, j):
            return pref_demand[j + 1] - pref_demand[i]

        def seg_service(i, j):
            return pref_service[j + 1] - pref_service[i]

        def seg_feasible(i, j):
            return seg_demand(i, j) <= Q and seg_travel(i, j) + seg_service(i, j) <= D

        INF = float('inf')
        dp = [[INF] * (n_seq + 1) for _ in range(m + 1)]
        parent = [[-1] * (n_seq + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for r in range(1, m + 1):
            for i in range(1, n_seq + 1):
                for j in range(i):
                    if dp[r - 1][j] < INF and seg_feasible(j, i - 1):
                        cost = dp[r - 1][j] + seg_travel(j, i - 1) + seg_service(j, i - 1)
                        if cost < dp[r][i]:
                            dp[r][i] = cost
                            parent[r][i] = j
        best_cost = INF
        best_r = -1
        for r in range(1, m + 1):
            if dp[r][n_seq] < best_cost:
                best_cost = dp[r][n_seq]
                best_r = r
        if best_r == -1:
            return None
        result_routes = []
        cur_i = n_seq
        cur_r = best_r
        while cur_r > 0:
            j = parent[cur_r][cur_i]
            route = seq[j:cur_i]
            result_routes.append(route)
            cur_i = j
            cur_r -= 1
        result_routes.reverse()
        return result_routes

    def savings_algorithm(dep, customers):
        if not customers:
            return []
        routes = [[c] for c in customers]
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist_dc(dep, ci) + dist_dc(dep, cj) - dist_cc(ci, cj)
                savings.append((s, ci, cj))
        savings.sort(reverse=True, key=lambda x: x[0])

        def find_route(c):
            for idx, r in enumerate(routes):
                if c in r:
                    return idx, r
            return -1, None

        for s, ci, cj in savings:
            idx_i, r_i = find_route(ci)
            idx_j, r_j = find_route(cj)
            if idx_i == -1 or idx_j == -1 or idx_i == idx_j:
                continue
            if r_i[-1] == ci and r_j[0] == cj:
                new_route = r_i + r_j
            elif r_i[0] == ci and r_j[-1] == cj:
                new_route = r_j + r_i
            elif r_i[-1] == ci and r_j[-1] == cj:
                new_route = r_i + r_j[::-1]
            elif r_i[0] == ci and r_j[0] == cj:
                new_route = r_i[::-1] + r_j
            else:
                continue
            if route_demand(new_route) <= Q and route_duration(dep, new_route) <= D:
                routes = [r for k, r in enumerate(routes) if k != idx_i and k != idx_j]
                routes.append(new_route)

        if len(routes) > m:
            merged = True
            while merged and len(routes) > m:
                merged = False
                for i in range(len(routes)):
                    for j in range(i + 1, len(routes)):
                        for ri, rj in [(routes[i], routes[j]),
                                       (routes[i], routes[j][::-1]),
                                       (routes[i][::-1], routes[j]),
                                       (routes[i][::-1], routes[j][::-1])]:
                            new_route = ri + rj
                            if route_demand(new_route) <= Q and route_duration(dep, new_route) <= D:
                                routes = [r for k, r in enumerate(routes) if k != i and k != j]
                                routes.append(new_route)
                                merged = True
                                break
                        if merged:
                            break
                    if merged:
                        break
        return routes

    def first_fit_decreasing(dep, customers):
        sorted_custs = sorted(customers, key=lambda c: demands[c], reverse=True)
        routes = [[] for _ in range(m)]
        for c in sorted_custs:
            placed = False
            for i in range(m):
                new_route = routes[i] + [c]
                if route_demand(new_route) <= Q and route_duration(dep, new_route) <= D:
                    routes[i].append(c)
                    placed = True
                    break
            if not placed:
                for i in range(m):
                    for pos in range(len(routes[i]) + 1):
                        new_route = routes[i][:pos] + [c] + routes[i][pos:]
                        if route_demand(new_route) <= Q and route_duration(dep, new_route) <= D:
                            routes[i] = new_route
                            placed = True
                            break
                    if placed:
                        break
            if not placed:
                return None
        routes = [r for r in routes if r]
        return routes

    # ------------------------------------------------------------------
    # Initial routing for every depot and day
    # ------------------------------------------------------------------
    for dep in range(d):
        for day in range(1, t + 1):
            custs = day_customers.get((dep, day), [])
            if not custs:
                continue
            seq = tsp_heuristic(dep, custs)
            day_routes = dp_split(dep, seq)
            if day_routes is None or len(day_routes) > m:
                day_routes = savings_algorithm(dep, custs)
                if day_routes is None or len(day_routes) > m:
                    day_routes = first_fit_decreasing(dep, custs)
            if day_routes is None:
                day_routes = []
            for v_idx, route in enumerate(day_routes):
                if route:
                    routes.append({
                        "depot": dep,
                        "period": day,
                        "vehicle": v_idx + 1,
                        "customers": route
                    })

    # ------------------------------------------------------------------
    # Feasibility & best solution tracking
    # ------------------------------------------------------------------
    def is_feasible():
        for dep in range(d):
            for day in range(1, t + 1):
                day_routes = [r for r in routes if r['depot'] == dep and r['period'] == day]
                if len(day_routes) > m:
                    return False
                covered = set()
                for r in day_routes:
                    if not route_feasible(dep, r['customers']):
                        return False
                    for c in r['customers']:
                        if c in covered:
                            return False
                        covered.add(c)
                if covered != set(day_customers.get((dep, day), [])):
                    return False
        return True

    best_solution = None
    best_cost = float('inf')

    def update_best():
        nonlocal best_solution, best_cost
        if not is_feasible():
            return
        cost = sum(route_cost(r['depot'], r['customers']) for r in routes)
        if cost < best_cost:
            best_cost = cost
            best_solution = {
                "objective_value": cost,
                "routes": [dict(r) for r in routes],
                "assignments": {str(c): {"depot": assignments[c][0], "pattern": list(assignments[c][1])}
                                 for c in range(n)}
            }
            if logger:
                logger.log_solution(best_cost, best_solution)

    update_best()

    # ------------------------------------------------------------------
    # Repair infeasible assignments
    # ------------------------------------------------------------------
    if not is_feasible():
        while time.time() - start_time < time_limit and not is_feasible():
            infeasible_day = None
            for dep in range(d):
                for day in range(1, t + 1):
                    day_routes = [r for r in routes if r['depot'] == dep and r['period'] == day]
                    if len(day_routes) > m:
                        infeasible_day = (dep, day)
                        break
                    if not day_routes and day_customers.get((dep, day), []):
                        infeasible_day = (dep, day)
                        break
                    for r in day_routes:
                        if not route_feasible(dep, r['customers']):
                            infeasible_day = (dep, day)
                            break
                    if infeasible_day:
                        break
                if infeasible_day:
                    break
            if not infeasible_day:
                break

            dep, day = infeasible_day
            moved = False
            for c in day_customers.get((dep, day), []):
                for new_dep in range(d):
                    if new_dep == dep:
                        continue
                    for new_pat in pattern_lists[c]:
                        if day in new_pat:
                            continue
                        feasible = True
                        for new_day in new_pat:
                            new_custs = day_customers.get((new_dep, new_day), [])
                            total_demand = sum(demands[c2] for c2 in new_custs) + demands[c]
                            if total_demand > m * Q:
                                feasible = False
                                break
                        if not feasible:
                            continue
                        assignments[c] = (new_dep, new_pat)
                        day_customers[(dep, day)].remove(c)
                        for new_day in new_pat:
                            if (new_dep, new_day) not in day_customers:
                                day_customers[(new_dep, new_day)] = []
                            day_customers[(new_dep, new_day)].append(c)
                        affected_days = [(dep, day)] + [(new_dep, new_day) for new_day in new_pat]
                        routes = [r for r in routes if (r['depot'], r['period']) not in affected_days]
                        for d2, day2 in affected_days:
                            custs2 = day_customers.get((d2, day2), [])
                            if not custs2:
                                continue
                            seq2 = tsp_heuristic(d2, custs2)
                            day_routes2 = dp_split(d2, seq2)
                            if day_routes2 is None or len(day_routes2) > m:
                                day_routes2 = savings_algorithm(d2, custs2)
                                if day_routes2 is None or len(day_routes2) > m:
                                    day_routes2 = first_fit_decreasing(d2, custs2)
                            if day_routes2 is None:
                                day_routes2 = []
                            for v_idx, route2 in enumerate(day_routes2):
                                if route2:
                                    routes.append({
                                        "depot": d2,
                                        "period": day2,
                                        "vehicle": v_idx + 1,
                                        "customers": route2
                                    })
                        moved = True
                        update_best()
                        break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                break

    # ------------------------------------------------------------------
    # Local search (2-opt + relocate)
    # ------------------------------------------------------------------
    if is_feasible() and time.time() - start_time < time_limit:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # 2-opt within each route
            for r in routes:
                if time.time() - start_time > time_limit:
                    break
                route = r['customers']
                if len(route) < 3:
                    continue
                dep = r['depot']
                best_cost_r = route_cost(dep, route)
                best_route = route
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        if route_feasible(dep, new_route):
                            new_cost = route_cost(dep, new_route)
                            if new_cost < best_cost_r - 1e-6:
                                best_cost_r = new_cost
                                best_route = new_route
                if best_route != route:
                    r['customers'] = best_route
                    improved = True

            # Relocate between routes in same depot/day
            routes_by_day = {}
            for r in routes:
                key = (r['depot'], r['period'])
                if key not in routes_by_day:
                    routes_by_day[key] = []
                routes_by_day[key].append(r)

            for (dep, day), day_routes in routes_by_day.items():
                if time.time() - start_time > time_limit:
                    break
                for i, r1 in enumerate(day_routes):
                    for j, r2 in enumerate(day_routes):
                        if i == j:
                            continue
                        for c in list(r1['customers']):
                            best_cost_delta = float('inf')
                            best_pos = -1
                            for pos in range(len(r2['customers']) + 1):
                                new_r2 = r2['customers'][:pos] + [c] + r2['customers'][pos:]
                                if route_feasible(dep, new_r2):
                                    new_r1 = [x for x in r1['customers'] if x != c]
                                    cost_old = route_cost(dep, r1['customers']) + route_cost(dep, r2['customers'])
                                    cost_new = route_cost(dep, new_r1) + route_cost(dep, new_r2)
                                    delta = cost_new - cost_old
                                    if delta < best_cost_delta - 1e-6:
                                        best_cost_delta = delta
                                        best_pos = pos
                            if best_pos != -1:
                                new_r2 = r2['customers'][:best_pos] + [c] + r2['customers'][best_pos:]
                                new_r1 = [x for x in r1['customers'] if x != c]
                                r1['customers'] = new_r1
                                r2['customers'] = new_r2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                update_best()

    # ------------------------------------------------------------------
    # Write final solution
    # ------------------------------------------------------------------
    if best_solution is None:
        cost = sum(route_cost(r['depot'], r['customers']) for r in routes)
        best_solution = {
            "objective_value": cost,
            "routes": [dict(r) for r in routes],
            "assignments": {str(c): {"depot": assignments[c][0], "pattern": list(assignments[c][1])}
                            for c in range(n)}
        }
        if logger:
            logger.log_solution(cost, best_solution)

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == '__main__':
    main()