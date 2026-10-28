import argparse
import json
import math
import sys
import time


def build_effective_times(n, dist_matrix, service_times, ready_times, due_dates, includes_service):
    eff_time = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                eff_time[i][j] = float('inf')
                continue
            base = dist_matrix[i][j]
            if i > 0:
                if not includes_service:
                    adjusted = base + service_times[i]
                else:
                    adjusted = base
                diff = ready_times[j] - due_dates[i]
                eff_time[i][j] = max(adjusted, diff)
            else:
                eff_time[i][j] = base
    return eff_time


def compute_feasible_arcs(n, eff_time, ready_times, due_dates, demands, vehicle_capacity):
    feasible = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ready_times[i] + eff_time[i][j] > due_dates[j] + 1e-6:
                continue
            if j == 0:
                if due_dates[i] + eff_time[i][0] > due_dates[0] + 1e-6:
                    continue
            if i > 0 and j > 0:
                if demands[i] + demands[j] > vehicle_capacity:
                    continue
            feasible.add((i, j))
    return feasible


def evaluate_route(route, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity):
    current_time = ready_times[0]
    current_load = 0
    total_dist = 0.0
    dep_times = {}
    loads = {}
    prev = 0
    for cust in route:
        if (prev, cust) not in feasible:
            return None
        current_time = max(current_time + eff_time[prev][cust], ready_times[cust])
        if current_time > due_dates[cust] + 1e-6:
            return None
        current_load += demands[cust]
        if current_load > vehicle_capacity + 1e-6:
            return None
        dep_times[cust] = current_time
        loads[cust] = current_load
        total_dist += dist_matrix[prev][cust]
        prev = cust
    if (prev, 0) not in feasible:
        return None
    ret = current_time + eff_time[prev][0]
    if ret > due_dates[0] + 1e-6:
        return None
    total_dist += dist_matrix[prev][0]
    return (dep_times, loads, total_dist)


def greedy_insertion(n, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity, deadline):
    sorted_custs = sorted(range(1, n), key=lambda c: (due_dates[c], ready_times[c]))
    routes = []
    for c in sorted_custs:
        if time.time() > deadline:
            routes.append([c])
            continue
        best_cost = float('inf')
        best_ridx = -1
        best_pos = -1
        for ridx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                result = evaluate_route(new_route, feasible, eff_time, dist_matrix,
                                        ready_times, due_dates, demands, vehicle_capacity)
                if result is not None and result[2] < best_cost:
                    best_cost = result[2]
                    best_ridx = ridx
                    best_pos = pos
        if best_ridx >= 0:
            routes[best_ridx].insert(best_pos, c)
        else:
            routes.append([c])
    return routes


def merge_routes(routes, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity, deadline):
    improved = True
    while improved and time.time() <= deadline:
        improved = False
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for j in range(len(routes)):
                if i == j or time.time() > deadline:
                    continue
                for opt in range(4):
                    if opt == 0:
                        merged = routes[i] + routes[j]
                    elif opt == 1:
                        merged = routes[i] + routes[j][::-1]
                    elif opt == 2:
                        merged = routes[i][::-1] + routes[j]
                    else:
                        merged = routes[i][::-1] + routes[j][::-1]
                    if evaluate_route(merged, feasible, eff_time, dist_matrix,
                                      ready_times, due_dates, demands, vehicle_capacity) is not None:
                        if j > i:
                            routes[i] = merged
                            routes.pop(j)
                        else:
                            routes[j] = merged
                            routes.pop(i)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes


def two_opt_route(route, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity, deadline):
    improved = True
    while improved and time.time() <= deadline:
        improved = False
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                old_r = evaluate_route(route, feasible, eff_time, dist_matrix,
                                       ready_times, due_dates, demands, vehicle_capacity)
                new_r = evaluate_route(new_route, feasible, eff_time, dist_matrix,
                                       ready_times, due_dates, demands, vehicle_capacity)
                if old_r is not None and new_r is not None and new_r[2] < old_r[2] - 1e-10:
                    route[:] = new_route
                    improved = True
                    break
            if improved:
                break
    return route


def routes_to_solution(routes, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity):
    all_dep = {}
    all_load = {}
    total_dist = 0.0
    full_routes = []
    for route in routes:
        result = evaluate_route(route, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity)
        if result is None:
            return None
        dep, ld, dist = result
        all_dep.update(dep)
        all_load.update(ld)
        total_dist += dist
        full_routes.append([0] + route + [0])
    return {
        "objective_value": float(len(routes)),
        "num_vehicles": len(routes),
        "routes": full_routes,
        "total_distance": total_dist,
        "departure_times": {str(k): v for k, v in all_dep.items()},
        "loads": {str(k): int(v) for k, v in all_load.items()}
    }


def extract_mip_solution(x, feasible, n, dist_matrix, eff_time, ready_times, demands):
    arcs = []
    for (i, j) in feasible:
        try:
            if x[i, j].X > 0.5:
                arcs.append((i, j))
        except Exception:
            pass
    next_node = {}
    for (i, j) in arcs:
        next_node[i] = j
    routes = []
    visited = set()
    for j in range(1, n):
        if (0, j) in feasible:
            try:
                if x[0, j].X > 0.5 and j not in visited:
                    route = []
                    current = j
                    while current != 0 and current not in visited:
                        visited.add(current)
                        route.append(current)
                        current = next_node.get(current, 0)
                    if route:
                        routes.append(route)
            except Exception:
                pass
    if len(visited) != n - 1:
        return None
    all_dep = {}
    all_load = {}
    total_dist = 0.0
    full_routes = []
    for route in routes:
        current_time = ready_times[0]
        current_load = 0
        prev = 0
        for cust in route:
            current_time = max(current_time + eff_time[prev][cust], ready_times[cust])
            current_load += demands[cust]
            all_dep[cust] = current_time
            all_load[cust] = current_load
            total_dist += dist_matrix[prev][cust]
            prev = cust
        total_dist += dist_matrix[prev][0]
        full_routes.append([0] + route + [0])
    return {
        "objective_value": float(len(routes)),
        "num_vehicles": len(routes),
        "routes": full_routes,
        "total_distance": total_dist,
        "departure_times": {str(k): v for k, v in all_dep.items()},
        "loads": {str(k): int(v) for k, v in all_load.items()}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist_matrix = instance['distance_matrix']
    includes_service = instance.get('travel_time_includes_service_time', False)

    n = num_customers + 1

    service_times = [0] * n
    ready_times = [0] * n
    due_dates = [0] * n
    demands = [0] * n

    service_times[0] = depot['service_time']
    ready_times[0] = depot['ready_time']
    due_dates[0] = depot['due_date']
    demands[0] = depot['demand']

    for c in customers:
        cid = c['id']
        service_times[cid] = c['service_time']
        ready_times[cid] = c['ready_time']
        due_dates[cid] = c['due_date']
        demands[cid] = c['demand']

    eff_time = build_effective_times(n, dist_matrix, service_times, ready_times, due_dates, includes_service)
    feasible = compute_feasible_arcs(n, eff_time, ready_times, due_dates, demands, vehicle_capacity)

    for j in range(1, n):
        if (0, j) not in feasible:
            print(f"Warning: Customer {j} unreachable from depot", file=sys.stderr)

    best_solution = None
    heur_routes = None

    heur_deadline = start_time + min(args.time_limit * 0.3, 30)
    heur_routes = greedy_insertion(n, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity, heur_deadline)
    heur_routes = merge_routes(heur_routes, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity, heur_deadline)
    for r in heur_routes:
        two_opt_route(r, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity, heur_deadline)
    heur_routes = merge_routes(heur_routes, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity, heur_deadline)

    sol = routes_to_solution(heur_routes, feasible, eff_time, dist_matrix, ready_times, due_dates, demands, vehicle_capacity)
    if sol is not None:
        best_solution = sol
        if logger:
            logger.log_solution(sol["objective_value"], sol)

    try:
        import gurobipy as gp
        from gurobipy import GRB

        env = gp.Env(empty=True)
        env.setParam('Seed', 0)
        env.setParam('MIPGap', 1e-4)
        env.setParam('NumericFocus', 0)
        env.setParam('Threads', 1)
        env.setParam('OutputFlag', 0)
        env.start()

        model = gp.Model(env=env)

        x = {}
        for (i, j) in feasible:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

        t_vars = [None] * n
        l_vars = [None] * n
        for i in range(1, n):
            t_vars[i] = model.addVar(lb=ready_times[i], ub=due_dates[i], name=f't_{i}')
            l_vars[i] = model.addVar(lb=demands[i], ub=vehicle_capacity, name=f'l_{i}')

        model.update()

        K_expr = gp.quicksum(x[0, j] for j in range(1, n) if (0, j) in feasible)
        dist_expr = gp.quicksum(dist_matrix[i][j] * x[i, j] for (i, j) in feasible)

        for j in range(1, n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(n) if (i, j) in feasible) == 1)
        for i in range(1, n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if (i, j) in feasible) == 1)
        model.addConstr(
            gp.quicksum(x[0, j] for j in range(1, n) if (0, j) in feasible) ==
            gp.quicksum(x[i, 0] for i in range(1, n) if (i, 0) in feasible)
        )

        for (i, j) in feasible:
            if i > 0 and j > 0:
                M_t = max(0.0, due_dates[i] + eff_time[i][j] - ready_times[j])
                model.addConstr(t_vars[j] >= t_vars[i] + eff_time[i][j] - M_t * (1 - x[i, j]))
                model.addConstr(l_vars[j] >= l_vars[i] + demands[j] - float(vehicle_capacity) * (1 - x[i, j]))
            elif i == 0 and j > 0:
                M_t = max(0.0, ready_times[0] + eff_time[0][j] - ready_times[j])
                model.addConstr(t_vars[j] >= ready_times[0] + eff_time[0][j] - M_t * (1 - x[0, j]))
            elif i > 0 and j == 0:
                M_t = max(0.0, due_dates[i] - due_dates[0] + eff_time[i][0])
                model.addConstr(t_vars[i] <= due_dates[0] - eff_time[i][0] + M_t * (1 - x[i, 0]))

        if best_solution:
            for (i, j) in feasible:
                x[i, j].Start = 0
            for route in heur_routes:
                prev = 0
                for cust in route:
                    if (prev, cust) in x:
                        x[prev, cust].Start = 1
                    prev = cust
                if (prev, 0) in x:
                    x[prev, 0].Start = 1
            for i in range(1, n):
                key = str(i)
                if key in best_solution["departure_times"]:
                    t_vars[i].Start = best_solution["departure_times"][key]
                if key in best_solution["loads"]:
                    l_vars[i].Start = best_solution["loads"][key]

        elapsed = time.time() - start_time
        phase1_time = max((args.time_limit - elapsed) * 0.5, 5)
        model.setObjective(K_expr, GRB.MINIMIZE)
        model.setParam('TimeLimit', phase1_time)
        model.optimize()

        best_K = None
        if model.SolCount > 0:
            best_K = int(round(model.ObjVal))
            mip_sol = extract_mip_solution(x, feasible, n, dist_matrix, eff_time, ready_times, demands)
            if mip_sol:
                if best_solution is None or mip_sol["objective_value"] < best_solution["objective_value"] or \
                   (mip_sol["objective_value"] == best_solution["objective_value"] and
                    mip_sol["total_distance"] < best_solution["total_distance"] - 1e-10):
                    best_solution = mip_sol
                    if logger:
                        logger.log_solution(best_solution["objective_value"], best_solution)

        if best_K is not None:
            elapsed = time.time() - start_time
            remaining = max(args.time_limit - elapsed, 1)
            model.addConstr(K_expr <= best_K)
            model.setObjective(dist_expr, GRB.MINIMIZE)
            model.setParam('TimeLimit', remaining)
            model.optimize()

            if model.SolCount > 0:
                mip_sol = extract_mip_solution(x, feasible, n, dist_matrix, eff_time, ready_times, demands)
                if mip_sol:
                    if best_solution is None or mip_sol["objective_value"] < best_solution["objective_value"] or \
                       (mip_sol["objective_value"] == best_solution["objective_value"] and
                        mip_sol["total_distance"] < best_solution["total_distance"] - 1e-10):
                        best_solution = mip_sol
                        if logger:
                            logger.log_solution(best_solution["objective_value"], best_solution)

        model.dispose()
        env.dispose()
    except Exception as e:
        print(f"MIP error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    if best_solution is None:
        best_solution = {
            "objective_value": 0.0,
            "num_vehicles": 0,
            "routes": [],
            "total_distance": 0.0,
            "departure_times": {},
            "loads": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == '__main__':
    main()