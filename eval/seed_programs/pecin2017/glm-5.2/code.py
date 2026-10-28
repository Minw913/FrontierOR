import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    random.seed(0)

    # ---------- Read instance ----------
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    params = data['problem_parameters']
    nodes = data['nodes']
    N = len(nodes)

    depot_source = None
    depot_sink = None
    customers = []
    for node in nodes:
        if node['type'] == 'depot_source':
            depot_source = node
        elif node['type'] == 'depot_sink':
            depot_sink = node
        else:
            customers.append(node)

    id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
    idx_to_id = [node['id'] for node in nodes]
    ds_idx = id_to_idx[depot_source['id']]
    dt_idx = id_to_idx[depot_sink['id']]
    customer_indices = [id_to_idx[c['id']] for c in customers]
    n_customers = len(customer_indices)

    demand = [0] * N
    service_time = [0] * N
    tw_open = [0.0] * N
    tw_close = [0.0] * N
    for i, node in enumerate(nodes):
        demand[i] = node['demand']
        service_time[i] = node['service_time']
        tw_open[i] = float(node['time_window_open'])
        tw_close[i] = float(node['time_window_close'])

    Q = int(params['vehicle_capacity'])
    H = float(params['planning_horizon'])

    # ---------- Distance matrix ----------
    coords = [(node['x'], node['y']) for node in nodes]
    dist = [[0.0] * N for _ in range(N)]
    for i in range(N):
        xi, yi = coords[i]
        for j in range(N):
            xj, yj = coords[j]
            dist[i][j] = math.hypot(xi - xj, yi - yj)

    # ---------- Helper functions ----------
    EPS = 1e-9

    def compute_depart(route):
        """Return departure time for each position in route, or None if infeasible."""
        depart = [0.0] * len(route)
        time = 0.0
        prev = route[0]
        for i in range(1, len(route)):
            curr = route[i]
            travel = dist[prev][curr]
            arrival = time + travel
            if arrival > tw_close[curr] + EPS:
                return None
            start = max(arrival, tw_open[curr])
            time = start + service_time[curr]
            depart[i] = time
            prev = curr
        return depart

    def is_feasible(route):
        if sum(demand[node] for node in route) > Q:
            return False
        return compute_depart(route) is not None

    def route_cost(route):
        c = 0.0
        for i in range(len(route) - 1):
            c += dist[route[i]][route[i+1]]
        return c

    def solution_cost(routes):
        return sum(route_cost(r) for r in routes)

    def feasible_insertion(route, c, pos, depart=None, route_load=None):
        if route_load is None:
            route_load = sum(demand[node] for node in route)
        if route_load + demand[c] > Q:
            return False
        if depart is None:
            depart = compute_depart(route)
            if depart is None:
                return False
        time = depart[pos]
        # Insert c
        travel = dist[route[pos]][c]
        arrival = time + travel
        if arrival > tw_close[c] + EPS:
            return False
        start = max(arrival, tw_open[c])
        time = start + service_time[c]
        prev = c
        # Check remaining
        for i in range(pos + 1, len(route)):
            curr = route[i]
            travel = dist[prev][curr]
            arrival = time + travel
            if arrival > tw_close[curr] + EPS:
                return False
            start = max(arrival, tw_open[curr])
            time = start + service_time[curr]
            prev = curr
        return True

    def insert_customer(route, c, pos):
        return route[:pos+1] + [c] + route[pos+1:]

    def best_insertion(route, c):
        route_load = sum(demand[node] for node in route)
        if route_load + demand[c] > Q:
            return None, None
        depart = compute_depart(route)
        if depart is None:
            return None, None
        best_cost = float('inf')
        best_pos = -1
        for pos in range(len(route) - 1):
            inc = dist[route[pos]][c] + dist[c][route[pos+1]] - dist[route[pos]][route[pos+1]]
            if inc < best_cost:
                if feasible_insertion(route, c, pos, depart, route_load):
                    best_cost = inc
                    best_pos = pos
        return best_pos, best_cost

    # ---------- Initial solution (sequential insertion) ----------
    def initial_solution():
        sorted_custs = sorted(customer_indices, key=lambda x: tw_open[x])
        routes = []
        for c in sorted_custs:
            best_idx = -1
            best_pos = -1
            best_inc = float('inf')
            for r_idx, route in enumerate(routes):
                pos, inc = best_insertion(route, c)
                if pos != -1 and inc < best_inc:
                    best_inc = inc
                    best_pos = pos
                    best_idx = r_idx
            if best_idx != -1:
                routes[best_idx] = insert_customer(routes[best_idx], c, best_pos)
            else:
                new_route = [ds_idx, c, dt_idx]
                if is_feasible(new_route):
                    routes.append(new_route)
                else:
                    # Fallback: try any feasible insertion
                    inserted = False
                    for r_idx, route in enumerate(routes):
                        depart = compute_depart(route)
                        if depart is None:
                            continue
                        route_load = sum(demand[node] for node in route)
                        for pos in range(len(route) - 1):
                            if feasible_insertion(route, c, pos, depart, route_load):
                                routes[r_idx] = insert_customer(route, c, pos)
                                inserted = True
                                break
                        if inserted:
                            break
                    if not inserted:
                        routes.append(new_route)  # last resort
        return routes

    # ---------- ALNS operators ----------
    def destroy_random(routes, q):
        all_custs = [node for r in routes for node in r if node != ds_idx and node != dt_idx]
        if not all_custs:
            return routes, []
        q = min(q, len(all_custs))
        to_remove = random.sample(all_custs, q)
        new_routes = []
        for r in routes:
            nr = [node for node in r if node not in to_remove]
            if len(nr) > 2:
                new_routes.append(nr)
        return new_routes, to_remove

    def destroy_shaw(routes, q):
        all_custs = [node for r in routes for node in r if node != ds_idx and node != dt_idx]
        if not all_custs:
            return routes, []
        seed = random.choice(all_custs)
        related = sorted(all_custs,
                         key=lambda x: dist[seed][x] + abs(tw_open[seed]-tw_open[x]) + abs(tw_close[seed]-tw_close[x]))
        q = min(q, len(related))
        to_remove = related[:q]
        new_routes = []
        for r in routes:
            nr = [node for node in r if node not in to_remove]
            if len(nr) > 2:
                new_routes.append(nr)
        return new_routes, to_remove

    def destroy_worst(routes, q):
        all_custs = []
        for r in routes:
            for i in range(1, len(r)-1):
                c = r[i]
                prev = r[i-1]
                nxt = r[i+1]
                gain = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                all_custs.append((gain, c))
        if not all_custs:
            return routes, []
        all_custs.sort(key=lambda x: x[0], reverse=True)
        q = min(q, len(all_custs))
        to_remove = [c for _, c in all_custs[:q]]
        new_routes = []
        for r in routes:
            nr = [node for node in r if node not in to_remove]
            if len(nr) > 2:
                new_routes.append(nr)
        return new_routes, to_remove

    def repair_greedy(routes, unassigned):
        while unassigned:
            best_cust = -1
            best_ridx = -1
            best_pos = -1
            best_inc = float('inf')
            for c in unassigned:
                for r_idx, route in enumerate(routes):
                    pos, inc = best_insertion(route, c)
                    if pos != -1 and inc < best_inc:
                        best_inc = inc
                        best_pos = pos
                        best_ridx = r_idx
                        best_cust = c
            if best_cust != -1:
                routes[best_ridx] = insert_customer(routes[best_ridx], best_cust, best_pos)
                unassigned.remove(best_cust)
            else:
                c = unassigned.pop(0)
                new_route = [ds_idx, c, dt_idx]
                if is_feasible(new_route):
                    routes.append(new_route)
                else:
                    # try any insertion
                    inserted = False
                    for r_idx, route in enumerate(routes):
                        depart = compute_depart(route)
                        if depart is None:
                            continue
                        rl = sum(demand[node] for node in route)
                        for pos in range(len(route)-1):
                            if feasible_insertion(route, c, pos, depart, rl):
                                routes[r_idx] = insert_customer(route, c, pos)
                                inserted = True
                                break
                        if inserted:
                            break
                    if not inserted:
                        routes.append(new_route)
        return routes

    def repair_regret(routes, unassigned):
        while unassigned:
            best_regret = -1.0
            best_cust = -1
            best_ridx = -1
            best_pos = -1
            for c in unassigned:
                costs = []
                for r_idx, route in enumerate(routes):
                    pos, inc = best_insertion(route, c)
                    if pos != -1:
                        costs.append((inc, r_idx, pos))
                if not costs:
                    continue
                costs.sort(key=lambda x: x[0])
                regret = costs[1][0] - costs[0][0] if len(costs) >= 2 else float('inf')
                if regret > best_regret:
                    best_regret = regret
                    best_cust = c
                    best_ridx = costs[0][1]
                    best_pos = costs[0][2]
            if best_cust != -1:
                routes[best_ridx] = insert_customer(routes[best_ridx], best_cust, best_pos)
                unassigned.remove(best_cust)
            else:
                c = unassigned.pop(0)
                new_route = [ds_idx, c, dt_idx]
                if is_feasible(new_route):
                    routes.append(new_route)
                else:
                    inserted = False
                    for r_idx, route in enumerate(routes):
                        depart = compute_depart(route)
                        if depart is None:
                            continue
                        rl = sum(demand[node] for node in route)
                        for pos in range(len(route)-1):
                            if feasible_insertion(route, c, pos, depart, rl):
                                routes[r_idx] = insert_customer(route, c, pos)
                                inserted = True
                                break
                        if inserted:
                            break
                    if not inserted:
                        routes.append(new_route)
        return routes

    # ---------- Local search polish ----------
    def two_opt(route):
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                delta = (dist[route[i-1]][route[j]] + dist[route[i]][route[j+1]]
                         - dist[route[i-1]][route[i]] - dist[route[j]][route[j+1]])
                if delta < -EPS:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if is_feasible(new_route):
                        route[:] = new_route
                        improved = True
        return improved

    def or_opt(route):
        improved = False
        for i in range(1, len(route)-1):
            c = route[i]
            base = route[:i] + route[i+1:]
            for j in range(1, len(base)-1):
                if j == i:
                    continue
                new_route = base[:j] + [c] + base[j:]
                if route_cost(new_route) < route_cost(route) - EPS:
                    if is_feasible(new_route):
                        route[:] = new_route
                        improved = True
                        return improved
        return improved

    # ---------- Main loop ----------
    start_time = time.time()
    deadline = start_time + args.time_limit
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    routes = initial_solution()
    best_routes = [r[:] for r in routes]
    best_cost = solution_cost(best_routes)
    current_routes = [r[:] for r in routes]
    current_cost = best_cost

    if logger:
        sol_dict = {
            "objective_value": best_cost,
            "routes": [[idx_to_id[node] for node in r] for r in best_routes]
        }
        logger.log_solution(best_cost, sol_dict)

    max_destroy = max(2, min(n_customers // 2, 15)) if n_customers > 0 else 1
    T = 0.1 * best_cost if best_cost > 0 else 1.0
    alpha = 0.99

    destroy_ops = [destroy_random, destroy_shaw, destroy_worst]
    repair_ops = [repair_greedy, repair_regret]

    n_iter = 0
    while time.time() < deadline and n_iter < 1000000:
        n_iter += 1
        destroy = random.choice(destroy_ops)
        repair = random.choice(repair_ops)
        q = random.randint(1, max_destroy)
        destroyed, removed = destroy(current_routes, q)
        new_routes = repair(destroyed, removed[:])
        new_cost = solution_cost(new_routes)

        if new_cost < current_cost - EPS:
            current_routes = new_routes
            current_cost = new_cost
            if new_cost < best_cost - EPS:
                best_routes = [r[:] for r in new_routes]
                best_cost = new_cost
                if logger:
                    sol_dict = {
                        "objective_value": best_cost,
                        "routes": [[idx_to_id[node] for node in r] for r in best_routes]
                    }
                    logger.log_solution(best_cost, sol_dict)
        else:
            delta = new_cost - current_cost
            if T > 1e-10:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if random.random() < prob:
                current_routes = new_routes
                current_cost = new_cost

        T = max(T * alpha, 1e-6)

    # ---------- Local search on best solution if time remains ----------
    while time.time() < deadline:
        improved = False
        for r in best_routes:
            if two_opt(r):
                improved = True
            if or_opt(r):
                improved = True
        if not improved:
            break
        best_cost = solution_cost(best_routes)
        if logger:
            sol_dict = {
                "objective_value": best_cost,
                "routes": [[idx_to_id[node] for node in r] for r in best_routes]
            }
            logger.log_solution(best_cost, sol_dict)

    # ---------- Verify all customers covered ----------
    all_set = set(customer_indices)
    covered = set()
    for r in best_routes:
        for node in r:
            if node in all_set:
                covered.add(node)
    if covered != all_set:
        missing = list(all_set - covered)
        for c in missing:
            inserted = False
            for r_idx, r in enumerate(best_routes):
                pos, inc = best_insertion(r, c)
                if pos != -1:
                    best_routes[r_idx] = insert_customer(r, c, pos)
                    inserted = True
                    break
            if not inserted:
                best_routes.append([ds_idx, c, dt_idx])
        best_cost = solution_cost(best_routes)
        if logger:
            sol_dict = {
                "objective_value": best_cost,
                "routes": [[idx_to_id[node] for node in r] for r in best_routes]
            }
            logger.log_solution(best_cost, sol_dict)

    # ---------- Write final solution ----------
    output = {
        "objective_value": best_cost,
        "routes": [[idx_to_id[node] for node in r] for r in best_routes]
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    main()