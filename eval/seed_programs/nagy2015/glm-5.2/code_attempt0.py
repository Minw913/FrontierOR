import json
import time
import argparse
import random
from solution_logger import SolutionLogger

def build_distance_matrix(n, distance_matrix):
    # Nodes: 0 (depot), 1..n (linehaul), n+1..2n (backhaul)
    size = 2 * n + 1
    dist = [[0] * size for _ in range(size)]
    for i in range(1, n + 1):
        d0i = distance_matrix[0][i]
        dist[0][i] = d0i
        dist[0][n + i] = d0i
        dist[i][0] = d0i
        dist[n + i][0] = d0i
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            dij = distance_matrix[i][j]
            dist[i][j] = dij
            dist[i][n + j] = dij
            dist[n + i][j] = dij
            dist[n + i][n + j] = dij
    for i in range(1, n + 1):
        dist[n + i][n + i] = 0
    return dist

def evaluate_route(route, delivery, pickup, dist, C, n):
    """Return (feasible, distance) for a route (list of node IDs, no depot)."""
    if not route:
        return True, 0
    D = 0
    for node in route:
        if node <= n:
            D += delivery[node]
    if D > C:
        return False, 0
    load = D
    d = dist[0][route[0]]
    for i in range(len(route)):
        node = route[i]
        if node <= n:
            load -= delivery[node]
        else:
            load += pickup[node - n]
        if load < 0 or load > C:
            return False, 0
        if i < len(route) - 1:
            d += dist[node][route[i + 1]]
        else:
            d += dist[node][0]
    return True, d

def route_dist(route, dist):
    if not route:
        return 0
    d = dist[0][route[0]]
    for i in range(len(route) - 1):
        d += dist[route[i]][route[i + 1]]
    d += dist[route[-1]][0]
    return d

def greedy_insertion(delivery, pickup, dist, C, n, order='distance_desc'):
    requests = []
    for i in range(1, n + 1):
        requests.append(i)
        requests.append(n + i)
    if order == 'distance_desc':
        requests.sort(key=lambda x: -dist[0][x])
    elif order == 'demand_desc':
        requests.sort(key=lambda x: -(delivery[x] if x <= n else pickup[x - n]))
    elif order == 'random':
        random.shuffle(requests)
    else:
        random.shuffle(requests)
    routes = []
    for req in requests:
        best_delta = float('inf')
        best_r = -1
        best_pos = -1
        for r_idx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [req] + route[pos:]
                feas, d = evaluate_route(new_route, delivery, pickup, dist, C, n)
                if not feas:
                    continue
                delta = d - route_dist(route, dist)
                if delta < best_delta:
                    best_delta = delta
                    best_r = r_idx
                    best_pos = pos
        if best_r != -1:
            routes[best_r].insert(best_pos, req)
        else:
            routes.append([req])
    return routes

def two_opt_intra(routes, route_dists, delivery, pickup, dist, C, n):
    for r_idx in range(len(routes)):
        route = routes[r_idx]
        L = len(route)
        if L < 2:
            continue
        for i in range(L):
            for j in range(i + 1, L):
                prev_i = route[i - 1] if i > 0 else 0
                next_j = route[j + 1] if j < L - 1 else 0
                old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                delta = new - old
                if delta < -1e-6:
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    feas, d = evaluate_route(new_route, delivery, pickup, dist, C, n)
                    if feas and d < route_dists[r_idx] - 1e-6:
                        routes[r_idx] = new_route
                        route_dists[r_idx] = d
                        return True
    return False

def or_opt_intra(routes, route_dists, delivery, pickup, dist, C, n):
    for r_idx in range(len(routes)):
        route = routes[r_idx]
        L = len(route)
        for seg_len in [1, 2, 3]:
            if seg_len > L:
                continue
            for i in range(L - seg_len + 1):
                seg = route[i:i + seg_len]
                rest = route[:i] + route[i + seg_len:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    new_route = rest[:j] + seg + rest[j:]
                    feas, d = evaluate_route(new_route, delivery, pickup, dist, C, n)
                    if feas and d < route_dists[r_idx] - 1e-6:
                        routes[r_idx] = new_route
                        route_dists[r_idx] = d
                        return True
    return False

def relocate(routes, route_dists, delivery, pickup, dist, C, n):
    for a_idx in range(len(routes)):
        for i in range(len(routes[a_idx])):
            node = routes[a_idx][i]
            new_a = routes[a_idx][:i] + routes[a_idx][i + 1:]
            feas_a, dist_a = evaluate_route(new_a, delivery, pickup, dist, C, n)
            if not feas_a:
                continue
            for b_idx in range(len(routes)):
                if b_idx == a_idx:
                    continue
                for pos in range(len(routes[b_idx]) + 1):
                    new_b = routes[b_idx][:pos] + [node] + routes[b_idx][pos:]
                    feas_b, dist_b = evaluate_route(new_b, delivery, pickup, dist, C, n)
                    if not feas_b:
                        continue
                    delta = (dist_a + dist_b) - (route_dists[a_idx] + route_dists[b_idx])
                    if delta < -1e-6:
                        routes[a_idx] = new_a
                        routes[b_idx] = new_b
                        route_dists[a_idx] = dist_a
                        route_dists[b_idx] = dist_b
                        return True
    return False

def swap(routes, route_dists, delivery, pickup, dist, C, n):
    for a_idx in range(len(routes)):
        for i in range(len(routes[a_idx])):
            node_a = routes[a_idx][i]
            for b_idx in range(a_idx + 1, len(routes)):
                for j in range(len(routes[b_idx])):
                    node_b = routes[b_idx][j]
                    new_a = routes[a_idx][:i] + [node_b] + routes[a_idx][i + 1:]
                    new_b = routes[b_idx][:j] + [node_a] + routes[b_idx][j + 1:]
                    feas_a, dist_a = evaluate_route(new_a, delivery, pickup, dist, C, n)
                    if not feas_a:
                        continue
                    feas_b, dist_b = evaluate_route(new_b, delivery, pickup, dist, C, n)
                    if not feas_b:
                        continue
                    delta = (dist_a + dist_b) - (route_dists[a_idx] + route_dists[b_idx])
                    if delta < -1e-6:
                        routes[a_idx] = new_a
                        routes[b_idx] = new_b
                        route_dists[a_idx] = dist_a
                        route_dists[b_idx] = dist_b
                        return True
    return False

def two_opt_star(routes, route_dists, delivery, pickup, dist, C, n):
    for a_idx in range(len(routes)):
        for i in range(len(routes[a_idx]) + 1):
            for b_idx in range(a_idx + 1, len(routes)):
                for j in range(len(routes[b_idx]) + 1):
                    new_a = routes[a_idx][:i] + routes[b_idx][j:]
                    new_b = routes[b_idx][:j] + routes[a_idx][i:]
                    feas_a, dist_a = evaluate_route(new_a, delivery, pickup, dist, C, n)
                    if not feas_a:
                        continue
                    feas_b, dist_b = evaluate_route(new_b, delivery, pickup, dist, C, n)
                    if not feas_b:
                        continue
                    delta = (dist_a + dist_b) - (route_dists[a_idx] + route_dists[b_idx])
                    if delta < -1e-6:
                        routes[a_idx] = new_a
                        routes[b_idx] = new_b
                        route_dists[a_idx] = dist_a
                        route_dists[b_idx] = dist_b
                        return True
    return False

def perturb(routes, route_dists, delivery, pickup, dist, C, n):
    if len(routes) < 2:
        if routes:
            r = routes[0]
            if len(r) >= 2:
                i = random.randint(0, len(r) - 1)
                j = random.randint(i + 1, len(r) - 1)
                new_r = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                feas, d = evaluate_route(new_r, delivery, pickup, dist, C, n)
                if feas:
                    routes[0] = new_r
                    route_dists[0] = d
        return
    a_idx = random.randint(0, len(routes) - 1)
    b_idx = random.randint(0, len(routes) - 1)
    if a_idx == b_idx:
        return
    i = random.randint(0, len(routes[a_idx]))
    j = random.randint(0, len(routes[b_idx]))
    new_a = routes[a_idx][:i] + routes[b_idx][j:]
    new_b = routes[b_idx][:j] + routes[a_idx][i:]
    feas_a, dist_a = evaluate_route(new_a, delivery, pickup, dist, C, n)
    feas_b, dist_b = evaluate_route(new_b, delivery, pickup, dist, C, n)
    if feas_a and feas_b:
        routes[a_idx] = new_a
        routes[b_idx] = new_b
        route_dists[a_idx] = dist_a
        route_dists[b_idx] = dist_b

def build_solution_dict(routes, delivery, pickup, n, dist):
    sol_routes = []
    sol_detailed = []
    for route in routes:
        full_route = [0] + route + [0]
        sol_routes.append(full_route)
        detailed = []
        for node in full_route:
            if node == 0:
                detailed.append({
                    "node_id": 0,
                    "role": "depot",
                    "customer_id": 0,
                    "quantity": 0
                })
            elif node <= n:
                detailed.append({
                    "node_id": node,
                    "role": "linehaul",
                    "customer_id": node,
                    "quantity": delivery[node]
                })
            else:
                cust_id = node - n
                detailed.append({
                    "node_id": node,
                    "role": "backhaul",
                    "customer_id": cust_id,
                    "quantity": pickup[cust_id]
                })
        sol_detailed.append(detailed)
    total_dist = sum(route_dist(r, dist) for r in routes)
    return {
        "objective_value": total_dist,
        "routes": sol_routes,
        "routes_detailed": sol_detailed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    random.seed(0)

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_customers']
    C = instance['vehicle_capacity']
    customers = instance['customers']
    distance_matrix = instance['distance_matrix']

    delivery = [0] * (n + 1)
    pickup = [0] * (n + 1)
    for cust in customers:
        cid = cust['id']
        delivery[cid] = cust['delivery_demand']
        pickup[cid] = cust['pickup_demand']

    dist = build_distance_matrix(n, distance_matrix)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Generate initial solution
    best_routes = None
    best_init_dist = float('inf')
    for order in ['distance_desc', 'demand_desc', 'random']:
        routes = greedy_insertion(delivery, pickup, dist, C, n, order)
        total = sum(route_dist(r, dist) for r in routes)
        if total < best_init_dist:
            best_init_dist = total
            best_routes = routes
    routes = best_routes
    route_dists = [route_dist(r, dist) for r in routes]
    best_dist = best_init_dist

    if logger:
        logger.log_solution(best_dist, build_solution_dict(routes, delivery, pickup, n, dist))

    start_time = time.time()
    while time.time() - start_time < args.time_limit - 1:
        improved = False
        if time.time() - start_time >= args.time_limit - 1:
            break
        improved = two_opt_intra(routes, route_dists, delivery, pickup, dist, C, n)
        if improved:
            total = sum(route_dists)
            if total < best_dist - 1e-6:
                best_dist = total
                best_routes = [r[:] for r in routes]
                if logger:
                    logger.log_solution(best_dist, build_solution_dict(best_routes, delivery, pickup, n, dist))
            continue
        if time.time() - start_time >= args.time_limit - 1:
            break
        improved = or_opt_intra(routes, route_dists, delivery, pickup, dist, C, n)
        if improved:
            total = sum(route_dists)
            if total < best_dist - 1e-6:
                best_dist = total
                best_routes = [r[:] for r in routes]
                if logger:
                    logger.log_solution(best_dist, build_solution_dict(best_routes, delivery, pickup, n, dist))
            continue
        if time.time() - start_time >= args.time_limit - 1:
            break
        improved = relocate(routes, route_dists, delivery, pickup, dist, C, n)
        if improved:
            total = sum(route_dists)
            if total < best_dist - 1e-6:
                best_dist = total
                best_routes = [r[:] for r in routes]
                if logger:
                    logger.log_solution(best_dist, build_solution_dict(best_routes, delivery, pickup, n, dist))
            continue
        if time.time() - start_time >= args.time_limit - 1:
            break
        improved = swap(routes, route_dists, delivery, pickup, dist, C, n)
        if improved:
            total = sum(route_dists)
            if total < best_dist - 1e-6:
                best_dist = total
                best_routes = [r[:] for r in routes]
                if logger:
                    logger.log_solution(best_dist, build_solution_dict(best_routes, delivery, pickup, n, dist))
            continue
        if time.time() - start_time >= args.time_limit - 1:
            break
        improved = two_opt_star(routes, route_dists, delivery, pickup, dist, C, n)
        if improved:
            total = sum(route_dists)
            if total < best_dist - 1e-6:
                best_dist = total
                best_routes = [r[:] for r in routes]
                if logger:
                    logger.log_solution(best_dist, build_solution_dict(best_routes, delivery, pickup, n, dist))
            continue

        # No improvement: perturb
        perturb(routes, route_dists, delivery, pickup, dist, C, n)
        # Clean up empty routes
        routes[:] = [r for r in routes if r]
        route_dists[:] = [route_dist(r, dist) for r in routes]

    # Final solution from best found
    final_solution = build_solution_dict(best_routes, delivery, pickup, n, dist)
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == '__main__':
    main()