import argparse
import json
import math
import random
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    num_requests = instance['num_requests']
    num_nodes = instance['num_nodes']
    num_vehicles = instance['num_vehicles']
    capacity = instance['vehicle_capacity']

    nodes = instance['nodes']
    node_by_id = {n['id']: n for n in nodes}

    origin_id = None
    dest_id = None
    for n in nodes:
        if n['type'] == 'depot_origin':
            origin_id = n['id']
        elif n['type'] == 'depot_destination':
            dest_id = n['id']

    coords = {n['id']: (n['x'], n['y']) for n in nodes}
    demand = {n['id']: n['demand'] for n in nodes}
    service_time = {n['id']: n['service_time'] for n in nodes}
    tw = {n['id']: n['time_window'] for n in nodes}

    def dist(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        return math.hypot(xi - xj, yi - yj)

    D = {}
    for n1 in nodes:
        D[n1['id']] = {}
        for n2 in nodes:
            D[n1['id']][n2['id']] = dist(n1['id'], n2['id'])

    requests = instance['requests']
    req_list = []
    for req in requests:
        req_list.append({
            'id': req['request_id'],
            'pickup': req['pickup_node'],
            'delivery': req['delivery_node'],
            'load': req['load'],
            'pickup_tw': req['pickup_time_window'],
            'delivery_tw': req['delivery_time_window']
        })

    def check_route(route):
        # route: list of node ids, starting at origin, ending at dest
        if not route:
            return False, None
        time = max(0.0, tw[route[0]][0])
        load = 0
        prev = route[0]
        cost = 0.0
        for idx in range(1, len(route)):
            node = route[idx]
            d = D[prev][node]
            cost += d
            arrival = time + d
            e, l = tw[node]
            if node == dest_id:
                if arrival > l + 1e-6:
                    return False, None
                # destination has no service, no load change
                break
            else:
                if arrival > l + 1e-6:
                    return False, None
                start = max(arrival, e)
                if start > l + 1e-6:
                    return False, None
                time = start + service_time[node]
                load += demand[node]
                if load < -1e-6 or load > capacity + 1e-6:
                    return False, None
                prev = node
        return True, cost

    def route_cost(route):
        return sum(D[route[i]][route[i+1]] for i in range(len(route)-1))

    def try_insert(route, pickup_node, delivery_node):
        best_cost = float('inf')
        best_route = None
        for i in range(1, len(route)):
            for j in range(i, len(route)):
                new_route = route[:i] + [pickup_node] + route[i:j] + [delivery_node] + route[j:]
                feasible, cost = check_route(new_route)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_route = new_route
        return best_route, best_cost

    # Regret-based insertion heuristic for initial solution
    unserved = list(req_list)
    routes = []

    while unserved:
        best_req = None
        best_insertion = None
        best_route_idx = -1
        best_regret = -1
        best_cost = float('inf')

        for req in unserved:
            p = req['pickup']
            d = req['delivery']
            costs = []
            for idx, route in enumerate(routes):
                new_route, cost = try_insert(route, p, d)
                if new_route is not None:
                    costs.append((cost, idx, new_route))
            # Consider opening a new route
            if len(routes) < num_vehicles:
                new_route = [origin_id, p, d, dest_id]
                feasible, cost = check_route(new_route)
                if feasible:
                    costs.append((cost, len(routes), new_route))
            if not costs:
                continue
            costs.sort(key=lambda x: x[0])
            best = costs[0][0]
            second = costs[1][0] if len(costs) > 1 else best + 10000
            regret = second - best
            if regret > best_regret:
                best_regret = regret
                best_req = req
                best_insertion = costs[0][2]
                best_route_idx = costs[0][1]
                best_cost = best

        if best_req is None:
            # Fallback: insert greedily
            for req in unserved:
                p = req['pickup']
                d = req['delivery']
                inserted = False
                for idx, route in enumerate(routes):
                    new_route, cost = try_insert(route, p, d)
                    if new_route is not None:
                        routes[idx] = new_route
                        inserted = True
                        break
                if not inserted and len(routes) < num_vehicles:
                    new_route = [origin_id, p, d, dest_id]
                    feasible, _ = check_route(new_route)
                    if feasible:
                        routes.append(new_route)
                        inserted = True
                if not inserted:
                    raise RuntimeError(f"Cannot insert request {req['id']}")
            break
        else:
            if best_route_idx == len(routes):
                routes.append(best_insertion)
            else:
                routes[best_route_idx] = best_insertion
            unserved.remove(best_req)

    # Fill empty routes
    while len(routes) < num_vehicles:
        routes.append([origin_id, dest_id])

    def make_solution_dict(routes_list):
        sol_routes = []
        for v, r in enumerate(routes_list):
            sol_routes.append({
                "vehicle": v,
                "nodes": r
            })
        return {
            "objective_value": sum(route_cost(r) for r in routes_list),
            "routes": sol_routes
        }

    curr_routes = [list(r) for r in routes]
    curr_cost = sum(route_cost(r) for r in curr_routes)
    best_routes = curr_routes
    best_cost = curr_cost
    best_sol_dict = make_solution_dict(best_routes)

    if logger:
        logger.log_solution(best_cost, best_sol_dict)

    # LNS improvement
    start_time = time.time()
    time_limit = args.time_limit
    random.seed(0)

    req_by_id = {r['id']: r for r in req_list}

    def remove_request(route, req_id):
        req = req_by_id[req_id]
        p = req['pickup']
        d = req['delivery']
        return [node for node in route if node != p and node != d]

    def insert_request(route, req_id):
        req = req_by_id[req_id]
        p = req['pickup']
        d = req['delivery']
        best_route = None
        best_cost = float('inf')
        for i in range(1, len(route)):
            for j in range(i, len(route)):
                new_route = route[:i] + [p] + route[i:j] + [d] + route[j:]
                feasible, cost = check_route(new_route)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_route = new_route
        return best_route, best_cost

    while time.time() - start_time < time_limit:
        q = random.randint(1, min(5, num_requests))
        all_req_ids = [r['id'] for r in req_list]
        removed = random.sample(all_req_ids, q)

        new_routes = []
        for route in curr_routes:
            new_route = route[:]
            for rid in removed:
                new_route = remove_request(new_route, rid)
            new_routes.append(new_route)

        random.shuffle(removed)
        feasible = True
        for rid in removed:
            best_idx = -1
            best_ins = None
            best_c = float('inf')
            for idx, route in enumerate(new_routes):
                ins_route, c = insert_request(route, rid)
                if ins_route is not None and c < best_c:
                    best_c = c
                    best_ins = ins_route
                    best_idx = idx
            if best_idx == -1:
                feasible = False
                break
            new_routes[best_idx] = best_ins

        if not feasible:
            continue

        new_total = sum(route_cost(r) for r in new_routes)
        if new_total < curr_cost - 1e-6:
            curr_routes = new_routes
            curr_cost = new_total
            if curr_cost < best_cost - 1e-6:
                best_routes = curr_routes
                best_cost = curr_cost
                best_sol_dict = make_solution_dict(best_routes)
                if logger:
                    logger.log_solution(best_cost, best_sol_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol_dict, f, indent=2)

if __name__ == '__main__':
    main()