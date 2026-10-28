import json
import math
import argparse
import random
import time

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def get_dist(n1, n2):
    return math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])

def check_capacity(route, nodes, capacity):
    load = 0
    for v in route:
        load += nodes[v]['load']
        if load < 0 or load > capacity:
            return False
    return True

def quick_feasibility_check(route, nodes, max_ride_time, max_route_duration, dist_matrix):
    n = len(route)
    if n == 0: return True
    
    # 1. Forward pass for time windows (Earliest possible arrival)
    E = [0.0] * n
    E[0] = nodes[route[0]]['earliest_time']
    for i in range(1, n):
        u = route[i-1]
        v = route[i]
        E[i] = max(nodes[v]['earliest_time'], E[i-1] + nodes[u]['service_duration'] + dist_matrix[u][v])
        if E[i] > nodes[v]['latest_time'] + 1e-5:
            return False
            
    # 2. Backward pass for time windows (Latest possible departure)
    L = [0.0] * n
    L[-1] = nodes[route[-1]]['latest_time']
    for i in range(n-2, -1, -1):
        u = route[i]
        v = route[i+1]
        L[i] = min(nodes[u]['latest_time'], L[i+1] - nodes[u]['service_duration'] - dist_matrix[u][v])
        if L[i] < E[i] - 1e-5:
            return False
            
    # Minimum route duration check
    if E[-1] - L[0] > max_route_duration + 1e-5:
        return False
        
    # 3. Ride time lower bound check
    for i in range(n):
        u = route[i]
        if nodes[u]['node_type'] == 'pickup':
            p = u
            d = nodes[u]['paired_node']
            ride_lb = 0.0
            d_idx = -1
            for j in range(i, n-1):
                x = route[j]
                y = route[j+1]
                ride_lb += (0 if x == p else nodes[x]['service_duration']) + dist_matrix[x][y]
                if y == d:
                    d_idx = j+1
                    break
            if d_idx != -1 and ride_lb > max_ride_time + 1e-5:
                return False
                
    return True

def check_schedule(route, nodes, max_ride_time, max_route_duration, dist_matrix):
    V = route + ['zero']
    edges = []
    
    for k in range(len(route) - 1):
        u = route[k]
        v = route[k+1]
        weight = nodes[u]['service_duration'] + dist_matrix[u][v]
        edges.append((u, v, weight))
        
    for v in route:
        edges.append(('zero', v, nodes[v]['earliest_time']))
        if nodes[v]['latest_time'] < 1e8:
            edges.append((v, 'zero', -nodes[v]['latest_time']))
            
    for v in route:
        if nodes[v]['node_type'] == 'dropoff':
            p = nodes[v]['paired_node']
            if p in route:
                weight = -nodes[p]['service_duration'] - max_ride_time
                edges.append((v, p, weight))
                
    if len(route) > 0:
        first = route[0]
        last = route[-1]
        edges.append((last, first, -max_route_duration))
        
    B = {v: -math.inf for v in V}
    B['zero'] = 0.0
    
    num_nodes_V = len(V)
    for _ in range(num_nodes_V - 1):
        updated = False
        for u, v, w in edges:
            if B[u] != -math.inf and B[u] + w > B[v] + 1e-5:
                B[v] = B[u] + w
                updated = True
                if v == 'zero' and B['zero'] > 1e-5:
                    return False, None
        if not updated:
            break
            
    for u, v, w in edges:
        if B[u] != -math.inf and B[u] + w > B[v] + 1e-5:
            return False, None
            
    return True, {v: B[v] for v in route}

def build_solution_regret_robust(users, initial_routes, nodes, max_ride_time, max_route_duration, capacity, dist_matrix, end_time=None):
    routes = [r.copy() for r in initial_routes]
    unrouted = set(user['user_id'] for user in users)
    user_dict = {u['user_id']: u for u in users}
    
    while unrouted:
        if end_time and time.time() > end_time:
            break
            
        best_uid = None
        best_move_for_user = None
        max_regret = -math.inf
        
        for uid in list(unrouted):
            u = user_dict[uid]
            insertions = []
            
            for k, route in enumerate(routes):
                n = len(route)
                # Ensure we only insert at valid intermediate locations, leaving depot at route[-1]
                for i in range(1, n):
                    for j in range(i, n):
                        if i == j:
                            cost_diff = dist_matrix[route[i-1]][u['pickup']] + \
                                        dist_matrix[u['pickup']][u['dropoff']] + \
                                        dist_matrix[u['dropoff']][route[i]] - \
                                        dist_matrix[route[i-1]][route[i]]
                        else:
                            cost_diff = dist_matrix[route[i-1]][u['pickup']] + \
                                        dist_matrix[u['pickup']][route[i]] - \
                                        dist_matrix[route[i-1]][route[i]] + \
                                        dist_matrix[route[j-1]][u['dropoff']] + \
                                        dist_matrix[u['dropoff']][route[j]] - \
                                        dist_matrix[route[j-1]][route[j]]
                                        
                        insertions.append((cost_diff, k, i, j))
                        
            insertions.sort(key=lambda x: x[0])
            
            feasible_insertions = []
            for cost_diff, k, i, j in insertions:
                route = routes[k]
                new_route = route[:i] + [u['pickup']] + route[i:j] + [u['dropoff']] + route[j:]
                
                if not check_capacity(new_route, nodes, capacity): 
                    continue
                if not quick_feasibility_check(new_route, nodes, max_ride_time, max_route_duration, dist_matrix): 
                    continue
                
                feasible, _ = check_schedule(new_route, nodes, max_ride_time, max_route_duration, dist_matrix)
                if feasible:
                    feasible_insertions.append((cost_diff, k, new_route))
                    if len(feasible_insertions) >= 2:
                        break
                        
            if not feasible_insertions:
                continue 
                
            delta1 = feasible_insertions[0][0]
            delta2 = feasible_insertions[1][0] if len(feasible_insertions) > 1 else delta1 + 100000.0
            
            regret = delta2 - delta1
            if regret > max_regret:
                max_regret = regret
                best_uid = uid
                best_move_for_user = feasible_insertions[0]
                
        if best_uid is None:
            break
            
        k = best_move_for_user[1]
        routes[k] = best_move_for_user[2]
        unrouted.remove(best_uid)
        
    return routes, unrouted

def evaluate_full_state(routes, unrouted, dist_matrix):
    cost = 0.0
    for r in routes:
        cost += sum(dist_matrix[r[x]][r[x+1]] for x in range(len(r)-1))
    return cost + len(unrouted) * 1e7

def ruin_route(current_routes, current_unrouted, users, q, nodes, num_nodes):
    valid_routes = [i for i, r in enumerate(current_routes) if len(r) > 2]
    removed_uids = set()
    new_routes = [r.copy() for r in current_routes]
    
    if valid_routes and random.random() < 0.2:
        idx = random.choice(valid_routes)
        r = current_routes[idx]
        for v in r:
            if v != 0 and v != num_nodes-1:
                uid = nodes[v].get('user_id')
                if uid is not None:
                    removed_uids.add(uid)
        new_routes[idx] = [0, num_nodes-1]
    
    routed_uids = [u['user_id'] for u in users if u['user_id'] not in current_unrouted and u['user_id'] not in removed_uids]
    if routed_uids:
        more = min(q, len(routed_uids))
        removed_uids.update(random.sample(routed_uids, more))
        
    final_routes = []
    for r in new_routes:
        final_routes.append([v for v in r if v == 0 or v == num_nodes-1 or nodes[v].get('user_id') not in removed_uids])
        
    return final_routes, removed_uids

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    time_limit_end = start_time + args.time_limit
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    nodes.sort(key=lambda x: x['node_id'])
    
    num_nodes = data['num_nodes']
    num_vehicles = data['num_vehicles']
    capacity = data['vehicle_capacity']
    max_ride_time = data['maximum_ride_time']
    max_route_duration = data['maximum_route_duration']

    dist_matrix = [[0.0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i][j] = get_dist(nodes[i], nodes[j])

    unique_users = {}
    for n in nodes:
        if n['node_type'] == 'pickup':
            uid = n['user_id']
            if uid not in unique_users: unique_users[uid] = {}
            unique_users[uid]['pickup'] = n['node_id']
        elif n['node_type'] == 'dropoff':
            uid = n['user_id']
            if uid not in unique_users: unique_users[uid] = {}
            unique_users[uid]['dropoff'] = n['node_id']
            
    users = [{'user_id': uid, 'pickup': ud['pickup'], 'dropoff': ud['dropoff']} for uid, ud in unique_users.items()]
    user_dict = {u['user_id']: u for u in users}

    empty_routes = [[0, num_nodes-1] for _ in range(num_vehicles)]
    sorted_users = sorted(users, key=lambda a: nodes[a['pickup']]['earliest_time'])
    
    current_routes, current_unrouted = build_solution_regret_robust(
        sorted_users, empty_routes, nodes, max_ride_time, max_route_duration, capacity, dist_matrix, time_limit_end - 0.5
    )
    current_cost = evaluate_full_state(current_routes, current_unrouted, dist_matrix)
    
    best_feasible_routes = current_routes if not current_unrouted else None
    best_feasible_cost = current_cost if not current_unrouted else math.inf
    
    if logger and best_feasible_routes is not None:
        logger.log(best_feasible_cost)

    T = max(100.0, current_cost * 0.05)
    alpha = 0.995

    while time.time() < time_limit_end - 0.5:
        q = random.randint(1, min(10, len(users)//4 + 1))
        
        new_routes, removed_uids = ruin_route(current_routes, current_unrouted, users, q, nodes, num_nodes)
                
        users_to_insert_uids = removed_uids.union(current_unrouted)
        users_to_insert = [user_dict[uid] for uid in users_to_insert_uids]
        
        cand_routes, cand_unrouted = build_solution_regret_robust(
            users_to_insert, new_routes, nodes, max_ride_time, max_route_duration, capacity, dist_matrix, time_limit_end - 0.5
        )
        cand_cost = evaluate_full_state(cand_routes, cand_unrouted, dist_matrix)
        
        if cand_cost < current_cost:
            current_routes = cand_routes
            current_unrouted = cand_unrouted
            current_cost = cand_cost
        else:
            if T > 1e-4 and math.exp((current_cost - cand_cost) / T) > random.random():
                current_routes = cand_routes
                current_unrouted = cand_unrouted
                current_cost = cand_cost
                
        if not cand_unrouted and cand_cost < best_feasible_cost:
            best_feasible_cost = cand_cost
            best_feasible_routes = cand_routes
            if logger:
                logger.log(best_feasible_cost)
                
        T *= alpha
        if T < 1e-3:
            T = max(100.0, current_cost * 0.05)

    if best_feasible_routes is None:
        best_feasible_routes = current_routes
        best_feasible_cost = evaluate_full_state(current_routes, [], dist_matrix)

    routes_out = {}
    service_times_out = {}
    ride_times_out = {}
    
    for k, r in enumerate(best_feasible_routes):
        feasible, B_times = check_schedule(r, nodes, max_ride_time, max_route_duration, dist_matrix)
        routes_out[str(k)] = r
        if B_times:
            service_times_out[f"depot_start_{k}"] = float(B_times[r[0]])
            service_times_out[f"depot_end_{k}"] = float(B_times[r[-1]])
            for v in r[1:-1]:
                service_times_out[str(v)] = float(B_times[v])
                
    for u in users:
        p = u['pickup']
        d = u['dropoff']
        if str(p) in service_times_out and str(d) in service_times_out:
            rt = service_times_out[str(d)] - service_times_out[str(p)] - nodes[p]['service_duration']
            ride_times_out[str(u['user_id'])] = float(max(0.0, rt))
            
    solution = {
        "objective_value": float(best_feasible_cost),
        "routes": routes_out,
        "service_times": service_times_out,
        "ride_times": ride_times_out
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()