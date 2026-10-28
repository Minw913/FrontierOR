import sys
import json
import math
import time
import random
import argparse

from solution_logger import SolutionLogger

def euclidean_distance(n1, n2):
    return math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])

def check_route(route, node_tw_early, node_tw_late, node_demand, node_srv_time, dist_matrix, capacity):
    load = 0
    t = node_tw_early[route[0]]
    
    for k in range(len(route) - 1):
        idx = route[k]
        nxt = route[k+1]
        
        tw_early = node_tw_early[idx]
        if t < tw_early:
            t = tw_early
        elif t > node_tw_late[idx]:
            return False
            
        load += node_demand[idx]
        if load > capacity or load < 0:
            return False
            
        t += node_srv_time[idx] + dist_matrix[idx][nxt]
        
    idx = route[-1]
    if t > node_tw_late[idx]:
        return False
        
    return True

def evaluate_insertion(route, req_p, req_d, dist_matrix, node_tw_early, node_tw_late, node_demand, node_srv_time, capacity):
    best_cost = float('inf')
    best_i, best_j = -1, -1
    route_len = len(route)
    
    for i in range(1, route_len):
        for j in range(i, route_len):
            if i == j:
                new_route = route[:i] + [req_p, req_d] + route[i:]
            else:
                new_route = route[:i] + [req_p] + route[i:j] + [req_d] + route[j:]
                
            if check_route(new_route, node_tw_early, node_tw_late, node_demand, node_srv_time, dist_matrix, capacity):
                if i == j:
                    delta = (dist_matrix[route[i-1]][req_p] + 
                             dist_matrix[req_p][req_d] + 
                             dist_matrix[req_d][route[i]] - 
                             dist_matrix[route[i-1]][route[i]])
                else:                 
                    delta = (dist_matrix[route[i-1]][req_p] + 
                             dist_matrix[req_p][route[i]] - 
                             dist_matrix[route[i-1]][route[i]] +
                             dist_matrix[route[j-1]][req_d] + 
                             dist_matrix[req_d][route[j]] - 
                             dist_matrix[route[j-1]][route[j]])
                             
                if delta < best_cost:
                    best_cost = delta
                    best_i = i
                    best_j = j
                    
    return best_i, best_j, best_cost

def random_removal(routes, num_to_remove, req_mapping, unassigned_list, req_id_to_req, *args, **kwargs):
    assigned_reqs = []
    for r_idx, route in enumerate(routes):
        for node in route:
            if node in req_mapping and req_mapping[node]['is_pickup']:
                assigned_reqs.append((req_mapping[node]['req_id'], r_idx))
                
    if not assigned_reqs: return routes, unassigned_list
    
    to_remove_count = min(num_to_remove, len(assigned_reqs))
    to_remove = random.sample(assigned_reqs, to_remove_count)
    
    new_unassigned = list(unassigned_list)
    to_remove_req_ids = set([x[0] for x in to_remove])
    for req_id, _ in to_remove:
        new_unassigned.append(req_id_to_req[req_id])
        
    new_routes = []
    for route in routes:
        new_r = [n for n in route if not (n in req_mapping and req_mapping[n]['req_id'] in to_remove_req_ids)]
        new_routes.append(new_r)
        
    return new_routes, new_unassigned

def worst_removal(routes, num_to_remove, req_mapping, unassigned_list, req_id_to_req, dist_matrix, *args, **kwargs):
    assigned_reqs = [] 
    for r_idx, route in enumerate(routes):
        reqs_in_route = set()
        for node in route:
            if node in req_mapping:
                reqs_in_route.add(req_mapping[node]['req_id'])
                
        if not reqs_in_route:
            continue
            
        orig_cost = 0
        for i in range(len(route)-1):
            orig_cost += dist_matrix[route[i]][route[i+1]]
            
        for req_id in reqs_in_route:
            req_info = req_id_to_req[req_id]
            req_p, req_d = req_info['pickup_node'], req_info['delivery_node']
            
            new_r = [n for n in route if n != req_p and n != req_d]
            new_cost = 0
            for i in range(len(new_r)-1):
                new_cost += dist_matrix[new_r[i]][new_r[i+1]]
                
            saving = orig_cost - new_cost
            assigned_reqs.append((req_id, saving, r_idx))
            
    if not assigned_reqs: return routes, unassigned_list
    
    assigned_reqs.sort(key=lambda x: x[1], reverse=True)
    
    to_remove = []
    candidates = assigned_reqs[:]
    p = 3
    while len(to_remove) < num_to_remove and candidates:
        idx = int(random.random()**p * len(candidates))
        to_remove.append(candidates.pop(idx))
    
    new_unassigned = list(unassigned_list)
    to_remove_req_ids = set([x[0] for x in to_remove])
    for req_id, _ , _ in to_remove:
        new_unassigned.append(req_id_to_req[req_id])
        
    new_routes = []
    for route in routes:
        new_r = [n for n in route if not (n in req_mapping and req_mapping[n]['req_id'] in to_remove_req_ids)]
        new_routes.append(new_r)
        
    return new_routes, new_unassigned

def shaw_removal(routes, num_to_remove, req_mapping, unassigned_list, req_id_to_req, dist_matrix, *args, **kwargs):
    assigned_reqs = [] 
    for r_idx, route in enumerate(routes):
        for node in route:
            if node in req_mapping and req_mapping[node]['is_pickup']:
                assigned_reqs.append(req_mapping[node]['req_id'])
                
    if not assigned_reqs: return routes, unassigned_list
    
    seed_req_id = random.choice(assigned_reqs)
    seed_p = req_id_to_req[seed_req_id]['pickup_node']
    
    assigned_reqs.sort(key=lambda r_id: dist_matrix[seed_p][req_id_to_req[r_id]['pickup_node']])
    
    candidates = assigned_reqs[:]
    to_remove_req_ids = set()
    p = 3
    while len(to_remove_req_ids) < num_to_remove and candidates:
        idx = int(random.random()**p * len(candidates))
        to_remove_req_ids.add(candidates.pop(idx))
        
    new_unassigned = list(unassigned_list)
    for req_id in to_remove_req_ids:
        new_unassigned.append(req_id_to_req[req_id])
        
    new_routes = []
    for route in routes:
        new_r = [n for n in route if not (n in req_mapping and req_mapping[n]['req_id'] in to_remove_req_ids)]
        new_routes.append(new_r)
        
    return new_routes, new_unassigned

def greedy_insertion(routes, unassigned, dist_matrix, node_tw_early, node_tw_late, node_demand, node_srv_time, capacity):
    random.shuffle(unassigned)
    unassigned_remaining = []
    for req in unassigned:
        req_p = req['pickup_node']
        req_d = req['delivery_node']
        
        b_route = -1
        b_cost = float('inf')
        b_i = -1
        b_j = -1
        
        for k, route in enumerate(routes):
            i, j, cost = evaluate_insertion(route, req_p, req_d, dist_matrix, node_tw_early, node_tw_late, node_demand, node_srv_time, capacity)
            if cost < b_cost:
                b_cost = cost
                b_route = k
                b_i = i
                b_j = j
                
        if b_route != -1:
            if b_i == b_j:
                routes[b_route] = routes[b_route][:b_i] + [req_p, req_d] + routes[b_route][b_i:]
            else:
                routes[b_route] = routes[b_route][:b_i] + [req_p] + routes[b_route][b_i:b_j] + [req_d] + routes[b_route][b_j:]
        else:
            unassigned_remaining.append(req)
            
    return routes, unassigned_remaining


def regret_insertion(routes, unassigned, dist_matrix, node_tw_early, node_tw_late, node_demand, node_srv_time, capacity):
    while unassigned:
        best_req_idx = -1
        max_regret = -1
        best_insertion = None
        
        for idx, req in enumerate(unassigned):
            req_p = req['pickup_node']
            req_d = req['delivery_node']
            
            c1 = float('inf')
            c2 = float('inf')
            b_route = -1
            b_i = -1
            b_j = -1
            
            for k, route in enumerate(routes):
                i, j, cost = evaluate_insertion(route, req_p, req_d, dist_matrix, node_tw_early, node_tw_late, node_demand, node_srv_time, capacity)
                if cost < c1:
                    c2 = c1
                    c1 = cost
                    b_route = k
                    b_i = i
                    b_j = j
                elif cost < c2:
                    c2 = cost
                    
            regret = c2 - c1
            if c1 == float('inf'):
                regret = 1e9
                
            if regret > max_regret:
                max_regret = regret
                best_req_idx = idx
                best_insertion = (b_route, b_i, b_j, c1)
                
        if best_insertion is None or best_insertion[3] == float('inf'):
            break
            
        req = unassigned.pop(best_req_idx)
        r_idx, b_i, b_j, _ = best_insertion
        req_p = req['pickup_node']
        req_d = req['delivery_node']
        if b_i == b_j:
            routes[r_idx] = routes[r_idx][:b_i] + [req_p, req_d] + routes[r_idx][b_i:]
        else:
            routes[r_idx] = routes[r_idx][:b_i] + [req_p] + routes[r_idx][b_i:b_j] + [req_d] + routes[r_idx][b_j:]

    return routes, unassigned

def calc_solution_cost(routes, unassigned, dist_matrix):
    cost = 0
    for route in routes:
        for i in range(len(route)-1):
            cost += dist_matrix[route[i]][route[i+1]]
    cost += 10000000 * len(unassigned)
    return cost

def get_true_cost(routes, dist_matrix):
    cost = 0
    for route in routes:
        for i in range(len(route)-1):
            cost += dist_matrix[route[i]][route[i+1]]
    return cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    max_time = args.time_limit
    random.seed(42)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    max_id = max(n['id'] for n in data['nodes']) + 1
    node_tw_early = [0] * max_id
    node_tw_late = [0] * max_id
    node_demand = [0] * max_id
    node_srv_time = [0] * max_id
    dist_matrix = [[0] * max_id for _ in range(max_id)]
    
    node_objs = {n['id']: n for n in data['nodes']}
    for i in range(max_id):
        if i in node_objs:
            n = node_objs[i]
            node_tw_early[i] = n['time_window'][0]
            node_tw_late[i] = n['time_window'][1]
            node_demand[i] = n['demand']
            node_srv_time[i] = n['service_time']
            
    for i in range(max_id):
        for j in range(max_id):
            if i in node_objs and j in node_objs:
                n1 = node_objs[i]
                n2 = node_objs[j]
                dist_matrix[i][j] = math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])

    depot_origin_idx = next(n['id'] for n in data['nodes'] if n['type'] == 'depot_origin')
    depot_dest_idx = next(n['id'] for n in data['nodes'] if n['type'] == 'depot_destination')

    requests = data['requests']
    req_id_to_req = {r['request_id']: r for r in requests}
    num_requests = data['num_requests']
    capacity = data['vehicle_capacity']

    req_mapping = {}
    for r in requests:
        req_mapping[r['pickup_node']] = {'req_id': r['request_id'], 'is_pickup': True}
        req_mapping[r['delivery_node']] = {'req_id': r['request_id'], 'is_pickup': False}

    req_list = list(requests)
    req_list.sort(key=lambda x: node_objs[x['pickup_node']]['time_window'][0])
    
    best_init_routes = None
    best_init_un = None
    best_un_count = float('inf')
    
    for attempt in range(20):
        if time.time() - start_time > max_time - 1.0:
            break
            
        if attempt > 0:
            random.shuffle(req_list)
        
        routes = [[depot_origin_idx, depot_dest_idx] for _ in range(data['num_vehicles'])]
        un = []
        for req in req_list:
            req_p = req['pickup_node']
            req_d = req['delivery_node']
            
            b_route = -1
            b_cost = float('inf')
            b_i = -1
            b_j = -1
            
            for k, route in enumerate(routes):
                i, j, cost = evaluate_insertion(route, req_p, req_d, dist_matrix, node_tw_early, node_tw_late, node_demand, node_srv_time, capacity)
                if cost < b_cost:
                    b_cost = cost
                    b_route = k
                    b_i = i
                    b_j = j
                    
            if b_route != -1:
                if b_i == b_j:
                    routes[b_route] = routes[b_route][:b_i] + [req_p, req_d] + routes[b_route][b_i:]
                else:
                    routes[b_route] = routes[b_route][:b_i] + [req_p] + routes[b_route][b_i:b_j] + [req_d] + routes[b_route][b_j:]
            else:
                un.append(req)
                
        if len(un) < best_un_count:
            best_un_count = len(un)
            best_init_routes = [list(r) for r in routes]
            best_init_un = list(un)
            
        if best_un_count == 0:
            break

    global_best_valid_cost = float('inf')
    best_routes_valid = None
    
    current_routes = best_init_routes
    current_unassigned = best_init_un
    current_cost = calc_solution_cost(current_routes, current_unassigned, dist_matrix)
    
    best_sa_cost = current_cost
    best_sa_routes = [list(r) for r in current_routes]
    best_sa_unassigned = list(current_unassigned)
    
    if len(current_unassigned) == 0:
        global_best_valid_cost = current_cost
        best_routes_valid = [list(r) for r in current_routes]
        if logger:
            logger.log(global_best_valid_cost)
            
    T = current_cost * 0.05 / math.log(2)
    if T < 1: T = 100
    cooling_rate = 0.9995

    destroy_ops = [random_removal, worst_removal, shaw_removal]
    repair_ops = [greedy_insertion, regret_insertion]

    iterations_since_improve = 0

    while time.time() - start_time < max_time - 0.5:
        routes_copy = [list(r) for r in current_routes]
        unassigned_copy = list(current_unassigned)
        
        destroy_op = random.choice(destroy_ops)
        repair_op = random.choice(repair_ops)
        
        pct = random.uniform(0.1, 0.4)
        num_to_remove = int(pct * num_requests)
        num_to_remove = max(1, min(num_to_remove, num_requests))
        
        routes_copy, unassigned_copy = destroy_op(routes_copy, num_to_remove, req_mapping, unassigned_copy, req_id_to_req, dist_matrix=dist_matrix)
        routes_copy, unassigned_copy = repair_op(routes_copy, unassigned_copy, dist_matrix, node_tw_early, node_tw_late, node_demand, node_srv_time, capacity)
        
        new_cost = calc_solution_cost(routes_copy, unassigned_copy, dist_matrix)
        
        if new_cost < current_cost:
            accept = True
        else:
            delta = new_cost - current_cost
            try:
                prob = math.exp(-delta / T)
            except OverflowError:
                prob = 0.0
            accept = (random.random() < prob)
            
        if accept:
            current_routes = routes_copy
            current_unassigned = unassigned_copy
            current_cost = new_cost
            
            if current_cost < best_sa_cost:
                best_sa_cost = current_cost
                best_sa_routes = [list(r) for r in current_routes]
                best_sa_unassigned = list(current_unassigned)
                iterations_since_improve = 0
                
                if len(current_unassigned) == 0 and current_cost < global_best_valid_cost:
                    global_best_valid_cost = current_cost
                    best_routes_valid = [list(r) for r in current_routes]
                    if logger:
                        logger.log(global_best_valid_cost)
            else:
                iterations_since_improve += 1
        else:
            iterations_since_improve += 1
            
        T *= cooling_rate
        if T < 1e-5:
            T = 1e-5
            
        if iterations_since_improve > 1000:
            current_routes = [list(r) for r in best_sa_routes]
            current_unassigned = list(best_sa_unassigned)
            current_cost = best_sa_cost
            T = best_sa_cost * 0.05 / math.log(2)
            if T < 1: T = 100
            iterations_since_improve = 0

    final_routes = best_routes_valid if best_routes_valid is not None else best_sa_routes
    final_cost = get_true_cost(final_routes, dist_matrix)
            
    solution_output = {
        "objective_value": final_cost,
        "routes": []
    }
    
    for k, route in enumerate(final_routes):
        solution_output["routes"].append({
            "vehicle": k,
            "nodes": route
        })
        
    with open(args.solution_path, 'w') as f:
        json.dump(solution_output, f, indent=4)

if __name__ == '__main__':
    main()