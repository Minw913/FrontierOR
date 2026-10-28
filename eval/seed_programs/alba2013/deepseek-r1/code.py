import argparse
import json
import time
from solution_logger import SolutionLogger

def compute_tour_cost(tour, dist_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i+1]
        cost += dist_matrix[u][v]
    return cost

def nearest_neighbor_tsp(depot, vertices, dist_matrix):
    n = len(vertices)
    if n == 0:
        return [depot, depot]
    unvisited = set(vertices)
    tour = [depot]
    current = depot
    while unvisited:
        min_cost = float('inf')
        next_vertex = None
        for j in unvisited:
            if dist_matrix[current][j] < min_cost:
                min_cost = dist_matrix[current][j]
                next_vertex = j
        tour.append(next_vertex)
        unvisited.remove(next_vertex)
        current = next_vertex
    tour.append(depot)
    return tour

def two_opt(tour, dist_matrix, start_time, time_limit):
    if len(tour) <= 3:
        return tour
    n = len(tour) - 1
    improvement = True
    best_tour = tour
    best_cost = compute_tour_cost(tour, dist_matrix)
    while improvement:
        if time.time() - start_time > time_limit:
            break
        improvement = False
        for i in range(1, n-1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i+1, n):
                a, b, c, d = best_tour[i-1], best_tour[i], best_tour[j], best_tour[j+1]
                old_edge_cost = dist_matrix[a][b] + dist_matrix[c][d]
                new_edge_cost = dist_matrix[a][c] + dist_matrix[b][d]
                if new_edge_cost < old_edge_cost:
                    new_tour = best_tour[:i] + list(reversed(best_tour[i:j+1])) + best_tour[j+1:]
                    new_cost = best_cost - old_edge_cost + new_edge_cost
                    best_tour = new_tour
                    best_cost = new_cost
                    improvement = True
                    break
            if improvement:
                break
    return best_tour

def check_compatibility(pickup_tour, delivery_tour, requests, s, l, pickup_vertex_to_request, delivery_vertex_to_request, request_id_to_delivery_vertex):
    if len(pickup_tour) < 3 or len(delivery_tour) < 3:
        return True
    pickup_vertices = pickup_tour[1:-1]
    delivery_vertices = delivery_tour[1:-1]
    
    pickup_requests = []
    for v in pickup_vertices:
        if v in pickup_vertex_to_request:
            pickup_requests.append(pickup_vertex_to_request[v])
    
    delivery_requests = []
    for v in delivery_vertices:
        if v in delivery_vertex_to_request:
            delivery_requests.append(delivery_vertex_to_request[v])
    
    request_id_to_delivery_index = {}
    for idx, req_id in enumerate(delivery_requests):
        request_id_to_delivery_index[req_id] = idx
    
    delivery_sequence = []
    for req_id in pickup_requests:
        delivery_sequence.append(request_id_to_delivery_index[req_id])
    
    stacks = [None] * s
    stack_sizes = [0] * s
    for d_index in delivery_sequence:
        assigned = False
        for stack_id in range(s):
            if stack_sizes[stack_id] < l:
                if stacks[stack_id] is None or stacks[stack_id] > d_index:
                    stacks[stack_id] = d_index
                    stack_sizes[stack_id] += 1
                    assigned = True
                    break
        if not assigned:
            return False
    return True

def build_fallback_delivery_tour(pickup_tour, requests, delivery_depot, pickup_vertex_to_request, request_id_to_delivery_vertex):
    pickup_vertices = pickup_tour[1:-1]
    pickup_requests = []
    for v in pickup_vertices:
        if v in pickup_vertex_to_request:
            pickup_requests.append(pickup_vertex_to_request[v])
    delivery_tour = [delivery_depot]
    for req_id in reversed(pickup_requests):
        delivery_vertex = request_id_to_delivery_vertex[req_id]
        delivery_tour.append(delivery_vertex)
    delivery_tour.append(delivery_depot)
    return delivery_tour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    logger = SolutionLogger(args.log_path, "minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    parameters = instance['parameters']
    s = parameters['num_stacks_s']
    l_val = parameters['stack_height_l']
    n = parameters['num_requests_n']
    pickup_region = instance['pickup_region']
    delivery_region = instance['delivery_region']
    requests_data = instance['requests']
    
    pickup_depot_idx = pickup_region['depot_index']
    delivery_depot_idx = delivery_region['depot_index']
    pickup_dist = pickup_region['distance_matrix']
    delivery_dist = delivery_region['distance_matrix']
    
    num_pickup_vertices = parameters['num_pickup_vertices']
    num_delivery_vertices = parameters['num_delivery_vertices']
    
    pickup_vertices = list(range(num_pickup_vertices))
    delivery_vertices = list(range(num_delivery_vertices))
    
    pickup_vertex_to_request = {}
    delivery_vertex_to_request = {}
    request_id_to_pickup_vertex = {}
    request_id_to_delivery_vertex = {}
    for req in requests_data:
        req_id = req['request_id']
        p_vertex = req['pickup_vertex']
        d_vertex = req['delivery_vertex']
        pickup_vertex_to_request[p_vertex] = req_id
        delivery_vertex_to_request[d_vertex] = req_id
        request_id_to_pickup_vertex[req_id] = p_vertex
        request_id_to_delivery_vertex[req_id] = d_vertex
    
    best_objective = float('inf')
    best_pickup_tour = None
    best_delivery_tour = None
    
    def maybe_log_and_update(obj_val, pickup_tour, delivery_tour):
        nonlocal best_objective, best_pickup_tour, best_delivery_tour
        if obj_val < best_objective:
            best_objective = obj_val
            best_pickup_tour = pickup_tour
            best_delivery_tour = delivery_tour
            if logger:
                logger.log(obj_val)
    
    initial_pickup_tour = nearest_neighbor_tsp(pickup_depot_idx, [v for v in pickup_vertices if v != pickup_depot_idx], pickup_dist)
    cost_initial_pickup = compute_tour_cost(initial_pickup_tour, pickup_dist)
    
    initial_delivery_tour = nearest_neighbor_tsp(delivery_depot_idx, [v for v in delivery_vertices if v != delivery_depot_idx], delivery_dist)
    cost_initial_delivery = compute_tour_cost(initial_delivery_tour, delivery_dist)
    
    if time.time() - start_time < args.time_limit:
        initial_pickup_tour = two_opt(initial_pickup_tour, pickup_dist, start_time, args.time_limit)
        cost_initial_pickup = compute_tour_cost(initial_pickup_tour, pickup_dist)
    
    if time.time() - start_time < args.time_limit:
        initial_delivery_tour = two_opt(initial_delivery_tour, delivery_dist, start_time, args.time_limit)
        cost_initial_delivery = compute_tour_cost(initial_delivery_tour, delivery_dist)
    
    if check_compatibility(initial_pickup_tour, initial_delivery_tour, requests_data, s, l_val, pickup_vertex_to_request, delivery_vertex_to_request, request_id_to_delivery_vertex):
        total_cost = cost_initial_pickup + cost_initial_delivery
        maybe_log_and_update(total_cost, initial_pickup_tour, initial_delivery_tour)
    else:
        fallback_delivery_tour = build_fallback_delivery_tour(initial_pickup_tour, requests_data, delivery_depot_idx, pickup_vertex_to_request, request_id_to_delivery_vertex)
        cost_fallback_delivery = compute_tour_cost(fallback_delivery_tour, delivery_dist)
        total_cost = cost_initial_pickup + cost_fallback_delivery
        maybe_log_and_update(total_cost, initial_pickup_tour, fallback_delivery_tour)
    
    if time.time() - start_time < args.time_limit:
        new_pickup_tour = two_opt(initial_pickup_tour, pickup_dist, start_time, args.time_limit)
        cost_new_pickup = compute_tour_cost(new_pickup_tour, pickup_dist)
        
        if check_compatibility(new_pickup_tour, initial_delivery_tour, requests_data, s, l_val, pickup_vertex_to_request, delivery_vertex_to_request, request_id_to_delivery_vertex):
            total_cost = cost_new_pickup + cost_initial_delivery
            maybe_log_and_update(total_cost, new_pickup_tour, initial_delivery_tour)
        else:
            fallback_delivery_tour_new = build_fallback_delivery_tour(new_pickup_tour, requests_data, delivery_depot_idx, pickup_vertex_to_request, request_id_to_delivery_vertex)
            cost_fallback_delivery_new = compute_tour_cost(fallback_delivery_tour_new, delivery_dist)
            total_cost = cost_new_pickup + cost_fallback_delivery_new
            maybe_log_and_update(total_cost, new_pickup_tour, fallback_delivery_tour_new)
    
    if time.time() - start_time < args.time_limit:
        new_delivery_tour = two_opt(initial_delivery_tour, delivery_dist, start_time, args.time_limit)
        cost_new_delivery = compute_tour_cost(new_delivery_tour, delivery_dist)
        
        if check_compatibility(initial_pickup_tour, new_delivery_tour, requests_data, s, l_val, pickup_vertex_to_request, delivery_vertex_to_request, request_id_to_delivery_vertex):
            total_cost = cost_initial_pickup + cost_new_delivery
            maybe_log_and_update(total_cost, initial_pickup_tour, new_delivery_tour)
        
        if best_pickup_tour is not None and 'new_pickup_tour' in locals():
            if check_compatibility(new_pickup_tour, new_delivery_tour, requests_data, s, l_val, pickup_vertex_to_request, delivery_vertex_to_request, request_id_to_delivery_vertex):
                total_cost = cost_new_pickup + cost_new_delivery
                maybe_log_and_update(total_cost, new_pickup_tour, new_delivery_tour)
    
    if best_pickup_tour is None:
        fallback_delivery_tour = build_fallback_delivery_tour(initial_pickup_tour, requests_data, delivery_depot_idx, pickup_vertex_to_request, request_id_to_delivery_vertex)
        cost_fallback_delivery = compute_tour_cost(fallback_delivery_tour, delivery_dist)
        total_cost = cost_initial_pickup + cost_fallback_delivery
        best_pickup_tour = initial_pickup_tour
        best_delivery_tour = fallback_delivery_tour
        best_objective = total_cost
        if logger:
            logger.log(total_cost)
    
    solution = {
        "objective_value": best_objective,
        "pickup_tour": best_pickup_tour,
        "delivery_tour": best_delivery_tour
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()