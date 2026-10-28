import json
import math
import time
import argparse
from solution_logger import SolutionLogger

def evaluate_route(route, id_to_index, all_nodes, dist_matrix, depot_data, vehicle_capacity):
    if len(route) == 0:
        return True, 0.0
    
    full_route_indices = [0]
    for cid in route:
        full_route_indices.append(id_to_index[cid])
    full_route_indices.append(0)
    
    total_demand = 0
    total_distance = 0.0
    current_time = depot_data['ready_time']
    prev_index = full_route_indices[0]
    
    for i in range(1, len(full_route_indices)):
        curr_index = full_route_indices[i]
        d = dist_matrix[prev_index][curr_index]
        total_distance += d
        current_time += d
        
        if curr_index == 0:
            if current_time > depot_data['due_date']:
                return False, None
        else:
            node = all_nodes[curr_index]
            if current_time < node['ready_time']:
                current_time = node['ready_time']
            elif current_time > node['due_date']:
                return False, None
            current_time += node['service_time']
            total_demand += node['demand']
        prev_index = curr_index

    if total_demand > vehicle_capacity:
        return False, None
        
    return True, total_distance

def main():
    parser = argparse.ArgumentParser(description='Vehicle Routing Problem with Time Windows')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file for intermediate solutions')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    depot = instance_data['depot']
    customers = instance_data['customers']
    vehicle_capacity = instance_data['vehicle_capacity']
    num_vehicles = instance_data['num_vehicles']

    all_nodes = [depot] + customers
    id_to_index = {}
    index_to_id = []
    id_to_index[depot['id']] = 0
    index_to_id.append(depot['id'])
    for idx, c in enumerate(customers, start=1):
        id_to_index[c['id']] = idx
        index_to_id.append(c['id'])

    n_nodes = len(all_nodes)
    dist_matrix = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            dx = all_nodes[i]['x'] - all_nodes[j]['x']
            dy = all_nodes[i]['y'] - all_nodes[j]['y']
            dist_matrix[i][j] = math.sqrt(dx*dx + dy*dy)

    for c in customers:
        if c['demand'] > vehicle_capacity:
            with open(args.solution_path, 'w') as f:
                json.dump({"objective_value": None, "routes": []}, f)
            return
        idx = id_to_index[c['id']]
        travel_time = dist_matrix[0][idx]
        earliest_arrival = depot['ready_time'] + travel_time
        if earliest_arrival > c['due_date']:
            with open(args.solution_path, 'w') as f:
                json.dump({"objective_value": None, "routes": []}, f)
            return

    customer_ids = [c['id'] for c in customers]
    routes = [[cid] for cid in customer_ids]
    customer_route = {}
    for i, cid in enumerate(customer_ids):
        customer_route[cid] = i

    savings_list = []
    for i in customer_ids:
        for j in customer_ids:
            if i != j:
                idx_i = id_to_index[i]
                idx_j = id_to_index[j]
                s = dist_matrix[0][idx_i] + dist_matrix[0][idx_j] - dist_matrix[idx_i][idx_j]
                savings_list.append((s, i, j))
    savings_list.sort(key=lambda x: x[0], reverse=True)

    active_routes = [True] * len(routes)
    for s_val, i, j in savings_list:
        r_i = customer_route[i]
        r_j = customer_route[j]
        if r_i == r_j or not active_routes[r_i] or not active_routes[r_j]:
            continue
        route_i = routes[r_i]
        route_j = routes[r_j]
        if route_i[-1] != i or route_j[0] != j:
            continue
        new_route = route_i + route_j
        feasible, _ = evaluate_route(new_route, id_to_index, all_nodes, dist_matrix, depot, vehicle_capacity)
        if not feasible:
            continue
        new_route_index = len(routes)
        routes.append(new_route)
        active_routes.append(True)
        active_routes[r_i] = False
        active_routes[r_j] = False
        for cid in new_route:
            customer_route[cid] = new_route_index

    initial_routes = []
    for i in range(len(routes)):
        if active_routes[i]:
            initial_routes.append(routes[i])

    current_solution = initial_routes
    current_total_distance = 0.0
    route_distances = []
    for route in current_solution:
        _, dist = evaluate_route(route, id_to_index, all_nodes, dist_matrix, depot, vehicle_capacity)
        route_distances.append(dist)
        current_total_distance += dist

    if logger:
        logger.log(current_total_distance)

    best_solution = current_solution
    best_distance = current_total_distance

    start_time = time.time()
    time_limit = args.time_limit

    while time.time() - start_time < time_limit:
        improvement_found = False
        new_solution = current_solution[:]
        new_route_distances = route_distances[:]
        new_total_distance = current_total_distance

        for r1_idx, route1 in enumerate(current_solution):
            for i, customer in enumerate(route1):
                new_route1 = route1[:i] + route1[i+1:]
                if len(new_route1) == 0:
                    feasible1 = True
                    dist1 = 0.0
                else:
                    feasible1, dist1 = evaluate_route(new_route1, id_to_index, all_nodes, dist_matrix, depot, vehicle_capacity)
                if not feasible1:
                    continue

                for r2_idx, route2 in enumerate(current_solution):
                    if r1_idx == r2_idx:
                        continue
                    for j in range(len(route2)+1):
                        new_route2 = route2[:j] + [customer] + route2[j:]
                        feasible2, dist2 = evaluate_route(new_route2, id_to_index, all_nodes, dist_matrix, depot, vehicle_capacity)
                        if not feasible2:
                            continue
                        new_total = current_total_distance - route_distances[r1_idx] - route_distances[r2_idx] + dist1 + dist2
                        if new_total < new_total_distance - 1e-5:
                            new_solution = []
                            new_route_distances = []
                            for idx, rt in enumerate(current_solution):
                                if idx == r1_idx:
                                    if len(new_route1) > 0:
                                        new_solution.append(new_route1)
                                        new_route_distances.append(dist1)
                                elif idx == r2_idx:
                                    new_solution.append(new_route2)
                                    new_route_distances.append(dist2)
                                else:
                                    new_solution.append(rt)
                                    new_route_distances.append(route_distances[idx])
                            new_total_distance = new_total
                            improvement_found = True
                            if logger:
                                logger.log(new_total)
                            break
                    if improvement_found:
                        break
                if improvement_found:
                    break

                if not improvement_found:
                    new_route2 = [customer]
                    feasible2, dist2 = evaluate_route(new_route2, id_to_index, all_nodes, dist_matrix, depot, vehicle_capacity)
                    if feasible2:
                        new_total = current_total_distance - route_distances[r1_idx] + dist1 + dist2
                        if new_total < new_total_distance - 1e-5:
                            new_solution = []
                            new_route_distances = []
                            for idx, rt in enumerate(current_solution):
                                if idx == r1_idx:
                                    if len(new_route1) > 0:
                                        new_solution.append(new_route1)
                                        new_route_distances.append(dist1)
                                else:
                                    new_solution.append(rt)
                                    new_route_distances.append(route_distances[idx])
                            new_solution.append(new_route2)
                            new_route_distances.append(dist2)
                            new_total_distance = new_total
                            improvement_found = True
                            if logger:
                                logger.log(new_total)
                if improvement_found:
                    break
            if improvement_found:
                break

        if improvement_found:
            current_solution = new_solution
            route_distances = new_route_distances
            current_total_distance = new_total_distance
            if current_total_distance < best_distance - 1e-5:
                best_solution = [r[:] for r in current_solution]
                best_distance = current_total_distance
            continue

        for route_idx, route in enumerate(current_solution):
            n = len(route)
            if n < 2:
                continue
            improved_two_opt = False
            for i in range(n-1):
                for j in range(i+1, n):
                    new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                    feasible, dist_new = evaluate_route(new_route, id_to_index, all_nodes, dist_matrix, depot, vehicle_capacity)
                    if feasible and dist_new < route_distances[route_idx] - 1e-5:
                        current_solution[route_idx] = new_route
                        route_distances[route_idx] = dist_new
                        current_total_distance = current_total_distance - route_distances[route_idx] + dist_new
                        improvement_found = True
                        improved_two_opt = True
                        if logger:
                            logger.log(current_total_distance)
                        if current_total_distance < best_distance - 1e-5:
                            best_solution = [r[:] for r in current_solution]
                            best_distance = current_total_distance
                        break
                if improved_two_opt:
                    break
            if improved_two_opt:
                break
        if not improvement_found:
            break

    solution_routes = []
    depot_id = depot['id']
    for route in best_solution:
        path = [depot_id] + route + [depot_id]
        solution_routes.append({
            "customer_ids": route,
            "path": path
        })
    
    solution = {
        "objective_value": best_distance,
        "routes": solution_routes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()