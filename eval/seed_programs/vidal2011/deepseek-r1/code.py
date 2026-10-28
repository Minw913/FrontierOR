import math
import json
import argparse
import time
from solution_logger import SolutionLogger

def compute_distance_matrix(depots, customers):
    points = []
    for depot in depots:
        points.append((depot['x'], depot['y']))
    for customer in customers:
        points.append((customer['x'], customer['y']))
    n_total = len(points)
    dist_matrix = [[0.0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        xi, yi = points[i]
        for j in range(i + 1, n_total):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d_val = math.sqrt(dx * dx + dy * dy)
            dist_matrix[i][j] = d_val
            dist_matrix[j][i] = d_val
    return dist_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    d = data['d']
    t = data['t']
    m = data['m']
    vehicle_capacity = data['vehicle_capacity']
    route_duration_limit = data['route_duration_limit']
    depots = data['depots']
    customers = data['customers']
    
    depots_sorted = sorted(depots, key=lambda depot: depot['id'])
    customers_sorted = sorted(customers, key=lambda customer: customer['id'])
    depot_ids = [depot['id'] for depot in depots_sorted]
    
    dist_matrix = compute_distance_matrix(depots_sorted, customers_sorted)
    assignments = {}
    current_demand = [[0] * t for _ in range(d)]
    
    sorted_customers = sorted(customers_sorted, key=lambda cust: cust['demand'] * len(cust['pattern_list']), reverse=True)
    
    for customer in sorted_customers:
        cid = customer['id']
        c_demand = customer['demand']
        customer_index = d + cid
        best_cost = float('inf')
        best_depot = None
        best_pattern = None
        for depot_idx in range(d):
            for pattern in customer['pattern_list']:
                pattern_days_0_indexed = [day - 1 for day in pattern]
                frequency = len(pattern_days_0_indexed)
                cost1 = 0.0
                for day_0 in pattern_days_0_indexed:
                    cost1 += dist_matrix[depot_idx][customer_index]
                penalty = 0.0
                for day_0 in pattern_days_0_indexed:
                    if current_demand[depot_idx][day_0] + c_demand > m * vehicle_capacity:
                        penalty += (current_demand[depot_idx][day_0] + c_demand - m * vehicle_capacity)
                total_cost = cost1 + 1000.0 * penalty
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_depot = depot_idx
                    best_pattern = pattern
        assignments[cid] = (best_depot, best_pattern)
        pattern_days = [day - 1 for day in best_pattern]
        for day_0 in pattern_days:
            current_demand[best_depot][day_0] += customer['demand']
    
    elapsed_time = time.perf_counter() - start_time
    time_remaining = args.time_limit - elapsed_time
    
    total_objective = 0.0
    routes_output = []
    assignments_output = {}
    
    for customer in customers_sorted:
        cid = customer['id']
        depot_index, pattern_list = assignments[cid]
        assignments_output[str(cid)] = {
            "depot": depot_ids[depot_index],
            "pattern": sorted(pattern_list)
        }
    
    if time_remaining <= 0:
        if logger:
            logger.log_solution(total_objective, {"objective_value": total_objective, "assignments": assignments_output, "routes": routes_output})
        with open(args.solution_path, 'w') as f:
            json.dump({"objective_value": total_objective, "assignments": assignments_output, "routes": routes_output}, f)
        return
        
    for depot_idx in range(d):
        for day in range(t):
            elapsed_time = time.perf_counter() - start_time
            time_remaining = args.time_limit - elapsed_time
            if time_remaining <= 0:
                break
            customers_this_day = []
            for customer in customers_sorted:
                cid = customer['id']
                assigned_depot, pattern = assignments[cid]
                day_0_indexed = day
                if assigned_depot == depot_idx:
                    pattern_days_0_indexed = [p - 1 for p in pattern]
                    if day_0_indexed in pattern_days_0_indexed:
                        customers_this_day.append(customer)
            if not customers_this_day:
                continue
            n_local = len(customers_this_day)
            if time_remaining > 0 and n_local > 1 and time_remaining > 0.0005 * n_local * n_local:
                routes = []
                for customer in customers_this_day:
                    cid = customer['id']
                    customer_index = d + cid
                    dist_to_depot = dist_matrix[depot_idx][customer_index]
                    travel_cost = 2 * dist_to_depot
                    service_time = customer['service_duration']
                    total_duration = travel_cost + service_time
                    if total_duration > route_duration_limit:
                        total_duration = route_duration_limit
                    route_data = {
                        'sequence': [cid],
                        'end1': cid,
                        'end2': cid,
                        'total_demand': customer['demand'],
                        'travel_cost': travel_cost,
                        'service_time': service_time,
                        'total_duration': total_duration
                    }
                    routes.append(route_data)
                num_routes = len(routes)
                while num_routes > m:
                    best_saving = -10**9
                    best_i = None
                    best_j = None
                    best_connect = None
                    for i in range(num_routes):
                        for j in range(i + 1, num_routes):
                            routeA = routes[i]
                            routeB = routes[j]
                            if routeA['total_demand'] + routeB['total_demand'] > vehicle_capacity:
                                continue
                            a2 = routeA['end2']
                            a2_idx = d + a2
                            b1 = routeB['end1']
                            b1_idx = d + b1
                            b2 = routeB['end2']
                            b2_idx = d + b2
                            a1 = routeA['end1']
                            a1_idx = d + a1
                            saving1 = dist_matrix[a2_idx][depot_idx] + dist_matrix[depot_idx][b1_idx] - dist_matrix[a2_idx][b1_idx]
                            saving2 = dist_matrix[a2_idx][depot_idx] + dist_matrix[depot_idx][b2_idx] - dist_matrix[a2_idx][b2_idx]
                            saving3 = dist_matrix[a1_idx][depot_idx] + dist_matrix[depot_idx][b1_idx] - dist_matrix[a1_idx][b1_idx]
                            saving4 = dist_matrix[a1_idx][depot_idx] + dist_matrix[depot_idx][b2_idx] - dist_matrix[a1_idx][b2_idx]
                            savings = [saving1, saving2, saving3, saving4]
                            max_saving = max(savings)
                            if max_saving < 1e-7:
                                continue
                            idx_max = savings.index(max_saving)
                            new_service_time = routeA['service_time'] + routeB['service_time']
                            new_demand = routeA['total_demand'] + routeB['total_demand']
                            new_travel_cost = routeA['travel_cost'] + routeB['travel_cost'] - max_saving
                            new_duration = new_travel_cost + new_service_time
                            if new_duration > route_duration_limit:
                                continue
                            if max_saving > best_saving:
                                best_saving = max_saving
                                best_i = i
                                best_j = j
                                best_connect = idx_max
                    if best_saving == -10**9:
                        break
                    A = routes[best_i]
                    B = routes[best_j]
                    if best_connect == 0:
                        sequence_new = A['sequence'] + B['sequence']
                        end1_new = A['end1']
                        end2_new = B['end2']
                    elif best_connect == 1:
                        sequence_new = A['sequence'] + B['sequence'][::-1]
                        end1_new = A['end1']
                        end2_new = B['end1']
                    elif best_connect == 2:
                        sequence_new = A['sequence'][::-1] + B['sequence']
                        end1_new = A['end2']
                        end2_new = B['end2']
                    else:
                        sequence_new = A['sequence'][::-1] + B['sequence'][::-1]
                        end1_new = A['end2']
                        end2_new = B['end1']
                    new_service_time = A['service_time'] + B['service_time']
                    new_demand = A['total_demand'] + B['total_demand']
                    new_travel_cost = A['travel_cost'] + B['travel_cost'] - best_saving
                    new_duration = new_travel_cost + new_service_time
                    new_route = {
                        'sequence': sequence_new,
                        'end1': end1_new,
                        'end2': end2_new,
                        'total_demand': new_demand,
                        'travel_cost': new_travel_cost,
                        'service_time': new_service_time,
                        'total_duration': new_duration
                    }
                    del routes[best_j]
                    del routes[best_i]
                    routes.append(new_route)
                    num_routes = len(routes)
                for vehicle_index, route in enumerate(routes):
                    total_objective += route['travel_cost']
                    routes_output.append({
                        'depot': depot_ids[depot_idx],
                        'period': day + 1,
                        'vehicle': vehicle_index + 1,
                        'customers': route['sequence']
                    })
            else:
                for vehicle_idx, customer in enumerate(customers_this_day[:m]):
                    cid = customer['id']
                    customer_index = d + cid
                    dist_to_depot = dist_matrix[depot_idx][customer_index]
                    travel_cost = 2 * dist_to_depot
                    total_objective += travel_cost
                    routes_output.append({
                        'depot': depot_ids[depot_idx],
                        'period': day + 1,
                        'vehicle': vehicle_idx + 1,
                        'customers': [cid]
                    })
                for customer in customers_this_day[m:]:
                    if len(customers_this_day) > m:
                        raise RuntimeError("Too many customers for the available vehicles")
        if time_remaining <= 0:
            break
    solution_dict = {
        "objective_value": total_objective,
        "assignments": assignments_output,
        "routes": routes_output
    }
    if logger:
        logger.log_solution(total_objective, solution_dict)
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == '__main__':
    main()