import math
import json
import sys

def main():
    data = json.load(sys.stdin)
    t = data['periods']
    m = data['vehicles_per_depot']
    vehicle_capacity = data['vehicle_capacity']
    route_duration_limit = data['route_duration_limit']
    depots = data['depots']
    customers = data['customers']

    depots_sorted = sorted(depots, key=lambda x: x['id'])
    customers_sorted = sorted(customers, key=lambda x: x['id'])
    d = len(depots_sorted)
    n = len(customers_sorted)
    depot_ids = [depot['id'] for depot in depots_sorted]

    points = []
    for depot in depots_sorted:
        points.append((depot['x'], depot['y']))
    for customer in customers_sorted:
        points.append((customer['x'], customer['y']))
    n_total = len(points)
    dist_matrix = [[0.0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        for j in range(i + 1, n_total):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d_val = math.sqrt(dx * dx + dy * dy)
            dist_matrix[i][j] = d_val
            dist_matrix[j][i] = d_val

    current_demand = [[0] * t for _ in range(d)]
    assignments = {}
    sorted_customers = sorted(customers_sorted, key=lambda c: c['demand'] * len(c['pattern']), reverse=True)
    for customer in sorted_customers:
        best_cost = float('inf')
        best_depot = None
        best_pattern = None
        cid = customer['id']
        c_demand = customer['demand']
        c_index = d + cid
        for depot_idx in range(d):
            for pattern_int in customer['pattern']:
                days = []
                for day in range(t):
                    if pattern_int & (1 << day):
                        days.append(day)
                frequency = len(days)
                cost1 = frequency * dist_matrix[depot_idx][c_index]
                penalty = 0
                for day in days:
                    if current_demand[depot_idx][day] + c_demand > m * vehicle_capacity:
                        penalty += (current_demand[depot_idx][day] + c_demand - m * vehicle_capacity)
                cost = cost1 + 1000 * penalty
                if cost < best_cost:
                    best_cost = cost
                    best_depot = depot_idx
                    best_pattern = pattern_int
        assignments[cid] = (best_depot, best_pattern)
        for day in range(t):
            if best_pattern & (1 << day):
                current_demand[best_depot][day] += customer['demand']

    routes_output = []
    for depot_idx in range(d):
        for day in range(t):
            customers_this_day = []
            for customer in customers_sorted:
                cid = customer['id']
                assigned_depot, pattern_int = assignments[cid]
                if assigned_depot == depot_idx and (pattern_int & (1 << day)):
                    customers_this_day.append(customer)
            if not customers_this_day:
                continue
            routes = []
            for c in customers_this_day:
                cid = c['id']
                c_index = d + cid
                travel_cost = 2 * dist_matrix[depot_idx][c_index]
                service_time = c['service_duration']
                total_duration = travel_cost + service_time
                route = {
                    'sequence': [cid],
                    'end1': cid,
                    'end2': cid,
                    'total_demand': c['demand'],
                    'travel_cost': travel_cost,
                    'service_time': service_time,
                    'total_duration': total_duration
                }
                routes.append(route)
            num_routes = len(routes)
            while num_routes > m:
                best_saving = -10**9
                best_i = None
                best_j = None
                best_connect = None
                for i in range(num_routes):
                    for j in range(i + 1, num_routes):
                        A = routes[i]
                        B = routes[j]
                        a1 = A['end1']
                        a2 = A['end2']
                        b1 = B['end1']
                        b2 = B['end2']
                        a1_idx = d + a1
                        a2_idx = d + a2
                        b1_idx = d + b1
                        b2_idx = d + b2
                        saving1 = dist_matrix[a2_idx][depot_idx] + dist_matrix[depot_idx][b1_idx] - dist_matrix[a2_idx][b1_idx]
                        saving2 = dist_matrix[a2_idx][depot_idx] + dist_matrix[depot_idx][b2_idx] - dist_matrix[a2_idx][b2_idx]
                        saving3 = dist_matrix[a1_idx][depot_idx] + dist_matrix[depot_idx][b1_idx] - dist_matrix[a1_idx][b1_idx]
                        saving4 = dist_matrix[a1_idx][depot_idx] + dist_matrix[depot_idx][b2_idx] - dist_matrix[a1_idx][b2_idx]
                        savings = [saving1, saving2, saving3, saving4]
                        max_saving = max(savings)
                        which = savings.index(max_saving)
                        if A['total_demand'] + B['total_demand'] > vehicle_capacity:
                            continue
                        new_travel_cost = A['travel_cost'] + B['travel_cost'] - max_saving
                        new_service_time = A['service_time'] + B['service_time']
                        new_duration = new_travel_cost + new_service_time
                        if new_duration > route_duration_limit:
                            continue
                        if max_saving > best_saving:
                            best_saving = max_saving
                            best_i = i
                            best_j = j
                            best_connect = which
                if best_saving == -10**9:
                    break
                A = routes[best_i]
                B = routes[best_j]
                if best_connect == 0:
                    new_sequence = A['sequence'] + B['sequence']
                    new_end1 = A['end1']
                    new_end2 = B['end2']
                elif best_connect == 1:
                    new_sequence = A['sequence'] + list(reversed(B['sequence']))
                    new_end1 = A['end1']
                    new_end2 = B['end1']
                elif best_connect == 2:
                    new_sequence = list(reversed(A['sequence'])) + B['sequence']
                    new_end1 = A['end2']
                    new_end2 = B['end2']
                else:
                    new_sequence = list(reversed(A['sequence'])) + list(reversed(B['sequence']))
                    new_end1 = A['end2']
                    new_end2 = B['end1']
                new_travel_cost = A['travel_cost'] + B['travel_cost'] - best_saving
                new_service_time = A['service_time'] + B['service_time']
                new_duration = new_travel_cost + new_service_time
                new_total_demand = A['total_demand'] + B['total_demand']
                new_route = {
                    'sequence': new_sequence,
                    'end1': new_end1,
                    'end2': new_end2,
                    'total_demand': new_total_demand,
                    'travel_cost': new_travel_cost,
                    'service_time': new_service_time,
                    'total_duration': new_duration
                }
                routes.pop(best_j)
                routes.pop(best_i)
                routes.append(new_route)
                num_routes = len(routes)
            for vehicle_index, route in enumerate(routes):
                routes_output.append({
                    "depot": depot_ids[depot_idx],
                    "period": day + 1,
                    "vehicle": vehicle_index,
                    "customers": route['sequence']
                })
    assignments_output = {}
    for customer in customers_sorted:
        cid = customer['id']
        depot_idx, pattern_int = assignments[cid]
        pattern_days = []
        for day in range(t):
            if pattern_int & (1 << day):
                pattern_days.append(day + 1)
        pattern_days.sort()
        assignments_output[str(cid)] = {
            "depot": depot_ids[depot_idx],
            "pattern": pattern_days
        }
    solution = {
        "assignments": assignments_output,
        "routes": routes_output
    }
    json.dump(solution, sys.stdout)

if __name__ == "__main__":
    main()