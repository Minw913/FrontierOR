import json
import argparse
import time
from collections import defaultdict

def main(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    start_time = time.time()

    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_vehicles_available = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    schedule_horizon = instance['schedule_horizon']

    free_edges_set = set()
    for edge in instance['free_edges']:
        i, j = edge
        if i > j:
            i, j = j, i
        free_edges_set.add((i, j))

    edge_sets_dict = {}
    for es in instance['edge_sets']:
        edge_set = set()
        for edge in es['edges']:
            i, j = edge
            if i > j:
                i, j = j, i
            edge_set.add((i, j))
        edge_sets_dict[es['id']] = (es['access_cost'], edge_set)

    depot = None
    customers = []
    node_id_to_node = {}
    for node in instance['nodes']:
        node_id_to_node[node['id']] = node
        if node['id'] == 0:
            depot = node
        else:
            customers.append(node)

    dist = instance['distance_matrix']

    def compute_route_schedule(route, new_customer=None, pos=None):
        if new_customer is not None and pos is not None:
            new_route = route[:pos] + [new_customer] + route[pos:]
        else:
            new_route = route

        total_demand = sum(c['demand'] for c in new_route)
        if total_demand > vehicle_capacity:
            return False, None, None, None

        times = {}
        current_time = 0.0
        total_distance = 0.0
        if new_route:
            first = new_route[0]
            current_time += dist[0][first['id']]
            total_distance += dist[0][first['id']]
            if current_time < first['time_window_start']:
                current_time = first['time_window_start']
            if current_time > first['time_window_end']:
                return False, None, None, None
            times[first['id']] = current_time
            current_time += first['service_time']

            prev = first
            for i in range(1, len(new_route)):
                c = new_route[i]
                current_time += dist[prev['id']][c['id']]
                total_distance += dist[prev['id']][c['id']]
                if current_time < c['time_window_start']:
                    current_time = c['time_window_start']
                if current_time > c['time_window_end']:
                    return False, None, None, None
                times[c['id']] = current_time
                current_time += c['service_time']
                prev = c

            current_time += dist[prev['id']][0]
            total_distance += dist[prev['id']][0]
            if current_time > schedule_horizon:
                return False, None, None, None
        else:
            return True, 0.0, [], {}

        return True, total_distance, new_route, times

    routes = []
    unassigned = customers[:]
    base_route_costs = {}

    while unassigned and (time.time() - start_time) < time_limit:
        best_cost_increase = float('inf')
        best_customer = None
        best_route_index = None
        best_pos = None
        best_new_route = None

        for customer in unassigned:
            feas, cost, new_route, _ = compute_route_schedule([customer])
            if feas and cost < best_cost_increase:
                best_cost_increase = cost
                best_customer = customer
                best_route_index = -1
                best_pos = None
                best_new_route = [customer]

        for route_idx, route in enumerate(routes):
            if route_idx not in base_route_costs:
                feas_base, base_cost, _, _ = compute_route_schedule(route)
                if not feas_base:
                    continue
                base_route_costs[route_idx] = base_cost
            base_cost = base_route_costs[route_idx]

            for pos in range(len(route) + 1):
                for customer in unassigned:
                    feas, new_cost, new_route, _ = compute_route_schedule(route, customer, pos)
                    if feas:
                        cost_increase = new_cost - base_cost
                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_customer = customer
                            best_route_index = route_idx
                            best_pos = pos
                            best_new_route = new_route

        if best_customer is None:
            for customer in unassigned:
                feas, cost, new_route, _ = compute_route_schedule([customer])
                if feas:
                    best_customer = customer
                    best_route_index = -1
                    best_pos = None
                    best_new_route = [customer]
                    best_cost_increase = cost
                    break
            if best_customer is None and unassigned:
                customer = unassigned[0]
                best_customer = customer
                best_route_index = -1
                best_pos = None
                best_new_route = [customer]
                best_cost_increase = dist[0][customer['id']] + dist[customer['id']][0]

        if best_route_index == -1:
            routes.append(best_new_route)
            unassigned.remove(best_customer)
        else:
            routes[best_route_index] = best_new_route
            unassigned.remove(best_customer)
            base_route_costs[best_route_index] = base_route_costs.get(best_route_index, 0) + best_cost_increase

    for customer in unassigned:
        feas, cost, new_route, _ = compute_route_schedule([customer])
        if feas:
            routes.append([customer])
        else:
            routes.append([customer])

    edges_used_set = set()
    for route in routes:
        if not route:
            continue
        edges_used_set.add((0, route[0]['id']))
        for i in range(len(route) - 1):
            a, b = route[i]['id'], route[i+1]['id']
            if a > b:
                a, b = b, a
            edges_used_set.add((a, b))
        edges_used_set.add((route[-1]['id'], 0))

    non_free_edges_used = edges_used_set - free_edges_set

    covered = set()
    selected_set_ids = []
    total_access_cost = 0.0
    universe = non_free_edges_used

    while covered != universe:
        best_ratio = float('inf')
        best_set_id = None
        best_cover = set()
        for set_id, (cost, edge_set) in edge_sets_dict.items():
            if set_id in selected_set_ids:
                continue
            cover = (edge_set & universe) - covered
            if not cover:
                continue
            ratio = cost / len(cover)
            if ratio < best_ratio:
                best_ratio = ratio
                best_set_id = set_id
                best_cover = cover
        if best_set_id is None:
            break
        selected_set_ids.append(best_set_id)
        covered |= best_cover
        total_access_cost += edge_sets_dict[best_set_id][0]

    total_traversal_cost = 0.0
    detailed_routes = []
    for route in routes:
        if not route:
            continue
        total_traversal_cost += dist[0][route[0]['id']]
        for i in range(len(route) - 1):
            total_traversal_cost += dist[route[i]['id']][route[i+1]['id']]
        total_traversal_cost += dist[route[-1]['id']][0]

    objective_value = total_traversal_cost + total_access_cost

    routes_output = []
    for vehicle_id, route in enumerate(routes):
        if not route:
            continue
        sequence = [0]
        arcs = []
        _, _, _, times_dict = compute_route_schedule(route)
        if not times_dict:
            times_dict = {}
        times_dict[0] = 0.0

        first = route[0]
        sequence.append(first['id'])
        arcs.append([0, first['id']])

        prev = first
        for i in range(1, len(route)):
            c = route[i]
            sequence.append(c['id'])
            arcs.append([prev['id'], c['id']])
            prev = c
        sequence.append(0)
        arcs.append([prev['id'], 0])

        all_visit_times = {}
        for node in instance['nodes']:
            node_id = node['id']
            if node_id in times_dict:
                all_visit_times[str(node_id)] = times_dict[node_id]
            else:
                all_visit_times[str(node_id)] = node['time_window_start']

        routes_output.append({
            "vehicle": vehicle_id,
            "sequence": sequence,
            "arcs": arcs,
            "visit_times": all_visit_times
        })

    solution = {
        "objective_value": objective_value,
        "routes": routes_output,
        "edge_sets_used": selected_set_ids
    }

    if logger:
        logger.log(objective_value)

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)