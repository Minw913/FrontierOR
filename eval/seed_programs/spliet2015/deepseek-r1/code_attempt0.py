import argparse
import json
import time
from solution_logger import SolutionLogger

def compute_route_cost(nodes, distance_matrix):
    cost = 0
    for i in range(len(nodes) - 1):
        cost += distance_matrix[nodes[i]][nodes[i+1]]
    return cost

def is_route_feasible(nodes, distance_matrix, time_windows, customer_widths, depot_start, depot_end, end_depot_id):
    current_time = depot_start
    for j in range(1, len(nodes)):
        prev = nodes[j-1]
        curr = nodes[j]
        current_time += distance_matrix[prev][curr]
        if curr == end_depot_id:
            if current_time > depot_end:
                return False
        else:
            a_i = time_windows[curr][0]
            L_i = customer_widths[curr]
            if current_time > a_i + L_i:
                return False
            current_time = max(current_time, a_i)
    return True

def compute_route_schedule(nodes, distance_matrix, time_windows, customer_widths, depot_start, depot_end, end_depot_id):
    arrival_times = [depot_start]
    current_time = depot_start
    for j in range(1, len(nodes)):
        prev = nodes[j-1]
        curr = nodes[j]
        current_time += distance_matrix[prev][curr]
        arrival_times.append(current_time)
        if curr != end_depot_id and curr != 0:
            a_i = time_windows[curr][0]
            current_time = max(current_time, a_i)
    return arrival_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None
    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_customers']
    depot_info = data['depot']
    D_start = depot_info['exogenous_time_window'][0]
    D_end = depot_info['exogenous_time_window'][1]
    vehicle_capacity = data['vehicle_capacity']
    global_endogenous_width = data.get('endogenous_time_window_width', None)
    distance_matrix = data['distance_matrix']
    end_depot_id = n + 1

    customers = data['customers']
    customer_ids = [c['id'] for c in customers]
    customer_exogenous = {c['id']: c['exogenous_time_window'] for c in customers}
    customer_endogenous_widths = {}
    for c in customers:
        if 'endogenous_time_window_width' in c:
            customer_endogenous_widths[c['id']] = c['endogenous_time_window_width']
        else:
            customer_endogenous_widths[c['id']] = global_endogenous_width

    scenarios = data['scenarios']
    scenario_dict = {s['scenario_id']: (s['probability'], s['demands']) for s in scenarios}

    assignments = []
    assignment1 = {}
    assignment2 = {}
    assignment3 = {}
    for c in customers:
        cid = c['id']
        exo_start = c['exogenous_time_window'][0]
        exo_end = c['exogenous_time_window'][1]
        L_i = customer_endogenous_widths[cid]
        assignment1[cid] = exo_start
        assignment2[cid] = exo_end - L_i
        assignment3[cid] = (exo_start + exo_end - L_i) / 2.0
    assignments = [assignment1, assignment2, assignment3]
    assignment_names = ['earliest', 'latest', 'centered']

    best_objective = float('inf')
    best_solution = None
    best_assignment_idx = -1

    for idx, assignment_dict in enumerate(assignments):
        if time.time() - start_time > args.time_limit:
            break

        time_windows_dict = {}
        for cid in customer_ids:
            start_time_win = assignment_dict[cid]
            time_windows_dict[cid] = [start_time_win, start_time_win + customer_endogenous_widths[cid]]

        scenario_routes = {}
        total_expected_cost = 0.0
        feasible_all_scenarios = True

        for scenario_id, (prob, demands) in scenario_dict.items():
            if time.time() - start_time > args.time_limit:
                feasible_all_scenarios = False
                break

            routes = []
            unassigned = customer_ids[:]
            demand_dict = {cid: d for cid, d in zip(customer_ids, demands)}
            sorted_unassigned = sorted(unassigned, key=lambda cid: assignment_dict[cid])

            while sorted_unassigned:
                if time.time() - start_time > args.time_limit:
                    break
                i = sorted_unassigned.pop(0)
                i_demand = demand_dict[i]
                best_cost_increase = float('inf')
                best_route = None
                best_position = None
                best_new_nodes = None
                found = False

                for route in routes:
                    if route['total_demand'] + i_demand > vehicle_capacity:
                        continue
                    for pos in range(1, len(route['nodes'])):
                        new_nodes = route['nodes'][:]
                        new_nodes.insert(pos, i)
                        new_cost = compute_route_cost(new_nodes, distance_matrix)
                        if is_route_feasible(new_nodes, distance_matrix, assignment_dict, customer_endogenous_widths, D_start, D_end, end_depot_id):
                            cost_increase = new_cost - route['cost']
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_route = route
                                best_position = pos
                                best_new_nodes = new_nodes
                                found = True

                if found:
                    new_cost = compute_route_cost(best_new_nodes, distance_matrix)
                    best_route['nodes'] = best_new_nodes
                    best_route['total_demand'] += i_demand
                    best_route['cost'] = new_cost
                else:
                    new_nodes = [0, i, end_depot_id]
                    new_cost = compute_route_cost(new_nodes, distance_matrix)
                    if is_route_feasible(new_nodes, distance_matrix, assignment_dict, customer_endogenous_widths, D_start, D_end, end_depot_id):
                        routes.append({
                            'nodes': new_nodes,
                            'total_demand': i_demand,
                            'cost': new_cost
                        })
                    else:
                        feasible_all_scenarios = False
                        break

            if not feasible_all_scenarios or time.time() - start_time > args.time_limit:
                feasible_all_scenarios = False
                break
            else:
                scenario_cost = sum(route['cost'] for route in routes)
                total_expected_cost += prob * scenario_cost
                scenario_routes[scenario_id] = routes

        if not feasible_all_scenarios:
            continue

        if total_expected_cost < best_objective:
            best_objective = total_expected_cost
            best_solution = (time_windows_dict, scenario_routes)
            best_assignment_idx = idx
            if logger:
                logger.log(best_objective)

    if best_solution is None:
        time_windows_dict = {cid: [assignment1[cid], assignment1[cid] + customer_endogenous_widths[cid]] for cid in customer_ids}
        scenario_routes = {}
        total_expected_cost = 0.0
        for scenario_id, (prob, demands) in scenario_dict.items():
            demand_dict = {cid: d for cid, d in zip(customer_ids, demands)}
            routes = []
            for cid in customer_ids:
                new_nodes = [0, cid, end_depot_id]
                new_cost = compute_route_cost(new_nodes, distance_matrix)
                routes.append({
                    'nodes': new_nodes,
                    'total_demand': demand_dict[cid],
                    'cost': new_cost
                })
            scenario_cost = sum(route['cost'] for route in routes)
            total_expected_cost += prob * scenario_cost
            scenario_routes[scenario_id] = routes
        best_solution = (time_windows_dict, scenario_routes)
        best_objective = total_expected_cost

    time_windows_dict, scenario_routes_dict = best_solution
    output_time_windows = {str(cid): [tw[0], tw[1]] for cid, tw in time_windows_dict.items()}

    output_routes = {}
    for scenario_id, routes in scenario_routes_dict.items():
        scenario_list = []
        for route in routes:
            schedule = compute_route_schedule(
                route['nodes'], distance_matrix, 
                {cid: tw for cid, tw in time_windows_dict.items()}, 
                customer_endogenous_widths, D_start, D_end, end_depot_id
            )
            arrival_times_dict = {str(node): schedule[i] for i, node in enumerate(route['nodes'])}
            scenario_list.append({
                'route': route['nodes'],
                'arrival_times': arrival_times_dict,
                'cost': route['cost']
            })
        output_routes[str(scenario_id)] = scenario_list

    output = {
        "objective_value": best_objective,
        "time_windows": output_time_windows,
        "routes": output_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    main()