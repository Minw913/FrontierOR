import json
import math
import time
import sys
from copy import deepcopy
import argparse
from solution_logger import SolutionLogger  # Provided utility

# We'll use this function to get customer info or depot info for a node id
def get_node_info(node_id, data):
    if node_id == 0:
        return data['depot']
    else:
        idx = node_id - 1
        return data['customers'][idx]

def build_effective_time_matrix(data):
    n_nodes = data['num_customers'] + 1
    effective_time = [[0.0]*n_nodes for _ in range(n_nodes)]
    dist_mat = data['distance_matrix']
    travel_includes_service = data['travel_time_includes_service_time']

    # Precomputed due_date and ready_time for nodes (for depots and customers)
    due_date = [0] * n_nodes
    ready_time = [0] * n_nodes
    due_date[0] = data['depot']['due_date']
    ready_time[0] = data['depot']['ready_time']
    service_time = [0] * n_nodes
    for i, cust in enumerate(data['customers']):
        node_id = i+1
        due_date[node_id] = cust['due_date']
        ready_time[node_id] = cust['ready_time']
        service_time[node_id] = cust['service_time']

    for i in range(n_nodes):
        for j in range(n_nodes):
            base = dist_mat[i][j]
            adjust_base = base
            if i > 0 and not travel_includes_service:
                adjust_base += service_time[i]
            diff = ready_time[j] - due_date[i]
            effective_time[i][j] = max(adjust_base, diff)
    return effective_time

def simulate_route(route, effective_time, data):
    if not route:
        return None, None, True
    n_nodes = len(route)
    depart_times = [0] * n_nodes
    loads = [0] * n_nodes
    vehicle_cap = data['vehicle_capacity']
    # first node (depot)
    depart_times[0] = data['depot']['ready_time']
    loads[0] = 0

    for i in range(1, n_nodes):
        prev_id = route[i-1]
        curr_id = route[i]
        # Get from the travel time matrix
        travel = effective_time[prev_id][curr_id]
        arrival_curr = depart_times[i-1] + travel
        node_info = get_node_info(curr_id, data)
        ready_curr = node_info['ready_time']
        due_date_curr = node_info['due_date']
        depart_curr = max(arrival_curr, ready_curr)
        if depart_curr > due_date_curr:
            return [], [], False
        new_load = loads[i-1] + node_info['demand'] if curr_id != 0 else 0
        if new_load > vehicle_cap and curr_id != 0:  # depot has zero demand
            return [], [], False
        depart_times[i] = depart_curr
        loads[i] = new_load
    arrival_at_final_depot = depart_times[-1]  # last node's departure, which for depot is already the arrival time
    if route[-1] == 0:  # the last node is depot
        if arrival_at_final_depot > data['depot']['due_date']:
            return [], [], False
    return depart_times, loads, True

def compute_total_distance(route, data):
    if len(route) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(route)-1):
        from_node = route[i]
        to_node = route[i+1]
        total += data['distance_matrix'][from_node][to_node]
    return total

def best_insertion(route, unassigned, effective_time, data):
    if not route:
        return None, None, float('inf'), None, None
    vehicle_cap = data['vehicle_capacity']
    best_delta = float('inf')
    best_pos = -1
    best_cust = None
    best_state_dep = []
    best_state_load = []

    for cust_id in unassigned:
        cust_info = next(c for c in data['customers'] if c['id'] == cust_id)
        for pos in range(1, len(route)):
            prev_id = route[pos-1]
            next_id = route[pos]
            # Create new_route with cust_id inserted at 'pos'
            new_route = route[:pos] + [cust_id] + route[pos:]
            depart_times, loads, feasible = simulate_route(new_route, effective_time, data)
            if not feasible:
                continue
            # Compute increase in distance (delta)
            current_dist = compute_total_distance(route, data)
            new_dist = compute_total_distance(new_route, data)
            delta = new_dist - current_dist
            if delta < best_delta:
                best_delta = delta
                best_cust = cust_id
                best_pos = pos
                best_state_dep = depart_times
                best_state_load = loads
    return best_cust, best_pos, best_delta, best_state_dep, best_state_load

def construct_initial_routes(data, effective_time, time_limit):
    start_time = time.time()
    unassigned = [cust['id'] for cust in data['customers']]
    routes = []
    state_per_route = []  # Each is a state: for the route, we store the state at a higher level later?
    # Sort unassigned by due_date and then by demand (descending) to have critical ones first
    customers_info = {cust['id']: cust for cust in data['customers']}
    unassigned.sort(key=lambda cid: (customers_info[cid]['due_date'], -customers_info[cid]['demand']))
    while unassigned and (time.time() - start_time) < time_limit:
        route = [0, 0]  # start at depot0 and end at depot0
        state_dep, state_load, feasible = simulate_route(route, effective_time, data)
        # This base route is always feasible? 
        if not feasible:
            routes.append(route)
            state_per_route.append((state_dep, state_load))
            break
        inserted = True
        # Initially, the total travel distance for the empty route
        current_route = route
        while unassigned and inserted and (time.time() - start_time) < time_limit:
            best_cust, best_pos, best_delta, n_state_dep, n_state_load = best_insertion(current_route, unassigned, effective_time, data)
            if best_cust is None:
                inserted = False
            else:
                # Apply the insertion
                new_route = current_route[:best_pos] + [best_cust] + current_route[best_pos:]
                unassigned.remove(best_cust)
                current_route = new_route
                state_dep, state_load = n_state_dep, n_state_load
        # End of the route: close
        routes.append(current_route)
        state_per_route.append((state_dep, state_load))
    # There might be leftover unassigned? then we have to create additional routes? but we already do one route per unassigned? 
    # However, we break by unassigned.
    return routes, unassigned, state_per_route

def try_reduce_vehicles(routes, effective_time, data, time_left, start_total_time):
    num_v = len(routes)
    if num_v <= 1:
        return routes, False  # cannot reduce
    # Try remove the smallest route and reassign its customers
    customer_count_per_route = [len(route) - 2 for route in routes]  # without depots
    min_route_idx = customer_count_per_route.index(min(customer_count_per_route))
    # Also get the one with the fewest customer if tie by distance
    removed_route = routes.pop(min_route_idx)
    customers_to_reassign = [node for node in removed_route if node != 0]
    # Now try to insert these customers into the remaining routes
    for cust in customers_to_reassign:
        feasible_found = False
        for idx, route in enumerate(routes):
            new_route = route[:] 
            depart_times, loads, feasible = simulate_route(route, effective_time, data)
            if not feasible:
                continue
            # We try all positions in the route? but we only know current state, we have to try insert this customer
            for pos in range(1, len(new_route)):
                trial_route = new_route[0:pos] + [cust] + new_route[pos:]
                _, _, feasible = simulate_route(trial_route, effective_time, data)
                if feasible:
                    routes[idx] = trial_route
                    feasible_found = True
                    break
            if time.time() - start_total_time > time_left:
                return None, False
            if feasible_found:
                break
        if not feasible_found:
            # Cannot insert customer `cust` -> reduction fails
            routes.insert(min_route_idx, removed_route)
            return routes, False
    return routes, True

def prepare_output(routes, data):
    num_vehicles = len(routes)
    objective_value = num_vehicles
    total_distance = 0.0
    dept_times_dict = {}
    loads_dict = {}
    effective_time = build_effective_time_matrix(data)  # rebuild to ensure consistency? or pass it
    for route in routes:
        dep_times, loads, valid = simulate_route(route, effective_time, data)
        if not valid:
            continue
        total_distance += compute_total_distance(route, data)
        for i, node_id in enumerate(route):
            if node_id == 0:
                continue
            dept_times_dict[str(node_id)] = dep_times[i]
            loads_dict[str(node_id)] = loads[i]
    solution_dict = {
        "objective_value": float(objective_value),
        "num_vehicles": num_vehicles,
        "routes": routes,
        "total_distance": round(total_distance, 1),
        "departure_times": dept_times_dict,
        "loads": loads_dict
    }
    return solution_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    with open(args.instance_path) as f:
        data = json.load(f)

    effective_time = build_effective_time_matrix(data)

    unassigned = [cust['id'] for cust in data['customers']] 
    initial_routes, unassigned, _ = construct_initial_routes(data, effective_time, args.time_limit - (time.time()-start_time))
    
    if unassigned:
        # Create additional routes for unassigned?
        for cust in unassigned:
            initial_routes.append([0, cust, 0])
    
    solution_dict = prepare_output(initial_routes, data)
    incum_value = solution_dict['num_vehicles']  # primary objective, the number of vehicles
    if logger:
        logger.log_solution(incum_value, solution_dict)

    # Now, try to reduce the number of vehicles
    current_routes = initial_routes
    while (time.time() - start_time) < args.time_limit:
        next_routes, reduced = try_reduce_vehicles(
            current_routes, 
            effective_time,
            data, 
            args.time_limit - (time.time()-start_time), 
            start_time
        )
        if not reduced:
            break
        current_routes = next_routes
        new_sol = prepare_output(current_routes, data)
        new_value = new_sol['num_vehicles']
        if new_value < incum_value:
            solution_dict = new_sol
            incum_value = new_value
            if logger:
                logger.log_solution(incum_value, solution_dict)
        else:
            break

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=0)

if __name__ == '__main__':
    main()