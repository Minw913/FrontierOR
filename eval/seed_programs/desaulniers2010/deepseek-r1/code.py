import json
import argparse
import time
from collections import defaultdict
from solution_logger import SolutionLogger

def compute_route_cost(route, distance_matrix):
    cost = 0.0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i+1]
        cost += distance_matrix[from_node][to_node]
    return cost

def compute_route_time(route, distance_matrix, customer_by_id, depot_earliest, depot_latest):
    n = len(route)
    times = [0] * n
    times[0] = depot_earliest
    for i in range(1, n):
        from_node = route[i-1]
        to_node = route[i]
        travel_time = distance_matrix[from_node][to_node]
        arrival = times[i-1] + travel_time
        if to_node == 0:
            times[i] = arrival
        else:
            customer = customer_by_id[to_node]
            a, b = customer['time_window']
            if arrival < a:
                start_time = a
            elif arrival > b:
                return None
            else:
                start_time = arrival
            times[i] = start_time
    if route[-1] == 0 and times[-1] > depot_latest:
        return None
    return times

def build_initial_solution(instance, start_time, time_limit, logger):
    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    num_vehicles_upper_bound = instance['num_vehicles_upper_bound']
    depot_info = instance['depot']
    depot_earliest = depot_info['time_window'][0]
    depot_latest = depot_info['time_window'][1]
    customers = instance['customers']
    distance_matrix = instance['distance_matrix']
    
    customer_by_id = {c['id']: c for c in customers}
    remaining_demand = {c['id']: c['demand'] for c in customers}
    total_demand = sum(remaining_demand.values())
    
    routes = []
    total_cost = 0.0
    num_vehicles_used = 0
    remaining_vehicles = num_vehicles_upper_bound
    
    while remaining_vehicles > 0 and total_demand > 0 and time.time() - start_time < time_limit:
        route = [0, 0]
        delivery_amounts = {}
        current_load = 0
        
        improved = True
        while improved:
            if time.time() - start_time >= time_limit:
                break
            improved = False
            best_ratio = float('inf')
            best_customer = None
            best_position = None
            best_amount = 0
            best_new_route = None
            
            for c_id, rem in remaining_demand.items():
                if rem <= 0:
                    continue
                for k in range(1, len(route)):
                    new_route = route[:k] + [c_id] + route[k:]
                    times = compute_route_time(new_route, distance_matrix, customer_by_id, depot_earliest, depot_latest)
                    if times is None:
                        continue
                    amount = min(rem, vehicle_capacity - current_load)
                    if amount <= 0:
                        continue
                    cost_old = compute_route_cost(route, distance_matrix)
                    cost_new = compute_route_cost(new_route, distance_matrix)
                    additional_cost = cost_new - cost_old
                    ratio = additional_cost / amount
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_customer = c_id
                        best_position = k
                        best_amount = amount
                        best_new_route = new_route
            
            if best_customer is not None:
                route = best_new_route
                current_load += best_amount
                delivery_amounts[best_customer] = delivery_amounts.get(best_customer, 0) + best_amount
                remaining_demand[best_customer] -= best_amount
                total_demand -= best_amount
                improved = True
        
        route_cost = compute_route_cost(route, distance_matrix)
        total_cost += route_cost
        num_vehicles_used += 1
        routes.append({
            'nodes': route,
            'deliveries': delivery_amounts,
            'load': current_load,
            'cost': route_cost
        })
        remaining_vehicles -= 1
        
        if total_demand <= 0:
            break
    
    if total_demand > 0:
        return None
    
    return {
        'total_cost': total_cost,
        'num_vehicles_used': num_vehicles_used,
        'routes': routes
    }

def local_search(solution, instance, start_time, time_limit, logger):
    if solution is None:
        return None
    
    customer_by_id = {c['id']: c for c in instance['customers']}
    distance_matrix = instance['distance_matrix']
    depot_earliest = instance['depot']['time_window'][0]
    depot_latest = instance['depot']['time_window'][1]
    vehicle_capacity = instance['vehicle_capacity']
    
    routes = solution['routes']
    total_cost = solution['total_cost']
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        customer_visits = defaultdict(list)
        for idx, route in enumerate(routes):
            for c_id in route['deliveries']:
                amount = route['deliveries'][c_id]
                if amount > 0:
                    customer_visits[c_id].append((idx, amount))
        
        for c_id, visits in customer_visits.items():
            if len(visits) < 2:
                continue
            for visit in visits:
                if time.time() - start_time >= time_limit:
                    break
                route_idx = visit[0]
                amount_to_reassign = visit[1]
                other_visits = [v for v in visits if v[0] != route_idx]
                if not other_visits:
                    continue
                total_spare = 0
                for other in other_visits:
                    other_route_idx = other[0]
                    spare = vehicle_capacity - routes[other_route_idx]['load']
                    total_spare += spare
                if total_spare < amount_to_reassign:
                    continue
                
                route = routes[route_idx]
                if c_id not in route['nodes']:
                    continue
                pos = route['nodes'].index(c_id)
                new_route_nodes = route['nodes'][:pos] + route['nodes'][pos+1:]
                times = compute_route_time(new_route_nodes, distance_matrix, customer_by_id, depot_earliest, depot_latest)
                if times is None:
                    continue
                
                old_segment_cost = distance_matrix[route['nodes'][pos-1]][c_id] + distance_matrix[c_id][route['nodes'][pos+1]]
                new_segment_cost = distance_matrix[route['nodes'][pos-1]][route['nodes'][pos+1]]
                cost_saved = old_segment_cost - new_segment_cost
                new_route_cost = route['cost'] - cost_saved
                
                reassigned = 0
                for other in other_visits:
                    if reassigned >= amount_to_reassign:
                        break
                    other_route_idx = other[0]
                    other_route = routes[other_route_idx]
                    spare = vehicle_capacity - other_route['load']
                    if spare <= 0:
                        continue
                    assign_amount = min(spare, amount_to_reassign - reassigned)
                    other_route['deliveries'][c_id] += assign_amount
                    other_route['load'] += assign_amount
                    reassigned += assign_amount
                
                if reassigned < amount_to_reassign:
                    continue
                
                route['nodes'] = new_route_nodes
                route['deliveries'][c_id] -= amount_to_reassign
                if route['deliveries'][c_id] == 0:
                    del route['deliveries'][c_id]
                route['load'] -= amount_to_reassign
                route['cost'] = new_route_cost
                total_cost -= cost_saved
                
                improved = True
                if logger:
                    logger.log(total_cost)
                break
            if improved:
                break
    
    solution['total_cost'] = total_cost
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    solution = build_initial_solution(instance, start_time, args.time_limit, logger)
    
    if solution is None:
        solution = {
            'objective_value': 0.0,
            'num_vehicles_used': 0,
            'routes': [],
            'deliveries': {}
        }
        for customer in instance['customers']:
            cid = customer['id']
            solution['deliveries'][str(cid)] = {
                'demand': customer['demand'],
                'total_delivered': 0.0
            }
    else:
        solution = local_search(solution, instance, start_time, args.time_limit, logger)
        if solution is None:
            solution = {
                'objective_value': 0.0,
                'num_vehicles_used': 0,
                'routes': [],
                'deliveries': {}
            }
            for customer in instance['customers']:
                cid = customer['id']
                solution['deliveries'][str(cid)] = {
                    'demand': customer['demand'],
                    'total_delivered': 0.0
                }
        else:
            total_delivered = defaultdict(float)
            for customer in instance['customers']:
                cid = customer['id']
                total_delivered[cid] = 0.0
            
            output_routes = []
            for idx, route in enumerate(solution['routes']):
                r = {
                    'vehicle': idx + 1,
                    'route': route['nodes'],
                    'deliveries': {str(cid): amt for cid, amt in route['deliveries'].items()}
                }
                output_routes.append(r)
                for cid, amt in route['deliveries'].items():
                    total_delivered[int(cid)] += amt
            
            deliveries_output = {}
            for customer in instance['customers']:
                cid = customer['id']
                total = total_delivered.get(cid, 0.0)
                deliveries_output[str(cid)] = {
                    'demand': customer['demand'],
                    'total_delivered': total
                }
            
            solution = {
                'objective_value': solution['total_cost'],
                'num_vehicles_used': solution['num_vehicles_used'],
                'routes': output_routes,
                'deliveries': deliveries_output
            }
    
    if logger and 'objective_value' in solution and solution['objective_value'] < float('inf'):
        logger.log(solution['objective_value'])
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()