import argparse
import json
import math
import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

def parse_arguments():
    parser = argparse.ArgumentParser(description='Solve the Cyclic Inventory Routing Problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log file for intermediate solutions.')
    return parser.parse_args()

def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def compute_distance_matrix(depot, customers):
    locations = []
    locations.append((depot['x'], depot['y']))
    customer_id_to_index = {}
    for idx, c in enumerate(customers):
        customer_id_to_index[c['id']] = idx + 1
        locations.append((c['x'], c['y']))
    
    n = len(locations)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = locations[i]
        for j in range(n):
            x2, y2 = locations[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = math.sqrt(dx*dx + dy*dy)
            distance_matrix[i][j] = dist
    return distance_matrix, customer_id_to_index

def schedule_selection_mip(instance, distance_matrix, customer_id_to_index, time_limit):
    num_days = instance['num_days']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    num_customers = len(customers)
    
    model = gp.Model('ScheduleSelection')
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(1, time_limit))
    
    y = {}
    for c in customers:
        c_id = c['id']
        c_idx = customer_id_to_index[c_id]
        dist_to_c = distance_matrix[0][c_idx]
        schedules = c['schedules']
        y[c_id] = {}
        for s_idx, s in enumerate(schedules):
            y_var = model.addVar(vtype=GRB.BINARY, name=f'y_{c_id}_{s_idx}')
            y[c_id][s_idx] = y_var
            visit_freq = s['visit_frequency']
            model.addConstr(y_var * dist_to_c * 2 * visit_freq <= 1e9)
    
    for c in customers:
        c_id = c['id']
        model.addConstr(gp.quicksum(y[c_id][s_idx] for s_idx in range(len(c['schedules']))) == 1)
    
    for d in range(num_days):
        total_demand = 0.0
        for c in customers:
            c_id = c['id']
            for s_idx, s in enumerate(c['schedules']):
                demand_d = 0
                for part in s['parts']:
                    if part['visit_day'] == d:
                        demand_d = part['demand_per_visit']
                        break
                total_demand += demand_d * y[c_id][s_idx]
        model.addConstr(total_demand <= num_vehicles * vehicle_capacity)
    
    obj = gp.quicksum(
        y[c['id']][s_idx] * 2 * distance_matrix[0][customer_id_to_index[c['id']]] * s['visit_frequency']
        for c in customers for s_idx, s in enumerate(c['schedules'])
    )
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    chosen_schedules = {}
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        for c in customers:
            c_id = c['id']
            for s_idx in range(len(c['schedules'])):
                if y[c_id][s_idx].X > 0.5:
                    chosen_schedules[c_id] = s_idx
                    break
    return chosen_schedules

def check_route_feasibility(customer_route, instance, distance_matrix, customer_id_to_index, return_cost=False):
    depot_tw = instance['depot']['time_window']
    a_depot, b_depot = depot_tw[0], depot_tw[1]
    customers_data = {c['id']: c for c in instance['customers']}
    
    if not customer_route:
        return (True, 0.0) if return_cost else True
    
    current_time = 0.0
    locations = [0]
    for c_id in customer_route:
        c = customers_data[c_id]
        c_idx = customer_id_to_index[c_id]
        prev_idx = locations[-1]
        dist = distance_matrix[prev_idx][c_idx]
        current_time += dist
        a_i, b_i = c['time_window']
        if current_time > b_i + 1e-5:
            return (False, None) if return_cost else False
        if current_time < a_i:
            current_time = a_i
        current_time += c['service_time']
        locations.append(c_idx)
    
    last_c_idx = locations[-1]
    dist_back = distance_matrix[last_c_idx][0]
    current_time += dist_back
    
    if current_time > b_depot + 1e-5:
        return (False, None) if return_cost else False
    if return_cost:
        cost = 0.0
        locs = [0] + [customer_id_to_index[c_id] for c_id in customer_route] + [0]
        for i in range(len(locs)-1):
            cost += distance_matrix[locs[i]][locs[i+1]]
        return True, cost
    return True

def two_opt_swap(route, i, j):
    return route[:i] + list(reversed(route[i:j+1])) + route[j+1:]

def solve_vrp_day(visits, instance, distance_matrix, customer_id_to_index, num_vehicles, vehicle_capacity, max_route_duration, depot_tw):
    if not visits:
        return [], 0.0
    
    sorted_visits = sorted(visits, key=lambda v: v['time_window'][1])
    routes = []
    route_demands = []
    total_cost = 0.0
    
    for v in sorted_visits:
        c_id = v['customer_id']
        best_cost_increase = float('inf')
        best_route_idx = None
        best_position = None
        best_new_route = None
        
        for ridx, route in enumerate(routes):
            current_demand = route_demands[ridx]
            if current_demand + v['demand_per_visit'] > vehicle_capacity:
                continue
            route_customer_ids = [visit['customer_id'] for visit in route]
            if c_id in route_customer_ids:
                continue
            for pos in range(0, len(route)+1):
                candidate_route = route[:pos] + [v] + route[pos:]
                candidate_route_customer_ids = [visit['customer_id'] for visit in candidate_route]
                feasible, cost = check_route_feasibility(candidate_route_customer_ids, instance, distance_matrix, customer_id_to_index, return_cost=True)
                if not feasible:
                    continue
                if cost < best_cost_increase:
                    best_cost_increase = cost
                    best_route_idx = ridx
                    best_position = pos
                    best_new_route = candidate_route
        
        if best_route_idx is not None:
            routes[best_route_idx] = best_new_route
            route_demands[best_route_idx] += v['demand_per_visit']
        else:
            if len(routes) < num_vehicles:
                candidate_route = [v]
                candidate_route_customer_ids = [v['customer_id']]
                feasible, cost = check_route_feasibility(candidate_route_customer_ids, instance, distance_matrix, customer_id_to_index, return_cost=True)
                if feasible:
                    routes.append(candidate_route)
                    route_demands.append(v['demand_per_visit'])
                else:
                    return None, None
            else:
                return None, None
    
    routes_with_costs = []
    total_cost = 0.0
    for route in routes:
        current_route = route
        current_route_customer_ids = [v['customer_id'] for v in current_route]
        current_cost = 0.0
        locs = [0] + [customer_id_to_index[v['customer_id']] for v in current_route] + [0]
        for i in range(len(locs)-1):
            current_cost += distance_matrix[locs[i]][locs[i+1]]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route)-1):
                for j in range(i+1, len(route)):
                    new_route = two_opt_swap(route, i, j)
                    new_route_customer_ids = [v['customer_id'] for v in new_route]
                    new_cost = 0.0
                    new_locs = [0] + [customer_id_to_index[v['customer_id']] for v in new_route] + [0]
                    for k in range(len(new_locs)-1):
                        new_cost += distance_matrix[new_locs[k]][new_locs[k+1]]
                    if new_cost < current_cost:
                        feasible, _ = check_route_feasibility(new_route_customer_ids, instance, distance_matrix, customer_id_to_index, return_cost=True)
                        if feasible:
                            route = new_route
                            current_route = new_route
                            current_cost = new_cost
                            improved = True
        routes_with_costs.append( (route, current_cost) )
        total_cost += current_cost
    
    return routes_with_costs, total_cost

def build_solution(instance, chosen_schedules, day_routes, total_cost):
    schedules_output = {}
    for c in instance['customers']:
        c_id = c['id']
        s_idx = chosen_schedules.get(c_id, 0)
        s = c['schedules'][s_idx]
        schedules_output[str(c_id)] = {
            "schedule_index": s_idx,
            "days": s['days'],
            "visit_frequency": s['visit_frequency']
        }
    
    routes_output = {}
    for day in range(instance['num_days']):
        routes_day = []
        if day in day_routes:
            for route_visits, route_cost in day_routes[day]:
                if not route_visits:
                    continue
                demand = 0
                parts_dict = {}
                customer_ids = []
                for visit in route_visits:
                    c_id = visit['customer_id']
                    customer_ids.append(c_id)
                    demand += visit['demand_per_visit']
                    parts_dict[str(c_id)] = {
                        "visit_day": day,
                        "days_covered": visit['days_covered'],
                        "demand_per_visit": visit['demand_per_visit']
                    }
                route_dict = {
                    "customers": customer_ids,
                    "cost": route_cost,
                    "demand": demand,
                    "parts": parts_dict
                }
                routes_day.append(route_dict)
            routes_output[str(day)] = routes_day
        else:
            routes_output[str(day)] = []
    
    return {
        "objective_value": total_cost,
        "routes": routes_output,
        "schedules": schedules_output
    }

def main():
    args = parse_arguments()
    start_time = time.time()
    time_limit = args.time_limit
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    instance = load_instance(args.instance_path)
    distance_matrix, customer_id_to_index = compute_distance_matrix(instance['depot'], instance['customers'])
    
    time_used = time.time() - start_time
    time_left = max(1, time_limit - time_used)
    chosen_schedules = schedule_selection_mip(instance, distance_matrix, customer_id_to_index, time_left)
    
    if not chosen_schedules:
        chosen_schedules = {}
        for c in instance['customers']:
            chosen_schedules[c['id']] = 0
    
    visits_per_day = defaultdict(list)
    for c in instance['customers']:
        c_id = c['id']
        s_idx = chosen_schedules[c_id]
        s = c['schedules'][s_idx]
        for part in s['parts']:
            day = part['visit_day']
            visits_per_day[day].append({
                'customer_id': c_id,
                'demand_per_visit': part['demand_per_visit'],
                'days_covered': part['days_covered'],
                'time_window': c['time_window'],
                'service_time': c['service_time']
            })
    
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    max_route_duration = instance['max_route_duration']
    depot_tw = instance['depot']['time_window']
    
    day_routes = {}
    total_cost = 0.0
    for day in range(instance['num_days']):
        if day in visits_per_day:
            visits = visits_per_day[day]
            routes_with_costs, cost = solve_vrp_day(visits, instance, distance_matrix, customer_id_to_index, num_vehicles, vehicle_capacity, max_route_duration, depot_tw)
            if routes_with_costs is not None:
                day_routes[day] = routes_with_costs
                total_cost += cost
            else:
                day_routes[day] = []
        else:
            day_routes[day] = []
    
    if logger:
        logger.log(total_cost)
    
    solution = build_solution(instance, chosen_schedules, day_routes, total_cost)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()