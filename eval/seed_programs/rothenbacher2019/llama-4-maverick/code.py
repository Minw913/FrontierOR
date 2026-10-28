import argparse
import json
import math
from gurobipy import Model, GRB, quicksum

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def solve(instance_path, time_limit, log_path=None):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_customers = instance['num_customers']
    num_days = instance['num_days']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    max_route_duration = instance['max_route_duration']
    depot = instance['depot']
    customers = instance['customers']
    planning_horizon_cyclic = instance['planning_horizon_cyclic']

    depot_x, depot_y = depot['x'], depot['y']
    depot_time_window = depot['time_window']
    depot_service_time = depot['service_time']

    customer_locations = [(customer['x'], customer['y']) for customer in customers]
    customer_time_windows = [customer['time_window'] for customer in customers]
    customer_service_times = [customer['service_time'] for customer in customers]
    customer_demands = [customer['demand_per_day'] for customer in customers]

    distance_matrix = [[0 for _ in range(num_customers+1)] for _ in range(num_customers+1)]
    for i in range(num_customers+1):
        for j in range(num_customers+1):
            if i == 0:
                x1, y1 = depot_x, depot_y
            else:
                x1, y1 = customer_locations[i-1]
            if j == 0:
                x2, y2 = depot_x, depot_y
            else:
                x2, y2 = customer_locations[j-1]
            distance_matrix[i][j] = distance(x1, y1, x2, y2)

    model = Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    schedule_selection = {}
    for c in range(num_customers):
        customer = customers[c]
        num_schedules = len(customer['schedules'])
        schedule_selection[c] = model.addVar(vtype=GRB.INTEGER, name=f'schedule_selection_{c}')
        model.addConstr(schedule_selection[c] >= 0)
        model.addConstr(schedule_selection[c] <= num_schedules - 1)
        model.addConstr(schedule_selection[c] == round(schedule_selection[c]))

    visit_vars = {}
    for d in range(num_days):
        for v in range(num_vehicles):
            for c in range(num_customers):
                customer = customers[c]
                for s in range(len(customer['schedules'])):
                    for p in customer['schedules'][s]['parts']:
                        if p['visit_day'] == d:
                            if (d, v, c, s) not in visit_vars:
                                visit_vars[(d, v, c, s)] = model.addVar(vtype=GRB.BINARY, name=f'visit_{d}_{v}_{c}_{s}')

    # Simplified objective function
    obj = 0
    for d in range(num_days):
        for v in range(num_vehicles):
            for c in range(num_customers):
                customer = customers[c]
                for s in range(len(customer['schedules'])):
                    for p in customer['schedules'][s]['parts']:
                        if p['visit_day'] == d and (d, v, c, s) in visit_vars:
                            obj += distance_matrix[0][c+1] * visit_vars[(d, v, c, s)]
                            obj += distance_matrix[c+1][0] * visit_vars[(d, v, c, s)]
    model.setObjective(obj, GRB.MINIMIZE)

    for c in range(num_customers):
        customer = customers[c]
        model.addConstr(quicksum(visit_vars.get((d, v, c, s), 0) for d in range(num_days) for v in range(num_vehicles) for s in range(len(customer['schedules']))) == 
                        quicksum([schedule['visit_frequency'] for schedule in customer['schedules']][int(schedule_selection[c].x)]))

    for d in range(num_days):
        for v in range(num_vehicles):
            model.addConstr(quicksum(customers[c]['demand_per_day'] * p['days_covered'] * visit_vars.get((d, v, c, s), 0) for c in range(num_customers) for s in range(len(customers[c]['schedules'])) for p in customers[c]['schedules'][s]['parts'] if p['visit_day'] == d) <= vehicle_capacity)

    model.optimize()

    solution = {'routes': {}, 'schedules': {}}
    for d in range(num_days):
        solution['routes'][str(d)] = []
        for v in range(num_vehicles):
            route = {'customers': [], 'cost': 0, 'demand': 0, 'parts': {}}
            for c in range(num_customers):
                for s in range(len(customers[c]['schedules'])):
                    if (d, v, c, s) in visit_vars and visit_vars[(d, v, c, s)].x > 0.5:
                        route['customers'].append(customers[c]['id'])
                        for p in customers[c]['schedules'][s]['parts']:
                            if p['visit_day'] == d:
                                route['demand'] += p['demand_per_visit']
                                route['parts'][str(customers[c]['id'])] = {'visit_day': p['visit_day'], 'days_covered': p['days_covered'], 'demand_per_visit': p['demand_per_visit']}
            if route['customers']:
                solution['routes'][str(d)].append(route)

    for c in range(num_customers):
        customer = customers[c]
        if schedule_selection[c].x is not None and schedule_selection[c].x >= 0 and schedule_selection[c].x < len(customer['schedules']):
            selected_schedule_index = int(schedule_selection[c].x)
            solution['schedules'][str(customer['id'])] = {
                'schedule_index': selected_schedule_index,
                'days': customer['schedules'][selected_schedule_index]['days'],
                'visit_frequency': customer['schedules'][selected_schedule_index]['visit_frequency']
            }

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution['objective_value'] = model.ObjVal
    else:
        solution['objective_value'] = None

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(args.instance_path, args.time_limit, args.log_path)
    if logger and solution['objective_value'] is not None:
        logger.log(solution['objective_value'])
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()