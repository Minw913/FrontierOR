import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

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

    # Precompute distance matrix
    distance_matrix = [[distance(depot_x, depot_y, customer_locations[i][0], customer_locations[i][1]) for i in range(num_customers)]]
    for i in range(num_customers):
        row = [distance(customer_locations[i][0], customer_locations[i][1], customer_locations[j][0], customer_locations[j][1]) for j in range(num_customers)]
        row.insert(0, distance(customer_locations[i][0], customer_locations[i][1], depot_x, depot_y))
        distance_matrix.append(row)

    # Initialize Gurobi model
    model = Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Decision variables for schedule selection
    schedule_selection = {}
    for c in range(num_customers):
        customer = customers[c]
        num_schedules = len(customer['schedules'])
        schedule_selection[c] = model.addVar(vtype=GRB.INTEGER, name=f'schedule_selection_{c}')
        model.addConstr(schedule_selection[c] >= 0)
        model.addConstr(schedule_selection[c] <= num_schedules - 1)
        model.addConstr(schedule_selection[c] == round(schedule_selection[c]))

    # Decision variables for routing
    visit = {}
    for d in range(num_days):
        for v in range(num_vehicles):
            for c in range(num_customers):
                customer = customers[c]
                for p in customer['schedules'][int(schedule_selection[c].getValue())]['parts']:
                    if p['visit_day'] == d:
                        visit[(d, v, c)] = model.addVar(vtype=GRB.BINARY, name=f'visit_{d}_{v}_{c}')

    # Objective function
    model.setObjective(quicksum(distance_matrix[0][c] * visit[(d, v, c)] for d in range(num_days) for v in range(num_vehicles) for c in range(num_customers) if (d, v, c) in visit) + 
                       quicksum(distance_matrix[c+1][c2+1] * visit[(d, v, c)] * visit[(d, v, c2)] for d in range(num_days) for v in range(num_vehicles) for c in range(num_customers) for c2 in range(num_customers) if (d, v, c) in visit and (d, v, c2) in visit), GRB.MINIMIZE)

    # Constraints
    # Each customer is visited according to its selected schedule
    for c in range(num_customers):
        customer = customers[c]
        selected_schedule = schedule_selection[c]
        for s in range(len(customer['schedules'])):
            schedule = customer['schedules'][s]
            model.addConstr(quicksum(visit[(d, v, c)] for d in range(num_days) for v in range(num_vehicles) if (d, v, c) in visit) == schedule['visit_frequency'] * (selected_schedule == s))

    # Vehicle capacity constraint
    for d in range(num_days):
        for v in range(num_vehicles):
            model.addConstr(quicksum(customers[c]['demand_per_day'] * p['days_covered'] * visit[(d, v, c)] for c in range(num_customers) for p in customers[c]['schedules'][int(schedule_selection[c].getValue())]['parts'] if p['visit_day'] == d and (d, v, c) in visit) <= vehicle_capacity)

    # Route duration constraint
    for d in range(num_days):
        for v in range(num_vehicles):
            route_duration = quicksum(distance_matrix[0][c] * visit[(d, v, c)] for c in range(num_customers) if (d, v, c) in visit) + \
                             quicksum(distance_matrix[c+1][c2+1] * visit[(d, v, c)] * visit[(d, v, c2)] for c in range(num_customers) for c2 in range(num_customers) if (d, v, c) in visit and (d, v, c2) in visit) + \
                             quicksum(customers[c]['service_time'] * visit[(d, v, c)] for c in range(num_customers) if (d, v, c) in visit)
            model.addConstr(route_duration <= max_route_duration)

    # Time window constraints
    for d in range(num_days):
        for v in range(num_vehicles):
            for c in range(num_customers):
                if (d, v, c) in visit:
                    arrival_time = quicksum(distance_matrix[0][c2] * visit[(d, v, c2)] for c2 in range(num_customers) if (d, v, c2) in visit) + \
                                   quicksum(distance_matrix[c2+1][c+1] * visit[(d, v, c2)] for c2 in range(num_customers) if (d, v, c2) in visit)
                    model.addConstr(arrival_time >= customer_time_windows[c][0])
                    model.addConstr(arrival_time <= customer_time_windows[c][1])

    # Solve the model
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    model.optimize()
    if logger:
        logger.log(model.ObjVal)

    # Extract solution
    solution = {'routes': {}, 'schedules': {}}
    for d in range(num_days):
        solution['routes'][str(d)] = []
        for v in range(num_vehicles):
            route = {'customers': [], 'cost': 0, 'demand': 0, 'parts': {}}
            for c in range(num_customers):
                if (d, v, c) in visit and visit[(d, v, c)].x > 0.5:
                    route['customers'].append(customers[c]['id'])
                    route['demand'] += customers[c]['demand_per_day'] * next(p['days_covered'] for p in customers[c]['schedules'][int(schedule_selection[c].x)]['parts'] if p['visit_day'] == d)
                    route['parts'][customers[c]['id']] = next({'visit_day': p['visit_day'], 'days_covered': p['days_covered'], 'demand_per_visit': p['demand_per_visit']} for p in customers[c]['schedules'][int(schedule_selection[c].x)]['parts'] if p['visit_day'] == d)
            if route['customers']:
                route['cost'] = sum(distance_matrix[0][c] for c in range(num_customers) if (d, v, c) in visit and visit[(d, v, c)].x > 0.5) + \
                                sum(distance_matrix[c+1][c2+1] for c in range(num_customers) for c2 in range(num_customers) if (d, v, c) in visit and (d, v, c2) in visit and visit[(d, v, c)].x > 0.5 and visit[(d, v, c2)].x > 0.5)
                solution['routes'][str(d)].append(route)

    for c in range(num_customers):
        customer = customers[c]
        selected_schedule_index = int(schedule_selection[c].x)
        solution['schedules'][str(customer['id'])] = {
            'schedule_index': selected_schedule_index,
            'days': customer['schedules'][selected_schedule_index]['days'],
            'visit_frequency': customer['schedules'][selected_schedule_index]['visit_frequency']
        }

    solution['objective_value'] = model.ObjVal
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    solution = solve(args.instance_path, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()