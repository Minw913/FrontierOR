import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_customers = instance['num_customers']
    customers = instance['customers']
    depot = instance['depot']
    scenarios = instance['scenarios']
    distance_matrix = instance['distance_matrix']
    vehicle_capacity = instance['vehicle_capacity']
    endogenous_time_window_width = instance.get('endogenous_time_window_width', customers[0]['endogenous_time_window_width'])

    model = Model('VRPTW')
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    # Variables for time window start times
    time_window_start = model.addVars(num_customers, lb=[c['exogenous_time_window'][0] for c in customers], 
                                      ub=[c['exogenous_time_window'][1] - c['endogenous_time_window_width'] for c in customers], 
                                      vtype=GRB.CONTINUOUS, name='time_window_start')

    # Variables for routing in each scenario
    x = model.addVars([(s['scenario_id'], i, j) for s in scenarios for i in range(num_customers+2) for j in range(num_customers+2)], 
                      vtype=GRB.BINARY, name='x')
    arrival_times = model.addVars([(s['scenario_id'], i) for s in scenarios for i in range(num_customers+2)], 
                                  vtype=GRB.CONTINUOUS, name='arrival_times')
    demands = {s['scenario_id']: s['demands'] for s in scenarios}

    # Objective: Minimize expected total travel cost
    model.setObjective(quicksum(s['probability'] * quicksum(distance_matrix[i][j] * x[s['scenario_id'], i, j] for i in range(num_customers+2) for j in range(num_customers+2)) for s in scenarios))

    # Constraints for time windows and routing
    for s in scenarios:
        scenario_id = s['scenario_id']
        model.addConstrs((arrival_times[scenario_id, i] + distance_matrix[i][j] <= arrival_times[scenario_id, j] + (1 - x[scenario_id, i, j]) * 1000000 for i in range(num_customers+2) for j in range(num_customers+2)), name=f'time_consistency_{scenario_id}')
        model.addConstrs((arrival_times[scenario_id, i] >= depot['exogenous_time_window'][0] for i in [0]), name=f'depot_start_{scenario_id}')
        model.addConstrs((arrival_times[scenario_id, i] <= depot['exogenous_time_window'][1] for i in [num_customers+1]), name=f'depot_end_{scenario_id}')
        for i, customer in enumerate(customers):
            model.addConstr(arrival_times[scenario_id, i+1] >= time_window_start[i], name=f'lower_time_window_{scenario_id}_{i}')
            model.addConstr(arrival_times[scenario_id, i+1] <= time_window_start[i] + endogenous_time_window_width, name=f'upper_time_window_{scenario_id}_{i}')
            model.addConstr(arrival_times[scenario_id, i+1] >= customer['exogenous_time_window'][0], name=f'exogenous_lower_{scenario_id}_{i}')
            model.addConstr(arrival_times[scenario_id, i+1] <= customer['exogenous_time_window'][1], name=f'exogenous_upper_{scenario_id}_{i}')

        # Routing constraints
        model.addConstrs((quicksum(x[scenario_id, i, j] for j in range(num_customers+2)) == 1 for i in range(1, num_customers+1)), name=f'customer_visited_once_{scenario_id}')
        model.addConstrs((quicksum(x[scenario_id, i, j] for i in range(num_customers+2)) == quicksum(x[scenario_id, j, i] for i in range(num_customers+2)) for j in range(num_customers+2)), name=f'flow_conservation_{scenario_id}')
        model.addConstr(quicksum(x[scenario_id, 0, j] for j in range(num_customers+2)) == quicksum(x[scenario_id, j, num_customers+1] for j in range(num_customers+2)), name=f'vehicle_count_{scenario_id}')
        model.addConstrs((quicksum(demands[scenario_id][i-1] * x[scenario_id, i, j] for i in range(1, num_customers+1)) <= vehicle_capacity for j in range(num_customers+2)), name=f'capacity_{scenario_id}')

    model.update()
    return model, time_window_start, x, arrival_times

def extract_solution(instance, model, time_window_start, x, arrival_times):
    num_customers = instance['num_customers']
    customers = instance['customers']
    scenarios = instance['scenarios']
    distance_matrix = instance['distance_matrix']

    solution = {'time_windows': {}, 'routes': {}}
    for i in range(num_customers):
        solution['time_windows'][str(i)] = [time_window_start[i].X, time_window_start[i].X + instance.get('endogenous_time_window_width', customers[0]['endogenous_time_window_width'])]

    for s in scenarios:
        scenario_id = s['scenario_id']
        solution['routes'][str(scenario_id)] = []
        route = []
        arrival_times_route = {}
        for i in range(num_customers+2):
            for j in range(num_customers+2):
                if x[scenario_id, i, j].X > 0.5:
                    route.append(i)
        route.append(num_customers+1)
        route = list(dict.fromkeys(route))
        cost = sum(distance_matrix[route[j]][route[j+1]] for j in range(len(route)-1))
        for node in route:
            arrival_times_route[str(node)] = arrival_times[scenario_id, node].X
        solution['routes'][str(scenario_id)].append({'route': route, 'arrival_times': arrival_times_route, 'cost': cost})

    solution['objective_value'] = model.ObjVal if model.SolCount > 0 else float('inf')
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model, time_window_start, x, arrival_times = solve(instance)
    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.SolCount > 0 and logger:
        logger.log(model.ObjVal)

    solution = extract_solution(instance, model, time_window_start, x, arrival_times)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()