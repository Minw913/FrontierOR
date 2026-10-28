import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    parser = argparse.ArgumentParser(description='Solve the dial-a-ride problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)

    num_vehicles = instance['num_vehicles']
    num_users = instance['num_users']
    num_nodes = instance['num_nodes']
    vehicle_capacity = instance['vehicle_capacity']
    maximum_ride_time = instance['maximum_ride_time']
    maximum_route_duration = instance['maximum_route_duration']

    nodes = instance['nodes']
    node_coords = {(node['node_id'],): (node['x'], node['y']) for node in nodes}

    model = Model('dial-a-ride')
    model.Params.TimeLimit = args.time_limit

    # Decision Variables
    x = model.addVars([(v, i, j) for v in range(num_vehicles) for i in range(num_nodes) for j in range(num_nodes) if i != j], vtype=GRB.BINARY, name='x')
    t = model.addVars([(v, i) for v in range(num_vehicles) for i in range(num_nodes)], vtype=GRB.CONTINUOUS, name='t')
    l = model.addVars([(v, i) for v in range(num_vehicles) for i in range(num_nodes)], vtype=GRB.INTEGER, name='l')
    u_ride_time = model.addVars([user['user_id'] for node in nodes if node['node_type'] in ['pickup', 'dropoff'] for user in [node] if 'user_id' in node], vtype=GRB.CONTINUOUS, name='u_ride_time')

    # Objective Function
    model.setObjective(quicksum(calculate_distance(node_coords[(i,)][0], node_coords[(i,)][1], node_coords[(j,)][0], node_coords[(j,)][1]) * x[v, i, j] for v in range(num_vehicles) for i in range(num_nodes) for j in range(num_nodes) if i != j), GRB.MINIMIZE)

    # Constraints
    for user in range(num_users):
        pickup_node = next(node['node_id'] for node in nodes if node['node_type'] == 'pickup' and node['user_id'] == user)
        dropoff_node = next(node['node_id'] for node in nodes if node['node_type'] == 'dropoff' and node['user_id'] == user)
        model.addConstr(quicksum(x[v, pickup_node, j] for v in range(num_vehicles) for j in range(num_nodes) if j != pickup_node) == 1)
        model.addConstr(quicksum(x[v, i, pickup_node] for i in range(num_nodes) if i != pickup_node) - quicksum(x[v, pickup_node, j] for j in range(num_nodes) if j != pickup_node) == 0)
        model.addConstr(quicksum(x[v, dropoff_node, j] for j in range(num_nodes) if j != dropoff_node) - quicksum(x[v, i, dropoff_node] for i in range(num_nodes) if i != dropoff_node) == 0)
        model.addConstr(quicksum(x[v, pickup_node, j] for j in range(num_nodes) if j != pickup_node) - quicksum(x[v, dropoff_node, j] for j in range(num_nodes) if j != dropoff_node) == 0)

    for v in range(num_vehicles):
        model.addConstr(quicksum(x[v, 0, j] for j in range(1, num_nodes)) == 1)
        model.addConstr(quicksum(x[v, i, num_nodes-1] for i in range(num_nodes-1)) == 1)
        for i in range(1, num_nodes-1):
            model.addConstr(quicksum(x[v, j, i] for j in range(num_nodes) if j != i) == quicksum(x[v, i, j] for j in range(num_nodes) if j != i))

    for v in range(num_vehicles):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    model.addConstr(t[v, i] + nodes[i]['service_duration'] + calculate_distance(node_coords[(i,)][0], node_coords[(i,)][1], node_coords[(j,)][0], node_coords[(j,)][1]) - t[v, j] <= (1 - x[v, i, j]) * 1e6)
                    model.addConstr(l[v, i] + nodes[j]['load'] - l[v, j] <= (1 - x[v, i, j]) * 1e6)

    for v in range(num_vehicles):
        model.addConstr(t[v, num_nodes-1] - t[v, 0] <= maximum_route_duration)

    for node in nodes:
        for v in range(num_vehicles):
            model.addConstr(t[v, node['node_id']] >= node['earliest_time'])
            model.addConstr(t[v, node['node_id']] <= node['latest_time'])

    for user in range(num_users):
        pickup_node = next(node for node in nodes if node['node_type'] == 'pickup' and node['user_id'] == user)
        dropoff_node = next(node for node in nodes if node['node_type'] == 'dropoff' and node['user_id'] == user)
        for v in range(num_vehicles):
            model.addConstr(u_ride_time[user] >= t[v, dropoff_node['node_id']] - t[v, pickup_node['node_id']] - pickup_node['service_duration'])
        model.addConstr(u_ride_time[user] <= maximum_ride_time)
        direct_travel_time = calculate_distance(node_coords[(pickup_node['node_id'],)][0], node_coords[(pickup_node['node_id'],)][1], node_coords[(dropoff_node['node_id'],)][0], node_coords[(dropoff_node['node_id'],)][1])
        model.addConstr(u_ride_time[user] >= direct_travel_time)

    for v in range(num_vehicles):
        for node in nodes:
            model.addConstr(l[v, node['node_id']] >= max(0, node['load']))
            model.addConstr(l[v, node['node_id']] <= min(vehicle_capacity, vehicle_capacity + node['load']))

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            'objective_value': model.getObjective().getValue(),
            'routes': {},
            'service_times': {},
            'ride_times': {}
        }

        for v in range(num_vehicles):
            route = [0]
            current_node = 0
            while True:
                next_node = next((j for j in range(num_nodes) if j != current_node and x[v, current_node, j].x > 0.5), None)
                if next_node is None:
                    break
                route.append(next_node)
                current_node = next_node
            solution['routes'][f'vehicle_{v}'] = route

        for v in range(num_vehicles):
            for i in range(num_nodes):
                if t[v, i].x > 0:
                    solution['service_times'][i] = t[v, i].x
            solution['service_times'][f'depot_start_{v}'] = t[v, 0].x
            solution['service_times'][f'depot_end_{v}'] = t[v, num_nodes-1].x

        for user in range(num_users):
            solution['ride_times'][user] = u_ride_time[user].x

        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)

        if logger:
            logger.log(model.getObjective().getValue())

if __name__ == '__main__':
    main()