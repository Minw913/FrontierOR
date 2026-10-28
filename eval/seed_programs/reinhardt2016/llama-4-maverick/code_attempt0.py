import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def calculate_distance(node1, node2):
    return math.sqrt((node1['x'] - node2['x']) ** 2 + (node1['y'] - node2['y']) ** 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    num_edge_sets = instance['num_edge_sets']
    nodes = instance['nodes']
    edge_sets = instance['edge_sets']
    distance_matrix = instance['distance_matrix']

    model = Model('VRP')
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = args.time_limit

    # Decision Variables
    x = model.addVars(num_vehicles, num_customers + 1, num_customers + 1, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_edge_sets, vtype=GRB.BINARY, name='y')
    visit_times = model.addVars(num_vehicles, num_customers + 1, vtype=GRB.CONTINUOUS, name='visit_times')
    flow = model.addVars(num_vehicles, num_customers + 1, num_customers + 1, vtype=GRB.CONTINUOUS, name='flow')

    # Objective Function
    model.setObjective(
        quicksum(distance_matrix[i][j] * x[v, i, j] for v in range(num_vehicles) for i in range(num_customers + 1) for j in range(num_customers + 1)) +
        quicksum(y[e] * edge_sets[e]['access_cost'] for e in range(num_edge_sets)),
        GRB.MINIMIZE
    )

    # Constraints
    for v in range(num_vehicles):
        model.addConstr(quicksum(x[v, 0, j] for j in range(1, num_customers + 1)) == quicksum(x[v, j, 0] for j in range(1, num_customers + 1)), name=f'flow_conservation_{v}')
        for i in range(num_customers + 1):
            model.addConstr(quicksum(x[v, i, j] for j in range(num_customers + 1)) == quicksum(x[v, j, i] for j in range(num_customers + 1)), name=f'flow_balance_{v}_{i}')
        for customer in range(1, num_customers + 1):
            model.addConstr(flow[v, customer, 0] == 0, name=f'flow_init_{v}_{customer}')
            model.addConstr(quicksum(flow[v, customer, j] for j in range(num_customers + 1)) - quicksum(flow[v, j, customer] for j in range(num_customers + 1)) == nodes[customer]['demand'], name=f'flow_demand_{v}_{customer}')
        model.addConstr(quicksum(flow[v, 0, j] for j in range(num_customers + 1)) == quicksum(nodes[j]['demand'] * x[v, 0, j] for j in range(1, num_customers + 1)), name=f'flow_depot_out_{v}')
        model.addConstr(quicksum(flow[v, j, 0] for j in range(num_customers + 1)) == quicksum(nodes[j]['demand'] * x[v, j, 0] for j in range(1, num_customers + 1)), name=f'flow_depot_in_{v}')
        for j in range(num_customers + 1):
            model.addConstr(quicksum(flow[v, i, j] for i in range(num_customers + 1)) == quicksum(flow[v, j, i] for i in range(num_customers + 1)), name=f'flow_conserve_{v}_{j}')

    for customer in range(1, num_customers + 1):
        model.addConstr(quicksum(x[v, customer, j] for v in range(num_vehicles) for j in range(num_customers + 1)) == 1, name=f'visit_once_{customer}')
        model.addConstr(quicksum(x[v, j, customer] for v in range(num_vehicles) for j in range(num_customers + 1)) == 1, name=f'visit_once_in_{customer}')

    for v in range(num_vehicles):
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                model.addConstr(visit_times[v, j] >= visit_times[v, i] + (distance_matrix[i][j] + nodes[i]['service_time']) * x[v, i, j] - (1 - x[v, i, j]) * instance['schedule_horizon'], name=f'time_window_{v}_{i}_{j}')
            model.addConstr(visit_times[v, i] >= nodes[i]['time_window_start'], name=f'time_window_start_{v}_{i}')
            model.addConstr(visit_times[v, i] <= nodes[i]['time_window_end'], name=f'time_window_end_{v}_{i}')

    for v in range(num_vehicles):
        model.addConstr(quicksum(nodes[j]['demand'] * x[v, 0, j] for j in range(1, num_customers + 1)) <= vehicle_capacity, name=f'capacity_{v}')

    for e in range(num_edge_sets):
        edge_set = edge_sets[e]
        for customer in range(num_customers + 1):
            node_customer = nodes[customer]
            for edge in edge_set['edges']:
                i, j = edge
                model.addConstr(quicksum(x[v, i, j] for v in range(num_vehicles)) <= y[e], name=f'edge_set_access_{e}_{i}_{j}')
                model.addConstr(quicksum(x[v, j, i] for v in range(num_vehicles)) <= y[e], name=f'edge_set_access_in_{e}_{j}_{i}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        if logger:
            logger.log(objective_value)

        routes = []
        edge_sets_used = []

        for v in range(num_vehicles):
            route = []
            sequence = [0]
            arcs = []
            visit_times_dict = {}
            current_node = 0
            while True:
                next_node = None
                for j in range(num_customers + 1):
                    if x[v, current_node, j].x > 0.5:
                        next_node = j
                        break
                if next_node is None:
                    break
                arcs.append([current_node, next_node])
                sequence.append(next_node)
                visit_times_dict[str(next_node)] = visit_times[v, next_node].x
                current_node = next_node
                if current_node == 0:
                    break
            if len(sequence) > 2:
                routes.append({
                    'vehicle': v,
                    'sequence': sequence,
                    'arcs': arcs,
                    'visit_times': visit_times_dict
                })

        for e in range(num_edge_sets):
            if y[e].x > 0.5:
                edge_sets_used.append(edge_sets[e]['id'])

        solution = {
            'objective_value': objective_value,
            'routes': routes,
            'edge_sets_used': edge_sets_used
        }

        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)

if __name__ == '__main__':
    main()