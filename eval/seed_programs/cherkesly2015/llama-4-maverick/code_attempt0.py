import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    n_requests = instance['n_requests']
    n_nodes = instance['n_nodes']
    vehicle_capacity = instance['vehicle_capacity']
    planning_horizon = instance['planning_horizon']
    vehicle_fixed_cost = instance['vehicle_fixed_cost']
    nodes = instance['nodes']
    arcs = instance['arcs']

    # Create a Gurobi model
    model = gp.Model('PickupAndDelivery')
    model.Params.LogToConsole = 0  # Silence Gurobi output

    # Decision variables
    x = model.addVars([(arc['from'], arc['to']) for arc in arcs], vtype=GRB.BINARY, name='x')
    start_time = model.addVars(range(n_nodes), lb=0, name='start_time')
    load = model.addVars(range(n_nodes), lb=0, ub=vehicle_capacity, name='load')

    # Objective function: minimize total travel cost
    model.setObjective(gp.quicksum(arc['cost'] * x[arc['from'], arc['to']] for arc in arcs), GRB.MINIMIZE)

    # Constraints
    # Flow conservation and visit exactly once for each pickup and delivery
    for node in range(1, n_nodes - 1):
        model.addConstr(gp.quicksum(x[from_node, node] for from_node in range(n_nodes) if (from_node, node) in x) ==
                        gp.quicksum(x[node, to_node] for to_node in range(n_nodes) if (node, to_node) in x), name=f'flow_conservation_{node}')

    for request in range(1, n_requests + 1):
        pickup_node = request
        delivery_node = request + n_requests
        model.addConstr(gp.quicksum(x[pickup_node, to_node] for to_node in range(n_nodes) if (pickup_node, to_node) in x) == 1, name=f'pickup_visit_{request}')
        model.addConstr(gp.quicksum(x[delivery_node, to_node] for to_node in range(n_nodes) if (delivery_node, to_node) in x) == 1, name=f'delivery_visit_{request}')

    # Pickup before delivery for each request
    for request in range(1, n_requests + 1):
        pickup_node = request
        delivery_node = request + n_requests
        model.addConstr(start_time[pickup_node] + nodes[pickup_node]['service_time'] + next((arc['travel_time'] for arc in arcs if arc['from'] == pickup_node and arc['to'] == delivery_node), 0) <= start_time[delivery_node], name=f'pickup_before_delivery_{request}')

    # Time window constraints
    for node in range(n_nodes):
        model.addConstr(start_time[node] >= nodes[node]['tw_early'], name=f'tw_early_{node}')
        model.addConstr(start_time[node] <= nodes[node]['tw_late'], name=f'tw_late_{node}')

    # Time consistency
    for arc in arcs:
        from_node, to_node = arc['from'], arc['to']
        if (from_node, to_node) in x:
            model.addConstr(start_time[from_node] + nodes[from_node]['service_time'] + arc['travel_time'] <= start_time[to_node] + (1 - x[from_node, to_node]) * (planning_horizon[1] - planning_horizon[0]), name=f'time_consistency_{from_node}_{to_node}')

    # Load consistency and capacity
    for arc in arcs:
        from_node, to_node = arc['from'], arc['to']
        if (from_node, to_node) in x:
            model.addConstr(load[from_node] + nodes[to_node]['load'] <= load[to_node] + (1 - x[from_node, to_node]) * vehicle_capacity, name=f'load_consistency_{from_node}_{to_node}')

    for node in range(n_nodes):
        model.addConstr(load[node] >= max(0, nodes[node]['load']), name=f'load_lower_bound_{node}')
        model.addConstr(load[node] <= min(vehicle_capacity, vehicle_capacity + nodes[node]['load']), name=f'load_upper_bound_{node}')

    # LIFO and request splitting constraints are complex and might not be directly implementable without further analysis.
    # For simplicity, we assume that the arcs already exclude direct arcs from a pickup to a different request's delivery.

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Optimize the model
    model.optimize()

    # Log intermediate solutions
    if logger:
        model.Params.SolutionCallback = lambda model, where: log_intermediate_solutions(model, where, logger)

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        routes = extract_routes(x, nodes, n_requests)
        solution = {
            'objective_value': objective_value,
            'routes': routes
        }
        return solution
    else:
        return None

def log_intermediate_solutions(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        logger.log(objective_value)

def extract_routes(x, nodes, n_requests):
    n_nodes = 2 * n_requests + 2
    routes = []
    vehicle_id = 0

    # Find routes by tracing from the origin depot
    remaining_nodes = set(range(1, n_nodes - 1))
    while remaining_nodes:
        route_sequence = [0]
        current_node = 0
        route_details = []
        load = 0

        while True:
            next_node = None
            for node in range(n_nodes):
                if (current_node, node) in x and x[current_node, node].X > 0.5:
                    next_node = node
                    break

            if next_node is None or next_node == n_nodes - 1:
                route_sequence.append(next_node)
                route_details.append({
                    'node_id': next_node,
                    'arrival_time': 0,  # This will be calculated later
                    'load_after': load
                })
                break

            route_sequence.append(next_node)
            load += nodes[next_node]['load']
            route_details.append({
                'node_id': next_node,
                'arrival_time': 0,  # This will be calculated later
                'load_after': load
            })

            if next_node in remaining_nodes:
                remaining_nodes.remove(next_node)

            current_node = next_node

        # Calculate arrival times
        start_time = 0
        for i, node_id in enumerate(route_sequence):
            route_details[i]['arrival_time'] = start_time
            if node_id != n_nodes - 1:
                start_time += nodes[node_id]['service_time'] + next((arc['travel_time'] for arc in nodes if arc['from'] == node_id and arc['to'] == route_sequence[i+1]), 0)

        routes.append({
            'vehicle_id': vehicle_id,
            'route_sequence': route_sequence,
            'route_details': route_details
        })
        vehicle_id += 1

    return routes

def main():
    parser = argparse.ArgumentParser(description='Solve Pickup and Delivery Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)

    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    main()