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

    model = gp.Model('PickupAndDelivery')
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  

    x = model.addVars([(arc['from'], arc['to']) for arc in arcs], vtype=GRB.BINARY, name='x')
    start_time = model.addVars(range(n_nodes), lb=0, name='start_time')
    load = model.addVars(range(n_nodes), lb=0, ub=vehicle_capacity, name='load')

    model.setObjective(gp.quicksum(arc['cost'] * x[arc['from'], arc['to']] for arc in arcs), GRB.MINIMIZE)

    for node in range(1, n_nodes - 1):
        model.addConstr(gp.quicksum(x[from_node, node] for from_node in range(n_nodes) if (from_node, node) in x) ==
                        gp.quicksum(x[node, to_node] for to_node in range(n_nodes) if (node, to_node) in x), name=f'flow_conservation_{node}')

    for request in range(1, n_requests + 1):
        pickup_node = request
        delivery_node = request + n_requests
        model.addConstr(gp.quicksum(x[pickup_node, to_node] for to_node in range(n_nodes) if (pickup_node, to_node) in x) == 1, name=f'pickup_visit_{request}')
        model.addConstr(gp.quicksum(x[delivery_node, to_node] for to_node in range(n_nodes) if (delivery_node, to_node) in x) == 1, name=f'delivery_visit_{request}')
        model.addConstr(gp.quicksum(x[from_node, pickup_node] for from_node in range(n_nodes) if (from_node, pickup_node) in x) ==
                        gp.quicksum(x[from_node, delivery_node] for from_node in range(n_nodes) if (from_node, delivery_node) in x), name=f'same_vehicle_{request}')

    for request in range(1, n_requests + 1):
        pickup_node = request
        delivery_node = request + n_requests
        travel_time = next((arc['travel_time'] for arc in arcs if arc['from'] == pickup_node and arc['to'] == delivery_node), 0)
        model.addConstr(start_time[pickup_node] + nodes[pickup_node-1]['service_time'] + travel_time <= start_time[delivery_node], name=f'pickup_before_delivery_{request}')

    for node in range(n_nodes):
        model.addConstr(start_time[node] >= nodes[node]['tw_early'], name=f'tw_early_{node}')
        model.addConstr(start_time[node] <= nodes[node]['tw_late'], name=f'tw_late_{node}')

    for arc in arcs:
        from_node, to_node = arc['from'], arc['to']
        if (from_node, to_node) in x:
            model.addConstr(start_time[from_node] + nodes[from_node]['service_time'] + arc['travel_time'] <= start_time[to_node] + (1 - x[from_node, to_node]) * (planning_horizon[1] - planning_horizon[0]), name=f'time_consistency_{from_node}_{to_node}')

    for arc in arcs:
        from_node, to_node = arc['from'], arc['to']
        if (from_node, to_node) in x:
            model.addConstr(load[from_node] + nodes[to_node]['load'] <= load[to_node] + (1 - x[from_node, to_node]) * vehicle_capacity, name=f'load_consistency_{from_node}_{to_node}')

    model.addConstr(gp.quicksum(x[0, to_node] for to_node in range(n_nodes) if (0, to_node) in x) == gp.quicksum(x[from_node, n_nodes-1] for from_node in range(n_nodes) if (from_node, n_nodes-1) in x), name='vehicle_balance')

    model.Params.TimeLimit = time_limit

    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(objective_value)
        model.optimize(callback)
    else:
        model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        if model.SolCount > 0:
            routes = extract_routes(x, nodes, n_requests, arcs, model)
            solution = {
                'objective_value': objective_value,
                'routes': routes
            }
            return solution
        else:
            return None
    else:
        return None

def extract_routes(x, nodes, n_requests, arcs, model):
    n_nodes = 2 * n_requests + 2
    routes = []
    vehicle_id = 0

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
                if next_node != 0:
                    route_details.append({
                        'node_id': next_node,
                        'arrival_time': model.getVarByName(f'start_time[{next_node}]').X,  
                        'load_after': model.getVarByName(f'load[{next_node}]').X
                    })
                break

            route_sequence.append(next_node)
            load = model.getVarByName(f'load[{next_node}]').X
            route_details.append({
                'node_id': next_node,
                'arrival_time': model.getVarByName(f'start_time[{next_node}]').X,  
                'load_after': load
            })

            if next_node in remaining_nodes:
                remaining_nodes.remove(next_node)

            current_node = next_node

        if len(route_sequence) > 2:  
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