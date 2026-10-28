import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as file:
        instance = json.load(file)

    num_satellites = instance['num_satellites']
    num_customers = instance['num_customers']
    num_groups = instance['num_groups']
    Q1 = instance['Q1']
    Q2 = instance['Q2']
    T1 = instance['T1']
    T2 = instance['T2']
    depot = instance['depot']
    satellites = instance['satellites']
    customers = instance['customers']
    groups = instance['groups']

    model = Model('2E-VRP')
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = False

    # Preprocessing
    customer_dict = {customer['id']: customer for customer in customers}
    satellite_dict = {satellite['id']: satellite for satellite in satellites}
    group_dict = {group['group_id']: group for group in groups}

    all_nodes = [depot['id']] + [satellite['id'] for satellite in satellites] + [customer['id'] for customer in customers]
    satellite_nodes = [satellite['id'] for satellite in satellites]
    customer_nodes = [customer['id'] for customer in customers]

    # Calculate distances
    distances = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                x1, y1 = get_coordinates(i, depot, satellites, customers)
                x2, y2 = get_coordinates(j, depot, satellites, customers)
                distances[i, j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # First-echelon variables
    x = model.addVars([(i, j, k) for k in range(T1) for i in [depot['id']] + satellite_nodes for j in [depot['id']] + satellite_nodes if i != j], vtype=GRB.BINARY)
    y = model.addVars([(i, k) for k in range(T1) for i in satellite_nodes], vtype=GRB.CONTINUOUS)

    # Second-echelon variables
    z = model.addVars([(i, j, k) for k in satellite_nodes for i in [k] + customer_nodes for j in [k] + customer_nodes if i != j], vtype=GRB.BINARY)
    load = model.addVars([(i, j, k) for k in satellite_nodes for i in [k] + customer_nodes for j in [k] + customer_nodes if i != j], vtype=GRB.CONTINUOUS)

    # Group assignment variables
    a = model.addVars([(g['group_id'], s) for g in groups for s in satellite_nodes], vtype=GRB.BINARY)

    # Objective function
    total_cost = quicksum(distances[i, j] * x[i, j, k] for k in range(T1) for i in [depot['id']] + satellite_nodes for j in [depot['id']] + satellite_nodes if i != j)
    total_cost += quicksum(satellite['handling_cost'] * y[i, k] for k in range(T1) for i in satellite_nodes)
    total_cost += quicksum(distances[i, j] * z[i, j, k] for k in satellite_nodes for i in [k] + customer_nodes for j in [k] + customer_nodes if i != j)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints
    # First-echelon constraints
    model.addConstrs((quicksum(x[depot['id'], j, k] for j in satellite_nodes) == 1 for k in range(T1)), 'first_echelon_depart_depot')
    model.addConstrs((quicksum(x[i, depot['id'], k] for i in satellite_nodes) == 1 for k in range(T1)), 'first_echelon_return_depot')
    model.addConstrs((quicksum(x[i, j, k] for i in [depot['id']] + satellite_nodes if i != j) == quicksum(x[j, i, k] for i in [depot['id']] + satellite_nodes if i != j) for j in satellite_nodes for k in range(T1)), 'first_echelon_flow_conservation')
    model.addConstrs((quicksum(y[i, k] for k in range(T1)) <= satellite_dict[i]['capacity'] for i in satellite_nodes), 'satellite_capacity')
    model.addConstrs((y[i, k] <= Q1 * quicksum(x[i, j, k] for j in [depot['id']] + satellite_nodes if i != j) for i in satellite_nodes for k in range(T1)), 'delivery_quantity')
    model.addConstrs((quicksum(y[i, k] for i in satellite_nodes) <= Q1 for k in range(T1)), 'vehicle_capacity')

    # Subtour elimination for first-echelon
    model.addConstrs((quicksum(x[i, j, k] for i in subset for j in subset if i != j) <= len(subset) - 1 for subset in get_subsets(satellite_nodes) for k in range(T1)), 'subtour_elimination_first_echelon')

    # Second-echelon constraints
    model.addConstrs((quicksum(z[j, i, s] for s in satellite_nodes for j in [s] + customer_nodes if j != i) == 1 for i in customer_nodes), 'customer_visited_once')
    model.addConstrs((quicksum(z[i, j, s] for i in [s] + customer_nodes if i != j) == quicksum(z[j, i, s] for i in [s] + customer_nodes if i != j) for s in satellite_nodes for j in [s] + customer_nodes), 'flow_conservation_second_echelon')
    model.addConstrs((load[i, j, s] <= Q2 * z[i, j, s] for s in satellite_nodes for i in [s] + customer_nodes for j in [s] + customer_nodes if i != j), 'capacity_constraint_second_echelon')
    model.addConstrs((load[i, j, s] >= customer_dict[j]['demand'] * z[i, j, s] for s in satellite_nodes for i in [s] + customer_nodes for j in customer_nodes if i != j), 'demand_constraint_second_echelon')
    model.addConstrs((quicksum(z[s, j, s] for j in customer_nodes) <= satellite_dict[s]['max_vehicles'] for s in satellite_nodes), 'max_vehicles_per_satellite')
    model.addConstrs((quicksum(quicksum(z[s, j, s] for j in customer_nodes) for s in satellite_nodes) <= T2), 'total_second_echelon_vehicles')

    # Group assignment constraints
    model.addConstrs((a[g['group_id'], s] == quicksum(z[s, i, s] for i in g['customer_ids']) / len(g['customer_ids']) for g in groups for s in satellite_nodes), 'group_assignment')

    # Load and delivery constraints
    model.addConstrs((quicksum(y[i, k] for k in range(T1)) == quicksum(load[s, j, s] - load[j, s, s] for j in customer_nodes) for i in [s] for s in satellite_nodes), 'load_delivered_equals_load_departed')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = process_solution(model, instance, x, y, z, a, depot, satellites, customers, groups)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
        if logger:
            logger.log(solution['objective_value'])
    else:
        print('No solution found')

def get_coordinates(node_id, depot, satellites, customers):
    if node_id == depot['id']:
        return depot['x'], depot['y']
    elif node_id in [satellite['id'] for satellite in satellites]:
        satellite = next(s for s in satellites if s['id'] == node_id)
        return satellite['x'], satellite['y']
    else:
        customer = next(c for c in customers if c['id'] == node_id)
        return customer['x'], customer['y']

def get_subsets(satellite_nodes):
    subsets = []
    num_satellites = len(satellite_nodes)
    for mask in range(1, 2**num_satellites):
        subset = [satellite_nodes[i] for i in range(num_satellites) if (mask & (1 << i))]
        subsets.append(subset)
    return subsets

def process_solution(model, instance, x, y, z, a, depot, satellites, customers, groups):
    satellite_dict = {satellite['id']: satellite for satellite in satellites}
    customer_dict = {customer['id']: customer for customer in customers}
    group_dict = {group['group_id']: group for group in groups}

    solution = {
        'objective_value': model.ObjVal,
        'first_echelon_routes': [],
        'second_echelon_routes': [],
        'group_assignments': {}
    }

    # First-echelon routes
    for k in range(instance['T1']):
        route = [depot['id']]
        current_node = depot['id']
        while True:
            next_node = next((j for j in [depot['id']] + [satellite['id'] for satellite in satellites] if x[current_node, j, k].X > 0.5), None)
            if next_node is None or next_node == depot['id']:
                break
            route.append(next_node)
            current_node = next_node
        route.append(depot['id'])
        if len(route) > 2:
            deliveries = {i: y[i, k].X for i in [satellite['id'] for satellite in satellites] if y[i, k].X > 0}
            solution['first_echelon_routes'].append({'vehicle': k, 'route': route, 'deliveries': deliveries})

    # Second-echelon routes
    for s in [satellite['id'] for satellite in satellites]:
        vehicle_routes = []
        for i in [s] + [customer['id'] for customer in customers]:
            if z[s, i, s].X > 0.5:
                route = [s, i]
                current_node = i
                while True:
                    next_node = next((j for j in [s] + [customer['id'] for customer in customers] if z[current_node, j, s].X > 0.5 and j != route[-1]), None)
                    if next_node is None or next_node == s:
                        break
                    route.append(next_node)
                    current_node = next_node
                route.append(s)
                vehicle_routes.append(route)
        for route in vehicle_routes:
            solution['second_echelon_routes'].append({'satellite': s, 'route': route})

    # Group assignments
    for group in groups:
        group_id = group['group_id']
        satellite_id = next((s for s in [satellite['id'] for satellite in satellites] if a[group_id, s].X > 0.5), None)
        solution['group_assignments'][str(group_id)] = satellite_id

    return solution

if __name__ == '__main__':
    main()