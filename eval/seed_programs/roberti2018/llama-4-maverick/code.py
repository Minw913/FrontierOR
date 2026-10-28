import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
from itertools import combinations

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
    model.setParam("Threads", 1)
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = False

    customer_dict = {customer['id']: customer for customer in customers}
    satellite_dict = {satellite['id']: satellite for satellite in satellites}
    group_dict = {group['group_id']: group for group in groups}

    all_nodes = [depot['id']] + [satellite['id'] for satellite in satellites] + [customer['id'] for customer in customers]
    satellite_nodes = [satellite['id'] for satellite in satellites]
    customer_nodes = [customer['id'] for customer in customers]

    distances = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                x1, y1 = get_coordinates(i, depot, satellites, customers)
                x2, y2 = get_coordinates(j, depot, satellites, customers)
                distances[i, j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    x = model.addVars([(i, j, k) for k in range(T1) for i in [depot['id']] + satellite_nodes for j in [depot['id']] + satellite_nodes if i != j], vtype=GRB.BINARY)
    y = model.addVars([(i, k) for k in range(T1) for i in satellite_nodes], vtype=GRB.CONTINUOUS)

    z = model.addVars([(i, j, k) for k in satellite_nodes for i in [k] + customer_nodes for j in [k] + customer_nodes if i != j], vtype=GRB.BINARY)
    load = model.addVars([(i, j, k) for k in satellite_nodes for i in [k] + customer_nodes for j in [k] + customer_nodes if i != j], vtype=GRB.CONTINUOUS)

    a = model.addVars([(g['group_id'], s) for g in groups for s in satellite_nodes], vtype=GRB.BINARY)

    total_cost = quicksum(distances[i, j] * x[i, j, k] for k in range(T1) for i in [depot['id']] + satellite_nodes for j in [depot['id']] + satellite_nodes if i != j)
    total_cost += quicksum(satellite_dict[i]['handling_cost'] * y[i, k] for k in range(T1) for i in satellite_nodes)
    total_cost += quicksum(distances[i, j] * z[i, j, k] for k in satellite_nodes for i in [k] + customer_nodes for j in [k] + customer_nodes if i != j)
    model.setObjective(total_cost, GRB.MINIMIZE)

    model.addConstrs((quicksum(x[depot['id'], j, k] for j in satellite_nodes) == 1 for k in range(T1)), 'first_echelon_depart_depot')
    model.addConstrs((quicksum(x[i, depot['id'], k] for i in satellite_nodes) == 1 for k in range(T1)), 'first_echelon_return_depot')
    for k in range(T1):
        for j in satellite_nodes:
            model.addConstr(quicksum(x[i, j, k] for i in [depot['id']] + satellite_nodes if i != j) == quicksum(x[j, i, k] for i in [depot['id']] + satellite_nodes if i != j), f'first_echelon_flow_conservation_{j}_{k}')
    model.addConstrs((quicksum(y[i, k] for k in range(T1)) <= satellite_dict[i]['capacity'] for i in satellite_nodes), 'satellite_capacity')
    model.addConstrs((y[i, k] <= Q1 * quicksum(x[i, j, k] for j in [depot['id']] + satellite_nodes if i != j) for i in satellite_nodes for k in range(T1)), 'delivery_quantity')
    model.addConstrs((quicksum(y[i, k] for i in satellite_nodes) <= Q1 for k in range(T1)), 'vehicle_capacity')

    for k in range(T1):
        for size in range(2, len(satellite_nodes) + 1):
            for subset in combinations(satellite_nodes, size):
                model.addConstr(quicksum(x[i, j, k] for i in subset for j in subset if i != j) <= len(subset) - 1)

    model.addConstrs((quicksum(z[j, i, s] for s in satellite_nodes for j in [s] + customer_nodes if j != i) == 1 for i in customer_nodes), 'customer_visited_once')
    for s in satellite_nodes:
        for j in [s] + customer_nodes:
            model.addConstr(quicksum(z[i, j, s] for i in [s] + customer_nodes if i != j) == quicksum(z[j, i, s] for i in [s] + customer_nodes if i != j), f'flow_conservation_second_echelon_{j}_{s}')
    model.addConstrs((load[i, j, s] <= Q2 * z[i, j, s] for s in satellite_nodes for i in [s] + customer_nodes for j in [s] + customer_nodes if i != j), 'capacity_constraint_second_echelon')
    model.addConstrs((load[i, j, s] >= customer_dict[j]['demand'] * z[i, j, s] for s in satellite_nodes for i in [s] + customer_nodes for j in customer_nodes if i != j), 'demand_constraint_second_echelon')
    model.addConstrs((quicksum(z[s, j, s] for j in customer_nodes) <= satellite_dict[s]['max_vehicles'] for s in satellite_nodes), 'max_vehicles_per_satellite')
    model.addConstr((quicksum(z[s, j, s] for s in satellite_nodes for j in customer_nodes) <= T2), 'total_second_echelon_vehicles')

    for g in groups:
        for s in satellite_nodes:
            model.addConstr(a[g['group_id'], s] == quicksum(z[s, i, s] for i in g['customer_ids']) / len(g['customer_ids']))

    for s in satellite_nodes:
        model.addConstr(quicksum(y[s, k] for k in range(T1)) == quicksum(load[s, j, s] for j in customer_nodes if (s, j, s) in load))

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.getVars():
            try:
                solution = process_solution(model, instance, x, y, z, a, depot, satellites, customers, groups, distances)
                with open(args.solution_path, 'w') as file:
                    json.dump(solution, file, indent=4)
                if logger:
                    logger.log(solution['objective_value'])
            except Exception as e:
                print(f"Error processing solution: {e}")
        else:
            print("Model has no variables.")
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

def process_solution(model, instance, x, y, z, a, depot, satellites, customers, groups, distances):
    satellite_dict = {satellite['id']: satellite for satellite in satellites}
    customer_dict = {customer['id']: customer for customer in customers}
    group_dict = {group['group_id']: group for group in groups}

    solution = {
        'objective_value': model.ObjVal if model.ObjVal != GRB.INFINITY else float('inf'),
        'first_echelon_routes': [],
        'second_echelon_routes': [],
        'group_assignments': {}
    }

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

    for group in groups:
        group_id = group['group_id']
        satellite_id = next((s for s in [satellite['id'] for satellite in satellites] if a[group_id, s].X > 0.5), None)
        if satellite_id is not None:
            solution['group_assignments'][str(group_id)] = satellite_id

    return solution

if __name__ == '__main__':
    main()