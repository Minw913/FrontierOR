import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit):
    num_delivery_sites = instance['num_delivery_sites']
    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    num_vehicles = instance['num_vehicles']
    depot_id = instance['depot']['id']
    delivery_site_ids = [site['id'] for site in instance['delivery_sites']]
    customer_ids = [customer['id'] for customer in instance['customers']]
    customer_demands = {customer['id']: customer['demand'] for customer in instance['customers']}
    routing_cost_matrix = instance['routing_cost_matrix']
    assignment_cost_matrix = instance['assignment_cost_matrix']

    model = Model('distribution_problem')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    x = model.addVars(num_vehicles, num_delivery_sites + 1, num_delivery_sites + 1, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_customers, num_delivery_sites, vtype=GRB.BINARY, name='y')
    z = model.addVars(num_vehicles, num_delivery_sites, vtype=GRB.BINARY, name='z')
    q = model.addVars(num_vehicles, num_delivery_sites + 1, vtype=GRB.INTEGER, name='q', lb=0, ub=vehicle_capacity)
    s = model.addVars(num_vehicles, num_customers, num_delivery_sites, vtype=GRB.BINARY, name='s')

    model.setObjective(
        quicksum(routing_cost_matrix[i][j] * x[v, i, j] for v in range(num_vehicles) for i in range(num_delivery_sites + 1) for j in range(num_delivery_sites + 1)) +
        quicksum(assignment_cost_matrix[c][d] * y[c, d] for c in range(num_customers) for d in range(num_delivery_sites)),
        GRB.MINIMIZE
    )

    model.addConstrs((quicksum(y[c, d] for d in range(num_delivery_sites)) == 1 for c in range(num_customers)), name='customer_assignment')
    model.addConstrs((s[v, c, d] <= y[c, d] for v in range(num_vehicles) for c in range(num_customers) for d in range(num_delivery_sites)), name='service_pairing_upper')
    model.addConstrs((s[v, c, d] <= z[v, d] for v in range(num_vehicles) for c in range(num_customers) for d in range(num_delivery_sites)), name='service_pairing_lower')
    model.addConstrs((quicksum(s[v, c, d] for v in range(num_vehicles)) == y[c, d] for c in range(num_customers) for d in range(num_delivery_sites)), name='service_pairing')
    model.addConstrs((quicksum(x[v, i, j] for i in range(num_delivery_sites + 1)) == quicksum(x[v, j, i] for i in range(num_delivery_sites + 1)) for v in range(num_vehicles) for j in range(num_delivery_sites + 1)), name='flow_conservation')
    model.addConstrs((x[v, 0, j] <= 1 for v in range(num_vehicles) for j in range(num_delivery_sites + 1)), name='max_one_route_segment_leaving_depot')
    model.addConstrs((quicksum(customer_demands[customer_ids[c]] * s[v, c, d] for c in range(num_customers)) <= vehicle_capacity * z[v, d] for v in range(num_vehicles) for d in range(num_delivery_sites)), name='vehicle_capacity')
    model.addConstrs((quicksum(x[v, i, d + 1] for i in range(num_delivery_sites + 1)) == z[v, d] for v in range(num_vehicles) for d in range(num_delivery_sites)), name='delivery_site_visit')
    model.addConstrs((q[v, j] >= q[v, i] + quicksum(customer_demands[customer_ids[c]] * s[v, c, j - 1] for c in range(num_customers)) - vehicle_capacity * (1 - x[v, i, j]) for v in range(num_vehicles) for i in range(num_delivery_sites + 1) for j in range(1, num_delivery_sites + 1)), name='cumulative_quantity_update_lower')
    model.addConstrs((q[v, j] <= vehicle_capacity * x[v, i, j] + q[v, i] - quicksum(customer_demands[customer_ids[c]] * s[v, c, j - 1] for c in range(num_customers)) + vehicle_capacity * (1 - x[v, i, j]) for v in range(num_vehicles) for i in range(num_delivery_sites + 1) for j in range(1, num_delivery_sites + 1)), name='cumulative_quantity_update_upper')
    model.addConstrs((q[v, 0] == 0 for v in range(num_vehicles)), name='initial_cumulative_quantity')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        return model
    else:
        return None

def extract_solution(instance, model):
    num_delivery_sites = instance['num_delivery_sites']
    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    depot_id = instance['depot']['id']
    delivery_site_ids = [site['id'] for site in instance['delivery_sites']]
    customer_ids = [customer['id'] for customer in instance['customers']]

    x = model.getAttr('X', model.getVars())
    x_dict = {}
    for var in model.getVars():
        if var.VarName.startswith('x'):
            name = var.VarName.split('[')[1].split(']')[0]
            v, i, j = map(int, name.split(','))
            x_dict[(v, i, j)] = var.X

    y = model.getAttr('X', model.getVars())
    y_dict = {}
    for var in model.getVars():
        if var.VarName.startswith('y'):
            name = var.VarName.split('[')[1].split(']')[0]
            c, d = map(int, name.split(','))
            y_dict[(c, d)] = var.X

    z = model.getAttr('X', model.getVars())
    z_dict = {}
    for var in model.getVars():
        if var.VarName.startswith('z'):
            name = var.VarName.split('[')[1].split(']')[0]
            v, d = map(int, name.split(','))
            z_dict[(v, d)] = var.X

    s = model.getAttr('X', model.getVars())
    s_dict = {}
    for var in model.getVars():
        if var.VarName.startswith('s'):
            name = var.VarName.split('[')[1].split(']')[0]
            v, c, d = map(int, name.split(','))
            s_dict[(v, c, d)] = var.X

    solution = {
        'objective_value': model.getObjective().getValue(),
        'routes': {},
        'assignments': {},
        'customer_tours': {}
    }

    vehicle_id = 0
    for v in range(num_vehicles):
        route = []
        sequence = [0]
        current_node = 0
        next_node = None
        while True:
            for j in range(num_delivery_sites + 1):
                if x_dict.get((v, current_node, j), 0) > 0.5:
                    route.append([current_node, j])
                    sequence.append(j)
                    next_node = j
                    break
            if next_node == 0:
                break
            if next_node is None:
                break
            current_node = next_node
        if len(route) > 0:
            solution['routes'][str(vehicle_id)] = {
                'arcs': route,
                'sequence': sequence
            }
            vehicle_id += 1

    for c in range(num_customers):
        for d in range(num_delivery_sites):
            if y_dict.get((c, d), 0) > 0.5:
                solution['assignments'][str(customer_ids[c])] = delivery_site_ids[d]
                for v in range(num_vehicles):
                    if s_dict.get((v, c, d), 0) > 0.5:
                        solution['customer_tours'][str(customer_ids[c])] = {
                            'vehicle': v,
                            'site': delivery_site_ids[d]
                        }
                break

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')

    args = parser.parse_args()

    instance = read_instance(args.instance_path)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = solve(instance, args.time_limit)

    if model:
        solution = extract_solution(instance, model)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        if logger:
            logger.log(solution['objective_value'])
    else:
        print("No solution found")

if __name__ == '__main__':
    main()