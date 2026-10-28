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
    delivery_sites = [site['id'] for site in instance['delivery_sites']]
    customers = [customer['id'] for customer in instance['customers']]
    customer_demands = {customer['id']: customer['demand'] for customer in instance['customers']}
    routing_cost_matrix = instance['routing_cost_matrix']
    assignment_cost_matrix = instance['assignment_cost_matrix']

    model = Model('distribution_problem')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    # Variables
    x = model.addVars(num_vehicles, num_delivery_sites + 1, num_delivery_sites + 1, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_customers, num_delivery_sites, vtype=GRB.BINARY, name='y')
    z = model.addVars(num_vehicles, num_delivery_sites, vtype=GRB.BINARY, name='z')
    q = model.addVars(num_vehicles, num_delivery_sites, vtype=GRB.INTEGER, name='q')
    s = model.addVars(num_vehicles, num_customers, num_delivery_sites, vtype=GRB.BINARY, name='s')

    # Objective
    model.setObjective(
        quicksum(routing_cost_matrix[i][j] * x[v, i, j] for v in range(num_vehicles) for i in range(num_delivery_sites + 1) for j in range(num_delivery_sites + 1)) +
        quicksum(assignment_cost_matrix[c][d] * y[c, d] for c in customers for d in delivery_sites),
        GRB.MINIMIZE
    )

    # Constraints
    # Each customer is assigned to exactly one delivery site
    model.addConstrs((quicksum(y[c, d] for d in delivery_sites) == 1 for c in customers), name='customer_assignment')

    # If a customer is assigned to a delivery site, that site must be visited by exactly one vehicle
    model.addConstrs((quicksum(s[v, c, d] for v in range(num_vehicles)) == y[c, d] for c in customers for d in delivery_sites), name='service_pairing')

    # Flow conservation constraints for vehicle tours
    model.addConstrs((quicksum(x[v, i, j] for i in range(num_delivery_sites + 1)) == quicksum(x[v, j, i] for i in range(num_delivery_sites + 1)) for v in range(num_vehicles) for j in range(num_delivery_sites + 1)), name='flow_conservation')

    # Vehicle capacity constraints
    model.addConstrs((quicksum(customer_demands[c] * s[v, c, d] for c in customers) <= vehicle_capacity * z[v, d] for v in range(num_vehicles) for d in delivery_sites), name='vehicle_capacity')

    # Delivery site visit constraints
    model.addConstrs((quicksum(x[v, i, d] for i in range(num_delivery_sites + 1)) == z[v, d] for v in range(num_vehicles) for d in delivery_sites), name='delivery_site_visit')

    # Cumulative quantity constraints
    model.addConstrs((q[v, d] >= quicksum(customer_demands[c] * s[v, c, d] for c in customers) for v in range(num_vehicles) for d in delivery_sites), name='cumulative_quantity_lower')
    model.addConstrs((q[v, d] <= vehicle_capacity * z[v, d] for v in range(num_vehicles) for d in delivery_sites), name='cumulative_quantity_upper')

    # Service pairing constraints
    model.addConstrs((s[v, c, d] <= y[c, d] for v in range(num_vehicles) for c in customers for d in delivery_sites), name='service_pairing_upper')
    model.addConstrs((s[v, c, d] <= z[v, d] for v in range(num_vehicles) for c in customers for d in delivery_sites), name='service_pairing_lower')

    # Subtour elimination constraints
    model.addConstrs((x[v, i, j] + x[v, j, i] <= 1 for v in range(num_vehicles) for i in range(1, num_delivery_sites + 1) for j in range(i + 1, num_delivery_sites + 1)), name='subtour_elimination')

    # Limit the number of vehicles used
    model.addConstr(quicksum(z[v, d] for v in range(num_vehicles) for d in delivery_sites) <= num_vehicles, name='num_vehicles')

    # Symmetry breaking: assign vehicles to tours in a specific order
    model.addConstrs((quicksum(x[v, 0, j] for j in range(1, num_delivery_sites + 1)) >= quicksum(x[v + 1, 0, j] for j in range(1, num_delivery_sites + 1)) for v in range(num_vehicles - 1)), name='symmetry_breaking')

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
    delivery_sites = [site['id'] for site in instance['delivery_sites']]
    customers = [customer['id'] for customer in instance['customers']]

    x = model.getAttr('X', model.getVars())
    x_dict = {(v, i, j): x_val for (v, i, j), x_val in zip([(v, i, j) for v in range(num_vehicles) for i in range(num_delivery_sites + 1) for j in range(num_delivery_sites + 1)], x)}

    y = model.getAttr('X', model.getVars())
    y_dict = {(c, d): y_val for (c, d), y_val in zip([(c, d) for c in customers for d in delivery_sites], model.getAttr('X', [var for var in model.getVars() if var.VarName.startswith('y')]))}

    z = model.getAttr('X', model.getVars())
    z_dict = {(v, d): z_val for (v, d), z_val in zip([(v, d) for v in range(num_vehicles) for d in delivery_sites], model.getAttr('X', [var for var in model.getVars() if var.VarName.startswith('z')]))}

    s = model.getAttr('X', model.getVars())
    s_dict = {(v, c, d): s_val for (v, c, d), s_val in zip([(v, c, d) for v in range(num_vehicles) for c in customers for d in delivery_sites], model.getAttr('X', [var for var in model.getVars() if var.VarName.startswith('s')]))}

    solution = {
        'objective_value': model.getObjective().getValue(),
        'routes': {},
        'assignments': {},
        'customer_tours': {}
    }

    vehicle_id = 0
    for v in range(num_vehicles):
        route = []
        sequence = []
        current_node = depot_id
        while True:
            found_next = False
            for j in range(num_delivery_sites + 1):
                if x_dict.get((v, current_node, j), 0) > 0.5:
                    route.append([current_node, j])
                    sequence.append(j)
                    current_node = j
                    found_next = True
                    break
            if not found_next:
                break
        if len(route) > 0:
            solution['routes'][str(vehicle_id)] = {
                'arcs': route,
                'sequence': [depot_id] + [node for node in sequence if node != depot_id] + [depot_id]
            }
            vehicle_id += 1

    for c in customers:
        for d in delivery_sites:
            if y_dict.get((c, d), 0) > 0.5:
                solution['assignments'][str(c)] = d
                for v in range(num_vehicles):
                    if s_dict.get((v, c, d), 0) > 0.5:
                        solution['customer_tours'][str(c)] = {
                            'vehicle': v,
                            'site': d
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