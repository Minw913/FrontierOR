import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve_instance(instance):
    num_customers = instance['num_customers']
    num_periods = instance['num_periods']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    num_nodes = instance['num_nodes']
    distance_matrix = instance['distance_matrix']
    holding_costs = instance['holding_costs']
    customer_demands = instance['customer_demands']
    supplier_production = instance['supplier_production']
    inventory_capacities = instance['inventory_capacities']
    initial_inventories = instance['initial_inventories']

    model = Model('IRP')
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    # Decision Variables
    visit = model.addVars(num_periods, num_customers, num_vehicles, vtype=GRB.BINARY, name='visit')
    quantity = model.addVars(num_periods, num_customers, num_vehicles, vtype=GRB.CONTINUOUS, name='quantity')
    route = model.addVars(num_periods, num_vehicles, num_nodes, num_nodes, vtype=GRB.BINARY, name='route')
    inventory_depot = model.addVars(num_periods, vtype=GRB.CONTINUOUS, name='inventory_depot')
    inventory_customer = model.addVars(num_periods, num_customers, vtype=GRB.CONTINUOUS, name='inventory_customer')

    # Objective Function
    model.setObjective(
        quicksum(distance_matrix[i][j] * route[t, v, i, j] for t in range(num_periods) for v in range(num_vehicles) for i in range(num_nodes) for j in range(num_nodes)) +
        quicksum(holding_costs[i] * inventory_customer[t, i-1] for t in range(num_periods) for i in range(1, num_nodes)) +
        quicksum(holding_costs[0] * inventory_depot[t] for t in range(num_periods)),
        GRB.MINIMIZE
    )

    # Constraints
    for t in range(num_periods):
        for i in range(1, num_nodes):
            model.addConstr(inventory_customer[t, i-1] == (initial_inventories[i] if t == 0 else inventory_customer[t-1, i-1]) + quicksum(quantity[t, i-1, v] for v in range(num_vehicles)) - customer_demands[i-1][t])
            model.addConstr(inventory_customer[t, i-1] >= 0)
            model.addConstr(inventory_customer[t, i-1] <= inventory_capacities[i])

        model.addConstr(inventory_depot[t] == (initial_inventories[0] if t == 0 else inventory_depot[t-1]) + supplier_production[t] - quicksum(quantity[t, i, v] for i in range(num_customers) for v in range(num_vehicles)))
        model.addConstr(inventory_depot[t] >= 0)
        model.addConstr(inventory_depot[t] <= inventory_capacities[0])

        for v in range(num_vehicles):
            model.addConstr(quicksum(route[t, v, 0, j] for j in range(1, num_nodes)) == quicksum(route[t, v, j, 0] for j in range(1, num_nodes)))
            model.addConstr(quicksum(route[t, v, 0, j] for j in range(1, num_nodes)) <= 1)

            for i in range(1, num_nodes):
                model.addConstr(quicksum(route[t, v, i, j] for j in range(num_nodes)) == quicksum(route[t, v, j, i] for j in range(num_nodes)))

            model.addConstr(quicksum(quantity[t, i, v] for i in range(num_customers)) <= vehicle_capacity)

            for i in range(num_customers):
                model.addConstr(quantity[t, i, v] <= vehicle_capacity * visit[t, i, v])
                model.addConstr(visit[t, i, v] == quicksum(route[t, v, i+1, j] for j in range(num_nodes)))

        for i in range(num_customers):
            model.addConstr(quicksum(visit[t, i, v] for v in range(num_vehicles)) <= 1)

    # Subtour elimination
    def subtour_elimination(model, where):
        if where == GRB.Callback.MIPSOL:
            for t in range(num_periods):
                for v in range(num_vehicles):
                    edges = []
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            if route[t, v, i, j].x > 0.5:
                                edges.append((i, j))
                    nodes = set()
                    for edge in edges:
                        nodes.add(edge[0])
                        nodes.add(edge[1])
                    if len(nodes) > 0:
                        for S in generate_subsets(list(nodes)):
                            if len(S) > 0 and 0 not in S:
                                lhs = quicksum(route[t, v, i, j] for i in S for j in S)
                                model.cbLazy(lhs <= len(S) - 1)

    model.Params.LazyConstraints = 1
    model.optimize(subtour_elimination)

    # Extract solution
    objective_value = model.getObjective().getValue()
    routes = {}
    inventories = {}

    for t in range(num_periods):
        routes[t] = {}
        for v in range(num_vehicles):
            route_sequence = []
            delivery_quantities = {}
            if sum([route[t, v, 0, j].x for j in range(num_nodes)]) > 0.5:
                current_node = 0
                route_sequence.append(current_node)
                next_node = get_next_node(route, t, v, current_node, num_nodes)
                while next_node != 0:
                    route_sequence.append(next_node)
                    if next_node != 0:
                        delivery_quantities[next_node] = quantity[t, next_node-1, v].x
                    current_node = next_node
                    next_node = get_next_node(route, t, v, current_node, num_nodes)
                route_sequence.append(0)
                routes[t][v] = {'route': route_sequence, 'deliveries': delivery_quantities}

    for i in range(num_nodes):
        inventories[i] = {}
        for t in range(num_periods):
            if i == 0:
                inventories[i][t] = inventory_depot[t].x
            else:
                inventories[i][t] = inventory_customer[t, i-1].x

    return objective_value, routes, inventories

def get_next_node(route, t, v, current_node, num_nodes):
    for j in range(num_nodes):
        if route[t, v, current_node, j].x > 0.5:
            return j

def generate_subsets(S):
    subsets = []
    for i in range(1, 2**len(S)):
        subset = [S[j] for j, bit in enumerate(bin(i)[2:].zfill(len(S))) if bit == '1']
        subsets.append(subset)
    return subsets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve IRP instance')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model('IRP')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = args.time_limit

    objective_value, routes, inventories = solve_instance(instance)

    solution = {
        'objective_value': objective_value,
        'routes': routes,
        'inventories': inventories
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)