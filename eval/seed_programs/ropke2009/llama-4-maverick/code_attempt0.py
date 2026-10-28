import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit, logger):
    num_requests = instance['num_requests']
    num_nodes = instance['num_nodes']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    nodes = instance['nodes']
    requests = instance['requests']
    depot = instance['depot']

    model = gp.Model("PDPTW")
    model.Params.LogToConsole = 0  # Disable Gurobi logging to console

    # Sets
    N = range(num_nodes)  # All nodes including depot
    V = range(num_vehicles)  # Vehicles
    P = [req['pickup_node'] for req in requests]  # Pickup nodes
    D = [req['delivery_node'] for req in requests]  # Delivery nodes
    PD = P + D  # Pickup and delivery nodes

    # Parameters
    Q = vehicle_capacity
    C = [[gp.quicksum([nodes[i]['x'] - nodes[j]['x'], nodes[i]['y'] - nodes[j]['y']])**0.5 for j in N] for i in N]  # Cost matrix
    T = [[model.addVar(lb=0) for _ in N] for _ in V]  # Time variables
    L = [[model.addVar(lb=0, ub=Q) for _ in N] for _ in V]  # Load variables
    x = model.addVars(V, N, N, vtype=GRB.BINARY)  # Routing variables

    # Objective: Minimize total routing cost
    model.setObjective(gp.quicksum(x[v, i, j] * C[i][j] for v in V for i in N for j in N), GRB.MINIMIZE)

    # Constraints
    # Each request is served exactly once
    for i in P:
        model.addConstr(gp.quicksum(x[v, i, j] for v in V for j in N if j != i) == 1)
    for i in D:
        model.addConstr(gp.quicksum(x[v, i, j] for v in V for j in N if j != i) == 1)

    # Pickup and delivery are served by the same vehicle
    for r in range(num_requests):
        p = requests[r]['pickup_node']
        d = requests[r]['delivery_node']
        for v in V:
            model.addConstr(gp.quicksum(x[v, p, j] for j in N if j != p) == gp.quicksum(x[v, d, j] for j in N if j != d))

    # Flow conservation
    for v in V:
        for i in N:
            model.addConstr(gp.quicksum(x[v, i, j] for j in N if j != i) == gp.quicksum(x[v, j, i] for j in N if j != i))

    # Time window constraints
    for v in V:
        for i in N:
            model.addConstr(T[v][i] >= nodes[i]['time_window'][0])
            model.addConstr(T[v][i] <= nodes[i]['time_window'][1])

    # Time consistency
    for v in V:
        for i in N:
            for j in N:
                if i != j:
                    model.addConstr(T[v][j] >= T[v][i] + nodes[i]['service_time'] + C[i][j] - (1 - x[v, i, j]) * (depot['time_window'][1] - depot['time_window'][0]))

    # Load consistency and capacity
    for v in V:
        for i in N:
            for j in N:
                if i != j:
                    model.addConstr(L[v][j] >= L[v][i] + nodes[j]['demand'] - (1 - x[v, i, j]) * Q)
                    model.addConstr(L[v][j] <= L[v][i] + nodes[j]['demand'] + (1 - x[v, i, j]) * Q)
            model.addConstr(L[v][i] >= 0)
            model.addConstr(L[v][i] <= Q)

    # Pickup before delivery
    for r in range(num_requests):
        p = requests[r]['pickup_node']
        d = requests[r]['delivery_node']
        for v in V:
            model.addConstr(gp.quicksum(T[v][p] for j in N if x[v, p, j].x > 0.5) <= gp.quicksum(T[v][d] for j in N if x[v, d, j].x > 0.5))

    # Depot time window for all vehicles
    depot_start = depot['time_window'][0]
    depot_end = depot['time_window'][1]
    for v in V:
        model.addConstr(T[v][0] >= depot_start)  # Origin depot
        model.addConstr(T[v][num_nodes-1] <= depot_end)  # Destination depot

    model.Params.TimeLimit = time_limit
    model.optimize()

    if logger:
        logger.log(model.getObjective().getValue())

    # Extract solution
    routes = []
    for v in V:
        route = [0]
        current_node = 0
        while True:
            next_node = None
            for j in N:
                if j != current_node and x[v, current_node, j].x > 0.5:
                    next_node = j
                    break
            if next_node is None:
                break
            route.append(next_node)
            current_node = next_node
        if len(route) > 2:  # If vehicle serves any requests
            routes.append({"vehicle": v, "nodes": route})

    objective_value = model.getObjective().getValue()

    return objective_value, routes

def write_solution(solution_path, objective_value, routes):
    solution = {
        "objective_value": objective_value,
        "routes": routes
    }
    with open(solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='PDPTW Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, routes = solve(instance, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, routes)

if __name__ == "__main__":
    main()