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
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  

    N = list(range(num_nodes))  
    V = list(range(num_vehicles))  
    P = [req['pickup_node'] for req in requests]  
    D = [req['delivery_node'] for req in requests]  
    PD = P + D  

    Q = vehicle_capacity
    dist = lambda x1, y1, x2, y2: ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    C = [[dist(nodes[i]['x'], nodes[i]['y'], nodes[j]['x'], nodes[j]['y']) for j in N] for i in N]  
    x = model.addVars(V, N, N, vtype=GRB.BINARY)  
    T = model.addVars(V, N, lb=0)  
    L = model.addVars(V, N, lb=0, ub=Q)  

    model.setObjective(gp.quicksum(x[v, i, j] * C[i][j] for v in V for i in N for j in N), GRB.MINIMIZE)

    for r in range(num_requests):
        p = requests[r]['pickup_node']
        d = requests[r]['delivery_node']
        model.addConstr(gp.quicksum(x[v, p, j] for v in V for j in N if j != p) == 1)
        model.addConstr(gp.quicksum(x[v, d, j] for v in V for j in N if j != d) == 1)
        for v in V:
            model.addConstr(gp.quicksum(x[v, p, j] for j in N if j != p) - gp.quicksum(x[v, i, d] for i in N if i != d) == 0)

    for v in V:
        model.addConstr(gp.quicksum(x[v, 0, j] for j in N if j != 0) <= 1)  
        model.addConstr(gp.quicksum(x[v, j, num_nodes-1] for j in N if j != num_nodes-1) <= 1)  
        model.addConstr(gp.quicksum(x[v, j, 0] for j in N if j != 0) == 0)  
        model.addConstr(gp.quicksum(x[v, num_nodes-1, j] for j in N if j != num_nodes-1) == 0)  
        for i in N:
            model.addConstr(gp.quicksum(x[v, j, i] for j in N if j != i) - gp.quicksum(x[v, i, j] for j in N if j != i) == 0)

    for v in V:
        for i in N:
            model.addConstr(T[v, i] >= nodes[i]['time_window'][0])
            model.addConstr(T[v, i] <= nodes[i]['time_window'][1])
            for j in N:
                if i != j:
                    M = max(depot['time_window'][1], nodes[j]['time_window'][1]) - min(depot['time_window'][0], nodes[i]['time_window'][0])
                    model.addConstr(T[v, j] >= T[v, i] + nodes[i]['service_time'] + C[i][j] - M * (1 - x[v, i, j]))
                    model.addConstr(L[v, j] >= L[v, i] + nodes[j]['demand'] - Q * (1 - x[v, i, j]))
                    model.addConstr(L[v, j] <= L[v, i] + nodes[j]['demand'] + Q * (1 - x[v, i, j]))

    for r in range(num_requests):
        p = requests[r]['pickup_node']
        d = requests[r]['delivery_node']
        for v in V:
            model.addConstr(T[v, d] >= T[v, p] + requests[r]['travel_time_pickup_to_delivery'])

    model.Params.TimeLimit = time_limit
    model.optimize()

    if logger and model.SolCount > 0:
        logger.log(model.getObjective().getValue())

    routes = []
    for v in V:
        route = []
        for i in N:
            for j in N:
                if x[v, i, j].x > 0.5:
                    if len(route) == 0:
                        route.append(i)
                    route.append(j)
        if len(route) > 1 and route[0] == 0 and route[-1] == num_nodes - 1:  
            routes.append({"vehicle": v, "nodes": route})

    objective_value = model.getObjective().getValue() if model.SolCount > 0 else float('inf')

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