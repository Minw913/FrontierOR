import json
import argparse
import numpy as np
from math import sqrt
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def compute_travel_cost(instance, i, j):
    node_i = instance['nodes'][i]
    node_j = instance['nodes'][j]
    x_i, y_i = node_i['x'], node_i['y']
    x_j, y_j = node_j['x'], node_j['y']
    return sqrt((x_i - x_j)**2 + (y_i - y_j)**2)

def compute_travel_time(instance, i, j):
    node_i = instance['nodes'][i]
    node_j = instance['nodes'][j]
    x_i, y_i = node_i['x'], node_i['y']
    x_j, y_j = node_j['x'], node_j['y']
    return sqrt((x_i - x_j)**2 + (y_i - y_j)**2)

def is_feasible_arc(instance, i, j):
    nodes = instance['nodes']
    vehicle_capacity = instance['problem_parameters']['vehicle_capacity']
    demand_i = nodes[i]['demand']
    demand_j = nodes[j]['demand']
    if demand_i + demand_j > vehicle_capacity:
        return False
    
    travel_time = compute_travel_time(instance, i, j)
    service_time_i = nodes[i]['service_time']
    time_window_open_j = nodes[j]['time_window_open']
    time_window_close_j = nodes[j]['time_window_close']
    earliest_start_i = nodes[i]['time_window_open']
    latest_arrival_j = earliest_start_i + service_time_i + travel_time
    if latest_arrival_j > time_window_close_j:
        return False
    
    return True

def build_model(instance):
    n_nodes = instance['problem_parameters']['n_nodes']
    n_customers = instance['problem_parameters']['n_customers']
    vehicle_capacity = instance['problem_parameters']['vehicle_capacity']
    planning_horizon = instance['problem_parameters']['planning_horizon']
    nodes = instance['nodes']

    model = gp.Model('VRPTW')
    model.modelSense = GRB.MINIMIZE

    # Variables
    x = model.addVars(n_nodes, n_nodes, vtype=GRB.BINARY, name='x')
    s = model.addVars(n_nodes, vtype=GRB.CONTINUOUS, name='s')

    # Objective function
    model.setObjective(gp.quicksum(compute_travel_cost(instance, i, j) * x[i, j] for i in range(n_nodes) for j in range(n_nodes)))

    # Constraints
    depot_source = 0
    depot_sink = n_nodes - 1

    # Flow conservation constraints
    model.addConstrs((gp.quicksum(x[i, k] for i in range(n_nodes)) == gp.quicksum(x[k, j] for j in range(n_nodes)) for k in range(n_nodes)), name='flow_conservation')
    
    # Customer visit exactly once
    model.addConstrs((gp.quicksum(x[i, k] for i in range(n_nodes)) == 1 for k in range(1, n_customers + 1)), name='visit_once')

    # Depot source and sink constraints
    model.addConstr(gp.quicksum(x[depot_source, j] for j in range(1, n_nodes)) <= n_customers, name='depot_source_out')
    model.addConstr(gp.quicksum(x[i, depot_sink] for i in range(n_nodes - 1)) <= n_customers, name='depot_sink_in')

    # Time window and service time constraints
    M = 1e6
    model.addConstrs((s[i] + nodes[i]['service_time'] + compute_travel_time(instance, i, j) <= s[j] + M * (1 - x[i, j]) for i in range(n_nodes) for j in range(n_nodes)), name='time_window')
    model.addConstrs((s[i] >= nodes[i]['time_window_open'] for i in range(n_nodes)), name='time_window_open')
    model.addConstrs((s[i] <= nodes[i]['time_window_close'] for i in range(n_nodes)), name='time_window_close')

    # Capacity and feasible arc constraints
    model.addConstrs((x[i, j] == 0 for i in range(n_nodes) for j in range(n_nodes) if not is_feasible_arc(instance, i, j)), name='feasible_arc')

    # Vehicle return to depot within planning horizon
    model.addConstrs((s[i] + nodes[i]['service_time'] + compute_travel_time(instance, i, depot_sink) <= planning_horizon + M * (1 - x[i, depot_sink]) for i in range(n_nodes)), name='return_to_depot')

    # Subtour elimination constraints
    def subtour_elimination(model, where):
        if where == GRB.Callback.MIPSOL:
            x_sol = model.cbGetSolution(model._x)
            visited = set()
            to_visit = [0]
            while to_visit:
                node = to_visit.pop()
                visited.add(node)
                for j in range(n_nodes):
                    if x_sol[node, j] > 0.5 and j not in visited:
                        to_visit.append(j)
            if len(visited) < n_customers + 1:
                model.cbLazy(gp.quicksum(model._x[i, j] for i in visited for j in visited) <= len(visited) - 1)

    model._x = x
    model.Params.lazyConstraints = 1

    return model

def extract_solution(instance, model):
    n_nodes = instance['problem_parameters']['n_nodes']
    x = model.getAttr('X', model.getVars())[:n_nodes*n_nodes]
    x = np.reshape(x, (n_nodes, n_nodes))

    routes = []
    depot_source = 0
    depot_sink = n_nodes - 1

    for _ in range(int(np.sum(x[depot_source, :]))):
        route = [depot_source]
        current_node = depot_source
        while current_node != depot_sink:
            for j in range(n_nodes):
                if x[current_node, j] > 0.5:
                    route.append(j)
                    current_node = j
                    break
        routes.append(route)

    objective_value = model.ObjVal

    return objective_value, routes

def solve_instance(instance, time_limit, log_path=None):
    model = build_model(instance)

    logger = SolutionLogger(log_path, sense='minimize') if log_path else None

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value, routes = extract_solution(instance, model)
        return objective_value, routes
    else:
        return None, None

def write_solution(solution_path, objective_value, routes):
    solution = {
        'objective_value': objective_value,
        'routes': [[int(node) for node in route] for route in routes]
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)

    objective_value, routes = solve_instance(instance, args.time_limit, args.log_path)

    if objective_value is not None and routes is not None:
        write_solution(args.solution_path, objective_value, routes)
    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    main()