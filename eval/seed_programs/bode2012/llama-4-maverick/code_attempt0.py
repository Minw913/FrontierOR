import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve_mcpcarp(instance):
    graph = instance['graph']
    num_nodes = graph['num_nodes']
    num_edges = graph['num_edges']
    depot = instance['depot']
    num_vehicles = instance['fleet']['num_vehicles']
    vehicle_capacity = instance['fleet']['vehicle_capacity']
    edges = instance['edges']
    required_edges = [edge for edge in edges if edge['is_required']]
    num_required_edges = len(required_edges)
    edge_index = {edge['edge_id']: i for i, edge in enumerate(edges)}

    # Create a Gurobi model
    model = gp.Model('MCPCARP')
    model.modelSense = GRB.MINIMIZE

    # Variables
    x = model.addVars(num_required_edges, num_vehicles, vtype=GRB.BINARY, name='x')  # Assignment of required edges to vehicles
    y = model.addVars(num_edges, num_vehicles, vtype=GRB.INTEGER, name='y')  # Traversal count of edges by vehicles
    f = model.addVars(num_edges, num_vehicles, vtype=GRB.INTEGER, name='f')  # Flow variable to ensure connectivity

    # Objective: minimize total deadheading cost
    obj = gp.quicksum(edges[i]['cost'] * y[i, k] for i in range(num_edges) for k in range(num_vehicles))
    model.setObjective(obj)

    # Constraints
    # 1. Each required edge is serviced by exactly one vehicle
    for i, required_edge in enumerate(required_edges):
        model.addConstr(gp.quicksum(x[i, k] for k in range(num_vehicles)) == 1, name=f'required_edge_assignment_{i}')

    # 2. Vehicle capacity constraints
    for k in range(num_vehicles):
        model.addConstr(gp.quicksum(required_edges[i]['demand'] * x[i, k] for i in range(num_required_edges)) <= vehicle_capacity, name=f'vehicle_capacity_{k}')

    # 3. Flow conservation and connectivity constraints
    for k in range(num_vehicles):
        for i, edge in enumerate(edges):
            u, v = edge['endpoints']
            if u == depot:
                model.addConstr(f[i, k] <= y[i, k], name=f'flow_depot_{i}_{k}')
            if v == depot:
                model.addConstr(f[i, k] >= -y[i, k], name=f'flow_depot_{i}_{k}')
            for j, other_edge in enumerate(edges):
                if u == other_edge['endpoints'][0] or u == other_edge['endpoints'][1]:
                    model.addConstr(gp.quicksum(f[j, k]) >= f[i, k] - num_nodes * y[i, k], name=f'flow_conservation_{i}_{k}')

        # Ensure that required edges assigned to a vehicle are connected to the depot
        for i, required_edge in enumerate(required_edges):
            model.addConstr(y[edge_index[required_edge['edge_id']], k] >= x[i, k], name=f'required_edge_traversal_{i}_{k}')

    # 4. Additional constraints to ensure y variables are correctly set for required edges
    for i, required_edge in enumerate(required_edges):
        for k in range(num_vehicles):
            model.addConstr(y[edge_index[required_edge['edge_id']], k] >= x[i, k], name=f'required_edge_traversal_{i}_{k}')

    return model, x, y

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    model, x, y = solve_mcpcarp(instance)

    # Set time limit
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi logging to console

    # Optimize the model
    model.optimize()

    # Extract solution
    edges = instance['edges']
    edge_index = {edge['edge_id']: i for i, edge in enumerate(edges)}
    required_edges = [edge for edge in edges if edge['is_required']]
    num_vehicles = instance['fleet']['num_vehicles']

    solution = {'routes': [], 'objective_value': model.ObjVal if model.SolCount > 0 else float('inf')}

    if model.SolCount > 0:
        for k in range(num_vehicles):
            route = {'vehicle': k, 'serviced_edges': [], 'deadheaded_edges': []}
            for i, required_edge in enumerate(required_edges):
                if x[i, k].X > 0.5:
                    route['serviced_edges'].append(required_edge['edge_id'])
            for i, edge in enumerate(edges):
                deadheading_count = int(y[i, k].X)
                if deadheading_count > 0:
                    route['deadheaded_edges'].append({'edge_id': edge['edge_id'], 'times': deadheading_count})
            solution['routes'].append(route)

    # Log intermediate solutions
    if logger:
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            logger.log(model.PoolObjVal)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()