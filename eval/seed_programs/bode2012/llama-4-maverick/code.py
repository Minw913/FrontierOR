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

    model = gp.Model('MCPCARP')
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    x = model.addVars(num_required_edges, num_vehicles, vtype=GRB.BINARY, name='x')  
    y = model.addVars(num_edges, num_vehicles, vtype=GRB.INTEGER, name='y')  
    f = model.addVars(num_edges, num_vehicles, vtype=GRB.INTEGER, name='f')  

    obj = gp.quicksum(edges[i]['cost'] * y[i, k] for i in range(num_edges) for k in range(num_vehicles))
    model.setObjective(obj)

    for i, required_edge in enumerate(required_edges):
        model.addConstr(gp.quicksum(x[i, k] for k in range(num_vehicles)) == 1, name=f'required_edge_assignment_{i}')

    for k in range(num_vehicles):
        model.addConstr(gp.quicksum(required_edges[i]['demand'] * x[i, k] for i in range(num_required_edges)) <= vehicle_capacity, name=f'vehicle_capacity_{k}')

    for k in range(num_vehicles):
        for i, edge in enumerate(edges):
            u, v = edge['endpoints']
            if edge['is_required']:
                model.addConstr(y[i, k] >= x[required_edges.index(edge), k], name=f'required_edge_traversal_{i}_{k}')

    model.update()
    return model, x, y

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    model, x, y = solve_mcpcarp(instance)

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0  

    model.optimize()

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

    if logger:
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            logger.log(model.PoolObjVal)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()