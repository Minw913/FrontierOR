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

def solve(instance):
    num_nodes = instance['num_nodes']
    num_edges = instance['num_edges']
    bisection_capacity = instance['bisection_capacity_F']
    node_weights = instance['node_weights']
    edges = instance['edges']
    edge_weights = instance['edge_weights']

    model = gp.Model()
    model.Params.LogToConsole = 0  # Disable console logging
    model.Params.TimeLimit = args.time_limit  # Set time limit

    # Node assignment variables
    node_assign = model.addVars(num_nodes, vtype=GRB.BINARY, name='node_assign')

    # Edge cut variables
    edge_cut = model.addVars(num_edges, vtype=GRB.BINARY, name='edge_cut')

    # Objective: minimize total cost of cut edges
    model.setObjective(gp.quicksum(edge_weights[e] * edge_cut[e] for e in range(num_edges)), GRB.MINIMIZE)

    # Constraints: node weights in each partition do not exceed bisection capacity
    model.addConstr(gp.quicksum(node_weights[n] * node_assign[n] for n in range(num_nodes)) <= bisection_capacity, 'capacity_S')
    model.addConstr(gp.quicksum(node_weights[n] * (1 - node_assign[n]) for n in range(num_nodes)) <= bisection_capacity, 'capacity_complement')

    # Constraints: edge cut logic
    for e in range(num_edges):
        i, j = edges[e]
        model.addConstr(edge_cut[e] >= node_assign[i] - node_assign[j], f'edge_cut_logic_{e}_1')
        model.addConstr(edge_cut[e] >= node_assign[j] - node_assign[i], f'edge_cut_logic_{e}_2')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        partition_S = [n for n in range(num_nodes) if node_assign[n].X >= 0.5]
        partition_complement = [n for n in range(num_nodes) if node_assign[n].X < 0.5]
        return objective_value, partition_S, partition_complement
    else:
        raise Exception(f'Gurobi model failed to find a feasible solution. Status: {model.status}')

def main():
    global args
    args = parse_args()
    instance = read_instance(args.instance_path)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def log_callback(model, where):
        if where == GRB.Callback.MIP:
            current_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            if model._log_obj != current_obj:
                model._log_obj = current_obj
                if logger:
                    logger.log(current_obj)

    model = gp.Model()
    model._log_obj = float('inf')
    num_nodes = instance['num_nodes']
    num_edges = instance['num_edges']
    bisection_capacity = instance['bisection_capacity_F']
    node_weights = instance['node_weights']
    edges = instance['edges']
    edge_weights = instance['edge_weights']

    node_assign = model.addVars(num_nodes, vtype=GRB.BINARY, name='node_assign')
    edge_cut = model.addVars(num_edges, vtype=GRB.BINARY, name='edge_cut')

    model.setObjective(gp.quicksum(edge_weights[e] * edge_cut[e] for e in range(num_edges)), GRB.MINIMIZE)

    model.addConstr(gp.quicksum(node_weights[n] * node_assign[n] for n in range(num_nodes)) <= bisection_capacity, 'capacity_S')
    model.addConstr(gp.quicksum(node_weights[n] * (1 - node_assign[n]) for n in range(num_nodes)) <= bisection_capacity, 'capacity_complement')

    for e in range(num_edges):
        i, j = edges[e]
        model.addConstr(edge_cut[e] >= node_assign[i] - node_assign[j], f'edge_cut_logic_{e}_1')
        model.addConstr(edge_cut[e] >= node_assign[j] - node_assign[i], f'edge_cut_logic_{e}_2')

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0
    model.optimize(callback=log_callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        partition_S = [n for n in range(num_nodes) if node_assign[n].X >= 0.5]
        partition_complement = [n for n in range(num_nodes) if node_assign[n].X < 0.5]

        solution = {
            'objective_value': objective_value,
            'partition_S': partition_S,
            'partition_complement': partition_complement
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        raise Exception(f'Gurobi model failed to find a feasible solution. Status: {model.status}')

if __name__ == '__main__':
    main()