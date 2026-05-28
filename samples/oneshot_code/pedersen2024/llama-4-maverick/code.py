import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    return instance_data

def solve(instance_data, time_limit, log_path=None):
    num_nodes = instance_data['parameters']['total_nodes']
    num_edges = instance_data['parameters']['total_edges']
    nodes = instance_data['nodes']
    edges = instance_data['edges']
    quota = instance_data['parameters']['quota']
    alpha = instance_data['parameters']['alpha']

    # Create Gurobi model
    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogFile = 'gurobi.log'

    # Variables
    x = {}  # Binary variable for selecting edges
    for edge in edges:
        x[edge['id']] = model.addVar(vtype=GRB.BINARY, name=f'x_{edge["id"]}')

    y = {}  # Binary variable for selecting nodes (turbines)
    for node in nodes:
        y[node['id']] = model.addVar(vtype=GRB.BINARY, name=f'y_{node["id"]}')

    # Objective function
    total_cost = gp.quicksum(x[edge['id']] * edge['cost'] for edge in edges) + \
                 gp.quicksum(y[node['id']] * node['cost'] for node in nodes if node['type'] == 'potential_turbine')
    total_scenic_impact = gp.quicksum(x[edge['id']] * edge['scenic_impact'] for edge in edges) + \
                          gp.quicksum(y[node['id']] * node['scenic_impact'] for node in nodes if node['type'] == 'potential_turbine')
    objective = alpha * total_cost + (1 - alpha) * total_scenic_impact
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # 1. Energy quota
    model.addConstr(gp.quicksum(y[node['id']] * node['profit'] for node in nodes if node['type'] == 'potential_turbine') >= quota)

    # 2. Connectivity and tree structure
    substations = [node['id'] for node in nodes if node['type'] == 'substation']
    for substation in substations:
        model.addConstr(gp.quicksum(x[edge['id']] for edge in edges if edge['from'] == substation or edge['to'] == substation) >= 1)

    # Flow-based connectivity constraints
    flow = {}
    for edge in edges:
        flow[edge['id']] = model.addVar(lb=0, ub=num_nodes, vtype=GRB.CONTINUOUS, name=f'flow_{edge["id"]}')
    for node in nodes:
        if node['type'] == 'substation':
            model.addConstr(gp.quicksum(flow[edge['id']] for edge in edges if edge['to'] == node['id']) -
                            gp.quicksum(flow[edge['id']] for edge in edges if edge['from'] == node['id']) ==
                            gp.quicksum(y[i] for i in [n['id'] for n in nodes]) - y[node['id']])
        else:
            model.addConstr(gp.quicksum(flow[edge['id']] for edge in edges if edge['to'] == node['id']) -
                            gp.quicksum(flow[edge['id']] for edge in edges if edge['from'] == node['id']) ==
                            y[node['id']] - y[node['id']])
    for edge in edges:
        model.addConstr(flow[edge['id']] <= num_nodes * x[edge['id']])

    # 3. Selected turbines are connected by selected edges
    for node in nodes:
        if node['type'] == 'potential_turbine':
            model.addConstr(y[node['id']] <= gp.quicksum(x[edge['id']] for edge in edges if edge['from'] == node['id'] or edge['to'] == node['id']))

    # Optimize
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                logger.log(model.getObjective())
    model.optimize(callback)

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected_turbines = [node['id'] for node in nodes if node['id'] in y and y[node['id']].x > 0.5]
        selected_edges = [{'from': edge['from'], 'to': edge['to']} for edge in edges if edge['id'] in x and x[edge['id']].x > 0.5]
        objective_value = model.getObjective().getValue()
        return objective_value, selected_turbines, selected_edges
    else:
        return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    instance_data = read_instance(args.instance_path)
    objective_value, selected_turbines, selected_edges = solve(instance_data, args.time_limit, args.log_path)

    if objective_value is not None:
        solution = {
            'objective_value': objective_value,
            'selected_turbines': selected_turbines,
            'selected_edges': selected_edges
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    main()