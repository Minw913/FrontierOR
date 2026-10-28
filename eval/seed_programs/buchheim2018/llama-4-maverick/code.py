import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit):
    num_arcs = instance['num_arcs']
    source_node = instance['source_node']
    target_node = instance['target_node']
    arcs = instance['arcs']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']

    # Create a Gurobi model
    model = gp.Model()
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  # Silence Gurobi output
    model.Params.TimeLimit = time_limit  # Set time limit

    # Arc selection variables
    x = model.addVars(num_arcs, vtype=GRB.BINARY, name='x')

    # Objective function: linear costs + quadratic interaction costs
    linear_cost = gp.quicksum(linear_costs[i] * x[i] for i in range(num_arcs))
    quadratic_cost = gp.quicksum(quadratic_costs[i][j] * x[i] * x[j] for i in range(num_arcs) for j in range(num_arcs))
    model.setObjective(linear_cost + quadratic_cost, GRB.MINIMIZE)

    # Flow conservation constraints
    node_in_arcs = {node['id']: [] for node in instance['nodes']}
    node_out_arcs = {node['id']: [] for node in instance['nodes']}
    for arc in arcs:
        node_out_arcs[arc['from_node']].append(arc['id'])
        node_in_arcs[arc['to_node']].append(arc['id'])

    for node in instance['nodes']:
        node_id = node['id']
        if node_id == source_node:
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_out_arcs[node_id]) == 1, f"source_{node_id}")
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_in_arcs[node_id]) == 0, f"source_in_{node_id}")
        elif node_id == target_node:
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_in_arcs[node_id]) == 1, f"target_{node_id}")
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_out_arcs[node_id]) == 0, f"target_out_{node_id}")
        else:
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_in_arcs[node_id]) - gp.quicksum(x[arc_id] for arc_id in node_out_arcs[node_id]) == 0, f"flow_{node_id}")

    # Optimize the model
    model.optimize()

    # Extract solution
    objective_value = model.getObjective().getValue()
    solution_arcs = []
    if model.SolCount > 0:
        for arc in arcs:
            arc_id = arc['id']
            if x[arc_id].x > 0.5:  # Considered selected if value is greater than 0.5
                solution_arcs.append({
                    'id': arc['id'],
                    'from_node': arc['from_node'],
                    'to_node': arc['to_node']
                })

    return objective_value, solution_arcs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback function to log intermediate solutions
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    # Solve instance with callback
    model = gp.Model()
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = args.time_limit
    num_arcs = instance['num_arcs']
    source_node = instance['source_node']
    target_node = instance['target_node']
    arcs = instance['arcs']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']

    x = model.addVars(num_arcs, vtype=GRB.BINARY, name='x')
    linear_cost = gp.quicksum(linear_costs[i] * x[i] for i in range(num_arcs))
    quadratic_cost = gp.quicksum(quadratic_costs[i][j] * x[i] * x[j] for i in range(num_arcs) for j in range(num_arcs))
    model.setObjective(linear_cost + quadratic_cost, GRB.MINIMIZE)

    node_in_arcs = {node['id']: [] for node in instance['nodes']}
    node_out_arcs = {node['id']: [] for node in instance['nodes']}
    for arc in arcs:
        node_out_arcs[arc['from_node']].append(arc['id'])
        node_in_arcs[arc['to_node']].append(arc['id'])

    for node in instance['nodes']:
        node_id = node['id']
        if node_id == source_node:
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_out_arcs[node_id]) == 1, f"source_{node_id}")
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_in_arcs[node_id]) == 0, f"source_in_{node_id}")
        elif node_id == target_node:
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_in_arcs[node_id]) == 1, f"target_{node_id}")
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_out_arcs[node_id]) == 0, f"target_out_{node_id}")
        else:
            model.addConstr(gp.quicksum(x[arc_id] for arc_id in node_in_arcs[node_id]) - gp.quicksum(x[arc_id] for arc_id in node_out_arcs[node_id]) == 0, f"flow_{node_id}")

    model.optimize(log_callback)

    objective_value = model.getObjective().getValue()
    solution_arcs = []
    if model.SolCount > 0:
        for arc in arcs:
            arc_id = arc['id']
            if x[arc_id].x > 0.5:
                solution_arcs.append({
                    'id': arc['id'],
                    'from_node': arc['from_node'],
                    'to_node': arc['to_node']
                })

    solution = {
        'objective_value': objective_value,
        'solution_arcs': solution_arcs
    }

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()