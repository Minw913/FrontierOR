import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
from itertools import combinations

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_nodes = instance['num_nodes']
    depot = instance['depot']
    arcs = instance['arcs']
    profitable_arcs = instance['profitable_arcs']
    VI_nodes = instance['VI_nodes']
    strong_incompatibilities = instance['strong_incompatibilities']
    weak_incompatibilities = instance['weak_incompatibilities']

    arc_dict = {(i, j): cost for i, j, cost in arcs}
    profitable_arc_dict = {(i, j): profit for i, j, profit in profitable_arcs}
    weak_incompatibility_dict = {(i, j): penalty for i, j, penalty in weak_incompatibilities}

    model = Model('Profitable Tour Problem')
    model.setParam("Threads", 1)
    model.modelSense = GRB.MAXIMIZE

    # Decision Variables
    x = model.addVars(arc_dict.keys(), vtype=GRB.INTEGER, name='x')  # Number of times each arc is traversed
    y = model.addVars(profitable_arc_dict.keys(), vtype=GRB.BINARY, name='y')  # Whether a profitable arc is served
    z = model.addVars(VI_nodes, vtype=GRB.BINARY, name='z')  # Whether a node in VI_nodes is active
    w = model.addVars(weak_incompatibility_dict.keys(), vtype=GRB.BINARY, name='w')  # Whether a weak incompatibility is overridden

    # Objective Function
    total_profit = quicksum(profit * y[i, j] for (i, j), profit in profitable_arc_dict.items())
    total_cost = quicksum(cost * x[i, j] for (i, j), cost in arc_dict.items())
    total_penalty = quicksum(penalty * w[i, j] for (i, j), penalty in weak_incompatibility_dict.items())
    model.setObjective(total_profit - total_cost - total_penalty)

    # Constraints
    # Flow conservation
    for node in range(num_nodes):
        model.addConstr(quicksum(x[i, j] for i, j in arc_dict if j == node) == quicksum(x[i, j] for i, j in arc_dict if i == node), name=f'flow_conservation_{node}')

    # Profitable arc serving condition
    for i, j in profitable_arc_dict:
        model.addConstr(y[i, j] <= x[i, j], name=f'serve_condition_{i}_{j}')

    # Strong incompatibilities
    for i, j in strong_incompatibilities:
        model.addConstr(z[i] + z[j] <= 1, name=f'strong_incompatibility_{i}_{j}')

    # Weak incompatibilities
    for i, j in weak_incompatibility_dict:
        model.addConstr(z[i] + z[j] <= 1 + w[i, j], name=f'weak_incompatibility_{i}_{j}')

    # Node activation based on profitable arc serving
    for node in VI_nodes:
        model.addConstr(z[node] <= quicksum(y[i, j] for i, j in profitable_arc_dict if i == node), name=f'node_activation_{node}')
        model.addConstr(quicksum(y[i, j] for i, j in profitable_arc_dict if i == node) <= z[node], name=f'node_activation_upper_{node}')

    # Connectivity constraints for served profitable arcs
    nodes = list(range(num_nodes))
    for size in range(1, num_nodes):
        for subset in combinations(nodes, size):
            if depot not in subset:
                arcs_in_S = [(i, j) for i, j in profitable_arc_dict if i in subset and j in subset]
                arcs_leaving_S = [(i, j) for i in subset for j in range(num_nodes) if (i, j) in arc_dict and j not in subset]
                if len(arcs_in_S) > 0:
                    model.addConstr(quicksum(x[i, j] for i, j in arcs_leaving_S) >= quicksum(y[i, j] for i, j in arcs_in_S), name=f'connectivity_{subset}')

    model.setParam('TimeLimit', time_limit)
    model.optimize()

    if logger and model.SolCount > 0:
        logger.log(model.ObjVal)

    if model.SolCount > 0:
        objective_value = model.ObjVal
        served_arcs = []
        for i, j in profitable_arc_dict:
            if y[i, j].X > 0.5:
                served_arcs.append({
                    'from': i,
                    'to': j,
                    'profit': profitable_arc_dict[i, j],
                    'cost': arc_dict.get((i, j), 0)
                })

        tour_arcs = []
        for i, j in arc_dict:
            count = int(x[i, j].X)
            if count > 0:
                tour_arcs.append({
                    'from': i,
                    'to': j,
                    'count': count,
                    'cost': arc_dict[i, j]
                })

        return objective_value, served_arcs, tour_arcs
    else:
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Solve Profitable Tour Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    objective_value, served_arcs, tour_arcs = solve(instance, args.time_limit, logger)

    if objective_value is not None:
        solution = {
            'objective_value': objective_value,
            'served_arcs': served_arcs,
            'tour_arcs': tour_arcs
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        with open(args.solution_path, 'w') as f:
            json.dump({'objective_value': float('-inf')}, f)

if __name__ == '__main__':
    main()