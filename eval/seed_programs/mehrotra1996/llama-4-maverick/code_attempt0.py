import argparse
import json
from gurobipy import Model, GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_graph_coloring(instance_data, time_limit, logger):
    graph_data = instance_data['graph']
    num_nodes = graph_data['num_nodes']
    num_edges = graph_data['num_edges']
    nodes = graph_data['nodes']
    edges = graph_data['edges']

    model = Model('graph_coloring')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Create variables
    node_colors = model.addVars(nodes, range(num_nodes), vtype=GRB.BINARY, name='node_colors')
    color_used = model.addVars(range(num_nodes), vtype=GRB.BINARY, name='color_used')

    # Objective: Minimize the number of colors used
    model.setObjective(sum(color_used[i] for i in range(num_nodes)), GRB.MINIMIZE)

    # Constraints:
    # 1. Each node must be assigned exactly one color
    model.addConstrs((sum(node_colors[node, i] for i in range(num_nodes)) == 1 for node in nodes), name='assign_one_color')

    # 2. Two nodes connected by an edge must not have the same color
    model.addConstrs((node_colors[node1, i] + node_colors[node2, i] <= color_used[i] for i in range(num_nodes) for node1, node2 in edges), name='different_colors')

    # 3. Link node_colors to color_used (if any node is assigned color i, then color_used[i] must be 1)
    model.addConstrs((node_colors[node, i] <= color_used[i] for node in nodes for i in range(num_nodes)), name='link_node_color_to_color_used')

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        vertex_colors = {node: next(i for i in range(num_nodes) if node_colors[node, i].X > 0.5) for node in nodes}
        color_classes = [[] for _ in range(num_nodes)]
        for node in nodes:
            color = vertex_colors[node]
            color_classes[color].append(node)
        color_classes = [group for group in color_classes if group]

        solution = {
            'objective_value': len(color_classes),
            'vertex_colors': vertex_colors,
            'color_classes': color_classes
        }

        if logger and model.status == GRB.TIME_LIMIT:
            logger.log(len(color_classes))
        elif logger and model.status == GRB.OPTIMAL:
            logger.log(len(color_classes))

        return solution
    else:
        raise Exception(f"Gurobi model failed to find a solution. Status code: {model.status}")

def main():
    args = parse_args()
    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solution = solve_graph_coloring(instance_data, args.time_limit, logger)
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == '__main__':
    main()