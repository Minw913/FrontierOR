import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path):
    graph = instance['graph']
    num_vertices = graph['num_vertices']
    edges = graph['edges']
    partition = instance['partition']
    num_clusters = partition['num_clusters']
    clusters = partition['clusters']

    # Create a Gurobi model
    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable console logging

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create variables
    x = model.addVars(num_clusters, len(clusters[0]), vtype=GRB.BINARY, name='x')
    y = model.addVars(num_vertices, vtype=GRB.BINARY, name='y')
    c = model.addVar(vtype=GRB.INTEGER, name='c')

    # Objective: minimize the number of colors used
    model.setObjective(c, GRB.MINIMIZE)

    # Constraints:
    # 1. Select exactly one vertex from each cluster
    for i in range(num_clusters):
        model.addConstr(gp.quicksum(x[i, j] for j in range(len(clusters[i]))) == 1, name=f'cluster_{i}')

    # 2. Link x and y variables: if a vertex is selected, its y variable should be 1
    for i in range(num_clusters):
        for j in range(len(clusters[i])):
            vertex = clusters[i][j]
            model.addConstr(y[vertex] <= x[i, j], name=f'link_xy_{vertex}')

    # 3. Color assignment and conflict constraints
    for i in range(num_clusters):
        for j in range(len(clusters[i])):
            vertex = clusters[i][j]
            model.addConstr(c >= y[vertex], name=f'color_bound_{vertex}')

    for edge in edges:
        u, v = edge
        model.addConstr(y[u] + y[v] <= 1 + c, name=f'color_conflict_{u}_{v}')

    # Optimize the model
    model.optimize()

    # Process the solution
    solution = {}
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected_vertices = {}
        coloring = {}
        num_colors = 0
        for i in range(num_clusters):
            for j in range(len(clusters[i])):
                if x[i, j].x > 0.5:
                    vertex = clusters[i][j]
                    selected_vertices[str(i)] = vertex
                    coloring[str(vertex)] = int(y[vertex].x)
                    num_colors = max(num_colors, int(y[vertex].x))
        solution['objective_value'] = num_colors
        solution['selected_vertices'] = selected_vertices
        solution['coloring'] = coloring

        # Log intermediate solutions
        if logger and model.status == GRB.TIME_LIMIT:
            best_obj = model.getObjective().getValue()
            logger.log(best_obj)
    else:
        raise Exception(f"Model status: {model.status}")

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()