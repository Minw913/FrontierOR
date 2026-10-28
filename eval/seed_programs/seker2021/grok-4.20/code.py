import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    partition = instance['partition']
    num_vertices = graph['num_vertices']
    edges = graph['edges']
    num_clusters = partition['num_clusters']
    clusters = partition['clusters']

    # Build adjacency list (undirected)
    adj = [set() for _ in range(num_vertices)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute possible colors: since it's perfect graph, chromatic number = clique number
    # But computing omega exactly is hard, so we use a reasonable upper bound
    # For MIP we will let solver decide the max color used

    model = gp.Model("PerfectGraphColoringSelection")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)

    # Decision variables
    # x[c][v] = 1 if vertex v from cluster c is selected
    x = []
    for c in range(num_clusters):
        x_c = {}
        for v in clusters[c]:
            x_c[v] = model.addVar(vtype=GRB.BINARY, name=f"x_c{c}_v{v}")
        x.append(x_c)

    # y_v = 1 if vertex v is selected (for any cluster)
    y = [model.addVar(vtype=GRB.BINARY, name=f"y_{v}") for v in range(num_vertices)]

    # Color assignment: we limit max colors to num_clusters (worst case)
    # But to tighten, we can use min(num_vertices, max clique size upper bound), but for simplicity use num_clusters
    K = num_clusters
    color = [[model.addVar(vtype=GRB.BINARY, name=f"color_v{v}_k{k}") for k in range(K)] for v in range(num_vertices)]

    # z_k = 1 if color k is used
    z = [model.addVar(vtype=GRB.BINARY, name=f"z_{k}") for k in range(K)]

    # Objective: minimize number of colors used
    model.setObjective(gp.quicksum(z), GRB.MINIMIZE)

    # Constraints

    # Exactly one vertex per cluster
    for c in range(num_clusters):
        model.addConstr(gp.quicksum(x[c][v] for v in clusters[c]) == 1)

    # Link y_v to selection
    for c in range(num_clusters):
        for v in clusters[c]:
            model.addConstr(y[v] == x[c][v])

    # Each selected vertex gets exactly one color
    for v in range(num_vertices):
        model.addConstr(gp.quicksum(color[v][k] for k in range(K)) == y[v])

    # If color k is used, z_k = 1
    for k in range(K):
        for v in range(num_vertices):
            model.addConstr(color[v][k] <= z[k])

    # Conflicting vertices cannot have same color
    for u, v in edges:
        for k in range(K):
            model.addConstr(color[u][k] + color[v][k] <= 1)

    # Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Callback to log improving solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        selected = {}
        coloring = {}
        used_colors = 0
        color_map = {}  # to remap colors to 0,1,2,... if needed but actually we can keep as is

        for c in range(num_clusters):
            for v in clusters[c]:
                if abs(x[c][v].X - 1.0) < 1e-6:
                    selected[str(c)] = v
                    # Find color
                    for k in range(K):
                        if abs(color[v][k].X - 1.0) < 1e-6:
                            coloring[str(v)] = k
                            used_colors = max(used_colors, k + 1)
                            break
                    break

        objective = float(model.objVal)  # This should be equal to used_colors but solver might have gaps

        # Final consistency check
        objective = float(used_colors)

        solution = {
            "objective_value": objective,
            "selected_vertices": selected,
            "coloring": coloring
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible? Should not happen for this problem
        solution = {
            "objective_value": 0.0,
            "selected_vertices": {},
            "coloring": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()