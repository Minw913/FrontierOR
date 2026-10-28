import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
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
    vertices = graph['vertices']
    edges = graph['edges']  # list of [u, v, weight]

    n = len(vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    # Build adjacency list and edge dict
    adj = {v: set() for v in vertices}
    edge_weight = {}
    edge_list = []
    for u, v, w in edges:
        if u > v:
            u, v = v, u
        adj[u].add(v)
        adj[v].add(u)
        edge_weight[(u, v)] = w
        edge_list.append((u, v, w))

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model = gp.Model("MaxWeightClique")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.MIPGap, 0.0)
    model.setParam(GRB.Param.LogToConsole, 0)

    x = model.addVars(vertices, vtype=GRB.BINARY, name="x")

    # Objective: sum w_e * y_e  (will link y_e <= x_u and y_e <= x_v)
    obj_expr = gp.LinExpr()
    y_vars = {}
    for u, v, w in edge_list:
        y = model.addVar(vtype=GRB.BINARY, name=f"y_{u}_{v}")
        y_vars[(u, v)] = y
        obj_expr.add(y, w)
        model.addConstr(y <= x[u])
        model.addConstr(y <= x[v])

    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Clique constraints: for every non-edge, at most one vertex selected
    non_edges_checked = set()
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            if v not in adj[u]:
                key = (min(u, v), max(u, v))
                if key not in non_edges_checked:
                    model.addConstr(x[u] + x[v] <= 1)
                    non_edges_checked.add(key)

    # Callback to log improving solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        selected_vertices = [v for v in vertices if x[v].X > 0.5]
        selected_vertices.sort()

        selected_edges = []
        total_weight = 0.0
        for u, v in y_vars:
            if y_vars[(u, v)].X > 0.5:
                selected_edges.append([u, v])
                total_weight += edge_weight[(u, v)]
        selected_edges.sort()

        objective_value = float(total_weight)
    else:
        # No solution found
        selected_vertices = []
        selected_edges = []
        objective_value = 0.0

    solution = {
        "objective_value": objective_value,
        "clique_vertices": selected_vertices,
        "clique_edges": selected_edges
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Final log of best solution
    if logger and model.SolCount > 0:
        logger.log(objective_value)

if __name__ == "__main__":
    main()