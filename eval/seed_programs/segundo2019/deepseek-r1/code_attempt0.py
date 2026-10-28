import json
import argparse
import gurobipy as grb
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    graph = data['graph']
    vertices = graph['vertices']
    edges = graph['edges']
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']

    adj = {}
    for v in vertices:
        adj[v] = set()
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adj[u].add(v)
        adj[v].add(u)

    model = grb.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)

    x = model.addVars(vertices, vtype=grb.GRB.BINARY, name='x')

    y_vars = {}
    for edge in edges:
        u, v, w = edge[0], edge[1], edge[2]
        key = (min(u, v), max(u, v))
        y_vars[key] = model.addVar(vtype=grb.GRB.BINARY, name=f'y_{key[0]}_{key[1]}')

    for edge in edges:
        u, v, w = edge
        key = (min(u, v), max(u, v))
        model.addConstr(y_vars[key] <= x[u])
        model.addConstr(y_vars[key] <= x[v])
        model.addConstr(y_vars[key] >= x[u] + x[v] - 1)

    obj = grb.quicksum(edge[2] * y_vars[(min(edge[0], edge[1]), max(edge[0], edge[1]))] for edge in edges)
    model.setObjective(obj, grb.GRB.MAXIMIZE)

    added_non_edges = set()

    def callback_function(model, where):
        nonlocal added_non_edges
        if where == grb.GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(model._x)
            S = [v for v in vertices if x_vals[v] > 0.5]
            if not S:
                return
            S_sorted = sorted(S)
            n = len(S_sorted)
            adj = model._adj
            non_edges_found = []
            is_clique = True

            for i in range(n):
                u = S_sorted[i]
                for j in range(i+1, n):
                    v = S_sorted[j]
                    if v not in adj[u]:
                        is_clique = False
                        key = (min(u, v), max(u, v))
                        if key not in added_non_edges:
                            non_edges_found.append(key)
            if is_clique:
                objval = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(objval)
            else:
                for key in non_edges_found:
                    u, v = key
                    model.cbLazy(model._x[u] + model._x[v] <= 1)
                    added_non_edges.add(key)

    model._x = x
    model._adj = adj
    model.optimize(callback_function)

    if model.SolCount > 0:
        objective_value = model.ObjVal
        clique_vertices = [v for v in vertices if x[v].X > 0.5]
        clique_vertices.sort()
        clique_edges = []
        for edge in edges:
            u, v, w = edge
            if x[u].X > 0.5 and x[v].X > 0.5:
                u1, v1 = min(u, v), max(u, v)
                clique_edges.append([u1, v1])
    else:
        objective_value = 0.0
        clique_vertices = []
        clique_edges = []

    solution = {
        "objective_value": float(objective_value),
        "clique_vertices": clique_vertices,
        "clique_edges": clique_edges
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()