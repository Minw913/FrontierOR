import gurobipy as gp
from gurobipy import GRB
import json
import argparse
from solution_logger import SolutionLogger

def max_clique_size(vertices, edges_list):
    if not vertices:
        return 0
    adj = {v: set() for v in vertices}
    for e in edges_list:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    sorted_vertices = sorted(vertices, key=lambda v: len(adj[v]), reverse=True)
    n = len(sorted_vertices)
    v_to_index = {v: i for i, v in enumerate(sorted_vertices)}
    adj_index = [set() for _ in range(n)]
    for i, v in enumerate(sorted_vertices):
        for w in adj[v]:
            j = v_to_index[w]
            adj_index[i].add(j)
            adj_index[j].add(i)
    
    best = 0
    if n == 0:
        return 0

    def bron_kerbosch(R, P, X):
        nonlocal best
        if not P and not X:
            if len(R) > best:
                best = len(R)
            return
        if not P:
            return
        
        u = max(P | X, key=lambda u: len(P & adj_index[u])) if (P | X) else next(iter(P))
        P_minus = P - adj_index[u]
        for v in list(P_minus):
            bron_kerbosch(R | {v}, P & adj_index[v], X & adj_index[v])
            P.remove(v)
            X.add(v)
    
    R = set()
    P = set(range(n))
    X = set()
    bron_kerbosch(R, P, X)
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_vertices = instance['num_vertices']
    num_edges = instance['num_edges']
    k = instance['interdiction_budget_k']
    vertices = instance['vertices']
    edges = instance['edges']
    
    edge_tuples = []
    for e in edges:
        u, v = e
        if u > v:
            u, v = v, u
        edge_tuples.append((u, v))
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    x = model.addVars(edge_tuples, vtype=GRB.BINARY, name="x")
    orig_z = max_clique_size(vertices, edge_tuples)
    t = model.addVar(lb=1, ub=orig_z, vtype=GRB.INTEGER, name="t")
    
    model.addConstr(gp.quicksum(x[e] for e in edge_tuples) <= k, "budget")
    model.setObjective(t, GRB.MINIMIZE)
    
    model._vertices = vertices
    model._edge_tuples = edge_tuples
    model._x = x
    model._t = t
    model._best_z = float('inf')
    model._best_x = None
    model._logger = logger
    
    def callback(model_param, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = {}
            for e in model_param._edge_tuples:
                x_vals[e] = model_param.cbGetSolution(model_param._x[e])
            kept_edges = [e for e in model_param._edge_tuples if x_vals[e] < 0.5]
            z = max_clique_size(model_param._vertices, kept_edges)
            model_param.cbLazy(model_param._t >= z)
            if z < model_param._best_z:
                model_param._best_z = z
                model_param._best_x = x_vals
                if model_param._logger:
                    model_param._logger.log(int(z))
    
    model.setParam('LazyConstraints', 1)
    model.optimize(callback)
    
    if model.SolCount > 0:
        x_sol = model.getAttr('X', x)
        interdicted_edges = [list(e) for e in edge_tuples if x_sol[e] > 0.5]
        kept_edges = [e for e in edge_tuples if x_sol[e] <= 0.5]
        true_z = max_clique_size(vertices, kept_edges)
        objective_value = true_z
    else:
        if model._best_x is not None:
            interdicted_edges = [list(e) for e in edge_tuples if model._best_x[e] > 0.5]
            objective_value = model._best_z
        else:
            interdicted_edges = []
            objective_value = orig_z
    
    solution = {
        "objective_value": int(objective_value),
        "interdicted_edges": interdicted_edges
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()