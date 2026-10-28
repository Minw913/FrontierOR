import json
import argparse
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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    edges = instance['edges']
    
    # Build adjacency list and non-edges for faster lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find a good initial clique using greedy heuristic ( degeneracy ordering like)
    # This gives a warm start and early logging
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    def greedy_clique():
        best_clique = []
        for start in range(n):
            clique = [start]
            candidates = set(range(n)) - {start}
            for v in order:
                if v == start:
                    continue
                if all(v in adj[u] for u in clique):
                    clique.append(v)
                    candidates.discard(v)
            if len(clique) > len(best_clique):
                best_clique = clique[:]
        return best_clique
    
    init_clique = greedy_clique()
    best_size = len(init_clique)
    best_vertices = sorted(init_clique)
    
    if logger:
        logger.log(best_size)
    
    # Build model for exact solving with Gurobi
    model = gp.Model("MaxClique")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPFocus = 1  # focus on finding good feasible solutions
    model.Params.Heuristics = 0.8
    model.Params.Cuts = 2
    
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]
    
    # Objective
    model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)
    
    # Edge constraints: for every non-edge, at most one can be selected
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
    
    for i, j in non_edges:
        model.addConstr(x[i] + x[j] <= 1, f"nonedge_{i}_{j}")
    
    # Warm start
    for i in range(n):
        x[i].Start = 1.0 if i in init_clique else 0.0
    
    # Callback to log improving solutions
    def callback(model, where):
        nonlocal best_size, best_vertices
        if where == GRB.Callback.MIPSOL:
            obj = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
            if obj > best_size:
                best_size = obj
                xsol = model.cbGetSolution(x)
                best_vertices = [i for i in range(n) if xsol[i] > 0.5]
                if logger:
                    logger.log(best_size)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount > 0:
        objective_value = int(round(model.ObjVal))
        clique_vertices = [i for i in range(n) if x[i].X > 0.5]
    else:
        # Fallback to heuristic solution
        objective_value = best_size
        clique_vertices = best_vertices
    
    # Write output
    solution = {
        "objective_value": objective_value,
        "clique_vertices": sorted(clique_vertices)
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()