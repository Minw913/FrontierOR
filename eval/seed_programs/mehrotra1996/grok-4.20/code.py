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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    num_nodes = graph['num_nodes']
    nodes = graph['nodes']
    edges = graph['edges']
    
    # Build adjacency list
    adj = {u: set() for u in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedy coloring for upper bound and initial solution
    def greedy_coloring():
        color = {}
        for node in nodes:
            used = set()
            for nei in adj[node]:
                if nei in color:
                    used.add(color[nei])
            c = 0
            while c in used:
                c += 1
            color[node] = c
        return color
    
    init_color = greedy_coloring()
    best_k = max(init_color.values()) + 1
    best_coloring = init_color
    best_classes = None
    
    if logger:
        logger.log(best_k)
    
    # Try to improve with MIP using Gurobi with time limit
    start_time = time.time()
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time < 1:
        remaining_time = 1
    
    model = gp.Model("GraphColoring")
    model.setParam('TimeLimit', remaining_time)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)
    model.setParam('Heuristics', 0.8)
    
    # Upper bound on colors
    K = best_k
    x = {}
    for v in nodes:
        for c in range(K):
            x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")
    
    # Each vertex gets exactly one color
    for v in nodes:
        model.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1)
    
    # Adjacent vertices different colors
    for u, v in edges:
        for c in range(K):
            model.addConstr(x[u, c] + x[v, c] <= 1)
    
    # Objective: minimize number of used colors
    y = [model.addVar(vtype=GRB.BINARY, name=f"y_{c}") for c in range(K)]
    for c in range(K):
        for v in nodes:
            model.addConstr(x[v, c] <= y[c])
    model.setObjective(gp.quicksum(y), GRB.MINIMIZE)
    
    # Provide initial solution
    for v, col in init_color.items():
        if col < K:
            x[v, col].Start = 1.0
            for c2 in range(K):
                if c2 != col:
                    x[v, c2].Start = 0.0
    for c in range(K):
        y[c].Start = 1.0 if c < best_k else 0.0
    
    model.update()
    
    # Callback to log improved solutions
    def callback(model, where):
        nonlocal best_k, best_coloring, best_classes
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_k - 0.5:  # improved
                best_k = int(round(obj))
                # Extract solution
                sol = model.cbGetSolution(x)
                coloring = {}
                for v in nodes:
                    for c in range(K):
                        if sol[v, c] > 0.5:
                            coloring[v] = c
                            break
                best_coloring = coloring
                if logger:
                    logger.log(best_k)
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        obj = int(round(model.ObjVal))
        if obj < best_k:
            best_k = obj
            sol = model.getAttr('X', x)
            coloring = {}
            for v in nodes:
                for c_idx in range(K):
                    if sol[v, c_idx] > 0.5:
                        coloring[v] = c_idx
                        break
            best_coloring = coloring
            if logger:
                logger.log(best_k)
    
    # Build color classes
    color_to_vertices = {}
    for v, c in best_coloring.items():
        if c not in color_to_vertices:
            color_to_vertices[c] = []
        color_to_vertices[c].append(v)
    
    # Renumber colors from 0 to best_k-1 consecutively
    sorted_colors = sorted(color_to_vertices.keys())
    color_map = {old: new for new, old in enumerate(sorted_colors)}
    final_coloring = {v: color_map[c] for v, c in best_coloring.items()}
    color_classes = [color_to_vertices[old] for old in sorted_colors]
    
    # Prepare output
    solution = {
        "objective_value": best_k,
        "vertex_colors": {str(node): final_coloring[node] for node in nodes},
        "color_classes": color_classes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()