import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from solution_logger import SolutionLogger

def dsatur(n, adj):
    if n == 0:
        return [], 0
    
    colors = [-1] * n
    saturation = [0] * n
    neighbor_colors = [set() for _ in range(n)]
    degrees = [len(adj[i]) for i in range(n)]
    uncolored = set(range(n))
    
    while uncolored:
        u = max(uncolored, key=lambda u: (saturation[u], degrees[u]))
        c = 0
        while c in neighbor_colors[u]:
            c += 1
        colors[u] = c
        uncolored.remove(u)
        
        for v in adj[u]:
            if colors[v] == -1:
                if c not in neighbor_colors[v]:
                    neighbor_colors[v].add(c)
                    saturation[v] += 1
    
    k_upper = max(colors) + 1 if n > 0 else 0
    return colors, k_upper

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
    
    graph = instance['graph']
    n = graph['num_vertices']
    m = graph['num_edges']
    edges = graph['edges']
    
    start_time = time.time()
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = e
        adj[u].append(v)
        adj[v].append(u)
    
    # Run DSatur to get initial solution
    colors, k_upper = dsatur(n, adj)
    
    if logger and n > 0:
        logger.log(k_upper)
    
    # If no vertices, output empty solution
    if n == 0:
        sol = {
            "objective_value": 0,
            "coloring": {},
            "color_classes": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return
    
    time_remaining = args.time_limit - (time.time() - start_time)
    if time_remaining <= 0:
        # Time's up, return greedy solution
        coloring_dict = {str(i): int(colors[i]) for i in range(n)}
        color_classes_dict = defaultdict(list)
        for v, c in enumerate(colors):
            color_classes_dict[str(c)].append(v)
        sol = {
            "objective_value": k_upper,
            "coloring": coloring_dict,
            "color_classes": dict(color_classes_dict)
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return
    
    try:
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', time_remaining)
        
        # Variables
        x = model.addVars(n, k_upper, vtype=gp.GRB.BINARY, name="x")
        y = model.addVars(k_upper, vtype=gp.GRB.BINARY, name="y")
        
        # Constraints
        for v in range(n):
            model.addConstr(gp.quicksum(x[v, c] for c in range(k_upper)) == 1, f"assign_{v}")
        
        for u, v in edges:
            for c in range(k_upper):
                model.addConstr(x[u, c] + x[v, c] <= 1, f"edge_{u}_{v}_{c}")
        
        for v in range(n):
            for c in range(k_upper):
                model.addConstr(x[v, c] <= y[c], f"color_used_{v}_{c}")
        
        model.setObjective(gp.quicksum(y[c] for c in range(k_upper)), gp.GRB.MINIMIZE)
        
        # Set initial solution
        for v in range(n):
            c_val = colors[v]
            x[v, c_val].Start = 1
            for c in range(k_upper):
                if c != c_val:
                    x[v, c].Start = 0
        
        colors_set = set(colors)
        for c in range(k_upper):
            y[c].Start = 1 if c in colors_set else 0
        
        # Callback for logging
        if logger:
            class CallbackWrapper:
                def __init__(self, logger):
                    self.logger = logger
                def __call__(self, model, where):
                    if where == gp.GRB.Callback.MIPSOL:
                        obj = int(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ))
                        self.logger.log(obj)
            model.optimize(CallbackWrapper(logger))
        else:
            model.optimize()
        
        # Extract solution
        if model.SolCount > 0:
            obj_val = int(model.objVal)
            coloring_sol = [-1] * n
            for v in range(n):
                for c in range(k_upper):
                    if x[v, c].X > 0.5:
                        coloring_sol[v] = c
        else:
            coloring_sol = colors
            obj_val = k_upper
        
        # Build output dictionaries
        coloring_dict = {str(i): int(coloring_sol[i]) for i in range(n)}
        color_classes_dict = defaultdict(list)
        for v, c in enumerate(coloring_sol):
            color_classes_dict[str(c)].append(v)
        
        sol = {
            "objective_value": obj_val,
            "coloring": coloring_dict,
            "color_classes": dict(color_classes_dict)
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
    
    except (gp.GurobiError, Exception) as e:
        # Fallback to greedy solution on error
        coloring_dict = {str(i): int(colors[i]) for i in range(n)}
        color_classes_dict = defaultdict(list)
        for v, c in enumerate(colors):
            color_classes_dict[str(c)].append(v)
        sol = {
            "objective_value": k_upper,
            "coloring": coloring_dict,
            "color_classes": dict(color_classes_dict)
        }
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)

if __name__ == "__main__":
    main()