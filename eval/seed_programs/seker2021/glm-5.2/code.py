import json
import argparse
import random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def compute_ub_incremental(current_colors, current_max, v, adj):
    used = set()
    for neighbor in adj[v]:
        if neighbor in current_colors:
            used.add(current_colors[neighbor])
    c = 1
    while c in used:
        c += 1
    return max(current_max, c), c

def greedy_color(nodes, adj):
    nodes_sorted = sorted(nodes, key=lambda x: -len(adj[x]))
    colors = {}
    max_c = 0
    for node in nodes_sorted:
        used = set()
        for neighbor in adj[node]:
            if neighbor in colors:
                used.add(colors[neighbor])
        c = 1
        while c in used:
            c += 1
        colors[node] = c
        if c > max_c:
            max_c = c
    return colors, max_c

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    random.seed(0)

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    num_vertices = instance["graph"]["num_vertices"]
    edges = instance["graph"]["edges"]
    num_clusters = instance["partition"]["num_clusters"]
    clusters = instance["partition"]["clusters"]

    adj = {i: set() for i in range(num_vertices)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    best_ub = num_clusters
    best_selected = []
    best_colors = {}

    # Fast incremental heuristic
    for _ in range(50):
        selected = [None] * num_clusters
        clusters_order = list(range(num_clusters))
        random.shuffle(clusters_order)
        
        current_colors = {}
        current_max = 0
        
        for c_idx in clusters_order:
            cluster = clusters[c_idx]
            best_v = -1
            best_ub_v = float('inf')
            best_c_v = 0
            
            for v in cluster:
                ub_v, c_v = compute_ub_incremental(current_colors, current_max, v, adj)
                if ub_v < best_ub_v:
                    best_ub_v = ub_v
                    best_v = v
                    best_c_v = c_v
                    
            selected[c_idx] = best_v
            current_colors[best_v] = best_c_v
            current_max = best_ub_v
            
        if current_max < best_ub:
            best_ub = current_max
            best_selected = list(selected)
            best_colors = current_colors
            if logger:
                sol = {
                    "objective_value": float(best_ub),
                    "selected_vertices": {str(i): best_selected[i] for i in range(len(best_selected))},
                    "coloring": {str(v): best_colors[v] for v in best_colors}
                }
                logger.log_solution(best_ub, sol)
        if best_ub == 1:
            break

    # Final greedy coloring improvement on the best selection found
    if best_selected:
        final_colors, final_ub = greedy_color(best_selected, adj)
        if final_ub < best_ub:
            best_ub = final_ub
            best_colors = final_colors
            if logger:
                sol = {
                    "objective_value": float(best_ub),
                    "selected_vertices": {str(i): best_selected[i] for i in range(len(best_selected))},
                    "coloring": {str(v): best_colors[v] for v in best_colors}
                }
                logger.log_solution(best_ub, sol)

    # ILP Formulation
    if best_ub > 1 and args.time_limit > 5 and num_clusters > 0:
        try:
            model = gp.Model("min_color")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1, args.time_limit - 5)
            
            x = model.addVars(num_vertices, vtype=GRB.BINARY, name="x")
            y = model.addVars(num_vertices, best_ub, vtype=GRB.BINARY, name="y")
            z = model.addVars(best_ub, vtype=GRB.BINARY, name="z")
            
            for i, cluster in enumerate(clusters):
                model.addConstr(gp.quicksum(x[v] for v in cluster) == 1)
                
            model.addConstrs((y[v, c] <= x[v] for v in range(num_vertices) for c in range(best_ub)))
            model.addConstrs((y[v, c] <= z[c] for v in range(num_vertices) for c in range(best_ub)))
            model.addConstrs((gp.quicksum(y[v, c] for c in range(best_ub)) == x[v] for v in range(num_vertices)))
            
            for u, v in edges:
                for c in range(best_ub):
                    model.addConstr(y[u, c] + y[v, c] <= 1)
                    
            for c in range(best_ub - 1):
                model.addConstr(z[c] >= z[c+1])
                
            model.setObjective(gp.quicksum(z[c] for c in range(best_ub)), GRB.MINIMIZE)
            
            # Warm start
            for i, v in enumerate(best_selected):
                x[v].Start = 1
            for v, c in best_colors.items():
                y[v, c-1].Start = 1
            for c in range(1, best_ub+1):
                z[c-1].Start = 1
                
            model.optimize()
            
            if model.SolCount > 0:
                obj_val = int(round(model.ObjVal))
                if obj_val < best_ub:
                    best_ub = obj_val
                    selected_v = []
                    for v in range(num_vertices):
                        if x[v].X > 0.5:
                            selected_v.append(v)
                    coloring = {}
                    for v in selected_v:
                        for c in range(best_ub):
                            if y[v, c].X > 0.5:
                                coloring[v] = c + 1
                                break
                    best_selected = []
                    for cluster in clusters:
                        for v in cluster:
                            if v in coloring:
                                best_selected.append(v)
                                break
                    best_colors = coloring
                    if logger:
                        sol = {
                            "objective_value": float(best_ub),
                            "selected_vertices": {str(i): best_selected[i] for i in range(len(best_selected))},
                            "coloring": {str(v): best_colors[v] for v in best_colors}
                        }
                        logger.log_solution(best_ub, sol)
        except Exception as e:
            pass

    solution = {
        "objective_value": float(best_ub),
        "selected_vertices": {str(i): best_selected[i] for i in range(len(best_selected))},
        "coloring": {str(v): best_colors[v] for v in best_colors}
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    solve()