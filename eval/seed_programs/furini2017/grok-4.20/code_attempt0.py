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

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    n = graph['n']
    edges = graph['edges']
    m = len(edges)
    
    # Build adjacency for faster checks
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    best_k = n
    best_coloring = None
    best_color_classes = None
    
    # Greedy coloring for upper bound and initial solution
    def greedy_coloring(order):
        color = [-1] * n
        for v in order:
            used = set()
            for nei in adj[v]:
                if color[nei] != -1:
                    used.add(color[nei])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        return color
    
    # Basic greedy in natural order
    color_greedy = greedy_coloring(range(n))
    k_greedy = max(color_greedy) + 1
    if k_greedy < best_k:
        best_k = k_greedy
        best_coloring = {str(i): int(color_greedy[i]) for i in range(n)}
        color_classes_dict = {}
        for c in range(best_k):
            color_classes_dict[str(c)] = [i for i in range(n) if color_greedy[i] == c]
        best_color_classes = color_classes_dict
        if logger:
            logger.log(int(best_k))
    
    # Try DSATUR-like ordering for potentially better coloring
    def dsatur_ordering():
        degree = [len(adj[i]) for i in range(n)]
        colored = [-1] * n
        saturation = [0] * n
        order = []
        remaining = set(range(n))
        
        # Start with highest degree
        v = max(remaining, key=lambda x: degree[x])
        remaining.remove(v)
        order.append(v)
        colored[v] = 0
        
        while remaining:
            best_v = None
            best_sat = -1
            best_deg = -1
            for u in remaining:
                sat = len({colored[w] for w in adj[u] if colored[w] != -1})
                if sat > best_sat or (sat == best_sat and degree[u] > best_deg):
                    best_sat = sat
                    best_deg = degree[u]
                    best_v = u
            remaining.remove(best_v)
            order.append(best_v)
            # We don't need to assign color here, just ordering
        return order
    
    color_dsatur_order = greedy_coloring(dsatur_ordering())
    k_dsatur = max(color_dsatur_order) + 1
    if k_dsatur < best_k:
        best_k = k_dsatur
        best_coloring = {str(i): int(color_dsatur_order[i]) for i in range(n)}
        color_classes_dict = {}
        for c_idx in range(best_k):
            color_classes_dict[str(c_idx)] = [i for i in range(n) if color_dsatur_order[i] == c_idx]
        best_color_classes = color_classes_dict
        if logger:
            logger.log(int(best_k))
    
    # Try to improve with simple local search / recoloring
    def improve_coloring(coloring, k):
        improved = True
        iterations = 0
        max_iter = 1000
        current_color = coloring[:]
        current_k = max(current_color) + 1
        
        while improved and iterations < max_iter and (time.time() - start_time) < args.time_limit * 0.7:
            improved = False
            iterations += 1
            for v in range(n):
                used = set()
                for nei in adj[v]:
                    if current_color[nei] != -1:
                        used.add(current_color[nei])
                if current_color[v] in used or current_color[v] >= k:
                    for c in range(k):
                        if c not in used:
                            current_color[v] = c
                            improved = True
                            break
            if improved:
                new_k = max(current_color) + 1
                if new_k < current_k:
                    current_k = new_k
                    if current_k < best_k:
                        best_k = current_k
                        nonlocal best_coloring, best_color_classes
                        best_coloring = {str(i): int(current_color[i]) for i in range(n)}
                        color_classes_dict = {}
                        for c_idx in range(current_k):
                            color_classes_dict[str(c_idx)] = [i for i in range(n) if current_color[i] == c_idx]
                        best_color_classes = color_classes_dict
                        if logger:
                            logger.log(int(best_k))
        return current_color
    
    if best_k > 1:
        color_improved = improve_coloring(color_dsatur_order, best_k)
    
    # If we have small n or good upper bound, try exact MIP with Gurobi
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed - 2.0  # reserve some margin
    if n <= 80 and best_k <= 15 and remaining_time > 5.0:
        try:
            # Model for graph coloring
            model = gp.Model("GraphColoring")
            model.setParam(GRB.Param.TimeLimit, remaining_time)
            model.setParam(GRB.Param.Threads, 1)
            model.setParam(GRB.Param.LogToConsole, 0)
            model.setParam(GRB.Param.MIPGap, 0.0)
            
            # Upper bound on colors
            K = best_k
            x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
            y = model.addVars(K, vtype=GRB.BINARY, name="y")
            
            # Each vertex gets exactly one color
            for v in range(n):
                model.addConstr(gp.quicksum(x[v, c] for c in range(K)) == 1)
            
            # Adjacent vertices cannot have same color
            for u, v_edge in edges:
                for c in range(K):
                    model.addConstr(x[u, c] + x[v_edge, c] <= 1)
            
            # Link y variables
            for c in range(K):
                for v in range(n):
                    model.addConstr(x[v, c] <= y[c])
            
            # Objective: minimize number of colors used
            model.setObjective(gp.quicksum(y[c] for c in range(K)), GRB.MINIMIZE)
            
            # Callback to log new incumbents
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < best_k - 0.5:  # new integer solution better
                        nonlocal best_k, best_coloring, best_color_classes
                        best_k = int(round(obj))
                        sol_x = model.cbGetSolution(x)
                        color_assign = [-1] * n
                        for v in range(n):
                            for c in range(K):
                                if sol_x[v, c] > 0.5:
                                    color_assign[v] = c
                                    break
                        best_coloring = {str(i): int(color_assign[i]) for i in range(n)}
                        color_classes_dict = {}
                        for c_idx in range(best_k):
                            color_classes_dict[str(c_idx)] = [i for i in range(n) if color_assign[i] == c_idx]
                        best_color_classes = color_classes_dict
                        if logger:
                            logger.log(int(best_k))
            
            model.optimize(callback)
            
            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
                obj_val = int(round(model.ObjVal))
                if obj_val < best_k:
                    best_k = obj_val
                    sol_x = model.getAttr('X', x)
                    color_assign = [-1] * n
                    for v_idx in range(n):
                        for c_idx in range(K):
                            if sol_x[v_idx, c_idx] > 0.5:
                                color_assign[v_idx] = c_idx
                                break
                    best_coloring = {str(i): int(color_assign[i]) for i in range(n)}
                    color_classes_dict = {}
                    for c_idx in range(best_k):
                        color_classes_dict[str(c_idx)] = [i for i in range(n) if color_assign[i] == c_idx]
                    best_color_classes = color_classes_dict
                    if logger:
                        logger.log(int(best_k))
        except Exception:
            # If MIP fails for any reason, fall back to heuristic solution
            pass
    
    # Final output
    solution = {
        "objective_value": int(best_k),
        "coloring": best_coloring,
        "color_classes": best_color_classes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()