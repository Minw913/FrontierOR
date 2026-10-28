import argparse
import json
import random
import time
import gurobipy as gp
from gurobipy import GRB

def find_large_clique_advanced(n, adj, iterations=50):
    if n == 0:
        return []
    if not any(adj):
        return [0]
        
    best_clique = []
    adj_sets = [set(neighbors) for neighbors in adj]
    
    for seed in range(iterations):
        random.seed(seed)
        u = random.randint(0, n-1)
        clique = [u]
        candidates = set(adj[u])
        
        while candidates:
            best_c = -1
            max_deg = -1
            
            for c in candidates:
                deg = len(adj_sets[c].intersection(candidates))
                if deg > max_deg:
                    max_deg = deg
                    best_c = c
                elif deg == max_deg:
                    if random.random() < 0.5:
                        best_c = c
            
            clique.append(best_c)
            candidates = candidates.intersection(adj_sets[best_c])
            
        if len(clique) > len(best_clique):
            best_clique = clique
            if len(best_clique) == n:
                break
                
    return best_clique

def dsatur_with_init(n, adj, initial_coloring, iterations=20):
    best_color_count = n + 1
    best_coloring = None
    
    for seed in range(iterations):
        random.seed(seed + 100)
        color = [-1] * n
        degree = [len(adj[i]) for i in range(n)]
        sat_degree = [0] * n
        adj_colors = [set() for _ in range(n)]
        
        sat_groups = {0: set()}
        max_sat = 0
        uncolored_count = n
        
        for v in range(n):
            if v not in initial_coloring:
                sat_groups[0].add(v)
                
        for u, c in initial_coloring.items():
            color[u] = c
            uncolored_count -= 1
            for v in adj[u]:
                if color[v] == -1: 
                    degree[v] -= 1
                    if c not in adj_colors[v]:
                        old_sat = sat_degree[v]
                        if v in sat_groups.get(old_sat, set()):
                            sat_groups[old_sat].remove(v)
                        sat_degree[v] += 1
                        new_sat = sat_degree[v]
                        if new_sat not in sat_groups:
                            sat_groups[new_sat] = set()
                        sat_groups[new_sat].add(v)
                        if new_sat > max_sat:
                            max_sat = new_sat
                            
        while uncolored_count > 0:
            candidates = sat_groups.get(max_sat, set())
            while not candidates and max_sat > 0:
                max_sat -= 1
                candidates = sat_groups.get(max_sat, set())
                
            best_u = -1
            max_deg = -1
            
            for u in candidates:
                d = degree[u]
                if d > max_deg:
                    max_deg = d
                    best_u = u
                elif d == max_deg:
                    if random.random() < 0.5:
                        best_u = u
                    
            u = best_u
            
            c = 0
            while c in adj_colors[u]:
                c += 1
                
            color[u] = c
            sat_groups[max_sat].remove(u)
            uncolored_count -= 1
            
            for v in adj[u]:
                if color[v] == -1:
                    degree[v] -= 1
                    if c not in adj_colors[v]:
                        old_sat = sat_degree[v]
                        sat_groups[old_sat].remove(v)
                        sat_degree[v] += 1
                        new_sat = sat_degree[v]
                        if new_sat not in sat_groups:
                            sat_groups[new_sat] = set()
                        sat_groups[new_sat].add(v)
                        if new_sat > max_sat:
                            max_sat = new_sat
                            
        max_c = max(color) + 1
        if max_c < best_color_count:
            best_color_count = max_c
            best_coloring = list(color)
            if best_color_count == len(initial_coloring):
                break
            
    return best_coloring, best_color_count

def extract_dsatur_solution(coloring, n):
    actual_colors = set(coloring)
    sorted_colors = sorted(list(actual_colors))
    color_map = {old: i for i, old in enumerate(sorted_colors)}
    
    obj_val = len(actual_colors)
    selected_sets = [[] for _ in range(obj_val)]
    coloring_dict = {}
    
    for v in range(n):
        new_c = color_map[coloring[v]]
        coloring_dict[str(v)] = new_c
        selected_sets[new_c].append(v)
        
    for s in selected_sets:
        s.sort()
        
    return obj_val, selected_sets, coloring_dict

def save_solution(obj_val, selected_sets, coloring_dict, path):
    out = {
        "objective_value": obj_val,
        "selected_sets": selected_sets,
        "coloring": coloring_dict
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    n = instance_data["graph"]["num_vertices"]
    edges_input = instance_data["graph"]["edges"]
    
    if n == 0:
        save_solution(0, [], {}, args.solution_path)
        return

    adj = [[] for _ in range(n)]
    edges = []
    for u, v in edges_input:
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    degrees = [len(adj[i]) for i in range(n)]
    
    best_clique = find_large_clique_advanced(n, adj, iterations=50)
    k = len(best_clique)

    initial_coloring = {v: i for i, v in enumerate(best_clique)}
    best_dsatur_coloring, U = dsatur_with_init(n, adj, initial_coloring, iterations=20)
    
    dsatur_obj, dsatur_selected, dsatur_color_dict = extract_dsatur_solution(best_dsatur_coloring, n)

    if logger:
        logger.log(dsatur_obj)

    bounds = instance_data.get("bounds", {})
    clique_lb = bounds.get("clique_lower_bound", 0)
    L = max(k, clique_lb)

    if U <= L or args.time_limit - (time.time() - start_time) < 5:
        save_solution(dsatur_obj, dsatur_selected, dsatur_color_dict, args.solution_path)
        return

    model = gp.Model("GraphColoring")
    model.Params.LogToConsole = 0
    model.Params.Threads = 1

    remaining_time = args.time_limit - (time.time() - start_time) - 2
    model.Params.TimeLimit = max(0.1, remaining_time)

    x = {}
    for v in range(n):
        for c in range(min(U, degrees[v] + 1)):
            x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

    y = model.addVars(U, vtype=GRB.BINARY, name="y")

    model.setObjective(gp.quicksum(y[c] for c in range(U)), GRB.MINIMIZE)

    for c in range(min(L, U)):
        y[c].LB = 1

    for c in range(U - 1):
        model.addConstr(y[c] >= y[c + 1])

    for v in range(n):
        model.addConstr(gp.quicksum(x[v, c] for c in range(min(U, degrees[v] + 1))) == 1)

    for v in range(n):
        for c in range(min(U, degrees[v] + 1)):
            model.addConstr(x[v, c] <= y[c])

    for u, v in edges:
        for c in range(U):
            var_u = x.get((u, c))
            var_v = x.get((v, c))
            if var_u is not None and var_v is not None:
                model.addConstr(var_u + var_v <= y[c])

    for i, v in enumerate(best_clique):
        if i < U:
            var = x.get((v, i))
            if var is not None:
                var.LB = 1

    for v in range(n):
        color = best_dsatur_coloring[v]
        for c in range(min(U, degrees[v] + 1)):
            if c == color:
                x[v, c].Start = 1
            else:
                x[v, c].Start = 0

    for c in range(U):
        y[c].Start = 1

    best_logged_obj = dsatur_obj

    def callback(model, where):
        nonlocal best_logged_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_int = int(obj_val + 0.5)
            if obj_int < best_logged_obj:
                best_logged_obj = obj_int
                if logger:
                    logger.log(obj_int)

    try:
        model.optimize(callback)
    except Exception:
        pass

    if model.SolCount > 0 and int(model.ObjVal + 0.5) < U:
        actual_colors = set()
        v_colors = {}
        for v in range(n):
            c_assigned = 0
            for c in range(min(U, degrees[v] + 1)):
                if (v, c) in x and x[v, c].X > 0.5:
                    c_assigned = c
                    break
            actual_colors.add(c_assigned)
            v_colors[v] = c_assigned

        sorted_colors = sorted(list(actual_colors))
        color_map = {old: i for i, old in enumerate(sorted_colors)}

        gurobi_obj = len(actual_colors)
        gurobi_selected = [[] for _ in range(gurobi_obj)]
        gurobi_dicts = {}

        for v in range(n):
            new_c = color_map[v_colors[v]]
            gurobi_dicts[str(v)] = new_c
            gurobi_selected[new_c].append(v)

        for s in gurobi_selected:
            s.sort()

        save_solution(gurobi_obj, gurobi_selected, gurobi_dicts, args.solution_path)
    else:
        save_solution(dsatur_obj, dsatur_selected, dsatur_color_dict, args.solution_path)

if __name__ == "__main__":
    main()