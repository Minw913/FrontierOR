import argparse
import json
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

def find_large_clique_advanced(n, adj, iterations=50, max_time=5.0):
    start_c_time = time.time()
    if n == 0:
        return []
    if not any(adj):
        return [0]
        
    best_clique = []
    adj_sets = [set(neighbors) for neighbors in adj]
    
    for seed in range(iterations):
        if time.time() - start_c_time > max_time:
            break
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

def dsatur_with_init(n, adj, initial_coloring, iterations, time_limit, start_time):
    best_color_count = n + 1
    best_coloring = None
    
    for seed in range(iterations):
        if time.time() - start_time > min(time_limit * 0.3, 10):
            break
            
        random.seed(seed ** 2 + 100)
        color = [-1] * n
        degree = [len(adj[i]) for i in range(n)]
        sat_degree = [0] * n
        adj_colors = [set() for _ in range(n)]
        
        sat_groups = {0: set()}
        max_sat = 0
        
        for u, c in initial_coloring.items():
            color[u] = c
            
        uncolored_count = n - len(initial_coloring)
        
        for v in range(n):
            if v not in initial_coloring:
                sat_groups[0].add(v)
                
        for u, c in initial_coloring.items():
            for v in adj[u]:
                if color[v] == -1: 
                    degree[v] -= 1
                    if c not in adj_colors[v]:
                        adj_colors[v].add(c)
                        old_sat = sat_degree[v]
                        if old_sat in sat_groups and v in sat_groups[old_sat]:
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
            if u == -1:
                break
                
            c = 0
            while c in adj_colors[u]:
                c += 1
                
            color[u] = c
            if max_sat in sat_groups and u in sat_groups[max_sat]:
                sat_groups[max_sat].discard(u)
            uncolored_count -= 1
            
            for v in adj[u]:
                if color[v] == -1:
                    degree[v] -= 1
                    if c not in adj_colors[v]:
                        adj_colors[v].add(c)
                        old_sat = sat_degree[v]
                        if old_sat in sat_groups and v in sat_groups[old_sat]:
                            sat_groups[old_sat].discard(v)
                        sat_degree[v] += 1
                        new_sat = sat_degree[v]
                        if new_sat not in sat_groups:
                            sat_groups[new_sat] = set()
                        sat_groups[new_sat].add(v)
                        if new_sat > max_sat:
                            max_sat = new_sat
                            
        if uncolored_count == 0:
            max_c = max(color) + 1
            if max_c < best_color_count:
                best_color_count = max_c
                best_coloring = list(color)
                if best_color_count == len(initial_coloring):
                    break
            
    return best_coloring, best_color_count

def solve_tabucol(n, adj, initial_k, initial_coloring, time_limit, start_time, logger):
    best_valid_coloring = list(initial_coloring)
    best_k = initial_k
    k = best_k - 1
    
    alpha = 0.6
    p0 = 10
    
    while k > 0 and (time.time() - start_time) < time_limit - 1.5:
        found_valid = False
        iteration_for_this_k = 0
        
        while not found_valid and iteration_for_this_k < 5 and (time.time() - start_time) < time_limit - 1.5:
            iteration_for_this_k += 1
            
            color = list(best_valid_coloring)
            for v in range(n):
                if color[v] >= k:
                    color[v] = random.randint(0, k - 1)
                    
            adj_color_count = [[0] * k for _ in range(n)]
            for v in range(n):
                for u in adj[v]:
                    adj_color_count[v][color[u]] += 1
                    
            conflicts = 0
            conflict_set = set()
            for v in range(n):
                c_v = adj_color_count[v][color[v]]
                conflicts += c_v
                if c_v > 0:
                    conflict_set.add(v)
            conflicts //= 2
            
            best_conflicts = conflicts
            tabu_tenure = [[0] * k for _ in range(n)]
            step = 0
            
            while step < 200000:
                if step % 1000 == 0 and (time.time() - start_time) >= time_limit - 1.5:
                    break
                    
                if conflicts == 0:
                    found_valid = True
                    break
                    
                if not conflict_set:
                    break
                    
                cv_list = list(conflict_set)
                if len(cv_list) > 30:
                    cv_subset = random.sample(cv_list, 30)
                else:
                    cv_subset = cv_list
                    
                best_moves = []
                min_delta = float('inf')
                
                for v in cv_subset:
                    old_c = color[v]
                    for new_c in range(k):
                        if old_c == new_c: continue
                        delta = adj_color_count[v][new_c] - adj_color_count[v][old_c]
                        
                        is_tabu = tabu_tenure[v][new_c] > step
                        
                        if delta < min_delta:
                            if not is_tabu or (conflicts + delta < best_conflicts):
                                min_delta = delta
                                best_moves = [(v, old_c, new_c)]
                        elif delta == min_delta:
                            if not is_tabu or (conflicts + delta < best_conflicts):
                                best_moves.append((v, old_c, new_c))
                                
                if not best_moves:
                    for v in cv_subset:
                        old_c = color[v]
                        for new_c in range(k):
                            if old_c == new_c: continue
                            delta = adj_color_count[v][new_c] - adj_color_count[v][old_c]
                            if delta < min_delta:
                                min_delta = delta
                                best_moves = [(v, old_c, new_c)]
                            elif delta == min_delta:
                                best_moves.append((v, old_c, new_c))
                                
                if not best_moves:
                    break
                    
                v, old_c, new_c = random.choice(best_moves)
                
                color[v] = new_c
                conflicts += min_delta
                
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    
                if adj_color_count[v][new_c] > 0:
                    conflict_set.add(v)
                else:
                    conflict_set.discard(v)
                    
                for u in adj[v]:
                    adj_color_count[u][old_c] -= 1
                    if color[u] == old_c and adj_color_count[u][old_c] == 0:
                        conflict_set.discard(u)
                        
                    adj_color_count[u][new_c] += 1
                    if color[u] == new_c and adj_color_count[u][new_c] > 0:
                        conflict_set.add(u)
                        
                tenure = step + int(p0 + random.random() * 10 + alpha * conflicts)
                tabu_tenure[v][old_c] = tenure
                
                step += 1
                
        if found_valid:
            best_valid_coloring = list(color)
            best_k = k
            if logger:
                logger.log(best_k)
            k -= 1
        else:
            break
            
    return best_valid_coloring, best_k

def extract_solution(coloring, n):
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
    bounds = instance_data.get("bounds", {})
    clique_lb = bounds.get("clique_lower_bound", 0)
    
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
    
    best_clique = find_large_clique_advanced(n, adj, iterations=50, max_time=min(5.0, args.time_limit * 0.1))
    k = len(best_clique)
    L = max(k, clique_lb)

    initial_coloring = {v: i for i, v in enumerate(best_clique)}
    best_dsatur_coloring, U_dsatur = dsatur_with_init(n, adj, initial_coloring, iterations=20, time_limit=args.time_limit, start_time=start_time)
    
    if logger:
        logger.log(U_dsatur)

    tabucol_coloring, U = solve_tabucol(n, adj, U_dsatur, best_dsatur_coloring, args.time_limit, start_time, logger)
    tabu_obj, tabu_selected, tabu_color_dict = extract_solution(tabucol_coloring, n)

    if not GUROBI_AVAILABLE or U <= L or args.time_limit - (time.time() - start_time) < 10:
        save_solution(tabu_obj, tabu_selected, tabu_color_dict, args.solution_path)
        return

    # Use Gurobi only if we still have plenty of time and Guorbi is available
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
        limit = min(U, degrees[u] + 1, degrees[v] + 1)
        for c in range(limit):
            if (u, c) in x and (v, c) in x:
                model.addConstr(x[u, c] + x[v, c] <= y[c])

    for i, v in enumerate(best_clique):
        if i < U:
            var = x.get((v, i))
            if var is not None:
                var.LB = 1

    for v in range(n):
        color = tabucol_coloring[v]
        for c in range(min(U, degrees[v] + 1)):
            if c == color:
                x[v, c].Start = 1
            else:
                x[v, c].Start = 0

    for c in range(U):
        y[c].Start = 1

    best_logged_obj = U

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
            return
    except Exception:
        pass

    save_solution(tabu_obj, tabu_selected, tabu_color_dict, args.solution_path)

if __name__ == "__main__":
    main()