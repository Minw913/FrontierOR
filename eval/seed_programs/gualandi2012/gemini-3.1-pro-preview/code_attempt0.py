import sys
import json
import time
import argparse
import random
import traceback

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs):
            pass
        def log(self, *args):
            pass

def heuristic_cliques(n, adj, adj_set):
    nodes_by_degree = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    best_clique = []
    
    limit = min(n, max(50, 20000 // max(1, n)))
    
    for i in range(limit):
        clique = [nodes_by_degree[i]]
        for v in nodes_by_degree:
            if v == nodes_by_degree[i]:
                continue
            connected = True
            for u in clique:
                if u not in adj_set[v]:
                    connected = False
                    break
            if connected:
                clique.append(v)
        if len(clique) > len(best_clique):
            best_clique = clique
            
    return best_clique

def dsatur_with_clique(n, adj, clique):
    uncolored_degrees = [len(adj[v]) for v in range(n)]
    colors = [-1] * n
    uncolored = set(range(n))
    adj_colors = [set() for _ in range(n)]
    
    for i, v in enumerate(clique):
        colors[v] = i
        uncolored.remove(v)
        for u in adj[v]:
            adj_colors[u].add(i)
            uncolored_degrees[u] -= 1
            
    while uncolored:
        best_u = -1
        max_sat = -1
        max_deg = -1
        
        for u in uncolored:
            sat = len(adj_colors[u])
            deg = uncolored_degrees[u]
            if sat > max_sat or (sat == max_sat and deg > max_deg):
                best_u = u
                max_sat = sat
                max_deg = deg
                
        c = 0
        while c in adj_colors[best_u]:
            c += 1
        
        colors[best_u] = c
        uncolored.remove(best_u)
        
        for u in adj[best_u]:
            if colors[u] == -1:
                adj_colors[u].add(c)
                uncolored_degrees[u] -= 1
                
    return colors, max(colors) + 1

def tabucol(n, adj, k, initial_colors, max_iter, time_limit, start_time):
    colors = list(initial_colors)
    conf = [0] * n
    total_conflicts = 0
    for u in range(n):
        c = 0
        u_color = colors[u]
        for v in adj[u]:
            if u_color == colors[v]:
                c += 1
        conf[u] = c
        total_conflicts += c
    total_conflicts //= 2
    
    if total_conflicts == 0:
        return colors
        
    color_adj_count = [[0]*k for _ in range(n)]
    for u in range(n):
        u_adj_count = color_adj_count[u]
        for v in adj[u]:
            u_adj_count[colors[v]] += 1
            
    best_conflicts = total_conflicts
    best_colors = list(colors)
    
    tabu = [[0]*k for _ in range(n)]
    iter_count = 0
    
    while iter_count < max_iter and total_conflicts > 0:
        if (iter_count % 1000) == 0:
            if time.time() - start_time > time_limit:
                break
                
        best_moves = []
        best_delta = float('inf')
        
        conflicting_vertices = [v for v in range(n) if conf[v] > 0]
        if not conflicting_vertices:
            break
            
        for v in conflicting_vertices:
            old_c = colors[v]
            v_counts = color_adj_count[v]
            base_count = v_counts[old_c]
            v_tabu = tabu[v]
            
            for new_c in range(k):
                if new_c == old_c:
                    continue
                delta = v_counts[new_c] - base_count
                
                is_tabu = v_tabu[new_c] > iter_count
                
                if delta < best_delta:
                    if not is_tabu or (total_conflicts + delta < best_conflicts):
                        best_delta = delta
                        best_moves = [(v, new_c)]
                elif delta == best_delta:
                    if not is_tabu or (total_conflicts + delta < best_conflicts):
                        best_moves.append((v, new_c))
                        
        if not best_moves:
            v = random.choice(conflicting_vertices)
            new_c = random.randint(0, k-1)
            while new_c == colors[v]:
                new_c = random.randint(0, k-1)
            best_move = (v, new_c)
            best_delta = color_adj_count[v][new_c] - color_adj_count[v][colors[v]]
        else:
            best_move = random.choice(best_moves)
            
        v, new_c = best_move
        old_c = colors[v]
        
        colors[v] = new_c
        total_conflicts += best_delta
        
        if total_conflicts < best_conflicts:
            best_conflicts = total_conflicts
            best_colors = list(colors)
            
        conf[v] = color_adj_count[v][new_c]
        for u in adj[v]:
            u_adj_count = color_adj_count[u]
            u_adj_count[old_c] -= 1
            u_adj_count[new_c] += 1
            u_color = colors[u]
            if u_color == old_c:
                conf[u] -= 1
            if u_color == new_c:
                conf[u] += 1
                
        tenure = int(0.6 * total_conflicts + random.randint(0, 10))
        tabu[v][old_c] = iter_count + tenure
        iter_count += 1
        
    if best_conflicts == 0:
        return best_colors
    return None

def run_tabu_phases(n, adj, initial_colors, u_bound, L, time_limit, start_time, logger):
    current_colors = initial_colors
    current_k = u_bound
    
    while current_k > L:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        target_k = current_k - 1
        found = False
        
        for attempt in range(5):
            rem_time = time_limit - (time.time() - start_time)
            if rem_time <= 0:
                break
                
            new_colors = []
            for c in current_colors:
                if c == target_k:
                    new_colors.append(random.randint(0, target_k-1))
                else:
                    new_colors.append(c)
                    
            res = tabucol(n, adj, target_k, new_colors, 200000, rem_time, time.time())
            if res is not None:
                current_colors = res
                current_k = target_k
                if logger:
                    logger.log(current_k)
                found = True
                break
                
        if not found:
            break
            
    return current_colors, current_k

def compress_colors(colors):
    unique = sorted(list(set(colors)))
    mapping = {old_c: new_c for new_c, old_c in enumerate(unique)}
    return [mapping[c] for c in colors], len(unique)

def remap_colors(n, current_colors, U, clique):
    K = len(clique)
    mapping = {}
    
    for i, v in enumerate(clique):
        c = current_colors[v]
        mapping[c] = i
        
    next_color = K
    for c in range(U):
        if c not in mapping:
            mapping[c] = next_color
            next_color += 1
            
    remapped = [mapping[current_colors[v]] for v in range(n)]
    return remapped

def solve_with_gurobi(n, edges, U, L, current_colors, clique, time_limit, logger):
    import gurobipy as gp
    from gurobipy import GRB
    
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Threads', 1)
    env.start()
    model = gp.Model("GraphColoring", env=env)
    
    y = model.addVars(U, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, U, vtype=GRB.BINARY, name="x")
    
    model.setObjective(gp.quicksum(y[c] for c in range(U)), GRB.MINIMIZE)
    
    model.addConstrs((x.sum(v, '*') == 1 for v in range(n)), name="assign")
    model.addConstrs((x[u, c] + x[v, c] <= y[c] for u, v in edges for c in range(U)), name="edge")
    model.addConstrs((y[c] <= y[c-1] for c in range(1, U)), name="sym")
    
    for idx, v in enumerate(clique):
        model.addConstr(x[v, idx] == 1)
        model.addConstr(y[idx] == 1)
        
    if L > 0:
        model.addConstr(gp.quicksum(y[c] for c in range(U)) >= L)
        
    for v in range(n):
        c = current_colors[v]
        x[v, c].Start = 1
        for c_other in range(U):
            if c_other != c:
                x[v, c_other].Start = 0
                
    for c in range(U):
        y[c].Start = 1
        
    model.Params.TimeLimit = time_limit
    
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(int(round(obj)))
                
    model.optimize(my_callback)
    
    if model.SolCount > 0:
        best_k = int(round(model.objVal))
        final_colors = [-1] * n
        for v in range(n):
            for c in range(U):
                if x[v, c].X > 0.5:
                    final_colors[v] = c
                    break
        return final_colors, best_k
    else:
        return current_colors, U

def output_solution(path, U, colors):
    coloring = {str(i): colors[i] for i in range(len(colors))}
    color_classes = {}
    for i, c in enumerate(colors):
        str_c = str(c)
        if str_c not in color_classes:
            color_classes[str_c] = []
        color_classes[str_c].append(i)
        
    res = {
        "objective_value": U,
        "coloring": coloring,
        "color_classes": color_classes
    }
    with open(path, 'w') as f:
        json.dump(res, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    graph = data['graph']
    n = graph['num_vertices']
    input_edges = graph['edges']
    
    adj_set = [set() for _ in range(n)]
    for u, v in input_edges:
        if u != v:
            adj_set[u].add(v)
            adj_set[v].add(u)
            
    adj = [list(c) for c in adj_set]
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                
    num_edges = len(edges)
    
    if num_edges == 0:
        colors = [0] * n
        U = 1
        if logger:
            logger.log(U)
        output_solution(args.solution_path, U, colors)
        return
        
    clique = heuristic_cliques(n, adj, adj_set)
    L = len(clique)
    
    colors, U = dsatur_with_clique(n, adj, clique)
    if logger:
        logger.log(U)
        
    num_edges_times_u = num_edges * U
    
    if num_edges_times_u <= 4_000_000:
        tabu_time = args.time_limit * 0.4
    else:
        tabu_time = args.time_limit - 2.0
        
    colors, U = run_tabu_phases(n, adj, colors, U, L, tabu_time, start_time, logger)
    colors, actual_U = compress_colors(colors)
    if actual_U < U:
        U = actual_U
        if logger:
            logger.log(U)
            
    rem_time = args.time_limit - (time.time() - start_time) - 1.0
    
    if (num_edges * U <= 4_000_000) and U > L and rem_time > 5.0:
        colors = remap_colors(n, colors, U, clique)
        final_colors, final_U = solve_with_gurobi(n, edges, U, L, colors, clique, rem_time, logger)
        colors, U = final_colors, final_U
        
        colors, actual_U = compress_colors(colors)
        if actual_U < U:
            U = actual_U
            
    output_solution(args.solution_path, U, colors)

if __name__ == "__main__":
    main()