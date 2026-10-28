import time
import random
import json
import argparse
from solution_logger import SolutionLogger

def dsatur(V, adj):
    N = len(V)
    color = [-1] * N
    sat = [0] * N
    deg = [len(adj[i]) for i in range(N)]
    uncolored = set(V)
    
    if not uncolored:
        return color
        
    u = max(V, key=lambda x: deg[x])
    color[u] = 0
    uncolored.remove(u)
    
    for v in adj[u]:
        if color[v] == -1:
            sat[v] += 1
            
    while uncolored:
        max_sat = -1
        max_deg = -1
        best = -1
        for v in uncolored:
            if sat[v] > max_sat or (sat[v] == max_sat and deg[v] > max_deg):
                max_sat = sat[v]
                max_deg = deg[v]
                best = v
                
        u = best
        used_colors = set()
        for v in adj[u]:
            if color[v] != -1:
                used_colors.add(color[v])
        c = 0
        while c in used_colors:
            c += 1
        color[u] = c
        uncolored.remove(u)
        
        for v in adj[u]:
            if color[v] == -1:
                sat[v] += 1
                
    return color

def tabu_search(V, E, adj, k, init_color, time_limit, start_time, max_iter=100000):
    end_time = start_time + time_limit
    color = init_color.copy()
    
    for v in V:
        if color[v] >= k:
            color[v] = random.randint(0, k - 1)
            
    N = len(V)
    gamma = [[0] * k for _ in range(N)]
    for edge in E:
        u, v = edge
        gamma[u][color[v]] += 1
        gamma[v][color[u]] += 1
        
    conflicts = 0
    conflicting_vertices = set()
    for u in V:
        if gamma[u][color[u]] > 0:
            conflicting_vertices.add(u)
            conflicts += gamma[u][color[u]]
    conflicts //= 2
    
    tabu = [[0] * k for _ in range(N)]
    best_conflicts = conflicts
    iter = 0
    
    while conflicts > 0 and iter < max_iter:
        iter += 1
        if iter % 100 == 0:
            if time.time() > end_time:
                break
                
        if not conflicting_vertices:
            break
            
        best_move = None
        best_delta = float('inf')
        
        for u in list(conflicting_vertices):
            c = color[u]
            for c_new in range(k):
                if c_new == c: continue
                delta = gamma[u][c] - gamma[u][c_new]
                if tabu[u][c_new] > iter and delta >= 0:
                    continue
                if delta < best_delta or (delta == best_delta and random.random() < 0.1):
                    best_delta = delta
                    best_move = (u, c_new, c)
                    
        if best_move is None:
            u = random.choice(list(conflicting_vertices))
            c_new = random.randint(0, k - 1)
            if c_new == color[u]:
                c_new = (c_new + 1) % k
            best_move = (u, c_new, color[u])
            best_delta = gamma[u][color[u]] - gamma[u][c_new]
            
        u, c_new, c_old = best_move
        color[u] = c_new
        conflicts += best_delta
        
        for v in adj[u]:
            gamma[v][c_old] -= 1
            gamma[v][c_new] += 1
            
            if gamma[v][color[v]] > 0:
                conflicting_vertices.add(v)
            else:
                conflicting_vertices.discard(v)
                
        if gamma[u][color[u]] > 0:
            conflicting_vertices.add(u)
        else:
            conflicting_vertices.discard(u)
            
        tabu[u][c_old] = iter + 10 + random.randint(0, 10)
        
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            if conflicts == 0:
                return color, 0
                
    return color, conflicts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path) as f:
        data = json.load(f)
        
    N = data["graph"]["num_vertices"]
    E = data["graph"]["edges"]
    
    V = list(range(N))
    adj = [set() for _ in range(N)]
    for edge in E:
        u, v = edge
        adj[u].add(v)
        adj[v].add(u)
        
    start_time = time.time()
    
    best_color = dsatur(V, adj)
    UB = max(best_color) + 1 if best_color else 0
    
    def make_sol_dict(color):
        obj = max(color) + 1 if color else 0
        cc = {}
        for v in V:
            c = color[v]
            if c not in cc:
                cc[c] = []
            cc[c].append(v)
        return {
            "objective_value": obj,
            "coloring": {str(v): c for v, c in enumerate(color)},
            "color_classes": {str(c): cls for c, cls in cc.items()}
        }
        
    if logger:
        logger.log_solution(UB, make_sol_dict(best_color))
        
    while UB > 1 and time.time() - start_time < args.time_limit - 1:
        k = UB - 1
        success = False
        while time.time() - start_time < args.time_limit - 1 and not success:
            init_color = best_color.copy()
            for v in V:
                if init_color[v] >= k:
                    init_color[v] = random.randint(0, k-1)
                    
            remaining = args.time_limit - (time.time() - start_time)
            time_slice = min(remaining, max(5, k * 2)) 
            
            new_color, conflicts = tabu_search(V, E, adj, k, init_color, time_slice, start_time)
            if conflicts == 0:
                UB = k
                best_color = new_color
                if logger:
                    logger.log_solution(UB, make_sol_dict(best_color))
                success = True
            else:
                if time.time() - start_time >= args.time_limit - 1:
                    break
        if not success:
            break
            
    final_sol = make_sol_dict(best_color)
    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=4)

if __name__ == "__main__":
    main()