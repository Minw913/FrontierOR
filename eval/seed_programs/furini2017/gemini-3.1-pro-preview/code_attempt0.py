import argparse
import json
import time
import random

def check_time(start_time, limit, margin=1.0):
    return time.time() - start_time > limit - margin

def dsatur_randomized(n, adj):
    color = [-1] * n
    if n == 0:
        return color, 0
    degree = [len(x) for x in adj]
    sat_degree = [0] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))
    
    while uncolored:
        max_sat = -1
        max_deg = -1
        best_vs = []
        for v in uncolored:
            if sat_degree[v] > max_sat:
                max_sat = sat_degree[v]
                max_deg = degree[v]
                best_vs = [v]
            elif sat_degree[v] == max_sat:
                if degree[v] > max_deg:
                    max_deg = degree[v]
                    best_vs = [v]
                elif degree[v] == max_deg:
                    best_vs.append(v)
                    
        v = best_vs[0] if len(best_vs) == 1 else random.choice(best_vs)
        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1
            
        color[v] = c
        uncolored.remove(v)
        for u in adj[v]:
            if color[u] == -1:
                degree[u] -= 1
                if c not in neighbor_colors[u]:
                    neighbor_colors[u].add(c)
                    sat_degree[u] += 1
                    
    return color, max(color) + 1

def tabu_search_for_k(n, adj, target_k, initial_color, time_limit_total, start_time):
    color = list(initial_color)
    for v in range(n):
        if color[v] >= target_k:
            counts = [0] * target_k
            for u in adj[v]:
                c = color[u]
                if c < target_k:
                    counts[c] += 1
            min_c = min(counts)
            best_cs = [c for c in range(target_k) if counts[c] == min_c]
            color[v] = random.choice(best_cs)
            
    adj_color_count = [[0] * target_k for _ in range(n)]
    for i in range(n):
        for u in adj[i]:
            if color[u] < target_k:
                adj_color_count[i][color[u]] += 1
                
    conflicts = []
    conflict_pos = [-1] * n

    f = 0
    for i in range(n):
        if adj_color_count[i][color[i]] > 0:
            conflict_pos[i] = len(conflicts)
            conflicts.append(i)
            f += adj_color_count[i][color[i]]
    f //= 2
    
    best_f = f
    if f == 0:
        return color, f
        
    tabu_matrix = [[0] * target_k for _ in range(n)]
    iter_count = 0
    max_iter_without_improve = 2000 + 10 * n
    iter_since_improve = 0

    while True:
        if iter_count % 500 == 0:
            if check_time(start_time, time_limit_total, margin=1.0):
                break
        
        if not conflicts:
            best_f = 0
            break
            
        best_delta = 999999
        candidates = []
        
        max_c = max(1, 4000 // target_k)
        if len(conflicts) > max_c:
            eval_conflicts = random.sample(conflicts, max_c)
        else:
            eval_conflicts = conflicts
            
        for v in eval_conflicts:
            cv = color[v]
            acc_v = adj_color_count[v]
            cv_count = acc_v[cv]
            tabu_v = tabu_matrix[v]
            
            for c in range(target_k):
                if c == cv: continue
                delta = acc_v[c] - cv_count
                
                if tabu_v[c] > iter_count and f + delta >= best_f:
                    continue
                    
                if delta < best_delta:
                    best_delta = delta
                    candidates = [(v, c)]
                elif delta == best_delta:
                    candidates.append((v, c))
                    
        if not candidates:
            v_move = random.choice(conflicts)
            c_new = random.randint(0, target_k - 1)
            while c_new == color[v_move]:
                c_new = random.randint(0, target_k - 1)
        else:
            v_move, c_new = random.choice(candidates)
            
        c_old = color[v_move]
        
        # Inline move execution
        f += (adj_color_count[v_move][c_new] - adj_color_count[v_move][c_old])
        color[v_move] = c_new
        
        for u in adj[v_move]:
            adj_color_count[u][c_old] -= 1
            if color[u] == c_old:
                if adj_color_count[u][c_old] == 0:
                    pos = conflict_pos[u]
                    if pos != -1:
                        last_u = conflicts[-1]
                        conflicts[pos] = last_u
                        conflict_pos[last_u] = pos
                        conflicts.pop()
                        conflict_pos[u] = -1
            elif color[u] == c_new:
                if adj_color_count[u][c_new] == 1:
                    if conflict_pos[u] == -1:
                        conflict_pos[u] = len(conflicts)
                        conflicts.append(u)
            adj_color_count[u][c_new] += 1
            
        if adj_color_count[v_move][c_new] > 0:
            if conflict_pos[v_move] == -1:
                conflict_pos[v_move] = len(conflicts)
                conflicts.append(v_move)
        else:
            pos = conflict_pos[v_move]
            if pos != -1:
                last_v = conflicts[-1]
                conflicts[pos] = last_v
                conflict_pos[last_v] = pos
                conflicts.pop()
                conflict_pos[v_move] = -1
        
        if f < best_f:
            best_f = f
            iter_since_improve = 0
        else:
            iter_since_improve += 1
            
        tabu_tenure = random.randint(0, 9) + int(0.6 * len(conflicts))
        tabu_matrix[v_move][c_old] = iter_count + tabu_tenure
        
        iter_count += 1
        
        if f == 0:
            break
            
        if iter_since_improve > max_iter_without_improve:
            num_perturb = max(1, n // 10)
            for _ in range(num_perturb):
                if conflicts:
                    v_p = random.choice(conflicts)
                else:
                    v_p = random.randint(0, n - 1)
                new_c = random.randint(0, target_k - 1)
                
                old_c = color[v_p]
                if old_c == new_c: 
                    continue
                
                f += (adj_color_count[v_p][new_c] - adj_color_count[v_p][old_c])
                color[v_p] = new_c
                
                for u in adj[v_p]:
                    adj_color_count[u][old_c] -= 1
                    if color[u] == old_c:
                        if adj_color_count[u][old_c] == 0:
                            pos = conflict_pos[u]
                            if pos != -1:
                                last_u = conflicts[-1]
                                conflicts[pos] = last_u
                                conflict_pos[last_u] = pos
                                conflicts.pop()
                                conflict_pos[u] = -1
                    elif color[u] == new_c:
                        if adj_color_count[u][new_c] == 1:
                            if conflict_pos[u] == -1:
                                conflict_pos[u] = len(conflicts)
                                conflicts.append(u)
                    adj_color_count[u][new_c] += 1
                    
                if adj_color_count[v_p][new_c] > 0:
                    if conflict_pos[v_p] == -1:
                        conflict_pos[v_p] = len(conflicts)
                        conflicts.append(v_p)
                else:
                    pos = conflict_pos[v_p]
                    if pos != -1:
                        last_v = conflicts[-1]
                        conflicts[pos] = last_v
                        conflict_pos[last_v] = pos
                        conflicts.pop()
                        conflict_pos[v_p] = -1

            iter_since_improve = 0

    return color, f

def solve_graph_coloring():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data["graph"]["n"]
    adj = data["graph"]["adjacency_list"]

    best_color = []
    best_k = n + 1
    
    # Fast Greedy Randomized Multi-start
    for _ in range(10):
        col, k = dsatur_randomized(n, adj)
        if k < best_k:
            best_k = k
            best_color = col
            if logger:
                logger.log(best_k)
        if check_time(start_time, args.time_limit, margin=1.0):
            break

    # Tabucol Optimization Loop
    target_k = best_k - 1
    current_color = best_color
    
    while target_k > 0:
        if check_time(start_time, args.time_limit, margin=1.0):
            break
            
        color, f = tabu_search_for_k(n, adj, target_k, current_color, args.time_limit, start_time)
        
        if f == 0:
            best_k = target_k
            best_color = color
            current_color = color
            if logger:
                logger.log(best_k)
            target_k -= 1
        else:
            # Time limit reached or strictly trapped locally without improvements
            break

    # Solution Formatting
    coloring_str = {str(i): best_color[i] for i in range(n)}
    color_classes_str = {}
    for i in range(n):
        c_str = str(best_color[i])
        if c_str not in color_classes_str:
            color_classes_str[c_str] = []
        color_classes_str[c_str].append(i)
        
    solution = {
        "objective_value": best_k,
        "coloring": coloring_str,
        "color_classes": color_classes_str
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    solve_graph_coloring()