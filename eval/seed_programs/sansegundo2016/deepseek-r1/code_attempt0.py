import json
import time
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['graph']['num_vertices']
    adj_list_dict = data['graph']['adjacency_list']
    
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        key = str(i)
        if key in adj_list_dict:
            nbrs = adj_list_dict[key]
            adj_set[i] = set(nbrs)
    
    best_solution = set()
    best_size = 0

    # Greedy heuristic for initial solution
    candidates = set(range(n))
    current_clique = set()
    while candidates:
        max_deg = -1
        v = None
        for u in candidates:
            deg = len(adj_set[u] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = u
        if v is None:
            break
        current_clique.add(v)
        candidates = candidates & adj_set[v]
    
    if len(current_clique) > best_size:
        best_size = len(current_clique)
        best_solution = set(current_clique)
        if logger:
            logger.log(best_size)
    
    stack = []
    R0 = set()
    P0 = set(range(n))
    stack.append((R0, P0))
    
    start_time = time.time()
    
    while stack:
        current_time = time.time()
        if current_time - start_time > args.time_limit:
            break
        
        R, P = stack.pop()
        
        if len(R) > best_size:
            best_size = len(R)
            best_solution = set(R)
            if logger:
                logger.log(best_size)
        
        if not P:
            continue
        
        if len(R) + len(P) <= best_size:
            continue
        
        deg_in_P = {}
        for u in P:
            deg_in_P[u] = len(adj_set[u] & P)
        
        sorted_vertices = sorted(deg_in_P.keys(), key=lambda u: deg_in_P[u], reverse=True)
        
        color = {}
        for u in sorted_vertices:
            used_colors = set()
            for v in adj_set[u]:
                if v in P and v in color:
                    used_colors.add(color[v])
            c = 0
            while c in used_colors:
                c += 1
            color[u] = c
        
        color_count = max(color.values()) + 1 if color else 0
        bound = len(R) + color_count
        if bound <= best_size:
            continue
        
        v = None
        max_color = -1
        max_deg = -1
        for u in P:
            c_val = color[u]
            d_val = deg_in_P[u]
            if c_val > max_color or (c_val == max_color and d_val > max_deg):
                max_color = c_val
                max_deg = d_val
                v = u
        
        if v is None:
            continue
        
        P_exclude = P - {v}
        stack.append((R, P_exclude))
        
        R_include = R | {v}
        P_include = adj_set[v] & P
        stack.append((R_include, P_include))
    
    with open(args.solution_path, 'w') as f:
        json.dump({
            "objective_value": best_size,
            "clique_vertices": sorted(list(best_solution))
        }, f)

if __name__ == "__main__":
    main()