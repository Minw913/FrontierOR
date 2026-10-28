import json
import argparse
import random
import time
import sys
from solution_logger import SolutionLogger

def solution_from_coloring(coloring):
    max_c = max(coloring)
    obj = max_c + 1
    color_classes = {}
    for i, c in enumerate(coloring):
        color_classes.setdefault(c, []).append(i)
    return {
        "objective_value": obj,
        "coloring": {str(v): coloring[v] for v in range(len(coloring))},
        "color_classes": {str(c): color_classes[c] for c in color_classes}
    }

def greedy_color(n, adj, order):
    color = [-1] * n
    for v in order:
        used = set()
        for u in adj[v]:
            if color[u] != -1:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color

def smallest_last_order(n, adj):
    degree = [len(a) for a in adj]
    removed = [False] * n
    order = []
    for _ in range(n):
        v = -1
        min_deg = n + 1
        for i in range(n):
            if not removed[i] and degree[i] < min_deg:
                min_deg = degree[i]
                v = i
        removed[v] = True
        order.append(v)
        for u in adj[v]:
            if not removed[u]:
                degree[u] -= 1
    return order

def dsatur(n, adj):
    color = [-1] * n
    saturation = [0] * n
    degree = [len(a) for a in adj]
    uncolored = [True] * n
    uncolored_count = n
    
    v = max(range(n), key=lambda x: degree[x])
    color[v] = 0
    uncolored[v] = False
    uncolored_count -= 1
    for u in adj[v]:
        saturation[u] += 1
        
    while uncolored_count > 0:
        max_sat = -1
        best_v = -1
        best_deg = -1
        for i in range(n):
            if uncolored[i]:
                if saturation[i] > max_sat or (saturation[i] == max_sat and degree[i] > best_deg):
                    max_sat = saturation[i]
                    best_deg = degree[i]
                    best_v = i
        v = best_v
        used = set(color[u] for u in adj[v] if color[u] != -1)
        c = 0
        while c in used:
            c += 1
        color[v] = c
        uncolored[v] = False
        uncolored_count -= 1
        for u in adj[v]:
            if uncolored[u]:
                saturation[u] += 1
    return color

def dsatur_random(n, adj, rng):
    color = [-1] * n
    saturation = [0] * n
    degree = [len(a) for a in adj]
    uncolored = [True] * n
    uncolored_count = n
    
    max_deg = max(degree)
    candidates = [i for i in range(n) if degree[i] == max_deg]
    v = rng.choice(candidates)
    color[v] = 0
    uncolored[v] = False
    uncolored_count -= 1
    for u in adj[v]:
        saturation[u] += 1
        
    while uncolored_count > 0:
        max_sat = -1
        for i in range(n):
            if uncolored[i] and saturation[i] > max_sat:
                max_sat = saturation[i]
        
        sat_cands = [i for i in range(n) if uncolored[i] and saturation[i] == max_sat]
        if len(sat_cands) > 1:
            max_deg = max(degree[i] for i in sat_cands)
            deg_cands = [i for i in sat_cands if degree[i] == max_deg]
            v = rng.choice(deg_cands)
        else:
            v = sat_cands[0]
            
        used = set(color[u] for u in adj[v] if color[u] != -1)
        c = 0
        while c in used:
            c += 1
        color[v] = c
        uncolored[v] = False
        uncolored_count -= 1
        for u in adj[v]:
            if uncolored[u]:
                saturation[u] += 1
    return color

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
    
    n = instance['graph']['n']
    adj = instance['graph']['adjacency_list']
    
    start_time = time.time()
    
    # Fast greedy first
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    best_coloring = greedy_color(n, adj, order)
    best_sol = solution_from_coloring(best_coloring)
    if logger:
        logger.log_solution(best_sol["objective_value"], best_sol)
        
    # Smallest-last greedy
    if time.time() - start_time < args.time_limit * 0.5:
        order = smallest_last_order(n, adj)
        c = greedy_color(n, adj, order)
        sol = solution_from_coloring(c)
        if sol["objective_value"] < best_sol["objective_value"]:
            best_sol = sol
            best_coloring = c
            if logger:
                logger.log_solution(best_sol["objective_value"], best_sol)
                
    # Standard DSATUR
    if time.time() - start_time < args.time_limit * 0.8:
        try:
            c = dsatur(n, adj)
            sol = solution_from_coloring(c)
            if sol["objective_value"] < best_sol["objective_value"]:
                best_sol = sol
                best_coloring = c
                if logger:
                    logger.log_solution(best_sol["objective_value"], best_sol)
        except Exception:
            pass
            
    # Randomized DSATUR restarts
    rng = random.Random(0)
    while time.time() - start_time < args.time_limit - 1.0:
        try:
            c = dsatur_random(n, adj, rng)
            sol = solution_from_coloring(c)
            if sol["objective_value"] < best_sol["objective_value"]:
                best_sol = sol
                best_coloring = c
                if logger:
                    logger.log_solution(best_sol["objective_value"], best_sol)
        except Exception:
            break
            
    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f, indent=2)

if __name__ == '__main__':
    main()