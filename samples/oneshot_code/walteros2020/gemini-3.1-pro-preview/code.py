import argparse
import json
import time
import random
import sys

from solution_logger import SolutionLogger

# Global variables for B&B
max_size = 0
best_clique_new_ids = []
search_time_limit = 0
logger_global = None

def get_colored_candidates(P_list, adj_masks):
    """
    Greedy coloring of candidates to find independent sets.
    Returns the color assigned to each candidate, while preserving the color-ascending order in P_list.
    """
    color_classes = [] 
    color_of = {}
    for v in P_list:
        c = 0
        v_adj = adj_masks[v]
        while c < len(color_classes):
            if not (color_classes[c] & v_adj):
                break
            c += 1
        if c == len(color_classes):
            color_classes.append(1 << v)
        else:
            color_classes[c] |= (1 << v)
        color_of[v] = c + 1
        
    # Sort such that lower colors (pruned later when iterating backwards) are at start.
    P_list.sort(key=lambda x: color_of[x])
    return [color_of[v] for v in P_list]

def bitset_search(R, P_list, adj_masks):
    """
    Branch and Bound recursive search leveraging bitmasks and coloring bounds.
    """
    global max_size, best_clique_new_ids, search_time_limit, logger_global
    
    if time.time() > search_time_limit:
        return
        
    if not P_list:
        if len(R) > max_size:
            max_size = len(R)
            best_clique_new_ids = R.copy()
            if logger_global:
                logger_global.log(max_size)
        return
        
    # Get bounds
    colors = get_colored_candidates(P_list, adj_masks)
    
    while P_list:
        v = P_list.pop()
        c = colors.pop()
        
        # Pruning based on the maximum possible clique size extension
        if len(R) + c <= max_size:
            break
            
        v_adj = adj_masks[v]
        # Intersect with neighborhood
        new_P_list = [u for u in P_list if (1 << u) & v_adj]
        
        R.append(v)
        bitset_search(R, new_P_list, adj_masks)
        R.pop()


def degeneracy_ordering(n, adj_list):
    """
    Computes a degeneracy ordering of the graph.
    Vertices removed last (highest core) will appear at the end of the order list.
    """
    deg = [len(adj_list[i]) for i in range(n)]
    active = [True] * n
    
    order = []
    
    max_deg = max(deg) if n > 0 else 0
    deg_lists = [[] for _ in range(max_deg + 1)]
    for v in range(n):
        deg_lists[deg[v]].append(v)
        
    for _ in range(n):
        v = -1
        for d in range(max_deg + 1):
            while deg_lists[d]:
                candidate = deg_lists[d].pop()
                if active[candidate]:
                    v = candidate
                    break
            else:
                continue
            break
            
        if v == -1: 
            break
            
        active[v] = False
        order.append(v)
        
        for u in adj_list[v]:
            if active[u]:
                deg[u] -= 1
                deg_lists[deg[u]].append(u)
                
    # Place highest-core vertices at the start of our new index space
    return order[::-1]

def main():
    global max_size, best_clique_new_ids, search_time_limit, logger_global
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    proc_start_time = time.time()
    
    # Initialize the solution logger tracking progress
    logger_global = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Buffer time logic ensures output dump gets processed before the runner strictly kills the logic
    search_time_limit = proc_start_time + args.time_limit - 1.0 
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['n']
    edges = data['edges']
    
    if n == 0:
        output_data = {
            "objective_value": 0,
            "clique_vertices": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        return
        
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    adj_set = [set(neighbors) for neighbors in adj_list]
    
    # --- PHASE 1: Randomized Greedy Heuristic ---
    heur_time_allowed = max(1.0, min(5.0, args.time_limit * 0.05))
    heuristic_end = min(search_time_limit, proc_start_time + heur_time_allowed)
    
    best_size = 0
    best_clique_old_ids = []
    
    deg = [len(adj_set[i]) for i in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: deg[x], reverse=True)
    pool_size = max(1, n // 5)
    
    while time.time() < heuristic_end:
        start_node = random.choice(vertices[:pool_size]) 
        clq = [start_node]
        candidates = set(adj_set[start_node])
        while candidates:
            if random.random() < 0.2:
                for c in candidates:
                    best_c = c
                    break
            else:
                sample = random.sample(list(candidates), min(10, len(candidates)))
                best_c = max(sample, key=lambda c: len(candidates.intersection(adj_set[c])))
            clq.append(best_c)
            candidates.intersection_update(adj_set[best_c])
            
        if len(clq) > best_size:
            best_size = len(clq)
            best_clique_old_ids = clq
            if logger_global:
                logger_global.log(best_size)
                
    max_size = best_size
    
    # --- PHASE 2: Graph Reductions and Exact Fast Branch & Bound ---
    new_to_old = degeneracy_ordering(n, adj_list)
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    
    best_clique_new_ids = [old_to_new[v] for v in best_clique_old_ids]
    
    # Build integer bitmasks for efficient bitwise set intersections mapped to new IDs
    adj_masks = [0] * n
    for u in range(n):
        for v_old in adj_list[new_to_old[u]]:
            v = old_to_new[v_old]
            adj_masks[u] |= (1 << v)
            
    # Core pruning - Strip sub-graphs not capable of exceeding our heuristically found best sizing
    active = [True] * n
    new_deg = [len(adj_list[new_to_old[i]]) for i in range(n)]
    queue = [i for i in range(n) if new_deg[i] < max_size - 1]
    
    while queue:
        u = queue.pop()
        if not active[u]: continue
        active[u] = False
        for v_old in adj_list[new_to_old[u]]:
            v = old_to_new[v_old]
            if active[v]:
                new_deg[v] -= 1
                if new_deg[v] == max_size - 2:
                    queue.append(v)
                    
    # Generate pruned top level candidates for B&B
    P_list = [i for i in range(n) if active[i]]
    P_list.sort(key=lambda x: new_deg[x])
    
    # Launch branch-and-bound recursion core
    bitset_search([], P_list, adj_masks)
    
    final_clique = [new_to_old[v] for v in best_clique_new_ids]
    
    output_data = {
        "objective_value": len(final_clique),
        "clique_vertices": [int(v) for v in final_clique]
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    sys.setrecursionlimit(20000)
    main()