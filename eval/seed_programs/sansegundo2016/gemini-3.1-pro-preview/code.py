import sys
import json
import random
import time
import argparse

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def get_bit_count(n):
    """Fallback for bit counting if Python < 3.10 is used."""
    try:
        return n.bit_count()
    except AttributeError:
        return bin(n).count('1')

def solve_max_clique(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    graph_data = data["graph"]
    num_vertices = graph_data["num_vertices"]
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path and SolutionLogger else None
    
    if num_vertices == 0:
        if logger:
            logger.log(0)
        sol = {"objective_value": 0, "clique_vertices": []}
        with open(solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    # Parse adjacency list and ensure integer keys/values
    adj_list_raw = graph_data["adjacency_list"]
    v_ints = [int(k) for k in adj_list_raw.keys()]
    v_ints.sort()
    
    if len(v_ints) == 0:
        if logger:
            logger.log(0)
        sol = {"objective_value": 0, "clique_vertices": []}
        with open(solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    # Vertex mapping to 0...N-1 for efficient bitmask operations
    v2idx = {v: i for i, v in enumerate(v_ints)}
    idx2v = {i: v for i, v in enumerate(v_ints)}
    
    N = len(v_ints)
    adj_mask = [0] * N
    degree = [0] * N
    
    for k_str, neighbors in adj_list_raw.items():
        u_orig = int(k_str)
        u_idx = v2idx[u_orig]
        mask = 0
        deg = 0
        for v_orig in neighbors:
            v_orig = int(v_orig)
            if v_orig != u_orig and v_orig in v2idx:
                v_idx = v2idx[v_orig]
                mask |= (1 << v_idx)
                deg += 1
        adj_mask[u_idx] = mask
        degree[u_idx] = deg
        
    # ----------------------------------------------------
    # Heuristic phase: Construct strong bounds quickly
    # ----------------------------------------------------
    best_clique = []
    max_size = 0
    
    def try_update(chosen_idx_list):
        nonlocal best_clique, max_size
        if len(chosen_idx_list) > max_size:
            max_size = len(chosen_idx_list)
            best_clique = chosen_idx_list[:]
            if logger:
                logger.log(max_size)

    # 1. Deterministic standard greedy (highest degree first)
    sorted_by_deg = sorted(range(N), key=lambda x: degree[x], reverse=True)
    c1 = []
    c1_mask = (1 << N) - 1
    for v in sorted_by_deg:
        if c1_mask & (1 << v):
            c1.append(v)
            c1_mask &= adj_mask[v]
    try_update(c1)
    
    # 2. Randomized Greedy with bitwise Local Search
    random.seed(42)
    heur_start = time.time()
    heur_time_limit = min(3.0, time_limit * 0.1)
    
    for _ in range(20):
        if time.time() - heur_start > heur_time_limit:
            break
            
        c_rand = []
        c_mask = (1 << N) - 1
        rand_v = list(range(N))
        
        # Mix completely random with degree-biased sequences
        if random.random() < 0.7:
            rand_v.sort(key=lambda x: degree[x] + random.randint(0, min(5, N)), reverse=True)
        else:
            random.shuffle(rand_v)
            
        for v in rand_v:
            if c_mask & (1 << v):
                c_rand.append(v)
                c_mask &= adj_mask[v]
                
        try_update(c_rand)
        
        # 1-opt Local Search (Swap exploration)
        c_list = c_rand[:]
        c_set = set(c_list)
        c_mask_val = c_mask
        
        improved = True
        while improved and time.time() - heur_start < heur_time_limit:
            improved = False
            cand_swap = []
            
            for v in range(N):
                if v not in c_set:
                    missing_edges = c_mask_val & ~adj_mask[v]
                    if missing_edges == 0:
                        # Full connection: can expand seamlessly
                        c_list.append(v)
                        c_set.add(v)
                        c_mask_val |= (1 << v)
                        try_update(c_list)
                        improved = True
                        break
                    else:
                        # Exactly one missing edge: isolated swap candidate structure
                        if (missing_edges & (missing_edges - 1)) == 0:
                            missing_u = missing_edges.bit_length() - 1
                            cand_swap.append((v, missing_u))
                            
            if not improved:
                for v, u in cand_swap:
                    # Simulated swap mask (c_mask without u, with v)
                    new_c_mask = (c_mask_val & ~(1 << u)) | (1 << v)
                    for w in range(N):
                        if w != v and w not in c_set:
                            # Verify w dominates the simulated mask entirely
                            if (new_c_mask & ~adj_mask[w]) == 0:
                                c_list.remove(u)
                                c_set.remove(u)
                                c_list.append(v)
                                c_set.add(v)
                                c_list.append(w)
                                c_set.add(w)
                                c_mask_val = new_c_mask | (1 << w)
                                try_update(c_list)
                                improved = True
                                break
                    if improved:
                        break

    # ----------------------------------------------------
    # Exact phase: Custom Vertex Coloring Branch-and-Bound
    # ----------------------------------------------------
    time_out = start_time + time_limit - 0.2
    sys.setrecursionlimit(max(sys.getrecursionlimit(), N + 500))
    
    def color_and_bounds_bitmask(P_mask):
        """Graph partitions to form strict coloring bounds for unexpanded search branches."""
        P_list = []
        temp_P = P_mask
        while temp_P:
            lsb = temp_P & (-temp_P)
            P_list.append(lsb.bit_length() - 1)
            temp_P &= temp_P - 1
            
        # Coloring prioritization mapping heuristics: process stable sub-degrees sequentially
        P_list.sort(key=lambda x: get_bit_count(adj_mask[x] & P_mask), reverse=True)
        
        C_masks = []
        C_lists = []
        
        for v in P_list:
            v_bit = 1 << v
            c = 0
            while c < len(C_masks):
                # Independent Set validation condition
                if not (adj_mask[v] & C_masks[c]):
                    C_masks[c] |= v_bit
                    C_lists[c].append(v)
                    break
                c += 1
            if c == len(C_masks):
                C_masks.append(v_bit)
                C_lists.append([v])
                
        P_sorted = []
        bounds = []
        color_idx = 1
        for c_list in C_lists:
            for v in c_list:
                P_sorted.append(v)
                bounds.append(color_idx)
            color_idx += 1
            
        return P_sorted, bounds

    def solve_exact(C, P_mask):
        nonlocal max_size, best_clique
        
        if time.time() > time_out:
            return
            
        if P_mask == 0:
            if len(C) > max_size:
                max_size = len(C)
                best_clique = C[:]
                if logger: logger.log(max_size)
            return
            
        P_sorted, bounds = color_and_bounds_bitmask(P_mask)
        
        # Build masks for overlapping hierarchical subsets in prefix order limits
        prefix_masks = [0] * len(P_sorted)
        curr = 0
        for i in range(len(P_sorted)):
            prefix_masks[i] = curr
            curr |= (1 << P_sorted[i])
            
        # Ascend back the reversed branch boundary definitions
        for i in range(len(P_sorted) - 1, -1, -1):
            if len(C) + bounds[i] <= max_size:
                break
            
            v = P_sorted[i]
            C.append(v)
            
            P_new_mask = prefix_masks[i] & adj_mask[v]
            if P_new_mask:
                solve_exact(C, P_new_mask)
            else:
                if len(C) > max_size:
                    max_size = len(C)
                    best_clique = C[:]
                    if logger: logger.log(max_size)
                    
            C.pop()

    # Trigger exact search phase
    if time_out > time.time():
        initial_P_mask = (1 << N) - 1
        solve_exact([], initial_P_mask)

    # ----------------------------------------------------
    # Solution evaluation & saving routines
    # ----------------------------------------------------
    final_clique_orig = [idx2v[idx] for idx in best_clique]
    final_clique_orig.sort()
    
    sol = {
        "objective_value": len(final_clique_orig),
        "clique_vertices": final_clique_orig
    }
    
    with open(solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Solve Maximum Clique Problem via Native Branch-and-Bound")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to JSON structural framework input")
    parser.add_argument("--solution_path", type=str, required=True, help="Destination directory bounds output string")
    parser.add_argument("--time_limit", type=int, required=True, help="Execution allocation boundaries in secs")
    parser.add_argument("--log_path", type=str, default=None, help="Diagnostic traversal file mapping")
    
    args = parser.parse_args()
    solve_max_clique(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == "__main__":
    main()