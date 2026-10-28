import argparse
import json
import time
import sys
from collections import defaultdict

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    edges = instance['edges']
    omega = instance.get('omega', None)
    degeneracy = instance.get('degeneracy', None)

    start_time = time.time()
    time_limit = args.time_limit

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Use bitset representation for faster intersection
    # For large graphs, we'll use frozensets; for smaller ones, integers as bitmasks
    
    best_clique = []
    
    def time_up():
        return time.time() - start_time >= time_limit - 0.5

    # Greedy initial solution
    def greedy_clique():
        # Start from highest degree vertex
        degrees = [(len(adj[v]), v) for v in range(n)]
        degrees.sort(reverse=True)
        
        best = []
        # Try starting from several high-degree vertices
        for _, start in degrees[:min(n, 50)]:
            if time_up():
                break
            clique = [start]
            clique_set = {start}
            # Candidates: neighbors of start
            candidates = list(adj[start])
            # Sort by number of connections to current clique (desc), then degree (desc)
            while candidates:
                # Score each candidate
                scored = []
                for c in candidates:
                    connections = sum(1 for v in clique if c in adj[v])
                    if connections == len(clique):
                        scored.append((len(adj[c]), c))
                if not scored:
                    break
                scored.sort(reverse=True)
                chosen = scored[0][1]
                clique.append(chosen)
                clique_set.add(chosen)
                candidates = [c for c in candidates if c != chosen and c in adj[chosen]]
            
            if len(clique) > len(best):
                best = clique[:]
        return best

    best_clique = greedy_clique()
    if logger:
        logger.log(len(best_clique))

    # For small/medium graphs, use Bron-Kerbosch with pivoting and pruning
    # For very large graphs, rely on heuristics
    
    use_bitmask = n <= 512
    
    if use_bitmask and n <= 512:
        # Use integer bitmasks for fast set operations
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Degeneracy ordering
        def degeneracy_ordering():
            deg = [len(adj[v]) for v in range(n)]
            removed = [False] * n
            order = []
            for _ in range(n):
                # Find vertex with minimum degree among remaining
                min_deg = n + 1
                min_v = -1
                for v in range(n):
                    if not removed[v] and deg[v] < min_deg:
                        min_deg = deg[v]
                        min_v = v
                order.append(min_v)
                removed[min_v] = True
                for u in adj[min_v]:
                    if not removed[u]:
                        deg[u] -= 1
            return order
        
        deg_order = degeneracy_ordering()
        vertex_pos = [0] * n
        for i, v in enumerate(deg_order):
            vertex_pos[v] = i
        
        # Branch and bound with Bron-Kerbosch, processing in reverse degeneracy order
        node_count = [0]
        
        def bron_kerbosch_bitmask(clique_mask, clique_size, P_mask, X_mask):
            nonlocal best_clique
            
            if P_mask == 0 and X_mask == 0:
                if clique_size > len(best_clique):
                    best_clique = []
                    temp = clique_mask
                    bit = 0
                    while temp:
                        if temp & 1:
                            best_clique.append(bit)
                        temp >>= 1
                        bit += 1
                    if logger:
                        logger.log(len(best_clique))
                return
            
            if P_mask == 0:
                return
            
            # Upper bound: clique_size + |P|
            p_count = bin(P_mask).count('1')
            if clique_size + p_count <= len(best_clique):
                return
            
            if time_up():
                return
            
            node_count[0] += 1
            
            # Choose pivot that maximizes |P ∩ N(pivot)| from P ∪ X
            union_mask = P_mask | X_mask
            best_pivot = -1
            best_pivot_count = -1
            
            temp = union_mask
            while temp:
                v = (temp & -temp).bit_length() - 1
                count = bin(P_mask & adj_mask[v]).count('1')
                if count > best_pivot_count:
                    best_pivot_count = count
                    best_pivot = v
                temp &= temp - 1
            
            # Iterate over P \ N(pivot)
            candidates = P_mask & ~adj_mask[best_pivot]
            
            while candidates:
                v_bit = candidates & -candidates
                v = v_bit.bit_length() - 1
                candidates &= candidates - 1
                
                new_P = P_mask & adj_mask[v]
                new_X = X_mask & adj_mask[v]
                
                bron_kerbosch_bitmask(clique_mask | v_bit, clique_size + 1, new_P, new_X)
                
                P_mask &= ~v_bit
                X_mask |= v_bit
                
                if time_up():
                    return
        
        # Process vertices in reverse degeneracy order
        # For each vertex v (in reverse order), find cliques containing v
        # where all other vertices come after v in the ordering
        
        remaining_mask = 0
        for v in reversed(deg_order):
            if time_up():
                break
            
            P_mask = remaining_mask & adj_mask[v]
            X_mask = 0  # No excluded vertices needed in this formulation
            
            # Upper bound check
            p_count = bin(P_mask).count('1')
            if 1 + p_count <= len(best_clique):
                remaining_mask |= (1 << v)
                continue
            
            bron_kerbosch_bitmask(1 << v, 1, P_mask, X_mask)
            remaining_mask |= (1 << v)
    
    elif n <= 5000:
        # Use set-based Bron-Kerbosch with degeneracy ordering
        
        def degeneracy_ordering():
            deg = [len(adj[v]) for v in range(n)]
            removed = [False] * n
            order = []
            # Use bucket sort approach
            max_deg = max(deg) if deg else 0
            buckets = [set() for _ in range(max_deg + 1)]
            for v in range(n):
                buckets[deg[v]].add(v)
            
            current_min = 0
            for _ in range(n):
                while current_min <= max_deg and not buckets[current_min]:
                    current_min += 1
                if current_min > max_deg:
                    break
                v = buckets[current_min].pop()
                order.append(v)
                removed[v] = True
                for u in adj[v]:
                    if not removed[u]:
                        old_deg = deg[u]
                        buckets[old_deg].discard(u)
                        deg[u] -= 1
                        buckets[deg[u]].add(u)
                        if deg[u] < current_min:
                            current_min = deg[u]
            return order
        
        deg_order = degeneracy_ordering()
        vertex_pos = {v: i for i, v in enumerate(deg_order)}
        
        def bron_kerbosch_set(clique, P, X):
            nonlocal best_clique
            
            if not P and not X:
                if len(clique) > len(best_clique):
                    best_clique = list(clique)
                    if logger:
                        logger.log(len(best_clique))
                return
            
            if not P:
                return
            
            if len(clique) + len(P) <= len(best_clique):
                return
            
            if time_up():
                return
            
            # Choose pivot maximizing |P ∩ N(pivot)|
            pivot = max(P | X, key=lambda u: len(P & adj[u]))
            
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                clique.append(v)
                bron_kerbosch_set(clique, new_P, new_X)
                clique.pop()
                
                P = P - {v}
                X = X | {v}
                
                if time_up():
                    return
        
        # Convert adj to frozensets for faster operations - actually keep as sets
        adj_sets = [adj[v] for v in range(n)]
        
        remaining = set()
        for v in reversed(deg_order):
            if time_up():
                break
            
            P = remaining & adj_sets[v]
            
            if 1 + len(P) <= len(best_clique):
                remaining.add(v)
                continue
            
            bron_kerbosch_set([v], P, set())
            remaining.add(v)
    
    else:
        # For very large graphs, use more aggressive heuristics
        # Local search: try to improve the greedy solution
        
        def try_improve_clique(clique):
            nonlocal best_clique
            clique_set = set(clique)
            improved = True
            while improved and not time_up():
                improved = False
                # Try swapping: remove one vertex, add two
                for v in list(clique_set):
                    if time_up():
                        break
                    # Remove v, find candidates that connect to all remaining
                    remaining = clique_set - {v}
                    # Candidates must be neighbors of all in remaining
                    if not remaining:
                        continue
                    candidates = set(adj[next(iter(remaining))])
                    for u in remaining:
                        candidates &= adj[u]
                    candidates -= clique_set
                    
                    # Try to add two from candidates
                    cand_list = list(candidates)
                    found = False
                    for i in range(len(cand_list)):
                        if found or time_up():
                            break
                        for j in range(i + 1, len(cand_list)):
                            if cand_list[j] in adj[cand_list[i]]:
                                # Swap: remove v, add cand_list[i] and cand_list[j]
                                clique_set.discard(v)
                                clique_set.add(cand_list[i])
                                clique_set.add(cand_list[j])
                                improved = True
                                found = True
                                break
                
                if len(clique_set) > len(best_clique):
                    best_clique = list(clique_set)
                    if logger:
                        logger.log(len(best_clique))
            return list(clique_set)
        
        try_improve_clique(best_clique[:])

    # Verify the solution
    result_set = set(best_clique)
    valid = True
    for i, u in enumerate(best_clique):
        for j in range(i + 1, len(best_clique)):
            v = best_clique[j]
            if v not in adj[u]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: find largest valid sub-clique
        new_clique = [best_clique[0]]
        for v in best_clique[1:]:
            if all(v in adj[u] for u in new_clique):
                new_clique.append(v)
        best_clique = new_clique

    solution = {
        "objective_value": len(best_clique),
        "clique_vertices": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()