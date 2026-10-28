import json
import argparse
import time
import sys

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)
    
    n = data["n"]
    edges = data["edges"]
    omega = data.get("omega", 0)

    # Build adjacency list using bitsets (Python integers)
    adj = [0] * n
    for u, v in edges:
        if u == v: 
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    best_clique = 0
    best_vertices = []

    start_time = time.time()
    
    # Use Python's built-in bit_count if available (Python 3.8+), else fallback
    try:
        (0).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    class OptimalFound(Exception):
        pass

    class TimeoutException(Exception):
        pass

    def get_solution_dict(vertices):
        return {
            "objective_value": len(vertices),
            "clique_vertices": sorted(vertices)
        }

    def update_best(R):
        nonlocal best_clique, best_vertices
        cnt = popcount(R)
        if cnt > best_clique:
            best_clique = cnt
            verts = []
            temp = R
            while temp:
                w = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                verts.append(w)
            best_vertices = verts
            if logger:
                logger.log_solution(best_clique, get_solution_dict(best_vertices))
            if best_clique == omega and omega > 0:
                raise OptimalFound

    def greedy():
        nonlocal best_clique, best_vertices
        order = sorted(range(n), key=lambda i: popcount(adj[i]), reverse=True)
        for v in order:
            if time.time() - start_time > args.time_limit * 0.2:
                break
            curr = 1 << v
            P = adj[v]
            while P:
                u = (P & -P).bit_length() - 1
                curr |= 1 << u
                P &= adj[u]
            update_best(curr)

    def bron_kerbosch(R, P, X):
        if time.time() - start_time > args.time_limit:
            raise TimeoutException
        
        if P == 0 and X == 0:
            update_best(R)
            return

        if popcount(R) + popcount(P) <= best_clique:
            return
        
        # Tomita's pivot selection: pick u in P | X that maximizes |P & adj[u]|
        u = -1
        max_inter = -1
        union = P | X
        temp_union = union
        while temp_union:
            w = (temp_union & -temp_union).bit_length() - 1
            temp_union &= temp_union - 1
            inter = popcount(adj[w] & P)
            if inter > max_inter:
                max_inter = inter
                u = w
        
        to_iter = P & (~adj[u] if u != -1 else 0)
        
        temp_to_iter = to_iter
        while temp_to_iter:
            v = (temp_to_iter & -temp_to_iter).bit_length() - 1
            temp_to_iter &= temp_to_iter - 1
            
            bron_kerbosch(R | (1 << v), P & adj[v], X & adj[v])
            
            P &= ~(1 << v)
            X |= 1 << v

    order = sorted(range(n), key=lambda i: popcount(adj[i]), reverse=True)
    
    P_all = 0
    for v in order:
        P_all |= 1 << v

    try:
        greedy()
        P = P_all
        X = 0
        for v in order:
            if time.time() - start_time > args.time_limit:
                break
            if popcount(1 << v) + popcount(P) <= best_clique:
                P &= ~(1 << v)
                X |= 1 << v
                continue
            bron_kerbosch(1 << v, P & adj[v], X & adj[v])
            P &= ~(1 << v)
            X |= 1 << v
    except OptimalFound:
        pass
    except TimeoutException:
        pass

    solution = {
        "objective_value": best_clique,
        "clique_vertices": sorted(best_vertices)
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)

if __name__ == "__main__":
    solve()