import json
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="Maximum Clique Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=3600)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    m = data['m']
    edges = data['edges']

    start_time = time.time()
    deadline = start_time + args.time_limit - 0.5

    # Build adjacency as bitsets (Python integers)
    adj = [0] * n
    degree = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degree[u] += 1
        degree[v] += 1

    best_clique = []
    best_size = 0

    def time_left():
        return deadline - time.time()

    def greedy_clique(candidates_bitset=None):
        """Greedy heuristic to find a clique."""
        if candidates_bitset is None:
            candidates_bitset = (1 << n) - 1

        clique = []
        P = candidates_bitset

        while P:
            # Pick vertex with highest degree in current subgraph
            best_v = -1
            best_d = -1
            temp = P
            while temp:
                v = (temp & -temp).bit_length() - 1
                d = bin(adj[v] & P).count('1')
                if d > best_d:
                    best_d = d
                    best_v = v
                temp &= temp - 1

            if best_v == -1:
                break

            clique.append(best_v)
            P = P & adj[best_v]

        return clique

    def greedy_clique_fast(candidates_bitset=None):
        """Faster greedy heuristic - picks vertex with max degree in original graph."""
        if candidates_bitset is None:
            candidates_bitset = (1 << n) - 1

        clique = []
        P = candidates_bitset

        # Get vertices sorted by degree
        verts = []
        temp = candidates_bitset
        while temp:
            v = (temp & -temp).bit_length() - 1
            verts.append(v)
            temp &= temp - 1
        verts.sort(key=lambda x: degree[x], reverse=True)

        for v in verts:
            if P & (1 << v):
                clique.append(v)
                P = P & adj[v]

        return clique

    def greedy_color_order(P_bitset):
        """Greedy coloring of subgraph induced by P_bitset.
        Returns (order, colors) sorted by color ascending."""
        verts = []
        temp = P_bitset
        while temp:
            v = (temp & -temp).bit_length() - 1
            verts.append(v)
            temp &= temp - 1

        if not verts:
            return [], []

        color = [0] * n
        num_colors = 0

        # Color vertices in order of decreasing degree for tighter bounds
        verts.sort(key=lambda x: degree[x], reverse=True)

        for v in verts:
            used_colors = 0  # bitmask of used colors
            temp = adj[v] & P_bitset
            while temp:
                u = (temp & -temp).bit_length() - 1
                if color[u] > 0:
                    used_colors |= (1 << color[u])
                temp &= temp - 1

            # Find smallest available color
            c = 1
            while used_colors & (1 << c):
                c += 1
            color[v] = c
            if c > num_colors:
                num_colors = c

        # Sort by color ascending
        order = sorted(verts, key=lambda x: color[x])
        colors = [color[v] for v in order]
        return order, colors

    def expand(clique, P_bitset):
        nonlocal best_clique, best_size

        if time_left() <= 0:
            return

        if P_bitset == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
                sol = {
                    "objective_value": best_size,
                    "clique_vertices": best_clique[:]
                }
                if logger:
                    logger.log_solution(best_size, sol)
            return

        # Quick check: if |P| + |clique| <= best_size, prune
        popcount_P = bin(P_bitset).count('1')
        if len(clique) + popcount_P <= best_size:
            return

        order, colors = greedy_color_order(P_bitset)

        if not order:
            return

        if len(clique) + colors[-1] <= best_size:
            return

        # Process in reverse order (highest color first)
        for i in range(len(order) - 1, -1, -1):
            if time_left() <= 0:
                return

            v = order[i]

            # Upper bound check
            if len(clique) + colors[i] <= best_size:
                break

            clique.append(v)
            new_P = P_bitset & adj[v]

            if new_P == 0:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                    sol = {
                        "objective_value": best_size,
                        "clique_vertices": best_clique[:]
                    }
                    if logger:
                        logger.log_solution(best_size, sol)
            else:
                # Quick upper bound check before recursion
                popcount_newP = bin(new_P).count('1')
                if len(clique) + popcount_newP > best_size:
                    expand(clique, new_P)

            clique.pop()

    # Step 1: Get initial solution via greedy heuristic
    if time_left() > 0:
        clique = greedy_clique()
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique[:]
            sol = {
                "objective_value": best_size,
                "clique_vertices": best_clique[:]
            }
            if logger:
                logger.log_solution(best_size, sol)

    # Step 2: Try greedy from each high-degree vertex
    if time_left() > 1:
        sorted_verts = sorted(range(n), key=lambda x: degree[x], reverse=True)
        for v in sorted_verts[:min(50, n)]:
            if time_left() <= 0.5:
                break
            clique = greedy_clique(adj[v] | (1 << v))
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
                sol = {
                    "objective_value": best_size,
                    "clique_vertices": best_clique[:]
                }
                if logger:
                    logger.log_solution(best_size, sol)

    # Step 3: Exact B&B if time permits and graph is manageable
    if time_left() > 1:
        # For very large graphs, limit the search
        if n <= 5000:
            all_vertices = (1 << n) - 1
            expand([], all_vertices)
        else:
            # For large graphs, try B&B on neighborhoods of high-degree vertices
            sorted_verts = sorted(range(n), key=lambda x: degree[x], reverse=True)
            for v in sorted_verts[:min(20, n)]:
                if time_left() <= 1:
                    break
                # Try to extend from vertex v
                P = adj[v]
                clique = [v]
                expand(clique, P)

    # Step 4: Local search improvement
    if time_left() > 0.5:
        improved = True
        while improved and time_left() > 0.5:
            improved = False
            # Try adding vertices
            clique_set = set(best_clique)
            clique_bits = 0
            for v in best_clique:
                clique_bits |= (1 << v)

            # Find vertices connected to all clique members
            if best_clique:
                common_neighbors = (1 << n) - 1
                for v in best_clique:
                    common_neighbors &= adj[v]
                # Remove clique members
                common_neighbors &= ~clique_bits

                temp = common_neighbors
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    best_clique.append(v)
                    best_size = len(best_clique)
                    sol = {
                        "objective_value": best_size,
                        "clique_vertices": best_clique[:]
                    }
                    if logger:
                        logger.log_solution(best_size, sol)
                    improved = True
                    break
                    temp &= temp - 1

    # Final output
    solution = {
        "objective_value": best_size,
        "clique_vertices": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

    if logger:
        logger.log_solution(best_size, solution)

if __name__ == "__main__":
    main()