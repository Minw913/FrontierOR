import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    graph = data['graph']
    num_vertices = graph['num_vertices']
    adjacency_list = graph['adjacency_list']

    # Build adjacency sets
    adj = [set() for _ in range(num_vertices)]
    for v_str, neighbors in adjacency_list.items():
        v = int(v_str)
        for u in neighbors:
            adj[v].add(u)

    start_time = time.time()
    time_limit = args.time_limit

    best_clique = []
    best_size = 0
    time_up = False

    def update_best(clique):
        nonlocal best_clique, best_size
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = list(clique)
            if logger:
                solution = {
                    "objective_value": best_size,
                    "clique_vertices": sorted(best_clique)
                }
                logger.log_solution(best_size, solution)

    def greedy_color(vertices):
        """Greedy sequential coloring. Returns (sorted_vertices, color_bounds)."""
        n = len(vertices)
        if n == 0:
            return [], []
        
        color = [0] * n
        for i in range(n):
            v = vertices[i]
            used_colors = set()
            for j in range(i):
                if vertices[j] in adj[v]:
                    used_colors.add(color[j])
            c = 1
            while c in used_colors:
                c += 1
            color[i] = c

        # Sort by color ascending
        indices = sorted(range(n), key=lambda i: color[i])
        sorted_v = [vertices[i] for i in indices]
        bounds = [color[i] for i in indices]
        return sorted_v, bounds

    def expand(clique, candidates):
        nonlocal best_size, time_up

        if time_up:
            return

        if time.time() - start_time > time_limit - 1.0:
            time_up = True
            return

        if len(clique) + len(candidates) <= best_size:
            return

        if not candidates:
            update_best(clique)
            return

        # Greedy coloring for upper bound
        sorted_cands, bounds = greedy_color(candidates)

        if len(clique) + bounds[-1] <= best_size:
            return

        for i in range(len(sorted_cands) - 1, -1, -1):
            if time_up:
                return
            if time.time() - start_time > time_limit - 1.0:
                time_up = True
                return

            if len(clique) + bounds[i] <= best_size:
                return

            v = sorted_cands[i]
            clique.append(v)

            new_candidates = [u for u in sorted_cands[:i] if u in adj[v]]

            if len(clique) + len(new_candidates) > best_size:
                expand(clique, new_candidates)

            clique.pop()

    # Greedy heuristic for initial solution
    def greedy_clique():
        remaining = set(range(num_vertices))
        clique = []
        while remaining:
            # Pick vertex with max degree within remaining
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.append(best_v)
            remaining = remaining & adj[best_v]
        return clique

    # Multiple greedy attempts with different orderings
    initial = greedy_clique()
    update_best(initial)

    # Try greedy with different starting vertices
    if num_vertices <= 5000 and time.time() - start_time < time_limit * 0.1:
        for start_v in range(min(num_vertices, 50)):
            remaining = adj[start_v].copy()
            clique = [start_v]
            while remaining:
                best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
                clique.append(best_v)
                remaining = remaining & adj[best_v]
            if len(clique) > best_size:
                update_best(clique)

    # Run Tomita's exact algorithm
    sys.setrecursionlimit(max(10000, num_vertices + 1000))

    # Order vertices by degree (ascending) for better pruning
    vertices = list(range(num_vertices))
    vertices.sort(key=lambda v: len(adj[v]))

    expand([], vertices)

    # If we still have time and didn't find optimal, try local search improvement
    if not time_up and time.time() - start_time < time_limit - 2.0:
        # Try expanding from each vertex in current best clique neighborhood
        pass

    solution = {
        "objective_value": best_size,
        "clique_vertices": sorted(best_clique)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()