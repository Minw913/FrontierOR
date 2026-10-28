import json
import argparse
import time
from solution_logger import SolutionLogger


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_vertices']
    edges = instance['edges']

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    start_time = time.time()
    time_limit = args.time_limit

    # Initialize with greedy clique if available
    best_clique = list(instance.get('greedy_clique', []))
    best_size = len(best_clique)

    if logger and best_size > 0:
        logger.log_solution(best_size, {
            "objective_value": best_size,
            "clique_vertices": best_clique[:]
        })

    timed_out = [False]

    def greedy_color(P):
        """
        Greedy sequential coloring of the subgraph induced by P.
        Returns list of (vertex, color) sorted by color (non-decreasing).
        """
        color = {}
        for v in P:
            # Find colors used by already-colored neighbors
            neighbor_colors = set()
            for u in adj[v]:
                if u in color:
                    neighbor_colors.add(color[u])
            c = 1
            while c in neighbor_colors:
                c += 1
            color[v] = c

        result = [(v, color[v]) for v in P]
        result.sort(key=lambda x: x[1])
        return result

    def expand(C, P):
        """
        Branch and bound for maximum clique.
        C: current clique (list of vertices)
        P: candidate vertices (list)
        """
        nonlocal best_clique, best_size

        if timed_out[0]:
            return

        if time.time() - start_time > time_limit - 1.0:
            timed_out[0] = True
            return

        if not P:
            if len(C) > best_size:
                best_size = len(C)
                best_clique = C[:]
                if logger:
                    logger.log_solution(best_size, {
                        "objective_value": best_size,
                        "clique_vertices": best_clique[:]
                    })
            return

        # Simple upper bound check
        if len(C) + len(P) <= best_size:
            return

        # Greedy coloring for tighter upper bound
        colored = greedy_color(P)

        # Process vertices from highest color to lowest
        for i in range(len(colored) - 1, -1, -1):
            if timed_out[0]:
                return

            if time.time() - start_time > time_limit - 1.0:
                timed_out[0] = True
                return

            v, c = colored[i]

            # Pruning: even with all remaining vertices, can't beat best
            if len(C) + c <= best_size:
                break

            # New candidates: vertices at positions 0..i-1 that are neighbors of v
            new_P = [colored[j][0] for j in range(i) if colored[j][0] in adj[v]]

            C.append(v)

            if len(C) > best_size:
                best_size = len(C)
                best_clique = C[:]
                if logger:
                    logger.log_solution(best_size, {
                        "objective_value": best_size,
                        "clique_vertices": best_clique[:]
                    })

            if len(C) + len(new_P) > best_size:
                expand(C, new_P)

            C.pop()

    # Sort vertices by degree (descending) for better initial exploration
    all_vertices = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

    expand([], all_vertices)

    solution = {
        "objective_value": best_size,
        "clique_vertices": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    solve()