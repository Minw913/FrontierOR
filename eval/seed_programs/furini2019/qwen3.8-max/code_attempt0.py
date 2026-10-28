import json
import argparse
import time
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_vertices']
    edges = instance['edges']

    # Build adjacency structures
    adj_bits = [0] * n
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    start_time = time.time()
    time_limit = args.time_limit

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    best_clique = []

    def update_best(clique):
        nonlocal best_clique
        if len(clique) > len(best_clique):
            best_clique = clique[:]
            if logger:
                sol = {
                    "objective_value": len(best_clique),
                    "clique_vertices": sorted(best_clique)
                }
                logger.log_solution(len(best_clique), sol)

    def is_time_up():
        return time.time() - start_time > time_limit * 0.95

    def get_vertices_from_bits(bits):
        vertices = []
        while bits:
            lsb = bits & (-bits)
            v = lsb.bit_length() - 1
            vertices.append(v)
            bits ^= lsb
        return vertices

    def greedy_color(P_bits):
        """
        Greedy coloring of the subgraph induced by P_bits.
        Returns (ordered_vertices, color_of) where vertices are ordered
        by color class (ascending color).
        """
        vertices = get_vertices_from_bits(P_bits)
        if not vertices:
            return [], {}

        # Sort by degree within P for better coloring (fewer colors = tighter bound)
        vertices.sort(key=lambda v: bin(adj_bits[v] & P_bits).count('1'), reverse=True)

        color_of = {}
        color_classes = []

        for v in vertices:
            used_colors = set()
            for u in adj_list[v]:
                if u in color_of:
                    used_colors.add(color_of[u])
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            while len(color_classes) <= c:
                color_classes.append([])
            color_classes[c].append(v)

        ordered = []
        for cls in color_classes:
            ordered.extend(cls)

        return ordered, color_of

    def expand(P_bits, clique):
        """
        Branch and bound expansion using Tomita's algorithm with coloring bounds.
        P_bits: bitset of candidate vertices
        clique: current clique being built
        """
        if is_time_up():
            return

        if P_bits == 0:
            update_best(clique)
            return

        # Compute greedy coloring for upper bound
        ordered, color_of = greedy_color(P_bits)
        num_colors = max(color_of.values()) + 1 if color_of else 0

        # Prune if upper bound can't beat current best
        if len(clique) + num_colors <= len(best_clique):
            return

        # Process vertices from highest color to lowest
        for i in range(len(ordered) - 1, -1, -1):
            if is_time_up():
                return

            v = ordered[i]

            # Upper bound: clique + at most color_of[v]+1 more vertices
            if len(clique) + color_of[v] + 1 <= len(best_clique):
                break

            # Branch: include v in the clique
            new_P_bits = P_bits & adj_bits[v]
            expand(new_P_bits, clique + [v])

            # Remove v from candidates for subsequent iterations
            P_bits &= ~(1 << v)

    # --- Find initial greedy clique ---
    vertices_sorted = sorted(range(n), key=lambda v: len(adj_list[v]), reverse=True)
    greedy_clique = []
    candidates = (1 << n) - 1 if n > 0 else 0
    for v in vertices_sorted:
        if candidates & (1 << v):
            greedy_clique.append(v)
            candidates &= adj_bits[v]
    update_best(greedy_clique)

    # --- Run exact branch and bound ---
    if n > 0 and not is_time_up():
        expand((1 << n) - 1, [])

    # Write final solution
    solution = {
        "objective_value": len(best_clique),
        "clique_vertices": sorted(best_clique)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()