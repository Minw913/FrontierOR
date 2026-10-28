import argparse
import json
import os
import sys
import time

from solution_logger import SolutionLogger


class TimeLimitReached(Exception):
    pass


def iter_bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def main():
    parser = argparse.ArgumentParser(description="Maximum clique solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["num_vertices"])
    edges = instance.get("edges", [])

    # Build the graph in the original vertex numbering.
    original_adjacency = [0] * n
    valid_edges = []
    for edge in edges:
        if len(edge) < 2:
            continue
        u, v = int(edge[0]), int(edge[1])
        if 0 <= u < n and 0 <= v < n and u != v:
            bit_u = 1 << u
            bit_v = 1 << v
            if not (original_adjacency[u] & bit_v):
                original_adjacency[u] |= bit_v
                original_adjacency[v] |= bit_u
                valid_edges.append((u, v))

    # Fall back to the adjacency list if an input omits the edge array.
    if not edges and instance.get("adjacency_list"):
        for key, neighbors in instance["adjacency_list"].items():
            u = int(key)
            if not (0 <= u < n):
                continue
            for raw_v in neighbors:
                v = int(raw_v)
                if 0 <= v < n and u != v:
                    original_adjacency[u] |= 1 << v
                    original_adjacency[v] |= 1 << u

        valid_edges = []
        for u in range(n):
            later = original_adjacency[u] & ~((1 << (u + 1)) - 1)
            for v in iter_bits(later):
                valid_edges.append((u, v))

    best_original = []

    def is_valid_original_clique(vertices):
        seen = 0
        for raw_v in vertices:
            v = int(raw_v)
            if not (0 <= v < n):
                return False
            bit = 1 << v
            if seen & bit:
                return False
            if (original_adjacency[v] & seen) != seen:
                return False
            seen |= bit
        return True

    def publish_original(vertices):
        nonlocal best_original
        candidate = sorted(int(v) for v in vertices)
        if len(candidate) <= len(best_original):
            return
        best_original = candidate
        solution = {
            "objective_value": float(len(best_original)),
            "clique_vertices": best_original.copy(),
        }
        if logger:
            logger.log_solution(solution["objective_value"], solution)

    if n == 0:
        solution = {"objective_value": 0.0, "clique_vertices": []}
        if logger:
            logger.log_solution(0.0, solution)
        output_dir = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    # Always establish a feasible incumbent.
    publish_original([0])

    supplied_clique = instance.get("greedy_clique", [])
    if supplied_clique and is_valid_original_clique(supplied_clique):
        publish_original(supplied_clique)

    # Relabel vertices by nondecreasing degree. This generally makes the
    # coloring-based branch order considerably more effective.
    original_degrees = [mask.bit_count() for mask in original_adjacency]
    permutation = sorted(range(n), key=lambda v: (original_degrees[v], v))
    inverse = [0] * n
    for internal, original in enumerate(permutation):
        inverse[original] = internal

    adjacency = [0] * n
    for u, v in valid_edges:
        iu, iv = inverse[u], inverse[v]
        adjacency[iu] |= 1 << iv
        adjacency[iv] |= 1 << iu

    # In case the fallback adjacency list was used and valid_edges was empty.
    if not valid_edges:
        for original_u in range(n):
            iu = inverse[original_u]
            mask = original_adjacency[original_u]
            converted = 0
            for original_v in iter_bits(mask):
                converted |= 1 << inverse[original_v]
            adjacency[iu] = converted

    degrees = [mask.bit_count() for mask in adjacency]
    best_internal = [inverse[v] for v in best_original]

    def publish_internal(vertices):
        nonlocal best_internal
        if len(vertices) <= len(best_internal):
            return
        original_vertices = [permutation[v] for v in vertices]
        # Search-generated cliques are valid by construction.
        best_internal = list(vertices)
        publish_original(original_vertices)

    all_vertices = (1 << n) - 1

    # Fast greedy completions. Since internal labels are degree ordered,
    # choosing the highest set bit is a cheap high-degree heuristic.
    def static_greedy(prefix, candidates, choose_high=True):
        clique = list(prefix)
        p = candidates
        while p:
            if time.monotonic() >= heuristic_deadline:
                raise TimeLimitReached
            if choose_high:
                v = p.bit_length() - 1
            else:
                bit = p & -p
                v = bit.bit_length() - 1
            clique.append(v)
            p &= adjacency[v]
        publish_internal(clique)

    now = time.monotonic()
    heuristic_allowance = min(2.0, max(0.05, max(0, args.time_limit) * 0.08))
    heuristic_deadline = min(deadline, now + heuristic_allowance)

    try:
        if time.monotonic() < heuristic_deadline:
            static_greedy([], all_vertices, True)
        if time.monotonic() < heuristic_deadline:
            static_greedy([], all_vertices, False)

        # Try forced high-degree starting vertices. These runs are very cheap
        # because each completion only performs bit intersections.
        max_starts = min(n, 512)
        for offset in range(max_starts):
            if time.monotonic() >= heuristic_deadline:
                break
            v = n - 1 - offset
            static_greedy([v], adjacency[v], True)

        # One more expensive adaptive greedy pass for moderate-sized graphs.
        if n <= 4000 and time.monotonic() < heuristic_deadline:
            clique = []
            p = all_vertices
            while p:
                if time.monotonic() >= heuristic_deadline:
                    break
                scan = p
                selected = -1
                selected_score = -1
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    scan ^= bit
                    score = (adjacency[v] & p).bit_count()
                    if score > selected_score or (
                        score == selected_score
                        and selected >= 0
                        and degrees[v] > degrees[selected]
                    ):
                        selected = v
                        selected_score = score
                if selected < 0:
                    break
                clique.append(selected)
                p &= adjacency[selected]
            publish_internal(clique)
    except TimeLimitReached:
        pass

    # Exact branch-and-bound using greedy sequential coloring. If interrupted,
    # best_internal remains a valid incumbent.
    current_clique = []
    operation_counter = 0

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeLimitReached

    def color_sort(candidate_mask):
        """
        Greedily color the candidate subgraph.

        Vertices in each color class form an independent set. The returned
        color number is therefore an upper bound for prefixes of the order.
        """
        nonlocal operation_counter
        ordered = []
        color_bounds = []
        uncolored = candidate_mask
        color = 0

        while uncolored:
            color += 1
            available = uncolored
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                ordered.append(v)
                color_bounds.append(color)

                uncolored ^= bit
                available ^= bit
                available &= ~adjacency[v]

                operation_counter += 1
                if (operation_counter & 255) == 0:
                    check_time()

        return ordered, color_bounds

    def expand(candidate_mask):
        nonlocal operation_counter

        operation_counter += 1
        if (operation_counter & 63) == 0:
            check_time()

        ordered, bounds = color_sort(candidate_mask)

        for i in range(len(ordered) - 1, -1, -1):
            # bounds[i] colors suffice for every remaining vertex 0..i.
            if len(current_clique) + bounds[i] <= len(best_internal):
                return

            v = ordered[i]
            bit = 1 << v
            current_clique.append(v)

            next_candidates = candidate_mask & adjacency[v]
            if not next_candidates:
                publish_internal(current_clique)
            elif len(current_clique) + next_candidates.bit_count() > len(best_internal):
                expand(next_candidates)

            current_clique.pop()
            candidate_mask &= ~bit

    try:
        check_time()

        # To improve a clique of size k, every participating vertex must have
        # global degree at least k.
        threshold = len(best_internal)
        root_candidates = 0
        for v in range(n):
            if degrees[v] >= threshold:
                root_candidates |= 1 << v

        if root_candidates and root_candidates.bit_count() > len(best_internal):
            expand(root_candidates)
    except TimeLimitReached:
        pass

    final_solution = {
        "objective_value": float(len(best_original)),
        "clique_vertices": sorted(best_original),
    }

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    main()