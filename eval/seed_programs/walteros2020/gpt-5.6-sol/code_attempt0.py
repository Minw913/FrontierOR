import argparse
import json
import sys
import time
from collections import deque

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
    pass


class TargetReached(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="Maximum clique solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    edges = data["edges"]

    # Relabel vertices by nonincreasing degree. This generally improves greedy
    # coloring and branching while retaining a mapping to original indices.
    degrees = [0] * n
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        degrees[u] += 1
        degrees[v] += 1

    permutation = sorted(range(n), key=lambda v: (-degrees[v], v))
    inverse = [0] * n
    for new_v, old_v in enumerate(permutation):
        inverse[old_v] = new_v

    adjacency = [0] * n
    for edge in edges:
        old_u, old_v = int(edge[0]), int(edge[1])
        u, v = inverse[old_u], inverse[old_v]
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u

    try:
        sys.setrecursionlimit(max(2000, n + 100))
    except Exception:
        pass

    best_clique = []

    def make_solution(internal_clique):
        vertices = sorted(permutation[v] for v in internal_clique)
        return {
            "objective_value": len(vertices),
            "clique_vertices": vertices,
        }

    def record(clique):
        nonlocal best_clique
        if len(clique) <= len(best_clique):
            return
        best_clique = list(clique)
        solution = make_solution(best_clique)
        if logger:
            logger.log_solution(solution["objective_value"], solution)

    if n == 0:
        empty_solution = {
            "objective_value": 0,
            "clique_vertices": [],
        }
        if logger:
            logger.log_solution(0, empty_solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(empty_solution, f)
        return

    # A singleton is always feasible.
    record([0])

    known_omega = data.get("omega")
    target = None
    if known_omega is not None:
        try:
            value = int(known_omega)
            if 1 <= value <= n:
                target = value
        except (TypeError, ValueError):
            pass

    upper_bound = n
    if target is not None:
        upper_bound = min(upper_bound, target)

    if "degeneracy" in data:
        try:
            upper_bound = min(upper_bound, int(data["degeneracy"]) + 1)
        except (TypeError, ValueError):
            pass

    if "max_degree" in data:
        try:
            upper_bound = min(upper_bound, int(data["max_degree"]) + 1)
        except (TypeError, ValueError):
            pass

    def check_time():
        if time.monotonic() >= deadline:
            raise SearchTimeout

    def choose_greedy_vertex(candidates, sample_limit=80):
        """Choose a candidate with high degree inside the current candidate set."""
        first_bit = candidates & -candidates
        best_v = first_bit.bit_length() - 1
        best_score = -1

        remaining = candidates
        checked = 0
        while remaining and checked < sample_limit:
            bit = remaining & -remaining
            remaining ^= bit
            v = bit.bit_length() - 1
            score = (adjacency[v] & candidates).bit_count()
            if score > best_score:
                best_score = score
                best_v = v
            checked += 1

        return best_v

    def greedy_clique(initial_candidates, prefix=None):
        clique = list(prefix) if prefix else []
        candidates = initial_candidates
        steps = 0

        while candidates:
            if (steps & 31) == 0:
                check_time()
            v = choose_greedy_vertex(candidates)
            clique.append(v)
            candidates &= adjacency[v]
            steps += 1

        return clique

    # Spend a bounded fraction of the runtime obtaining a strong initial lower
    # bound. A good incumbent substantially reduces exact-search effort.
    try:
        now = time.monotonic()
        heuristic_budget = min(2.0, max(0.02, max(0, args.time_limit) * 0.15))
        heuristic_deadline = min(deadline, now + heuristic_budget)

        if now < heuristic_deadline:
            record(greedy_clique((1 << n) - 1))

        seed_count = min(n, 64)
        if seed_count > 0:
            seeds = list(range(min(seed_count, 32)))
            if n > 32:
                extra_count = seed_count - len(seeds)
                for i in range(extra_count):
                    # Deterministically distribute additional seeds.
                    seed = 32 + (i * max(1, n - 32)) // max(1, extra_count)
                    if seed < n and seed not in seeds:
                        seeds.append(seed)

            for seed in seeds:
                if time.monotonic() >= heuristic_deadline:
                    break
                clique = greedy_clique(adjacency[seed], [seed])
                record(clique)
                if len(best_clique) >= upper_bound:
                    break
    except SearchTimeout:
        pass

    # Remove vertices that cannot belong to a clique larger than the incumbent.
    # Any vertex in a clique of size best+1 must lie in the best-core.
    def root_core_filter(vertices, threshold):
        if threshold <= 0 or not vertices:
            return vertices

        active = vertices
        local_degree = [0] * n
        queue = deque()

        remaining = active
        counter = 0
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            v = bit.bit_length() - 1
            d = (adjacency[v] & active).bit_count()
            local_degree[v] = d
            if d < threshold:
                queue.append(v)
            counter += 1
            if (counter & 255) == 0:
                check_time()

        removed_count = 0
        while queue:
            v = queue.popleft()
            v_bit = 1 << v
            if not (active & v_bit) or local_degree[v] >= threshold:
                continue

            neighbors = adjacency[v] & active
            active ^= v_bit

            while neighbors:
                bit = neighbors & -neighbors
                neighbors ^= bit
                u = bit.bit_length() - 1
                local_degree[u] -= 1
                if local_degree[u] == threshold - 1:
                    queue.append(u)

            removed_count += 1
            if (removed_count & 255) == 0:
                check_time()

        return active

    node_count = 0
    color_work = 0
    current_clique = []

    def color_sort(candidates):
        """
        Greedily color the candidate subgraph.

        Returns vertices ordered by nondecreasing color and a corresponding
        upper bound. Vertices sharing a color form an independent set.
        """
        nonlocal color_work

        order = []
        color_bounds = []
        uncolored = candidates
        color = 0

        while uncolored:
            color += 1
            available = uncolored

            while available:
                bit = available & -available
                available ^= bit
                v = bit.bit_length() - 1

                uncolored ^= bit
                order.append(v)
                color_bounds.append(color)

                available &= ~adjacency[v]

                color_work += 1
                if (color_work & 511) == 0:
                    check_time()

            check_time()

        return order, color_bounds

    def expand(candidates):
        nonlocal node_count

        node_count += 1
        if (node_count & 63) == 0:
            check_time()

        current_size = len(current_clique)
        if current_size + candidates.bit_count() <= len(best_clique):
            return

        order, bounds = color_sort(candidates)

        for i in range(len(order) - 1, -1, -1):
            if current_size + bounds[i] <= len(best_clique):
                return

            v = order[i]
            v_bit = 1 << v

            # The candidate set is progressively reduced as branches are
            # processed, preventing duplicate clique enumeration.
            if not (candidates & v_bit):
                continue

            current_clique.append(v)
            record(current_clique)

            if len(best_clique) >= upper_bound:
                raise TargetReached

            child_candidates = candidates & adjacency[v]
            if (
                child_candidates
                and len(current_clique) + child_candidates.bit_count()
                > len(best_clique)
            ):
                expand(child_candidates)

            current_clique.pop()
            candidates ^= v_bit

    try:
        if len(best_clique) < upper_bound:
            check_time()
            root_candidates = (1 << n) - 1
            root_candidates = root_core_filter(
                root_candidates, len(best_clique)
            )

            if root_candidates:
                expand(root_candidates)
    except (SearchTimeout, TargetReached):
        pass

    final_solution = make_solution(best_clique)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()