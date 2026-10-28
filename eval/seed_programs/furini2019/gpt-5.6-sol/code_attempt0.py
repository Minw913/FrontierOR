import argparse
import json
import sys
import time
from collections import deque

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_vertices"])

    if n == 0:
        solution = {"objective_value": 0, "clique_vertices": []}
        if logger:
            logger.log_solution(0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    # Relabel vertices so that larger internal identifiers tend to have
    # larger degree. This generally improves greedy coloring and branching.
    adjacency_input = data.get("adjacency_list")
    if adjacency_input is not None and len(adjacency_input) == n:
        input_degrees = [len(neighbors) for neighbors in adjacency_input]
    else:
        input_degrees = [0] * n
        for edge in data["edges"]:
            u, v = int(edge[0]), int(edge[1])
            if u != v:
                input_degrees[u] += 1
                input_degrees[v] += 1

    internal_to_original = sorted(range(n), key=lambda v: (input_degrees[v], v))
    original_to_internal = [0] * n
    for internal, original in enumerate(internal_to_original):
        original_to_internal[original] = internal

    adjacency = [0] * n
    for edge in data["edges"]:
        original_u, original_v = int(edge[0]), int(edge[1])
        if original_u == original_v:
            continue
        u = original_to_internal[original_u]
        v = original_to_internal[original_v]
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u

    degrees = [bits.bit_count() for bits in adjacency]
    all_vertices = (1 << n) - 1

    sys.setrecursionlimit(max(10000, n + 100))

    best_clique = []

    def make_solution(clique):
        vertices = sorted(internal_to_original[v] for v in clique)
        return {
            "objective_value": len(vertices),
            "clique_vertices": vertices,
        }

    def record(clique):
        nonlocal best_clique
        if len(clique) <= len(best_clique):
            return
        best_clique = list(clique)
        if logger:
            solution = make_solution(best_clique)
            logger.log_solution(len(best_clique), solution)

    # A singleton is always feasible.
    record([n - 1])

    def fixed_greedy(initial_candidates, prefix=None, take_high=True):
        clique = list(prefix) if prefix else []
        candidates = initial_candidates
        while candidates:
            if take_high:
                v = candidates.bit_length() - 1
            else:
                bit = candidates & -candidates
                v = bit.bit_length() - 1
            clique.append(v)
            candidates &= adjacency[v]
        record(clique)

    # Very inexpensive deterministic greedy solutions.
    fixed_greedy(all_vertices, take_high=True)
    fixed_greedy(all_vertices, take_high=False)

    high_degree_vertices = list(range(n - 1, max(-1, n - 65), -1))
    for seed in high_degree_vertices:
        fixed_greedy(adjacency[seed], prefix=[seed], take_high=True)

    # Spend a small fraction of the budget on stronger induced-degree
    # greedy solutions, leaving most time for exact branch-and-bound.
    heuristic_deadline = min(
        deadline,
        start_time + min(1.0, max(0.0, args.time_limit * 0.08)),
    )

    def dynamic_greedy(seed):
        clique = [seed]
        candidates = adjacency[seed]
        work = 0

        while candidates:
            if time.monotonic() >= heuristic_deadline:
                break

            candidate_count = candidates.bit_count()
            scan = candidates

            # For huge candidate sets, inspect the statically strongest
            # vertices first rather than spending excessive time scoring all.
            if candidate_count > 2500:
                sample = 0
                sampled = 0
                temp = scan
                while temp and sampled < 384:
                    v = temp.bit_length() - 1
                    sample |= 1 << v
                    temp &= ~(1 << v)
                    sampled += 1
                scan = sample

            best_v = -1
            best_score = -1
            temp = scan
            while temp:
                bit = temp & -temp
                v = bit.bit_length() - 1
                score = (adjacency[v] & candidates).bit_count()
                if score > best_score or (
                    score == best_score
                    and best_v >= 0
                    and degrees[v] > degrees[best_v]
                ):
                    best_score = score
                    best_v = v
                temp ^= bit
                work += 1
                if (work & 511) == 0 and time.monotonic() >= heuristic_deadline:
                    break

            if best_v < 0:
                break
            clique.append(best_v)
            candidates &= adjacency[best_v]

        record(clique)

    if time.monotonic() < heuristic_deadline:
        # Include the highest-degree starts and a spread across the ordering.
        seeds = high_degree_vertices[:32]
        if n > 32:
            step = max(1, n // 16)
            seeds.extend(range(n - 1, -1, -step))

        seen_seeds = set()
        for seed in seeds:
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            if time.monotonic() >= heuristic_deadline:
                break
            dynamic_greedy(seed)

    # Compute the |best|-core. Any clique that improves an incumbent of size
    # B must be contained in the B-core.
    def incumbent_core(threshold):
        if threshold <= 0:
            return all_vertices

        active = [True] * n
        current_degrees = degrees[:]
        queue = deque(v for v in range(n) if current_degrees[v] < threshold)

        while queue:
            v = queue.popleft()
            if not active[v] or current_degrees[v] >= threshold:
                continue

            active[v] = False
            neighbors = adjacency[v]
            while neighbors:
                bit = neighbors & -neighbors
                u = bit.bit_length() - 1
                neighbors ^= bit
                if active[u]:
                    current_degrees[u] -= 1
                    if current_degrees[u] == threshold - 1:
                        queue.append(u)

        result = 0
        for v, is_active in enumerate(active):
            if is_active:
                result |= 1 << v
        return result

    root_candidates = incumbent_core(len(best_clique))
    current_clique = []
    node_counter = 0
    color_work = 0

    def check_deadline():
        if time.monotonic() >= deadline:
            raise SearchTimeout

    def greedy_coloring(candidates):
        """
        Greedily partition candidates into independent color classes.

        The returned bound for each ordered vertex is a valid upper bound on
        the number of additional clique vertices among that vertex and all
        preceding vertices in the order.
        """
        nonlocal color_work

        ordered = []
        bounds = []
        uncolored = candidates
        color = 0

        while uncolored:
            check_deadline()
            color += 1
            available = uncolored

            while available:
                bit = available & -available
                v = bit.bit_length() - 1

                ordered.append(v)
                bounds.append(color)

                uncolored ^= bit
                available ^= bit
                available &= ~adjacency[v]

                color_work += 1
                if (color_work & 1023) == 0:
                    check_deadline()

        return ordered, bounds

    def expand(candidates):
        nonlocal node_counter

        node_counter += 1
        if (node_counter & 255) == 0:
            check_deadline()

        ordered, color_bounds = greedy_coloring(candidates)

        for i in range(len(ordered) - 1, -1, -1):
            # All remaining branches have no larger coloring bound.
            if len(current_clique) + color_bounds[i] <= len(best_clique):
                return

            v = ordered[i]
            bit = 1 << v

            # The vertex may already have disappeared from candidates only if
            # a higher-index branch removed it; normally it remains present.
            if not (candidates & bit):
                continue

            current_clique.append(v)
            record(current_clique)

            new_candidates = candidates & adjacency[v]
            if new_candidates and (
                len(current_clique) + new_candidates.bit_count() > len(best_clique)
            ):
                expand(new_candidates)

            current_clique.pop()
            candidates &= ~bit

    try:
        if root_candidates and time.monotonic() < deadline:
            expand(root_candidates)
    except SearchTimeout:
        pass

    final_solution = make_solution(best_clique)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()