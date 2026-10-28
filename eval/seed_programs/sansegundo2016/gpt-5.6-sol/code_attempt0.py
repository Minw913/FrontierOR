import argparse
import json
import os
import sys
import time
from collections import deque

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
    pass


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data["graph"]
    n = int(graph["num_vertices"])
    raw_adjacency = graph.get("adjacency_list", {})

    labels = set()
    parsed = {}

    for raw_vertex, raw_neighbors in raw_adjacency.items():
        vertex = int(raw_vertex)
        neighbors = [int(x) for x in raw_neighbors]
        parsed[vertex] = neighbors
        labels.add(vertex)
        labels.update(neighbors)

    # The schema normally lists every vertex. Handle omitted isolated vertices
    # by inferring whether the instance uses zero-based or one-based labels.
    if len(labels) < n:
        zero_based = set(range(n))
        one_based = set(range(1, n + 1))

        if labels.issubset(zero_based):
            labels.update(zero_based)
        elif labels.issubset(one_based):
            labels.update(one_based)
        elif not labels:
            labels.update(zero_based)
        else:
            candidate = 0
            while len(labels) < n:
                if candidate not in labels:
                    labels.add(candidate)
                candidate += 1

    labels = sorted(labels)

    if len(labels) > n:
        raise ValueError(
            f"Instance declares {n} vertices but adjacency data contains "
            f"{len(labels)} distinct vertex labels"
        )

    label_to_index = {label: i for i, label in enumerate(labels)}
    adjacency = [0] * len(labels)

    # Symmetrize defensively. Valid instances are already undirected.
    for vertex, neighbors in parsed.items():
        if vertex not in label_to_index:
            continue
        u = label_to_index[vertex]
        for neighbor in neighbors:
            if neighbor == vertex or neighbor not in label_to_index:
                continue
            v = label_to_index[neighbor]
            adjacency[u] |= 1 << v
            adjacency[v] |= 1 << u

    return labels, adjacency


def iter_bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def make_solution(clique_indices, labels):
    vertices = sorted(labels[i] for i in clique_indices)
    return {
        "objective_value": len(vertices),
        "clique_vertices": vertices,
    }


def solve(labels, adjacency, deadline, logger):
    n = len(labels)
    best_clique = []

    def record(clique):
        nonlocal best_clique
        if len(clique) <= len(best_clique):
            return
        best_clique = list(clique)
        solution = make_solution(best_clique, labels)
        if logger:
            logger.log_solution(solution["objective_value"], solution)

    if n == 0:
        solution = make_solution([], labels)
        if logger:
            logger.log_solution(0, solution)
        return []

    # A singleton is always feasible and ensures a useful result even under
    # an extremely small time limit.
    record([0])

    degrees = [mask.bit_count() for mask in adjacency]
    degree_order = sorted(
        range(n), key=lambda v: (degrees[v], -labels[v]), reverse=True
    )

    def greedy_scan(order, forced_vertex=None):
        if forced_vertex is None:
            clique = []
            clique_mask = 0
        else:
            clique = [forced_vertex]
            clique_mask = 1 << forced_vertex

        for position, vertex in enumerate(order):
            if (position & 255) == 0 and time.monotonic() >= deadline:
                break
            bit = 1 << vertex
            if clique_mask & bit:
                continue
            if adjacency[vertex] & clique_mask == clique_mask:
                clique.append(vertex)
                clique_mask |= bit

        record(clique)

    if time.monotonic() < deadline:
        greedy_scan(degree_order)

    # Tie-breaking in the opposite label direction can produce a substantially
    # different clique while preserving the useful degree ordering.
    if time.monotonic() < deadline:
        alternate_order = sorted(
            range(n), key=lambda v: (degrees[v], labels[v]), reverse=True
        )
        greedy_scan(alternate_order)

    # Try several high-degree and spread-out starting vertices.
    if time.monotonic() < deadline:
        starts = []
        seen = set()

        for v in degree_order[: min(32, n)]:
            if v not in seen:
                seen.add(v)
                starts.append(v)

        spread_count = min(16, n)
        if spread_count > 1:
            for i in range(spread_count):
                pos = i * (n - 1) // (spread_count - 1)
                v = degree_order[pos]
                if v not in seen:
                    seen.add(v)
                    starts.append(v)

        for start_vertex in starts:
            if time.monotonic() >= deadline:
                break
            greedy_scan(degree_order, start_vertex)

    # One dynamic greedy construction chooses a vertex with high degree in the
    # current candidate subgraph at every step.
    if n <= 1800 and time.monotonic() < deadline:
        candidates = (1 << n) - 1
        clique = []

        while candidates and time.monotonic() < deadline:
            chosen = -1
            chosen_score = (-1, -1)
            scan = candidates

            while scan:
                bit = scan & -scan
                vertex = bit.bit_length() - 1
                scan ^= bit
                score = (
                    (adjacency[vertex] & candidates).bit_count(),
                    degrees[vertex],
                )
                if score > chosen_score:
                    chosen = vertex
                    chosen_score = score

            if chosen < 0:
                break
            clique.append(chosen)
            candidates &= adjacency[chosen]

        record(clique)

    if time.monotonic() >= deadline:
        return best_clique

    # Any clique improving an incumbent of size B consists solely of vertices
    # in the B-core. Peel vertices that cannot participate in such a clique.
    threshold = len(best_clique)
    active = (1 << n) - 1

    if threshold > 0:
        active_degrees = [(adjacency[v] & active).bit_count() for v in range(n)]
        queue = deque(v for v in range(n) if active_degrees[v] < threshold)
        queued = [False] * n
        for v in queue:
            queued[v] = True

        peel_steps = 0
        while queue:
            if (peel_steps & 1023) == 0 and time.monotonic() >= deadline:
                return best_clique
            peel_steps += 1

            v = queue.popleft()
            bit_v = 1 << v
            if not (active & bit_v):
                continue

            neighbors = adjacency[v] & active
            active ^= bit_v

            while neighbors:
                bit_u = neighbors & -neighbors
                neighbors ^= bit_u
                u = bit_u.bit_length() - 1
                active_degrees[u] -= 1
                if active_degrees[u] < threshold and not queued[u]:
                    queued[u] = True
                    queue.append(u)

    if not active or time.monotonic() >= deadline:
        return best_clique

    active_vertices = list(iter_bits(active))
    active_set = active

    # Low-degree vertices receive low search indices. The coloring routine is
    # traversed in reverse, so high-degree vertices are generally branched on
    # first.
    active_vertices.sort(
        key=lambda v: (
            (adjacency[v] & active_set).bit_count(),
            degrees[v],
            -labels[v],
        )
    )

    search_size = len(active_vertices)
    old_to_new = {old: new for new, old in enumerate(active_vertices)}
    search_adjacency = [0] * search_size

    for new_v, old_v in enumerate(active_vertices):
        neighbors = adjacency[old_v] & active_set
        converted = 0
        while neighbors:
            bit = neighbors & -neighbors
            neighbors ^= bit
            old_u = bit.bit_length() - 1
            converted |= 1 << old_to_new[old_u]
        search_adjacency[new_v] = converted

    sys.setrecursionlimit(max(2000, search_size + 200))

    nodes = 0
    color_calls = 0

    def check_time(force=False):
        nonlocal nodes
        if force or (nodes & 1023) == 0:
            if time.monotonic() >= deadline:
                raise SearchTimeout

    def color_sort(candidate_mask):
        """Greedy coloring providing an upper bound for each ordered prefix."""
        nonlocal color_calls
        color_calls += 1

        order = []
        bounds = []
        uncolored = candidate_mask
        color = 0

        while uncolored:
            if (color_calls & 31) == 0 and time.monotonic() >= deadline:
                raise SearchTimeout

            color += 1
            available = uncolored

            # Build one independent color class.
            while available:
                bit = available & -available
                vertex = bit.bit_length() - 1

                order.append(vertex)
                bounds.append(color)

                uncolored ^= bit
                available ^= bit
                available &= uncolored
                available &= ~search_adjacency[vertex]

        return order, bounds

    search_clique = []

    def expand(candidate_mask):
        nonlocal nodes
        nodes += 1
        check_time()

        order, color_bounds = color_sort(candidate_mask)

        for i in range(len(order) - 1, -1, -1):
            if len(search_clique) + color_bounds[i] <= len(best_clique):
                return

            vertex = order[i]
            bit = 1 << vertex

            # Earlier branches may already have removed this vertex.
            if not (candidate_mask & bit):
                continue

            search_clique.append(vertex)

            if len(search_clique) > len(best_clique):
                original_clique = [active_vertices[x] for x in search_clique]
                record(original_clique)

            next_candidates = candidate_mask & search_adjacency[vertex]
            if next_candidates:
                expand(next_candidates)

            search_clique.pop()
            candidate_mask ^= bit

            if (nodes & 1023) == 0:
                check_time()

    try:
        expand((1 << search_size) - 1)
    except SearchTimeout:
        pass

    return best_clique


def write_solution(path, solution):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))
        f.write("\n")
    os.replace(temporary_path, path)


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

    labels, adjacency = read_instance(args.instance_path)
    best_clique = solve(labels, adjacency, deadline, logger)
    solution = make_solution(best_clique, labels)
    write_solution(args.solution_path, solution)


if __name__ == "__main__":
    main()