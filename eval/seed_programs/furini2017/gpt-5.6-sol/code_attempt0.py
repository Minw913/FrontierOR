import argparse
import heapq
import json
import os
import random
import sys
import time

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
    pass


def normalize_coloring(coloring):
    if not coloring:
        return [], 0
    remap = {}
    normalized = []
    next_color = 0
    for color in coloring:
        if color not in remap:
            remap[color] = next_color
            next_color += 1
        normalized.append(remap[color])
    return normalized, next_color


def make_solution(coloring):
    coloring, objective = normalize_coloring(coloring)
    color_classes = {str(c): [] for c in range(objective)}
    coloring_dict = {}
    for vertex, color in enumerate(coloring):
        coloring_dict[str(vertex)] = color
        color_classes[str(color)].append(vertex)
    return {
        "objective_value": objective,
        "coloring": coloring_dict,
        "color_classes": color_classes,
    }


def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data["graph"]
    n = int(graph["n"])
    adjacency = [set() for _ in range(n)]

    if "edges" in graph:
        for edge in graph["edges"]:
            u, v = int(edge[0]), int(edge[1])
            if u == v:
                continue
            if 0 <= u < n and 0 <= v < n:
                adjacency[u].add(v)
                adjacency[v].add(u)
    else:
        for u, neighbors in enumerate(graph["adjacency_list"]):
            for raw_v in neighbors:
                v = int(raw_v)
                if u != v and 0 <= v < n:
                    adjacency[u].add(v)
                    adjacency[v].add(u)

    adjacency_list = [sorted(neighbors) for neighbors in adjacency]
    adjacency_bits = []
    for neighbors in adjacency_list:
        bits = 0
        for v in neighbors:
            bits |= 1 << v
        adjacency_bits.append(bits)

    return n, adjacency_list, adjacency_bits


def bipartite_coloring(adjacency_list):
    n = len(adjacency_list)
    colors = [-1] * n
    has_edge = False

    for start in range(n):
        if colors[start] != -1:
            continue
        colors[start] = 0
        queue = [start]
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adjacency_list[u]:
                has_edge = True
                if colors[v] == -1:
                    colors[v] = colors[u] ^ 1
                    queue.append(v)
                elif colors[v] == colors[u]:
                    return None

    if n == 0:
        return []
    if not has_edge:
        return [0] * n
    return colors


def greedy_dsatur(adjacency_list, deadline, random_values=None):
    n = len(adjacency_list)
    if n == 0:
        return []

    degrees = [len(neighbors) for neighbors in adjacency_list]
    colors = [-1] * n
    saturation_masks = [0] * n
    versions = [0] * n
    heap = []

    if random_values is None:
        random_values = [0.0] * n
        deterministic = True
    else:
        deterministic = False

    def push(v):
        tie = v if deterministic else random_values[v]
        heapq.heappush(
            heap,
            (-saturation_masks[v].bit_count(), -degrees[v], tie, versions[v], v),
        )

    for v in range(n):
        push(v)

    colored_count = 0
    max_color = -1

    while colored_count < n:
        if colored_count % 128 == 0 and time.perf_counter() >= deadline:
            # Complete immediately with fresh private colors.
            next_color = max_color + 1
            for v in range(n):
                if colors[v] == -1:
                    colors[v] = next_color
                    next_color += 1
            return colors

        while heap:
            neg_sat, neg_degree, tie, version, v = heapq.heappop(heap)
            if colors[v] == -1 and version == versions[v]:
                break
        else:
            break

        occupied = saturation_masks[v]
        color = 0
        while occupied & (1 << color):
            color += 1

        colors[v] = color
        max_color = max(max_color, color)
        colored_count += 1
        bit = 1 << color

        for w in adjacency_list[v]:
            if colors[w] == -1 and not (saturation_masks[w] & bit):
                saturation_masks[w] |= bit
                versions[w] += 1
                push(w)

    return colors


def heuristic_clique(adjacency_bits, degrees, deadline, rng):
    n = len(adjacency_bits)
    if n == 0:
        return []

    best = [max(range(n), key=lambda v: degrees[v])]
    base_order = sorted(range(n), key=lambda v: (-degrees[v], v))
    orders = [base_order, list(reversed(base_order))]

    extra_orders = min(6, max(0, n // 20 + 1))
    for _ in range(extra_orders):
        order = list(range(n))
        rng.shuffle(order)
        order.sort(key=lambda v: -degrees[v] + rng.random() * 2.0)
        orders.append(order)

    for order in orders:
        if time.perf_counter() >= deadline:
            break

        clique = []
        candidates = (1 << n) - 1
        for v in order:
            if candidates & (1 << v):
                clique.append(v)
                candidates &= adjacency_bits[v]
        if len(clique) > len(best):
            best = clique

        for start in order[:min(30, n)]:
            if time.perf_counter() >= deadline:
                break
            clique = [start]
            candidates = adjacency_bits[start]
            while candidates:
                # Static-degree selection is cheap and generally effective.
                chosen = -1
                chosen_score = -1
                scan = candidates
                while scan:
                    lsb = scan & -scan
                    v = lsb.bit_length() - 1
                    score = degrees[v]
                    if score > chosen_score:
                        chosen = v
                        chosen_score = score
                    scan ^= lsb
                clique.append(chosen)
                candidates &= adjacency_bits[chosen]
            if len(clique) > len(best):
                best = clique

    return best


def improve_clique_exact(adjacency_bits, initial_clique, deadline):
    """Maximum-clique branch and bound, stopped safely at its deadline."""
    n = len(adjacency_bits)
    best = list(initial_clique)
    calls = 0

    def color_sort(vertices):
        order = []
        bounds = []
        remaining = vertices
        color_number = 0

        while remaining:
            color_number += 1
            available = remaining
            while available:
                lsb = available & -available
                v = lsb.bit_length() - 1
                order.append(v)
                bounds.append(color_number)
                remaining ^= lsb
                available ^= lsb
                available &= ~adjacency_bits[v]
        return order, bounds

    def expand(candidates, chosen):
        nonlocal best, calls
        calls += 1
        if calls & 255 == 0 and time.perf_counter() >= deadline:
            raise SearchTimeout

        if not candidates:
            if len(chosen) > len(best):
                best = list(chosen)
            return

        order, bounds = color_sort(candidates)
        for i in range(len(order) - 1, -1, -1):
            if len(chosen) + bounds[i] <= len(best):
                return

            v = order[i]
            bit = 1 << v
            if candidates & bit:
                chosen.append(v)
                new_candidates = candidates & adjacency_bits[v]
                if new_candidates:
                    expand(new_candidates, chosen)
                elif len(chosen) > len(best):
                    best = list(chosen)
                chosen.pop()
                candidates &= ~bit

    try:
        expand((1 << n) - 1, [])
    except SearchTimeout:
        pass

    return best


def exact_dsatur_search(
    adjacency_list,
    adjacency_bits,
    clique,
    initial_coloring,
    deadline,
    on_improvement,
):
    n = len(adjacency_list)
    degrees = [len(neighbors) for neighbors in adjacency_list]
    best_coloring = list(initial_coloring)
    best_k = len(set(initial_coloring)) if n else 0

    if n == 0 or best_k <= len(clique):
        return best_coloring, best_k

    colors = [-1] * n
    neighbor_color_masks = [0] * n
    neighbor_color_counts = [dict() for _ in range(n)]
    class_bits = [0] * max(best_k, n, 1)
    uncolored = (1 << n) - 1
    nodes = 0

    def assign(v, color):
        nonlocal uncolored
        colors[v] = color
        class_bits[color] |= 1 << v
        uncolored &= ~(1 << v)

        bit = 1 << color
        for w in adjacency_list[v]:
            counts = neighbor_color_counts[w]
            old = counts.get(color, 0)
            counts[color] = old + 1
            if old == 0:
                neighbor_color_masks[w] |= bit

    def unassign(v, color):
        nonlocal uncolored
        colors[v] = -1
        class_bits[color] &= ~(1 << v)
        uncolored |= 1 << v

        bit = 1 << color
        for w in adjacency_list[v]:
            counts = neighbor_color_counts[w]
            old = counts[color]
            if old == 1:
                del counts[color]
                neighbor_color_masks[w] &= ~bit
            else:
                counts[color] = old - 1

    # Fixing the colors of a clique is valid by color-label symmetry.
    for color, vertex in enumerate(clique):
        assign(vertex, color)

    initial_used = len(clique)

    def search(used_colors):
        nonlocal best_coloring, best_k, nodes
        nodes += 1
        if nodes & 255 == 0 and time.perf_counter() >= deadline:
            raise SearchTimeout

        if uncolored == 0:
            if used_colors < best_k:
                best_k = used_colors
                best_coloring = list(colors)
                on_improvement(best_coloring)
            return

        if used_colors >= best_k:
            return

        selected = -1
        selected_key = None
        max_saturation = 0

        scan = uncolored
        while scan:
            lsb = scan & -scan
            v = lsb.bit_length() - 1
            saturation = neighbor_color_masks[v].bit_count()
            if saturation > max_saturation:
                max_saturation = saturation
            residual_degree = (adjacency_bits[v] & uncolored).bit_count()
            key = (saturation, residual_degree, degrees[v], -v)
            if selected_key is None or key > selected_key:
                selected_key = key
                selected = v
            scan ^= lsb

        # The selected vertex must differ from every distinct color visible
        # in its neighborhood.
        if max(used_colors, max_saturation + 1) >= best_k:
            return

        forbidden = neighbor_color_masks[selected]
        existing_limit = min(used_colors, best_k - 1)

        # Prefer larger color classes, which generally creates tighter nodes.
        candidates = [
            c for c in range(existing_limit) if not (forbidden & (1 << c))
        ]
        candidates.sort(key=lambda c: class_bits[c].bit_count(), reverse=True)

        for color in candidates:
            assign(selected, color)
            search(used_colors)
            unassign(selected, color)

            if time.perf_counter() >= deadline:
                raise SearchTimeout

        # Introduce at most one new color; all unused colors are symmetric.
        if used_colors < best_k - 1:
            assign(selected, used_colors)
            search(used_colors + 1)
            unassign(selected, used_colors)

    try:
        search(initial_used)
    except SearchTimeout:
        pass

    return best_coloring, best_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    )

    start_time = time.perf_counter()
    deadline = start_time + max(0, args.time_limit)

    n, adjacency_list, adjacency_bits = load_graph(args.instance_path)
    sys.setrecursionlimit(max(2000, n * 3 + 100))

    best_coloring = list(range(n))
    best_solution = make_solution(best_coloring)
    best_k = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_k, best_solution)

    def accept_coloring(candidate):
        nonlocal best_coloring, best_solution, best_k
        candidate, candidate_k = normalize_coloring(candidate)
        if candidate_k < best_k:
            best_coloring = candidate
            best_k = candidate_k
            best_solution = make_solution(candidate)
            if logger:
                logger.log_solution(best_k, best_solution)
            return True
        return False

    if n == 0:
        pass
    elif time.perf_counter() < deadline:
        # This recognizes all instances with chromatic number at most two.
        bipartite = bipartite_coloring(adjacency_list)
        if bipartite is not None:
            accept_coloring(bipartite)
        else:
            rng = random.Random(0)

            # Obtain a strong feasible upper bound.
            deterministic = greedy_dsatur(adjacency_list, deadline)
            accept_coloring(deterministic)

            heuristic_budget = min(
                deadline,
                start_time + min(2.0, max(0.10, args.time_limit * 0.15)),
            )
            trial = 0
            while trial < 8 and time.perf_counter() < heuristic_budget:
                random_values = [rng.random() for _ in range(n)]
                candidate = greedy_dsatur(
                    adjacency_list, heuristic_budget, random_values
                )
                accept_coloring(candidate)
                trial += 1

            if best_k > 1 and time.perf_counter() < deadline:
                degrees = [len(neighbors) for neighbors in adjacency_list]
                clique_budget = min(
                    deadline,
                    time.perf_counter()
                    + min(2.0, max(0.05, args.time_limit * 0.10)),
                )
                clique = heuristic_clique(
                    adjacency_bits, degrees, clique_budget, rng
                )

                if time.perf_counter() < clique_budget:
                    clique = improve_clique_exact(
                        adjacency_bits, clique, clique_budget
                    )

                if best_k > len(clique) and time.perf_counter() < deadline:
                    best_coloring, best_k = exact_dsatur_search(
                        adjacency_list,
                        adjacency_bits,
                        clique,
                        best_coloring,
                        deadline,
                        accept_coloring,
                    )
                    # accept_coloring already records every strict improvement.
                    best_solution = make_solution(best_coloring)
                    best_k = best_solution["objective_value"]

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()