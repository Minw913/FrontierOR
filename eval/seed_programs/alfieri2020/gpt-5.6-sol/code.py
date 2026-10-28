import argparse
import heapq
import json
import sys
import time
from typing import Callable, List, Optional, Set, Tuple

from solution_logger import SolutionLogger


def solution_dict(colors: List[int], objective: int) -> dict:
    return {
        "objective_value": float(objective),
        "coloring": {str(v): int(colors[v]) for v in range(len(colors))},
    }


def greedy_coloring(adjacency: List[Set[int]], order: List[int]) -> Tuple[List[int], int]:
    n = len(adjacency)
    if n == 0:
        return [], 0

    colors = [-1] * n
    max_color = -1

    for vertex in order:
        forbidden = {colors[u] for u in adjacency[vertex] if colors[u] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[vertex] = color
        if color > max_color:
            max_color = color

    return colors, max_color + 1


def dsatur_greedy(adjacency: List[Set[int]], deadline: float) -> Optional[Tuple[List[int], int]]:
    n = len(adjacency)
    if n == 0:
        return [], 0
    if time.monotonic() >= deadline:
        return None

    colors = [-1] * n
    saturation_masks = [0] * n
    degrees = [len(adjacency[v]) for v in range(n)]
    heap = []

    for v in range(n):
        heapq.heappush(heap, (0, -degrees[v], v))

    max_color = -1
    colored_count = 0

    while colored_count < n:
        if time.monotonic() >= deadline:
            return None

        while heap:
            neg_sat, neg_degree, v = heapq.heappop(heap)
            if colors[v] >= 0:
                continue
            current_sat = saturation_masks[v].bit_count()
            if neg_sat != -current_sat or neg_degree != -degrees[v]:
                continue
            break
        else:
            return None

        forbidden = saturation_masks[v]
        color = 0
        while forbidden & (1 << color):
            color += 1

        colors[v] = color
        colored_count += 1
        max_color = max(max_color, color)

        color_bit = 1 << color
        for u in adjacency[v]:
            if colors[u] < 0 and not (saturation_masks[u] & color_bit):
                saturation_masks[u] |= color_bit
                heapq.heappush(
                    heap,
                    (-saturation_masks[u].bit_count(), -degrees[u], u),
                )

    return colors, max_color + 1


def connected_components(adjacency: List[Set[int]]) -> List[List[int]]:
    n = len(adjacency)
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue

        visited[start] = True
        stack = [start]
        component = []

        while stack:
            v = stack.pop()
            component.append(v)
            for u in adjacency[v]:
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)

        component.sort()
        components.append(component)

    return components


def normalize_component_coloring(
    component: List[int], global_colors: List[int]
) -> Tuple[List[int], int]:
    mapping = {}
    local_colors = []

    for vertex in component:
        old_color = global_colors[vertex]
        if old_color not in mapping:
            mapping[old_color] = len(mapping)
        local_colors.append(mapping[old_color])

    return local_colors, len(mapping)


def build_local_bit_graph(
    component: List[int], adjacency: List[Set[int]]
) -> List[int]:
    local_index = {vertex: i for i, vertex in enumerate(component)}
    local_adjacency = [0] * len(component)

    for i, vertex in enumerate(component):
        mask = 0
        for neighbor in adjacency[vertex]:
            j = local_index.get(neighbor)
            if j is not None:
                mask |= 1 << j
        local_adjacency[i] = mask

    return local_adjacency


def heuristic_clique(adjacency_bits: List[int], deadline: float) -> List[int]:
    n = len(adjacency_bits)
    if n == 0:
        return []

    degrees = [mask.bit_count() for mask in adjacency_bits]
    best = [max(range(n), key=lambda v: (degrees[v], -v))]

    # Ensure an edge supplies a clique lower bound of two.
    for v in range(n):
        if adjacency_bits[v]:
            low_bit = adjacency_bits[v] & -adjacency_bits[v]
            u = low_bit.bit_length() - 1
            best = [v, u]
            break

    order = sorted(range(n), key=lambda v: (-degrees[v], v))

    # A fast ordered greedy clique.
    clique = []
    clique_mask = 0
    for v in order:
        if not clique or (adjacency_bits[v] & clique_mask) == clique_mask:
            clique.append(v)
            clique_mask |= 1 << v
    if len(clique) > len(best):
        best = clique

    if time.monotonic() >= deadline:
        return best

    max_starts = min(n, 64 if n <= 500 else 24)
    for start in order[:max_starts]:
        if time.monotonic() >= deadline:
            break

        clique = [start]
        candidates = adjacency_bits[start]

        while candidates:
            if time.monotonic() >= deadline:
                break

            chosen = -1
            chosen_score = -1
            scan = candidates

            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                score = (adjacency_bits[v] & candidates).bit_count()
                if score > chosen_score or (
                    score == chosen_score and (chosen < 0 or degrees[v] > degrees[chosen])
                ):
                    chosen = v
                    chosen_score = score
                scan ^= bit

            clique.append(chosen)
            candidates &= adjacency_bits[chosen]
            candidates &= ~(1 << chosen)

        if len(clique) > len(best):
            best = clique

    return best


def exact_dsatur_component(
    adjacency_bits: List[int],
    initial_colors: List[int],
    initial_k: int,
    deadline: float,
    improvement_callback: Callable[[List[int], int], None],
) -> Tuple[List[int], int, bool]:
    """
    DSATUR branch-and-bound. Returns best coloring, its color count, and whether
    the search proved optimal (as opposed to stopping at its deadline).
    """
    n = len(adjacency_bits)
    if n == 0:
        return [], 0, True
    if initial_k <= 1:
        return initial_colors[:], initial_k, True
    if time.monotonic() >= deadline:
        return initial_colors[:], initial_k, False

    clique = heuristic_clique(adjacency_bits, deadline)
    clique_size = len(clique)

    best_colors = initial_colors[:]
    best_k = initial_k

    if clique_size >= best_k:
        return best_colors, best_k, True
    if time.monotonic() >= deadline:
        return best_colors, best_k, False

    colors = [-1] * n
    saturation_masks = [0] * n
    degrees = [mask.bit_count() for mask in adjacency_bits]
    class_sizes = [0] * max(n, best_k + 1)

    colored_count = 0
    for color, vertex in enumerate(clique):
        colors[vertex] = color
        class_sizes[color] += 1
        colored_count += 1

    for vertex in clique:
        color_bit = 1 << colors[vertex]
        neighbors = adjacency_bits[vertex]
        while neighbors:
            bit = neighbors & -neighbors
            u = bit.bit_length() - 1
            if colors[u] < 0:
                saturation_masks[u] |= color_bit
            neighbors ^= bit

    timed_out = False

    def recurse(used_colors: int, count_colored: int) -> bool:
        nonlocal best_colors, best_k, timed_out

        if time.monotonic() >= deadline:
            timed_out = True
            return True

        if best_k == clique_size:
            return True

        if used_colors >= best_k:
            return False

        if count_colored == n:
            best_k = used_colors
            best_colors = colors[:]
            improvement_callback(best_colors[:], best_k)
            return best_k == clique_size

        selected = -1
        selected_key = None

        for v in range(n):
            if colors[v] >= 0:
                continue
            key = (saturation_masks[v].bit_count(), degrees[v], -v)
            if selected_key is None or key > selected_key:
                selected = v
                selected_key = key

        forbidden = saturation_masks[selected]

        available_colors = [
            c for c in range(used_colors) if not (forbidden & (1 << c))
        ]
        available_colors.sort(key=lambda c: (-class_sizes[c], c))

        def assign_and_recurse(color: int, next_used: int) -> bool:
            colors[selected] = color
            class_sizes[color] += 1
            color_bit = 1 << color
            changed_neighbors = []

            neighbors = adjacency_bits[selected]
            while neighbors:
                bit = neighbors & -neighbors
                u = bit.bit_length() - 1
                if colors[u] < 0 and not (saturation_masks[u] & color_bit):
                    saturation_masks[u] |= color_bit
                    changed_neighbors.append(u)
                neighbors ^= bit

            stop = recurse(next_used, count_colored + 1)

            for u in changed_neighbors:
                saturation_masks[u] &= ~color_bit
            class_sizes[color] -= 1
            colors[selected] = -1
            return stop

        for color in available_colors:
            if assign_and_recurse(color, used_colors):
                return True

        if used_colors + 1 < best_k:
            if assign_and_recurse(used_colors, used_colors + 1):
                return True

        return False

    recurse(clique_size, colored_count)
    return best_colors, best_k, not timed_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    n = int(instance["num_nodes"])
    adjacency: List[Set[int]] = [set() for _ in range(n)]

    for edge in instance["edges"]:
        u = int(edge[0])
        v = int(edge[1])
        if u == v:
            continue
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge endpoint outside 0..{n - 1}: {edge}")
        adjacency[u].add(v)
        adjacency[v].add(u)

    if n == 0:
        final_solution = solution_dict([], 0)
        if logger:
            logger.log_solution(0.0, final_solution)
        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(final_solution, file)
        return

    # Immediate, always-available feasible incumbent.
    best_colors, best_k = greedy_coloring(adjacency, list(range(n)))
    if logger:
        logger.log_solution(float(best_k), solution_dict(best_colors, best_k))

    # Static largest-degree-first greedy coloring.
    if time.monotonic() < deadline:
        degree_order = sorted(range(n), key=lambda v: (-len(adjacency[v]), v))
        colors, k = greedy_coloring(adjacency, degree_order)
        if k < best_k:
            best_colors, best_k = colors, k
            if logger:
                logger.log_solution(float(best_k), solution_dict(best_colors, best_k))

    # Usually provides the strongest inexpensive initial upper bound.
    if time.monotonic() < deadline:
        dsatur_result = dsatur_greedy(adjacency, deadline)
        if dsatur_result is not None:
            colors, k = dsatur_result
            if k < best_k:
                best_colors, best_k = colors, k
                if logger:
                    logger.log_solution(
                        float(best_k), solution_dict(best_colors, best_k)
                    )

    components = connected_components(adjacency)
    component_colors: List[List[int]] = []
    component_k: List[int] = []

    for component in components:
        local_colors, local_k = normalize_component_coloring(component, best_colors)
        component_colors.append(local_colors)
        component_k.append(local_k)

    def assemble_global_coloring() -> Tuple[List[int], int]:
        colors = [0] * n
        objective = 0
        for idx, component in enumerate(components):
            objective = max(objective, component_k[idx])
            local = component_colors[idx]
            for j, vertex in enumerate(component):
                colors[vertex] = local[j]
        return colors, objective

    combined_colors, combined_k = assemble_global_coloring()
    if combined_k < best_k:
        best_colors, best_k = combined_colors, combined_k
        if logger:
            logger.log_solution(float(best_k), solution_dict(best_colors, best_k))
    else:
        best_colors = combined_colors
        best_k = combined_k

    sys.setrecursionlimit(max(10000, 2 * n + 100))

    candidates = [
        i
        for i, component in enumerate(components)
        if component_k[i] > 1 and len(component) > 1
    ]
    candidates.sort(
        key=lambda i: (-component_k[i], -len(components[i]), components[i][0])
    )

    timed_out_components = []

    for position, component_index in enumerate(candidates):
        now = time.monotonic()
        if now >= deadline:
            break

        remaining_candidates = len(candidates) - position
        remaining_time = deadline - now
        slice_seconds = remaining_time / max(1, remaining_candidates)
        local_deadline = min(deadline, now + max(0.01, slice_seconds))

        component = components[component_index]
        local_bits = build_local_bit_graph(component, adjacency)

        def on_local_improvement(
            local_solution: List[int],
            local_objective: int,
            idx: int = component_index,
        ) -> None:
            nonlocal best_colors, best_k
            component_colors[idx] = local_solution[:]
            component_k[idx] = local_objective
            global_colors, global_k = assemble_global_coloring()
            if global_k < best_k:
                best_colors, best_k = global_colors, global_k
                if logger:
                    logger.log_solution(
                        float(best_k), solution_dict(best_colors, best_k)
                    )

        local_best, local_k, proved = exact_dsatur_component(
            local_bits,
            component_colors[component_index],
            component_k[component_index],
            local_deadline,
            on_local_improvement,
        )
        component_colors[component_index] = local_best
        component_k[component_index] = local_k

        if not proved:
            timed_out_components.append(component_index)

    # If allocated slices finished early, spend remaining time on the currently
    # most relevant unfinished component.
    if timed_out_components and time.monotonic() < deadline:
        timed_out_components.sort(
            key=lambda i: (-component_k[i], -len(components[i]), components[i][0])
        )
        component_index = timed_out_components[0]
        component = components[component_index]
        local_bits = build_local_bit_graph(component, adjacency)

        def on_final_improvement(
            local_solution: List[int], local_objective: int
        ) -> None:
            nonlocal best_colors, best_k
            component_colors[component_index] = local_solution[:]
            component_k[component_index] = local_objective
            global_colors, global_k = assemble_global_coloring()
            if global_k < best_k:
                best_colors, best_k = global_colors, global_k
                if logger:
                    logger.log_solution(
                        float(best_k), solution_dict(best_colors, best_k)
                    )

        local_best, local_k, _ = exact_dsatur_component(
            local_bits,
            component_colors[component_index],
            component_k[component_index],
            deadline,
            on_final_improvement,
        )
        component_colors[component_index] = local_best
        component_k[component_index] = local_k

    final_colors, final_k = assemble_global_coloring()
    if final_k <= best_k:
        best_colors, best_k = final_colors, final_k

    final_solution = solution_dict(best_colors, best_k)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(final_solution, file)


if __name__ == "__main__":
    main()