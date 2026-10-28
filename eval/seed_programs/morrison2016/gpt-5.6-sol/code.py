import argparse
import json
import os
import random
import signal
import sys
import time

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
    pass


class OptimalFound(Exception):
    pass


def build_solution(colors):
    """Convert a complete internal coloring to the required output schema."""
    if not colors:
        return {
            "objective_value": 0,
            "selected_sets": [],
            "coloring": {}
        }

    used = sorted(set(colors))
    remap = {old: new for new, old in enumerate(used)}
    normalized = [remap[c] for c in colors]

    selected_sets = [[] for _ in used]
    for vertex, color in enumerate(normalized):
        selected_sets[color].append(vertex)

    return {
        "objective_value": len(selected_sets),
        "selected_sets": selected_sets,
        "coloring": {str(v): normalized[v] for v in range(len(normalized))}
    }


def greedy_degree_coloring(adj, degrees):
    """Fast deterministic greedy coloring in descending degree order."""
    n = len(adj)
    order = sorted(range(n), key=lambda v: (-degrees[v], v))
    colors = [-1] * n
    max_color = -1

    for v in order:
        forbidden = 0
        for u in adj[v]:
            c = colors[u]
            if c >= 0:
                forbidden |= 1 << c

        c = 0
        while forbidden & (1 << c):
            c += 1
        colors[v] = c
        if c > max_color:
            max_color = c

    return colors


def dsatur_greedy(adj, degrees, deadline, rng=None):
    """DSATUR constructive heuristic, optionally randomizing exact ties."""
    n = len(adj)
    if n == 0:
        return []

    colors = [-1] * n
    saturation_masks = [0] * n
    uncolored = set(range(n))

    for step in range(n):
        if (step & 63) == 0 and time.perf_counter() >= deadline:
            raise SearchTimeout

        best_sat = -1
        best_degree = -1
        candidates = []

        for v in uncolored:
            sat = saturation_masks[v].bit_count()
            degree = degrees[v]
            if sat > best_sat or (sat == best_sat and degree > best_degree):
                best_sat = sat
                best_degree = degree
                candidates = [v]
            elif sat == best_sat and degree == best_degree:
                candidates.append(v)

        if rng is None:
            v = min(candidates)
        else:
            v = candidates[rng.randrange(len(candidates))]

        forbidden = saturation_masks[v]
        color = 0
        while forbidden & (1 << color):
            color += 1

        colors[v] = color
        uncolored.remove(v)
        bit = 1 << color
        for u in adj[v]:
            if colors[u] < 0:
                saturation_masks[u] |= bit

    return colors


def find_greedy_clique(adj_bits, degrees, deadline, max_starts=24):
    """Find a valid clique to strengthen and symmetry-break exact search."""
    n = len(adj_bits)
    if n == 0:
        return []

    degree_order = sorted(range(n), key=lambda v: (-degrees[v], v))
    starts = degree_order[:max_starts]
    if degree_order and degree_order[-1] not in starts:
        starts.append(degree_order[-1])

    best = [degree_order[0]]

    for start_index, start in enumerate(starts):
        if time.perf_counter() >= deadline:
            break

        clique = [start]
        candidates = adj_bits[start]

        while candidates:
            if (len(clique) & 15) == 0 and time.perf_counter() >= deadline:
                break

            # Choose a candidate having many neighbors among the remaining
            # candidates. This is more expensive than static-degree ordering,
            # so limit full scoring on very large candidate sets.
            if candidates.bit_count() <= 256:
                scan = candidates
                chosen = -1
                chosen_score = -1
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    score = (adj_bits[v] & candidates).bit_count()
                    if score > chosen_score or (
                        score == chosen_score and
                        (chosen < 0 or degrees[v] > degrees[chosen])
                    ):
                        chosen = v
                        chosen_score = score
                    scan ^= bit
            else:
                scan = candidates
                chosen = -1
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    if chosen < 0 or degrees[v] > degrees[chosen]:
                        chosen = v
                    scan ^= bit

            clique.append(chosen)
            candidates &= adj_bits[chosen]

        if len(clique) > len(best):
            best = clique

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.perf_counter()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    graph = instance["graph"]
    n = int(graph["num_vertices"])

    adj_bits = [0] * n
    for edge in graph.get("edges", []):
        u = int(edge[0])
        v = int(edge[1])
        if u == v:
            continue
        if 0 <= u < n and 0 <= v < n:
            adj_bits[u] |= 1 << v
            adj_bits[v] |= 1 << u

    adj = [[] for _ in range(n)]
    degrees = [0] * n
    for v in range(n):
        bits = adj_bits[v]
        degrees[v] = bits.bit_count()
        neighbors = []
        while bits:
            bit = bits & -bits
            neighbors.append(bit.bit_length() - 1)
            bits ^= bit
        adj[v] = neighbors

    if n == 0:
        final_solution = build_solution([])
        if logger:
            logger.log_solution(0, final_solution)
        output_dir = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(final_solution, f, separators=(",", ":"))
        return

    # Always obtain a feasible solution before starting interruptible work.
    best_colors = greedy_degree_coloring(adj, degrees)
    best_count = len(set(best_colors))
    initial_solution = build_solution(best_colors)
    if logger:
        logger.log_solution(best_count, initial_solution)

    supplied_lower = int(
        instance.get("bounds", {}).get("clique_lower_bound", 1)
    )
    supplied_lower = max(1, min(n, supplied_lower))
    proven_lower = supplied_lower

    def register_incumbent(candidate):
        nonlocal best_colors, best_count
        count = len(set(candidate))
        if count >= best_count:
            return

        # Copy first so that a timer interruption cannot leave a reference to
        # mutable search state.
        candidate_copy = list(candidate)
        best_colors = candidate_copy
        best_count = count
        solution = build_solution(candidate_copy)
        if logger:
            logger.log_solution(count, solution)

        if best_count <= proven_lower:
            raise OptimalFound

    def alarm_handler(signum, frame):
        raise SearchTimeout

    advanced_search_started = False

    try:
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, remaining)
            advanced_search_started = True

            # Deterministic DSATUR is usually substantially better than a
            # simple degree-ordered greedy coloring.
            candidate = dsatur_greedy(adj, degrees, deadline)
            register_incumbent(candidate)

            # A few reproducible randomized tie-break attempts can improve the
            # upper bound without consuming too much of the search budget.
            rng = random.Random(0)
            heuristic_attempts = 3 if n <= 2500 else 1
            for _ in range(heuristic_attempts):
                if time.perf_counter() >= deadline:
                    raise SearchTimeout
                candidate = dsatur_greedy(adj, degrees, deadline, rng)
                register_incumbent(candidate)

            clique = find_greedy_clique(adj_bits, degrees, deadline)
            proven_lower = max(proven_lower, len(clique))

            if best_count <= proven_lower:
                raise OptimalFound

            # Exact DSATUR branch-and-bound. Fixing the discovered clique to
            # distinct colors is a valid symmetry break.
            sys.setrecursionlimit(max(2000, n + 200))

            colors = [-1] * n
            saturation_masks = [0] * n
            color_sizes = [0] * n
            uncolored_mask = (1 << n) - 1

            for color, v in enumerate(clique):
                colors[v] = color
                color_sizes[color] = 1
                uncolored_mask &= ~(1 << v)

            for color, v in enumerate(clique):
                bit = 1 << color
                for u in adj[v]:
                    if colors[u] < 0:
                        saturation_masks[u] |= bit

            initial_used = len(clique)
            nodes = 0

            def greedy_complete(mask, used):
                """Greedily complete a partial exact-search assignment."""
                temp_colors = list(colors)
                temp_sat = list(saturation_masks)
                temp_mask = mask
                temp_used = used

                while temp_mask:
                    if time.perf_counter() >= deadline:
                        raise SearchTimeout

                    scan = temp_mask
                    chosen = -1
                    chosen_sat = -1
                    chosen_degree = -1
                    while scan:
                        bit = scan & -scan
                        v = bit.bit_length() - 1
                        sat = temp_sat[v].bit_count()
                        if (
                            sat > chosen_sat
                            or (sat == chosen_sat and degrees[v] > chosen_degree)
                            or (
                                sat == chosen_sat
                                and degrees[v] == chosen_degree
                                and (chosen < 0 or v < chosen)
                            )
                        ):
                            chosen = v
                            chosen_sat = sat
                            chosen_degree = degrees[v]
                        scan ^= bit

                    forbidden = temp_sat[chosen]
                    c = 0
                    while forbidden & (1 << c):
                        c += 1

                    temp_colors[chosen] = c
                    if c == temp_used:
                        temp_used += 1
                    temp_mask &= ~(1 << chosen)
                    color_bit = 1 << c
                    for u in adj[chosen]:
                        if temp_colors[u] < 0:
                            temp_sat[u] |= color_bit

                return temp_colors

            # A clique-seeded completion is often stronger than the initial
            # whole-graph heuristics.
            if n <= 4000 and time.perf_counter() < deadline:
                register_incumbent(
                    greedy_complete(uncolored_mask, initial_used)
                )

            def search(mask, used):
                nonlocal nodes, best_count

                nodes += 1
                if (nodes & 255) == 0 and time.perf_counter() >= deadline:
                    raise SearchTimeout

                if used >= best_count:
                    return

                if not mask:
                    register_incumbent(colors)
                    return

                scan = mask
                v = -1
                best_sat = -1
                best_degree = -1
                while scan:
                    bit = scan & -scan
                    u = bit.bit_length() - 1
                    sat = saturation_masks[u].bit_count()
                    if (
                        sat > best_sat
                        or (sat == best_sat and degrees[u] > best_degree)
                        or (
                            sat == best_sat
                            and degrees[u] == best_degree
                            and (v < 0 or u < v)
                        )
                    ):
                        v = u
                        best_sat = sat
                        best_degree = degrees[u]
                    scan ^= bit

                forbidden = saturation_masks[v]

                # Existing colors: larger classes first generally encourage
                # compact colorings and early discovery of better incumbents.
                available = [
                    c for c in range(used)
                    if not (forbidden & (1 << c))
                ]
                available.sort(key=lambda c: (-color_sizes[c], c))

                # At most one new color is needed due to color-label symmetry.
                if used + 1 < best_count:
                    available.append(used)

                next_mask = mask & ~(1 << v)

                for c in available:
                    is_new = (c == used)
                    colors[v] = c
                    color_sizes[c] += 1
                    color_bit = 1 << c
                    changes = []

                    for u in adj[v]:
                        if colors[u] < 0:
                            old_mask = saturation_masks[u]
                            new_mask = old_mask | color_bit
                            if new_mask != old_mask:
                                saturation_masks[u] = new_mask
                                changes.append((u, old_mask))

                    search(next_mask, used + 1 if is_new else used)

                    for u, old_mask in changes:
                        saturation_masks[u] = old_mask
                    color_sizes[c] -= 1
                    colors[v] = -1

            search(uncolored_mask, initial_used)

    except (SearchTimeout, OptimalFound):
        pass
    finally:
        if advanced_search_started:
            signal.setitimer(signal.ITIMER_REAL, 0)

    final_solution = build_solution(best_colors)
    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()