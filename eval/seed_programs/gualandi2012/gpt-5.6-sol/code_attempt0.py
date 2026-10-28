import argparse
import heapq
import json
import random
import sys
import time

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
    pass


def make_solution(coloring):
    """Project an internal coloring onto the required output schema."""
    if not coloring:
        return {
            "objective_value": 0,
            "coloring": {},
            "color_classes": {}
        }

    # Normalize color labels to consecutive integers.
    used = sorted(set(coloring))
    remap = {old: new for new, old in enumerate(used)}
    normalized = [remap[c] for c in coloring]

    color_classes = {}
    for vertex, color in enumerate(normalized):
        color_classes.setdefault(str(color), []).append(vertex)

    return {
        "objective_value": len(used),
        "coloring": {str(v): normalized[v] for v in range(len(normalized))},
        "color_classes": color_classes
    }


def read_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data["graph"]
    n = int(graph["num_vertices"])
    adjacency = [set() for _ in range(n)]

    for edge in graph["edges"]:
        if len(edge) != 2:
            raise ValueError("Each edge must contain exactly two vertex indices")
        u, v = int(edge[0]), int(edge[1])
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge endpoint out of range: ({u}, {v})")
        if u == v:
            raise ValueError(
                f"Graph contains a self-loop at vertex {u}; no feasible coloring exists"
            )
        adjacency[u].add(v)
        adjacency[v].add(u)

    return n, adjacency


def connected_components(adjacency):
    n = len(adjacency)
    seen = [False] * n
    components = []

    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        vertices = []

        while stack:
            u = stack.pop()
            vertices.append(u)
            for v in adjacency[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        components.append(vertices)

    return components


def build_local_component(vertices, global_adjacency):
    index = {v: i for i, v in enumerate(vertices)}
    local_adjacency = [set() for _ in vertices]

    for i, v in enumerate(vertices):
        local_adjacency[i] = {
            index[w] for w in global_adjacency[v] if w in index
        }

    return {
        "vertices": vertices,
        "adjacency": local_adjacency,
        "best_coloring": None,
        "best_count": None,
        "lower_bound": 1 if vertices else 0,
    }


def greedy_degree_coloring(adjacency):
    """Fast largest-degree-first greedy coloring."""
    n = len(adjacency)
    if n == 0:
        return [], 0

    order = sorted(range(n), key=lambda v: (-len(adjacency[v]), v))
    colors = [-1] * n
    max_color = -1

    for v in order:
        forbidden = set()
        for w in adjacency[v]:
            if colors[w] >= 0:
                forbidden.add(colors[w])

        color = 0
        while color in forbidden:
            color += 1
        colors[v] = color
        if color > max_color:
            max_color = color

    return colors, max_color + 1


def dsatur_greedy(adjacency, rng, randomized, deadline):
    """
    Heap-based DSATUR greedy coloring. The randomized variant only changes
    tie-breaking, preserving the DSATUR saturation/degree priorities.
    """
    n = len(adjacency)
    if n == 0:
        return [], 0

    colors = [-1] * n
    forbidden_masks = [0] * n
    versions = [0] * n
    priorities = [rng.random() if randomized else float(v) for v in range(n)]
    heap = []

    for v in range(n):
        heapq.heappush(
            heap,
            (0, -len(adjacency[v]), priorities[v], v, 0)
        )

    class_sizes = []
    max_color = -1
    colored_count = 0

    while colored_count < n:
        if colored_count % 256 == 0 and time.monotonic() >= deadline:
            return None

        while heap:
            neg_sat, neg_degree, _, v, version = heapq.heappop(heap)
            if colors[v] == -1 and version == versions[v]:
                current_sat = forbidden_masks[v].bit_count()
                if -neg_sat == current_sat:
                    break
        else:
            return None

        mask = forbidden_masks[v]

        # Prefer larger existing color classes, which often creates fewer colors.
        existing = list(range(max_color + 1))
        existing.sort(key=lambda c: (-class_sizes[c], c))

        chosen = None
        for color in existing:
            if not (mask >> color) & 1:
                chosen = color
                break

        if chosen is None:
            chosen = max_color + 1
            max_color = chosen
            class_sizes.append(0)

        colors[v] = chosen
        class_sizes[chosen] += 1
        colored_count += 1
        bit = 1 << chosen

        for w in adjacency[v]:
            if colors[w] != -1:
                continue
            if not (forbidden_masks[w] & bit):
                forbidden_masks[w] |= bit
                versions[w] += 1
                heapq.heappush(
                    heap,
                    (
                        -forbidden_masks[w].bit_count(),
                        -len(adjacency[w]),
                        priorities[w],
                        w,
                        versions[w],
                    )
                )

    return colors, max_color + 1


def greedy_clique(adjacency, deadline):
    """Find a valid clique to provide a lower bound and symmetry breaking."""
    n = len(adjacency)
    if n == 0:
        return []

    degrees = [len(adjacency[v]) for v in range(n)]
    degree_order = sorted(range(n), key=lambda v: (-degrees[v], v))
    best = []

    starts = [None] + degree_order[:min(24, n)]

    for start in starts:
        if time.monotonic() >= deadline:
            break

        if start is None:
            clique = []
            candidates = set(range(n))
        else:
            clique = [start]
            candidates = set(adjacency[start])

        while candidates:
            if len(clique) + len(candidates) <= len(best):
                break

            # Favor vertices with many remaining compatible candidates.
            if len(candidates) <= 300:
                v = max(
                    candidates,
                    key=lambda x: (
                        len(adjacency[x] & candidates),
                        degrees[x],
                        -x
                    )
                )
            else:
                v = max(candidates, key=lambda x: (degrees[x], -x))

            clique.append(v)
            candidates.intersection_update(adjacency[v])

        if len(clique) > len(best):
            best = clique

    if not best:
        best = [degree_order[0]]
    return best


def exact_dsatur_optimize(adjacency, incumbent_coloring, clique, deadline,
                          improvement_callback):
    """
    Exact DSATUR branch-and-bound optimization for one connected component.
    Returns the best coloring found. It may terminate at the deadline.
    """
    n = len(adjacency)
    if n == 0:
        return [], 0, True

    best_coloring = incumbent_coloring[:]
    best_count = max(best_coloring) + 1

    if len(clique) >= best_count:
        return best_coloring, best_count, True

    colors = [-1] * n
    forbidden_masks = [0] * n
    color_neighbor_counts = [dict() for _ in range(n)]
    degrees = [len(adjacency[v]) for v in range(n)]
    uncolored = set(range(n))
    class_sizes = [0] * best_count

    # A clique may be assigned canonical distinct colors without loss of
    # generality, eliminating global color-label symmetry.
    for color, v in enumerate(clique):
        colors[v] = color
        uncolored.remove(v)
        class_sizes[color] += 1

    for color, v in enumerate(clique):
        bit = 1 << color
        for w in adjacency[v]:
            if colors[w] != -1:
                continue
            counts = color_neighbor_counts[w]
            old = counts.get(color, 0)
            counts[color] = old + 1
            if old == 0:
                forbidden_masks[w] |= bit

    node_count = 0
    timed_out = False

    def choose_vertex():
        best_v = None
        best_key = None
        for v in uncolored:
            key = (forbidden_masks[v].bit_count(), degrees[v], -v)
            if best_key is None or key > best_key:
                best_key = key
                best_v = v
        return best_v

    def search(max_used):
        nonlocal best_coloring, best_count, node_count

        node_count += 1
        if node_count & 127 == 0 and time.monotonic() >= deadline:
            raise SearchTimeout

        used_count = max_used + 1
        if used_count >= best_count:
            return

        if not uncolored:
            best_count = used_count
            best_coloring = colors[:]
            improvement_callback(best_coloring, best_count)
            return

        v = choose_vertex()
        mask = forbidden_masks[v]

        # Existing colors first, favoring large classes.
        candidates = [
            c for c in range(max_used + 1) if not ((mask >> c) & 1)
        ]
        candidates.sort(key=lambda c: (-class_sizes[c], c))

        # Canonical introduction of at most one new color.
        new_color = max_used + 1
        if new_color + 1 < best_count:
            candidates.append(new_color)

        if not candidates:
            return

        uncolored.remove(v)

        for color in candidates:
            if time.monotonic() >= deadline:
                uncolored.add(v)
                raise SearchTimeout

            colors[v] = color
            if color >= len(class_sizes):
                class_sizes.extend([0] * (color + 1 - len(class_sizes)))
            class_sizes[color] += 1

            changed_neighbors = []
            bit = 1 << color

            for w in adjacency[v]:
                if colors[w] != -1:
                    continue
                counts = color_neighbor_counts[w]
                old = counts.get(color, 0)
                counts[color] = old + 1
                if old == 0:
                    forbidden_masks[w] |= bit
                changed_neighbors.append(w)

            search(max(max_used, color))

            for w in changed_neighbors:
                counts = color_neighbor_counts[w]
                new_value = counts[color] - 1
                if new_value == 0:
                    del counts[color]
                    forbidden_masks[w] &= ~bit
                else:
                    counts[color] = new_value

            class_sizes[color] -= 1
            colors[v] = -1

            if best_count <= len(clique):
                break

        uncolored.add(v)

    try:
        search(len(clique) - 1)
    except SearchTimeout:
        timed_out = True

    return best_coloring, best_count, not timed_out


def main():
    program_start = time.monotonic()

    parser = argparse.ArgumentParser(description="Minimum vertex coloring solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    deadline = program_start + max(0, args.time_limit)
    rng = random.Random(0)

    n, global_adjacency = read_graph(args.instance_path)

    if n == 0:
        solution = make_solution([])
        if logger:
            logger.log_solution(0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)
        return

    component_vertices = connected_components(global_adjacency)
    components = [
        build_local_component(vertices, global_adjacency)
        for vertices in component_vertices
    ]

    global_coloring = [-1] * n

    # Obtain a guaranteed feasible initial solution quickly.
    for component in components:
        colors, count = greedy_degree_coloring(component["adjacency"])
        component["best_coloring"] = colors
        component["best_count"] = count
        for local_v, global_v in enumerate(component["vertices"]):
            global_coloring[global_v] = colors[local_v]

    best_global_objective = max(c["best_count"] for c in components)
    initial_solution = make_solution(global_coloring)
    if logger:
        logger.log_solution(best_global_objective, initial_solution)

    def refresh_global_coloring(component):
        for local_v, global_v in enumerate(component["vertices"]):
            global_coloring[global_v] = component["best_coloring"][local_v]

    def maybe_log_global_improvement():
        nonlocal best_global_objective
        objective = max(c["best_count"] for c in components)
        if objective < best_global_objective:
            best_global_objective = objective
            solution = make_solution(global_coloring)
            if logger:
                logger.log_solution(objective, solution)

    # Improve initial upper bounds using deterministic and randomized DSATUR.
    total_edges = sum(len(a) for a in global_adjacency) // 2
    if n <= 2000 and total_edges <= 200000:
        passes = 10
    elif n <= 10000 and total_edges <= 1000000:
        passes = 4
    else:
        passes = 1

    for pass_index in range(passes):
        if time.monotonic() >= deadline:
            break

        randomized = pass_index > 0
        component_order = sorted(
            components,
            key=lambda c: (-c["best_count"], -len(c["vertices"]))
        )

        for component in component_order:
            if time.monotonic() >= deadline:
                break

            result = dsatur_greedy(
                component["adjacency"], rng, randomized, deadline
            )
            if result is None:
                break

            colors, count = result
            if count < component["best_count"]:
                component["best_coloring"] = colors
                component["best_count"] = count
                refresh_global_coloring(component)
                maybe_log_global_improvement()

    # Compute clique lower bounds.
    for component in components:
        if time.monotonic() >= deadline:
            break
        clique = greedy_clique(component["adjacency"], deadline)
        component["clique"] = clique
        component["lower_bound"] = len(clique)

    for component in components:
        if "clique" not in component:
            component["clique"] = [0] if component["vertices"] else []
            component["lower_bound"] = len(component["clique"])

    global_lower_bound = max(c["lower_bound"] for c in components)

    # Exact component-wise DSATUR. Since components may reuse colors, the
    # chromatic number of the whole graph is the maximum component value.
    exact_order = sorted(
        components,
        key=lambda c: (
            -c["best_count"],
            -(c["best_count"] - c["lower_bound"]),
            -len(c["vertices"])
        )
    )

    for component in exact_order:
        if time.monotonic() >= deadline:
            break
        if best_global_objective <= global_lower_bound:
            break
        if component["best_count"] <= component["lower_bound"]:
            continue

        # The scan-based exact search is intended for moderate components.
        if len(component["vertices"]) > 3000:
            continue

        def on_component_improvement(colors, count, comp=component):
            comp["best_coloring"] = colors[:]
            comp["best_count"] = count
            refresh_global_coloring(comp)
            maybe_log_global_improvement()

        colors, count, _ = exact_dsatur_optimize(
            component["adjacency"],
            component["best_coloring"],
            component["clique"],
            deadline,
            on_component_improvement
        )

        if count < component["best_count"]:
            component["best_coloring"] = colors
            component["best_count"] = count
            refresh_global_coloring(component)
            maybe_log_global_improvement()

    # Reconstruct to ensure consistency even after interrupted searches.
    for component in components:
        refresh_global_coloring(component)

    final_solution = make_solution(global_coloring)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()