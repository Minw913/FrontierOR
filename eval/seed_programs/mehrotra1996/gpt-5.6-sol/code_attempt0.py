import argparse
import json
import sys
import time
import heapq
from collections import deque

from solution_logger import SolutionLogger


def normalize_coloring(colors):
    """Renumber used colors consecutively starting at zero."""
    mapping = {}
    normalized = []
    next_color = 0
    for color in colors:
        if color not in mapping:
            mapping[color] = next_color
            next_color += 1
        normalized.append(mapping[color])
    return normalized, next_color


def make_solution(nodes, colors):
    colors, num_colors = normalize_coloring(colors)
    classes = [[] for _ in range(num_colors)]
    vertex_colors = {}

    for node, color in zip(nodes, colors):
        vertex_colors[str(node)] = color
        classes[color].append(node)

    return {
        "objective_value": num_colors,
        "vertex_colors": vertex_colors,
        "color_classes": classes,
    }


def sequential_greedy_coloring(adjacency, order):
    n = len(adjacency)
    colors = [-1] * n
    max_color = -1

    for vertex in order:
        forbidden = 0
        for neighbor in adjacency[vertex]:
            color = colors[neighbor]
            if color >= 0:
                forbidden |= 1 << color

        color = 0
        while forbidden & (1 << color):
            color += 1

        colors[vertex] = color
        if color > max_color:
            max_color = color

    return colors, max_color + 1


def tie_hash(vertex, variant):
    if variant == 0:
        return vertex
    x = (vertex + 0x9E3779B97F4A7C15 * variant) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return x


def greedy_dsatur(adjacency, degrees, variant=0):
    n = len(adjacency)
    colors = [-1] * n
    saturation_masks = [0] * n
    versions = [0] * n
    heap = []

    for vertex in range(n):
        heapq.heappush(
            heap,
            (0, -degrees[vertex], tie_hash(vertex, variant), vertex, 0),
        )

    colored_count = 0
    max_color = -1

    while colored_count < n:
        while True:
            _, _, _, vertex, version = heapq.heappop(heap)
            if colors[vertex] == -1 and versions[vertex] == version:
                break

        forbidden = saturation_masks[vertex]
        color = 0
        while forbidden & (1 << color):
            color += 1

        colors[vertex] = color
        colored_count += 1
        if color > max_color:
            max_color = color

        bit = 1 << color
        for neighbor in adjacency[vertex]:
            if colors[neighbor] == -1 and not (saturation_masks[neighbor] & bit):
                saturation_masks[neighbor] |= bit
                versions[neighbor] += 1
                heapq.heappush(
                    heap,
                    (
                        -saturation_masks[neighbor].bit_count(),
                        -degrees[neighbor],
                        tie_hash(neighbor, variant),
                        neighbor,
                        versions[neighbor],
                    ),
                )

    return colors, max_color + 1


def bipartite_coloring(adjacency):
    n = len(adjacency)
    colors = [-1] * n

    for start in range(n):
        if colors[start] != -1:
            continue

        colors[start] = 0
        queue = deque([start])

        while queue:
            vertex = queue.popleft()
            expected = 1 - colors[vertex]
            for neighbor in adjacency[vertex]:
                if colors[neighbor] == -1:
                    colors[neighbor] = expected
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return False, None

    return True, colors


def greedy_clique(adjacency_bits, degrees, deadline):
    """Find a valid clique to provide a certified lower bound."""
    n = len(adjacency_bits)
    if n == 0:
        return []

    degree_order = sorted(range(n), key=lambda v: (-degrees[v], v))
    starts = degree_order[: min(48, n)]
    best = [degree_order[0]]

    for start_index, start in enumerate(starts):
        if start_index and time.monotonic() >= deadline:
            break

        clique = [start]
        clique_mask = 1 << start

        for vertex in degree_order:
            if vertex == start:
                continue
            if clique_mask & ~adjacency_bits[vertex] == 0:
                clique.append(vertex)
                clique_mask |= 1 << vertex

        if len(clique) > len(best):
            best = clique

    # Also try a construction without forcing a particular first vertex.
    clique = []
    clique_mask = 0
    for vertex in degree_order:
        if clique_mask & ~adjacency_bits[vertex] == 0:
            clique.append(vertex)
            clique_mask |= 1 << vertex
    if len(clique) > len(best):
        best = clique

    return best


class ExactDSATUR:
    def __init__(
        self,
        adjacency,
        degrees,
        initial_colors,
        clique,
        deadline,
        incumbent_callback,
    ):
        self.adjacency = adjacency
        self.degrees = degrees
        self.n = len(adjacency)
        self.best_colors = initial_colors[:]
        self.best_k = max(initial_colors) + 1 if initial_colors else 0
        self.clique = clique
        self.deadline = deadline
        self.incumbent_callback = incumbent_callback

        self.colors = [-1] * self.n
        self.saturation_masks = [0] * self.n
        self.remaining_degrees = degrees[:]
        self.color_sizes = [0] * max(1, self.best_k)
        self.nodes_searched = 0
        self.timed_out = False

    def assign(self, vertex, color):
        self.colors[vertex] = color
        self.color_sizes[color] += 1
        changed_saturation = []

        bit = 1 << color
        for neighbor in self.adjacency[vertex]:
            if self.colors[neighbor] == -1:
                self.remaining_degrees[neighbor] -= 1
                if not (self.saturation_masks[neighbor] & bit):
                    self.saturation_masks[neighbor] |= bit
                    changed_saturation.append(neighbor)

        return changed_saturation

    def unassign(self, vertex, color, changed_saturation):
        bit = 1 << color
        for neighbor in self.adjacency[vertex]:
            if self.colors[neighbor] == -1:
                self.remaining_degrees[neighbor] += 1

        for neighbor in changed_saturation:
            self.saturation_masks[neighbor] &= ~bit

        self.color_sizes[color] -= 1
        self.colors[vertex] = -1

    def select_vertex(self):
        selected = -1
        selected_key = None

        for vertex in range(self.n):
            if self.colors[vertex] != -1:
                continue
            key = (
                self.saturation_masks[vertex].bit_count(),
                self.remaining_degrees[vertex],
                self.degrees[vertex],
                -vertex,
            )
            if selected_key is None or key > selected_key:
                selected_key = key
                selected = vertex

        return selected

    def search(self, colored_count, used_colors):
        self.nodes_searched += 1
        if (self.nodes_searched & 31) == 0 and time.monotonic() >= self.deadline:
            self.timed_out = True
            return

        if used_colors >= self.best_k:
            return

        if colored_count == self.n:
            self.best_k = used_colors
            self.best_colors = self.colors[:]
            self.incumbent_callback(self.best_k, self.best_colors)
            return

        vertex = self.select_vertex()
        if vertex < 0:
            return

        forbidden = self.saturation_masks[vertex]

        available_existing = [
            color
            for color in range(used_colors)
            if not (forbidden & (1 << color))
        ]
        # Trying large classes first often causes stronger propagation.
        available_existing.sort(
            key=lambda color: (-self.color_sizes[color], color)
        )

        for color in available_existing:
            changed = self.assign(vertex, color)
            self.search(colored_count + 1, used_colors)
            self.unassign(vertex, color, changed)

            if self.timed_out:
                return

        # By color symmetry only one new color needs to be considered.
        if used_colors + 1 < self.best_k:
            color = used_colors
            changed = self.assign(vertex, color)
            self.search(colored_count + 1, used_colors + 1)
            self.unassign(vertex, color, changed)

    def solve(self):
        used_colors = 0
        colored_count = 0

        # Fix the clique to distinct colors. Any coloring can be relabeled to
        # satisfy this restriction, so it does not remove optimal solutions.
        for color, vertex in enumerate(self.clique):
            self.assign(vertex, color)
            used_colors += 1
            colored_count += 1

        self.search(colored_count, used_colors)
        return self.best_colors, self.best_k, not self.timed_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    graph = instance["graph"]
    nodes = graph["nodes"]
    n = len(nodes)

    if n == 0:
        solution = {
            "objective_value": 0,
            "vertex_colors": {},
            "color_classes": [],
        }
        if logger:
            logger.log_solution(0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(solution, file)
        return

    index_of = {node: index for index, node in enumerate(nodes)}
    adjacency = [set() for _ in range(n)]

    for edge in graph["edges"]:
        left = index_of[edge[0]]
        right = index_of[edge[1]]
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    degrees = [len(neighbors) for neighbors in adjacency]
    adjacency_bits = [0] * n
    for vertex in range(n):
        bits = 0
        for neighbor in adjacency[vertex]:
            bits |= 1 << neighbor
        adjacency_bits[vertex] = bits

    best_colors = None
    best_k = n + 1

    def submit(colors, claimed_k=None):
        nonlocal best_colors, best_k
        normalized, actual_k = normalize_coloring(colors)
        if claimed_k is not None:
            actual_k = min(actual_k, claimed_k)
        if actual_k < best_k:
            best_colors = normalized
            best_k = actual_k
            solution = make_solution(nodes, best_colors)
            if logger:
                logger.log_solution(best_k, solution)
            return True
        return False

    # A fast initial incumbent is always produced, even for very short limits.
    degree_order = sorted(range(n), key=lambda v: (-degrees[v], v))
    initial_colors, initial_k = sequential_greedy_coloring(
        adjacency, degree_order
    )
    submit(initial_colors, initial_k)

    edge_count = sum(degrees) // 2
    if edge_count == 0:
        final_solution = make_solution(nodes, best_colors)
        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(final_solution, file)
        return

    # Bipartite graphs are solved optimally in linear time.
    is_bipartite, bipartite_colors = bipartite_coloring(adjacency)
    if is_bipartite:
        submit(bipartite_colors, 2)
        final_solution = make_solution(nodes, best_colors)
        with open(args.solution_path, "w", encoding="utf-8") as file:
            json.dump(final_solution, file)
        return

    # Run several deterministic DSATUR tie-breaking variants.
    heuristic_deadline = min(
        deadline,
        start_time + max(0.05, 0.15 * max(0, args.time_limit)),
    )
    variant = 0
    while variant < 12 and time.monotonic() < heuristic_deadline:
        colors, color_count = greedy_dsatur(adjacency, degrees, variant)
        submit(colors, color_count)
        variant += 1

    clique_deadline = min(
        deadline,
        time.monotonic() + max(0.01, 0.03 * max(0, args.time_limit)),
    )
    clique = greedy_clique(adjacency_bits, degrees, clique_deadline)

    # Non-bipartiteness certifies a lower bound of 3.
    lower_bound = max(len(clique), 3)

    # If the bounds meet, the incumbent is already proven optimal.
    if best_k > lower_bound and time.monotonic() < deadline:
        # Very large exact searches can exhaust the Python call stack and are
        # generally unproductive under a single-core time limit.
        if n <= 3000:
            sys.setrecursionlimit(max(10000, n + 1000))

            def exact_incumbent(color_count, colors):
                submit(colors, color_count)

            # Leave a small amount of time for serialization.
            exact_deadline = max(
                time.monotonic(),
                deadline - min(0.05, 0.01 * max(1, args.time_limit)),
            )
            exact_solver = ExactDSATUR(
                adjacency=adjacency,
                degrees=degrees,
                initial_colors=best_colors,
                clique=clique,
                deadline=exact_deadline,
                incumbent_callback=exact_incumbent,
            )
            exact_colors, exact_k, _ = exact_solver.solve()
            submit(exact_colors, exact_k)

    final_solution = make_solution(nodes, best_colors)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(final_solution, file)


if __name__ == "__main__":
    main()