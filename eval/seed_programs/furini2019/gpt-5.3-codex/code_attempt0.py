import argparse
import json
import random
import sys
import time
from typing import List, Tuple, Optional

from solution_logger import SolutionLogger


def build_adjacency_masks(instance: dict) -> List[int]:
    n = int(instance["num_vertices"])
    adj = [0] * n

    edges = instance.get("edges", None)
    if edges is not None:
        for e in edges:
            if not isinstance(e, list) or len(e) != 2:
                continue
            u, v = int(e[0]), int(e[1])
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    else:
        # Fallback to adjacency_list if edges are not provided
        adjacency_list = instance.get("adjacency_list", [])
        for u in range(min(n, len(adjacency_list))):
            for v in adjacency_list[u]:
                v = int(v)
                if 0 <= v < n and v != u:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u

    return adj


class MaxCliqueSolver:
    def __init__(
        self,
        adjacency_masks: List[int],
        deadline: float,
        logger: Optional[SolutionLogger] = None,
    ):
        self.adj = adjacency_masks
        self.n = len(adjacency_masks)
        self.deg = [m.bit_count() for m in adjacency_masks]
        self.deadline = deadline
        self.logger = logger

        self.best_clique: List[int] = []
        self.best_size = 0

        self.nodes = 0
        self.timeout = False

    def time_up(self) -> bool:
        return time.time() >= self.deadline

    def update_incumbent(self, clique: List[int]):
        sz = len(clique)
        if sz > self.best_size:
            self.best_size = sz
            self.best_clique = clique.copy()
            if self.logger:
                self.logger.log(self.best_size)

    def greedy_clique(self, order: List[int]) -> List[int]:
        clique = []
        candidates = order[:]

        while candidates and not self.time_up():
            v = candidates[0]
            clique.append(v)
            vm = self.adj[v]

            nxt = [u for u in candidates[1:] if (vm >> u) & 1]
            if nxt:
                nxt.sort(key=lambda x: self.deg[x], reverse=True)
            candidates = nxt

        return clique

    def initial_heuristics(self):
        if self.n == 0:
            return

        # Always have at least a feasible clique of size 1 when n > 0
        start_v = max(range(self.n), key=lambda x: self.deg[x])
        self.update_incumbent([start_v])

        if self.time_up():
            return

        degree_desc = sorted(range(self.n), key=lambda x: self.deg[x], reverse=True)
        degree_asc = list(reversed(degree_desc))

        # Deterministic random tries
        rng = random.Random(0)

        # Reserve a small fraction of total time for heuristics
        heuristic_deadline = min(self.deadline, time.time() + max(0.1, 0.15 * max(self.deadline - time.time(), 0.0)))

        # Try structured orders
        for order in (degree_desc, degree_asc):
            if time.time() >= heuristic_deadline or self.time_up():
                return
            c = self.greedy_clique(order)
            self.update_incumbent(c)

        # Randomized restarts
        tries = 24
        for _ in range(tries):
            if time.time() >= heuristic_deadline or self.time_up():
                return
            order = degree_desc[:]
            rng.shuffle(order)
            c = self.greedy_clique(order)
            self.update_incumbent(c)

    def color_sort(self, P: List[int]) -> Tuple[List[int], List[int]]:
        """
        Greedy sequential coloring for an upper bound.
        Returns vertices reordered by color classes (nondecreasing color),
        and bounds where bounds[i] is color number of ordered[i].
        """
        color_class_masks: List[int] = []
        color_class_vertices: List[List[int]] = []

        for v in P:
            placed = False
            vm = self.adj[v]
            bit = 1 << v
            for c in range(len(color_class_masks)):
                # Same color class is an independent set:
                # v can be placed if it has no edge to any vertex in the class
                if (vm & color_class_masks[c]) == 0:
                    color_class_masks[c] |= bit
                    color_class_vertices[c].append(v)
                    placed = True
                    break
            if not placed:
                color_class_masks.append(bit)
                color_class_vertices.append([v])

        ordered: List[int] = []
        bounds: List[int] = []
        color = 0
        for cls in color_class_vertices:
            color += 1
            ordered.extend(cls)
            bounds.extend([color] * len(cls))

        return ordered, bounds

    def expand(self, C: List[int], P: List[int]):
        if self.timeout:
            return

        self.nodes += 1
        if (self.nodes & 1023) == 0 and self.time_up():
            self.timeout = True
            return

        if not P:
            self.update_incumbent(C)
            return

        ordered, bounds = self.color_sort(P)

        for i in range(len(ordered) - 1, -1, -1):
            # Upper bound pruning
            if len(C) + bounds[i] <= self.best_size:
                return

            if self.time_up():
                self.timeout = True
                return

            v = ordered[i]
            C.append(v)

            vm = self.adj[v]
            newP = [u for u in ordered[:i] if (vm >> u) & 1]

            # Simple cardinality pruning
            if len(C) + len(newP) > self.best_size:
                if newP:
                    self.expand(C, newP)
                else:
                    self.update_incumbent(C)

            C.pop()

    def solve(self):
        if self.n == 0:
            return

        self.initial_heuristics()
        if self.time_up():
            return

        # Initial candidate ordering by descending degree
        P0 = sorted(range(self.n), key=lambda x: self.deg[x], reverse=True)

        # Static filter: vertices that could potentially beat incumbent
        if self.best_size > 1:
            P0 = [v for v in P0 if self.deg[v] + 1 > self.best_size]

        self.expand([], P0)


def main():
    parser = argparse.ArgumentParser(description="Maximum Clique Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, required=False, default=None, help="Optional JSONL log path")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["num_vertices"])
    adj_masks = build_adjacency_masks(instance)

    start_time = time.time()
    deadline = start_time + max(0, int(args.time_limit))

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    sys.setrecursionlimit(max(10000, n + 1000))
    solver = MaxCliqueSolver(adj_masks, deadline, logger)
    solver.solve()

    clique_vertices = sorted(solver.best_clique)
    solution = {
        "objective_value": int(len(clique_vertices)),
        "clique_vertices": clique_vertices,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()