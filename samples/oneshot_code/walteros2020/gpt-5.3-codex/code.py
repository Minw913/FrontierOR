import argparse
import json
import time
import sys
from typing import List, Optional

from solution_logger import SolutionLogger


class MaxCliqueSolver:
    def __init__(self, n: int, edges: List[List[int]], time_limit: int, logger: Optional[SolutionLogger] = None):
        self.n = n
        self.edges = edges
        self.time_limit = max(0, int(time_limit))
        self.logger = logger

        self.all_mask = (1 << n) - 1 if n > 0 else 0
        self.adj = [0] * n
        self.adj_list = [[] for _ in range(n)]
        for u, v in edges:
            if u == v:
                continue
            self.adj[u] |= (1 << v)
            self.adj[v] |= (1 << u)
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

        self.deg = [len(self.adj_list[v]) for v in range(n)]
        self.vertices_by_deg = sorted(range(n), key=lambda x: self.deg[x], reverse=True)

        self.best_size = 0
        self.best_clique: List[int] = []

        self.start_time = 0.0
        self.deadline = 0.0
        self.timed_out = False
        self._check_counter = 0

    def time_exceeded(self) -> bool:
        self._check_counter += 1
        if (self._check_counter & 1023) == 0:
            if time.perf_counter() >= self.deadline:
                self.timed_out = True
                return True
        return False

    def update_best(self, clique: List[int]):
        sz = len(clique)
        if sz > self.best_size:
            self.best_size = sz
            self.best_clique = clique.copy()
            if self.logger:
                self.logger.log(self.best_size)

    def greedy_clique_from_candidates(self, cand: int) -> List[int]:
        clique = []
        while cand:
            chosen = -1
            for v in self.vertices_by_deg:
                if (cand >> v) & 1:
                    chosen = v
                    break
            if chosen < 0:
                break
            clique.append(chosen)
            cand &= self.adj[chosen]
        return clique

    def run_heuristics(self):
        if self.n == 0:
            return
        # Quick constructive heuristic
        self.update_best(self.greedy_clique_from_candidates(self.all_mask))

        # Seeded greedies on top-degree vertices, limited budget
        heuristic_deadline = min(self.deadline, self.start_time + max(0.2, 0.15 * self.time_limit))
        max_seeds = min(self.n, 40)
        for i in range(max_seeds):
            if time.perf_counter() >= heuristic_deadline:
                break
            v = self.vertices_by_deg[i]
            clique = [v]
            cand = self.adj[v]
            while cand:
                chosen = -1
                for u in self.vertices_by_deg:
                    if (cand >> u) & 1:
                        chosen = u
                        break
                if chosen < 0:
                    break
                clique.append(chosen)
                cand &= self.adj[chosen]
            self.update_best(clique)

    def connected_components_bitsets(self) -> List[int]:
        visited = [False] * self.n
        comps = []
        for s in range(self.n):
            if visited[s]:
                continue
            stack = [s]
            visited[s] = True
            comp_vertices = []
            while stack:
                v = stack.pop()
                comp_vertices.append(v)
                for u in self.adj_list[v]:
                    if not visited[u]:
                        visited[u] = True
                        stack.append(u)
            bits = 0
            for v in comp_vertices:
                bits |= (1 << v)
            comps.append(bits)
        comps.sort(key=lambda b: b.bit_count(), reverse=True)
        return comps

    def color_sort(self, P: int):
        order = []
        bounds = []
        uncolored = P
        color = 0
        while uncolored:
            color += 1
            Q = uncolored
            while Q:
                b = Q & -Q
                v = b.bit_length() - 1
                order.append(v)
                bounds.append(color)
                uncolored ^= b
                Q ^= b
                Q &= (~self.adj[v]) & self.all_mask
        return order, bounds

    def expand(self, C: List[int], P: int):
        if self.timed_out:
            return
        if time.perf_counter() >= self.deadline:
            self.timed_out = True
            return

        if len(C) + P.bit_count() <= self.best_size:
            return

        order, bounds = self.color_sort(P)

        for i in range(len(order) - 1, -1, -1):
            if self.time_exceeded():
                return

            if len(C) + bounds[i] <= self.best_size:
                return

            v = order[i]
            v_bit = 1 << v

            C.append(v)
            P_next = P & self.adj[v]

            if P_next == 0:
                self.update_best(C)
            else:
                if len(C) + P_next.bit_count() > self.best_size:
                    self.expand(C, P_next)

            C.pop()
            P &= ~v_bit

            if len(C) + P.bit_count() <= self.best_size:
                return

    def solve(self):
        self.start_time = time.perf_counter()
        self.deadline = self.start_time + self.time_limit

        if self.n == 0:
            return 0, []

        # Trivial incumbent
        self.update_best([self.vertices_by_deg[0]] if self.n > 0 else [])

        self.run_heuristics()
        if self.timed_out or time.perf_counter() >= self.deadline:
            return self.best_size, sorted(self.best_clique)

        components = self.connected_components_bitsets()

        for comp in components:
            if self.timed_out:
                break
            comp_size = comp.bit_count()
            if comp_size <= self.best_size:
                continue
            self.expand([], comp)

        return self.best_size, sorted(self.best_clique)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["n"])
    edges = instance.get("edges", [])

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    sys.setrecursionlimit(max(10000, n + 100))

    solver = MaxCliqueSolver(n=n, edges=edges, time_limit=args.time_limit, logger=logger)
    best_size, best_vertices = solver.solve()

    solution = {
        "objective_value": int(best_size),
        "clique_vertices": [int(v) for v in best_vertices],
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()