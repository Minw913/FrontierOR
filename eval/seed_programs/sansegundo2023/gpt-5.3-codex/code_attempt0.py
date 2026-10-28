import argparse
import json
import time
import sys
from typing import List, Dict, Optional

from solution_logger import SolutionLogger


class TimeoutException(Exception):
    pass


def iter_bits(x: int):
    while x:
        lsb = x & -x
        yield lsb.bit_length() - 1
        x ^= lsb


class MaxCliqueSolver:
    def __init__(self, adj_masks: List[int], time_limit: int, logger: Optional[SolutionLogger] = None):
        self.adj = adj_masks
        self.n = len(adj_masks)
        self.all_mask = (1 << self.n) - 1
        self.time_limit = max(0, int(time_limit))
        self.deadline = time.perf_counter() + self.time_limit
        self.logger = logger

        self.best_size = 0
        self.best_clique: List[int] = []
        self._check_counter = 0

    def _time_check(self):
        self._check_counter += 1
        if (self._check_counter & 1023) == 0:
            if time.perf_counter() >= self.deadline:
                raise TimeoutException

    def set_incumbent(self, clique: List[int]):
        if len(clique) > self.best_size:
            self.best_size = len(clique)
            self.best_clique = clique.copy()
            if self.logger:
                self.logger.log(float(self.best_size))

    def color_sort(self, P: int):
        """
        Greedy coloring bound on subgraph induced by P.
        Returns:
          order: vertices in coloring generation order
          bounds: color index assigned to each order vertex
        """
        order = []
        bounds = []
        U = P
        color = 0
        while U:
            color += 1
            Q = U
            while Q:
                lsb = Q & -Q
                v = lsb.bit_length() - 1
                order.append(v)
                bounds.append(color)
                U ^= lsb
                Q ^= lsb
                Q &= ~self.adj[v]  # keep vertices non-adjacent to v for same color class
        return order, bounds

    def expand(self, P: int, current: List[int]):
        self._time_check()

        if not P:
            self.set_incumbent(current)
            return

        # Quick bound
        if len(current) + P.bit_count() <= self.best_size:
            return

        order, bounds = self.color_sort(P)

        # Traverse in reverse so larger color bound considered first (Tomita-style)
        for i in range(len(order) - 1, -1, -1):
            v = order[i]
            if len(current) + bounds[i] <= self.best_size:
                return

            vbit = 1 << v
            if not (P & vbit):
                continue

            current.append(v)
            newP = P & self.adj[v]

            if not newP:
                self.set_incumbent(current)
            else:
                self.expand(newP, current)

            current.pop()
            P ^= vbit  # remove v from candidate set

            if len(current) + P.bit_count() <= self.best_size:
                return

            self._time_check()

    def greedy_construct(self, start_v: Optional[int] = None) -> List[int]:
        if start_v is None:
            cand = self.all_mask
            clique = []
        else:
            clique = [start_v]
            cand = self.adj[start_v]

        while cand:
            # pick vertex with maximum degree inside current candidate set
            best_v = -1
            best_score = -1
            x = cand
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                score = (self.adj[v] & cand).bit_count()
                if score > best_score:
                    best_score = score
                    best_v = v
                x ^= lsb

            clique.append(best_v)
            cand &= self.adj[best_v]

        return clique

    def solve(self, initial_clique: Optional[List[int]] = None):
        # seed incumbent
        if initial_clique:
            self.set_incumbent(initial_clique)

        # fast multi-start greedy
        try:
            if time.perf_counter() < self.deadline:
                self.set_incumbent(self.greedy_construct(None))

            max_starts = min(self.n, 25)
            for s in range(max_starts):
                if time.perf_counter() >= self.deadline:
                    raise TimeoutException
                self.set_incumbent(self.greedy_construct(s))

            # exact branch-and-bound
            if time.perf_counter() < self.deadline:
                self.expand(self.all_mask, [])
        except TimeoutException:
            pass

        return self.best_size, self.best_clique


def build_adjacency_masks(data: Dict) -> List[int]:
    n = int(data["num_vertices"])
    adj_sets = [set() for _ in range(n)]

    if "adjacency_list" in data and data["adjacency_list"] is not None:
        for k, neighs in data["adjacency_list"].items():
            u = int(k)
            for v in neighs:
                v = int(v)
                if v != u:
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
    else:
        for e in data.get("edges", []):
            u, v = int(e[0]), int(e[1])
            if u != v:
                adj_sets[u].add(v)
                adj_sets[v].add(u)

    adj_masks = [0] * n
    for u in range(n):
        m = 0
        for v in adj_sets[u]:
            m |= 1 << v
        adj_masks[u] = m
    return adj_masks


def relabel_by_degree(adj_masks_old: List[int]):
    n = len(adj_masks_old)
    degrees = [m.bit_count() for m in adj_masks_old]
    new_to_old = sorted(range(n), key=lambda u: degrees[u], reverse=True)
    old_to_new = [0] * n
    for new_idx, old_idx in enumerate(new_to_old):
        old_to_new[old_idx] = new_idx

    adj_masks_new = [0] * n
    for new_u, old_u in enumerate(new_to_old):
        m_old = adj_masks_old[old_u]
        m_new = 0
        x = m_old
        while x:
            lsb = x & -x
            old_v = lsb.bit_length() - 1
            m_new |= 1 << old_to_new[old_v]
            x ^= lsb
        adj_masks_new[new_u] = m_new

    return adj_masks_new, old_to_new, new_to_old


def map_clique_old_to_new(clique_old: List[int], old_to_new: List[int]) -> List[int]:
    return [old_to_new[v] for v in clique_old if 0 <= v < len(old_to_new)]


def main():
    parser = argparse.ArgumentParser(description="Maximum Clique Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    adj_old = build_adjacency_masks(data)
    adj_new, old_to_new, new_to_old = relabel_by_degree(adj_old)

    initial_clique_new = None
    greedy_clique_old = data.get("greedy_clique", None)
    if isinstance(greedy_clique_old, list) and len(greedy_clique_old) > 0:
        initial_clique_new = map_clique_old_to_new([int(v) for v in greedy_clique_old], old_to_new)

    solver = MaxCliqueSolver(adj_new, args.time_limit, logger=logger)
    best_size, best_clique_new = solver.solve(initial_clique=initial_clique_new)

    best_clique_old = [new_to_old[v] for v in best_clique_new]
    best_clique_old.sort()

    result = {
        "objective_value": float(best_size),
        "clique_vertices": best_clique_old,
    }

    with open(args.solution_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()