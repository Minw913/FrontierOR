import argparse
import json
import time
from typing import List, Tuple, Dict, Optional

from solution_logger import SolutionLogger


class TimeLimitExceeded(Exception):
    pass


class MaxCliqueSolver:
    def __init__(
        self,
        adj_bits: List[int],
        idx_to_label: List[int],
        time_limit: int,
        logger: Optional[SolutionLogger] = None,
    ):
        self.adj_bits = adj_bits
        self.n = len(adj_bits)
        self.idx_to_label = idx_to_label
        self.logger = logger

        self.start_time = time.perf_counter()
        self.deadline = self.start_time + max(0, time_limit)

        self.degrees = [a.bit_count() for a in adj_bits]

        self.best_size = 0
        self.best_clique: List[int] = []

        self.node_counter = 0

    def time_up(self) -> bool:
        return time.perf_counter() >= self.deadline

    def _check_time(self):
        if self.time_up():
            raise TimeLimitExceeded()

    def _update_best(self, clique_idx: List[int]):
        csz = len(clique_idx)
        if csz > self.best_size:
            self.best_size = csz
            self.best_clique = clique_idx.copy()
            if self.logger:
                self.logger.log(self.best_size)

    def greedy_initial_solutions(self):
        if self.n == 0:
            return

        # 1) Greedy by degree descending
        order = sorted(range(self.n), key=lambda v: self.degrees[v], reverse=True)
        clique = []
        clique_mask = 0
        for v in order:
            if clique_mask == 0 or (self.adj_bits[v] & clique_mask) == clique_mask:
                clique.append(v)
                clique_mask |= (1 << v)
        self._update_best(clique)

        # 2) Multi-start greedy expansion from top-degree vertices
        top_k = min(self.n, 60)
        for s in order[:top_k]:
            if self.time_up():
                return
            c = [s]
            cand = self.adj_bits[s]
            while cand:
                # Choose candidate with largest connectivity inside cand
                best_v = -1
                best_score = -1
                q = cand
                while q:
                    b = q & -q
                    v = b.bit_length() - 1
                    q ^= b
                    score = (self.adj_bits[v] & cand).bit_count()
                    if score > best_score:
                        best_score = score
                        best_v = v
                c.append(best_v)
                cand &= self.adj_bits[best_v]
            self._update_best(c)

    def color_sort(self, p: int) -> Tuple[List[int], List[int]]:
        """Sequential greedy coloring. Returns (vertices, color_bound_per_vertex)."""
        verts: List[int] = []
        colors: List[int] = []
        uncolored = p
        color = 0

        while uncolored:
            color += 1
            q = uncolored
            while q:
                b = q & -q
                v = b.bit_length() - 1
                verts.append(v)
                colors.append(color)

                uncolored ^= b
                q ^= b
                q &= ~self.adj_bits[v]

        return verts, colors

    def expand(self, p: int, r: List[int]):
        self.node_counter += 1
        if (self.node_counter & 1023) == 0:
            self._check_time()

        if p == 0:
            self._update_best(r)
            return

        verts, colors = self.color_sort(p)

        # Branch in reverse order (larger color bounds first)
        for i in range(len(verts) - 1, -1, -1):
            if len(r) + colors[i] <= self.best_size:
                return

            v = verts[i]
            vb = 1 << v
            if (p & vb) == 0:
                continue

            r.append(v)
            new_p = p & self.adj_bits[v]

            if new_p == 0:
                self._update_best(r)
            else:
                self.expand(new_p, r)

            r.pop()
            p &= ~vb

            if (self.node_counter & 2047) == 0:
                self._check_time()

    def solve(self) -> Tuple[int, List[int]]:
        # Fast feasible solution(s)
        self.greedy_initial_solutions()

        # Exact BnB (until time limit)
        try:
            if not self.time_up():
                all_vertices = (1 << self.n) - 1
                self.expand(all_vertices, [])
        except TimeLimitExceeded:
            pass

        clique_labels = sorted(self.idx_to_label[i] for i in self.best_clique)
        return self.best_size, clique_labels


def build_graph(instance: Dict) -> Tuple[List[int], List[int]]:
    graph = instance.get("graph", {})
    adj_raw = graph.get("adjacency_list", {})
    n_schema = int(graph.get("num_vertices", 0))

    labels = set()
    for k, nbrs in adj_raw.items():
        try:
            labels.add(int(k))
        except Exception:
            continue
        for nb in nbrs:
            labels.add(int(nb))

    if not labels and n_schema > 0:
        labels = set(range(n_schema))
    elif len(labels) < n_schema:
        # Try to infer missing isolated vertices
        if labels.issubset(set(range(n_schema))):
            labels.update(range(n_schema))
        elif labels.issubset(set(range(1, n_schema + 1))):
            labels.update(range(1, n_schema + 1))

    idx_to_label = sorted(labels)
    label_to_idx = {lab: i for i, lab in enumerate(idx_to_label)}
    n = len(idx_to_label)

    adj_bits = [0] * n

    # Build edges from adjacency list; enforce undirected
    for u_lab in idx_to_label:
        nbrs = adj_raw.get(str(u_lab), adj_raw.get(u_lab, []))
        u = label_to_idx[u_lab]
        for nb in nbrs:
            v_lab = int(nb)
            if v_lab not in label_to_idx or v_lab == u_lab:
                continue
            v = label_to_idx[v_lab]
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)

    return adj_bits, idx_to_label


def main():
    parser = argparse.ArgumentParser(description="Maximum Clique Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    adj_bits, idx_to_label = build_graph(instance)

    solver = MaxCliqueSolver(
        adj_bits=adj_bits,
        idx_to_label=idx_to_label,
        time_limit=args.time_limit,
        logger=logger,
    )
    obj, clique_vertices = solver.solve()

    solution = {
        "objective_value": int(obj),
        "clique_vertices": [int(v) for v in clique_vertices],
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()