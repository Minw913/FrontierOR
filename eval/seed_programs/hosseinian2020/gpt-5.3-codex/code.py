import argparse
import json
import random
import time
from typing import Dict, List, Set, Tuple

from solution_logger import SolutionLogger


class MaxWeightCliqueSolver:
    def __init__(self, instance: dict, time_limit: int, logger: SolutionLogger = None):
        self.logger = logger
        self.time_limit = max(0, int(time_limit))
        self.start_time = time.time()
        self.deadline = self.start_time + self.time_limit

        # ---------- Robust instance parsing ----------
        n_raw = instance.get("num_vertices", None)
        edges_raw = instance.get("edges", []) or []
        verts_raw = instance.get("vertices", None)

        # Build vertex labels robustly
        if isinstance(verts_raw, list) and len(verts_raw) > 0:
            self.vertices = list(verts_raw)
            self.n = len(self.vertices)
        else:
            if isinstance(n_raw, int) and n_raw > 0:
                # Default to 1..n (common in benchmark instances)
                self.n = n_raw
                self.vertices = list(range(1, self.n + 1))
            else:
                # Derive from edges if needed
                labels = set()
                for e in edges_raw:
                    if isinstance(e, dict):
                        if "i" in e:
                            labels.add(e["i"])
                        if "j" in e:
                            labels.add(e["j"])
                self.vertices = sorted(labels)
                self.n = len(self.vertices)

        self.idx_of = {v: i for i, v in enumerate(self.vertices)}

        # Graph structures
        self.adj: List[Set[int]] = [set() for _ in range(self.n)]
        self.w_adj: List[Dict[int, float]] = [dict() for _ in range(self.n)]

        # Parse edges
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            if "i" not in e or "j" not in e or "weight" not in e:
                continue

            u_lbl = e["i"]
            v_lbl = e["j"]

            # Robust weight parsing
            try:
                w = float(e["weight"])
            except Exception:
                continue
            if w <= 0:
                continue

            if u_lbl not in self.idx_of or v_lbl not in self.idx_of:
                continue

            u = self.idx_of[u_lbl]
            v = self.idx_of[v_lbl]
            if u == v:
                continue

            if v in self.w_adj[u]:
                # Aggregate duplicates
                self.w_adj[u][v] += w
                self.w_adj[v][u] += w
            else:
                self.w_adj[u][v] = w
                self.w_adj[v][u] = w
                self.adj[u].add(v)
                self.adj[v].add(u)

        # Unique edge list i<j
        self.edge_list_idx: List[Tuple[int, int, float]] = []
        for i in range(self.n):
            for j, w in self.w_adj[i].items():
                if i < j:
                    self.edge_list_idx.append((i, j, w))

        self.weighted_degree = [sum(self.w_adj[i].values()) for i in range(self.n)]

        self.best_obj = 0.0
        self.best_clique: List[int] = []

    def timed_out(self) -> bool:
        return time.time() >= self.deadline

    def time_left(self) -> float:
        return self.deadline - time.time()

    def clique_weight(self, clique: List[int]) -> float:
        if not clique:
            return 0.0
        s = set(clique)
        total = 0.0
        for i in clique:
            for j, w in self.w_adj[i].items():
                if j in s and i < j:
                    total += w
        return total

    def update_best(self, clique: List[int], obj: float):
        if obj > self.best_obj + 1e-9:
            self.best_obj = float(obj)
            self.best_clique = list(clique)
            if self.logger:
                self.logger.log(float(self.best_obj))

    def greedy_expand(self, seed: List[int], randomized: bool = False) -> Tuple[List[int], float]:
        if self.timed_out():
            return [], 0.0
        if not seed:
            return [], 0.0

        clique = list(seed)

        # Validate seed as clique
        for a in range(len(clique)):
            ua = clique[a]
            for b in range(a + 1, len(clique)):
                ub = clique[b]
                if ub not in self.adj[ua]:
                    return [], 0.0

        obj = self.clique_weight(clique)

        # Candidates adjacent to all clique members
        cand = None
        for u in clique:
            if cand is None:
                cand = set(self.adj[u])
            else:
                cand.intersection_update(self.adj[u])

        if cand is None:
            cand = set()
        for u in clique:
            cand.discard(u)

        while cand and not self.timed_out():
            scored = []
            for v in cand:
                gain = 0.0
                wv = self.w_adj[v]
                for u in clique:
                    gain += wv.get(u, 0.0)
                scored.append((gain, self.weighted_degree[v], v))

            if not scored:
                break
            scored.sort(reverse=True)

            if randomized:
                rcl_size = min(4, len(scored))
                _, _, chosen = random.choice(scored[:rcl_size])
            else:
                chosen = scored[0][2]

            # Add chosen
            gain = 0.0
            wc = self.w_adj[chosen]
            for u in clique:
                gain += wc.get(u, 0.0)

            clique.append(chosen)
            obj += gain

            cand.intersection_update(self.adj[chosen])
            cand.discard(chosen)

        return clique, obj

    def run_heuristics(self):
        if self.n == 0 or self.time_limit <= 0:
            return
        random.seed(0)

        # Time for heuristics
        heur_end = min(self.deadline, self.start_time + max(0.0, 0.8 * self.time_limit))

        order = sorted(range(self.n), key=lambda i: self.weighted_degree[i], reverse=True)
        seeds_v = order[: min(60, self.n)]
        if self.n > 0:
            seeds_v += random.sample(range(self.n), k=min(60, self.n))

        # Deduplicate seeds
        seen = set()
        dedup_seeds_v = []
        for s in seeds_v:
            if s not in seen:
                seen.add(s)
                dedup_seeds_v.append(s)

        sorted_edges = sorted(self.edge_list_idx, key=lambda t: t[2], reverse=True)
        seeds_e = sorted_edges[: min(80, len(sorted_edges))]

        # Deterministic from vertices
        for s in dedup_seeds_v:
            if time.time() >= heur_end:
                return
            c, obj = self.greedy_expand([s], randomized=False)
            self.update_best(c, obj)

        # Deterministic from edges
        for u, v, _ in seeds_e:
            if time.time() >= heur_end:
                return
            c, obj = self.greedy_expand([u, v], randomized=False)
            self.update_best(c, obj)

        # Randomized multi-start
        while time.time() < heur_end:
            if self.edge_list_idx and random.random() < 0.5:
                u, v, _ = random.choice(self.edge_list_idx)
                seed = [u, v]
            else:
                seed = [random.randrange(self.n)]
            c, obj = self.greedy_expand(seed, randomized=True)
            self.update_best(c, obj)

    def branch_and_bound_small(self):
        # Exact only for small n
        if self.n > 70 or self.timed_out():
            return

        P0 = list(range(self.n))
        g0 = {v: 0.0 for v in P0}
        current: List[int] = []
        best_ref = {"obj": self.best_obj}

        def upper_bound(P: List[int], g: Dict[int, float], cur_obj: float) -> float:
            ub = cur_obj + sum(g[v] for v in P)
            m = len(P)
            # Add all edges within P as relaxation
            for i in range(m):
                a = P[i]
                wa = self.w_adj[a]
                for j in range(i + 1, m):
                    ub += wa.get(P[j], 0.0)
            return ub

        def pick_vertex(P: List[int], g: Dict[int, float]) -> int:
            Pset = set(P)
            best_v = P[0]
            best_score = -1.0
            for v in P:
                score = g[v]
                for u, w in self.w_adj[v].items():
                    if u in Pset:
                        score += w
                if score > best_score:
                    best_score = score
                    best_v = v
            return best_v

        def dfs(P: List[int], g: Dict[int, float], cur_obj: float):
            if self.timed_out():
                raise TimeoutError

            if not P:
                if cur_obj > best_ref["obj"] + 1e-9:
                    best_ref["obj"] = cur_obj
                    self.update_best(current, cur_obj)
                return

            if upper_bound(P, g, cur_obj) <= best_ref["obj"] + 1e-9:
                return

            v = pick_vertex(P, g)

            # Include v
            P_in = [u for u in P if u != v and u in self.adj[v]]
            g_in = {}
            wv = self.w_adj[v]
            for u in P_in:
                g_in[u] = g[u] + wv.get(u, 0.0)

            current.append(v)
            dfs(P_in, g_in, cur_obj + g[v])
            current.pop()

            # Exclude v
            P_ex = [u for u in P if u != v]
            if P_ex:
                dfs(P_ex, g, cur_obj)
            else:
                if cur_obj > best_ref["obj"] + 1e-9:
                    best_ref["obj"] = cur_obj
                    self.update_best(current, cur_obj)

        try:
            dfs(P0, g0, 0.0)
        except TimeoutError:
            pass

    def solve(self) -> Tuple[float, List[int]]:
        if self.time_limit <= 0:
            return 0.0, []

        self.run_heuristics()

        if self.time_left() > 0.2:
            self.branch_and_bound_small()

        # Safety: ensure result is valid and objective consistent
        final_clique = self.best_clique if self.best_clique else []
        final_obj = self.clique_weight(final_clique)

        labels = [self.vertices[i] for i in final_clique]
        # Schema expects ints
        clique_out = [int(v) for v in labels]

        return float(final_obj), sorted(clique_out)


def main():
        parser = argparse.ArgumentParser(description="Maximum edge-weight clique solver")
        parser.add_argument("--instance_path", type=str, required=True)
        parser.add_argument("--solution_path", type=str, required=True)
        parser.add_argument("--time_limit", type=int, required=True)
        parser.add_argument("--log_path", type=str, default=None)
        args = parser.parse_args()

        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

        # Default fallback solution on any unexpected issue
        fallback = {"objective_value": 0.0, "clique": []}

        try:
            with open(args.instance_path, "r") as f:
                instance = json.load(f)

            solver = MaxWeightCliqueSolver(instance, args.time_limit, logger=logger)
            obj, clique = solver.solve()

            solution = {
                "objective_value": float(obj),
                "clique": clique
            }
        except Exception:
            solution = fallback

        with open(args.solution_path, "w") as f:
            json.dump(solution, f)


if __name__ == "__main__":
    main()