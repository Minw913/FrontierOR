import argparse
import heapq
import json
import sys
import time

from solution_logger import SolutionLogger


class SearchTimeout(Exception):
    pass


class SearchDone(Exception):
    pass


_popcount = getattr(int, "bit_count", None)
if _popcount is None:
    def popcount(x):
        return bin(x).count("1")
else:
    popcount = _popcount


def make_solution(clique):
    c = sorted(int(v) for v in clique)
    return {"objective_value": float(len(c)), "clique_vertices": c}


class BestTracker:
    def __init__(self, logger):
        self.logger = logger
        self.best_clique = []
        self.upper_bound = None

    @property
    def best_size(self):
        return len(self.best_clique)

    def consider(self, clique):
        sz = len(clique)
        if sz <= self.best_size:
            return False
        self.best_clique = sorted(int(v) for v in clique)
        if self.logger is not None:
            sol = {
                "objective_value": float(sz),
                "clique_vertices": list(self.best_clique),
            }
            self.logger.log_solution(float(sz), sol)
        if self.upper_bound is not None and sz >= self.upper_bound:
            raise SearchDone()
        return True


def solve_instance(data, deadline, logger):
    tracker = BestTracker(logger)

    try:
        n = int(data.get("num_vertices", len(data.get("vertices", []) or [])))
        if n <= 0:
            return []
        sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 1000))

        # A singleton is always feasible for a nonempty graph.
        tracker.consider([0])

        edges = data.get("edges", None)
        adj = [0] * n
        adj_lists = [[] for _ in range(n)]

        if edges is not None:
            for e in edges:
                a = int(e[0])
                b = int(e[1])
                if a == b:
                    continue
                if not (0 <= a < n and 0 <= b < n):
                    raise ValueError("edge endpoint out of range")
                bit_b = 1 << b
                bit_a = 1 << a
                if (adj[a] & bit_b) == 0:
                    adj[a] |= bit_b
                    adj[b] |= bit_a
                    adj_lists[a].append(b)
                    adj_lists[b].append(a)
        else:
            raw_adj = data.get("adjacency_list", {}) or {}
            for k, neigh in raw_adj.items():
                v = int(k)
                if not (0 <= v < n):
                    continue
                for uu in neigh:
                    u = int(uu)
                    if u == v or not (0 <= u < n):
                        continue
                    bit = 1 << u
                    if (adj[v] & bit) == 0:
                        adj[v] |= bit
                        adj_lists[v].append(u)

        all_mask = (1 << n) - 1
        num_edges_computed = sum(popcount(adj[v]) for v in range(n)) // 2

        def check_time():
            if time.monotonic() >= deadline:
                raise SearchTimeout()

        def valid_clique(clique):
            seen = set()
            mask = 0
            for x in clique:
                v = int(x)
                if v in seen or v < 0 or v >= n:
                    return False
                seen.add(v)
                mask |= 1 << v
            for v in seen:
                others = mask & ~(1 << v)
                if others & ~adj[v]:
                    return False
            return True

        def greedy_degree_clique():
            deg = [len(adj_lists[v]) for v in range(n)]
            order = sorted(range(n), key=lambda v: (-deg[v], v))
            clique = []
            cand = all_mask
            for v in order:
                if (cand >> v) & 1:
                    clique.append(v)
                    cand &= adj[v]
                    if cand == 0:
                        break
            return clique

        def grow_clique(start):
            clique = list(dict.fromkeys(int(v) for v in start))
            mask = 0
            for v in clique:
                mask |= 1 << v
            common = all_mask
            for v in clique:
                common &= adj[v]
            cand = common & ~mask

            while cand:
                if time.monotonic() >= deadline:
                    tracker.consider(clique)
                    raise SearchTimeout()

                if popcount(cand) <= 32:
                    best_v = -1
                    best_score = -1
                    x = cand
                    while x:
                        lsb = x & -x
                        v = lsb.bit_length() - 1
                        x ^= lsb
                        score = popcount(adj[v] & cand)
                        if score > best_score:
                            best_score = score
                            best_v = v
                    v = best_v
                else:
                    lsb = cand & -cand
                    v = lsb.bit_length() - 1

                clique.append(v)
                bit = 1 << v
                mask |= bit
                cand &= adj[v]
                cand &= ~mask

            tracker.consider(clique)
            return clique

        def find_swap(clique):
            # Try to replace one clique vertex u by two outside vertices v,w
            # such that v,w are adjacent and both are adjacent to clique\{u}.
            mask = 0
            for v in clique:
                mask |= 1 << v
            outside = all_mask & ~mask
            buckets = {}

            x = outside
            cnt = 0
            while x:
                cnt += 1
                if (cnt & 255) == 0:
                    check_time()
                lsb = x & -x
                v = lsb.bit_length() - 1
                x ^= lsb

                missing = mask & ~adj[v]
                pc = popcount(missing)
                if pc == 0:
                    return list(clique) + [v]
                if pc == 1:
                    u = (missing & -missing).bit_length() - 1
                    buckets[u] = buckets.get(u, 0) | lsb

            for u, bmask in buckets.items():
                y = bmask
                while y:
                    lsb = y & -y
                    v = lsb.bit_length() - 1
                    y ^= lsb
                    wm = adj[v] & bmask
                    if wm:
                        w = (wm & -wm).bit_length() - 1
                        if w != v:
                            return [z for z in clique if z != u] + [v, w]
            return None

        def local_improve(start):
            clique = grow_clique(start)
            while True:
                check_time()
                nxt = find_swap(clique)
                if nxt is None:
                    return clique
                new_clique = grow_clique(nxt)
                if len(new_clique) <= len(clique):
                    return clique
                clique = new_clique

        def degeneracy():
            deg = [len(adj_lists[v]) for v in range(n)]
            removed = bytearray(n)
            heap = [(deg[v], v) for v in range(n)]
            heapq.heapify(heap)
            max_core = 0
            pops = 0
            while heap:
                pops += 1
                if (pops & 1023) == 0:
                    check_time()
                d, v = heapq.heappop(heap)
                if removed[v] or d != deg[v]:
                    continue
                removed[v] = 1
                if d > max_core:
                    max_core = d
                for u in adj_lists[v]:
                    if not removed[u]:
                        deg[u] -= 1
                        heapq.heappush(heap, (deg[u], u))
            return max_core

        def kcore_mask(threshold):
            if threshold <= 0:
                return all_mask
            deg = [len(adj_lists[v]) for v in range(n)]
            removed = bytearray(n)
            stack = []
            for v, d in enumerate(deg):
                if d < threshold:
                    removed[v] = 1
                    stack.append(v)
            steps = 0
            while stack:
                steps += 1
                if (steps & 1023) == 0:
                    check_time()
                v = stack.pop()
                for u in adj_lists[v]:
                    if not removed[u]:
                        deg[u] -= 1
                        if deg[u] < threshold:
                            removed[u] = 1
                            stack.append(u)
            mask = 0
            for v in range(n):
                if not removed[v]:
                    mask |= 1 << v
            return mask

        node_count = 0

        def expand(cur, cand):
            nonlocal node_count
            node_count += 1
            if (node_count & 1023) == 0:
                check_time()

            cur_size = len(cur)
            if cur_size > tracker.best_size:
                tracker.consider(cur)

            if cand == 0:
                return
            if cur_size + popcount(cand) <= tracker.best_size:
                return

            # Greedy coloring of the candidate subgraph. Vertices in the same
            # color class are pairwise non-adjacent, so the color count bounds
            # any additional clique vertices obtainable from cand.
            order = []
            colors = []
            uncolored = cand
            color = 0
            while uncolored:
                color += 1
                avail = uncolored
                while avail:
                    lsb = avail & -avail
                    v = lsb.bit_length() - 1
                    order.append(v)
                    colors.append(color)
                    uncolored &= ~lsb
                    avail &= ~lsb
                    avail &= ~adj[v]
                    if (len(order) & 63) == 0 and time.monotonic() >= deadline:
                        raise SearchTimeout()

            rem = cand
            for i in range(len(order) - 1, -1, -1):
                if cur_size + colors[i] <= tracker.best_size:
                    return
                v = order[i]
                bit = 1 << v
                if rem & bit:
                    cur.append(v)
                    expand(cur, rem & adj[v])
                    cur.pop()
                    rem &= ~bit

        greedy_given = data.get("greedy_clique") or []
        if valid_clique(greedy_given):
            tracker.consider(greedy_given)
        tracker.consider(greedy_degree_clique())
        tracker.consider(local_improve(tracker.best_clique))

        if num_edges_computed == n * (n - 1) // 2:
            tracker.consider(range(n))
            return tracker.best_clique

        upper = degeneracy() + 1
        tracker.upper_bound = upper
        if tracker.best_size >= upper:
            return tracker.best_clique

        # Any clique larger than the current incumbent of size B must lie in
        # the B-core: every vertex of a (B+1)-clique has degree at least B.
        root = kcore_mask(tracker.best_size)
        if root == 0 or popcount(root) < tracker.best_size + 1:
            return tracker.best_clique

        expand([], root)

    except (SearchTimeout, SearchDone):
        pass

    return tracker.best_clique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    deadline = time.monotonic() + max(0, int(args.time_limit))

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    best = solve_instance(data, deadline, logger)
    solution = make_solution(best)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()