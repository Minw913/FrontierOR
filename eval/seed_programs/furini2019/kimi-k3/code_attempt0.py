import argparse
import heapq
import json
import random
import sys
import time

from solution_logger import SolutionLogger

if hasattr(int, "bit_count"):
    popcount = int.bit_count
else:
    def popcount(x):
        return bin(x).count("1")


class TimeUp(Exception):
    """Raised when the global time budget has been exhausted."""


def parse_args():
    parser = argparse.ArgumentParser(description="Maximum clique solver")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional JSONL path for incumbent logging.")
    return parser.parse_args()


def degeneracy_order(adj, n):
    """Return (degeneracy ordering, degeneracy) of the graph."""
    deg = [popcount(adj[v]) for v in range(n)]
    removed = [False] * n
    heap = [(deg[v], v) for v in range(n)]
    heapq.heapify(heap)
    order = []
    degeneracy = 0
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v] or d != deg[v]:
            continue
        removed[v] = True
        if d > degeneracy:
            degeneracy = d
        order.append(v)
        bits = adj[v]
        while bits:
            b = bits & -bits
            u = b.bit_length() - 1
            if not removed[u]:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))
            bits ^= b
    return order, degeneracy


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    t0 = time.perf_counter()
    deadline = t0 + max(0.05, float(args.time_limit) - 0.25)

    # ---------------- read instance ----------------
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data.get("num_vertices", 0))
    adj = [0] * n  # adjacency bitsets stored as Python integers

    edges = data.get("edges")
    if edges is None:
        edges = []
        for u, nbrs in enumerate(data.get("adjacency_list", [])):
            for v in nbrs:
                edges.append((u, v))
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    sys.setrecursionlimit(max(10000, n + 1000))

    best = []  # best clique found so far (list of vertex ids)

    def record(clique):
        nonlocal best
        best = list(clique)
        if logger:
            sol = {
                "objective_value": len(best),
                "clique_vertices": sorted(best),
            }
            logger.log_solution(len(best), sol)

    def check_time():
        if time.perf_counter() > deadline:
            raise TimeUp()

    if n > 0:
        order, degeneracy = degeneracy_order(adj, n)
    else:
        order, degeneracy = [], 0
    ub_global = (degeneracy + 1) if n > 0 else 0

    # ---------------- greedy heuristic phase ----------------
    h_budget = min(deadline,
                   t0 + max(0.1, min(0.2 * max(1, args.time_limit), 5.0)))

    def greedy(start_v, randomized=False, rng=None, adj=adj):
        clique = [start_v]
        cand = adj[start_v]
        cnt = 0
        while cand:
            scores = []
            tmp = cand
            while tmp:
                b = tmp & -tmp
                u = b.bit_length() - 1
                scores.append((popcount(adj[u] & cand), u))
                tmp ^= b
                cnt += 1
                if cnt >= 512:
                    cnt = 0
                    if time.perf_counter() > h_budget:
                        return clique
            if randomized and rng is not None and len(scores) > 1:
                scores.sort(reverse=True)
                u = rng.choice(scores[: min(3, len(scores))])[1]
            else:
                u = max(scores)[1]
            clique.append(u)
            cand &= adj[u]
        return clique

    try:
        if n > 0 and len(best) < ub_global:
            deg_sorted = sorted(range(n), key=lambda v: -popcount(adj[v]))
            for s in deg_sorted:
                if len(best) >= ub_global or time.perf_counter() > h_budget:
                    break
                c = greedy(s)
                if len(c) > len(best):
                    record(c)
            rng = random.Random(0)
            top = deg_sorted[: max(1, min(n, 64))]
            while len(best) < ub_global and time.perf_counter() < h_budget:
                s = rng.choice(top)
                c = greedy(s, randomized=True, rng=rng)
                if len(c) > len(best):
                    record(c)
    except TimeUp:
        pass

    if n > 0 and not best:
        record([order[-1]])

    # ------------- exact bit-parallel branch and bound -------------
    # Tomita-style search with greedy-graph-coloring upper bounds.
    node_count = 0

    def color_sort(P, adj=adj):
        order_v = []
        colors = []
        uncolored = P
        color = 0
        steps = 0
        while uncolored:
            color += 1
            avail = uncolored
            while avail:
                b = avail & -avail
                v = b.bit_length() - 1
                order_v.append(v)
                colors.append(color)
                uncolored ^= b
                avail &= ~b
                avail &= ~adj[v]
                steps += 1
                if steps >= 4096:
                    steps = 0
                    check_time()
        return order_v, colors

    def expand(R, P, adj=adj):
        nonlocal node_count
        node_count += 1
        if node_count >= 1024:
            node_count = 0
            check_time()
        order_v, colors = color_sort(P)
        rlen = len(R)
        for k in range(len(order_v) - 1, -1, -1):
            v = order_v[k]
            if rlen + colors[k] <= len(best):
                return
            newP = P & adj[v]
            if newP:
                R.append(v)
                expand(R, newP)
                R.pop()
            elif rlen + 1 > len(best):
                record(R + [v])
            P &= ~(1 << v)

    try:
        later = 0
        # Process vertices in reverse degeneracy order; vertex v is combined
        # only with neighbors that come later in the order, partitioning the
        # search space by the earliest vertex of the clique.
        for i in range(n - 1, -1, -1):
            if len(best) >= ub_global or len(best) >= i + 1:
                break
            v = order[i]
            P = adj[v] & later
            later |= 1 << v
            if not P:
                continue
            if popcount(P) + 1 <= len(best):
                continue
            expand([v], P)
    except TimeUp:
        pass

    # ---------------- write final solution ----------------
    solution = {
        "objective_value": len(best),
        "clique_vertices": sorted(best),
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()