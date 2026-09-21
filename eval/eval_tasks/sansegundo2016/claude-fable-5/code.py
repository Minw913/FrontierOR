import argparse
import json
import sys
import time
import random

from solution_logger import SolutionLogger

try:
    _bc = int.bit_count  # Python 3.10+
    def popcount(x):
        return x.bit_count()
except AttributeError:
    def popcount(x):
        return bin(x).count("1")


class TimeUp(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)
    g = data["graph"]
    al = g.get("adjacency_list", {}) or {}

    # Collect vertex labels
    labels = set()
    for k, vs in al.items():
        labels.add(int(k))
        for x in vs:
            labels.add(int(x))
    if not labels:
        nv = int(g.get("num_vertices", 0) or 0)
        labels = set(range(nv))
    labels = sorted(labels)
    n = len(labels)

    if n == 0:
        sol = {"objective_value": 0, "clique_vertices": []}
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    idx = {l: i for i, l in enumerate(labels)}
    sets = [set() for _ in range(n)]
    for k, vs in al.items():
        u = idx[int(k)]
        for x in vs:
            v = idx[int(x)]
            if v != u:
                sets[u].add(v)
                sets[v].add(u)

    # ---- Degeneracy ordering (bucket queue) ----
    deg = [len(sets[u]) for u in range(n)]
    maxd = max(deg) if n else 0
    buckets = [[] for _ in range(maxd + 2)]
    for u in range(n):
        buckets[deg[u]].append(u)
    removed = [False] * n
    curdeg = deg[:]
    order = []
    dptr = 0
    for _ in range(n):
        while True:
            while dptr <= maxd and not buckets[dptr]:
                dptr += 1
            u = buckets[dptr].pop()
            if not removed[u] and curdeg[u] == dptr:
                break
        removed[u] = True
        order.append(u)
        for v in sets[u]:
            if not removed[v]:
                curdeg[v] -= 1
                buckets[curdeg[v]].append(v)
                if curdeg[v] < dptr:
                    dptr = curdeg[v]

    # Relabel: last removed (highest core) -> lowest internal index
    newid = [0] * n
    for pos, u in enumerate(order):
        newid[u] = n - 1 - pos
    lbl = [0] * n  # internal index -> original label
    for u in range(n):
        lbl[newid[u]] = labels[u]

    adj = [0] * n
    for u in range(n):
        nu = newid[u]
        m = 0
        for v in sets[u]:
            m |= 1 << newid[v]
        adj[nu] = m

    best = 0
    best_clique = []  # internal indices

    def record(clique):
        nonlocal best, best_clique
        best = len(clique)
        best_clique = list(clique)
        sol = {
            "objective_value": best,
            "clique_vertices": sorted(lbl[v] for v in best_clique),
        }
        if logger:
            logger.log_solution(best, sol)

    # trivial start
    record([0])

    # ---- Heuristic: multi-start greedy ----
    rng = random.Random(0)

    def greedy(start, randomize):
        c = [start]
        cand = adj[start]
        while cand:
            bv = -1
            bd = -1
            Q = cand
            while Q:
                b = Q & -Q
                Q ^= b
                v = b.bit_length() - 1
                d = popcount(cand & adj[v])
                if d > bd or (randomize and d == bd and rng.random() < 0.5):
                    bd = d
                    bv = v
            c.append(bv)
            cand &= adj[bv]
        return c

    h_end = min(deadline, t0 + max(1.0, 0.15 * (deadline - t0)))
    starts = sorted(range(n), key=lambda v: -popcount(adj[v]))
    si = 0
    while time.time() < h_end:
        if si < n:
            s = starts[si]
            randomize = False
            si += 1
        else:
            s = rng.randrange(n)
            randomize = True
        c = greedy(s, randomize)
        if len(c) > best:
            record(c)
        if si >= n and n <= 50:
            break  # tiny graphs: move on to exact quickly

    # ---- Exact branch & bound with greedy coloring bound (bitsets) ----
    sys.setrecursionlimit(1000000)
    node_counter = [0]

    def color_sort(P):
        order_v = []
        colors = []
        color = 0
        while P:
            color += 1
            Q = P
            while Q:
                b = Q & -Q
                v = b.bit_length() - 1
                Q &= ~(adj[v] | b)
                P &= ~b
                order_v.append(v)
                colors.append(color)
        return order_v, colors

    def expand(R, P):
        nonlocal best
        node_counter[0] += 1
        if (node_counter[0] & 1023) == 0 and time.time() > deadline:
            raise TimeUp()
        if len(R) + popcount(P) <= best:
            return
        order_v, colors = color_sort(P)
        for i in range(len(order_v) - 1, -1, -1):
            if len(R) + colors[i] <= best:
                return
            v = order_v[i]
            R.append(v)
            newP = P & adj[v]
            if newP:
                expand(R, newP)
            elif len(R) > best:
                record(R)
            R.pop()
            P &= ~(1 << v)

    optimal = False
    try:
        full = (1 << n) - 1
        expand([], full)
        optimal = True
    except TimeUp:
        pass
    except RecursionError:
        pass

    # If exact finished early and there is leftover time but not optimal (won't happen),
    # or exact was too slow — keep best found.

    sol = {
        "objective_value": best,
        "clique_vertices": sorted(lbl[v] for v in best_clique),
    }

    # sanity verification: ensure it really is a clique; fall back if not
    cvs = [None]
    internal = best_clique
    ok = True
    for i in range(len(internal)):
        for j in range(i + 1, len(internal)):
            if not (adj[internal[i]] >> internal[j]) & 1:
                ok = False
                break
        if not ok:
            break
    if not ok:
        sol = {"objective_value": 1, "clique_vertices": [lbl[0]]}

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()