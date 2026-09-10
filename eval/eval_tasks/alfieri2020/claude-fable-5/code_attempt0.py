import argparse
import json
import random
import time
import heapq

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def dsatur(n, adj):
    """DSATUR greedy coloring. Returns list of colors 0..k-1."""
    color = [-1] * n
    satset = [set() for _ in range(n)]
    deg = [len(a) for a in adj]
    heap = [(0, -deg[v], v) for v in range(n)]
    heapq.heapify(heap)
    colored = 0
    while colored < n:
        while True:
            negs, negd, v = heapq.heappop(heap)
            if color[v] != -1:
                continue
            cur = len(satset[v])
            if -negs != cur:
                heapq.heappush(heap, (-cur, negd, v))
                continue
            break
        c = 0
        s = satset[v]
        while c in s:
            c += 1
        color[v] = c
        colored += 1
        for u in adj[v]:
            if color[u] == -1 and c not in satset[u]:
                satset[u].add(c)
                heapq.heappush(heap, (-len(satset[u]), -deg[u], u))
    return color


def greedy_clique_lb(n, adj_sets, deadline):
    """Greedy heuristic clique to get a lower bound on chromatic number."""
    if n == 0:
        return 0
    deg = [len(adj_sets[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    best = 1
    tries = min(8, n)
    for i in range(tries):
        if time.time() > deadline:
            break
        start = order[i]
        clique = [start]
        cand = set(adj_sets[start])
        while cand:
            # pick candidate with max degree inside candidate set (approx by global deg)
            u = max(cand, key=lambda x: deg[x])
            clique.append(u)
            cand &= adj_sets[u]
        if len(clique) > best:
            best = len(clique)
    return best


def compress_colors(color):
    """Relabel colors to contiguous 0..k-1. Returns (new_color, k)."""
    used = sorted(set(color))
    remap = {c: i for i, c in enumerate(used)}
    return [remap[c] for c in color], len(used)


def make_init_k_minus_1(color, k, adj):
    """Given a proper k-coloring, build an initial (k-1)-assignment by removing
    the smallest color class and reassigning its vertices greedily."""
    n = len(color)
    counts = [0] * k
    for c in color:
        counts[c] += 1
    r = min(range(k), key=lambda c: counts[c])
    newc = []
    for c in color:
        if c == r:
            newc.append(-1)
        elif c > r:
            newc.append(c - 1)
        else:
            newc.append(c)
    kk = k - 1
    for v in range(n):
        if newc[v] == -1:
            # choose color minimizing conflicts among neighbors
            cnt = [0] * kk
            for u in adj[v]:
                cu = newc[u]
                if cu >= 0:
                    cnt[cu] += 1
            m = min(cnt)
            choices = [c for c in range(kk) if cnt[c] == m]
            newc[v] = random.choice(choices)
    return newc


def tabucol(n, adj, k, color, deadline):
    """TabuCol: try to find a proper k-coloring starting from 'color'.
    Returns coloring list on success, None on timeout."""
    if k <= 0:
        return None
    cnt = [[0] * k for _ in range(n)]
    for v in range(n):
        cv = color[v]
        for u in adj[v]:
            cnt[u][cv] += 1
    conf = set()
    f = 0
    for v in range(n):
        cvv = cnt[v][color[v]]
        if cvv > 0:
            conf.add(v)
            f += cvv
    f //= 2
    if f == 0:
        return color
    tabu = [[0] * k for _ in range(n)]
    it = 0
    best_f = f
    rng_random = random.random
    rng_randrange = random.randrange
    while True:
        it += 1
        if (it & 127) == 0 and time.time() > deadline:
            return None
        best_d = 1 << 30
        ties = 0
        sel_v = -1
        sel_c = -1
        for v in conf:
            cc = cnt[v]
            cv = color[v]
            base = cc[cv]
            tv = tabu[v]
            for c in range(k):
                if c == cv:
                    continue
                d = cc[c] - base
                if d > best_d:
                    continue
                if tv[c] > it and f + d >= best_f:
                    continue
                if d < best_d:
                    best_d = d
                    ties = 1
                    sel_v = v
                    sel_c = c
                else:
                    ties += 1
                    if rng_random() < 1.0 / ties:
                        sel_v = v
                        sel_c = c
        if sel_v < 0:
            # everything tabu: random move among conflicted vertices
            sel_v = random.choice(tuple(conf))
            sel_c = rng_randrange(k)
            while sel_c == color[sel_v] and k > 1:
                sel_c = rng_randrange(k)
            best_d = cnt[sel_v][sel_c] - cnt[sel_v][color[sel_v]]
        v = sel_v
        c = sel_c
        old = color[v]
        color[v] = c
        f += best_d
        for u in adj[v]:
            cu = cnt[u]
            cu[old] -= 1
            cu[c] += 1
            cuc = color[u]
            if cuc == old:
                if cu[old] == 0:
                    conf.discard(u)
            elif cuc == c:
                conf.add(u)
        if cnt[v][c] > 0:
            conf.add(v)
        else:
            conf.discard(v)
        tabu[v][old] = it + 10 + int(0.6 * len(conf)) + rng_randrange(10)
        if f == 0:
            return color
        if f < best_f:
            best_f = f


def build_solution(color):
    obj = float(len(set(color)))
    return {
        "objective_value": obj,
        "coloring": {str(v): int(c) for v, c in enumerate(color)},
    }


def write_solution(path, sol):
    with open(path, "w") as f:
        json.dump(sol, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 1.0

    random.seed(0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = read_instance(args.instance_path)
    n = int(inst["num_nodes"])
    edges_raw = inst.get("edges", []) or []

    if n == 0:
        sol = {"objective_value": 0.0, "coloring": {}}
        if logger:
            logger.log_solution(0.0, sol)
        write_solution(args.solution_path, sol)
        return

    adj_sets = [set() for _ in range(n)]
    for e in edges_raw:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    adj = [list(s) for s in adj_sets]
    m = sum(len(a) for a in adj) // 2

    # Initial DSATUR coloring
    color = dsatur(n, adj)
    color, k = compress_colors(color)
    sol = build_solution(color)
    if logger:
        logger.log_solution(float(k), sol)
    write_solution(args.solution_path, sol)

    # Lower bound
    lb = 1 if m == 0 else 2
    clique_lb = greedy_clique_lb(n, adj_sets, min(deadline, time.time() + 2.0))
    lb = max(lb, clique_lb)

    # Iteratively try to reduce number of colors with TabuCol
    while k > lb and time.time() < deadline:
        target = k - 1
        init = make_init_k_minus_1(color, k, adj)
        res = tabucol(n, adj, target, init, deadline)
        if res is None:
            break
        color, k = compress_colors(res)
        sol = build_solution(color)
        if logger:
            logger.log_solution(float(k), sol)
        write_solution(args.solution_path, sol)

    write_solution(args.solution_path, build_solution(color))


if __name__ == "__main__":
    main()