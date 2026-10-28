import argparse
import heapq
import json
import random
import sys
import time
from collections import deque

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def normalize(colors):
    """Relabel colors to 0..k-1 preserving first-occurrence order."""
    mapping = {}
    out = []
    nxt = 0
    for c in colors:
        m = mapping.get(c)
        if m is None:
            m = nxt
            mapping[c] = m
            nxt += 1
        out.append(m)
    return out


def dsatur(adj, n, rng):
    """DSATUR greedy coloring with random tie-breaking (heap based)."""
    color = [-1] * n
    nbr_colors = [0] * n          # bitmask of colors used by neighbours
    sat = [0] * n
    deg = [len(a) for a in adj]
    rv = [rng.random() for _ in range(n)]
    heap = [(0, -deg[v], rv[v], v) for v in range(n)]
    heapq.heapify(heap)
    while heap:
        negs, _, _, v = heapq.heappop(heap)
        if color[v] != -1 or -negs != sat[v]:
            continue
        banned = nbr_colors[v]
        low = ~banned & (banned + 1)      # lowest zero bit -> smallest free color
        c = low.bit_length() - 1
        color[v] = c
        bit = 1 << c
        for u in adj[v]:
            if color[u] == -1 and not (nbr_colors[u] & bit):
                nbr_colors[u] |= bit
                sat[u] += 1
                heapq.heappush(heap, (-sat[u], -deg[u], rv[u], u))
    return color


def bicolor(adj, n):
    """Return a 2-coloring if the graph is bipartite, else None."""
    color = [-1] * n
    for s in range(n):
        if color[s] != -1:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            cv = color[v]
            for u in adj[v]:
                if color[u] == -1:
                    color[u] = cv ^ 1
                    q.append(u)
                elif color[u] == cv:
                    return None
    return color


def reduce_coloring(colors, k_colors, adj, rng):
    """Turn a k_colors-coloring into a (k_colors-1)-coloring with as few
    conflicts as possible by eliminating the smallest color class."""
    k = k_colors - 1
    sizes = [0] * k_colors
    for c in colors:
        sizes[c] += 1
    c_elim = min(range(k_colors), key=lambda c: sizes[c])
    remap = {}
    nxt = 0
    for c in range(k_colors):
        if c != c_elim:
            remap[c] = nxt
            nxt += 1
    newc = [remap[c] if c != c_elim else -1 for c in colors]
    verts = [v for v in range(len(colors)) if colors[v] == c_elim]
    rng.shuffle(verts)
    for v in verts:
        counts = [0] * k
        for u in adj[v]:
            cu = newc[u]
            if cu >= 0:
                counts[cu] += 1
        m = min(counts)
        cand = [c for c in range(k) if counts[c] == m]
        newc[v] = rng.choice(cand)
    return newc


# ---------------------------------------------------------------------------
# TABUCOL local search for k-colorability
# ---------------------------------------------------------------------------

def tabucol(adj, n, k, init, deadline, rng, max_iters=3_000_000):
    if k <= 0:
        return None
    color = list(init)
    cnt = [[0] * k for _ in range(n)]     # cnt[v][c] = #neighbours of v with color c
    for v in range(n):
        row = cnt[v]
        for u in adj[v]:
            row[color[u]] += 1

    in_conf = [False] * n
    conf_list = []
    pos = [-1] * n

    def add_conf(v):
        in_conf[v] = True
        pos[v] = len(conf_list)
        conf_list.append(v)

    def remove_conf(v):
        in_conf[v] = False
        i = pos[v]
        last = conf_list[-1]
        conf_list[i] = last
        pos[last] = i
        conf_list.pop()
        pos[v] = -1

    f2 = 0
    for v in range(n):
        if cnt[v][color[v]] > 0:
            add_conf(v)
            f2 += cnt[v][color[v]]
    f = f2 // 2                            # number of conflicting edges

    tabu = [[0] * k for _ in range(n)]
    best_f = f
    it = 0
    while f > 0 and it < max_iters:
        it += 1
        if (it & 255) == 0 and time.time() >= deadline:
            return None
        L = len(conf_list)
        if L == 0:
            break
        if L <= 64:
            cand = conf_list
        else:
            cand = [conf_list[rng.randrange(L)] for _ in range(64)]
        best_delta = None
        moves = []
        for v in cand:
            cv = color[v]
            row = cnt[v]
            cur = row[cv]
            trow = tabu[v]
            for c in range(k):
                if c == cv:
                    continue
                delta = row[c] - cur
                if best_delta is not None and delta > best_delta:
                    continue
                if trow[c] > it and f + delta >= best_f:   # tabu, no aspiration
                    continue
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    moves = [(v, c)]
                elif delta == best_delta:
                    moves.append((v, c))
        if not moves:
            v = conf_list[rng.randrange(L)]
            c = rng.randrange(k - 1)
            if c >= color[v]:
                c += 1
            moves = [(v, c)]
        v, c = moves[rng.randrange(len(moves))]
        old = color[v]
        delta = cnt[v][c] - cnt[v][old]
        color[v] = c
        f += delta
        tabu[v][old] = it + f + rng.randint(0, 9)
        for u in adj[v]:
            cnt[u][old] -= 1
            cnt[u][c] += 1
            if cnt[u][color[u]] > 0:
                if not in_conf[u]:
                    add_conf(u)
            elif in_conf[u]:
                remove_conf(u)
        if cnt[v][c] > 0:
            if not in_conf[v]:
                add_conf(v)
        elif in_conf[v]:
            remove_conf(v)
        if f < best_f:
            best_f = f
    return color if f == 0 else None


# ---------------------------------------------------------------------------
# Clique lower bound (greedy + time-capped Tomita)
# ---------------------------------------------------------------------------

def clique_lower_bound(adj, adj_set, n, rng, deadline):
    deg = [len(a) for a in adj]
    best = 2

    def greedy_set(start):
        cand = adj_set[start].copy()
        size = 1
        while cand:
            v = max(cand, key=lambda x: deg[x])
            size += 1
            cand.intersection_update(adj_set[v])
        return size

    trials = 30 if n <= 3000 else 8
    starts = sorted(range(n), key=lambda v: -deg[v])[:trials]
    for s in starts:
        if time.time() > deadline:
            break
        r = greedy_set(s)
        if r > best:
            best = r

    if n > 8000 or time.time() >= deadline:
        return best

    # Relabel by descending degree and build bitmask adjacency.
    order = sorted(range(n), key=lambda v: -deg[v])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    bits = [0] * n
    for v in range(n):
        b = 0
        for u in adj[v]:
            b |= 1 << pos[u]
        bits[pos[v]] = b

    sys.setrecursionlimit(max(10000, n + 100))
    best_box = [best]
    calls = [0]

    class _Timeout(Exception):
        pass

    def expand(R, P):
        calls[0] += 1
        if (calls[0] & 1023) == 0 and time.time() > deadline:
            raise _Timeout()
        if not P:
            if R > best_box[0]:
                best_box[0] = R
            return
        order_l = []
        bounds = []
        uncol = P
        colno = 0
        while uncol:
            colno += 1
            avail = uncol
            while avail:
                lsb = avail & -avail
                v = lsb.bit_length() - 1
                order_l.append(v)
                bounds.append(colno)
                uncol ^= lsb
                avail &= ~lsb
                avail &= ~bits[v]
        mask = P
        for i in range(len(order_l) - 1, -1, -1):
            if R + bounds[i] <= best_box[0]:
                return
            v = order_l[i]
            mask &= ~(1 << v)
            expand(R + 1, mask & bits[v])

    try:
        expand(0, (1 << n) - 1)
    except (_Timeout, RecursionError):
        pass
    return best_box[0]


# ---------------------------------------------------------------------------
# Solution construction / verification
# ---------------------------------------------------------------------------

def build_solution(colors, nodes):
    k = max(colors) + 1 if colors else 0
    vertex_colors = {str(nodes[i]): int(colors[i]) for i in range(len(nodes))}
    classes = [[] for _ in range(k)]
    for i, c in enumerate(colors):
        classes[c].append(nodes[i])
    return {"objective_value": int(k),
            "vertex_colors": vertex_colors,
            "color_classes": classes}


def is_proper(colors, edges_idx):
    for a, b in edges_idx:
        if colors[a] == colors[b]:
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as fh:
        instance = json.load(fh)
    g = instance["graph"]
    nodes = list(g["nodes"])
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    adj_set = [set() for _ in range(n)]
    edges_idx = []
    for e in g["edges"]:
        a = index[e[0]]
        b = index[e[1]]
        if a == b:
            continue
        if b not in adj_set[a]:
            adj_set[a].add(b)
            adj_set[b].add(a)
            edges_idx.append((a, b))
    adj = [list(s) for s in adj_set]
    m = len(edges_idx)

    # Trivial cases
    if n == 0:
        sol = {"objective_value": 0, "vertex_colors": {}, "color_classes": []}
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)
        return
    if m == 0:
        sol = build_solution([0] * n, nodes)
        if logger:
            logger.log_solution(1, sol)
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)
        return

    start = time.time()
    total = max(1, int(args.time_limit))
    deadline = start + max(0.2, total - 0.3)
    rng = random.Random(987654321)

    # ---- Upper bound: repeated DSATUR ----
    best = normalize(dsatur(adj, n, rng))
    best_k = max(best) + 1
    ds_end = min(deadline, time.time() + max(0.05, 0.08 * (deadline - time.time())))
    while time.time() < ds_end:
        c = normalize(dsatur(adj, n, rng))
        k = max(c) + 1
        if k < best_k:
            best, best_k = c, k

    # Bipartite shortcut
    if best_k > 2:
        bi = bicolor(adj, n)
        if bi is not None:
            best, best_k = bi, 2

    sol = build_solution(best, nodes)
    if logger:
        logger.log_solution(best_k, sol)

    # ---- Lower bound: maximum clique (time capped) ----
    lb = 2
    if best_k > 2 and time.time() < deadline:
        lb_end = min(deadline, time.time() + max(0.2, 0.15 * (deadline - time.time())))
        lb = max(2, clique_lower_bound(adj, adj_set, n, rng, lb_end))

    # ---- Iterated TABUCOL: try to reduce the number of colors ----
    while best_k > lb and time.time() < deadline - 0.05:
        k = best_k - 1
        if n * max(1, k) > 6_000_000:      # memory safeguard for huge instances
            break
        init = reduce_coloring(best, best_k, adj, rng)
        remaining = deadline - time.time()
        levels = best_k - lb
        budget = max(0.3, remaining / (levels + 1))
        att_deadline = min(deadline, time.time() + budget)
        res = tabucol(adj, n, k, init, att_deadline, rng)
        if res is not None:
            res = normalize(res)
            k2 = max(res) + 1
            if k2 < best_k and is_proper(res, edges_idx):
                best, best_k = res, k2
                sol = build_solution(best, nodes)
                if logger:
                    logger.log_solution(best_k, sol)
        else:
            if time.time() < deadline - 0.25:
                continue          # retry with a fresh randomized starting point
            break

    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()