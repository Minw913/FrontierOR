import argparse
import heapq
import json
import random
import sys
import time


# --------------------------------------------------------------------------
# Version-safe popcount (int.bit_count exists only on Python >= 3.10)
# --------------------------------------------------------------------------
if hasattr(int, "bit_count"):
    def popcount(x):
        return x.bit_count()
else:
    def popcount(x):
        return bin(x).count("1")


def lowest_free_color(mask):
    """Smallest non-negative integer whose bit is not set in mask."""
    return (~mask & (mask + 1)).bit_length() - 1


# --------------------------------------------------------------------------
# Heuristic coloring routines
# --------------------------------------------------------------------------

def dsatur_color(n, adj, deg, rng, randomize=False):
    """DSATUR heuristic coloring (heap based)."""
    color = [-1] * n
    sat = [0] * n  # bitmask of colors used by colored neighbors
    heap = []
    for v in range(n):
        heap.append((0, -deg[v], rng.random() if randomize else 0.0, v))
    heapq.heapify(heap)
    ncolors = 0
    colored = 0
    while colored < n:
        while True:
            negs, _d, _r, v = heapq.heappop(heap)
            if color[v] == -1 and -negs == popcount(sat[v]):
                break
        mask = sat[v]
        c = lowest_free_color(mask)
        color[v] = c
        colored += 1
        if c == ncolors:
            ncolors = c + 1
        bit = 1 << c
        for w in adj[v]:
            if color[w] == -1 and not (sat[w] & bit):
                sat[w] |= bit
                heapq.heappush(heap, (-popcount(sat[w]), -deg[w],
                                      rng.random() if randomize else 0.0, w))
    return color, ncolors


def greedy_color(n, adj, order):
    color = [-1] * n
    ncolors = 0
    for v in order:
        mask = 0
        for w in adj[v]:
            c = color[w]
            if c >= 0:
                mask |= (1 << c)
        c = lowest_free_color(mask)
        color[v] = c
        if c == ncolors:
            ncolors += 1
    return color, ncolors


def smallest_last_order(n, adj, deg):
    d = list(deg)
    removed = [False] * n
    heap = [(d[v], v) for v in range(n)]
    heapq.heapify(heap)
    order = []
    while heap:
        dv, v = heapq.heappop(heap)
        if removed[v] or dv != d[v]:
            continue
        removed[v] = True
        order.append(v)
        for w in adj[v]:
            if not removed[w]:
                d[w] -= 1
                heapq.heappush(heap, (d[w], w))
    order.reverse()
    return order


def heuristic_clique(n, adj_set, deg, rng, budget):
    """Greedy clique growth from many starting vertices, time-boxed."""
    end = time.time() + budget
    best = []
    starts = sorted(range(n), key=lambda v: -deg[v])
    total = len(starts)
    if total == 0:
        return best
    i = 0
    it = 0
    while True:
        it += 1
        if (it & 15) == 0 and time.time() >= end:
            break
        if i < total:
            s = starts[i]
            i += 1
        else:
            if time.time() >= end:
                break
            s = starts[rng.randrange(total)]
        clique = [s]
        cand = set(adj_set[s])
        while cand:
            v = max(cand, key=lambda x: deg[x])
            clique.append(v)
            cand &= adj_set[v]
        if len(clique) > len(best):
            best = clique
    return best


# --------------------------------------------------------------------------
# Tabucol: local search for k-colorability
# --------------------------------------------------------------------------

def tabucol(n, adj, k, init_color, rng, max_iters, deadline):
    color = list(init_color)
    cnt = [[0] * k for _ in range(n)]
    for v in range(n):
        row = cnt[v]
        for w in adj[v]:
            row[color[w]] += 1
    E = 0
    for v in range(n):
        E += cnt[v][color[v]]
    E //= 2
    conf = []
    pos = [-1] * n
    for v in range(n):
        if cnt[v][color[v]] > 0:
            pos[v] = len(conf)
            conf.append(v)
    if E == 0:
        return color, 0
    tabu = [[0] * k for _ in range(n)]
    best_E = E
    it = 0
    while it < max_iters:
        it += 1
        if (it & 1023) == 0 and time.time() > deadline:
            return None, E
        best_delta = None
        mv = -1
        mc = -1
        ns = 12 if len(conf) >= 12 else len(conf)
        for _ in range(ns):
            v = conf[rng.randrange(len(conf))]
            cv = color[v]
            row = cnt[v]
            base = row[cv]
            tabrow = tabu[v]
            for c in range(k):
                if c == cv:
                    continue
                d = row[c] - base
                if best_delta is not None and d > best_delta:
                    continue
                if tabrow[c] > it and E + d >= best_E:
                    continue
                if best_delta is None or d < best_delta:
                    best_delta = d
                    mv = v
                    mc = c
        if mv == -1:
            v = conf[rng.randrange(len(conf))]
            c = rng.randrange(k)
            if c == color[v]:
                continue
            mv, mc = v, c
            best_delta = cnt[v][c] - cnt[v][color[v]]
        v = mv
        c_new = mc
        c_old = color[v]
        if c_new == c_old:
            continue
        E += best_delta
        color[v] = c_new
        tabu[v][c_old] = it + rng.randint(0, 9) + E
        for w in adj[v]:
            row = cnt[w]
            row[c_old] -= 1
            row[c_new] += 1
            cw = color[w]
            if cw == c_old:
                if row[c_old] == 0 and pos[w] != -1:
                    i = pos[w]
                    last = conf[-1]
                    conf[i] = last
                    pos[last] = i
                    conf.pop()
                    pos[w] = -1
            elif cw == c_new:
                if row[c_new] == 1 and pos[w] == -1:
                    pos[w] = len(conf)
                    conf.append(w)
        if cnt[v][c_new] > 0:
            if pos[v] == -1:
                pos[v] = len(conf)
                conf.append(v)
        else:
            i = pos[v]
            if i != -1:
                last = conf[-1]
                conf[i] = last
                pos[last] = i
                conf.pop()
                pos[v] = -1
        if E < best_E:
            best_E = E
        if E == 0:
            return color, 0
    return None, E


def dissolve_init(n, adj, color, k, rng, dissolve=None):
    """Build an initial (k-1)-coloring by dissolving one color class."""
    classes = [[] for _ in range(k)]
    for v, c in enumerate(color):
        classes[c].append(v)
    if dissolve is None:
        dissolve = min(range(k), key=lambda c: len(classes[c]))
    K = k - 1
    mapping = [-1] * k
    nc = 0
    for c in range(k):
        if c != dissolve:
            mapping[c] = nc
            nc += 1
    init = [-1] * n
    for c in range(k):
        if c == dissolve:
            continue
        mc = mapping[c]
        for v in classes[c]:
            init[v] = mc
    order = list(classes[dissolve])
    rng.shuffle(order)
    for v in order:
        cnts = [0] * max(K, 1)
        for w in adj[v]:
            cw = init[w]
            if cw >= 0:
                cnts[cw] += 1
        mn = min(cnts)
        cands = [c for c in range(len(cnts)) if cnts[c] == mn]
        init[v] = cands[rng.randrange(len(cands))]
    return init


# --------------------------------------------------------------------------
# Exact DSATUR branch and bound (for small graphs)
# --------------------------------------------------------------------------

def exact_color(n, adj, deg, init_color, init_k, lb, clique, deadline, improve_cb):
    sys.setrecursionlimit(max(100000, n * 4 + 1000))
    color = [-1] * n
    sat = [0] * n
    K = init_k
    ccnt = [[0] * K for _ in range(n)]
    best = [init_k]
    best_col = [list(init_color)]
    nodes = [0]
    proved = [False]

    class Abort(Exception):
        pass

    class Done(Exception):
        pass

    def assign(v, c):
        color[v] = c
        for w in adj[v]:
            if color[w] == -1:
                row = ccnt[w]
                row[c] += 1
                if row[c] == 1:
                    sat[w] += 1

    def unassign(v, c):
        color[v] = -1
        for w in adj[v]:
            if color[w] == -1:
                row = ccnt[w]
                row[c] -= 1
                if row[c] == 0:
                    sat[w] -= 1

    def rec(uncolored, used):
        if uncolored == 0:
            if used < best[0]:
                best[0] = used
                best_col[0] = list(color)
                improve_cb(color, used)
            return
        nodes[0] += 1
        if (nodes[0] & 4095) == 0 and time.time() > deadline:
            raise Abort()
        bv = -1
        bs = -1
        bd = -1
        maxsat = 0
        for v in range(n):
            if color[v] == -1:
                s = sat[v]
                if s > maxsat:
                    maxsat = s
                if s > bs or (s == bs and deg[v] > bd):
                    bv = v
                    bs = s
                    bd = deg[v]
        if maxsat + 1 >= best[0]:
            return  # any completion would need >= best colors
        v = bv
        row = ccnt[v]
        for c in range(used):
            if row[c] == 0:
                assign(v, c)
                rec(uncolored - 1, used)
                unassign(v, c)
                if best[0] <= lb:
                    raise Done()
        if used + 1 <= best[0] - 1:
            assign(v, used)
            rec(uncolored - 1, used + 1)
            unassign(v, used)
            if best[0] <= lb:
                raise Done()

    # Pre-color the clique (valid by color-permutation symmetry).
    q = 0
    for i, v in enumerate(clique):
        if color[v] == -1:
            assign(v, i)
            q += 1

    try:
        rec(n - q, q)
        proved[0] = True
    except Done:
        proved[0] = True
    except Abort:
        proved[0] = False
    return best[0], best_col[0], proved[0]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Graph coloring solver")
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    t0 = time.time()
    tl = max(1, int(args.time_limit))
    eff_deadline = t0 + tl - 0.5  # reserve time to write the output

    with open(args.instance_path) as f:
        data = json.load(f)
    g = data.get("graph", {})
    n = int(g.get("num_vertices", 0))
    raw_edges = g.get("edges", [])

    # Detect 1-based vertex indexing defensively.
    mx = -1
    for e in raw_edges:
        if e and len(e) >= 2:
            if e[0] > mx:
                mx = e[0]
            if e[1] > mx:
                mx = e[1]
    shift = 1 if mx >= n else 0

    adj_set = [set() for _ in range(n)]
    edges = []
    for e in raw_edges:
        if not e or len(e) < 2:
            continue
        u = e[0] - shift
        v = e[1] - shift
        if 0 <= u < n and 0 <= v < n and u != v and v not in adj_set[u]:
            adj_set[u].add(v)
            adj_set[v].add(u)
            edges.append((u, v))
    adj = [sorted(s) for s in adj_set]
    deg = [len(s) for s in adj]
    m = len(edges)
    rng = random.Random(0)

    def build_solution(color):
        k = (max(color) + 1) if color else 0
        classes = [[] for _ in range(k)]
        for v, c in enumerate(color):
            classes[c].append(v)
        for cls in classes:
            cls.sort()
        return {
            "objective_value": len(classes),
            "selected_sets": classes,
            "coloring": {str(v): int(c) for v, c in enumerate(color)},
        }

    def normalize(color):
        mp = {}
        out = []
        for c in color:
            r = mp.get(c)
            if r is None:
                r = len(mp)
                mp[c] = r
            out.append(r)
        return out

    # Trivial cases ---------------------------------------------------------
    if n == 0:
        sol = {"objective_value": 0, "selected_sets": [], "coloring": {}}
        if logger:
            try:
                logger.log_solution(0, sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return
    if m == 0:
        sol = {
            "objective_value": 1,
            "selected_sets": [list(range(n))],
            "coloring": {str(v): 0 for v in range(n)},
        }
        if logger:
            try:
                logger.log_solution(1, sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    best_color, best_k = dsatur_color(n, adj, deg, rng, randomize=False)
    best_color = normalize(best_color)
    best_k = max(best_color) + 1
    safe_color = list(best_color)  # always-valid fallback

    def log_best():
        if logger:
            try:
                logger.log_solution(best_k, build_solution(best_color))
            except Exception:
                pass

    # ---- Phase 1: additional heuristic colorings --------------------------
    try:
        c2, k2 = greedy_color(n, adj, sorted(range(n), key=lambda v: -deg[v]))
        if k2 < best_k:
            c2n = normalize(c2)
            best_color, best_k = c2n, max(c2n) + 1
        if time.time() < eff_deadline:
            c3, k3 = greedy_color(n, adj, smallest_last_order(n, adj, deg))
            if k3 < best_k:
                c3n = normalize(c3)
                best_color, best_k = c3n, max(c3n) + 1
        extra = 6 if n <= 2500 else (2 if n <= 6000 else 0)
        for _ in range(extra):
            if time.time() > eff_deadline - 0.2:
                break
            c4, k4 = dsatur_color(n, adj, deg, rng, randomize=True)
            if k4 < best_k:
                c4n = normalize(c4)
                best_color, best_k = c4n, max(c4n) + 1
    except Exception:
        pass
    log_best()

    # ---- Phase 2: clique lower bound ---------------------------------------
    given_lb = 0
    try:
        given_lb = int(data.get("bounds", {}).get("clique_lower_bound", 0) or 0)
    except Exception:
        given_lb = 0
    clique = []
    try:
        rem = eff_deadline - time.time()
        if rem > 0.4:
            budget = min(3.0, max(0.25, 0.05 * (eff_deadline - time.time())))
            clique = heuristic_clique(n, adj_set, deg, rng, budget)
    except Exception:
        clique = []
    LB = max(given_lb, len(clique), 2)  # m > 0 here
    LB = min(LB, best_k)

    # ---- Phase 3: Tabucol descent ------------------------------------------
    try:
        rem = eff_deadline - time.time()
        if n <= 150:
            descent_deadline = time.time() + min(rem, max(0.5, 0.15 * rem))
        else:
            descent_deadline = eff_deadline
        descent_ok = (n * best_k <= 4_000_000)
        fails = 0
        while descent_ok and best_k > LB and fails < 10:
            now = time.time()
            if now >= descent_deadline:
                break
            target = best_k - 1
            if target <= 1:
                break
            remaining = descent_deadline - time.time()
            if remaining <= 0.05:
                break
            budget = max(0.5, min(10.0, remaining * 0.25))
            aend = min(descent_deadline, time.time() + budget)
            dissolve = None if fails == 0 else rng.randrange(best_k)
            init = dissolve_init(n, adj, best_color, best_k, rng, dissolve)
            res, E = tabucol(n, adj, target, init, rng, 10 ** 9, aend)
            if res is not None:
                res = normalize(res)
                best_color = res
                best_k = max(res) + 1
                fails = 0
                log_best()
            else:
                fails += 1
            if best_k <= LB:
                break
    except Exception:
        pass

    # ---- Phase 4: exact DSATUR branch & bound (small graphs) ----------------
    def on_improve(col, k):
        nonlocal best_color, best_k
        best_color = list(col)
        best_k = k
        log_best()

    try:
        if best_k > LB and n <= 150 and eff_deadline - time.time() > 0.5:
            rk, rcol, _proved = exact_color(
                n, adj, deg, best_color, best_k, LB, clique,
                eff_deadline, on_improve)
            if rk < best_k:
                best_color = list(rcol)
                best_k = rk
                log_best()
    except Exception:
        pass

    # ---- Final output --------------------------------------------------------
    best_color = normalize(best_color)
    ok = True
    for u, v in edges:
        if best_color[u] == best_color[v]:
            ok = False
            break
    if not ok:
        best_color = normalize(safe_color)
    sol = build_solution(best_color)
    if logger:
        try:
            logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()