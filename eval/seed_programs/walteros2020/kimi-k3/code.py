import sys
import json
import time
import argparse
import heapq


class _Timeout(Exception):
    pass


class _HeurTimeout(Exception):
    pass


if hasattr(int, "bit_count"):
    _popcount = int.bit_count
else:
    def _popcount(x):
        return bin(x).count("1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path) as f:
        data = json.load(f)

    n = int(data.get("n", 0))
    edges = data.get("edges", []) or []
    deadline = time.time() + max(0, int(args.time_limit))

    best = 0
    best_clique = []

    def finish():
        sol = {
            "objective_value": int(best),
            "clique_vertices": sorted(int(x) for x in best_clique),
        }
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    if n <= 0:
        if logger:
            logger.log_solution(0, {"objective_value": 0, "clique_vertices": []})
        finish()
        return

    use_bits = (n <= 20000)

    # Build adjacency (sets always; bit-parallel masks for modest n).
    adjset = [set() for _ in range(n)]
    barrs = None
    if use_bits:
        nbytes = (n + 7) // 8
        barrs = [bytearray(nbytes) for _ in range(n)]
    for e in edges:
        try:
            u = int(e[0])
            v = int(e[1])
        except Exception:
            continue
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            continue
        adjset[u].add(v)
        adjset[v].add(u)
        if use_bits:
            barrs[u][v >> 3] |= (1 << (v & 7))
            barrs[v][u >> 3] |= (1 << (u & 7))

    # Degeneracy ordering (min-degree peeling) and core numbers.
    deg0 = [len(adjset[v]) for v in range(n)]
    deg = deg0[:]
    heap = [(deg[v], v) for v in range(n)]
    heapq.heapify(heap)
    removed = [False] * n
    core = [0] * n
    order = []
    dmax = 0
    while heap:
        dd, v = heapq.heappop(heap)
        if removed[v] or dd != deg[v]:
            continue
        removed[v] = True
        core[v] = dd
        if dd > dmax:
            dmax = dd
        order.append(v)
        for u in adjset[v]:
            if not removed[u]:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))

    sys.setrecursionlimit(max(10000, dmax + 1000))

    # Trivial feasible solution (any single vertex is a clique).
    best = 1
    best_clique = [order[-1]]
    if logger:
        logger.log_solution(1, {"objective_value": 1,
                                "clique_vertices": list(best_clique)})

    remaining = max(0.0, deadline - time.time())
    heur_deadline = time.time() + max(0.2, 0.15 * remaining)
    if heur_deadline > deadline:
        heur_deadline = deadline

    nodes = [0]
    adj = None
    adjl = None

    try:
        if use_bits:
            adj = [int.from_bytes(a, "little") for a in barrs]
            barrs = None
            adjset = None

            def expand(P, size, current):
                """Tomita/BBMC branch and bound with greedy coloring bound."""
                nonlocal best
                nodes[0] += 1
                if (nodes[0] & 1023) == 0 and time.time() > deadline:
                    raise _Timeout
                adj_l = adj
                verts = []
                colors = []
                uncolored = P
                color = 0
                apv = verts.append
                apc = colors.append
                while uncolored:
                    color += 1
                    avail = uncolored
                    while avail:
                        lb = avail & -avail
                        v = lb.bit_length() - 1
                        apv(v)
                        apc(color)
                        uncolored &= ~lb
                        avail &= ~(lb | adj_l[v])
                for i in range(len(verts) - 1, -1, -1):
                    if size + colors[i] <= best:
                        return
                    v = verts[i]
                    current.append(v)
                    if size + 1 > best:
                        best = size + 1
                        best_clique[:] = current
                        if logger:
                            logger.log_solution(
                                best,
                                {"objective_value": best,
                                 "clique_vertices": sorted(best_clique)})
                    newP = P & adj_l[v]
                    if newP:
                        expand(newP, size + 1, current)
                    current.pop()
                    P &= ~(1 << v)

            def greedy_from(v0):
                clique = [v0]
                P = adj[v0]
                adj_l = adj
                while P:
                    t = time.time()
                    if t > deadline:
                        raise _Timeout
                    if t > heur_deadline:
                        raise _HeurTimeout
                    T = P
                    bu = -1
                    bc = -1
                    while T:
                        lb = T & -T
                        u = lb.bit_length() - 1
                        T &= ~lb
                        c = _popcount(P & adj_l[u])
                        if c > bc:
                            bc = c
                            bu = u
                    clique.append(bu)
                    P &= adj_l[bu]
                return clique

            seeds = sorted(range(n), key=lambda v: -deg0[v])[:12]
            for v0 in seeds:
                try:
                    c = greedy_from(v0)
                except _HeurTimeout:
                    break
                if len(c) > best:
                    best = len(c)
                    best_clique[:] = c
                    if logger:
                        logger.log_solution(
                            best, {"objective_value": best,
                                   "clique_vertices": sorted(c)})

            # Main loop: reverse degeneracy order with core-number bounds.
            later = 0
            for idx in range(n - 1, -1, -1):
                if best >= dmax + 1:
                    break
                if (idx & 63) == 0 and time.time() > deadline:
                    raise _Timeout
                v = order[idx]
                if core[v] + 1 <= best:
                    later |= (1 << v)
                    continue
                P = adj[v] & later
                if not P:
                    later |= (1 << v)
                    continue
                t = P
                while t:
                    lb = t & -t
                    u = lb.bit_length() - 1
                    t &= ~lb
                    if core[u] + 1 <= best:
                        P &= ~lb
                if _popcount(P) + 1 <= best:
                    later |= (1 << v)
                    continue
                current = [v]
                expand(P, 1, current)
                later |= (1 << v)
        else:
            # Set-based path for very large (typically sparse) graphs.
            adjl = adjset

            def expand_set(P, size, current):
                nonlocal best
                nodes[0] += 1
                if (nodes[0] & 1023) == 0 and time.time() > deadline:
                    raise _Timeout
                verts = []
                colors = []
                uncolored = P.copy()
                color = 0
                while uncolored:
                    color += 1
                    avail = uncolored.copy()
                    while avail:
                        v = avail.pop()
                        uncolored.discard(v)
                        verts.append(v)
                        colors.append(color)
                        avail.difference_update(adjl[v])
                for i in range(len(verts) - 1, -1, -1):
                    if size + colors[i] <= best:
                        return
                    v = verts[i]
                    current.append(v)
                    if size + 1 > best:
                        best = size + 1
                        best_clique[:] = current
                        if logger:
                            logger.log_solution(
                                best,
                                {"objective_value": best,
                                 "clique_vertices": sorted(best_clique)})
                    newP = P & adjl[v]
                    if newP:
                        expand_set(newP, size + 1, current)
                    current.pop()
                    P.discard(v)

            def greedy_from_set(v0):
                clique = [v0]
                P = set(adjl[v0])
                while P:
                    t = time.time()
                    if t > deadline:
                        raise _Timeout
                    if t > heur_deadline:
                        raise _HeurTimeout
                    if len(P) <= 512:
                        bu = max(P, key=lambda u: len(adjl[u] & P))
                    else:
                        it = iter(P)
                        sample = []
                        for _ in range(512):
                            try:
                                sample.append(next(it))
                            except StopIteration:
                                break
                        bu = max(sample, key=lambda u: len(adjl[u] & P))
                    clique.append(bu)
                    P &= adjl[bu]
                return clique

            seeds = sorted(range(n), key=lambda v: -deg0[v])[:8]
            for v0 in seeds:
                try:
                    c = greedy_from_set(v0)
                except _HeurTimeout:
                    break
                if len(c) > best:
                    best = len(c)
                    best_clique[:] = c
                    if logger:
                        logger.log_solution(
                            best, {"objective_value": best,
                                   "clique_vertices": sorted(c)})

            later = set()
            for idx in range(n - 1, -1, -1):
                if best >= dmax + 1:
                    break
                if (idx & 63) == 0 and time.time() > deadline:
                    raise _Timeout
                v = order[idx]
                if core[v] + 1 <= best:
                    later.add(v)
                    continue
                P = adjl[v] & later
                if not P:
                    later.add(v)
                    continue
                P = set(u for u in P if core[u] + 1 > best)
                if len(P) + 1 <= best:
                    later.add(v)
                    continue
                current = [v]
                expand_set(P, 1, current)
                later.add(v)
    except _Timeout:
        pass
    except Exception:
        pass

    # Sanity verification of the incumbent clique.
    k = len(best_clique)
    if k > 1:
        try:
            if adj is not None:
                cm = 0
                for v in best_clique:
                    cm |= (1 << v)
                valid = True
                for v in best_clique:
                    o = cm & ~(1 << v)
                    if (adj[v] & o) != o:
                        valid = False
                        break
            elif adjset is not None:
                s = set(best_clique)
                valid = True
                for v in best_clique:
                    if len(adjset[v] & s) != k - 1:
                        valid = False
                        break
            else:
                valid = True
        except Exception:
            valid = True
        if not valid:
            best = 1
            best_clique = best_clique[:1]

    finish()


if __name__ == "__main__":
    main()