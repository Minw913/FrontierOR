import argparse
import json
import sys
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


class TimeUp(Exception):
    pass


def degeneracy_order(n, adjsets):
    """Return (order, core, degeneracy) using bucket peeling, O(n+m)."""
    if n == 0:
        return [], [], 0
    deg = [len(adjsets[v]) for v in range(n)]
    md = max(deg) if deg else 0
    buckets = [[] for _ in range(md + 1)]
    for v in range(n):
        buckets[deg[v]].append(v)
    visited = [False] * n
    order = []
    core = [0] * n
    k = 0
    i = 0
    for _ in range(n):
        v = None
        while i < len(buckets):
            found = None
            while buckets[i]:
                u = buckets[i].pop()
                if not visited[u] and deg[u] == i:
                    found = u
                    break
            if found is not None:
                v = found
                break
            i += 1
        if v is None:
            break
        visited[v] = True
        if i > k:
            k = i
        core[v] = k
        order.append(v)
        for u in adjsets[v]:
            if not visited[u]:
                deg[u] -= 1
                buckets[deg[u]].append(u)
        if i > 0:
            i -= 1
    return order, core, k


def greedy_degeneracy_clique(order, adjsets):
    C = []
    Cset = set()
    for v in reversed(order):
        if Cset <= adjsets[v]:
            C.append(v)
            Cset.add(v)
    return C


def greedy_from_seed(v, adjsets):
    C = [v]
    cand = set(adjsets[v])
    while cand:
        best_u = None
        best_c = -1
        for u in cand:
            c = len(adjsets[u] & cand)
            if c > best_c:
                best_c = c
                best_u = u
        C.append(best_u)
        cand &= adjsets[best_u]
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="maximize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n"])
    edges = inst.get("edges", [])
    omega_known = inst.get("omega", None)

    if n == 0:
        sol = {"objective_value": 0, "clique_vertices": []}
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    adjsets = [set() for _ in range(n)]
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        adjsets[u].add(v)
        adjsets[v].add(u)

    best_clique = [0]
    best = 1

    def record(clique):
        nonlocal best, best_clique
        if len(clique) > best:
            best = len(clique)
            best_clique = list(clique)
            if logger:
                logger.log_solution(best, {"objective_value": best,
                                           "clique_vertices": list(best_clique)})

    # ---- degeneracy + greedy heuristics ----
    order, core, dgn = degeneracy_order(n, adjsets)
    record(greedy_degeneracy_clique(order, adjsets))

    # seeded greedy from high-core vertices (time bounded)
    seed_budget = start + 0.15 * max(1, args.time_limit)
    seeds = sorted(range(n), key=lambda v: (core[v], len(adjsets[v])), reverse=True)
    tried = 0
    for v in seeds:
        if tried >= 40 or time.time() > seed_budget or time.time() > deadline:
            break
        if core[v] < best:  # cannot yield larger clique
            continue
        record(greedy_from_seed(v, adjsets))
        tried += 1

    ub = dgn + 1
    done_optimal = (best >= ub) or (omega_known is not None and best >= int(omega_known))

    if not done_optimal and time.time() < deadline:
        # ---- reduce: only vertices with core >= best can be in a larger clique ----
        keep = [v for v in range(n) if core[v] >= best]
        if keep:
            keep.sort(key=lambda v: len(adjsets[v]), reverse=True)
            idx = {v: i for i, v in enumerate(keep)}
            K = len(keep)
            adj = [0] * K
            for v in keep:
                i = idx[v]
                m = 0
                for u in adjsets[v]:
                    j = idx.get(u)
                    if j is not None:
                        m |= 1 << j
                adj[i] = m

            sys.setrecursionlimit(100000)
            node_ctr = [0]
            target = None
            if omega_known is not None:
                target = int(omega_known)

            def expand(R, P):
                nonlocal best
                node_ctr[0] += 1
                if (node_ctr[0] & 127) == 0 and time.time() > deadline:
                    raise TimeUp()
                if len(R) + popcount(P) <= best:
                    return
                # greedy coloring for bound and branching order
                vorder = []
                vcolor = []
                Q = P
                color = 0
                while Q:
                    color += 1
                    cand = Q
                    while cand:
                        b = cand & (-cand)
                        v = b.bit_length() - 1
                        vorder.append(v)
                        vcolor.append(color)
                        cand = (cand ^ b) & ~adj[v]
                        Q &= ~b
                lr = len(R)
                for i in range(len(vorder) - 1, -1, -1):
                    if lr + vcolor[i] <= best:
                        return
                    v = vorder[i]
                    b = 1 << v
                    R.append(v)
                    newP = P & adj[v]
                    if newP:
                        expand(R, newP)
                    else:
                        if len(R) > best:
                            best = len(R)
                            bc = [keep[x] for x in R]
                            nonlocal_record(bc)
                    R.pop()
                    P &= ~b
                    if target is not None and best >= target:
                        raise TimeUp()

            def nonlocal_record(clique):
                nonlocal best_clique
                best_clique = list(clique)
                if logger:
                    logger.log_solution(best, {"objective_value": best,
                                               "clique_vertices": list(best_clique)})

            try:
                full = (1 << K) - 1
                expand([], full)
            except TimeUp:
                pass
            except RecursionError:
                pass

    # sanity check (defensive): verify best_clique is a clique
    ok = True
    cs = set(best_clique)
    for v in best_clique:
        if len(cs & adjsets[v]) != len(cs) - 1:
            ok = False
            break
    if not ok or len(cs) != len(best_clique):
        # fall back to trivial safe clique
        best_clique = [0]
        best = 1

    sol = {"objective_value": len(best_clique), "clique_vertices": list(best_clique)}
    if logger:
        logger.log_solution(sol["objective_value"], sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()