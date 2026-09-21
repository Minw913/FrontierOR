import argparse
import json
import random
import sys
import time

from solution_logger import SolutionLogger


class TimeUp(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)
    n = int(inst["num_vertices"])
    edges = inst.get("edges", []) or []

    if hasattr(int, "bit_count"):
        pc = int.bit_count
    else:
        def pc(x):
            return bin(x).count("1")

    adj = [0] * n
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    best = []
    best_size = 0

    def write_solution():
        sol = {"objective_value": best_size, "clique_vertices": sorted(best)}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    def record(clique):
        nonlocal best, best_size
        best = list(clique)
        best_size = len(best)
        if logger:
            logger.log_solution(best_size,
                                {"objective_value": best_size,
                                 "clique_vertices": sorted(best)})

    if n == 0:
        write_solution()
        return

    # trivial initial solution
    record([0])

    rng = random.Random(0)
    degs = [pc(adj[v]) for v in range(n)]

    # ------------------------------------------------------------------
    # Phase 1: randomized greedy multistart heuristic
    # ------------------------------------------------------------------
    def greedy_from(v0, randomized):
        clique = [v0]
        cand = adj[v0]
        while cand:
            best_u = -1
            best_d = -1
            Q = cand
            while Q:
                b = Q & -Q
                u = b.bit_length() - 1
                Q &= ~b
                d = pc(adj[u] & cand)
                if d > best_d or (randomized and d == best_d and rng.random() < 0.5):
                    best_d = d
                    best_u = u
            clique.append(best_u)
            cand &= adj[best_u]
        return clique

    heur_deadline = min(deadline, start + max(1.0, 0.25 * args.time_limit))
    order_by_deg = sorted(range(n), key=lambda v: -degs[v])

    # first pass: deterministic greedy from highest-degree vertices
    for v0 in order_by_deg[:min(n, 200)]:
        if time.time() > heur_deadline:
            break
        c = greedy_from(v0, False)
        if len(c) > best_size:
            record(c)

    # randomized restarts
    while time.time() < heur_deadline:
        v0 = order_by_deg[rng.randrange(min(n, max(1, n // 2)))]
        c = greedy_from(v0, True)
        if len(c) > best_size:
            record(c)

    # ------------------------------------------------------------------
    # Phase 2: branch-and-bound with greedy coloring bound (Tomita-style)
    # ------------------------------------------------------------------
    sys.setrecursionlimit(max(20000, n + 1000))
    full = (1 << n) - 1

    def expand(R, P):
        nonlocal best, best_size
        if time.time() > deadline:
            raise TimeUp()
        # greedy coloring of candidate set P
        order = []
        colors = []
        uncolored = P
        c = 0
        while uncolored:
            c += 1
            Q = uncolored
            while Q:
                b = Q & -Q
                v = b.bit_length() - 1
                Q &= ~adj[v]
                Q &= ~b
                uncolored &= ~b
                order.append(v)
                colors.append(c)
        # branch in reverse (decreasing color) order
        for i in range(len(order) - 1, -1, -1):
            if len(R) + colors[i] <= best_size:
                return
            v = order[i]
            b = 1 << v
            P &= ~b
            R.append(v)
            if len(R) > best_size:
                record(R)
            newP = P & adj[v]
            if newP:
                expand(R, newP)
            R.pop()

    proven_optimal = False
    try:
        expand([], full)
        proven_optimal = True
    except TimeUp:
        pass
    except RecursionError:
        pass

    # ------------------------------------------------------------------
    # If time remains and not optimal, keep doing randomized heuristics
    # ------------------------------------------------------------------
    if not proven_optimal:
        while time.time() < deadline:
            v0 = rng.randrange(n)
            c = greedy_from(v0, True)
            if len(c) > best_size:
                record(c)

    # defensive validation / repair
    ok = True
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            if not (adj[best[i]] >> best[j]) & 1:
                ok = False
                break
        if not ok:
            break
    if not ok:
        repaired = []
        cand = full
        for v in sorted(best, key=lambda x: -degs[x]):
            if (cand >> v) & 1:
                repaired.append(v)
                cand &= adj[v]
        best = repaired
        best_size = len(best)

    write_solution()


if __name__ == "__main__":
    main()