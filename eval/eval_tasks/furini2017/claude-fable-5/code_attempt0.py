import argparse
import json
import random
import time
import sys

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    g = data["graph"]
    n = g["n"]
    edges = [tuple(e) for e in g["edges"]]
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    adj = [sorted(s) for s in adj]
    return n, edges, adj


def dsatur(n, adj):
    """DSATUR greedy coloring; returns color list."""
    color = [-1] * n
    if n == 0:
        return color
    sat = [set() for _ in range(n)]
    deg = [len(adj[v]) for v in range(n)]
    uncolored = set(range(n))
    for _ in range(n):
        # pick vertex with max saturation, tie-break by degree
        best_v, best_key = -1, (-1, -1)
        for v in uncolored:
            key = (len(sat[v]), deg[v])
            if key > best_key:
                best_key = key
                best_v = v
        v = best_v
        used = sat[v]
        c = 0
        while c in used:
            c += 1
        color[v] = c
        uncolored.remove(v)
        for u in adj[v]:
            if color[u] == -1:
                sat[u].add(c)
    return color


def greedy_clique(n, adj, rng, tries=20, deadline=None):
    """Greedy max clique lower bound."""
    best = []
    adjset = [set(a) for a in adj]
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    for t in range(tries):
        if deadline and time.time() > deadline:
            break
        if t == 0:
            start_order = order
        else:
            start_order = order[:]
            rng.shuffle(start_order)
        if not start_order:
            break
        v0 = start_order[0]
        clique = [v0]
        cand = set(adjset[v0])
        while cand:
            # pick candidate with max degree within cand
            u = max(cand, key=lambda x: len(adjset[x] & cand))
            clique.append(u)
            cand &= adjset[u]
        if len(clique) > len(best):
            best = clique
    return len(best) if best else (1 if n > 0 else 0)


def build_solution(n, color):
    if n == 0:
        return 0, {"objective_value": 0, "coloring": {}, "color_classes": {}}
    # normalize colors to 0..k-1
    used = sorted(set(color))
    remap = {c: i for i, c in enumerate(used)}
    col = [remap[c] for c in color]
    k = len(used)
    coloring = {str(v): int(col[v]) for v in range(n)}
    classes = {}
    for v in range(n):
        classes.setdefault(str(col[v]), []).append(v)
    sol = {"objective_value": int(k), "coloring": coloring,
           "color_classes": classes}
    return k, sol


def tabucol(n, adj, k, init_color, deadline, rng, max_iters=None):
    """Try to find a proper k-coloring. Returns coloring list or None."""
    if k <= 0:
        return None
    color = init_color[:]
    # conflict counts: ncc[v][c] = number of neighbors of v with color c
    ncc = [[0] * k for _ in range(n)]
    for v in range(n):
        cv = color[v]
        for u in adj[v]:
            ncc[u][cv] += 1
    conflicts = 0
    conflicted = set()
    for v in range(n):
        if ncc[v][color[v]] > 0:
            conflicted.add(v)
            conflicts += ncc[v][color[v]]
    conflicts //= 2
    if conflicts == 0:
        return color

    tabu = {}
    it = 0
    best_conf = conflicts
    check_interval = 512
    while conflicts > 0:
        it += 1
        if it % check_interval == 0:
            if time.time() > deadline:
                return None
            if max_iters and it > max_iters:
                return None
        # pick a random conflicted vertex
        v = rng.choice(tuple(conflicted))
        cv = color[v]
        row = ncc[v]
        cur = row[cv]
        best_c = -1
        best_delta = 10**9
        for c in range(k):
            if c == cv:
                continue
            delta = row[c] - cur
            tkey = tabu.get((v, c), -1)
            if tkey >= it:
                # tabu unless aspiration
                if conflicts + delta >= best_conf:
                    continue
            if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                best_delta = delta
                best_c = c
        if best_c == -1:
            # all tabu: pick random color
            best_c = rng.randrange(k)
            if best_c == cv:
                best_c = (best_c + 1) % k
            best_delta = row[best_c] - cur
        # apply move
        color[v] = best_c
        for u in adj[v]:
            ncc[u][cv] -= 1
            ncc[u][best_c] += 1
            if color[u] == cv and ncc[u][cv] == 0:
                conflicted.discard(u)
            elif color[u] == best_c:
                conflicted.add(u)
        conflicts += best_delta
        if row[best_c] > 0:
            conflicted.add(v)
        else:
            conflicted.discard(v)
        if conflicts < best_conf:
            best_conf = conflicts
        # set tabu
        tenure = int(0.6 * conflicts) + rng.randint(1, 10)
        tabu[(v, cv)] = it + tenure
    return color


def reduce_to_k(color, k, n, adj, rng):
    """Given a valid coloring with >k colors, greedily map to k colors
    (may create conflicts) as an initial solution for tabucol."""
    new_color = [-1] * n
    order = list(range(n))
    rng.shuffle(order)
    for v in order:
        c = color[v]
        if c < k:
            new_color[v] = c
    for v in order:
        if new_color[v] == -1:
            counts = [0] * k
            for u in adj[v]:
                if new_color[u] != -1:
                    counts[new_color[u]] += 1
            mn = min(counts)
            choices = [c for c in range(k) if counts[c] == mn]
            new_color[v] = rng.choice(choices)
    return new_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 1.0

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    rng = random.Random(0)
    n, edges, adj = read_instance(args.instance_path)

    if n == 0:
        sol = {"objective_value": 0, "coloring": {}, "color_classes": {}}
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    if all(len(a) == 0 for a in adj):
        color = [0] * n
        k, sol = build_solution(n, color)
        if logger:
            logger.log_solution(k, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # Initial solution via DSATUR
    color = dsatur(n, adj)
    best_color = color[:]
    best_k, best_sol = build_solution(n, best_color)
    if logger:
        logger.log_solution(best_k, best_sol)

    # Lower bound via greedy clique
    lb = greedy_clique(n, adj, rng, tries=30,
                       deadline=min(deadline, start + 2.0))
    lb = max(lb, 2 if edges else 1)

    # Iteratively try to reduce number of colors using TabuCol
    current = best_color[:]
    k = best_k
    while k > lb and time.time() < deadline:
        target = k - 1
        init = reduce_to_k(current, target, n, adj, rng)
        result = tabucol(n, adj, target, init, deadline, rng)
        if result is not None:
            current = result
            k = target
            best_k, best_sol = build_solution(n, current)
            if logger:
                logger.log_solution(best_k, best_sol)
        else:
            # timed out on this k
            break

    # If time remains, keep retrying lower k with random restarts
    while k > lb and time.time() < deadline:
        target = k - 1
        init = reduce_to_k(current, target, n, adj, rng)
        result = tabucol(n, adj, target, init, deadline, rng)
        if result is not None:
            current = result
            k = target
            best_k, best_sol = build_solution(n, current)
            if logger:
                logger.log_solution(best_k, best_sol)
        else:
            break

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()