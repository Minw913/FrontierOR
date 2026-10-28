import argparse
import json
import random
import sys
import time

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    g = data["graph"]
    n = int(g["num_vertices"])
    edges_raw = g.get("edges", []) or []
    adj_sets = [set() for _ in range(n)]
    for e in edges_raw:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        if 0 <= u < n and 0 <= v < n:
            adj_sets[u].add(v)
            adj_sets[v].add(u)
    adj = [sorted(s) for s in adj_sets]
    return n, adj, adj_sets


def dsatur(n, adj, adj_sets):
    """DSATUR greedy coloring. Returns color list."""
    if n == 0:
        return []
    col = [-1] * n
    sat = [set() for _ in range(n)]  # colors of colored neighbors
    deg = [len(adj[v]) for v in range(n)]
    uncolored = set(range(n))
    for _ in range(n):
        # pick vertex with max saturation, tie-break max degree
        best_v = -1
        best_key = (-1, -1)
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
        col[v] = c
        uncolored.discard(v)
        for u in adj[v]:
            if col[u] == -1:
                sat[u].add(c)
    return col


def greedy_clique_lb(n, adj_sets, rng, tries=8):
    """Greedy clique heuristic for a lower bound on chromatic number."""
    if n == 0:
        return 0
    best = 1
    order_base = sorted(range(n), key=lambda v: -len(adj_sets[v]))
    for t in range(tries):
        if t == 0:
            order = order_base
        else:
            order = list(range(n))
            rng.shuffle(order)
            order.sort(key=lambda v: -len(adj_sets[v]))
        clique = []
        cand = None
        for v in order:
            if cand is None:
                clique = [v]
                cand = set(adj_sets[v])
            elif v in cand:
                clique.append(v)
                cand &= adj_sets[v]
            if cand is not None and not cand:
                break
        best = max(best, len(clique))
    return best


def normalize(col):
    """Remap colors to 0..k-1."""
    mapping = {}
    out = []
    for c in col:
        if c not in mapping:
            mapping[c] = len(mapping)
        out.append(mapping[c])
    return out, len(mapping)


def build_solution(col):
    col, k = normalize(col)
    coloring = {str(v): int(c) for v, c in enumerate(col)}
    classes = {}
    for v, c in enumerate(col):
        classes.setdefault(str(c), []).append(v)
    return {
        "objective_value": k,
        "coloring": coloring,
        "color_classes": classes,
    }


def init_k_coloring(col, k, adj, rng):
    """Given a valid (k+1)-coloring, produce an initial k-coloring (maybe conflicting)."""
    n = len(col)
    # count class sizes
    from collections import Counter
    cnt = Counter(col)
    # remove the smallest class -> fewest vertices to reassign
    remove_c = min(cnt, key=lambda c: cnt[c])
    # remap: colors other than remove_c to 0..k-1
    remap = {}
    for c in sorted(cnt):
        if c != remove_c:
            remap[c] = len(remap)
    newcol = [0] * n
    to_fix = []
    for v in range(n):
        if col[v] == remove_c:
            to_fix.append(v)
        else:
            newcol[v] = remap[col[v]]
    for v in to_fix:
        # assign color minimizing conflicts among neighbors
        counts = [0] * k
        for u in adj[v]:
            if u not in to_fix or u < v:
                pass
        for u in adj[v]:
            counts[newcol[u]] += 1
        m = min(counts)
        choices = [c for c in range(k) if counts[c] == m]
        newcol[v] = rng.choice(choices)
    return newcol


def tabucol(n, adj, k, col, deadline, rng, max_iters=2_000_000):
    """Tabu search for a proper k-coloring. col is initial assignment (modified in place).
    Returns valid coloring list or None if failed within deadline."""
    if k <= 0:
        return None
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        cv = col[v]
        for u in adj[v]:
            gamma[u][cv] += 1
    conflicts = 0
    for v in range(n):
        conflicts += gamma[v][col[v]]
    conflicts //= 2
    if conflicts == 0:
        return col
    tabu = [[0] * k for _ in range(n)]
    it = 0
    best_conf = conflicts
    INF = float("inf")
    while conflicts > 0 and it < max_iters:
        it += 1
        if (it & 255) == 0 and time.time() > deadline:
            return None
        best_delta = INF
        moves = []
        for v in range(n):
            gv = gamma[v]
            cv = col[v]
            base = gv[cv]
            if base == 0:
                continue
            tv = tabu[v]
            for c in range(k):
                if c == cv:
                    continue
                d = gv[c] - base
                if d > best_delta:
                    continue
                if tv[c] > it and conflicts + d >= best_conf:
                    continue
                if d < best_delta:
                    best_delta = d
                    moves = [(v, c)]
                else:
                    moves.append((v, c))
        if not moves:
            # everything tabu: random perturbation on a conflicting vertex
            confv = [v for v in range(n) if gamma[v][col[v]] > 0]
            if not confv:
                break
            v = rng.choice(confv)
            c = rng.randrange(k)
            while c == col[v] and k > 1:
                c = rng.randrange(k)
            best_delta = gamma[v][c] - gamma[v][col[v]]
        else:
            v, c = moves[rng.randrange(len(moves))]
        old = col[v]
        col[v] = c
        for u in adj[v]:
            gu = gamma[u]
            gu[old] -= 1
            gu[c] += 1
        conflicts += best_delta
        tenure = rng.randint(0, 9) + int(0.6 * conflicts)
        tabu[v][old] = it + tenure
        if conflicts < best_conf:
            best_conf = conflicts
    if conflicts == 0:
        return col
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)

    n, adj, adj_sets = read_instance(args.instance_path)

    if n == 0:
        sol = {"objective_value": 0, "coloring": {}, "color_classes": {}}
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # Initial DSATUR coloring
    best_col = dsatur(n, adj, adj_sets)
    best_col, best_k = normalize(best_col)
    best_sol = build_solution(best_col)
    if logger:
        logger.log_solution(best_sol["objective_value"], best_sol)

    lb = greedy_clique_lb(n, adj_sets, rng)
    num_edges = sum(len(a) for a in adj) // 2
    if num_edges == 0:
        lb = 1

    # Iteratively reduce k using TabuCol
    k = best_k - 1
    while k >= lb and time.time() < deadline:
        success = False
        # multiple restarts on current k until deadline
        while time.time() < deadline:
            init = init_k_coloring(best_col, k, adj, rng)
            res = tabucol(n, adj, k, init, deadline, rng)
            if res is not None:
                best_col, best_k = normalize(res)
                best_sol = build_solution(best_col)
                if logger:
                    logger.log_solution(best_sol["objective_value"], best_sol)
                success = True
                break
        if not success:
            break
        k = best_k - 1

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()