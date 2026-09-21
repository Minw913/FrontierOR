import argparse
import json
import random
import time
import sys

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    g = data["graph"]
    nodes = g["nodes"]
    edges = g["edges"]
    return nodes, edges


def build_adjacency(nodes, edges):
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    adj_sets = [set() for _ in range(n)]
    for e in edges:
        u, v = idx[e[0]], idx[e[1]]
        if u == v:
            continue
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    adj = [sorted(s) for s in adj_sets]
    return adj, idx


def greedy_clique(adj, n, rng, tries=20, deadline=None):
    """Heuristic max clique -> lower bound on chromatic number."""
    best = []
    deg = [len(adj[v]) for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    starts = order[: min(n, tries)]
    for s in starts:
        if deadline and time.time() > deadline:
            break
        clique = [s]
        cand = set(adj[s])
        while cand:
            # pick candidate with max degree within cand
            v = max(cand, key=lambda x: len(cand & set(adj[x])))
            clique.append(v)
            cand &= set(adj[v])
        if len(clique) > len(best):
            best = clique
    return best


def dsatur(adj, n):
    """DSATUR greedy coloring. Returns colors list and number of colors."""
    colors = [-1] * n
    if n == 0:
        return colors, 0
    sat = [set() for _ in range(n)]  # colors of colored neighbors
    deg = [len(adj[v]) for v in range(n)]
    uncolored = set(range(n))
    num_colors = 0
    for _ in range(n):
        # pick vertex with max saturation, tie-break by degree
        best_v = -1
        best_key = (-1, -1)
        for v in uncolored:
            key = (len(sat[v]), deg[v])
            if key > best_key:
                best_key = key
                best_v = v
        v = best_v
        # smallest available color
        used = sat[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        num_colors = max(num_colors, c + 1)
        uncolored.remove(v)
        for u in adj[v]:
            if colors[u] == -1:
                sat[u].add(c)
    return colors, num_colors


def reduce_to_k(colors, k, adj, n, rng):
    """Take a coloring with >k colors and produce an initial k-coloring
    (possibly with conflicts) by reassigning vertices from removed classes."""
    new_colors = [-1] * n
    for v in range(n):
        c = colors[v]
        if c < k:
            new_colors[v] = c
    for v in range(n):
        if new_colors[v] == -1:
            # assign color minimizing conflicts
            cnt = [0] * k
            for u in adj[v]:
                cu = new_colors[u]
                if cu != -1:
                    cnt[cu] += 1
            m = min(cnt)
            choices = [c for c in range(k) if cnt[c] == m]
            new_colors[v] = rng.choice(choices)
    return new_colors


def tabucol(adj, n, k, init_colors, deadline, rng, max_stagnation=30000):
    """Tabu search for k-coloring. Returns (success, colors)."""
    colors = list(init_colors)
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        cv_adj = adj[v]
        gv = gamma[v]
        for u in cv_adj:
            gv[colors[u]] += 1
    conflicts = 0
    for v in range(n):
        conflicts += gamma[v][colors[v]]
    conflicts //= 2
    if conflicts == 0:
        return True, colors

    tabu = [[0] * k for _ in range(n)]
    it = 0
    best_conf = conflicts
    stagnation = 0
    check_interval = 128
    since_check = 0

    while conflicts > 0:
        since_check += 1
        if since_check >= check_interval:
            since_check = 0
            if time.time() > deadline:
                return False, colors
        if stagnation > max_stagnation:
            return False, colors
        it += 1

        best_delta = None
        best_moves = []
        for v in range(n):
            cv = colors[v]
            gv = gamma[v]
            if gv[cv] == 0:
                continue
            tv = tabu[v]
            gvcv = gv[cv]
            for c in range(k):
                if c == cv:
                    continue
                delta = gv[c] - gvcv
                # tabu check with aspiration
                if tv[c] > it and (conflicts + delta) >= best_conf:
                    continue
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_moves = [(v, c)]
                elif delta == best_delta and len(best_moves) < 32:
                    best_moves.append((v, c))

        if not best_moves:
            # all moves tabu: random perturbation on a conflicting vertex
            confl = [v for v in range(n) if gamma[v][colors[v]] > 0]
            if not confl:
                break
            v = rng.choice(confl)
            c = rng.randrange(k)
            while c == colors[v] and k > 1:
                c = rng.randrange(k)
            best_delta = gamma[v][c] - gamma[v][colors[v]]
        else:
            v, c = rng.choice(best_moves)

        old = colors[v]
        colors[v] = c
        for u in adj[v]:
            gu = gamma[u]
            gu[old] -= 1
            gu[c] += 1
        conflicts += best_delta
        tabu[v][old] = it + int(0.6 * conflicts) + rng.randint(0, 9)

        if conflicts < best_conf:
            best_conf = conflicts
            stagnation = 0
        else:
            stagnation += 1

    return conflicts == 0, colors


def build_solution(nodes, best_colors, k):
    # normalize colors to 0..k-1
    used = sorted(set(best_colors))
    remap = {c: i for i, c in enumerate(used)}
    kk = len(used)
    vertex_colors = {}
    color_classes = [[] for _ in range(kk)]
    for i, node in enumerate(nodes):
        c = remap[best_colors[i]]
        vertex_colors[str(node)] = c
        color_classes[c].append(node)
    return {
        "objective_value": kk,
        "vertex_colors": vertex_colors,
        "color_classes": color_classes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 1.0  # safety margin

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    nodes, edges = read_instance(args.instance_path)
    n = len(nodes)
    rng = random.Random(0)

    if n == 0:
        sol = {"objective_value": 0, "vertex_colors": {}, "color_classes": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    adj, _ = build_adjacency(nodes, edges)

    # trivial case: no edges
    if all(len(a) == 0 for a in adj):
        best_colors = [0] * n
        sol = build_solution(nodes, best_colors, 1)
        if logger:
            logger.log_solution(1, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # lower bound via clique heuristic
    clique = greedy_clique(adj, n, rng, tries=min(30, n),
                           deadline=start_time + 0.1 * args.time_limit)
    lb = max(2, len(clique))

    # initial solution via DSATUR
    best_colors, k = dsatur(adj, n)
    sol = build_solution(nodes, best_colors, k)
    if logger:
        logger.log_solution(sol["objective_value"], sol)
    k = sol["objective_value"]
    best_colors = [sol["vertex_colors"][str(node)] for node in nodes]

    # iteratively try to find (k-1)-colorings with tabu search
    while k > lb and time.time() < deadline:
        target = k - 1
        found = False
        attempt = 0
        while time.time() < deadline and not found:
            attempt += 1
            if attempt == 1:
                init = reduce_to_k(best_colors, target, adj, n, rng)
            else:
                # perturbed / random restart
                if attempt % 2 == 0:
                    init = [rng.randrange(target) for _ in range(n)]
                else:
                    init = reduce_to_k(best_colors, target, adj, n, rng)
                    # random perturbation
                    for _ in range(max(1, n // 20)):
                        init[rng.randrange(n)] = rng.randrange(target)
            stag = 30000 + 5000 * min(attempt, 10)
            ok, cols = tabucol(adj, n, target, init, deadline, rng,
                               max_stagnation=stag)
            if ok:
                found = True
                best_colors = cols
                k = target
                sol = build_solution(nodes, best_colors, k)
                k = sol["objective_value"]
                best_colors = [sol["vertex_colors"][str(node)] for node in nodes]
                if logger:
                    logger.log_solution(sol["objective_value"], sol)
        if not found:
            break

    sol = build_solution(nodes, best_colors, k)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()