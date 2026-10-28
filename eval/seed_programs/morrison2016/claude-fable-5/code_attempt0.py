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
    n = int(g["num_vertices"])
    edges_raw = g.get("edges", [])
    edges = []
    seen = set()
    # Detect possible 1-indexing
    max_idx = -1
    min_idx = 10 ** 18
    for e in edges_raw:
        u, v = int(e[0]), int(e[1])
        if u > max_idx:
            max_idx = u
        if v > max_idx:
            max_idx = v
        if u < min_idx:
            min_idx = u
        if v < min_idx:
            min_idx = v
    offset = 0
    if edges_raw and max_idx >= n and min_idx >= 1:
        offset = 1
    for e in edges_raw:
        u, v = int(e[0]) - offset, int(e[1]) - offset
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
        edges.append((u, v))
    bounds = data.get("bounds", {}) or {}
    lb = int(bounds.get("clique_lower_bound", 1) or 1)
    ub = bounds.get("greedy_upper_bound", None)
    return n, edges, lb, ub


def build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def dsatur(n, adj):
    if n == 0:
        return []
    colors = [-1] * n
    sat = [set() for _ in range(n)]
    deg = [len(adj[v]) for v in range(n)]
    uncolored = set(range(n))
    for _ in range(n):
        best_v = -1
        best_key = (-1, -1)
        for v in uncolored:
            key = (len(sat[v]), deg[v])
            if key > best_key:
                best_key = key
                best_v = v
        v = best_v
        c = 0
        while c in sat[v]:
            c += 1
        colors[v] = c
        uncolored.discard(v)
        for u in adj[v]:
            sat[u].add(c)
    return colors


def greedy_clique(n, adj, tries=20, rng=None):
    if n == 0:
        return []
    if rng is None:
        rng = random.Random(0)
    adjset = [set(a) for a in adj]
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    best = []
    seeds = order[: min(tries, n)]
    for s in seeds:
        clique = [s]
        cand = list(adjset[s])
        cand.sort(key=lambda v: -len(adj[v]))
        for v in cand:
            ok = True
            for u in clique:
                if v not in adjset[u]:
                    ok = False
                    break
            if ok:
                clique.append(v)
        if len(clique) > len(best):
            best = clique
    return best


def normalize_colors(colors):
    used = sorted(set(colors))
    remap = {c: i for i, c in enumerate(used)}
    return [remap[c] for c in colors], len(used)


def build_solution(colors):
    cols, k = normalize_colors(colors)
    classes = [[] for _ in range(k)]
    for v, c in enumerate(cols):
        classes[c].append(v)
    for cl in classes:
        cl.sort()
    return {
        "objective_value": k,
        "selected_sets": classes,
        "coloring": {str(v): cols[v] for v in range(len(cols))},
    }


def reduce_coloring(colors, adj, k, rng):
    """Take a (k+1)-coloring and produce an initial k-coloring (possibly with conflicts)."""
    cols, kk = normalize_colors(colors)
    if kk <= k:
        return cols
    # find smallest class
    counts = [0] * kk
    for c in cols:
        counts[c] += 1
    smallest = min(range(kk), key=lambda c: counts[c])
    # relabel: move class 'smallest' to be class kk-1
    perm = list(range(kk))
    perm[smallest], perm[kk - 1] = perm[kk - 1], perm[smallest]
    inv = [0] * kk
    for i, p in enumerate(perm):
        inv[p] = i
    cols = [inv[c] for c in cols]
    # reassign vertices in class kk-1 (>= k) to best color < k
    for v in range(len(cols)):
        if cols[v] >= k:
            cnt = [0] * k
            for u in adj[v]:
                if cols[u] < k:
                    cnt[cols[u]] += 1
            m = min(cnt)
            choices = [c for c in range(k) if cnt[c] == m]
            cols[v] = rng.choice(choices)
    return cols


def tabucol(n, adj, k, init_colors, deadline, rng, stagnation_limit):
    """Try to find a conflict-free k-coloring; returns coloring or None."""
    if k <= 0:
        return None
    colors = list(init_colors)
    for v in range(n):
        if colors[v] >= k or colors[v] < 0:
            colors[v] = rng.randrange(k)
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        gv = gamma[v]
        for u in adj[v]:
            gv[colors[u]] += 1
    conflicts = 0
    for v in range(n):
        conflicts += gamma[v][colors[v]]
    conflicts //= 2
    if conflicts == 0:
        return colors
    tabu = [[0] * k for _ in range(n)]
    it = 0
    best_conf = conflicts
    last_improve = 0
    check_mask = 255
    while conflicts > 0:
        if (it & check_mask) == 0:
            if time.time() > deadline:
                return None
        it += 1
        if it - last_improve > stagnation_limit:
            return None
        best_delta = 1 << 30
        best_moves = []
        for v in range(n):
            gv = gamma[v]
            cv = colors[v]
            gcv = gv[cv]
            if gcv == 0:
                continue
            tv = tabu[v]
            for c in range(k):
                if c == cv:
                    continue
                d = gv[c] - gcv
                if d > best_delta:
                    continue
                if tv[c] > it and conflicts + d >= best_conf:
                    continue
                if d < best_delta:
                    best_delta = d
                    best_moves = [(v, c)]
                else:
                    if len(best_moves) < 64:
                        best_moves.append((v, c))
        if not best_moves:
            conf_v = [v for v in range(n) if gamma[v][colors[v]] > 0]
            if not conf_v:
                break
            v = rng.choice(conf_v)
            c = rng.randrange(k)
            while c == colors[v] and k > 1:
                c = rng.randrange(k)
            best_delta = gamma[v][c] - gamma[v][colors[v]]
            best_moves = [(v, c)]
        v, c = best_moves[rng.randrange(len(best_moves))]
        cv = colors[v]
        tabu[v][cv] = it + int(0.6 * conflicts) + rng.randrange(10) + 1
        colors[v] = c
        for u in adj[v]:
            gu = gamma[u]
            gu[cv] -= 1
            gu[c] += 1
        conflicts += best_delta
        if conflicts < best_conf:
            best_conf = conflicts
            last_improve = it
    if conflicts == 0:
        return colors
    return None


def gurobi_k_colorable(n, edges, adj, k, clique, deadline):
    """Returns (status, coloring) where status in {'feasible','infeasible','unknown'}."""
    remaining = deadline - time.time()
    if remaining < 2.0:
        return "unknown", None
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return "unknown", None
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, remaining)
        x = m.addVars(n, k, vtype=GRB.BINARY)
        for v in range(n):
            m.addConstr(gp.quicksum(x[v, c] for c in range(k)) == 1)
        for (u, v) in edges:
            for c in range(k):
                m.addConstr(x[u, c] + x[v, c] <= 1)
        # symmetry breaking via clique fixing
        for i, v in enumerate(clique[: min(len(clique), k)]):
            m.addConstr(x[v, i] == 1)
        m.setObjective(0, GRB.MINIMIZE)
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            return "infeasible", None
        if m.SolCount > 0:
            colors = [0] * n
            for v in range(n):
                for c in range(k):
                    if x[v, c].X > 0.5:
                        colors[v] = c
                        break
            return "feasible", colors
        return "unknown", None
    except Exception:
        return "unknown", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, edges, given_lb, _ = read_instance(args.instance_path)
    rng = random.Random(12345)

    if n == 0:
        sol = {"objective_value": 0, "selected_sets": [], "coloring": {}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    adj = build_adj(n, edges)

    # Initial coloring via DSATUR
    colors = dsatur(n, adj)
    best_colors, ub = normalize_colors(colors)
    best_sol = build_solution(best_colors)
    if logger:
        logger.log_solution(best_sol["objective_value"], best_sol)

    # Lower bound
    clique = greedy_clique(n, adj, tries=min(30, n), rng=rng)
    lb = max(given_lb, len(clique), 1 if n > 0 else 0)
    if not edges:
        lb = 1

    # Decide whether Gurobi exact phase is worthwhile
    gurobi_ok = (len(edges) * max(ub - 1, 1) <= 300000) and (n * max(ub - 1, 1) <= 40000)
    if gurobi_ok:
        heuristic_deadline = start + 0.6 * (deadline - start)
    else:
        heuristic_deadline = deadline

    current = list(best_colors)

    # Iteratively decrease color count using TabuCol
    while ub > lb and time.time() < heuristic_deadline:
        k = ub - 1
        stagnation = max(100000, 60 * n)
        init = reduce_coloring(current, adj, k, rng)
        found = None
        first = True
        while time.time() < heuristic_deadline:
            if first:
                cand = init
                first = False
            else:
                cand = [rng.randrange(k) for _ in range(n)]
            found = tabucol(n, adj, k, cand, heuristic_deadline, rng, stagnation)
            if found is not None:
                break
        if found is not None:
            current = found
            best_colors, ub = normalize_colors(found)
            best_sol = build_solution(best_colors)
            if logger:
                logger.log_solution(best_sol["objective_value"], best_sol)
        else:
            break

    # Exact phase with Gurobi for small instances
    if gurobi_ok and ub > lb and time.time() < deadline - 3.0:
        while ub > lb and time.time() < deadline - 2.0:
            k = ub - 1
            status, gcolors = gurobi_k_colorable(n, edges, adj, k, clique, deadline)
            if status == "feasible" and gcolors is not None:
                # verify
                valid = all(gcolors[u] != gcolors[v] for (u, v) in edges)
                if valid:
                    best_colors, ub = normalize_colors(gcolors)
                    best_sol = build_solution(best_colors)
                    if logger:
                        logger.log_solution(best_sol["objective_value"], best_sol)
                else:
                    break
            elif status == "infeasible":
                lb = ub  # proven optimal
                break
            else:
                break

    # Final verification & output
    valid = all(best_colors[u] != best_colors[v] for (u, v) in edges)
    if not valid:
        # fallback to DSATUR (always valid)
        best_colors, _ = normalize_colors(dsatur(n, adj))
        best_sol = build_solution(best_colors)
        if logger:
            logger.log_solution(best_sol["objective_value"], best_sol)
    else:
        best_sol = build_solution(best_colors)

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()