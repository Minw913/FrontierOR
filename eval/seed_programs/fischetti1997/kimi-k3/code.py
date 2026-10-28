import argparse
import json
import os
import random
import time

import numpy as np


# --------------------------------------------------------------------------- #
# Instance loading
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Generalized TSP (GTSP) solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    raw_nodes = data["nodes"]
    nodes = sorted(raw_nodes, key=lambda r: r["id"])
    n = len(nodes)
    id_of_idx = [int(r["id"]) for r in nodes]
    idx_of_id = {nid: i for i, nid in enumerate(id_of_idx)}

    clusters = sorted(data["clusters"], key=lambda c: c["id"])
    m = len(clusters)
    clusters_idx = []
    cluster_of = np.zeros(n, dtype=np.int64)
    for ci, c in enumerate(clusters):
        ids = c.get("node_ids")
        if not ids:
            ids = [r["id"] for r in raw_nodes if r.get("cluster") == c["id"]]
        arr = np.array([idx_of_id[int(v)] for v in ids], dtype=np.int64)
        clusters_idx.append(arr)
        cluster_of[arr] = ci

    D = None
    dmt = data.get("distance_matrix_upper_triangular")
    if isinstance(dmt, dict) and dmt.get("values") is not None:
        vals = np.asarray(dmt["values"], dtype=np.int64)
        if n > 1 and vals.shape[0] == n * (n - 1) // 2:
            D = np.zeros((n, n), dtype=np.int64)
            pos = 0
            for i in range(n - 1):
                k = n - 1 - i
                seg = vals[pos:pos + k]
                D[i, i + 1:] = seg
                D[i + 1:, i] = seg
                pos += k
    if D is None:  # fallback: compute rounded Euclidean distances
        xs = np.array([float(r["x"]) for r in nodes], dtype=np.float64)
        ys = np.array([float(r["y"]) for r in nodes], dtype=np.float64)
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        D = np.floor(np.hypot(dx, dy) + 0.5).astype(np.int64)
    return id_of_idx, clusters_idx, cluster_of, D, n, m


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def tour_cost(tour, D):
    if len(tour) <= 1:
        return 0
    total = 0
    prev = tour[-1]
    for v in tour:
        total += int(D[prev, v])
        prev = v
    return total


def greedy_nn(starts, D, cluster_of, m):
    """Clustered nearest-neighbour construction from several start nodes."""
    best_tour, best_cost = None, None
    BIG = np.iinfo(np.int64).max
    for s in starts:
        visited = np.zeros(m, dtype=bool)
        tour = [int(s)]
        visited[cluster_of[s]] = True
        cur = int(s)
        for _ in range(m - 1):
            row = D[cur].copy()
            row[visited[cluster_of]] = BIG
            cur = int(np.argmin(row))
            tour.append(cur)
            visited[cluster_of[cur]] = True
        c = tour_cost(tour, D)
        if best_cost is None or c < best_cost:
            best_tour, best_cost = tour, c
    return best_tour, best_cost


# --------------------------------------------------------------------------- #
# Local search moves
# --------------------------------------------------------------------------- #
def two_opt(tour, D, deadline):
    """In-place cyclic 2-opt on the tour order. Returns new tour cost."""
    m = len(tour)
    if m < 4:
        return tour_cost(tour, D)
    T = np.asarray(tour, dtype=np.int64)
    total = tour_cost(tour, D)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(m - 2):
            a = T[i]
            b = T[i + 1]
            js = np.arange(i + 2, m)
            if i == 0:
                js = js[:-1]  # skip move adjacent through the wrap-around edge
            if js.size == 0:
                continue
            js1 = js + 1
            js1[js1 == m] = 0
            Tj = T[js]
            Tj1 = T[js1]
            delta = D[a, Tj] + D[b, Tj1] - D[a, b] - D[Tj, Tj1]
            k = int(np.argmin(delta))
            if delta[k] < 0:
                j = int(js[k])
                T[i + 1:j + 1] = T[i + 1:j + 1][::-1]
                total += int(delta[k])
                improved = True
    tour[:] = T.tolist()
    return total


def cluster_opt_dp(tour, D, layers, deadline):
    """Exact representative selection for a fixed cluster order (cyclic layered
    shortest path). Returns (new_tour, new_cost); never worse than input."""
    m = len(tour)
    L0 = layers[0]
    best_total = tour_cost(tour, D)
    best_assign = list(tour)
    for si in range(L0.shape[0]):
        if time.time() > deadline:
            break
        s = int(L0[si])
        dp = D[s, layers[1]].astype(np.int64)
        parents = [None] * m
        for k in range(2, m):
            sub = D[np.ix_(layers[k - 1], layers[k])].astype(np.int64)
            M = dp[:, None] + sub
            par = np.argmin(M, axis=0)
            dp = M[par, np.arange(M.shape[1])]
            parents[k] = par
        close = dp + D[layers[-1], s].astype(np.int64)
        t = int(np.argmin(close))
        tot = int(close[t])
        if tot < best_total:
            best_total = tot
            sel = [0] * m
            sel[m - 1] = t
            idx = t
            for k in range(m - 1, 1, -1):
                idx = int(parents[k][idx])
                sel[k - 1] = idx
            sel[0] = si
            best_assign = [int(layers[k][sel[k]]) for k in range(m)]
    return best_assign, best_total


def cluster_opt_sweep(tour, D, layers, deadline):
    """Cheap best-representative sweep given current neighbours."""
    m = len(tour)
    t = list(tour)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(m):
            a = t[p - 1]
            b = t[(p + 1) % m]
            nodes = layers[p]
            cand = D[a, nodes].astype(np.int64) + D[nodes, b].astype(np.int64)
            k = int(np.argmin(cand))
            cur_c = int(D[a, t[p]]) + int(D[t[p], b])
            if cand[k] < cur_c:
                t[p] = int(nodes[k])
                improved = True
    return t, tour_cost(t, D)


def representative_opt(tour, D, clusters_idx, cluster_of, deadline):
    m = len(tour)
    layers = [clusters_idx[int(cluster_of[v])] for v in tour]
    pairs = 0
    for k in range(1, m):
        pairs += len(layers[k - 1]) * len(layers[k])
    est = len(layers[0]) * max(1, pairs)
    if est <= 4_000_000:
        return cluster_opt_dp(tour, D, layers, deadline)
    return cluster_opt_sweep(tour, D, layers, deadline)


def local_search(tour, D, clusters_idx, cluster_of, deadline):
    """Alternate order optimization (2-opt) and representative optimization."""
    m = len(tour)
    if m == 1:
        return list(tour), 0
    tour = list(tour)
    cur = tour_cost(tour, D)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        t2, c2 = representative_opt(tour, D, clusters_idx, cluster_of, deadline)
        if c2 < cur:
            tour, cur = t2, c2
            improved = True
        c3 = two_opt(tour, D, deadline)
        if c3 < cur:
            cur = c3
            improved = True
    return tour, cur


# --------------------------------------------------------------------------- #
# Iterated local search
# --------------------------------------------------------------------------- #
def perturb(tour, rng, clusters_idx, cluster_of):
    m = len(tour)
    t = list(tour)
    if m >= 4:  # double-bridge move
        a, b, c = sorted(rng.sample(range(1, m), 3))
        t = t[:a] + t[b:c] + t[a:b] + t[c:]
    k = max(1, m // 10)  # re-randomize some cluster representatives
    for p in rng.sample(range(m), min(k, m)):
        nodes = clusters_idx[int(cluster_of[t[p]])]
        if len(nodes) > 1:
            t[p] = int(nodes[rng.randrange(len(nodes))])
    return t


def ils(best_tour, best_cost, D, clusters_idx, cluster_of, rng, deadline,
        log_solution):
    m = len(best_tour)
    if m <= 2:
        return best_tour, best_cost
    current = list(best_tour)
    current_cost = best_cost
    while time.time() < deadline:
        cand = perturb(current, rng, clusters_idx, cluster_of)
        cand, c = local_search(cand, D, clusters_idx, cluster_of, deadline)
        if c < best_cost - 1e-9:
            best_tour, best_cost = cand, c
            log_solution(c, cand)
            current, current_cost = cand, c
        elif c <= current_cost + max(2.0, 0.01 * abs(current_cost)):
            current, current_cost = cand, c
        else:
            current, current_cost = list(best_tour), best_cost
    return best_tour, best_cost


# --------------------------------------------------------------------------- #
# Exact MIP (MTZ formulation) via Gurobi, with warm start
# --------------------------------------------------------------------------- #
def try_mip(D, clusters_idx, cluster_of, n, m, warm_tour, deadline,
            log_solution):
    if deadline - time.time() < 5.0 or m < 3:
        return None
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None
    try:
        model = gp.Model("gtsp")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0

        nbrs = [np.nonzero(cluster_of != cluster_of[i])[0].tolist()
                for i in range(n)]
        arcs = []
        costs = {}
        for i in range(n):
            Di = D[i]
            for j in nbrs[i]:
                arcs.append((i, j))
                costs[(i, j)] = float(Di[j])
        a = model.addVars(arcs, obj=costs, vtype=GRB.BINARY, name="a")
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        u = model.addVars(n, lb=0.0, ub=float(m - 1),
                          vtype=GRB.CONTINUOUS, name="u")

        for i in range(n):
            nb = nbrs[i]
            model.addConstr(gp.quicksum(a[i, j] for j in nb) == y[i])
            model.addConstr(gp.quicksum(a[j, i] for j in nb) == y[i])
        for nodes in clusters_idx:
            model.addConstr(gp.quicksum(y[int(v)] for v in nodes) == 1.0)

        root = np.zeros(n, dtype=bool)
        root[clusters_idx[0]] = True
        mf = float(m)
        model.addConstrs(
            (u[i] - u[j] + mf * a[i, j] <= mf - 1.0
             for (i, j) in arcs if (not root[i]) and (not root[j])),
            name="mtz")
        model.update()

        # warm start from the heuristic incumbent
        for k in range(m):
            v = int(warm_tour[k])
            w = int(warm_tour[(k + 1) % m])
            y[v].Start = 1.0
            u[v].Start = float(k)
            a[v, w].Start = 1.0

        arc_vars = [a[key] for key in arcs]
        y_vars = [y[i] for i in range(n)]
        root_nodes = [int(v) for v in clusters_idx[0]]

        def reconstruct(vals, yvals):
            succ = {}
            for idx, val in enumerate(vals):
                if val > 0.5:
                    succ[arcs[idx][0]] = arcs[idx][1]
            r = -1
            for v in root_nodes:
                if yvals[v] > 0.5:
                    r = v
                    break
            if r < 0:
                return None
            tour = [r]
            for _ in range(m - 1):
                r = succ.get(r, -1)
                if r < 0:
                    return None
                tour.append(r)
            if len(set(tour)) != m:
                return None
            return tour

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    vals = model.cbGetSolution(arc_vars)
                    yvals = model.cbGetSolution(y_vars)
                    tour = reconstruct(vals, yvals)
                    if tour is not None:
                        log_solution(tour_cost(tour, D), tour)
                except Exception:
                    pass

        model.Params.TimeLimit = max(1.0, deadline - time.time() - 0.5)
        model.optimize(callback)

        optimal = (model.Status == GRB.OPTIMAL)
        out = None
        if model.SolCount > 0:
            try:
                vals = model.getAttr("X", arc_vars)
                yvals = model.getAttr("X", y_vars)
                tour = reconstruct(vals, yvals)
                if tour is not None:
                    out = (tour, tour_cost(tour, D), optimal)
            except Exception:
                out = None
        del model
        return out
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    t_start = time.time()
    time_limit = max(1, int(args.time_limit))
    buffer = max(0.3, min(2.0, 0.03 * time_limit))
    deadline = t_start + time_limit - buffer

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    id_of_idx, clusters_idx, cluster_of, D, n, m = load_instance(
        args.instance_path)
    rng = random.Random(0)

    def sol_fn(tour, cost):
        reps = [0] * m
        for v in tour:
            reps[int(cluster_of[v])] = int(v)
        return {
            "objective_value": float(cost),
            "tour": [int(id_of_idx[v]) for v in tour],
            "visited_nodes": [int(id_of_idx[reps[c]]) for c in range(m)],
        }

    log_state = {"best": float("inf")}

    def log_solution(cost, tour):
        if logger is not None and tour is not None and \
                cost < log_state["best"] - 1e-9:
            log_state["best"] = cost
            try:
                logger.log_solution(float(cost), sol_fn(tour, cost))
            except Exception:
                try:
                    logger.log(float(cost))
                except Exception:
                    pass

    # trivial feasible fallback (first node of every cluster)
    best_tour = [int(c[0]) for c in clusters_idx]
    best_cost = tour_cost(best_tour, D)

    try:
        if m == 1:
            best_tour = [int(clusters_idx[0][0])]
            best_cost = 0
            log_solution(0.0, best_tour)
        else:
            # --- construction + local search ----------------------------- #
            starts = [int(v) for v in clusters_idx[0][:6]]
            init_tour, init_cost = greedy_nn(starts, D, cluster_of, m)
            log_solution(init_cost, init_tour)

            n_arcs = n * n - int(sum(len(c) * len(c) for c in clusters_idx))
            remaining = deadline - time.time()
            plan_mip = (3 <= m <= 150 and n_arcs <= 45000
                        and remaining >= 10.0)

            ls_deadline = deadline
            if plan_mip:  # keep most of the budget for the exact solve
                ls_deadline = time.time() + min(20.0, max(1.0, 0.2 * remaining))
            best_tour, best_cost = local_search(
                init_tour, D, clusters_idx, cluster_of, ls_deadline)
            log_solution(best_cost, best_tour)

            # --- exact MIP on small/medium instances --------------------- #
            mip_optimal = False
            if plan_mip and deadline - time.time() >= 5.0:
                res = try_mip(D, clusters_idx, cluster_of, n, m, best_tour,
                              deadline, log_solution)
                if res is not None:
                    t2, c2, mip_optimal = res
                    if t2 is not None and c2 < best_cost:
                        best_tour, best_cost = t2, c2
                        log_solution(best_cost, best_tour)

            # --- iterated local search with the remaining time ----------- #
            if not mip_optimal and m >= 3 and time.time() < deadline:
                best_tour, best_cost = ils(
                    best_tour, best_cost, D, clusters_idx, cluster_of, rng,
                    deadline, log_solution)
    except Exception:
        pass

    best_cost = tour_cost(best_tour, D)
    log_solution(best_cost, best_tour)
    solution = sol_fn(best_tour, best_cost)

    out_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()