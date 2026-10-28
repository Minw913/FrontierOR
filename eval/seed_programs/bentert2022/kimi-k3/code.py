import argparse
import json
import time
from collections import deque

import numpy as np

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:
    HAVE_GUROBI = False


# ----------------------------- helpers ---------------------------------- #

def norm(a, b):
    return (a, b) if a < b else (b, a)


def pair_w(a, b, X, Y):
    dx = X[a] - X[b]
    dy = Y[a] - Y[b]
    return float(dx * dx + dy * dy)


def prim_mst(X, Y):
    """O(n^2) Prim, returns list of (a, b, w) with a < b."""
    n = len(X)
    if n < 2:
        return []
    in_tree = np.zeros(n, dtype=bool)
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    dist[0] = 0.0
    for _ in range(n):
        tmp = np.where(in_tree, np.inf, dist)
        v = int(np.argmin(tmp))
        in_tree[v] = True
        dx = X - X[v]
        dy = Y - Y[v]
        dv = dx * dx + dy * dy
        mask = (~in_tree) & (dv < dist)
        idx = np.nonzero(mask)[0]
        parent[idx] = v
        dist[idx] = dv[idx]
    edges = []
    for v in range(1, n):
        p = int(parent[v])
        a, b = norm(p, v)
        edges.append((a, b, float(dist[v])))
    return edges


def knn_pairs(X, Y, k, block=1000):
    """Returns sorted unique normalized pairs among the k nearest neighbors."""
    n = len(X)
    if n < 2 or k < 1:
        return []
    k = min(k, n - 1)
    pairs = set()
    for s in range(0, n, block):
        e = min(s + block, n)
        bs = e - s
        xb = X[s:e]
        yb = Y[s:e]
        D = (X[None, :] - xb[:, None]) ** 2 + (Y[None, :] - yb[:, None]) ** 2
        D[np.arange(bs), np.arange(s, e)] = np.inf
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        for r in range(bs):
            i = s + r
            for j in idx[r]:
                j = int(j)
                pairs.add(norm(i, j))
    return sorted(pairs)


# ----------------------------- local search ----------------------------- #

def local_search(n, X, Y, tree_edges, cand_edges, deadline, logger,
                 last_logged, build_solution):
    adj = [[] for _ in range(n)]
    edgeW = {}
    tree_set = set()
    for a, b, w in tree_edges:
        adj[a].append((b, w))
        adj[b].append((a, w))
        tree_set.add((a, b))
        edgeW[(a, b)] = w

    max1 = [0.0] * n
    max2 = [0.0] * n
    cnt1 = [0] * n

    def recalc(z):
        m1 = 0.0
        m2 = 0.0
        c = 0
        for _, w in adj[z]:
            if w > m1:
                m2 = m1
                m1 = w
                c = 1
            elif w == m1:
                c += 1
            elif w > m2:
                m2 = w
        max1[z] = m1
        max2[z] = m2
        cnt1[z] = c

    for z in range(n):
        recalc(z)

    def drop_remove(z, w):
        """Cost decrease at z if its incident edge of weight w is removed."""
        if cnt1[z] == 1 and w == max1[z]:
            return max1[z] - max2[z]
        return 0.0

    parent = [-1] * n
    depth = [0] * n
    parW = [0.0] * n
    parDrop = [0.0] * n

    def reroot():
        for i in range(n):
            parent[i] = -1
        parent[0] = -2
        depth[0] = 0
        dq = deque([0])
        while dq:
            z = dq.popleft()
            dz = depth[z] + 1
            for nb, w in adj[z]:
                if parent[nb] == -1:
                    parent[nb] = z
                    depth[nb] = dz
                    parW[nb] = w
                    parDrop[nb] = drop_remove(z, w) + drop_remove(nb, w)
                    dq.append(nb)

    reroot()

    def delta_swap(u, v, w, a, b, wf):
        roles = {}
        roles[u] = [w, None]
        roles[v] = [w, None]
        for z in (a, b):
            if z in roles:
                roles[z][1] = wf
            else:
                roles[z] = [None, wf]
        d = 0.0
        for z, (addw, delw) in roles.items():
            base = max1[z]
            if delw is not None and cnt1[z] == 1 and delw == max1[z]:
                base = max2[z]
            if addw is not None and addw > base:
                base = addw
            d += base - max1[z]
        return d

    extra = []
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        worklist = cand_edges + extra
        extra = []
        for idx, (a, b, w) in enumerate(worklist):
            if (idx & 63) == 0 and time.monotonic() >= deadline:
                extra = []
                improved = False
                break
            if (a, b) in tree_set:
                continue
            ua = w - max1[a]
            if ua < 0.0:
                ua = 0.0
            ub = w - max1[b]
            if ub < 0.0:
                ub = 0.0
            if ua + ub <= 1e-12:
                pass  # possibly good, evaluate
            # find max drop edge on a-b path
            u, v = a, b
            bestd = 0.0
            bestnode = -1
            while u != v:
                if depth[u] >= depth[v]:
                    if parDrop[u] > bestd:
                        bestd = parDrop[u]
                        bestnode = u
                    u = parent[u]
                else:
                    if parDrop[v] > bestd:
                        bestd = parDrop[v]
                        bestnode = v
                    v = parent[v]
            if bestd - ua - ub <= 1e-9:
                continue
            p = bestnode
            q = parent[p]
            wf = parW[p]
            d = delta_swap(a, b, w, p, q, wf)
            if d < -1e-9:
                # apply swap: remove (p,q), add (a,b)
                tree_set.discard(norm(p, q))
                key = norm(a, b)
                tree_set.add(key)
                edgeW[key] = w
                edgeW.pop(norm(p, q), None)
                adj[p].remove((q, wf))
                adj[q].remove((p, wf))
                if wf > max1[p]:
                    pass
                adj[a].append((b, w))
                adj[b].append((a, w))
                recalc(a)
                recalc(b)
                recalc(p)
                recalc(q)
                reroot()
                extra.append((min(p, q), max(p, q), wf))
                improved = True
        if improved and logger:
            cur = float(sum(max1))
            if cur < last_logged - 1e-9:
                edges_now = [(a, b, edgeW[(a, b)]) for (a, b) in tree_set]
                o, sol = build_solution(edges_now)
                logger.log_solution(o, sol)
                last_logged = o

    final_edges = [(a, b, edgeW[(a, b)]) for (a, b) in tree_set]
    final_obj = float(sum(max1))
    return final_edges, final_obj, last_logged


# ------------------------------- Gurobi MIP ------------------------------ #

def mip_solve(n, edges, warm_set, time_limit, logger, last_logged,
              build_solution):
    model = gp.Model("power_tree")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    cap = float(n - 1) if n > 1 else 1.0
    x = {}
    for a, b, w in edges:
        x[(a, b)] = model.addVar(vtype=GRB.BINARY, name="x%d_%d" % (a, b))
    c = [model.addVar(lb=0.0, name="c%d" % v) for v in range(n)]
    model.setObjective(gp.quicksum(c), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(x.values()) == n - 1)

    arc_in = [[] for _ in range(n)]
    arc_out = [[] for _ in range(n)]
    for a, b, w in edges:
        xv = x[(a, b)]
        model.addConstr(c[a] >= w * xv)
        model.addConstr(c[b] >= w * xv)
        fab = model.addVar(lb=0.0)
        fba = model.addVar(lb=0.0)
        model.addConstr(fab <= cap * xv)
        model.addConstr(fba <= cap * xv)
        arc_out[a].append(fab)
        arc_in[b].append(fab)
        arc_out[b].append(fba)
        arc_in[a].append(fba)

    root = 0
    for v in range(n):
        if v == root:
            continue
        model.addConstr(gp.quicksum(arc_in[v]) - gp.quicksum(arc_out[v]) == 1)

    for a, b, w in edges:
        x[(a, b)].Start = 1.0 if (a, b) in warm_set else 0.0

    state = {"best": last_logged}

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
            except Exception:
                return
            if obj < state["best"] - 1e-7:
                sel = []
                for a, b, w in edges:
                    try:
                        if m.cbGetSolution(x[(a, b)]) > 0.5:
                            sel.append((a, b, w))
                    except Exception:
                        pass
                if len(sel) == max(0, n - 1):
                    o, sol = build_solution(sel)
                    state["best"] = min(state["best"], o)
                    if logger:
                        logger.log_solution(o, sol)

    model.Params.TimeLimit = max(1, int(time_limit))
    try:
        model.optimize(callback)
    except Exception:
        return None

    if model.SolCount > 0:
        sel = [(a, b, w) for a, b, w in edges if x[(a, b)].X > 0.5]
        if len(sel) == max(0, n - 1):
            o, sol = build_solution(sel)
            return o, sel, sol, state["best"]
    return None


# --------------------------------- main ---------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.monotonic()
    T = max(1, args.time_limit)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    vertices = data.get("vertices", [])
    ids = [int(v["id"]) for v in vertices]
    X = np.array([float(v["x"]) for v in vertices], dtype=float)
    Y = np.array([float(v["y"]) for v in vertices], dtype=float)
    n = len(ids)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def build_solution(edge_list):
        costs = [0.0] * n
        for a, b, w in edge_list:
            if w > costs[a]:
                costs[a] = w
            if w > costs[b]:
                costs[b] = w
        obj = float(sum(costs))
        sol = {
            "objective_value": obj,
            "selected_edges": [[int(ids[a]), int(ids[b])] for a, b, _ in edge_list],
            "vertex_costs": {str(ids[i]): float(costs[i]) for i in range(n)},
        }
        return obj, sol

    # --- initial MST solution ---
    mst_edges = prim_mst(X, Y)
    best_obj, best_sol = build_solution(mst_edges)
    best_edges = mst_edges
    last_logged = best_obj
    if logger:
        logger.log_solution(best_obj, best_sol)

    # --- local search improvement ---
    elapsed = time.monotonic() - t0
    remaining = T - elapsed
    if n > 2 and remaining > 0.5:
        if remaining > 4.0:
            ls_time = min(remaining * 0.35, 30.0)
        else:
            ls_time = max(0.2, remaining * 0.4)
        deadline = time.monotonic() + ls_time
        tree_set0 = set((a, b) for a, b, _ in mst_edges)
        neigh_pairs = knn_pairs(X, Y, min(n - 1, 16))
        cand = []
        for a, b in neigh_pairs:
            if (a, b) not in tree_set0:
                cand.append((a, b, pair_w(a, b, X, Y)))
        cand.sort(key=lambda e: e[2])
        ls_edges, ls_obj, last_logged = local_search(
            n, X, Y, mst_edges, cand, deadline, logger, last_logged,
            build_solution)
        if ls_obj < best_obj - 1e-9:
            best_obj, best_sol = build_solution(ls_edges)
            best_edges = ls_edges

    # --- restricted MIP polish via Gurobi ---
    elapsed = time.monotonic() - t0
    remaining = T - elapsed
    if HAVE_GUROBI and n > 1 and remaining > 4.0:
        warm_set = set((a, b) for a, b, _ in best_edges)
        if n <= 244:
            pair_set = set()
            for a in range(n):
                for b in range(a + 1, n):
                    pair_set.add((a, b))
        else:
            k = max(8, min(n - 1, 60000 // max(1, n)))
            pair_set = set(knn_pairs(X, Y, k))
        pair_set |= warm_set
        edges = [(a, b, pair_w(a, b, X, Y)) for (a, b) in sorted(pair_set)]
        res = mip_solve(n, edges, warm_set, remaining - 1.5, logger,
                        last_logged, build_solution)
        if res is not None:
            o, sel, sol, _ = res
            if o < best_obj - 1e-9:
                best_obj, best_sol, best_edges = o, sol, sel

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()