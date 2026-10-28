import argparse
import json
import random
import time

import numpy as np

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    verts = data["vertices"]
    n = len(verts)
    ids = [int(v["id"]) for v in verts]
    xs = np.array([float(v["x"]) for v in verts], dtype=np.float64)
    ys = np.array([float(v["y"]) for v in verts], dtype=np.float64)

    # ---------- trivial case ----------
    if n == 1:
        sol = {"objective_value": 0.0, "selected_edges": [],
               "vertex_costs": {str(ids[0]): 0.0}}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return

    # ---------- distances ----------
    use_matrix = n <= 4000
    if use_matrix:
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        D = dx * dx + dy * dy

        def row(j):
            return D[j]

        def w(u, v):
            return float(D[u, v])
    else:
        def row(j):
            ddx = xs - xs[j]
            ddy = ys - ys[j]
            return ddx * ddx + ddy * ddy

        def w(u, v):
            ddx = xs[u] - xs[v]
            ddy = ys[u] - ys[v]
            return float(ddx * ddx + ddy * ddy)

    # ---------- Prim MST ----------
    adj = [dict() for _ in range(n)]
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    mind = np.array(row(0), dtype=np.float64, copy=True)
    mind[0] = np.inf
    par = np.zeros(n, dtype=np.int64)
    for _ in range(n - 1):
        j = int(np.argmin(mind))
        p = int(par[j])
        wj = float(mind[j])
        adj[p][j] = wj
        adj[j][p] = wj
        in_tree[j] = True
        mind[j] = np.inf
        rj = row(j)
        upd = (~in_tree) & (rj < mind)
        mind[upd] = rj[upd]
        par[upd] = j

    c = [max(adj[t].values()) if adj[t] else 0.0 for t in range(n)]
    obj = float(sum(c))

    # ---------- helpers ----------
    parent = [0] * n
    depth = [0] * n

    def build_pd():
        parent[0] = -1
        depth[0] = 0
        seen = bytearray(n)
        seen[0] = 1
        stack = [0]
        while stack:
            u = stack.pop()
            du = depth[u] + 1
            for v2 in adj[u]:
                if not seen[v2]:
                    seen[v2] = 1
                    parent[v2] = u
                    depth[v2] = du
                    stack.append(v2)

    def path_edges(u, v):
        pe = []
        while u != v:
            if depth[u] >= depth[v]:
                pu = parent[u]
                pe.append((u, pu))
                u = pu
            else:
                pv = parent[v]
                pe.append((v, pv))
                v = pv
        return pe

    def eval_swap(u, v, wuv, a, b):
        wab = adj[a][b]
        delta = 0.0
        for t in {u, v, a, b}:
            if t == a or t == b:
                m1 = -1.0
                m2 = -1.0
                for wt in adj[t].values():
                    if wt > m1:
                        m2 = m1
                        m1 = wt
                    elif wt > m2:
                        m2 = wt
                base = m2 if wab == m1 else m1
            else:
                base = c[t]
            if t == u or t == v:
                nb = wuv if wuv > base else base
            else:
                nb = base
            delta += nb - c[t]
        return delta

    def apply_swap(u, v, wuv, a, b):
        del adj[a][b]
        del adj[b][a]
        adj[u][v] = wuv
        adj[v][u] = wuv
        for t in {u, v, a, b}:
            c[t] = max(adj[t].values())

    def make_sol(A, C, o):
        edges = []
        for u2 in range(n):
            for v2 in A[u2]:
                if u2 < v2:
                    edges.append([ids[u2], ids[v2]])
        return {"objective_value": float(o),
                "selected_edges": edges,
                "vertex_costs": {str(ids[t]): float(C[t]) for t in range(n)}}

    # ---------- candidate edges ----------
    cand = set()
    if n <= 200:
        for i in range(n):
            for j in range(i + 1, n):
                cand.add((i, j))
    else:
        K = 12
        for i in range(n):
            r = np.asarray(row(i))
            kk = min(K + 1, n - 1)
            idx = np.argpartition(r, kk)[:kk + 1]
            for j2 in idx:
                j2 = int(j2)
                if j2 != i:
                    cand.add((min(i, j2), max(i, j2)))
    cand = sorted(cand, key=lambda e: w(e[0], e[1]))

    def local_search(dl):
        nonlocal obj
        improved = True
        while improved:
            if time.time() > dl:
                return
            improved = False
            build_pd()
            cnt = 0
            for (u, v) in cand:
                cnt += 1
                if (cnt & 255) == 0 and time.time() > dl:
                    return
                if v in adj[u]:
                    continue
                wuv = w(u, v)
                pe = path_edges(u, v)
                best = -1e-9
                bestf = None
                for (a, b) in pe:
                    d = eval_swap(u, v, wuv, a, b)
                    if d < best:
                        best = d
                        bestf = (a, b)
                if bestf is not None:
                    a, b = bestf
                    apply_swap(u, v, wuv, a, b)
                    obj += best
                    improved = True
                    build_pd()
            obj = float(sum(c))

    # ---------- initial incumbent ----------
    best_adj = [dict(d) for d in adj]
    best_c = c[:]
    best_obj = obj
    if logger:
        logger.log_solution(best_obj, make_sol(best_adj, best_c, best_obj))

    try:
        local_search(deadline)
        obj = float(sum(c))
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_adj = [dict(d) for d in adj]
            best_c = c[:]
            if logger:
                logger.log_solution(best_obj, make_sol(best_adj, best_c, best_obj))
    except Exception:
        pass

    proven = (n <= 2)

    # ---------- exact MIP for small instances ----------
    if not proven and n <= 32 and time.time() < deadline - 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            E = [(i, j) for i in range(n) for j in range(i + 1, n)]
            we = [w(i, j) for (i, j) in E]
            mdl = gp.Model("mpst")
            mdl.Params.OutputFlag = 0
            mdl.Params.Seed = 0
            mdl.Params.MIPGap = 1e-4
            mdl.Params.NumericFocus = 0
            mdl.Params.Threads = 1
            mdl.Params.TimeLimit = max(1.0, deadline - time.time())

            x = mdl.addVars(len(E), vtype=GRB.BINARY, name="x")
            y = mdl.addVars(n, lb=0.0, name="y")
            fvar = mdl.addVars(2 * len(E), lb=0.0, name="f")

            for k, (i, j) in enumerate(E):
                mdl.addConstr(y[i] >= we[k] * x[k])
                mdl.addConstr(y[j] >= we[k] * x[k])
                mdl.addConstr(fvar[2 * k] <= (n - 1) * x[k])
                mdl.addConstr(fvar[2 * k + 1] <= (n - 1) * x[k])
            mdl.addConstr(gp.quicksum(x[k] for k in range(len(E))) == n - 1)

            inarc = [[] for _ in range(n)]
            outarc = [[] for _ in range(n)]
            for k, (i, j) in enumerate(E):
                outarc[i].append(2 * k)      # i -> j
                inarc[j].append(2 * k)
                outarc[j].append(2 * k + 1)  # j -> i
                inarc[i].append(2 * k + 1)
            for t in range(n):
                bal = gp.quicksum(fvar[a] for a in inarc[t]) - \
                      gp.quicksum(fvar[a] for a in outarc[t])
                if t == 0:
                    mdl.addConstr(bal == -(n - 1))
                else:
                    mdl.addConstr(bal == 1)

            mdl.setObjective(gp.quicksum(y[t] for t in range(n)), GRB.MINIMIZE)

            # warm start from heuristic incumbent
            for k, (i, j) in enumerate(E):
                x[k].Start = 1.0 if j in best_adj[i] else 0.0

            cbbest = [best_obj]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    ov = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if ov < cbbest[0] - 1e-9:
                        cbbest[0] = ov
                        try:
                            xv = model.cbGetSolution([x[k] for k in range(len(E))])
                            A = [dict() for _ in range(n)]
                            for k2, (i2, j2) in enumerate(E):
                                if xv[k2] > 0.5:
                                    A[i2][j2] = we[k2]
                                    A[j2][i2] = we[k2]
                            C = [max(A[t].values()) if A[t] else 0.0 for t in range(n)]
                            if logger:
                                logger.log_solution(float(sum(C)), make_sol(A, C, sum(C)))
                        except Exception:
                            if logger:
                                logger.log(ov)

            mdl.optimize(cb)

            if mdl.SolCount > 0:
                xv = [x[k].X for k in range(len(E))]
                A = [dict() for _ in range(n)]
                for k, (i, j) in enumerate(E):
                    if xv[k] > 0.5:
                        A[i][j] = we[k]
                        A[j][i] = we[k]
                C = [max(A[t].values()) if A[t] else 0.0 for t in range(n)]
                ov = float(sum(C))
                if ov < best_obj - 1e-9:
                    best_obj = ov
                    best_adj = A
                    best_c = C
                    if logger:
                        logger.log_solution(best_obj, make_sol(best_adj, best_c, best_obj))
            if mdl.Status == GRB.OPTIMAL:
                proven = True
        except Exception:
            pass

    # ---------- perturbation / restart loop ----------
    rng = random.Random(0)

    def perturb(kick):
        for _ in range(kick):
            edges = [(u2, v2) for u2 in range(n) for v2 in adj[u2] if u2 < v2]
            if not edges:
                return
            a, b = rng.choice(edges)
            del adj[a][b]
            del adj[b][a]
            # component containing a
            comp = set([a])
            stack = [a]
            while stack:
                u2 = stack.pop()
                for v2 in adj[u2]:
                    if v2 not in comp:
                        comp.add(v2)
                        stack.append(v2)
            u2 = rng.choice(list(comp))
            r = np.asarray(row(u2))
            order = np.argsort(r)
            choices = []
            for j2 in order:
                j2 = int(j2)
                if j2 != u2 and j2 not in comp:
                    choices.append(j2)
                    if len(choices) >= 5:
                        break
            v2 = rng.choice(choices)
            wnew = w(u2, v2)
            adj[u2][v2] = wnew
            adj[v2][u2] = wnew

    try:
        while not proven and time.time() < deadline - 0.1:
            # reset to best
            for i in range(n):
                adj[i].clear()
                adj[i].update(best_adj[i])
            perturb(rng.randint(1, 3))
            for t in range(n):
                c[t] = max(adj[t].values()) if adj[t] else 0.0
            obj = float(sum(c))
            local_search(deadline)
            obj = float(sum(c))
            if obj < best_obj - 1e-9:
                best_obj = obj
                best_adj = [dict(d) for d in adj]
                best_c = c[:]
                if logger:
                    logger.log_solution(best_obj, make_sol(best_adj, best_c, best_obj))
    except Exception:
        pass

    sol = make_sol(best_adj, best_c, best_obj)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()