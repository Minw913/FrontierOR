import argparse
import json
import time
import random
from collections import defaultdict

import numpy as np

from solution_logger import SolutionLogger

EPS = 1e-9


def mst(nodes, D, need_edges=False):
    """Prim's MST over the given node indices using distance matrix D."""
    k = len(nodes)
    if k <= 1:
        return 0.0, []
    idx = np.asarray(nodes, dtype=int)
    sub = D[np.ix_(idx, idx)]
    in_tree = np.zeros(k, dtype=bool)
    in_tree[0] = True
    dist = sub[0].copy()
    parent = np.zeros(k, dtype=int)
    total = 0.0
    edges = []
    for _ in range(k - 1):
        dm = np.where(in_tree, np.inf, dist)
        j = int(np.argmin(dm))
        total += dm[j]
        if need_edges:
            edges.append([int(idx[parent[j]]), int(idx[j])])
        in_tree[j] = True
        upd = sub[j] < dist
        parent = np.where(upd, j, parent)
        dist = np.minimum(dist, sub[j])
    return float(total), edges


def build_solution(S, C, D, w):
    """Given an active set S, build full feasible solution (nearest assignment + MST)."""
    S = sorted(set(int(x) for x in S))
    arr = np.array(S, dtype=int)
    m = C.shape[0]
    if m > 0:
        sub = C[:, arr]
        idx = sub.argmin(axis=1)
        assign_cost = float(sub[np.arange(m), idx].sum())
        assignments = {str(i): int(arr[idx[i]]) for i in range(m)}
    else:
        assign_cost = 0.0
        assignments = {}
    tcost, edges = mst(S, D, need_edges=True)
    obj = assign_cost + tcost + float(w[arr].sum())
    sol = {
        "objective_value": float(obj),
        "active_steiner_nodes": [int(x) for x in S],
        "assignments": assignments,
        "steiner_tree_edges": [[int(a), int(b)] for a, b in edges],
    }
    return obj, sol


class State:
    def __init__(self, logger, C, D, w):
        self.obj = float("inf")
        self.sol = None
        self.S = None
        self.logger = logger
        self.C = C
        self.D = D
        self.w = w

    def update(self, S):
        obj, sol = build_solution(S, self.C, self.D, self.w)
        if obj < self.obj - 1e-9:
            self.obj = obj
            self.sol = sol
            self.S = sorted(set(int(x) for x in S))
            if self.logger:
                self.logger.log_solution(obj, sol)
        return obj


def set_cost(S, C, D, w):
    arr = np.array(sorted(S), dtype=int)
    a = float(C[:, arr].min(axis=1).sum()) if C.shape[0] > 0 else 0.0
    t, _ = mst(list(arr), D)
    return a + t + float(w[arr].sum())


def local_search(S0, C, D, w, deadline, state):
    m, n = C.shape
    S = sorted(set(int(x) for x in S0))
    cur = set_cost(S, C, D, w)
    state.update(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        arr = np.array(S, dtype=int)
        k = len(S)
        if m > 0:
            sub = C[:, arr]
            bidx = sub.argmin(axis=1)
            best = sub[np.arange(m), bidx]
            if k >= 2:
                sub2 = sub.copy()
                sub2[np.arange(m), bidx] = np.inf
                second = sub2.min(axis=1)
            else:
                second = np.full(m, np.inf)
            assign_cur = float(best.sum())
        else:
            bidx = np.zeros(0, dtype=int)
            best = np.zeros(0)
            second = np.zeros(0)
            assign_cur = 0.0
        mst_cur, _ = mst(S, D)
        base = assign_cur + mst_cur + float(w[arr].sum())

        best_delta = -1e-9
        best_move = None
        inS = np.zeros(n, dtype=bool)
        inS[arr] = True

        # ---- ADD moves ----
        if m > 0:
            add_assign = np.minimum(best[:, None], C).sum(axis=0)
        else:
            add_assign = np.zeros(n)
        cnt = 0
        for j in range(n):
            if inS[j]:
                continue
            cnt += 1
            if cnt % 64 == 0 and time.time() > deadline:
                break
            # lower bound: mst_new >= 0
            if add_assign[j] - assign_cur + w[j] - mst_cur >= best_delta:
                continue
            mnew, _ = mst(S + [j], D)
            delta = (add_assign[j] - assign_cur) + w[j] + (mnew - mst_cur)
            if delta < best_delta:
                best_delta = delta
                best_move = ("add", j)

        # ---- REMOVE moves ----
        if k >= 2 and time.time() < deadline:
            for p in range(k):
                j = int(arr[p])
                if m > 0:
                    ra = float(np.where(bidx == p, second, best).sum())
                    if not np.isfinite(ra):
                        continue
                else:
                    ra = 0.0
                S2 = [x for x in S if x != j]
                mnew, _ = mst(S2, D)
                delta = (ra - assign_cur) - w[j] + (mnew - mst_cur)
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("rem", j)

        # ---- SWAP moves ----
        if time.time() < deadline and n > k:
            cand = []
            for p in range(k):
                j = int(arr[p])
                if m > 0:
                    bw = np.where(bidx == p, second, best)
                    sa = np.minimum(bw[:, None], C).sum(axis=0)
                else:
                    sa = np.zeros(n)
                dn = sa - assign_cur + w - w[j]
                dn = np.where(inS, np.inf, dn)
                take = min(5, n)
                if n > take:
                    top = np.argpartition(dn, take - 1)[:take]
                else:
                    top = np.arange(n)
                for kk in top:
                    if np.isfinite(dn[kk]):
                        cand.append((float(dn[kk]), p, int(kk)))
            cand.sort()
            for dnv, p, kk in cand[:60]:
                if time.time() > deadline:
                    break
                if dnv - mst_cur >= best_delta:
                    break  # sorted, no better possible (mst_new >= 0)
                S2 = [x for x in S if x != int(arr[p])] + [kk]
                mnew, _ = mst(S2, D)
                delta = dnv + (mnew - mst_cur)
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap", int(arr[p]), kk)

        if best_move is not None:
            improved = True
            if best_move[0] == "add":
                S = sorted(S + [best_move[1]])
            elif best_move[0] == "rem":
                S = [x for x in S if x != best_move[1]]
            else:
                S = sorted([x for x in S if x != best_move[1]] + [best_move[2]])
            cur = base + best_delta
            state.update(S)
    return S, cur


def perturb(S, n, rng):
    S = set(int(x) for x in S)
    for _ in range(rng.randint(1, 2)):
        if len(S) > 1 and rng.random() < 0.6:
            S.remove(rng.choice(sorted(S)))
    outs = [j for j in range(n) if j not in S]
    for _ in range(rng.randint(1, 3)):
        outs = [j for j in outs if j not in S]
        if not outs:
            break
        S.add(rng.choice(outs))
    if not S:
        S = {rng.randrange(n)}
    return sorted(S)


def run_mip(state, C, D, w, deadline, logger):
    import gurobipy as gp
    from gurobipy import GRB

    m, n = C.shape
    remaining = deadline - time.time()
    if remaining < 3:
        return

    mod = gp.Model("stst")
    mod.Params.OutputFlag = 0
    mod.Params.Seed = 0
    mod.Params.MIPGap = 1e-4
    mod.Params.NumericFocus = 0
    mod.Params.Threads = 1
    mod.Params.TimeLimit = max(1.0, remaining - 2.0)

    y = mod.addVars(n, vtype=GRB.BINARY, obj=w.tolist(), name="y")
    if m > 0:
        x = mod.addVars(m, n, lb=0.0, obj=C.reshape(-1).tolist(), name="x")
        for i in range(m):
            mod.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
        for i in range(m):
            for j in range(n):
                mod.addConstr(x[i, j] <= y[j])

    pairs = [(j, k) for j in range(n) for k in range(j + 1, n)]
    e = mod.addVars(pairs, vtype=GRB.BINARY,
                    obj=[float(D[j][k]) for (j, k) in pairs], name="e")
    dpairs = [(j, k) for (j, k) in pairs] + [(k, j) for (j, k) in pairs]
    f = mod.addVars(dpairs, lb=0.0, name="f")
    r = mod.addVars(n, vtype=GRB.BINARY, name="r")

    mod.addConstr(gp.quicksum(r[j] for j in range(n)) == 1)
    for j in range(n):
        mod.addConstr(r[j] <= y[j])
    mod.addConstr(gp.quicksum(e[p] for p in pairs) ==
                  gp.quicksum(y[j] for j in range(n)) - 1)
    cap = max(1, n - 1)
    for (j, k) in pairs:
        mod.addConstr(e[j, k] <= y[j])
        mod.addConstr(e[j, k] <= y[k])
        mod.addConstr(f[j, k] <= cap * e[j, k])
        mod.addConstr(f[k, j] <= cap * e[j, k])
    for j in range(n):
        inflow = gp.quicksum(f[k, j] for k in range(n) if k != j)
        outflow = gp.quicksum(f[j, k] for k in range(n) if k != j)
        mod.addConstr(inflow - outflow >= y[j] - n * r[j])

    # ----- warm start from heuristic incumbent -----
    if state.sol is not None:
        S = state.sol["active_steiner_nodes"]
        Sset = set(S)
        for j in range(n):
            y[j].Start = 1.0 if j in Sset else 0.0
            r[j].Start = 0.0
        root = S[0]
        r[root].Start = 1.0
        for p in pairs:
            e[p].Start = 0.0
        for dp in dpairs:
            f[dp].Start = 0.0
        edges = state.sol["steiner_tree_edges"]
        for a, b in edges:
            e[(min(a, b), max(a, b))].Start = 1.0
        # tree flows: parent->child = subtree size of child
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        parent = {root: None}
        order = []
        stack = [root]
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    stack.append(v)
        size = {u: 1 for u in order}
        for u in reversed(order):
            if parent[u] is not None:
                size[parent[u]] += size[u]
        for u in order:
            pu = parent[u]
            if pu is not None:
                f[(pu, u)].Start = float(size[u])
        if m > 0:
            for i in range(m):
                for j in range(n):
                    x[i, j].Start = 0.0
            for i in range(m):
                x[i, int(state.sol["assignments"][str(i)])].Start = 1.0

    yvars = [y[j] for j in range(n)]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            yv = model.cbGetSolution(yvars)
            Scb = [j for j in range(n) if yv[j] > 0.5]
            if Scb:
                state.update(Scb)

    mod.optimize(cb)

    if mod.SolCount > 0:
        yv = [y[j].X for j in range(n)]
        Sfin = [j for j in range(n) if yv[j] > 0.5]
        if Sfin:
            state.update(Sfin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(3, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    m = int(data["parameters"]["m"])
    n = int(data["parameters"]["n"])
    C = np.array(data["cost_matrices"]["assignment_costs"]["matrix"],
                 dtype=float).reshape(m, n) if m > 0 else np.zeros((0, n))
    D = np.array(data["cost_matrices"]["interconnection_costs"]["matrix"],
                 dtype=float).reshape(n, n)
    w = np.array(data["steiner_nodes"]["node_weights"], dtype=float)

    rng = random.Random(0)
    np.random.seed(0)

    state = State(logger, C, D, w)

    # Greedy start: best single hub
    if m > 0:
        single = w + C.sum(axis=0)
    else:
        single = w.copy()
    S0 = [int(np.argmin(single))]
    state.update(S0)

    use_mip = (n <= 200)

    # ----- heuristic phase -----
    if use_mip:
        heur_deadline = min(deadline, start + min(15.0, 0.2 * args.time_limit))
    else:
        heur_deadline = deadline

    S, _ = local_search(S0, C, D, w, heur_deadline, state)
    while time.time() < heur_deadline - 0.5:
        Sp = perturb(state.S if state.S else S, n, rng)
        S, _ = local_search(Sp, C, D, w, heur_deadline, state)

    # ----- exact phase (MIP) -----
    if use_mip and time.time() < deadline - 3:
        try:
            run_mip(state, C, D, w, deadline, logger)
        except Exception:
            # fall back to more local search if MIP fails
            while time.time() < deadline - 0.5:
                Sp = perturb(state.S if state.S else S, n, rng)
                local_search(Sp, C, D, w, deadline, state)

    # keep improving with local search if time remains
    while time.time() < deadline - 0.5:
        Sp = perturb(state.S if state.S else S, n, rng)
        local_search(Sp, C, D, w, deadline, state)

    if state.sol is None:
        _, sol = build_solution(S0, C, D, w)
        state.sol = sol

    with open(args.solution_path, "w") as fh:
        json.dump(state.sol, fh, indent=2)


if __name__ == "__main__":
    main()