import argparse
import json
import sys
import time
from itertools import combinations


def main():
    sys.setrecursionlimit(200000)
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    t_start = time.time()
    deadline = t_start + max(3, args.time_limit) - 1.5

    with open(args.instance_path) as f:
        data = json.load(f)
    verts = data["vertices"]
    n = len(verts)
    k = int(data["interdiction_budget_k"])
    idx = {v: i for i, v in enumerate(verts)}
    pair2eid = {}
    edges = []
    orig_edge = []
    for e in data["edges"]:
        u = idx[e[0]]
        v = idx[e[1]]
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in pair2eid:
            continue
        pair2eid[(a, b)] = len(edges)
        edges.append((a, b))
        orig_edge.append([e[0], e[1]])
    m = len(edges)
    adj = [0] * max(n, 1)
    for (a, b) in edges:
        adj[a] |= 1 << b
        adj[b] |= 1 << a

    try:
        (0).bit_count()

        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count("1")

    best = {"obj": None, "F": []}

    def write_solution():
        obj = best["obj"]
        if obj is None:
            obj = 1 if n >= 1 else 0
        sol = {"objective_value": int(obj),
               "interdicted_edges": [list(orig_edge[e]) for e in best["F"]]}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return sol

    def record(obj, F):
        best["obj"] = int(obj)
        best["F"] = list(F)
        sol = write_solution()
        if logger:
            try:
                logger.log_solution(int(obj), sol)
            except Exception:
                pass

    # ---------------- trivial cases ----------------
    if n == 0:
        record(0, [])
        return
    if m == 0:
        record(1, [])
        return
    if k >= m:
        record(1, list(range(m)))
        return

    # ---------------- max clique branch & bound (bitset, Tomita-style) ----------------
    def max_clique(adjl, lb, target, dl, init_best=None):
        """Return (best_clique, timed_out). Only records cliques strictly larger
        than max(lb, len(init_best)). Stops early once size >= target (if given)."""
        st_best = list(init_best) if init_best else []
        st = {"blen": max(lb, len(st_best)), "best": st_best,
              "timeout": False, "stop": False, "cnt": 0}
        if target is not None and st["blen"] >= target:
            return st["best"], False

        def expand(R, P):
            st["cnt"] += 1
            if (st["cnt"] & 127) == 0 and time.time() > dl:
                st["timeout"] = True
                st["stop"] = True
                return
            order = []
            cols = []
            uncolored = P
            c = 0
            while uncolored:
                c += 1
                Q = uncolored
                while Q:
                    bbit = Q & -Q
                    v = bbit.bit_length() - 1
                    Q &= ~adjl[v]
                    Q &= ~bbit
                    uncolored &= ~bbit
                    order.append(v)
                    cols.append(c)
            rlen = len(R)
            for i in range(len(order) - 1, -1, -1):
                if st["stop"]:
                    return
                if rlen + cols[i] <= st["blen"]:
                    return
                v = order[i]
                R.append(v)
                P2 = P & adjl[v]
                if P2:
                    expand(R, P2)
                else:
                    if rlen + 1 > st["blen"]:
                        st["blen"] = rlen + 1
                        st["best"] = R[:]
                        if target is not None and st["blen"] >= target:
                            st["stop"] = True
                R.pop()
                if st["stop"]:
                    return
                P &= ~(1 << v)

        mask = 0
        for v in range(n):
            mask |= 1 << v
        expand([], mask)
        return st["best"], st["timeout"]

    def degree_order(adjl):
        return sorted(range(n), key=lambda v: -popcount(adjl[v]))

    def greedy_from(adjl, order, v0):
        C = [v0]
        P = adjl[v0]
        if P:
            for v in order:
                if (P >> v) & 1:
                    C.append(v)
                    P &= adjl[v]
                    if not P:
                        break
        return C

    def extend_maximal(adjl, C):
        P = (1 << n) - 1
        for v in C:
            P &= adjl[v]
        C = list(C)
        while P:
            bv = -1
            bd = -1
            Q = P
            while Q:
                bbit = Q & -Q
                v = bbit.bit_length() - 1
                Q &= ~bbit
                d = popcount(adjl[v] & P)
                if d > bd:
                    bd = d
                    bv = v
            C.append(bv)
            P &= adjl[bv]
        return C

    # ---------------- initial clique number ----------------
    order0 = degree_order(adj)
    starts0 = order0[:min(n, 200)]
    hbest = []
    hcliques = []
    for v0 in starts0:
        C = greedy_from(adj, order0, v0)
        hcliques.append(C)
        if len(C) > len(hbest):
            hbest = C

    init_dl = min(deadline, time.time() + max(5.0, 0.35 * (deadline - time.time())))
    wclique, tmo = max_clique(adj, 0, None, init_dl, hbest)
    omega = max(len(wclique), 1)
    record(omega, [])  # baseline: remove nothing
    if tmo or omega <= 1:
        return

    # ---------------- MIP with lazy clique (Turan) covering cuts ----------------
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    model = gp.Model("clique_interdiction")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    yv = model.addVars(m, vtype=GRB.BINARY, name="y")
    model.addConstr(gp.quicksum(yv[e] for e in range(m)) <= k)
    model.setObjective(gp.quicksum(yv[e] for e in range(m)), GRB.MINIMIZE)

    stored = {}

    def register(C):
        fs = frozenset(C)
        if fs in stored:
            return None
        cl = sorted(fs)
        stored[fs] = cl
        return cl

    register(wclique)
    for C in hcliques:
        if len(C) >= 3:
            register(C)

    def add_cut(C, t):
        # Turan-strengthened cut: clique of size s needs >= s C2 - ex(s, K_{t+1}) removals
        s = len(C)
        if s <= t:
            return
        q, r = divmod(s, t)
        rhs = r * q * (q + 1) // 2 + (t - r) * q * (q - 1) // 2
        expr = gp.quicksum(yv[pair2eid[(a, b)]]
                           for a, b in combinations(sorted(C), 2))
        model.addConstr(expr >= rhs)

    t = omega - 1
    while t >= 2:
        if time.time() > deadline - 1:
            break
        # (re-)add cuts for all known cliques at current target level
        for C in list(stored.values()):
            add_cut(C, t)
        success = False
        while True:
            rem = deadline - time.time()
            if rem < 1:
                write_solution()
                return
            model.Params.TimeLimit = float(rem)
            model.optimize()
            if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                # target t provably unreachable => done
                write_solution()
                return
            if model.SolCount == 0:
                write_solution()
                return
            F = [e for e in range(m) if yv[e].X > 0.5]
            adj2 = adj[:]
            for e in F:
                a, b = edges[e]
                adj2[a] &= ~(1 << b)
                adj2[b] &= ~(1 << a)
            order2 = degree_order(adj2)
            # heuristic separation: greedy maximal cliques
            violated = []
            for v0 in order2[:min(n, 150)]:
                C = greedy_from(adj2, order2, v0)
                if len(C) >= t + 1:
                    cl = register(C)
                    if cl is not None:
                        violated.append(cl)
                        if len(violated) >= 15:
                            break
            if not violated:
                # exact separation: does a (t+1)-clique exist in G - F ?
                bc, tmo2 = max_clique(adj2, t, t + 1, deadline)
                if len(bc) >= t + 1:
                    Cm = extend_maximal(adj2, bc)
                    register(Cm)
                    violated.append(sorted(set(Cm)))
                elif tmo2:
                    write_solution()
                    return
                else:
                    # proven: clique number of remaining graph <= t
                    hb2 = []
                    for v0 in order2[:min(n, 60)]:
                        C = greedy_from(adj2, order2, v0)
                        if len(C) > len(hb2):
                            hb2 = C
                    real, tmo3 = max_clique(adj2, 0, None, deadline, hb2)
                    if tmo3:
                        obj = t
                    else:
                        obj = min(t, max(len(real), 1))
                    record(obj, F)
                    t = obj - 1
                    success = True
                    break
            for C in violated:
                add_cut(C, t)
        if not success:
            break

    write_solution()


if __name__ == "__main__":
    main()