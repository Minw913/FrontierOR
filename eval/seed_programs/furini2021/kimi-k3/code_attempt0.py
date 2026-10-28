import argparse
import json
import math
import random
import sys
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:
    gp = None
    GRB = None
    HAVE_GUROBI = False

from solution_logger import SolutionLogger

sys.setrecursionlimit(1000000)

if hasattr(int, "bit_count"):
    _popcount = int.bit_count
else:
    def _popcount(x):
        return bin(x).count("1")


class _Timeout(Exception):
    pass


class MaxCliqueSolver:
    """Maximum clique via bitset branch-and-bound with greedy-coloring bounds (Tomita MCQ).

    Time-boxed; always keeps the best clique found so far and reports whether the
    search completed (i.e. the returned size is proven optimal).
    """

    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.best = 0
        self.best_clique = []
        self.deadline = float("inf")
        self.calls = 0
        self.completed = False

    def solve(self, deadline, init_best=0, init_clique=None):
        self.best = init_best
        self.best_clique = list(init_clique) if init_clique else []
        self.deadline = deadline
        self.calls = 0
        self.completed = False
        if self.n == 0:
            self.completed = True
            return self.best, self.best_clique, self.completed
        if time.time() > deadline:
            return self.best, self.best_clique, False
        try:
            self._expand([], (1 << self.n) - 1)
            self.completed = True
        except (_Timeout, RecursionError):
            pass
        return self.best, self.best_clique, self.completed

    def _expand(self, R, P):
        self.calls += 1
        if (self.calls & 1023) == 0 and time.time() > self.deadline:
            raise _Timeout()
        if P == 0:
            if len(R) > self.best:
                self.best = len(R)
                self.best_clique = R[:]
            return
        order, bounds = self._color_sort(P)
        adj = self.adj
        P2 = P
        Rlen = len(R)
        for i in range(len(order) - 1, -1, -1):
            v = order[i]
            if Rlen + bounds[i] <= self.best:
                return
            bit = 1 << v
            if not (P2 & bit):
                continue
            R.append(v)
            self._expand(R, P2 & adj[v])
            R.pop()
            P2 &= ~bit

    def _color_sort(self, P):
        adj = self.adj
        order = []
        bounds = []
        uncolored = P
        color = 0
        while uncolored:
            color += 1
            avail = uncolored
            while avail:
                lsb = avail & (-avail)
                v = lsb.bit_length() - 1
                order.append(v)
                bounds.append(color)
                uncolored &= ~lsb
                avail &= ~lsb
                avail &= ~adj[v]
        return order, bounds


def dynamic_greedy_clique(n, adj):
    C = []
    cand = (1 << n) - 1
    while cand:
        tmp = cand
        bv = -1
        bd = -1
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            d = _popcount(adj[v] & cand)
            if d > bd:
                bd = d
                bv = v
        C.append(bv)
        cand &= adj[bv]
    return C


def heuristic_max_clique(n, adj, deadline, rng):
    """Multi-start greedy clique heuristic; returns (size, clique list)."""
    best = 0
    bestC = []
    if n <= 2500:
        bestC = dynamic_greedy_clique(n, adj)
        best = len(bestC)
    full = (1 << n) - 1
    deg = [_popcount(a) for a in adj]
    base = sorted(range(n), key=lambda v: -deg[v])
    while time.time() < deadline:
        if rng.random() < 0.5:
            order = base[:]
            for _ in range(max(1, n // 10)):
                i = rng.randrange(n)
                j = rng.randrange(n)
                order[i], order[j] = order[j], order[i]
        else:
            order = list(range(n))
            rng.shuffle(order)
        C = []
        cand = full
        for v in order:
            if (cand >> v) & 1:
                C.append(v)
                cand &= adj[v]
                if len(C) + _popcount(cand) <= best:
                    break
        if len(C) > best:
            best = len(C)
            bestC = C
    return best, bestC


def greedy_interdiction(n, adj, edge_list, edge_index, k, deadline, rng, init_removed=None):
    """Repeatedly delete an edge (max common-neighbor score) from a largest found clique."""
    a = adj[:]
    removed = list(init_removed) if init_removed else []
    remset = set(removed)
    for e in removed:
        u, v = edge_list[e]
        a[u] &= ~(1 << v)
        a[v] &= ~(1 << u)
    cliques = []
    seen = set()
    while len(removed) < k:
        now = time.time()
        if now > deadline:
            break
        s, C = heuristic_max_clique(n, a, min(deadline, now + 0.15), rng)
        if s <= 1:
            break
        fs = frozenset(C)
        if fs not in seen:
            seen.add(fs)
            cliques.append(C)
        bsc = -1
        bu, bv = C[0], C[1]
        for i in range(s):
            u = C[i]
            au = a[u]
            for j in range(i + 1, s):
                v = C[j]
                sc = _popcount(au & a[v])
                if sc > bsc:
                    bsc = sc
                    bu, bv = u, v
        u, v = (bu, bv) if bu < bv else (bv, bu)
        e = edge_index.get((u, v))
        if e is None or e in remset:
            break
        remset.add(e)
        removed.append(e)
        a[u] &= ~(1 << v)
        a[v] &= ~(1 << u)
    return removed, cliques


def partition_heuristic(n, adj_list, edge_list, t, deadline, rng):
    """Partition vertices into t parts; remove all intra-part edges (remaining graph is
    t-partite, hence clique number <= t). Local search minimizes intra-part edges."""
    if t < 1 or n == 0:
        return None
    if n * t > 4_000_000:
        return None
    order = list(range(n))
    rng.shuffle(order)
    part = [0] * n
    for i, v in enumerate(order):
        part[v] = i % t
    cnt = [[0] * t for _ in range(n)]
    for (u, v) in edge_list:
        cnt[u][part[v]] += 1
        cnt[v][part[u]] += 1
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(order)
        for v in order:
            pv = part[v]
            cv = cnt[v]
            cur = cv[pv]
            bq = -1
            bv = cur
            for q in range(t):
                cq = cv[q]
                if cq < bv:
                    bv = cq
                    bq = q
            if bq >= 0 and bq != pv:
                for u in adj_list[v]:
                    cnt[u][pv] -= 1
                    cnt[u][bq] += 1
                part[v] = bq
                improved = True
    removed = [i for i, (u, v) in enumerate(edge_list) if part[u] == part[v]]
    return removed


def turan_min_removals(s, t):
    """Min edges to delete from K_s so its clique number drops to <= t
    (edges inside a balanced t-partition)."""
    if t <= 0 or t >= s:
        return 0
    q, r = divmod(s, t)
    return r * (q + 1) * q // 2 + (t - r) * q * (q - 1) // 2


def solve(args, logger):
    t_start = time.time()
    total = float(max(1, args.time_limit))
    deadline = t_start + total

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    verts = list(data["vertices"])
    n = len(verts)
    k = int(data["interdiction_budget_k"])
    vidx = {v: i for i, v in enumerate(verts)}
    adj = [0] * n
    adj_list = [[] for _ in range(n)]
    edge_list = []
    edge_index = {}
    for e in data["edges"]:
        u = vidx[e[0]]
        v = vidx[e[1]]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in edge_index:
            continue
        edge_index[(u, v)] = len(edge_list)
        edge_list.append((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    m = len(edge_list)

    def make_sol_dict(obj, removed_idx):
        return {
            "objective_value": int(obj),
            "interdicted_edges": [[verts[edge_list[i][0]], verts[edge_list[i][1]]]
                                  for i in removed_idx],
        }

    def write_solution(obj, removed_idx):
        with open(args.solution_path, "w") as f:
            json.dump(make_sol_dict(obj, removed_idx), f)

    last_logged = {"obj": None}

    def maybe_log(obj, removed_idx):
        if logger is None:
            return
        if last_logged["obj"] is not None and obj >= last_logged["obj"]:
            return
        last_logged["obj"] = int(obj)
        try:
            logger.log_solution(int(obj), make_sol_dict(obj, removed_idx))
        except Exception:
            pass

    def surviving_adj(removed_idx):
        a = adj[:]
        for e in removed_idx:
            u, v = edge_list[e]
            a[u] &= ~(1 << v)
            a[v] &= ~(1 << u)
        return a

    # ---------------- trivial cases ----------------
    if n == 0:
        write_solution(0, [])
        return
    if m == 0 or n == 1:
        maybe_log(1, [])
        write_solution(1, [])
        return
    if k >= m:
        all_idx = list(range(m))
        maybe_log(1, all_idx)
        write_solution(1, all_idx)
        return

    rng = random.Random(20240501)

    if k <= 0:
        s0, C0 = heuristic_max_clique(n, adj, min(deadline - 0.2, time.time() + max(0.2, 0.05 * total)), rng)
        mcq = MaxCliqueSolver(n, adj)
        b, Cb, comp = mcq.solve(deadline - 0.15, init_best=s0, init_clique=C0)
        maybe_log(b, [])
        write_solution(b, [])
        return

    # ---------------- quick large clique on the original graph ----------------
    h_end = min(deadline - 0.3, time.time() + max(0.2, min(5.0, 0.05 * total)))
    s0, C0 = heuristic_max_clique(n, adj, h_end, rng)

    seen_cliques = set()
    cliques_for_cuts = []

    def add_clique(C):
        if len(C) >= 2 and len(cliques_for_cuts) < 500:
            fs = frozenset(C)
            if fs not in seen_cliques:
                seen_cliques.add(fs)
                cliques_for_cuts.append(list(C))

    add_clique(C0)

    # ---------------- candidate solutions via heuristics ----------------
    candidates = [[]]

    g_end = min(deadline - 0.3, time.time() + max(0.3, min(25.0, 0.18 * total)))
    try:
        rem1, cl1 = greedy_interdiction(n, adj, edge_list, edge_index, k, g_end, rng)
        candidates.append(rem1)
        for C in cl1:
            add_clique(C)
    except Exception:
        pass

    # lower bound on the optimum
    lb = 2 if (k < m and n >= 2) else 1
    lb = max(lb, s0 - k)
    t = lb
    while t < s0 and turan_min_removals(s0, t) > k:
        t += 1
    lb = max(lb, t)

    # partition-based heuristic (good when the budget is large)
    try:
        if n * max(2, lb + 2) <= 4_000_000:
            p_end = min(deadline - 0.3, time.time() + max(0.3, min(15.0, 0.10 * total)))
            tt = max(2, lb)
            tries = 0
            while tt <= s0 and tries < 25 and time.time() < p_end:
                r = partition_heuristic(n, adj_list, edge_list, tt,
                                        min(p_end, time.time() + 2.0), rng)
                tries += 1
                if r is not None and len(r) <= k:
                    candidates.append(r)
                    if len(r) < k and time.time() < p_end:
                        rem2, cl2 = greedy_interdiction(
                            n, adj, edge_list, edge_index, k,
                            min(p_end, time.time() + 5.0), rng, init_removed=r)
                        candidates.append(rem2)
                        for C in cl2:
                            add_clique(C)
                    break
                tt += 1
    except Exception:
        pass

    # ---------------- evaluate / verify candidates ----------------
    cand_scored = []
    for rem in candidates:
        if time.time() > deadline - 0.3:
            break
        a = surviving_adj(rem)
        hs, hC = heuristic_max_clique(n, a, min(deadline - 0.2, time.time() + 0.25), rng)
        cand_scored.append((hs, rem, hC))
    cand_scored.sort(key=lambda t: t[0])

    best_obj = None
    best_removed = []
    best_verified = False
    vphase_end = min(deadline - 0.3, time.time() + max(0.3, 0.15 * total))
    for hs, rem, hC in cand_scored[:2]:
        now = time.time()
        if now > vphase_end:
            if best_obj is None:
                best_obj = hs
                best_removed = list(rem)
                best_verified = False
            continue
        vbudget = max(0.2, min(8.0, vphase_end - now))
        mcq = MaxCliqueSolver(n, surviving_adj(rem))
        b, Cb, comp = mcq.solve(now + vbudget, init_best=hs, init_clique=hC)
        if best_obj is None or b < best_obj:
            best_obj = b
            best_removed = list(rem)
            best_verified = comp
    if best_obj is None:
        best_obj = s0
        best_removed = []
        best_verified = False

    maybe_log(best_obj, best_removed)

    # ---------------- MIP with lazy clique cuts ----------------
    if HAVE_GUROBI and best_obj is not None and lb <= best_obj - 1 and time.time() < deadline - 0.8:
        try:
            reserve = 0.5 if best_verified else max(1.0, 0.15 * total)
            tl = deadline - time.time() - reserve
            if tl > 0.2:
                model = gp.Model("clique_interdiction")
                p = model.Params
                p.Seed = 0
                p.MIPGap = 1e-4
                p.NumericFocus = 0
                p.Threads = 1
                p.LazyConstraints = 1
                p.OutputFlag = 0
                p.LogToConsole = 0
                p.TimeLimit = tl

                xvars = model.addVars(m, vtype=GRB.BINARY, name="x")
                xlist = [xvars[i] for i in range(m)]
                zvar = model.addVar(lb=float(lb), ub=float(best_obj - 1),
                                    vtype=GRB.INTEGER, name="z")
                model.addConstr(xvars.sum() <= float(k), name="budget")
                model.setObjective(zvar, GRB.MINIMIZE)

                def clique_edge_ids(C):
                    ids = []
                    lc = len(C)
                    for i in range(lc):
                        u = C[i]
                        for j in range(i + 1, lc):
                            v = C[j]
                            ids.append(edge_index[(u, v) if u < v else (v, u)])
                    return ids

                cliques_for_cuts.sort(key=len, reverse=True)
                for C in cliques_for_cuts:
                    if len(C) <= lb:
                        continue
                    ids = clique_edge_ids(C)
                    model.addConstr(zvar + gp.quicksum(xlist[i] for i in ids) >= len(C))

                state = {
                    "best_obj": int(best_obj),
                    "best_removed": list(best_removed),
                    "cb_count": 0,
                    "last_exact": 0.0,
                }

                def callback(mdl, where):
                    if where != GRB.Callback.MIPSOL:
                        return
                    now = time.time()
                    if now >= deadline - 0.2:
                        mdl.terminate()
                        return
                    state["cb_count"] += 1
                    try:
                        xv = mdl.cbGetSolution(xlist)
                        zv = mdl.cbGetSolution(zvar)
                    except Exception:
                        return
                    zf = int(math.floor(zv + 0.5))
                    removed = [i for i, val in enumerate(xv) if val > 0.5]
                    a = adj[:]
                    for e in removed:
                        u, v = edge_list[e]
                        a[u] &= ~(1 << v)
                        a[v] &= ~(1 << u)
                    sh, Ch = heuristic_max_clique(
                        n, a, min(deadline - 0.2, now + 0.05), rng)
                    if sh > zf:
                        try:
                            ids = clique_edge_ids(Ch)
                            mdl.cbLazy(zvar + gp.quicksum(xlist[i] for i in ids) >= sh)
                        except Exception:
                            pass
                    remaining = deadline - 0.3 - time.time()
                    if remaining < 0.15:
                        return
                    promising = sh < state["best_obj"]
                    need_confirm = sh <= zf
                    if promising:
                        ex_budget = max(0.15, min(2.0, remaining * 0.05))
                        init_best = sh
                        init_clique = Ch
                    elif need_confirm and state["cb_count"] % 3 == 0:
                        ex_budget = max(0.1, min(0.5, remaining * 0.02))
                        init_best = zf
                        init_clique = Ch if sh == zf else None
                    else:
                        return
                    mcq = MaxCliqueSolver(n, a)
                    b, Cb, comp = mcq.solve(time.time() + ex_budget,
                                            init_best=init_best,
                                            init_clique=init_clique)
                    if b > zf and b > sh:
                        try:
                            ids = clique_edge_ids(Cb)
                            mdl.cbLazy(zvar + gp.quicksum(xlist[i] for i in ids) >= b)
                        except Exception:
                            pass
                    if promising and comp and b < state["best_obj"]:
                        state["best_obj"] = int(b)
                        state["best_removed"] = list(removed)
                        maybe_log(int(b), removed)
                        if b <= lb:
                            mdl.terminate()

                model.optimize(callback)

                if state["best_obj"] < best_obj:
                    best_obj = state["best_obj"]
                    best_removed = state["best_removed"]
                    best_verified = True
        except Exception:
            pass

    # ---------------- final verification ----------------
    if not best_verified:
        rem_time = deadline - 0.15 - time.time()
        if rem_time > 0.05:
            mcq = MaxCliqueSolver(n, surviving_adj(best_removed))
            b, Cb, comp = mcq.solve(time.time() + rem_time, init_best=0)
            if comp:
                best_obj = b
                best_verified = True
            else:
                best_obj = max(best_obj, b)
            maybe_log(best_obj, best_removed)

    write_solution(best_obj, best_removed)


def emergency_fallback(args):
    """Last-resort: always produce a schema-valid output file."""
    try:
        with open(args.instance_path, "r") as f:
            data = json.load(f)
        verts = list(data["vertices"])
        n = len(verts)
        k = max(0, int(data["interdiction_budget_k"]))
        vidx = {v: i for i, v in enumerate(verts)}
        adj = [0] * n
        taken = []
        seen = set()
        for e in data["edges"]:
            u = vidx[e[0]]
            v = vidx[e[1]]
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            if len(taken) < k:
                taken.append([e[0], e[1]])
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        if n == 0:
            obj = 0
        elif not any(adj):
            obj = 1
        else:
            rng = random.Random(0)
            obj, _ = heuristic_max_clique(n, adj, time.time() + 1.0, rng)
            obj = max(obj, 1)
        with open(args.solution_path, "w") as f:
            json.dump({"objective_value": int(obj), "interdicted_edges": taken}, f)
    except Exception:
        try:
            with open(args.solution_path, "w") as f:
                json.dump({"objective_value": 1, "interdicted_edges": []}, f)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        solve(args, logger)
    except Exception:
        emergency_fallback(args)


if __name__ == "__main__":
    main()