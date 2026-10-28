import argparse
import json
import time

import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def pair_index(ii, jj, n):
    """Index of unordered pair (i,j), i<=j, in the enumeration
    (0,0),(0,1),...,(0,n-1),(1,1),...,(n-1,n-1)."""
    a = np.minimum(ii, jj)
    b = np.maximum(ii, jj)
    return a * n - (a * (a - 1)) // 2 + (b - a)


def build_solution(open_mask, best, bi, bj, fixed, origin, dest):
    S = np.flatnonzero(open_mask)
    obj = float(fixed[S].sum() + best.sum())
    assignments = []
    for c in range(best.size):
        i = int(bi[c])
        j = int(bj[c])
        assignments.append({
            "commodity": int(c),
            "origin": int(origin[c]),
            "destination": int(dest[c]),
            "hub_edge": [i] if i == j else [i, j],
            "cost": float(best[c]),
        })
    return {"objective_value": obj,
            "open_hubs": [int(s) for s in S],
            "assignments": assignments}, obj


def note_obj(state, logger, obj):
    if obj < state["logged"] - 1e-9 * max(1.0, abs(state["logged"])):
        state["logged"] = float(obj)
        if logger is not None:
            logger.log(float(obj))


def note_sol(state, logger, obj, sol, min_interval=0.0):
    if obj < state["logged"] - 1e-9 * max(1.0, abs(state["logged"])):
        state["logged"] = float(obj)
    if logger is None:
        return
    if obj < state["snapped"] - 1e-9 * max(1.0, abs(state["snapped"])):
        now = time.time()
        if now - state["last_snap"] >= min_interval:
            logger.log_solution(float(obj), sol)
            state["snapped"] = float(obj)
            state["last_snap"] = now
        else:
            logger.log(float(obj))


# --------------------------------------------------------------------------
# Vectorized local search over hub subsets
# --------------------------------------------------------------------------

class HubHeuristic:
    def __init__(self, D, fixed, coll, distb, tflow, same):
        self.D = D
        self.fixed = fixed
        self.coll = coll      # C x n : flow * chi * d(origin, hub)
        self.distb = distb    # C x n : flow * delta * d(hub, dest)
        self.tflow = tflow    # C     : flow * tau
        self.same = same      # C     : origin == destination
        self.C, self.n = coll.shape

    def _core(self, coll_S, distb_S, tf, D_SS, S, same_mask):
        R = coll_S.shape[0]
        m = S.shape[0]
        best = np.full(R, np.inf)
        bi = np.full(R, -1, dtype=np.int64)
        bj = np.full(R, -1, dtype=np.int64)
        ar = np.arange(R)
        for a in range(m):
            M = tf[:, None] * D_SS[a][None, :] + distb_S
            jm = M.argmin(axis=1)
            tot = coll_S[:, a] + M[ar, jm]
            upd = tot < best
            best[upd] = tot[upd]
            bi[upd] = S[a]
            bj[upd] = S[jm[upd]]
        if same_mask.any():
            rows = np.flatnonzero(same_mask)
            sd = coll_S[rows] + distb_S[rows]
            idx = sd.argmin(axis=1)
            best[rows] = sd[np.arange(rows.size), idx]
            bi[rows] = S[idx]
            bj[rows] = S[idx]
        return best, bi, bj

    def evaluate(self, S):
        S = np.asarray(S, dtype=np.int64)
        D_SS = self.D[np.ix_(S, S)]
        return self._core(self.coll[:, S], self.distb[:, S], self.tflow,
                          D_SS, S, self.same)

    def eval_rows(self, rows, S):
        D_SS = self.D[np.ix_(S, S)]
        return self._core(self.coll[rows][:, S], self.distb[rows][:, S],
                          self.tflow[rows], D_SS, S, self.same[rows])

    def add_scan(self, open_mask, best, deadline, subset=None):
        S = np.flatnonzero(open_mask)
        m = S.size
        closed = np.flatnonzero(~open_mask) if subset is None else subset
        if closed.size == 0:
            return closed, np.empty(0)
        C = self.C
        base = best.sum()
        if m:
            distb_S = self.distb[:, S]
            coll_S = self.coll[:, S]
        else:
            distb_S = coll_S = None
        denom = max(1, C * max(m, 1))
        block = int(max(1, min(32, 3000000 // denom)))
        deltas = np.full(closed.size, np.inf)
        for s0 in range(0, closed.size, block):
            H = closed[s0:s0 + block]
            a3 = self.coll[:, H] + self.distb[:, H]
            if m:
                D_HS = self.D[np.ix_(H, S)]
                TT = self.tflow[:, None, None] * D_HS[None, :, :]
                a1 = self.coll[:, H] + (TT + distb_S[:, None, :]).min(axis=2)
                a2 = self.distb[:, H] + (TT + coll_S[:, None, :]).min(axis=2)
                newc = np.minimum(best[:, None], np.minimum(a1, np.minimum(a2, a3)))
            else:
                newc = a3
            if self.same.any():
                newc[self.same] = np.minimum(best[self.same][:, None], a3[self.same])
            deltas[s0:s0 + H.size] = self.fixed[H] + newc.sum(axis=0) - base
            if time.time() > deadline:
                break
        return closed, deltas

    def drop_scan(self, open_mask, best, bi, bj, deadline):
        S = np.flatnonzero(open_mask)
        m = S.size
        deltas = np.full(m, np.inf)
        if m <= 1:
            return S, deltas
        for a in range(m):
            h = S[a]
            aff = (bi == h) | (bj == h)
            if not aff.any():
                deltas[a] = -self.fixed[h]
            else:
                rows = np.flatnonzero(aff)
                S2 = S[S != h]
                sb, _, _ = self.eval_rows(rows, S2)
                deltas[a] = -self.fixed[h] + sb.sum() - best[aff].sum()
            if time.time() > deadline:
                break
        return S, deltas

    def swap_pass(self, open_mask, best, bi, bj, closed, add_d, deadline, tol):
        S_open = np.flatnonzero(open_mask)
        if S_open.size == 0 or closed.size == 0:
            return False
        _, drop_d = self.drop_scan(open_mask, best, bi, bj, deadline)
        base = best.sum()
        nA = min(5, closed.size)
        nB = min(5, S_open.size)
        cand_add = closed[np.argsort(add_d)[:nA]]
        cand_drop = S_open[np.argsort(-drop_d)[:nB]]
        best_delta = -tol
        move = None
        for h in cand_drop:
            for h2 in cand_add:
                if time.time() > deadline:
                    return False
                nm = open_mask.copy()
                nm[h] = False
                nm[h2] = True
                nb, _, _ = self.evaluate(np.flatnonzero(nm))
                delta = nb.sum() - base + self.fixed[h2] - self.fixed[h]
                if delta < best_delta:
                    best_delta = delta
                    move = (int(h), int(h2))
        if move is not None:
            open_mask[move[0]] = False
            open_mask[move[1]] = True
            return True
        return False

    def descent(self, open_mask, deadline, move_cb=None):
        open_mask = open_mask.copy()
        cand = None
        it = 0
        while time.time() <= deadline:
            S = np.flatnonzero(open_mask)
            best, bi, bj = self.evaluate(S)
            total = best.sum() + self.fixed[S].sum()
            if move_cb is not None:
                move_cb(total)
            tol = 1e-7 * max(1.0, abs(total))
            if cand is None or it % 6 == 0:
                closed, add_d = self.add_scan(open_mask, best, deadline)
                if closed.size:
                    cand = closed[np.argsort(add_d)[:min(48, closed.size)]]
            else:
                cand = cand[~open_mask[cand]]
                if cand.size == 0:
                    cand = None
                    it += 1
                    continue
                closed, add_d = self.add_scan(open_mask, best, deadline, subset=cand)
            moved = False
            if closed.size:
                ai = int(np.argmin(add_d))
                if add_d[ai] < -tol:
                    open_mask[closed[ai]] = True
                    moved = True
            if not moved:
                So, drop_d = self.drop_scan(open_mask, best, bi, bj, deadline)
                if So.size:
                    di = int(np.argmin(drop_d))
                    if drop_d[di] < -tol:
                        open_mask[So[di]] = False
                        moved = True
                        cand = None
            if not moved:
                moved = self.swap_pass(open_mask, best, bi, bj, closed, add_d,
                                       deadline, tol)
                cand = None
            if not moved:
                break
            it += 1
        S = np.flatnonzero(open_mask)
        best, bi, bj = self.evaluate(S)
        total = best.sum() + self.fixed[S].sum()
        return open_mask, total, best, bi, bj


def ils_loop(heur, mask, total, deadline, logger, state, build_fn, move_cb):
    rng = np.random.default_rng(0)
    n = heur.n
    best_mask = mask.copy()
    best_total = total
    stall = 0
    while time.time() < deadline - 0.3:
        pm = best_mask.copy()
        S = np.flatnonzero(pm)
        Cl = np.flatnonzero(~pm)
        if stall >= 10:
            p = min(0.5, max(2.0 / n, S.size / max(1, n)))
            pm = rng.random(n) < p
            if not pm.any():
                pm[int(rng.integers(n))] = True
            stall = 0
        else:
            if S.size > 1:
                k = int(rng.integers(1, min(3, S.size)))
                pm[rng.choice(S, size=k, replace=False)] = False
            if Cl.size > 0:
                k = int(rng.integers(1, min(4, Cl.size + 1)))
                pm[rng.choice(Cl, size=k, replace=False)] = True
            if not pm.any():
                pm[int(rng.integers(n))] = True
        m2, t2, bb, bii, bjj = heur.descent(pm, deadline, move_cb)
        if t2 < best_total - 1e-7 * max(1.0, abs(best_total)):
            best_mask = m2
            best_total = t2
            stall = 0
            sol, obj = build_fn(m2, bb, bii, bjj)
            note_sol(state, logger, obj, sol)
        else:
            stall += 1
    return best_mask, best_total


# --------------------------------------------------------------------------
# MIP data builders (path formulation over unordered hub pairs)
# --------------------------------------------------------------------------

def make_exact_data(heur, warm_mask, coll, distb, tflow, same, n, C):
    P = n * (n + 1) // 2
    counts = n - np.arange(n)
    i_arr = np.repeat(np.arange(n), counts)
    j_arr = np.concatenate([np.arange(i, n) for i in range(n)])
    pair_D = heur.D[i_arr, j_arr]
    V = C * P
    cost = np.empty(V)
    chunk = max(1, int(4e6 // P))
    for c0 in range(0, C, chunk):
        c1 = min(C, c0 + chunk)
        costs = (coll[c0:c1][:, i_arr]
                 + tflow[c0:c1][:, None] * pair_D[None, :]
                 + distb[c0:c1][:, j_arr])
        cost[c0 * P:c1 * P] = costs.ravel()
    owner = np.repeat(np.arange(C), P)
    i_f = np.tile(i_arr, C)
    j_f = np.tile(j_arr, C)
    ub = np.ones(V)
    if same.any():
        diagf = (i_arr == j_arr).astype(float)
        for c in np.flatnonzero(same):
            ub[c * P:(c + 1) * P] = diagf
    S_w = np.flatnonzero(warm_mask)
    _, biw, bjw = heur.evaluate(S_w)
    hp = pair_index(biw, bjw, n)
    start_var = np.arange(C) * P + hp
    disaggregated = (V <= 300000)
    return dict(owner=owner, i_f=i_f, j_f=j_f, cost=cost, ub=ub,
                start_var=start_var, disaggregated=disaggregated)


def make_restricted_data(heur, warm_mask, coll, distb, tflow, same, n, C, K,
                         deadline):
    D = heur.D
    H = max(12, int(np.ceil((np.sqrt(1.0 + 8.0 * K) - 1.0) / 4.0)) + 2)
    use_all = (2 * H >= n)
    if use_all:
        M = n
        topc = topd = None
    else:
        H = min(H, n)
        M = 2 * H
        topc = np.argpartition(coll, H - 1, axis=1)[:, :H]
        topd = np.argpartition(distb, H - 1, axis=1)[:, :H]
    ii, jj = np.triu_indices(M)
    Q = ii.size
    Ksel = int(min(K, Q))
    owners, i_list, j_list = [], [], []
    chunk = max(1, int(2e6 // Q))
    all_idx = np.arange(n)
    for c0 in range(0, C, chunk):
        c1 = min(C, c0 + chunk)
        L = c1 - c0
        if use_all:
            hubs = np.tile(all_idx, (L, 1))
        else:
            hubs = np.concatenate([topc[c0:c1], topd[c0:c1]], axis=1)
        hi = hubs[:, ii]
        hj = hubs[:, jj]
        costs = (np.take_along_axis(coll[c0:c1], hi, axis=1)
                 + tflow[c0:c1][:, None] * D[hi, hj]
                 + np.take_along_axis(distb[c0:c1], hj, axis=1))
        sm = same[c0:c1]
        if sm.any():
            costs[sm[:, None] & (hi != hj)] = np.inf
        if Ksel < Q:
            part = np.argpartition(costs, Ksel - 1, axis=1)[:, :Ksel]
            hi_s = np.take_along_axis(hi, part, axis=1)
            hj_s = np.take_along_axis(hj, part, axis=1)
        else:
            hi_s, hj_s = hi, hj
        owners.append(np.repeat(np.arange(c0, c1), hi_s.shape[1]))
        i_list.append(hi_s.ravel())
        j_list.append(hj_s.ravel())
        if time.time() > deadline:
            raise RuntimeError("candidate generation timed out")
    owner = np.concatenate(owners)
    i_f = np.concatenate(i_list)
    j_f = np.concatenate(j_list)
    # append the pairs used by the warm-start solution
    S_w = np.flatnonzero(warm_mask)
    _, biw, bjw = heur.evaluate(S_w)
    owner = np.concatenate([owner, np.arange(C)])
    i_f = np.concatenate([i_f, np.minimum(biw, bjw)])
    j_f = np.concatenate([j_f, np.maximum(biw, bjw)])
    # sort by owner so variables are grouped per commodity
    order = np.argsort(owner, kind="stable")
    owner = owner[order]
    i_f = i_f[order]
    j_f = j_f[order]
    V = owner.size
    cost = (coll[owner, i_f] + tflow[owner] * D[i_f, j_f] + distb[owner, j_f])
    ub = np.ones(V)
    ub[same[owner] & (i_f != j_f)] = 0.0
    cnt = np.bincount(owner, minlength=C)
    offs = np.cumsum(cnt) - cnt
    start_var = offs + cnt - 1  # appended warm-start pair is last per commodity
    return dict(owner=owner, i_f=i_f, j_f=j_f, cost=cost, ub=ub,
                start_var=start_var, disaggregated=False)


def build_and_solve(heur, warm_mask, dat, fixed, origin, dest, n, C,
                    solve_deadline, logger, state, build_fn):
    import gurobipy as gp
    from gurobipy import GRB
    from scipy import sparse

    if solve_deadline - time.time() < 3.0:
        return None, None, False

    owner = np.asarray(dat["owner"], dtype=np.int64)
    i_f = np.asarray(dat["i_f"], dtype=np.int64)
    j_f = np.asarray(dat["j_f"], dtype=np.int64)
    cost = dat["cost"]
    ub = dat["ub"]
    V = owner.size

    model = gp.Model("uhl")
    prm = model.Params
    prm.Seed = 0
    prm.MIPGap = 1e-4
    prm.NumericFocus = 0
    prm.Threads = 1
    prm.LogToConsole = 0
    prm.Method = 1

    z = model.addMVar(V + n,
                      lb=np.zeros(V + n),
                      ub=np.concatenate([ub, np.ones(n)]),
                      obj=np.concatenate([cost, fixed]),
                      vtype=["C"] * V + ["B"] * n)

    cnt = np.bincount(owner, minlength=C)
    rows = np.repeat(np.arange(C, dtype=np.int64), cnt)
    A1 = sparse.csr_matrix((np.ones(V), (rows, np.arange(V, dtype=np.int64))),
                           shape=(C, V + n))
    model.addMConstr(A1, z, GRB.EQUAL, np.ones(C))

    if dat["disaggregated"]:
        r = np.arange(V, dtype=np.int64)
        off = i_f != j_f
        roff = r[off]
        rows_a = np.concatenate([r, V + roff])
        cols_a = np.concatenate([r, roff])
        cols_b = np.concatenate([V + i_f, V + j_f[off]])
        data = np.concatenate([np.ones(rows_a.size), -np.ones(rows_a.size)])
        A2 = sparse.csr_matrix((data,
                                (np.concatenate([rows_a, rows_a]),
                                 np.concatenate([cols_a, cols_b]))),
                               shape=(V + roff.size, V + n))
        model.addMConstr(A2, z, GRB.LESS_EQUAL, np.zeros(V + roff.size))
    else:
        r1 = owner * n + i_f
        off = i_f != j_f
        r2 = owner[off] * n + j_f[off]
        rx = np.concatenate([r1, r2])
        cx = np.concatenate([np.arange(V, dtype=np.int64),
                             np.arange(V, dtype=np.int64)[off]])
        urows, inv = np.unique(rx, return_inverse=True)
        rows_y = np.arange(urows.size, dtype=np.int64)
        cols_y = V + (urows % n)
        data = np.concatenate([np.ones(rx.size), -np.ones(urows.size)])
        A2 = sparse.csr_matrix((data,
                                (np.concatenate([inv, rows_y]),
                                 np.concatenate([cx, cols_y]))),
                               shape=(urows.size, V + n))
        model.addMConstr(A2, z, GRB.LESS_EQUAL, np.zeros(urows.size))

    try:
        start = np.zeros(V + n)
        start[V:] = warm_mask.astype(float)
        start[np.asarray(dat["start_var"], dtype=np.int64)] = 1.0
        z.Start = start
    except Exception:
        pass

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            except Exception:
                return
            if obj >= state["logged"] - 1e-7 * max(1.0, abs(state["logged"])):
                return
            now = time.time()
            if now - state.get("last_cb", 0.0) < 0.5:
                state["logged"] = float(obj)
                if logger is not None:
                    logger.log(float(obj))
                return
            state["last_cb"] = now
            try:
                zv = m.cbGetSolution(z)
            except Exception:
                state["logged"] = float(obj)
                if logger is not None:
                    logger.log(float(obj))
                return
            om = zv[V:] > 0.5
            if not om.any():
                return
            S = np.flatnonzero(om)
            bb, bii, bjj = heur.evaluate(S)
            sol, real_obj = build_fn(om, bb, bii, bjj)
            note_sol(state, logger, real_obj, sol)

    prm.TimeLimit = max(1.0, solve_deadline - time.time())
    model.optimize(cb)

    optimal = (model.Status == GRB.OPTIMAL)
    sol = None
    obj = None
    if model.SolCount > 0:
        try:
            zv = z.X
        except Exception:
            zv = None
        if zv is not None:
            om = zv[V:] > 0.5
            if om.any():
                S = np.flatnonzero(om)
                bb, bii, bjj = heur.evaluate(S)
                sol, obj = build_fn(om, bb, bii, bjj)
    return sol, obj, optimal


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["num_nodes"])
    fixed = np.zeros(n)
    for nd in data.get("nodes", []):
        fixed[int(nd["id"])] = float(nd["fixed_cost"])
    D = np.array(data["distances"], dtype=float)
    params = data.get("parameters", {}) or {}
    tau = float(params.get("tau", 1.0))
    chi = float(params.get("chi", 1.0))
    delta = float(params.get("delta", 1.0))
    comms = data.get("commodities", []) or []
    C = len(comms)
    origin = np.array([int(c["origin"]) for c in comms], dtype=np.int64)
    dest = np.array([int(c["destination"]) for c in comms], dtype=np.int64)
    flow = np.array([float(c["flow"]) for c in comms], dtype=float)
    return n, C, fixed, D, tau, chi, delta, origin, dest, flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    TL = max(1, int(args.time_limit))
    global_deadline = t0 + TL

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    n, C, fixed, D, tau, chi, delta, origin, dest, flow = load_instance(
        args.instance_path)

    if C == 0 or n == 0:
        sol = {"objective_value": 0.0, "open_hubs": [], "assignments": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        if logger is not None:
            logger.log_solution(0.0, sol)
        return

    # Cost arrays. Distances are already scaled by TC.
    coll = flow[:, None] * chi * D[origin, :]
    distb = flow[:, None] * delta * D[dest, :]
    tflow = flow * tau
    same = origin == dest

    heur = HubHeuristic(D, fixed, coll, distb, tflow, same)
    state = {"logged": np.inf, "snapped": np.inf, "last_snap": 0.0,
             "last_cb": 0.0}

    def build_fn(mask, bb, bii, bjj):
        return build_solution(mask, bb, bii, bjj, fixed, origin, dest)

    def move_cb(obj):
        note_obj(state, logger, obj)

    P = n * (n + 1) // 2
    exact_ok = (n >= 2) and (C * P <= 1500000)
    restr_ok = (not exact_ok) and (n >= 2) and (C <= 60000) and (n <= 2000)

    if exact_ok:
        hb = min(max(0.15 * TL, 2.0), 30.0)
        hb = min(hb, max(TL - 5.0, 0.5))
    elif restr_ok:
        hb = 0.5 * TL
    else:
        hb = max(TL - 1.5, 0.3)
    heur_deadline = t0 + hb

    # Trivial fallback: best single hub.
    single = fixed + (coll + distb).sum(axis=0)
    h0 = int(np.argmin(single))
    mask0 = np.zeros(n, dtype=bool)
    mask0[h0] = True
    b0, bi0, bj0 = heur.evaluate(np.array([h0]))
    best_mask = mask0
    best_total = float(fixed[h0] + b0.sum())
    sol0, obj0 = build_fn(best_mask, b0, bi0, bj0)
    note_sol(state, logger, obj0, sol0)

    try:
        best_mask, best_total, bb, bii, bjj = heur.descent(mask0, heur_deadline,
                                                           move_cb)
        sol, obj = build_fn(best_mask, bb, bii, bjj)
        note_sol(state, logger, obj, sol)
        best_mask, best_total = ils_loop(heur, best_mask, best_total,
                                         heur_deadline, logger, state,
                                         build_fn, move_cb)
    except Exception:
        pass

    mip_optimal = False
    remaining = global_deadline - time.time()
    if exact_ok:
        V = C * P
        min_rem = 4.0 if V <= 300000 else (12.0 if V <= 700000 else 20.0)
        if remaining > min_rem:
            try:
                dat = make_exact_data(heur, best_mask, coll, distb, tflow,
                                      same, n, C)
                sol2, obj2, opt = build_and_solve(
                    heur, best_mask, dat, fixed, origin, dest, n, C,
                    global_deadline - 1.0, logger, state, build_fn)
                if obj2 is not None and obj2 < best_total - 1e-7 * max(
                        1.0, abs(best_total)):
                    best_total = obj2
                    best_mask = np.zeros(n, dtype=bool)
                    best_mask[np.array(sol2["open_hubs"], dtype=int)] = True
                mip_optimal = opt
            except Exception:
                pass
    elif restr_ok and remaining > 25.0:
        try:
            K = int(max(15, min(2000, 800000 // C - 1)))
            dat = make_restricted_data(heur, best_mask, coll, distb, tflow,
                                       same, n, C, K, global_deadline - 3.0)
            sol2, obj2, opt = build_and_solve(
                heur, best_mask, dat, fixed, origin, dest, n, C,
                global_deadline - 1.0, logger, state, build_fn)
            if obj2 is not None and obj2 < best_total - 1e-7 * max(
                    1.0, abs(best_total)):
                best_total = obj2
                best_mask = np.zeros(n, dtype=bool)
                best_mask[np.array(sol2["open_hubs"], dtype=int)] = True
            # restricted model optimum is not a true optimum
        except Exception:
            pass

    remaining = global_deadline - time.time()
    if remaining > 2.0 and not mip_optimal:
        try:
            best_mask, best_total = ils_loop(heur, best_mask, best_total,
                                             global_deadline - 1.0, logger,
                                             state, build_fn, move_cb)
        except Exception:
            pass

    S = np.flatnonzero(best_mask)
    bb, bii, bjj = heur.evaluate(S)
    best_sol, best_obj = build_fn(best_mask, bb, bii, bjj)
    if logger is not None and best_obj < state["snapped"] - 1e-9 * max(
            1.0, abs(state["snapped"])):
        logger.log_solution(float(best_obj), best_sol)
        state["snapped"] = float(best_obj)

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()