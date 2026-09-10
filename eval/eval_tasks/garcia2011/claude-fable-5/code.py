import argparse
import json
import time
import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

CHUNK_ELEMS = 4_000_000


class DistProvider:
    """Serves distance columns either from a dense matrix or from coordinates."""

    def __init__(self, D=None, X=None):
        self.D = D
        self.X = X
        self.n = int(D.shape[0]) if D is not None else int(X.shape[0])

    def cols(self, J):
        J = np.asarray(J, dtype=np.int64)
        if self.D is not None:
            return self.D[:, J]
        dx = self.X[:, 0][:, None] - self.X[J, 0][None, :]
        dy = self.X[:, 1][:, None] - self.X[J, 1][None, :]
        return np.rint(np.sqrt(dx * dx + dy * dy)).astype(np.float32)

    def between(self, I, J):
        I = np.asarray(I, dtype=np.int64)
        J = np.asarray(J, dtype=np.int64)
        if self.D is not None:
            return self.D[np.ix_(I, J)]
        dx = self.X[I, 0][:, None] - self.X[J, 0][None, :]
        dy = self.X[I, 1][:, None] - self.X[J, 1][None, :]
        return np.rint(np.sqrt(dx * dx + dy * dy)).astype(np.float32)


def nearest_two(dp, M):
    """Returns (a, d1, d2): a = position (in M) of nearest median, d1/d2 nearest
    and second nearest distances. Memory-safe (chunked over medians)."""
    n = dp.n
    Marr = np.asarray(M, dtype=np.int64)
    p = len(Marr)
    d1 = np.full(n, np.inf, dtype=np.float32)
    d2 = np.full(n, np.inf, dtype=np.float32)
    a = np.zeros(n, dtype=np.int64)
    c = max(1, min(p, CHUNK_ELEMS // max(n, 1)))
    r = np.arange(n)
    for s in range(0, p, c):
        cb = Marr[s:s + c]
        C = dp.cols(cb)
        m = C.shape[1]
        if m == 1:
            e1 = C[:, 0]
            ka = np.zeros(n, dtype=np.int64)
            e2 = np.full(n, np.inf, dtype=np.float32)
        else:
            part = np.argpartition(C, 1, axis=1)[:, :2]
            v0 = C[r, part[:, 0]]
            v1 = C[r, part[:, 1]]
            sw = v0 > v1
            ka = np.where(sw, part[:, 1], part[:, 0]).astype(np.int64)
            e1 = np.where(sw, v1, v0)
            e2 = np.where(sw, v0, v1)
        cond = e1 < d1
        nd2 = np.where(cond, np.minimum(d1, e2), np.minimum(d2, e1))
        a = np.where(cond, s + ka, a)
        d1 = np.where(cond, e1, d1)
        d2 = nd2
    return a, d1, d2


def solution_objective(dp, M):
    _, d1, _ = nearest_two(dp, M)
    return float(d1.sum(dtype=np.float64))


def build_solution(dp, M):
    Ms = sorted(set(int(x) for x in M))
    a, d1, _ = nearest_two(dp, Ms)
    Marr = np.asarray(Ms, dtype=np.int64)
    facilities = Marr[a]
    n = dp.n
    assignments = []
    for i in range(n):
        assignments.append({
            "customer": int(i),
            "facility": int(facilities[i]),
            "cost": int(round(float(d1[i])))
        })
    return {
        "objective_value": int(round(float(d1.sum(dtype=np.float64)))),
        "opened_facilities": [int(x) for x in Ms],
        "assignments": assignments
    }


def greedy_construct(dp, p, deadline, rng, cap):
    n = dp.n
    d1 = np.full(n, np.float32(1e12), dtype=np.float32)
    chosen = np.zeros(n, dtype=bool)
    M = []
    cc = max(1, CHUNK_ELEMS // max(n, 1))
    while len(M) < p:
        if time.time() > deadline:
            break
        cand = np.flatnonzero(~chosen)
        if cand.size == 0:
            break
        if cand.size > cap:
            cand = rng.choice(cand, size=cap, replace=False)
        best_j = -1
        best_sc = np.inf
        for s in range(0, len(cand), cc):
            cb = cand[s:s + cc]
            C = dp.cols(cb)
            sc = np.minimum(C, d1[:, None]).sum(axis=0, dtype=np.float64)
            jl = int(np.argmin(sc))
            if float(sc[jl]) < best_sc:
                best_sc = float(sc[jl])
                best_j = int(cb[jl])
        if best_j < 0:
            break
        M.append(best_j)
        chosen[best_j] = True
        np.minimum(d1, dp.cols([best_j])[:, 0], out=d1)
    # fill remaining randomly if out of time
    if len(M) < p:
        rest = np.flatnonzero(~chosen)
        need = p - len(M)
        if rest.size > 0 and need > 0:
            extra = rng.choice(rest, size=min(need, rest.size), replace=False)
            for j in extra:
                M.append(int(j))
                chosen[int(j)] = True
    return M


def kmeanspp_seed(dp, p, rng):
    n = dp.n
    first = int(rng.integers(n))
    M = [first]
    chosen = np.zeros(n, dtype=bool)
    chosen[first] = True
    d1 = dp.cols([first])[:, 0].copy()
    while len(M) < p:
        w = d1.astype(np.float64)
        w[chosen] = 0.0
        tot = float(w.sum())
        if tot <= 0.0:
            rest = np.flatnonzero(~chosen)
            if rest.size == 0:
                break
            j = int(rng.choice(rest))
        else:
            r = float(rng.random()) * tot
            cs = np.cumsum(w)
            j = int(np.searchsorted(cs, r))
            j = min(j, n - 1)
            if chosen[j]:
                rest = np.flatnonzero(~chosen)
                if rest.size == 0:
                    break
                j = int(rng.choice(rest))
        M.append(j)
        chosen[j] = True
        np.minimum(d1, dp.cols([j])[:, 0], out=d1)
    return M


def alternate_medoid(dp, M, deadline, rng, max_cluster=3000):
    M = np.array(sorted(set(int(x) for x in M)), dtype=np.int64)
    p = len(M)
    for _ in range(60):
        if time.time() > deadline:
            break
        a, d1, _ = nearest_two(dp, M)
        changed = False
        for k in range(p):
            if time.time() > deadline:
                break
            C = np.flatnonzero(a == k)
            if C.size <= 1:
                continue
            if C.size > max_cluster:
                Cs = rng.choice(C, size=max_cluster, replace=False)
                if int(M[k]) not in set(int(x) for x in Cs):
                    Cs = np.append(Cs, M[k])
            else:
                Cs = C
            sub = dp.between(Cs, Cs)
            j = int(Cs[int(np.argmin(sub.sum(axis=0, dtype=np.float64)))])
            if j != int(M[k]):
                M[k] = j
                changed = True
        if not changed:
            break
    return [int(x) for x in M]


def swap_best(dp, M, cand, deadline):
    """Best (delta, position k, new node) over given candidate insertion nodes."""
    n = dp.n
    Marr = np.asarray(M, dtype=np.int64)
    p = len(Marr)
    a, d1, d2 = nearest_two(dp, Marr)
    order = np.argsort(a, kind="stable")
    a_s = a[order]
    starts = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1]])
    groups = a_s[starts]
    d1sum = float(d1.sum(dtype=np.float64))
    best_delta = -1e-6
    best_k = -1
    best_node = -1
    cc = max(1, CHUNK_ELEMS // max(n, 1))
    for s in range(0, len(cand), cc):
        if time.time() > deadline and best_k >= 0:
            break
        cb = cand[s:s + cc]
        DF = dp.cols(cb)
        base = np.minimum(DF, d1[:, None])
        total = base.sum(axis=0, dtype=np.float64) - d1sum
        diff = np.minimum(DF, d2[:, None]) - base
        Sseg = np.add.reduceat(diff[order], starts, axis=0, dtype=np.float64)
        Sfull = np.zeros((p, len(cb)), dtype=np.float64)
        Sfull[groups] = Sseg
        delta = total[None, :] + Sfull
        k, f = np.unravel_index(int(np.argmin(delta)), delta.shape)
        if float(delta[k, f]) < best_delta:
            best_delta = float(delta[k, f])
            best_k = int(k)
            best_node = int(cb[f])
    return best_delta, best_k, best_node


def local_search(dp, M_init, deadline, rng, record=None):
    n = dp.n
    M = list(dict.fromkeys(int(x) for x in M_init))
    full = (float(n) * float(n - len(M)) <= 3e8)
    fails = 0
    while time.time() < deadline:
        inS = np.zeros(n, dtype=bool)
        inS[M] = True
        cand = np.flatnonzero(~inS)
        if cand.size == 0:
            break
        if not full:
            samp = max(200, int(2e8 // max(n, 1)))
            if cand.size > samp:
                cand = rng.choice(cand, size=samp, replace=False)
        delta, k, node = swap_best(dp, M, cand, deadline)
        if k >= 0 and delta < -1e-7:
            M[k] = node
            fails = 0
            if record is not None:
                try:
                    record(M)
                except Exception:
                    pass
        else:
            if full:
                break
            fails += 1
            if fails >= 2:
                break
    return M


def load_instance(path):
    with open(path, "r") as fh:
        inst = json.load(fh)
    n = int(inst["n"])
    p = int(inst["p"])
    X = None
    coords = inst.get("node_coordinates")
    if isinstance(coords, list) and len(coords) == n and n > 0 and \
            isinstance(coords[0], (list, tuple)) and len(coords[0]) >= 2:
        try:
            X = np.array(coords, dtype=np.float64)[:, :2]
        except Exception:
            X = None
    D = None
    cm = inst.get("cost_matrix")
    cm_ok = isinstance(cm, list) and len(cm) == n and n > 0 and \
        isinstance(cm[0], (list, tuple)) and len(cm[0]) == n
    if cm_ok:
        try:
            D = np.empty((n, n), dtype=np.float32)
            for i in range(n):
                D[i] = cm[i]
                cm[i] = None
        except Exception:
            D = None
    inst["cost_matrix"] = None
    if D is None and X is None:
        raise ValueError("Instance has neither a valid cost matrix nor coordinates.")
    return n, p, D, X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    tl = max(5, int(args.time_limit))
    deadline = start + tl - max(2.0, 0.03 * tl)

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    n, p, D, X = load_instance(args.instance_path)
    dp = DistProvider(D=D, X=X)
    rng = np.random.default_rng(0)

    p = max(1, min(p, n))
    if p >= n:
        sol = build_solution(dp, list(range(n)))
        if logger:
            try:
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)
        return

    best = {"obj": float("inf"), "M": None}

    def record(M, obj=None):
        try:
            Ms = sorted(set(int(x) for x in M))
            if len(Ms) != p:
                return
            if obj is None:
                obj = solution_objective(dp, Ms)
            if obj < best["obj"] - 1e-9:
                best["obj"] = obj
                best["M"] = Ms
                if logger:
                    try:
                        logger.log_solution(int(round(obj)), build_solution(dp, Ms))
                    except Exception:
                        try:
                            logger.log(int(round(obj)))
                        except Exception:
                            pass
        except Exception:
            pass

    # Guaranteed fallback solution
    try:
        M_fallback = [int(x) for x in rng.choice(n, size=p, replace=False)]
        record(M_fallback)
    except Exception:
        M_fallback = list(range(p))
        record(M_fallback)
    if best["M"] is None:
        best["M"] = sorted(M_fallback)
        best["obj"] = float("inf")

    # ---------- Phase 1: seeding ----------
    try:
        seed_deadline = min(deadline, start + 0.25 * tl)
        cap = int(2.5e9 / (float(p) * float(n) + 1.0))
        if cap >= 40:
            M0 = greedy_construct(dp, p, seed_deadline, rng, min(cap, n))
        else:
            M0 = kmeanspp_seed(dp, p, rng)
        if len(set(M0)) == p:
            record(M0)
    except Exception:
        pass

    # ---------- Phase 2: alternate (medoid) improvement ----------
    try:
        alt_deadline = min(deadline, start + 0.45 * tl)
        M1 = alternate_medoid(dp, best["M"], alt_deadline, rng)
        record(M1)
    except Exception:
        pass

    # ---------- Phase 3: swap local search ----------
    try:
        ls_deadline = deadline
        if D is not None and n <= 350:
            ls_deadline = min(deadline, start + 0.6 * tl)
        M2 = local_search(dp, best["M"], ls_deadline, rng, record=record)
        record(M2)
    except Exception:
        pass

    # ---------- Phase 4: exact MIP for small dense instances ----------
    mip_done = False
    if D is not None and n <= 350 and (deadline - time.time()) > 10:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            Dd = D.astype(np.float64)
            m = gp.Model("pmedian")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time())

            y = m.addMVar(n, vtype=GRB.BINARY, name="y")
            x = m.addMVar((n, n), lb=0.0, ub=1.0, name="x")
            m.addConstr(x.sum(axis=1) == 1)
            for j in range(n):
                m.addConstr(x[:, j] <= y[j])
            m.addConstr(y.sum() == p)
            m.setObjective((Dd * x).sum(), GRB.MINIMIZE)

            ystart = np.zeros(n)
            for j in best["M"]:
                ystart[j] = 1.0
            y.Start = ystart
            yvars = y.tolist()

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = np.array(model.cbGetSolution(yvars))
                        med = sorted(int(z) for z in np.argsort(-vals)[:p])
                        record(med)
                    except Exception:
                        pass

            m.optimize(cb)
            if m.SolCount > 0:
                vals = np.array(y.X)
                med = sorted(int(z) for z in np.argsort(-vals)[:p])
                record(med)
            mip_done = (m.Status == GRB.OPTIMAL)
        except Exception:
            mip_done = False

    # ---------- Phase 5: perturb + re-optimize with remaining time ----------
    if not mip_done:
        while time.time() < deadline - 1.0:
            try:
                M = list(best["M"])
                k = max(1, p // 8)
                inS = np.zeros(n, dtype=bool)
                inS[M] = True
                F = np.flatnonzero(~inS)
                if F.size == 0:
                    break
                out_idx = rng.choice(p, size=min(k, p), replace=False)
                in_nodes = rng.choice(F, size=min(k, F.size), replace=False)
                for t in range(min(len(out_idx), len(in_nodes))):
                    M[int(out_idx[t])] = int(in_nodes[t])
                M = list(dict.fromkeys(M))
                while len(M) < p:
                    inS2 = np.zeros(n, dtype=bool)
                    inS2[M] = True
                    rest = np.flatnonzero(~inS2)
                    if rest.size == 0:
                        break
                    M.append(int(rng.choice(rest)))
                if len(M) != p:
                    break
                sub_dl = min(deadline, time.time() + max(5.0, 0.15 * tl))
                M3 = local_search(dp, M, sub_dl, rng)
                record(M3)
            except Exception:
                break

    if best["M"] is None or len(best["M"]) != p:
        best["M"] = sorted(int(x) for x in rng.choice(n, size=p, replace=False))

    sol = build_solution(dp, best["M"])
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()