import argparse
import heapq
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


# --------------------------------------------------------------------------
# Instance handling
# --------------------------------------------------------------------------

class Instance:
    def __init__(self, data):
        self.type = str(data.get("instance_type", "LOC")).strip().upper()
        if self.type not in ("LOC", "COV", "INF"):
            self.type = "LOC"
        self.n = int(data.get("n", 0) or 0)
        self.m = int(data.get("m", 0) or 0)
        self.k = int(data.get("k", 0) or 0)
        W = np.asarray(data.get("data_matrix", []), dtype=float)
        if W.size == 0:
            W = np.zeros((max(self.m, 1), max(self.n, 1)))
        W = np.atleast_2d(W)
        if self.m == 0:
            self.m = W.shape[0]
        if self.n == 0:
            self.n = W.shape[1]
        if W.shape != (self.m, self.n):
            W = W.reshape(self.m, self.n)
        self.W = W
        nw = np.asarray(data.get("node_weights") or [0.0] * self.m,
                        dtype=float).ravel()
        if nw.size != self.m:
            nw = np.zeros(self.m)
        self.nw = nw
        ew = np.asarray(data.get("element_weights") or [0.0] * self.n,
                        dtype=float).ravel()
        if ew.size != self.n:
            ew = np.zeros(self.n)
        self.ew = ew
        if self.type == "COV":
            self.A = (self.W > 0.5).astype(float)
        if self.type == "INF":
            P = self.W * self.ew[None, :]
            self.P = np.clip(P, 0.0, 1.0)
            self.Pc = np.clip(P, 0.0, 1.0 - 1e-12)


def exact_objective(inst, sel):
    """True value f(S) as defined by the instance type."""
    if sel is None or len(sel) == 0:
        return 0.0
    idx = np.asarray(list(sel), dtype=int)
    if inst.type == "LOC":
        return float(inst.W[:, idx].max(axis=1).sum())
    if inst.type == "COV":
        cov = inst.A[:, idx].max(axis=1) > 0.5
        return float(inst.nw[cov].sum())
    q = (1.0 - inst.P[:, idx]).prod(axis=1)
    return float(inst.m - q.sum())


class State:
    """Incremental state for fast marginal-gain evaluation."""
    def __init__(self, inst):
        self.inst = inst
        if inst.type == "LOC":
            self.cur = np.zeros(inst.m)
        elif inst.type == "COV":
            self.wunc = inst.nw.astype(float).copy()
        else:
            self.q = np.ones(inst.m)

    def gain(self, j):
        inst = self.inst
        if inst.type == "LOC":
            return float(np.maximum(inst.W[:, j] - self.cur, 0.0).sum())
        if inst.type == "COV":
            return float(self.wunc @ inst.A[:, j])
        return float(self.q @ inst.Pc[:, j])

    def apply(self, j):
        inst = self.inst
        if inst.type == "LOC":
            np.maximum(self.cur, inst.W[:, j], out=self.cur)
        elif inst.type == "COV":
            self.wunc[inst.A[:, j] > 0.5] = 0.0
        else:
            self.q *= 1.0 - inst.Pc[:, j]

    def all_gains(self):
        inst = self.inst
        if inst.type == "LOC":
            return np.maximum(inst.W - self.cur[:, None], 0.0).sum(axis=0)
        if inst.type == "COV":
            return inst.A.T @ self.wunc
        return inst.Pc.T @ self.q


def removal_losses(inst, S):
    """loss[p] = f(S) - f(S \\ {S[p]}) for each position p."""
    m = inst.m
    if inst.type == "LOC":
        sub = inst.W[:, S]
        argb = np.argmax(sub, axis=1)
        best = sub[np.arange(m), argb]
        if len(S) >= 2:
            masked = sub.copy()
            masked[np.arange(m), argb] = -np.inf
            second = masked.max(axis=1)
        else:
            second = np.zeros(m)
        return np.bincount(argb, weights=best - second, minlength=len(S))
    if inst.type == "COV":
        cnt = inst.A[:, S].sum(axis=1)
        single = (cnt > 0.5) & (cnt < 1.5)
        if single.any():
            return (inst.A[single][:, S] * inst.nw[single][:, None]).sum(axis=0)
        return np.zeros(len(S))
    P = inst.Pc
    q = (1.0 - P[:, S]).prod(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = q[:, None] * (P[:, S] / (1.0 - P[:, S]))
    contrib = np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
    return contrib.sum(axis=0)


# --------------------------------------------------------------------------
# Greedy (lazy) and randomized greedy
# --------------------------------------------------------------------------

def greedy(inst, k, deadline):
    n = inst.n
    st = State(inst)
    g0 = st.all_gains()
    heap = [(-float(g0[j]), j) for j in range(n)]
    heapq.heapify(heap)
    S = []
    kk = min(k, n)
    pops = 0
    while len(S) < kk and heap:
        _, j = heapq.heappop(heap)
        gj = st.gain(j)
        if heap and (-heap[0][0]) > gj + 1e-12:
            heapq.heappush(heap, (-gj, j))
        else:
            st.apply(j)
            S.append(j)
        pops += 1
        if (pops & 255) == 0 and time.time() > deadline:
            break
    return S


def greedy_random(inst, k, rng, deadline, top_r=3):
    n = inst.n
    st = State(inst)
    kk = min(k, n)
    S = []
    sel = np.zeros(n, dtype=bool)
    for _ in range(kk):
        g = st.all_gains()
        g[sel] = -np.inf
        remaining = n - len(S)
        r = min(top_r, remaining)
        if r <= 1:
            j = int(np.argmax(g))
        else:
            idx = np.argpartition(-g, r - 1)[:r]
            j = int(idx[int(rng.integers(r))])
        if not np.isfinite(g[j]):
            break
        st.apply(j)
        sel[j] = True
        S.append(j)
        if (len(S) & 7) == 0 and time.time() > deadline:
            break
    return S


# --------------------------------------------------------------------------
# Local search (1-swap hill climbing, vectorized per removed element)
# --------------------------------------------------------------------------

def local_search(inst, S, update_best, deadline):
    t = inst.type
    n, m = inst.n, inst.m
    S = list(S)
    if not S:
        return S
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[np.asarray(S, dtype=int)] = True
    obj = exact_objective(inst, S)
    tol = 1e-9 * max(1.0, abs(obj))
    improved = True
    while improved:
        improved = False
        if t == "LOC":
            sub = inst.W[:, S]
            argb = np.argmax(sub, axis=1)
            bestv = sub[np.arange(m), argb]
            if len(S) >= 2:
                masked = sub.copy()
                masked[np.arange(m), argb] = -np.inf
                second = masked.max(axis=1)
            else:
                second = np.zeros(m)
            loss_vec = np.bincount(argb, weights=bestv - second,
                                   minlength=len(S))
            order = np.argsort(loss_vec, kind="stable")
            for pos in order:
                if time.time() > deadline:
                    return S
                mask = (argb == pos)
                if mask.any():
                    curp = np.where(mask, second, bestv)
                    loss = float((bestv[mask] - second[mask]).sum())
                else:
                    curp = bestv
                    loss = 0.0
                gains = np.maximum(inst.W - curp[:, None], 0.0).sum(axis=0)
                gains[sel_mask] = -np.inf
                jbest = int(np.argmax(gains))
                dj = float(gains[jbest])
                if not np.isfinite(dj):
                    continue
                if dj - loss > tol:
                    old = S[pos]
                    S[pos] = jbest
                    sel_mask[old] = False
                    sel_mask[jbest] = True
                    obj = exact_objective(inst, S)
                    tol = 1e-9 * max(1.0, abs(obj))
                    update_best(S, obj)
                    improved = True
                    break
        elif t == "COV":
            A = inst.A
            nw = inst.nw
            cnt = A[:, S].sum(axis=1)
            obj = float(nw[cnt > 0.5].sum())
            single = (cnt > 0.5) & (cnt < 1.5)
            if single.any():
                loss_vec = (A[single][:, S] * nw[single][:, None]).sum(axis=0)
            else:
                loss_vec = np.zeros(len(S))
            order = np.argsort(loss_vec, kind="stable")
            for pos in order:
                if time.time() > deadline:
                    return S
                cntp = cnt - A[:, S[pos]]
                base = float(nw[cntp > 0.5].sum())
                uw = np.where(cntp > 0.5, 0.0, nw)
                gains = A.T @ uw
                gains[sel_mask] = -np.inf
                jbest = int(np.argmax(gains))
                dj = float(gains[jbest])
                if not np.isfinite(dj):
                    continue
                if base + dj - obj > tol:
                    old = S[pos]
                    S[pos] = jbest
                    sel_mask[old] = False
                    sel_mask[jbest] = True
                    obj = exact_objective(inst, S)
                    tol = 1e-9 * max(1.0, abs(obj))
                    update_best(S, obj)
                    improved = True
                    break
        else:
            P = inst.Pc
            q = (1.0 - P[:, S]).prod(axis=1)
            obj = exact_objective(inst, S)
            for pos in range(len(S)):
                if time.time() > deadline:
                    return S
                with np.errstate(divide="ignore", invalid="ignore"):
                    qp = q / (1.0 - P[:, S[pos]])
                qp = np.nan_to_num(qp, nan=0.0, posinf=0.0, neginf=0.0)
                base = float(m - qp.sum())
                gains = P.T @ qp
                gains[sel_mask] = -np.inf
                jbest = int(np.argmax(gains))
                dj = float(gains[jbest])
                if not np.isfinite(dj):
                    continue
                if base + dj - obj > tol:
                    old = S[pos]
                    S[pos] = jbest
                    sel_mask[old] = False
                    sel_mask[jbest] = True
                    obj = exact_objective(inst, S)
                    tol = 1e-9 * max(1.0, abs(obj))
                    update_best(S, obj)
                    improved = True
                    break
    return S


# --------------------------------------------------------------------------
# Gurobi models
# --------------------------------------------------------------------------

def _setup_model(name, time_left):
    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model(name)
    p = model.Params
    p.Seed = 0
    p.MIPGap = 1e-4
    p.NumericFocus = 0
    p.Threads = 1
    p.OutputFlag = 0
    p.TimeLimit = max(1.0, float(time_left))
    return model, gp, GRB


def _incumbent_callback(inst, update_best, GRB):
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = model.cbGetSolution(model._xvars)
                if isinstance(xv, dict):
                    xarr = np.array([float(xv[j]) for j in range(inst.n)])
                else:
                    xarr = np.asarray(xv, dtype=float).ravel()[: inst.n]
                sel = np.nonzero(xarr > 0.5)[0]
                if sel.size > inst.k:
                    sel = sel[: inst.k]
                update_best(sel.tolist())
            except Exception:
                pass
    return cb


def mip_loc(inst, time_left, update_best, warm):
    model, gp, GRB = _setup_model("loc", time_left)
    import scipy.sparse as sp
    n, m, k, W = inst.n, inst.m, inst.k, inst.W
    rows, cols = np.nonzero(W > 0)
    wvals = W[rows, cols].astype(float)
    P = int(rows.size)
    if P == 0:
        return False
    N = n + P
    vtype = [GRB.BINARY] * n + [GRB.CONTINUOUS] * P
    allv = model.addMVar(N, lb=0.0, ub=1.0, vtype=vtype)
    model.setObjective(allv[n:] @ wvals, GRB.MAXIMIZE)
    model.addConstr(allv[:n].sum() <= float(k))
    # sum_j y_ij <= 1 for each client i
    A2 = sp.csr_matrix((np.ones(P), (rows, n + np.arange(P))), shape=(m, N))
    model.addMConstr(A2, allv, '<', np.ones(m))
    # y_ij <= x_j
    A3 = sp.csr_matrix(
        (np.concatenate([np.ones(P), -np.ones(P)]),
         (np.concatenate([np.arange(P), np.arange(P)]),
          np.concatenate([n + np.arange(P), cols]))),
        shape=(P, N))
    model.addMConstr(A3, allv, '<', np.zeros(P))
    model.update()
    if warm:
        try:
            start = np.zeros(N)
            warm = list(warm)
            start[np.asarray(warm, dtype=int)] = 1.0
            sub = W[:, warm]
            argb = np.argmax(sub, axis=1)
            bestv = sub[np.arange(m), argb]
            pos = {}
            for pidx in range(P):
                pos[(int(rows[pidx]), int(cols[pidx]))] = pidx
            for i in range(m):
                if bestv[i] > 0:
                    pidx = pos.get((i, int(warm[argb[i]])))
                    if pidx is not None:
                        start[n + pidx] = 1.0
            allv.setAttr("Start", start)
        except Exception:
            pass
    model._xvars = allv
    model.optimize(_incumbent_callback(inst, update_best, GRB))
    if model.SolCount > 0:
        try:
            xv = np.asarray(allv.X, dtype=float).ravel()[:n]
            sel = np.nonzero(xv > 0.5)[0].tolist()
            update_best(sel)
        except Exception:
            pass
    return model.Status == GRB.OPTIMAL


def mip_cov(inst, time_left, update_best, warm):
    model, gp, GRB = _setup_model("cov", time_left)
    import scipy.sparse as sp
    n, m, k = inst.n, inst.m, inst.k
    A = inst.A
    rows, cols = np.nonzero(A > 0.5)
    P = int(rows.size)
    if P == 0:
        return False
    N = n + m
    vtype = [GRB.BINARY] * n + [GRB.CONTINUOUS] * m
    allv = model.addMVar(N, lb=0.0, ub=1.0, vtype=vtype)
    model.setObjective(allv[n:] @ inst.nw, GRB.MAXIMIZE)
    model.addConstr(allv[:n].sum() <= float(k))
    # y_i <= sum_{j covers i} x_j
    d = np.concatenate([-np.ones(P), np.ones(m)])
    r = np.concatenate([rows, np.arange(m)])
    c = np.concatenate([cols, n + np.arange(m)])
    Am = sp.csr_matrix((d, (r, c)), shape=(m, N))
    model.addMConstr(Am, allv, '<', np.zeros(m))
    model.update()
    if warm:
        try:
            start = np.zeros(N)
            warm = list(warm)
            start[np.asarray(warm, dtype=int)] = 1.0
            cov = A[:, warm].max(axis=1) > 0.5
            start[n + np.nonzero(cov)[0]] = 1.0
            allv.setAttr("Start", start)
        except Exception:
            pass
    model._xvars = allv
    model.optimize(_incumbent_callback(inst, update_best, GRB))
    if model.SolCount > 0:
        try:
            xv = np.asarray(allv.X, dtype=float).ravel()[:n]
            sel = np.nonzero(xv > 0.5)[0].tolist()
            update_best(sel)
        except Exception:
            pass
    return model.Status == GRB.OPTIMAL


def mip_nw(inst, time_left, update_best, warm):
    """Exact MIP for any monotone submodular f via lazy Nemhauser-Wolsey cuts:
       eta <= f(S) + sum_{j not in S} rho_j(S) x_j
                  - sum_{j in S} rho_j(S\\j) (1 - x_j)   for all S.
    """
    model, gp, GRB = _setup_model("nw", time_left)
    model.Params.LazyConstraints = 1
    n, k = inst.n, inst.k
    xs = model.addVars(n, vtype=GRB.BINARY, name="x")
    xs_list = [xs[j] for j in range(n)]
    full_val = exact_objective(inst, list(range(n)))
    eta = model.addVar(lb=0.0, ub=max(full_val, 0.0) + 1e-6, name="eta")
    model.setObjective(eta, GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(xs_list) <= float(k), name="card")

    def cut_data(sel):
        st = State(inst)
        for j in sel:
            st.apply(j)
        coef = np.array(st.all_gains(), dtype=float)   # rho_j(S), j not in S
        const = exact_objective(inst, sel)
        if sel:
            losses = removal_losses(inst, sel)
            coef[np.asarray(sel, dtype=int)] = losses
            const -= float(np.sum(losses))
        return const, coef

    def lin_expr(coef):
        idx = np.nonzero(np.abs(coef) > 1e-12)[0]
        return gp.LinExpr([float(coef[j]) for j in idx],
                          [xs_list[int(j)] for j in idx])

    # Static strengthening cuts at S = empty, S = full, S = incumbent.
    for S0 in ([], list(range(n))):
        try:
            const, coef = cut_data(S0)
            model.addConstr(eta <= const + lin_expr(coef))
        except Exception:
            pass
    if warm:
        try:
            const, coef = cut_data(list(warm))
            model.addConstr(eta <= const + lin_expr(coef))
        except Exception:
            pass

    model.update()
    if warm:
        try:
            for j in warm:
                xs[int(j)].Start = 1.0
            eta.Start = exact_objective(inst, warm)
        except Exception:
            pass

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = model.cbGetSolution(xs)
                xarr = np.array([float(xv[j]) for j in range(n)])
                sel = np.nonzero(xarr > 0.5)[0].tolist()
                fS = exact_objective(inst, sel)
                update_best(sel, fS)
                eta_val = float(model.cbGetSolution(eta))
                const, coef = cut_data(sel)
                if eta_val - (const + float(coef @ xarr)) > 1e-6:
                    model.cbLazy(eta <= const + lin_expr(coef))
            except Exception:
                pass

    model.optimize(cb)
    if model.SolCount > 0:
        try:
            sel = [j for j in range(n) if xs[j].X > 0.5]
            update_best(sel)
        except Exception:
            pass
    return model.Status == GRB.OPTIMAL


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    T = max(1, int(args.time_limit))
    deadline = t0 + float(T)

    best = {"obj": -float("inf"), "sel": []}

    try:
        with open(args.instance_path, "r") as f:
            data = json.load(f)
        inst = Instance(data)
    except Exception:
        try:
            with open(args.solution_path, "w") as f:
                json.dump({"objective_value": 0.0, "selected_elements": []}, f)
        except Exception:
            pass
        return

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except Exception:
            logger = None

    def update_best(sel, obj=None):
        if obj is None:
            try:
                obj = exact_objective(inst, sel)
            except Exception:
                return
        try:
            obj = float(obj)
        except Exception:
            return
        if not (obj > best["obj"] + 1e-9):
            return
        best["obj"] = obj
        best["sel"] = sorted(int(j) for j in sel)
        if logger is not None:
            sol = {"objective_value": obj,
                   "selected_elements": [int(j) for j in best["sel"]]}
            try:
                logger.log_solution(obj, sol)
            except Exception:
                pass

    def time_left():
        return deadline - time.time()

    try:
        n, m, k = inst.n, inst.m, inst.k

        if n <= 0 or k <= 0:
            update_best([], 0.0)
        elif k >= n:
            update_best(list(range(n)))
        else:
            # Phase 1: lazy greedy (1 - 1/e approximation for these
            # monotone submodular objectives).
            try:
                S = greedy(inst, k, deadline - 0.2)
            except Exception:
                S = list(range(min(k, n)))
            update_best(S)

            mn = float(m) * float(n)
            LS_GUARD = 2.0e7

            # Choose a MIP strategy.
            mip_fn = None
            have_gp = False
            have_sp = False
            try:
                import gurobipy  # noqa: F401
                have_gp = True
            except Exception:
                pass
            try:
                import scipy.sparse  # noqa: F401
                have_sp = True
            except Exception:
                pass

            if have_gp:
                nnz = int(np.count_nonzero(inst.W))
                if inst.type == "LOC":
                    if nnz > 0 and have_sp and nnz <= 300000:
                        mip_fn = mip_loc
                    elif nnz > 0 and n <= 3000:
                        mip_fn = mip_nw
                elif inst.type == "COV":
                    useful = nnz > 0 and float(inst.nw.sum()) > 0
                    if useful and have_sp and nnz <= 3000000:
                        mip_fn = mip_cov
                    elif useful and n <= 3000:
                        mip_fn = mip_nw
                else:
                    if np.any(inst.Pc) and n <= 3000:
                        mip_fn = mip_nw

            # Phase 2: local search on the greedy solution.
            if mn <= LS_GUARD and time_left() > 1.5:
                if mip_fn is not None:
                    ls_end = min(deadline - 2.0,
                                 time.time() + max(0.5, 0.20 * T))
                else:
                    ls_end = min(deadline - 1.0,
                                 time.time() + max(0.5, 0.35 * T))
                if time.time() < ls_end:
                    try:
                        local_search(inst,
                                     best["sel"] if best["sel"] else S,
                                     update_best, ls_end)
                    except Exception:
                        pass

            # Phase 3: exact MIP (time-limited, warm-started).
            proven_optimal = False
            if mip_fn is not None and time_left() > 3.0:
                try:
                    proven_optimal = mip_fn(inst, time_left() - 1.0,
                                            update_best, best["sel"])
                except Exception:
                    proven_optimal = False

            # Phase 4: use any leftover time for randomized restarts.
            if (not proven_optimal) and mn <= LS_GUARD:
                rng = np.random.default_rng(0)
                stall = 0
                iters = 0
                while time_left() > 2.0 and iters < 1000 and stall < 40:
                    iters += 1
                    before = best["obj"]
                    try:
                        S2 = greedy_random(inst, k, rng, deadline - 1.0)
                        update_best(S2)
                        if time_left() > 2.0:
                            local_search(inst, S2, update_best,
                                         deadline - 1.0)
                    except Exception:
                        break
                    if best["obj"] > before + 1e-12:
                        stall = 0
                    else:
                        stall += 1
    except Exception:
        pass

    if best["obj"] == -float("inf"):
        best["obj"] = 0.0
        best["sel"] = []
    sol = {"objective_value": float(best["obj"]),
           "selected_elements": [int(j) for j in best["sel"]]}
    try:
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
    except Exception:
        pass
    if logger is not None:
        try:
            if hasattr(logger, "close"):
                logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()