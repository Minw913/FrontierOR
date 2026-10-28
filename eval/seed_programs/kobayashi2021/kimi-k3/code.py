import argparse
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    return ap.parse_args()


def set_params(model, time_limit=None):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    if time_limit is not None:
        model.Params.TimeLimit = max(1e-3, float(time_limit))


def build_model(N, S, k, beta, gamma, mu_bar, mu, pvec, Rmat,
                allowed=None, with_z=False):
    m = gp.Model("portfolio_cvar")
    if allowed is None:
        idxs = None
        n = N
    else:
        idxs = np.asarray(allowed, dtype=int)
        n = int(idxs.size)

    x = m.addMVar(n, lb=0.0, ub=1.0, name="x")
    a = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="a")
    v = m.addVar(lb=0.0, ub=GRB.INFINITY, name="v")

    q = 1.0 / (2.0 * gamma)
    m.setObjective(q * (x @ x) + a + v, GRB.MINIMIZE)

    m.addConstr(x.sum() == 1.0)
    mu_sel = mu if idxs is None else mu[idxs]
    m.addConstr(mu_sel @ x >= mu_bar)

    z = None
    if with_z:
        z = m.addMVar(N, vtype=GRB.BINARY, name="z")
        m.addConstr(z.sum() <= float(k))
        link = (x <= z) if idxs is None else (x <= z[idxs])
        m.addConstr(link)

    if S > 0:
        y = m.addMVar(S, lb=0.0, name="y")
        Rsel = Rmat if idxs is None else Rmat[:, idxs]
        m.addConstr(Rsel @ x + y + a >= 0.0)
        w = 1.0 / (1.0 - beta)
        m.addConstr(v >= w * (pvec @ y))
    else:
        y = None

    return m, x, a, v, y, z, idxs


def polish(x, Rmat, pvec, beta, gamma):
    """Given x, optimally recompute a (minimizes a + v) and the exact v."""
    reg = float(x @ x) / (2.0 * gamma)
    S = Rmat.shape[0]
    if S == 0:
        return reg, 0.0, 0.0
    L = -(Rmat @ x)
    order = np.argsort(L)
    Ls = L[order]
    ps = pvec[order]
    cum_p = np.cumsum(ps)
    cum_pL = np.cumsum(ps * Ls)
    tot_p = cum_p[-1]
    tot_pL = cum_pL[-1]
    js = np.searchsorted(Ls, Ls, side="right")
    jsm1 = np.clip(js - 1, 0, len(Ls) - 1)
    before_p = np.where(js > 0, cum_p[jsm1], 0.0)
    before_pL = np.where(js > 0, cum_pL[jsm1], 0.0)
    Gt_p = tot_p - before_p
    Gt_pL = tot_pL - before_pL
    w = 1.0 / (1.0 - beta)
    v_vals = np.clip((Gt_pL - Ls * Gt_p) * w, 0.0, None)
    f = Ls + v_vals
    i = int(np.argmin(f))
    a = float(Ls[i])
    v = float(v_vals[i])
    return reg + a + v, a, v


def trivial_start(N, k, mu, mu_bar):
    kk = max(1, min(k, N))
    sel = np.argsort(-mu)[:kk]
    imax = int(np.argmax(mu))
    x = np.zeros(N)
    m_eq = float(mu[sel].mean())
    if (m_eq >= mu_bar) or (mu[imax] <= mu_bar):
        x[sel] = 1.0 / kk
    else:
        t = (mu_bar - m_eq) / (mu[imax] - m_eq)
        t = float(np.clip(t, 0.0, 1.0))
        x[sel] = (1.0 - t) / kk
        x[imax] += t
    return x


def solve_qp(N, S, k, beta, gamma, mu_bar, mu, pvec, Rmat, allowed, budget):
    try:
        m, x, a, v, y, z, idxs = build_model(
            N, S, k, beta, gamma, mu_bar, mu, pvec, Rmat,
            allowed=allowed, with_z=False)
        set_params(m, time_limit=budget)
        m.optimize()
        if m.SolCount > 0:
            xv = np.asarray(x.X, dtype=float)
            xfull = np.zeros(N)
            if idxs is None:
                xfull = xv
            else:
                xfull[idxs] = xv
            return xfull
    except Exception:
        return None
    return None


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()
    total = max(0.1, float(args.time_limit))
    deadline = t0 + total
    BUF = 0.3

    def rem():
        return deadline - time.time()

    with open(args.instance_path) as f:
        inst = json.load(f)

    N = int(inst["N"])
    S = int(inst["S"])
    k = int(inst["k"])
    beta = float(inst["beta"])
    gamma = float(inst["gamma"])
    mu_bar = float(inst["mu_bar"])
    mu = np.asarray(inst["mu"], dtype=float)
    if S > 0:
        Rmat = np.asarray(inst["scenarios"], dtype=float)
        if Rmat.size == 0:
            Rmat = np.zeros((0, N))
        ps = inst.get("p_s", 1.0 / S)
        if np.isscalar(ps):
            pvec = np.full(S, float(ps))
        else:
            pvec = np.asarray(ps, dtype=float)
    else:
        Rmat = np.zeros((0, N))
        pvec = np.zeros(0)

    best = {"obj": float("inf"), "x": None, "z": None, "a": 0.0, "v": 0.0}

    def consider(x, z=None):
        x = np.clip(np.asarray(x, dtype=float), 0.0, None)
        if z is not None:
            zz = (np.asarray(z) > 0.5).astype(int)
            x = x * zz
        s = float(x.sum())
        if s <= 0.0:
            return
        x = x / s
        if z is None:
            zz = (x > 0).astype(int)
        obj, a, v = polish(x, Rmat, pvec, beta, gamma)
        if obj < best["obj"]:
            best["obj"] = float(obj)
            best["x"] = x
            best["z"] = zz
            best["a"] = float(a)
            best["v"] = float(v)
            if logger:
                sol = {
                    "objective_value": float(obj),
                    "x": [float(t) for t in x],
                    "z": [int(t) for t in zz],
                    "a": float(a),
                    "v": float(v),
                }
                try:
                    logger.log_solution(float(obj), sol)
                except Exception:
                    pass

    # Guaranteed feasible (best-effort) starting point.
    try:
        consider(trivial_start(N, k, mu, mu_bar))
    except Exception:
        pass

    def finalize():
        sol = {
            "objective_value": float(best["obj"]),
            "x": [float(t) for t in best["x"]],
            "z": [int(t) for t in best["z"]],
            "a": float(best["a"]),
            "v": float(best["v"]),
        }
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        if logger is not None and hasattr(logger, "close"):
            try:
                logger.close()
            except Exception:
                pass

    # Case where cardinality is non-binding: solve the exact convex QP.
    if k >= N:
        b = max(0.05, rem() - BUF)
        xr = solve_qp(N, S, k, beta, gamma, mu_bar, mu, pvec, Rmat,
                      allowed=None, budget=b)
        if xr is not None:
            consider(xr)
        finalize()
        return

    # Heuristic phase: solve the continuous relaxation; if its support is
    # already within the cardinality limit, we have a provable optimum.
    heur_budget = min(0.35 * total, 45.0, max(0.0, rem() - BUF))
    if heur_budget >= 1.0:
        b1 = max(1.0, heur_budget * 0.55)
        xr = solve_qp(N, S, k, beta, gamma, mu_bar, mu, pvec, Rmat,
                      allowed=None, budget=b1)
        if xr is not None:
            supp = np.where(xr > 1e-8)[0]
            if supp.size <= max(1, k):
                consider(xr)
                finalize()
                return
            sel = np.argsort(-xr)[: max(1, k)]
            b2 = min(b1, max(0.0, rem() - BUF))
            if b2 >= 0.5 and sel.size > 0:
                xg = solve_qp(N, S, k, beta, gamma, mu_bar, mu, pvec, Rmat,
                              allowed=sel, budget=b2)
                if xg is not None:
                    consider(xg)

    # MIP phase with warm start and incumbent logging callback.
    remt = rem() - BUF
    if remt > 0.05:
        try:
            m, x, a, v, y, z, idxs = build_model(
                N, S, k, beta, gamma, mu_bar, mu, pvec, Rmat,
                allowed=None, with_z=True)
            set_params(m, time_limit=remt)

            if best["x"] is not None:
                try:
                    sx = best["x"]
                    sz = (sx > 0).astype(float)
                    obj0, a0, v0 = polish(sx, Rmat, pvec, beta, gamma)
                    x.Start = sx
                    z.Start = sz
                    a.Start = a0
                    v.Start = v0
                    if y is not None and S > 0:
                        L = -(Rmat @ sx)
                        y.Start = np.maximum(L - a0, 0.0)
                except Exception:
                    pass

            def cb(model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                if time.time() > deadline - BUF:
                    model.terminate()
                    return
                if logger is None:
                    return
                try:
                    obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                    xs = np.asarray(model.cbGetSolution(x), dtype=float)
                    zs = np.asarray(model.cbGetSolution(z), dtype=float)
                    av = float(model.cbGetSolution(a))
                    vv = float(model.cbGetSolution(v))
                    sol = {
                        "objective_value": obj,
                        "x": [float(t) for t in xs],
                        "z": [int(round(t)) for t in zs],
                        "a": av,
                        "v": vv,
                    }
                    logger.log_solution(obj, sol)
                except Exception:
                    pass

            m.optimize(cb if logger is not None else None)

            if m.SolCount > 0:
                xs = np.asarray(x.X, dtype=float)
                zs = (np.asarray(z.X) >= 0.5).astype(int)
                consider(xs, zs)
        except Exception:
            pass

    finalize()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise