import argparse
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

EPS = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    start = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    C = np.array(inst["scenarios"], dtype=float)
    N, n = C.shape
    raw_K = inst.get("K_values") or list(range(1, max(2, N)))
    K_list = sorted(set(int(k) for k in raw_K if int(k) >= 1))
    if not K_list:
        K_list = [max(1, N - 1)]

    buffer = max(2.0, min(10.0, 0.03 * args.time_limit))

    def rem():
        return args.time_limit - (time.time() - start) - buffer

    # ------------------------------------------------------------------
    # ratio matrix: r[i, s] = min_j C[s, j] / C[i, j]
    # (largest t such that scenario s dominates t * scenario i)
    # ------------------------------------------------------------------
    r = np.empty((N, N))
    for s in range(N):
        r[:, s] = np.min(C[s][None, :] / C, axis=1)

    Kmax_eff = min(N, max(min(k, N) for k in K_list))

    # greedy prefix selection (max-min coverage, k-center style)
    sel_order = []
    cover = np.full(N, -np.inf)
    for _ in range(Kmax_eff):
        M = np.maximum(cover[:, None], r)
        vals = M.min(axis=0)
        if sel_order:
            vals[np.array(sel_order)] = -np.inf
        s = int(np.argmax(vals))
        sel_order.append(s)
        cover = np.maximum(cover, r[:, s])

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def clean_lam(lamW):
        W = np.clip(np.asarray(lamW, dtype=float), 0.0, None)
        Wc = np.zeros_like(W)
        for k in range(W.shape[1]):
            col = W[:, k]
            idx = np.where(col > 1e-9)[0]
            if len(idx) == 0:
                Wc[0, k] = 1.0
            else:
                Wc[idx, k] = col[idx] / col[idx].sum()
        return Wc

    def witness(assign, W):
        D = W.T @ C  # K x n aggregates
        t = float(np.min(D[assign, :] / C))
        return t, D

    def assign_from_selection(selK):
        sub = r[:, selK]
        pos = np.argmax(sub, axis=1)
        label = {}
        assign = np.zeros(N, dtype=int)
        for i in range(N):
            p = int(pos[i])
            if p not in label:
                label[p] = len(label)
            assign[i] = label[p]
        rep = {v: selK[kk] for kk, v in label.items()}
        return assign, rep

    # ------------------------------------------------------------------
    # Phase 1: heuristic initialization for every K
    # ------------------------------------------------------------------
    one_state = {}
    one_meta = {}
    two_state = {}
    for K in K_list:
        Ke = min(K, N)
        if Ke >= N:
            assign = np.arange(N)
            lamW = np.zeros((N, K))
            lamW[np.arange(N), np.arange(N)] = 1.0
            for k in range(N, K):
                lamW[0, k] = 1.0
            W = lamW
            t1, _ = witness(assign, W)
            one_state[K] = {"assign": assign, "W": W, "t": t1}
            one_meta[K] = {"status": 2, "bound": 1.0, "gap": 0.0}
            sel = list(range(N)) + [0] * (K - N)
            two_state[K] = {"sel": sel, "t": 1.0}
        else:
            selK = sel_order[:Ke]
            t2 = float(np.min(np.max(r[:, selK], axis=1)))
            two_state[K] = {"sel": list(selK), "t": t2}
            assign, rep = assign_from_selection(selK)
            lamW = np.zeros((N, K))
            for k in range(K):
                lamW[rep.get(k, selK[0]), k] = 1.0
            W = clean_lam(lamW)
            t1, _ = witness(assign, W)
            one_state[K] = {"assign": assign, "W": W, "t": t1}
            one_meta[K] = {"status": 9, "bound": 1.0,
                           "gap": (1.0 - t1) / max(t1, 1e-10)}

    # ------------------------------------------------------------------
    # solution assembly / logging
    # ------------------------------------------------------------------
    def assemble():
        one = {}
        two = {}
        best = 0.0
        for K in K_list:
            st = one_state[K]
            t1 = float(st["t"])
            W = st["W"]
            assign = st["assign"]
            lw = {}
            for k in range(K):
                col = W[:, k]
                idx = np.where(col > 1e-12)[0]
                lw[str(k)] = {str(int(i)): float(col[i]) for i in idx}
            meta = one_meta[K]
            one[str(K)] = {
                "t": t1,
                "approximation_ratio": float(1.0 / max(t1, 1e-12)),
                "assignment": {str(i): int(assign[i]) for i in range(N)},
                "lambda_weights": lw,
                "solver_status": int(meta["status"]),
                "best_bound": float(meta["bound"]),
                "mip_gap": float(max(0.0, meta["gap"])),
            }
            ts = two_state[K]
            t2 = float(ts["t"])
            two[str(K)] = {
                "t": t2,
                "approximation_ratio": float(1.0 / max(t2, 1e-12)),
                "selected_scenarios": [int(s) for s in ts["sel"]],
            }
            best = max(best, t1, t2)
        return {"objective_value": best,
                "one_stage_IP_mu": one,
                "two_stage_IP": two}

    best_logged = [-1.0]

    def maybe_log():
        sol = assemble()
        if logger and sol["objective_value"] > best_logged[0] + 1e-12:
            best_logged[0] = sol["objective_value"]
            try:
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass
        return sol

    maybe_log()

    def new_model(tl):
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(0.5, float(tl))
        return m

    # ------------------------------------------------------------------
    # Phase 2: exact two-stage selection (K-center-like MIP)
    # ------------------------------------------------------------------
    def solve_two_stage(K, tl):
        Ke = min(K, N)
        ts = two_state[K]
        t_lb = ts["t"]
        if t_lb >= 1.0 - 1e-9 or Ke >= N:
            return
        m = new_model(tl)
        z = m.addVars(N, vtype=GRB.BINARY)
        t = m.addVar(lb=max(0.0, t_lb - 1e-9), ub=1.0)
        m.setObjective(t, GRB.MAXIMIZE)
        m.addConstr(gp.quicksum(z[s] for s in range(N)) == Ke)
        for i in range(N):
            cand = np.where(r[i] >= t_lb - 1e-9)[0]
            if len(cand) == 0:
                cand = np.array([int(np.argmax(r[i]))])
            e_sum = gp.LinExpr()
            e_t = gp.LinExpr()
            for s in cand:
                y = m.addVar(ub=1.0)
                m.addConstr(y <= z[int(s)])
                e_sum.add(y)
                e_t.addTerms(float(r[i, s]), y)
            m.addConstr(e_sum == 1)
            m.addConstr(t <= e_t)
        selset = set(ts["sel"])
        for s in range(N):
            z[s].Start = 1.0 if s in selset else 0.0
        m.optimize()
        if m.SolCount > 0:
            sel = [s for s in range(N) if z[s].X > 0.5]
            if len(sel) == Ke:
                t_w = float(np.min(np.max(r[:, sel], axis=1)))
                if t_w > ts["t"] + 1e-12:
                    ts["sel"] = sel
                    ts["t"] = t_w
                    st = one_state[K]
                    assign, rep = assign_from_selection(sel)
                    lamW = np.zeros((N, K))
                    for k in range(K):
                        lamW[rep.get(k, sel[0]), k] = 1.0
                    W = clean_lam(lamW)
                    t1, _ = witness(assign, W)
                    if t1 > st["t"] + 1e-12:
                        st["assign"] = assign
                        st["W"] = W
                        st["t"] = t1

    # ------------------------------------------------------------------
    # Phase 3: one-stage IP-mu (aggregates = convex combinations)
    # ------------------------------------------------------------------
    def solve_one_stage(K, tl):
        Ke = min(K, N)
        st = one_state[K]
        if st["t"] >= 1.0 - 1e-9 or Ke >= N:
            return
        if N * Ke * n > 5e6:
            return
        m = new_model(tl)
        t = m.addVar(lb=max(0.0, st["t"] - 1e-7), ub=1.0)
        m.setObjective(t, GRB.MAXIMIZE)
        mu = {}
        for i in range(N):
            for k in range(min(i + 1, Ke)):
                mu[(i, k)] = m.addVar(vtype=GRB.BINARY)
        lam = m.addVars(N, Ke, lb=0.0, ub=1.0)
        d = m.addVars(Ke, n, lb=0.0)
        for k in range(Ke):
            m.addConstr(gp.quicksum(lam[i, k] for i in range(N)) == 1)
            for j in range(n):
                e = gp.LinExpr()
                e.addTerms(C[:, j].tolist(), [lam[i, k] for i in range(N)])
                m.addConstr(d[k, j] == e)
        for i in range(N):
            ks = list(range(min(i + 1, Ke)))
            m.addConstr(gp.quicksum(mu[(i, k)] for k in ks) == 1)
            for k in ks:
                cij_row = C[i]
                for j in range(n):
                    # d_kj >= t*C_ij - C_ij*(1 - mu_ik)
                    m.addConstr(d[k, j] - cij_row[j] * t
                                + cij_row[j] * (1.0 - mu[(i, k)]) >= 0.0)
        # symmetry breaking (first-appearance cluster labels)
        for i in range(1, N):
            for k in range(1, min(i + 1, Ke)):
                m.addConstr(mu[(i, k)] <=
                            gp.quicksum(mu[(i2, k - 1)] for i2 in range(k - 1, i)))

        # warm start
        assign0 = st["assign"]
        W0 = st["W"]
        for (i, k), v in mu.items():
            v.Start = 1.0 if int(assign0[i]) == k else 0.0
        for i in range(N):
            for k in range(Ke):
                lam[i, k].Start = float(W0[i, k])
        t.Start = max(0.0, st["t"] - 1e-9)

        mu_items = list(mu.items())
        mu_vars = [v for _, v in mu_items]
        lam_vars = [lam[i, k] for i in range(N) for k in range(Ke)]

        def reconstruct(muv, lamv):
            assign = np.zeros(N, dtype=int)
            bestval = np.full(N, -1.0)
            for (key, _), val in zip(mu_items, muv):
                i, k = key
                if val > bestval[i]:
                    bestval[i] = val
                    assign[i] = k
            lamW = np.array(lamv, dtype=float).reshape(N, Ke)
            W = clean_lam(lamW)
            t_w, _ = witness(assign, W)
            return assign, W, t_w

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    muv = model.cbGetSolution(mu_vars)
                    lamv = model.cbGetSolution(lam_vars)
                    assign, W, t_w = reconstruct(muv, lamv)
                    if t_w > st["t"] + 1e-12:
                        st["assign"] = assign
                        st["W"] = W
                        st["t"] = t_w
                        one_meta[K] = {"status": 13, "bound": 1.0,
                                       "gap": (1.0 - t_w) / max(t_w, 1e-10)}
                        maybe_log()
                except Exception:
                    pass

        try:
            m.optimize(cb)
        except Exception:
            pass

        status = int(m.Status) if m.Status is not None else 9
        try:
            bound = float(m.ObjBound)
            if not np.isfinite(bound):
                bound = 1.0
        except Exception:
            bound = 1.0
        bound = min(bound, 1.0)
        if m.SolCount > 0:
            try:
                muv = [v.X for v in mu_vars]
                lamv = [v.X for v in lam_vars]
                assign, W, t_w = reconstruct(muv, lamv)
                if t_w > st["t"] + 1e-12:
                    st["assign"] = assign
                    st["W"] = W
                    st["t"] = t_w
            except Exception:
                pass
        gap = (bound - st["t"]) / max(abs(st["t"]), 1e-10)
        one_meta[K] = {"status": status, "bound": max(bound, st["t"]),
                       "gap": max(0.0, gap)}
        maybe_log()

    # ------------------------------------------------------------------
    # run phases with time budgeting
    # ------------------------------------------------------------------
    Ks_desc = sorted(K_list, reverse=True)

    two_jobs = [K for K in Ks_desc
                if min(K, N) < N and two_state[K]["t"] < 1.0 - 1e-9]
    if two_jobs and rem() > 2 and N <= 600:
        budget = 0.2 * rem()
        slice_t = budget / len(two_jobs)
        for K in two_jobs:
            if rem() <= 1:
                break
            try:
                solve_two_stage(K, min(max(1.0, slice_t), rem()))
            except Exception:
                pass
        maybe_log()

    one_jobs = [K for K in Ks_desc
                if min(K, N) < N and one_state[K]["t"] < 1.0 - 1e-9]
    left = len(one_jobs)
    for K in one_jobs:
        rr = rem()
        if rr <= 1:
            break
        tl = min(rr, max(2.0, 1.4 * rr / max(1, left)))
        try:
            solve_one_stage(K, tl)
        except Exception:
            pass
        left -= 1

    # ------------------------------------------------------------------
    # final output
    # ------------------------------------------------------------------
    sol = assemble()
    if logger and sol["objective_value"] > best_logged[0] + 1e-12:
        best_logged[0] = sol["objective_value"]
        try:
            logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()