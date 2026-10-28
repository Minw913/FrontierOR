import argparse
import json
import math
import os
import time

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GP = True
except Exception:
    HAS_GP = False

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# ---------------------------------------------------------------------------
# Basic geometric data
# ---------------------------------------------------------------------------

def pair_ratio_matrix(C):
    """R[i, s] = min_d C[s, d] / C[i, d]  (max t at which rep s covers i)."""
    N = C.shape[0]
    R = np.empty((N, N), dtype=float)
    for s in range(N):
        R[:, s] = (C[s] / C).min(axis=1)
    return R


def greedy_prefix(R, maxK):
    """Greedy k-center-like selection; returns dict K -> (S, t)."""
    N = R.shape[0]
    S = []
    cov = np.full(N, -np.inf)
    out = {}
    for k in range(maxK):
        cand = np.maximum(cov[:, None], R)
        vals = cand.min(axis=0)
        if S:
            vals[S] = -np.inf
        s = int(np.argmax(vals))
        S.append(s)
        cov = np.maximum(cov, R[:, s])
        out[k + 1] = (list(S), float(cov.min()))
    return out


def fix_empty(assign, K, R2=None):
    """Ensure every aggregate cluster 0..K-1 has at least one member."""
    assign = assign.copy()
    counts = np.bincount(assign, minlength=K)
    for j in range(K):
        if counts[j] == 0:
            largest = int(np.argmax(counts))
            members = np.where(assign == largest)[0]
            if R2 is not None:
                i = members[int(np.argmax(R2[members, j]))]
            else:
                i = int(members[0])
            assign[i] = j
            counts[largest] -= 1
            counts[j] = 1
    return assign


# ---------------------------------------------------------------------------
# Per-cluster optimal convex combination (small LP)
# ---------------------------------------------------------------------------

def cluster_lp(C, members):
    """Solve  max t s.t. sum_h lam_h c_h(d) >= t c_i(d) for all members."""
    idx = list(members)
    m = len(idx)
    if m == 1:
        return {idx[0]: 1.0}, 1.0
    sub = C[idx]  # m x n
    n = C.shape[1]

    if HAS_SCIPY and m * n > 0:
        # variables: lam_1..lam_m, t ; minimize -t
        lam_coef = -np.broadcast_to(sub.T[None, :, :], (m, n, m)).reshape(m * n, m)
        t_coef = sub.reshape(-1)
        A_ub = np.concatenate([lam_coef, t_coef.reshape(-1, 1)], axis=1)
        b_ub = np.zeros(m * n)
        A_eq = np.ones((1, m + 1))
        A_eq[0, m] = 0.0
        b_eq = np.array([1.0])
        c_obj = np.zeros(m + 1)
        c_obj[-1] = -1.0
        bounds = [(0.0, 1.0)] * m + [(0.0, 1.0)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")
            if res.status == 0:
                lam = res.x[:m]
                t = float(res.x[m])
                wts = {idx[h]: float(lam[h]) for h in range(m) if lam[h] > 1e-9}
                if not wts:
                    h = int(np.argmax(lam))
                    wts = {idx[h]: 1.0}
                # renormalize exactly
                tot = sum(wts.values())
                wts = {i: w / tot for i, w in wts.items()}
                return wts, t
        except Exception:
            pass

    # Fallback: uniform weights, exact recomputed t.
    a = sub.mean(axis=0)
    ratios = (a[None, :] / sub).min()
    wts = {i: 1.0 / m for i in idx}
    return wts, float(ratios)


def aggregate_of(C, lam):
    a = np.zeros(C.shape[1])
    for i, w in lam.items():
        a += w * C[i]
    return a


def one_stage_heuristic(C, R, reps, K, iters):
    """Iterative reassignment + per-cluster LP for aggregates."""
    N = C.shape[0]
    assign = fix_empty(np.argmax(R[:, reps], axis=1), K)

    best_assign = assign.copy()
    best_t = -1.0

    aggs = np.zeros((K, C.shape[1]))
    for _ in range(max(iters, 0)):
        lams_tmp = []
        for j in range(K):
            members = np.where(assign == j)[0]
            lam, _ = cluster_lp(C, members)
            lams_tmp.append(lam)
            aggs[j] = aggregate_of(C, lam)
        R2 = np.empty((N, K))
        for j in range(K):
            R2[:, j] = (aggs[j] / C).min(axis=1)
        new_assign = fix_empty(np.argmax(R2, axis=1), K, R2)
        t = float(R2[np.arange(N), new_assign].min())
        if t > best_t:
            best_t = t
            best_assign = new_assign.copy()
        assign = new_assign

    # Final, fully consistent lambda on the best assignment found.
    lams = {}
    aggs = np.zeros((K, C.shape[1]))
    for j in range(K):
        members = np.where(best_assign == j)[0]
        if len(members) == 0:  # safety
            best_assign = fix_empty(best_assign, K)
            members = np.where(best_assign == j)[0]
        lam, _ = cluster_lp(C, members)
        lams[j] = lam
        aggs[j] = aggregate_of(C, lam)

    # exact scaling factor of the witness
    t = np.inf
    for i in range(N):
        r = float((aggs[best_assign[i]] / C[i]).min())
        if r < t:
            t = r
    return {int(i): int(best_assign[i]) for i in range(N)}, lams, float(t)


# ---------------------------------------------------------------------------
# Gurobi models
# ---------------------------------------------------------------------------

def set_common_params(m, time_limit):
    p = m.Params
    p.Seed = 0
    p.MIPGap = 1e-4
    p.NumericFocus = 0
    p.Threads = 1
    p.OutputFlag = 0
    if time_limit is not None:
        p.TimeLimit = max(time_limit, 0.1)


def solve_two_stage(C, R, K, warm_S, t_lb, time_limit):
    N = C.shape[0]
    P = []
    thr = max(t_lb - 1e-9, 0.0)
    for i in range(N):
        Pi = [s for s in range(N) if R[i, s] >= thr]
        if not Pi:
            Pi = [int(np.argmax(R[i]))]
        P.append(Pi)
    npairs = sum(len(p) for p in P)

    m = gp.Model("two_stage")
    set_common_params(m, time_limit)
    x = m.addVars(N, vtype=GRB.BINARY, name="x")
    mu = m.addVars([(i, s) for i in range(N) for s in P[i]],
                   vtype=GRB.BINARY, name="mu")
    tvar = m.addVar(lb=0.0, ub=1.0, name="t")

    m.addConstr(gp.quicksum(x[s] for s in range(N)) == K)
    for i in range(N):
        m.addConstr(gp.quicksum(mu[i, s] for s in P[i]) == 1)
        m.addConstr(tvar <= gp.quicksum(R[i, s] * mu[i, s] for s in P[i]))
    for (i, s) in mu.keys():
        m.addConstr(mu[i, s] <= x[s])

    warm_assign = np.argmax(R[:, warm_S], axis=1)
    for s in range(N):
        x[s].Start = 1.0 if s in warm_S else 0.0
    for i in range(N):
        chosen = warm_S[int(warm_assign[i])]
        for s in P[i]:
            mu[i, s].Start = 1.0 if s == chosen else 0.0
    tvar.Start = min(max(t_lb, 0.0), 1.0)

    m.setObjective(tvar, GRB.MAXIMIZE)
    m.optimize()

    status = int(m.Status)
    if m.SolCount > 0:
        S = [s for s in range(N) if x[s].X > 0.5][:K]
        if len(S) < K:
            rest = [s for s in np.argsort(-R.min(axis=0)) if s not in S]
            S = (S + rest)[:K]
        tv = min(1.0, max(0.0, float(tvar.X)))
    else:
        S = list(warm_S)[:K]
        tv = t_lb
    try:
        bound = float(m.ObjBound)
        if not math.isfinite(bound):
            bound = 1.0
    except Exception:
        bound = 1.0
    obj = tv if m.SolCount > 0 else t_lb
    gap = abs(bound - obj) / max(abs(obj), 1e-9) if m.SolCount > 0 else 1.0
    return S, tv, status, bound, float(min(gap, 1e9))


def solve_one_stage(C, K, warm_assign, warm_lams, t_lb, time_limit):
    N, n = C.shape
    if t_lb >= 1.0 - 1e-12:
        return warm_assign, warm_lams, t_lb, -1, 1.0, 0.0

    m = gp.Model("one_stage")
    set_common_params(m, time_limit)
    mu = m.addVars(N, K, vtype=GRB.BINARY, name="mu")
    lam = m.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="lam")
    tvar = m.addVar(lb=0.0, ub=1.0, name="t")

    for i in range(N):
        m.addConstr(gp.quicksum(mu[i, j] for j in range(K)) == 1)
    for j in range(K):
        m.addConstr(gp.quicksum(mu[i, j] for i in range(N)) >= 1)
        m.addConstr(gp.quicksum(lam[j, i] for i in range(N)) == 1)
    for j in range(K):
        for i in range(N):
            m.addConstr(lam[j, i] <= mu[i, j])
    # ratio constraints
    for i in range(N):
        ci = C[i]
        for j in range(K):
            for d in range(n):
                expr = (gp.quicksum(C[h, d] * lam[j, h] for h in range(N))
                        - ci[d] * tvar - ci[d] * mu[i, j])
                m.addConstr(expr >= -ci[d])

    for i in range(N):
        for j in range(K):
            mu[i, j].Start = 1.0 if warm_assign.get(i, 0) == j else 0.0
    for j in range(K):
        for i in range(N):
            w = warm_lams.get(j, {}).get(i, 0.0)
            if w > 0:
                lam[j, i].Start = w
    tvar.Start = min(max(t_lb, 0.0), 1.0)

    m.setObjective(tvar, GRB.MAXIMIZE)
    m.optimize()

    status = int(m.Status)
    if m.SolCount > 0:
        # parse assignment
        assign = {}
        for i in range(N):
            bestj = int(np.argmax([mu[i, j].X for j in range(K)]))
            assign[i] = bestj
        # parse lambda, restricted to assigned support, renormalized
        lams = {}
        aggs = np.zeros((K, n))
        for j in range(K):
            wts = {}
            tot = 0.0
            for i in range(N):
                v = lam[j, i].X
                if v > 1e-7 and (assign.get(i) == j):
                    wts[i] = float(v)
                    tot += v
            if tot <= 1e-9:
                members = [i for i in range(N) if assign.get(i) == j]
                if members:
                    wts = {members[0]: 1.0}
                    tot = 1.0
                else:
                    wts = {0: 1.0}
                    tot = 1.0
            wts = {i: w / tot for i, w in wts.items()}
            lams[j] = wts
            aggs[j] = aggregate_of(C, wts)
        t = np.inf
        for i in range(N):
            r = float((aggs[assign[i]] / C[i]).min())
            if r < t:
                t = r
        bound = float(m.ObjBound) if math.isfinite(m.ObjBound) else 1.0
        val = float(tvar.X)
        gap = abs(bound - val) / max(abs(val), 1e-9)
        return assign, lams, float(t), status, bound, float(gap)
    return warm_assign, warm_lams, t_lb, status, 1.0, 1.0


# ---------------------------------------------------------------------------
# Helpers for exact evaluation of witnesses
# ---------------------------------------------------------------------------

def exact_two_stage_t(R, S):
    cov = np.max(R[:, S], axis=1)
    return float(cov.min())


def make_one_entry(assign, lams, t, status, bound, gap):
    Kagg = len(lams)
    lw = {str(j): {str(i): round(w, 10) for i, w in lams[j].items()}
          for j in range(Kagg)}
    return {
        "t": float(t),
        "approximation_ratio": float(1.0 / max(t, 1e-9)),
        "assignment": {str(i): int(a) for i, a in assign.items()},
        "lambda_weights": lw,
        "solver_status": int(status),
        "best_bound": float(bound),
        "mip_gap": float(max(gap, 0.0)),
    }


def make_two_entry(S, t, status=None):
    ent = {
        "t": float(t),
        "approximation_ratio": float(1.0 / max(t, 1e-9)),
        "selected_scenarios": [int(s) for s in S],
    }
    return ent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    reserve = max(0.5, 0.02 * args.time_limit)
    deadline = start + args.time_limit - reserve

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="maximize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    N = int(data["num_scenarios_N"])
    n = int(data["problem_dimension_n"])
    C = np.asarray(data["scenarios"], dtype=float)
    Ks = sorted({int(k) for k in data.get("K_values", []) if 1 <= int(k)})

    if N == 0 or not Ks:
        sol = {"objective_value": 1.0, "one_stage_IP_mu": {}, "two_stage_IP": {}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    R = pair_ratio_matrix(C)
    maxK = min(max(Ks), N)
    prefix = greedy_prefix(R, maxK)

    one_results = {}
    two_results = {}
    best_obj = -float("inf")

    def assemble():
        return {
            "objective_value": float(best_obj),
            "one_stage_IP_mu": one_results,
            "two_stage_IP": two_results,
        }

    # ---------------- heuristic pass over all K ----------------
    for K in Ks:
        Kc = min(K, N)
        reps, t2 = prefix[Kc]
        two_results[str(K)] = make_two_entry(reps, t2)
        best_obj = max(best_obj, t2)

        if Kc >= N:
            assign = {i: i for i in range(N)}
            lams = {i: {i: 1.0} for i in range(N)}
            t1 = 1.0
        else:
            remaining = deadline - time.time()
            iters = 3 if (Kc * N <= 4000 and remaining > 2.0) else \
                    (1 if remaining > 0.5 else 0)
            if iters <= 0:
                assign_arr = fix_empty(np.argmax(R[:, reps], axis=1), Kc)
                assign = {int(i): int(assign_arr[i]) for i in range(N)}
                lams = {}
                for j in range(Kc):
                    members = [i for i in range(N) if assign[i] == j]
                    lams[j] = {members[0]: 1.0}
                aggs = np.zeros((Kc, n))
                for j in range(Kc):
                    aggs[j] = aggregate_of(C, lams[j])
                t1 = min(float((aggs[assign[i]] / C[i]).min()) for i in range(N))
            else:
                assign, lams, t1 = one_stage_heuristic(C, R, reps, Kc, iters)

        one_results[str(K)] = make_one_entry(assign, lams, t1, -1, t1, 1.0)
        best_obj = max(best_obj, t1)

    if logger:
        try:
            logger.log_solution(float(best_obj), assemble())
        except Exception:
            pass

    # ---------------- MIP refinement pass ----------------
    total = len(Ks)
    for idx, K in enumerate(Ks):
        Kc = min(K, N)
        key = str(K)
        models_left = 2 * (total - idx)
        reps, t2h = prefix[Kc]

        # warm-start data from current stored results
        warm_assign = {int(i): int(a) for i, a in
                       one_results[key]["assignment"].items()}
        warm_lams = {int(j): {int(i): float(w) for i, w in d.items()}
                     for j, d in one_results[key]["lambda_weights"].items()}
        t1h = one_results[key]["t"]

        # ---- two-stage MIP ----
        remaining = deadline - time.time()
        share = remaining / models_left
        if HAS_GP and remaining > 1.0 and Kc < N:
            try:
                S, tv, status, bound, gap = solve_two_stage(
                    C, R, Kc, reps, t2h, share)
                t_exact = exact_two_stage_t(R, S)
                t_exact = min(t_exact, 1.0)
                if t_exact >= two_results[key]["t"] - 1e-9:
                    two_results[key] = make_two_entry(S, t_exact)
                if two_results[key]["t"] > best_obj + 1e-9:
                    best_obj = two_results[key]["t"]
                    if logger:
                        try:
                            logger.log_solution(float(best_obj), assemble())
                        except Exception:
                            pass
            except Exception:
                pass

        # ---- one-stage MIP ----
        models_left -= 1
        remaining = deadline - time.time()
        share = remaining / max(models_left, 1)
        skip = (N * Kc * n > 600000) or (Kc >= N)
        if HAS_GP and remaining > 1.0 and not skip:
            try:
                assign, lams, tv, status, bound, gap = solve_one_stage(
                    C, Kc, warm_assign, warm_lams, t1h, share)
                if tv >= one_results[key]["t"] - 1e-9 and status != -1:
                    ent = make_one_entry(assign, lams, tv, status, bound, gap)
                    ent["solver_status"] = status
                    ent["best_bound"] = bound
                    ent["mip_gap"] = gap
                    one_results[key] = ent
                if one_results[key]["t"] > best_obj + 1e-9:
                    best_obj = one_results[key]["t"]
                if best_obj > 0 and logger:
                    try:
                        logger.log_solution(float(best_obj), assemble())
                    except Exception:
                        pass
            except Exception:
                pass

    best_obj = max(
        [best_obj]
        + [e["t"] for e in one_results.values()]
        + [e["t"] for e in two_results.values()]
    )

    sol = {
        "objective_value": float(best_obj),
        "one_stage_IP_mu": one_results,
        "two_stage_IP": two_results,
    }
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()