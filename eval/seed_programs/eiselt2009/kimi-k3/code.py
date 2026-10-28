import argparse
import json
import math
import os
import time
from itertools import combinations

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = max(0, int(args.time_limit))
    deadline = start_time + time_limit
    reserve = max(0.05, min(1.0, 0.02 * max(1, time_limit)))
    work_deadline = max(start_time, deadline - reserve)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n_nodes"])
    p_req = int(data["p"])
    p = max(0, min(p_req, n))

    nodes = data.get("nodes", [])
    waiting = np.zeros(n, dtype=float)
    for node in nodes:
        i = int(node["id"])
        if 0 <= i < n:
            waiting[i] = float(node.get("waiting_time", 0.0))

    W = np.asarray(data["demand_matrix"], dtype=float).reshape(n, n)
    if "distance_matrix" in data:
        D = np.asarray(data["distance_matrix"], dtype=float).reshape(n, n)
    else:
        D = np.zeros((n, n), dtype=float)
        for a in nodes:
            i = int(a["id"])
            for b in nodes:
                j = int(b["id"])
                if 0 <= i < n and 0 <= j < n:
                    dx = float(a["x"]) - float(b["x"])
                    dy = float(a["y"]) - float(b["y"])
                    D[i, j] = math.hypot(dx, dy)

    params = data.get("parameters", {})
    alpha = float(params.get("alpha", 1.0))
    gamma = float(params.get("gamma", 0.5))
    beta = float(params.get("beta", 1.0))
    lam = float(params.get("lambda", 1.0))
    cost_scalar = float(params.get("cost_scalar", 1.0))
    attr_2 = float(params.get("attractiveness_2hub", 1.0))
    attr_1 = float(params.get("attractiveness_1hub", 1.0))

    comp_hubs = []
    for h in data.get("competitor_hub_locations", []):
        h = int(h)
        if 0 <= h < n and h not in comp_hubs:
            comp_hubs.append(h)

    def make_solution(hubs, obj):
        return {
            "objective_value": float(obj),
            "hub_locations": [int(x) for x in sorted(hubs)],
        }

    if n == 0 or p == 0:
        sol = make_solution([], 0.0)
        if logger:
            logger.log_solution(sol["objective_value"], sol)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(sol, f)
        return

    def pow_exp(arr, exp):
        if exp == 0.0:
            return np.ones_like(arr, dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out = np.power(np.maximum(arr, 1e-15), exp)
        return out

    def denom_tc(T, C):
        if gamma == 1.0:
            den = pow_exp(T, beta)
        elif gamma == 0.0:
            den = pow_exp(C, lam)
        else:
            den = gamma * pow_exp(T, beta) + (1.0 - gamma) * pow_exp(C, lam)
        den = np.where(np.isfinite(den), den, np.inf)
        return np.maximum(den, 1e-12)

    def util_tc(T, C, attr):
        return attr / denom_tc(T, C)

    def single_matrix(k):
        e = D[:, k][:, None] + D[k, :][None, :]
        T = waiting[:, None] + 2.0 * waiting[k] + e
        C = cost_scalar * e
        out = util_tc(T, C, attr_1)
        return np.nan_to_num(out, nan=0.0, posinf=1e100, neginf=0.0)

    def pair_matrix(a, b):
        # For fixed i,j and unordered hubs {a,b}, keep the better of i-a-b-j and i-b-a-j.
        A = D[:, a][:, None] + D[b, :][None, :]
        E = np.minimum(A, A.T)
        T = waiting[:, None] + waiting[a] + waiting[b] + D[a, b] + E
        C = cost_scalar * (alpha * D[a, b] + E)
        out = util_tc(T, C, attr_2)
        return np.nan_to_num(out, nan=0.0, posinf=1e100, neginf=0.0)

    def pair_u(a, b):
        if a == b:
            return single_matrix(a)
        if a < b:
            return pair_matrix(a, b)
        return pair_matrix(b, a)

    def frac_from_s(s, b):
        den = s + b
        out = np.zeros_like(s, dtype=float)
        mask = den > 1e-12
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out[mask] = s[mask] / den[mask]
        return np.clip(out, 0.0, 1.0)

    def utility_sum_for_set(S):
        s = np.zeros((n, n), dtype=float)
        S = list(S)
        for k in S:
            s += single_matrix(k)
        for a, b in combinations(S, 2):
            s += pair_u(a, b)
        return s

    def objective_from_s(s):
        return float(np.sum(W * frac_from_s(s, B)))

    def objective_for_set(S):
        s = utility_sum_for_set(S)
        return s, objective_from_s(s)

    B = utility_sum_for_set(comp_hubs)

    total_demand = float(np.sum(W))
    if total_demand <= 0.0:
        hubs = list(range(p))
        sol = make_solution(hubs, 0.0)
        if logger:
            logger.log_solution(sol["objective_value"], sol)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(sol, f)
        return

    demanded_od = np.argwhere(W > 0.0)

    def compute_single_scores(sample_limit=4000):
        if len(demanded_od) == 0:
            return np.zeros(n, dtype=float)
        q = len(demanded_od)
        if q * n <= 50_000_000:
            sel = demanded_od
        else:
            vals = W[demanded_od[:, 0], demanded_od[:, 1]]
            m = min(q, sample_limit)
            kth = q - m
            idx = np.argpartition(vals, kth)[kth:]
            sel = demanded_od[idx]
        scores = np.zeros(n, dtype=float)
        for i, j in sel:
            e = D[i, :] + D[:, j]
            u = util_tc(waiting[i] + 2.0 * waiting + e, cost_scalar * e, attr_1)
            u = np.nan_to_num(u, nan=0.0, posinf=1e100, neginf=0.0)
            b = B[i, j]
            den = u + b
            fr = np.zeros(n, dtype=float)
            mask = den > 1e-12
            fr[mask] = u[mask] / den[mask]
            scores += W[i, j] * np.clip(fr, 0.0, 1.0)
        return scores

    single_score = compute_single_scores()

    def choose_candidates():
        budget = 150_000_000.0
        est = (float(n) ** 3) * p * (p + 1) / 2.0
        if est <= budget or n <= 60:
            return list(range(n))
        den = (float(n) ** 2) * p * (p + 1) / 2.0
        keep = int(math.ceil(budget / max(1.0, den)))
        keep = max(p, min(n, max(2 * p + 5, keep)))
        if keep >= n:
            return list(range(n))
        order = np.argsort(-single_score, kind="mergesort")[:keep]
        return sorted(int(x) for x in order)

    candidate_nodes = choose_candidates()

    best_obj = -1e300
    best_S = []
    best_s = None

    def improve_tol(obj):
        return 1e-9 * max(1.0, abs(obj))

    def update_best(S, obj, s=None):
        nonlocal best_obj, best_S, best_s
        S = sorted(int(x) for x in S)
        if len(S) != p:
            return
        if obj > best_obj + improve_tol(best_obj):
            best_obj = float(obj)
            best_S = S
            best_s = s
            if logger:
                logger.log_solution(float(obj), make_solution(S, obj))

    def fill_to_p(S):
        S = list(dict.fromkeys(int(x) for x in S))
        if len(S) >= p:
            return sorted(S[:p])
        inS = np.zeros(n, dtype=bool)
        inS[S] = True
        for v in np.argsort(-single_score, kind="mergesort"):
            v = int(v)
            if not inS[v]:
                S.append(v)
                inS[v] = True
                if len(S) == p:
                    break
        return sorted(S)

    def greedy(cands, local_deadline):
        S = []
        inS = np.zeros(n, dtype=bool)
        s = np.zeros((n, n), dtype=float)
        cands = [int(x) for x in cands]
        timed_out = False
        for _ in range(p):
            if time.time() > local_deadline:
                timed_out = True
                break
            frac_s = frac_from_s(s, B)
            best_gain = -1e300
            best_v = None
            best_delta = None
            for ci, v in enumerate(cands):
                if inS[v]:
                    continue
                if (ci & 15) == 0 and time.time() > local_deadline:
                    timed_out = True
                    break
                delta = single_matrix(v)
                for k in S:
                    delta += pair_u(v, k)
                gain = float(np.sum(W * (frac_from_s(s + delta, B) - frac_s)))
                if gain > best_gain + 1e-12 or (
                    abs(gain - best_gain) <= 1e-12
                    and best_v is not None
                    and single_score[v] > single_score[best_v]
                ):
                    best_gain = gain
                    best_v = v
                    best_delta = delta
            if best_v is not None:
                S.append(best_v)
                inS[best_v] = True
                s += best_delta
            if timed_out:
                break
        if len(S) < p:
            S = fill_to_p(S)
            s, obj = objective_for_set(S)
        else:
            obj = objective_from_s(s)
        return S, s, obj

    if time.time() <= work_deadline:
        S0, s0, obj0 = greedy(candidate_nodes, work_deadline)
    else:
        S0 = fill_to_p([])
        s0, obj0 = objective_for_set(S0)
    update_best(S0, obj0, s0)

    pair_rows, pair_cols = np.triu_indices(n, k=1)
    m = len(pair_rows)
    coeff_est = int(len(demanded_od)) * int(n + m)
    use_exact = (
        gp is not None
        and n <= 60
        and coeff_est <= 4_000_000
        and time.time() < work_deadline - 0.05
    )

    def solve_exact(initial_S, initial_obj):
        eps_b = 1e-12
        rows = []
        d_vec = []
        b_vec = []
        const = 0.0
        for i, j in demanded_od:
            i = int(i)
            j = int(j)
            dem = float(W[i, j])
            b = float(B[i, j])
            if b <= eps_b:
                const += dem
            else:
                rows.append((i, j))
                d_vec.append(dem)
                b_vec.append(b)
        q = len(rows)
        if q == 0:
            return const, initial_S

        single_mat = np.empty((q, n), dtype=float)
        pair_mat = np.empty((q, m), dtype=float)
        d_vec = np.asarray(d_vec, dtype=float)
        b_vec = np.asarray(b_vec, dtype=float)

        for r, (i, j) in enumerate(rows):
            if (r & 63) == 0 and time.time() > work_deadline:
                return None
            e = D[i, :] + D[:, j]
            T1 = waiting[i] + 2.0 * waiting + e
            C1 = cost_scalar * e
            u1 = util_tc(T1, C1, attr_1)
            single_mat[r] = np.nan_to_num(u1, nan=0.0, posinf=1e100, neginf=0.0)

            A = D[i, :][:, None] + D[:, j][None, :]
            E = np.minimum(A, A.T)
            T = waiting[i] + waiting[:, None] + waiting[None, :] + D + E
            C = cost_scalar * (alpha * D + E)
            U = util_tc(T, C, attr_2)
            pair_mat[r] = np.nan_to_num(U[pair_rows, pair_cols], nan=0.0, posinf=1e100, neginf=0.0)

        try:
            model = gp.Model()
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.LazyConstraints = 1
            remaining = float(work_deadline - time.time())
            if remaining <= 0.05:
                return None
            model.Params.TimeLimit = remaining

            x_td = model.addVars(n, vtype=GRB.BINARY)
            y_td = model.addVars(m, vtype=GRB.BINARY)
            s_td = model.addVars(q, lb=0.0)
            z_td = model.addVars(q, lb=0.0, ub=1.0)
            x = [x_td[i] for i in range(n)]
            y = [y_td[i] for i in range(m)]
            s_vars = [s_td[i] for i in range(q)]
            z_vars = [z_td[i] for i in range(q)]
            pairs = list(zip(pair_rows.tolist(), pair_cols.tolist()))

            model.addConstr(gp.quicksum(x) == p)
            for idx, (a, b) in enumerate(pairs):
                model.addConstr(y[idx] <= x[a])
                model.addConstr(y[idx] <= x[b])
                model.addConstr(y[idx] >= x[a] + x[b] - 1.0)

            all_vars = x + y
            for r in range(q):
                row = np.concatenate([single_mat[r], pair_mat[r]])
                model.addConstr(s_vars[r] == gp.LinExpr(row, all_vars))

            tangent_fracs = (0.0, 0.25, 0.5, 0.75, 1.0)
            for r in range(q):
                b = float(b_vec[r])
                smax = float(np.sum(single_mat[r]) + np.sum(pair_mat[r]))
                if smax <= 1e-12:
                    model.addConstr(z_vars[r] <= 0.0)
                    continue
                for t in tangent_fracs:
                    s0 = t * smax
                    den = s0 + b
                    h0 = s0 / den
                    hp = b / (den * den)
                    if np.isfinite(h0) and np.isfinite(hp):
                        model.addConstr(z_vars[r] <= h0 + hp * (s_vars[r] - s0))

            model.setObjective(const + gp.quicksum(d_vec[r] * z_vars[r] for r in range(q)), GRB.MAXIMIZE)

            # MIP start from the heuristic solution.
            start_mask = np.zeros(n, dtype=float)
            start_mask[initial_S] = 1.0
            for i in range(n):
                x[i].Start = start_mask[i]
            for idx, (a, b) in enumerate(pairs):
                y[idx].Start = 1.0 if start_mask[a] > 0.5 and start_mask[b] > 0.5 else 0.0
            ystart = (start_mask[pair_rows] * start_mask[pair_cols]).astype(float)
            sstart = single_mat.dot(start_mask) + pair_mat.dot(ystart)
            zstart = frac_from_s(sstart, b_vec)
            for r in range(q):
                s_vars[r].Start = float(sstart[r])
                z_vars[r].Start = float(zstart[r])

            best = {"obj": float(initial_obj), "S": list(initial_S)}

            def actual_from_xv(xv):
                sel = [i for i, v in enumerate(xv) if v > 0.5]
                if len(sel) > p:
                    order = sorted(sel, key=lambda i: xv[i], reverse=True)[:p]
                    sel = sorted(order)
                elif len(sel) < p:
                    inS = np.zeros(n, dtype=bool)
                    inS[sel] = True
                    for v in np.argsort(-single_score, kind="mergesort"):
                        v = int(v)
                        if not inS[v]:
                            sel.append(v)
                            inS[v] = True
                            if len(sel) == p:
                                break
                    sel = sorted(sel)
                mask = np.zeros(n, dtype=float)
                mask[sel] = 1.0
                yv = (mask[pair_rows] * mask[pair_cols]).astype(float)
                sv = single_mat.dot(mask) + pair_mat.dot(yv)
                hv = frac_from_s(sv, b_vec)
                obj = const + float(np.dot(d_vec, hv))
                return sel, sv, hv, obj

            def callback(mdl, where):
                if where != GRB.Callback.MIPSOL:
                    return
                xv = np.array([mdl.cbGetSolution(x[i]) for i in range(n)], dtype=float)
                zv = np.array([mdl.cbGetSolution(z_vars[r]) for r in range(q)], dtype=float)
                sel, sv, hv, obj = actual_from_xv(xv)
                den = sv + b_vec
                hp = np.zeros(q, dtype=float)
                mask = den > 1e-12
                hp[mask] = b_vec[mask] / (den[mask] * den[mask])
                viol = zv - hv
                tol = 1e-6 + 1e-8 * np.maximum(1.0, hv)
                bad = np.nonzero(viol > tol)[0]
                for r in bad:
                    mdl.cbLazy(z_vars[r] <= hv[r] + hp[r] * (s_vars[r] - sv[r]))
                if obj > best["obj"] + improve_tol(best["obj"]):
                    best["obj"] = float(obj)
                    best["S"] = list(sel)
                    if logger:
                        logger.log_solution(float(obj), make_solution(sel, obj))

            try:
                model.optimize(callback)
            except Exception:
                return best["obj"], best["S"]

            if model.SolCount > 0:
                try:
                    xv = np.array([x[i].X for i in range(n)], dtype=float)
                    sel, sv, hv, obj = actual_from_xv(xv)
                    if obj > best["obj"] + improve_tol(best["obj"]):
                        best["obj"] = float(obj)
                        best["S"] = list(sel)
                        if logger:
                            logger.log_solution(float(obj), make_solution(sel, obj))
                except Exception:
                    pass
            return best["obj"], best["S"]
        except Exception:
            return None

    exact_result = None
    if use_exact:
        exact_result = solve_exact(best_S, best_obj)
        if exact_result is not None and exact_result[0] > best_obj + improve_tol(best_obj):
            best_obj, best_S = float(exact_result[0]), list(exact_result[1])
            best_s = None

    def local_search(S, s, obj, allowed, local_deadline):
        if p == 0 or len(S) != p:
            return S, s, obj
        rng = np.random.default_rng(0)
        inS = np.zeros(n, dtype=bool)
        inS[S] = True
        allowed = sorted(set(int(x) for x in allowed) | set(S))
        if len(allowed) <= p:
            return S, s, obj
        S = list(S)
        while time.time() < local_deadline:
            a = S[int(rng.integers(len(S)))]
            b = int(allowed[int(rng.integers(len(allowed)))])
            if b == a or inS[b]:
                continue
            Swo = [v for v in S if v != a]
            contrib_a = single_matrix(a)
            for k in Swo:
                contrib_a += pair_u(a, k)
            s_wo = s - contrib_a
            contrib_b = single_matrix(b)
            for k in Swo:
                contrib_b += pair_u(b, k)
            s_new = s_wo + contrib_b
            obj_new = objective_from_s(s_new)
            if obj_new > obj + improve_tol(obj):
                pos = S.index(a)
                S[pos] = b
                inS[a] = False
                inS[b] = True
                s = s_new
                obj = obj_new
                update_best(S, obj, s)
        return S, s, obj

    if (not use_exact or exact_result is None) and time.time() < work_deadline:
        if best_s is None:
            best_s, best_obj = objective_for_set(best_S)
        local_search(best_S, best_s, best_obj, candidate_nodes, work_deadline)

    if len(best_S) != p:
        best_S = fill_to_p(best_S)
        best_s, best_obj = objective_for_set(best_S)
        update_best(best_S, best_obj, best_s)

    sol = make_solution(best_S, best_obj)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()