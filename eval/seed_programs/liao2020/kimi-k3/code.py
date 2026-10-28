import argparse
import json
import math
import time

import numpy as np

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))
    soft_limit = time_limit * 0.88

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    P = int(inst["num_patients_P"])
    L = float(inst["provider_service_hours_L_minutes"])

    cost_structs = inst["cost_structures"]
    cost_name = "Cost1" if "Cost1" in cost_structs else next(iter(cost_structs))
    cs = cost_structs[cost_name]
    c_w = float(cs["c_w"])
    c_g = float(cs["c_g"])
    c_o = float(cs["c_o"])

    pats = inst["patients"]
    pid = [int(p.get("patient_index", k + 1)) for k, p in enumerate(pats)]
    muA = np.array([float(p["mean_duration_adequate_prep"]) for p in pats], dtype=float)
    muI = np.array([float(p["mean_duration_inadequate_prep"]) for p in pats], dtype=float)
    dAL = np.array([float(p["lower_bound_duration_adequate_prep"]) for p in pats], dtype=float)
    dAU = np.array([float(p["upper_bound_duration_adequate_prep"]) for p in pats], dtype=float)
    dIL = np.array([float(p["lower_bound_duration_inadequate_prep"]) for p in pats], dtype=float)
    dIU = np.array([float(p["upper_bound_duration_inadequate_prep"]) for p in pats], dtype=float)
    qpr = np.array([float(p["mean_prep_adequacy"]) for p in pats], dtype=float)
    muu = np.array([float(p["mean_arrival_time_deviation"]) for p in pats], dtype=float)
    uL = np.array([float(p["lower_bound_arrival_time_deviation"]) for p in pats], dtype=float)
    uU = np.array([float(p["upper_bound_arrival_time_deviation"]) for p in pats], dtype=float)

    mean_dur = qpr * muA + (1.0 - qpr) * muI
    risk = qpr * np.maximum(dAU - dAL, 0.0) + (1.0 - qpr) * np.maximum(dIU - dIL, 0.0) \
        + np.maximum(uU - uL, 0.0)

    # ------------------------------------------------------------------
    # Worst-case scenario grids.
    #
    # The realized cost is convex in each patient's random parameters, so
    # the worst-case marginals place mass on the support endpoints while
    # matching the given means; the worst-case coupling is comonotone
    # (a single uniform drives all randomness).  The expectation therefore
    # reduces to a one-dimensional integral, which we approximate on a
    # uniform midpoint grid.
    # ------------------------------------------------------------------
    def two_point(U, lo, hi, mu):
        if hi <= lo:
            return np.full(U.shape, lo, dtype=float)
        mu_c = min(max(mu, lo), hi)
        pi = (mu_c - lo) / (hi - lo)
        return np.where(U > 1.0 - pi, hi, lo)

    def build_scenarios(S, reverse_u=False):
        U = (np.arange(S, dtype=float) + 0.5) / S
        Uu = 1.0 - U if reverse_u else U
        dur = np.empty((P, S), dtype=float)
        arr = np.empty((P, S), dtype=float)
        for p in range(P):
            qq = min(max(qpr[p], 0.0), 1.0)
            if qq <= 0.0:
                dur[p] = two_point(U, dIL[p], dIU[p], muI[p])
            elif qq >= 1.0:
                dur[p] = two_point(U, dAL[p], dAU[p], muA[p])
            else:
                d = np.empty(S, dtype=float)
                mask = U <= qq
                d[mask] = two_point(U[mask] / qq, dAL[p], dAU[p], muA[p])
                d[~mask] = two_point((U[~mask] - qq) / (1.0 - qq), dIL[p], dIU[p], muI[p])
                dur[p] = d
            arr[p] = two_point(Uu, uL[p], uU[p], muu[p])
        return dur, arr

    if P <= 25:
        S_opt = 160
    elif P <= 50:
        S_opt = 110
    else:
        S_opt = 70
    S_final = int(max(500, min(4000, 150000 // max(P, 1))))

    dur, arr = build_scenarios(S_opt)

    # ------------------------------------------------------------------
    # Realized-cost recursion (per position i):
    #   ready_i = max(t_i + u_i, C_{i-1})
    #   A_i     = max(t_i, ready_i)        (actual procedure start)
    #   W_i     = A_i - t_i                (waiting past scheduled start)
    #   idle_i  = max(0, t_i - C_{i-1} - W_i) = max(0, t_i - C_{i-1} - u_i^+)
    #   C_i     = A_i + d_i
    #   overtime = max(0, C_P - L)
    # ------------------------------------------------------------------
    def advance(Cprev, cost, i, order, t, dur_, arr_):
        p = order[i]
        D = dur_[p]
        Ua = arr_[p]
        A = np.maximum(t[i], np.maximum(t[i] + Ua, Cprev))
        cost = cost + c_w * (A - t[i])
        if i >= 1:
            idle = t[i] - Cprev - np.maximum(Ua, 0.0)
            cost = cost + c_g * np.maximum(idle, 0.0)
        return A + D, cost

    def eval_full(order, t, dur_, arr_):
        S = dur_.shape[1]
        Cprev = np.zeros(S)
        cost = np.zeros(S)
        for i in range(P):
            Cprev, cost = advance(Cprev, cost, i, order, t, dur_, arr_)
        cost = cost + c_o * np.maximum(Cprev - L, 0.0)
        return float(cost.mean())

    def eval_from(order, t, dur_, arr_, start_i, Cprev0, cost0):
        Cp = Cprev0
        co = cost0
        for i in range(start_i, P):
            Cp, co = advance(Cp, co, i, order, t, dur_, arr_)
        return float((co + c_o * np.maximum(Cp - L, 0.0)).mean())

    GR = 0.6180339887498949

    def optimize_t(order, dur_, arr_, t0, sweeps=4, iters=22):
        """Coordinate descent with golden-section line search on start times."""
        t = np.array(t0, dtype=float).reshape(-1)
        t[0] = 0.0
        for i in range(1, P):
            if t[i] < t[i - 1]:
                t[i] = t[i - 1]
            if t[i] > L:
                t[i] = L
        for _ in range(sweeps):
            Cprev = np.zeros(dur_.shape[1])
            cost = np.zeros(dur_.shape[1])
            for i in range(P):
                if i >= 1:
                    lo = t[i - 1]
                    hi = L if i == P - 1 else min(t[i + 1], L)
                    if hi > lo + 1e-10:
                        t_old = t[i]

                        def g(x, i=i, Cp=Cprev, co=cost):
                            t[i] = x
                            return eval_from(order, t, dur_, arr_, i, Cp, co)

                        a, b = lo, hi
                        c1 = b - GR * (b - a)
                        c2 = a + GR * (b - a)
                        f1 = g(c1)
                        f2 = g(c2)
                        for _ in range(iters):
                            if f1 > f2:
                                a = c1
                                c1, f1 = c2, f2
                                c2 = a + GR * (b - a)
                                f2 = g(c2)
                            else:
                                b = c2
                                c2, f2 = c1, f1
                                c1 = b - GR * (b - a)
                                f1 = g(c1)
                        cands = [lo, hi, 0.5 * (a + b), t_old]
                        vals = [g(x) for x in cands]
                        t[i] = cands[int(np.argmin(vals))]
                Cprev, cost = advance(Cprev, cost, i, order, t, dur_, arr_)
        return t

    def initial_t(order):
        cum = np.concatenate(([0.0], np.cumsum(mean_dur[order])[:-1]))
        return np.minimum(cum, L)

    # ------------------------------------------------------------------
    # Incumbent management / logging
    # ------------------------------------------------------------------
    best = {"obj": math.inf, "order": None, "t": None}
    state = {"last_logged": math.inf}

    def make_solution(order, t, obj):
        assignment = {}
        patient_start = {}
        for i in range(P):
            p = pid[order[i]]
            assignment[str(p)] = i + 1
            patient_start[str(p)] = float(round(float(t[i]), 6))
        schedule = {str(i + 1): float(round(float(t[i]), 6)) for i in range(P)}
        mv = {}
        for i in range(P):
            mv["x[%d,%d]" % (pid[order[i]], i + 1)] = 1.0
        for i in range(P):
            mv["t[%d]" % (i + 1)] = float(round(float(t[i]), 6))
        return {
            "objective_value": float(obj),
            "model_variables": mv,
            "cost_structure": cost_name,
            "c_w": c_w,
            "c_g": c_g,
            "c_o": c_o,
            "assignment": assignment,
            "schedule": schedule,
            "patient_start_times": patient_start,
            "model_variable_name_patterns": {
                "x[p,i]": "patient-position assignment, p and i are 1..P",
                "t[i]": "position start time",
                "rho[i], alpha[i], lambda[i], gamma[i]": "dual variables",
                "beta[i]": "i is 1..P+2",
                "z[i], v[i], r[i], e[i]": "one-dimensional auxiliary variables",
                "eta[p,i], tau[p,i], nu[p,i], Gam[p,i]": "McCormick variables",
                "zeta[p,i], psi[p,i], phi[p,i]": "nonnegative McCormick variables",
            },
        }

    def time_ok():
        return (time.time() - t_start) < soft_limit

    def try_incumbent(order, t):
        obj = eval_full(order, t, dur, arr)
        if obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["order"] = [int(x) for x in order]
            best["t"] = np.array(t, dtype=float)
            if logger and obj < state["last_logged"] - 1e-9:
                state["last_logged"] = obj
                logger.log_solution(obj, make_solution(order, t, obj))
        return obj

    # ---------------- immediate feasible fallback ----------------
    base_order = [int(x) for x in np.argsort(mean_dur, kind="stable")]
    base_t = initial_t(base_order)
    try_incumbent(base_order, base_t)

    try:
        # ---------------- candidate sequences ----------------
        cand_orders = []

        def add_order(o):
            o = [int(x) for x in o]
            if o not in cand_orders:
                cand_orders.append(o)

        add_order(np.argsort(mean_dur, kind="stable"))
        add_order(np.argsort(-mean_dur, kind="stable"))
        add_order(np.argsort(mean_dur + 0.5 * risk, kind="stable"))
        add_order(np.argsort(risk, kind="stable"))
        add_order(range(P))

        for order in cand_orders:
            if not time_ok():
                break
            t = optimize_t(order, dur, arr, initial_t(order), sweeps=3, iters=18)
            try_incumbent(order, t)

        # ---------------- local search on the sequence ----------------
        order = best["order"]
        t = best["t"]
        cur_obj = best["obj"]
        improved = True
        n_pass = 0
        while improved and time_ok() and n_pass < 8:
            improved = False
            n_pass += 1
            # adjacent swaps
            for i in range(P - 1):
                if not time_ok():
                    break
                o2 = list(order)
                o2[i], o2[i + 1] = o2[i + 1], o2[i]
                t2 = optimize_t(o2, dur, arr, t, sweeps=2, iters=14)
                obj2 = eval_full(o2, t2, dur, arr)
                if obj2 < cur_obj - 1e-7:
                    t2 = optimize_t(o2, dur, arr, t2, sweeps=5, iters=26)
                    obj2 = eval_full(o2, t2, dur, arr)
                    if obj2 < cur_obj - 1e-9:
                        order, t, cur_obj = o2, t2, obj2
                        improved = True
                        try_incumbent(order, t)
            # reinsertion moves
            if time_ok() and P <= 60:
                for i in range(P):
                    if not time_ok():
                        break
                    for j in range(P):
                        if j == i:
                            continue
                        if not time_ok():
                            break
                        o2 = list(order)
                        x = o2.pop(i)
                        o2.insert(j, x)
                        t2 = optimize_t(o2, dur, arr, t, sweeps=1, iters=12)
                        obj2 = eval_full(o2, t2, dur, arr)
                        if obj2 < cur_obj - 1e-7:
                            t2 = optimize_t(o2, dur, arr, t2, sweeps=5, iters=26)
                            obj2 = eval_full(o2, t2, dur, arr)
                            if obj2 < cur_obj - 1e-9:
                                order, t, cur_obj = o2, t2, obj2
                                improved = True
                                try_incumbent(order, t)

        # ---------------- final polish of the incumbent schedule ----------------
        if time_ok() and best["order"] is not None:
            t = optimize_t(best["order"], dur, arr, best["t"], sweeps=8, iters=36)
            try_incumbent(best["order"], t)
    except Exception:
        pass

    if best["order"] is None:
        best["order"] = base_order
        best["t"] = base_t
        best["obj"] = eval_full(base_order, base_t, dur, arr)

    # ---------------- final objective evaluation (fine grids) ----------------
    order = best["order"]
    t = best["t"]
    durF, arrF = build_scenarios(S_final)
    obj1 = eval_full(order, t, durF, arrF)
    durR, arrR = build_scenarios(S_final, reverse_u=True)
    obj2 = eval_full(order, t, durR, arrR)
    final_obj = max(obj1, obj2)

    sol = make_solution(order, t, final_obj)
    if logger and final_obj < state["last_logged"] - 1e-9:
        logger.log_solution(final_obj, sol)

    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh, indent=2)


if __name__ == "__main__":
    main()