import argparse
import json
import math
import random
import time

import numpy as np

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(5, args.time_limit) - 1.5

    with open(args.instance_path) as fh:
        data = json.load(fh)

    n = int(data["n"])
    cp = data["cost_parameters"]
    chi = float(cp["collection_cost_chi"])
    alpha = float(cp["transfer_cost_alpha"])
    delta = float(cp["distribution_cost_delta"])
    d = np.asarray(data["distance_matrix"], dtype=float)
    w = np.asarray(data["flow_matrix"], dtype=float)
    f = np.asarray(data["setup_costs"], dtype=float)
    cap = np.asarray(data["capacities"], dtype=float)

    O = w.sum(axis=1)          # outgoing flow of each node
    Din = w.sum(axis=0)        # incoming flow of each node
    Dtot = float(O.sum())
    dT = np.ascontiguousarray(d.T)
    wT = np.ascontiguousarray(w.T)
    # Cas[i,k] = collection + distribution cost of assigning node i to hub k
    Cas = chi * O[:, None] * d + delta * Din[:, None] * dT
    idx = np.arange(n)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)
    np.random.seed(0)

    def full_cost(a):
        aa = np.asarray(a, dtype=np.int64)
        hubs = np.unique(aa)
        return float(f[hubs].sum() + Cas[idx, aa].sum()
                     + alpha * (w * d[np.ix_(aa, aa)]).sum())

    def sol_dict(a, obj):
        aa = [int(x) for x in a]
        return {"objective_value": float(obj),
                "hubs": sorted(set(aa)),
                "assignment": aa}

    def shift_delta(a, i, k, l):
        # move node i from hub k to hub l (a[i] currently == k)
        t = w[i] @ (d[l, a] - d[k, a]) + wT[i] @ (dT[l, a] - dT[k, a])
        t += w[i, i] * (d[l, l] + d[k, k] - d[l, k] - d[k, l])
        return Cas[i, l] - Cas[i, k] + alpha * t

    # ------------------- construction -------------------
    def construct():
        order = sorted(range(n), key=lambda k: (f[k] / max(cap[k], 1e-9), -cap[k]))
        hubs = []
        tc = 0.0
        for k in order:
            if cap[k] >= O[k] - 1e-9:
                hubs.append(k)
                tc += cap[k]
                if tc >= Dtot:
                    break
        if not hubs:
            hubs = [int(np.argmax(cap))]
        a = np.full(n, -1, dtype=np.int64)
        load = np.zeros(n)
        for k in hubs:
            a[k] = k
            load[k] = O[k]
        rest = sorted([i for i in range(n) if a[i] < 0], key=lambda i: -O[i])
        for i in rest:
            bestk = -1
            bc = math.inf
            for k in hubs:
                if load[k] + O[i] <= cap[k] + 1e-9 and Cas[i, k] < bc:
                    bc = Cas[i, k]
                    bestk = k
            if bestk < 0:
                cands = [m for m in range(n) if a[m] < 0 and
                         ((m == i and cap[m] >= O[i] - 1e-9) or
                          (m != i and cap[m] >= O[m] + O[i] - 1e-9))]
                if cands:
                    m2 = min(cands, key=lambda m: f[m] + Cas[i, m])
                    a[m2] = m2
                    load[m2] = O[m2]
                    hubs.append(m2)
                    if m2 == i:
                        continue
                    bestk = m2
                else:
                    bestk = max(hubs, key=lambda k: cap[k] - load[k])
            a[i] = bestk
            load[bestk] += O[i]
        return a, load

    # ------------------- local descent (best shift per node) -------------------
    def descent(a, load, end_time):
        improved = True
        cnt = 0
        while improved:
            improved = False
            for i in range(n):
                cnt += 1
                if (cnt & 31) == 0 and time.time() >= end_time:
                    return
                if a[i] == i:
                    continue
                k = int(a[i])
                hubs_arr = np.unique(a)
                if hubs_arr.size < 2:
                    return
                M1 = d[hubs_arr][:, a] @ w[i]
                M2 = dT[hubs_arr][:, a] @ wT[i]
                pk = int(np.searchsorted(hubs_arr, k))
                corr = w[i, i] * (d[hubs_arr, hubs_arr] + d[k, k]
                                  - d[hubs_arr, k] - d[k, hubs_arr])
                dv = (Cas[i, hubs_arr] - Cas[i, k]) + alpha * (M1 - M1[pk] + M2 - M2[pk] + corr)
                feas = load[hubs_arr] + O[i] <= cap[hubs_arr] + 1e-9
                dv = np.where(feas, dv, np.inf)
                dv[pk] = np.inf
                jb = int(np.argmin(dv))
                if dv[jb] < -1e-7:
                    l = int(hubs_arr[jb])
                    a[i] = l
                    load[k] -= O[i]
                    load[l] += O[i]
                    improved = True

    # ------------------- simulated annealing -------------------
    def anneal(a, load, cur, best, best_a, end_time):
        hubs = [k for k in range(n) if a[k] == k]
        samples = []
        for _ in range(300):
            if len(hubs) < 2:
                break
            i = rng.randrange(n)
            if a[i] == i:
                continue
            l = hubs[rng.randrange(len(hubs))]
            if l == a[i]:
                continue
            samples.append(abs(shift_delta(a, i, int(a[i]), l)))
        if samples:
            T0 = max(1e-8, 0.6 * sum(samples) / len(samples))
        else:
            T0 = max(1e-8, 0.001 * abs(cur) + 1.0)
        T = T0
        it = 0

        def acc(dlt):
            if dlt <= 1e-12:
                return True
            x = dlt / T
            return x < 50.0 and rng.random() < math.exp(-x)

        def on_improve():
            nonlocal best, best_a, cur
            ex = full_cost(a)
            cur = ex
            if ex < best - 1e-7:
                best = ex
                best_a = a.copy()
                if logger:
                    logger.log_solution(best, sol_dict(best_a, best))

        while True:
            it += 1
            if (it & 63) == 0:
                if time.time() >= end_time:
                    break
                T *= 0.9965
                if T < T0 * 3e-4:
                    a[:] = best_a
                    load[:] = 0.0
                    for i2 in range(n):
                        load[best_a[i2]] += O[i2]
                    hubs = [k for k in range(n) if a[k] == k]
                    cur = best
                    T = T0 * (0.2 + 0.3 * rng.random())
            if (it % 20000) == 0:
                cur = full_cost(a)

            r = rng.random()
            if r < 0.60:  # shift
                if len(hubs) < 2:
                    continue
                i = rng.randrange(n)
                if a[i] == i:
                    continue
                k = int(a[i])
                l = hubs[rng.randrange(len(hubs))]
                if l == k or load[l] + O[i] > cap[l] + 1e-9:
                    continue
                dlt = shift_delta(a, i, k, l)
                if acc(dlt):
                    a[i] = l
                    load[k] -= O[i]
                    load[l] += O[i]
                    cur += dlt
                    if cur < best - 1e-6:
                        on_improve()
            elif r < 0.82:  # swap
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j or a[i] == i or a[j] == j:
                    continue
                ki = int(a[i])
                kj = int(a[j])
                if ki == kj:
                    continue
                if load[ki] - O[i] + O[j] > cap[ki] + 1e-9:
                    continue
                if load[kj] - O[j] + O[i] > cap[kj] + 1e-9:
                    continue
                d1 = shift_delta(a, i, ki, kj)
                a[i] = kj
                d2 = shift_delta(a, j, kj, ki)
                dlt = d1 + d2
                if acc(dlt):
                    a[j] = ki
                    load[ki] += O[j] - O[i]
                    load[kj] += O[i] - O[j]
                    cur += dlt
                    if cur < best - 1e-6:
                        on_improve()
                else:
                    a[i] = ki
            elif r < 0.90:  # open new hub
                j = rng.randrange(n)
                if a[j] == j or cap[j] < O[j] - 1e-9:
                    continue
                k = int(a[j])
                dlt = f[j] + shift_delta(a, j, k, j)
                if acc(dlt):
                    a[j] = j
                    load[k] -= O[j]
                    load[j] = O[j]
                    hubs.append(j)
                    cur += dlt
                    if cur < best - 1e-6:
                        on_improve()
            elif r < 0.96:  # close hub
                if len(hubs) < 2:
                    continue
                k = hubs[rng.randrange(len(hubs))]
                members = np.where(a == k)[0]
                tload = load.copy()
                tload[k] = 0.0
                new_a = a.copy()
                ok = True
                for i2 in sorted(members.tolist(), key=lambda x: -O[x]):
                    bl = -1
                    bc = math.inf
                    for l in hubs:
                        if l == k:
                            continue
                        if tload[l] + O[i2] <= cap[l] + 1e-9 and Cas[i2, l] < bc:
                            bc = Cas[i2, l]
                            bl = l
                    if bl < 0:
                        ok = False
                        break
                    new_a[i2] = bl
                    tload[bl] += O[i2]
                if not ok:
                    continue
                newc = full_cost(new_a)
                dlt = newc - cur
                if acc(dlt):
                    a[:] = new_a
                    load[:] = tload
                    cur = newc
                    hubs.remove(k)
                    if cur < best - 1e-6:
                        on_improve()
            else:  # relocate hub inside its cluster
                k = hubs[rng.randrange(len(hubs))]
                members = np.where(a == k)[0]
                if len(members) < 2:
                    continue
                m2 = int(members[rng.randrange(len(members))])
                if m2 == k or cap[m2] < load[k] - 1e-9:
                    continue
                new_a = a.copy()
                new_a[members] = m2
                newc = full_cost(new_a)
                dlt = newc - cur
                if acc(dlt):
                    a[:] = new_a
                    load[m2] = load[k]
                    load[k] = 0.0
                    hubs.remove(k)
                    hubs.append(m2)
                    cur = newc
                    if cur < best - 1e-6:
                        on_improve()
        return best, best_a, a, load, cur

    # ------------------- run heuristic -------------------
    a, load = construct()
    descent(a, load, min(deadline, time.time() + 5))
    best = full_cost(a)
    best_a = a.copy()
    if logger:
        logger.log_solution(best, sol_dict(best_a, best))

    use_mip = (n <= 50) and (n > 1)
    if use_mip:
        sa_end = min(deadline, time.time() + max(3.0, 0.12 * args.time_limit))
    else:
        sa_end = deadline

    if n > 1 and time.time() < sa_end:
        best, best_a, a, load, cur = anneal(a, load, best, best, best_a, sa_end)

    # polish best with descent
    if n > 1 and time.time() < deadline:
        a2 = best_a.copy()
        load2 = np.zeros(n)
        for i in range(n):
            load2[a2[i]] += O[i]
        descent(a2, load2, min(deadline, time.time() + 10))
        ex = full_cost(a2)
        if ex < best - 1e-9:
            best = ex
            best_a = a2.copy()
            if logger:
                logger.log_solution(best, sol_dict(best_a, best))

    # ------------------- exact MIP for small instances -------------------
    if use_mip and time.time() < deadline - 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("csahlp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time())

            z = m.addVars(n, n, vtype=GRB.BINARY, name="z")
            keys = [(i, k, l) for i in range(n) for k in range(n)
                    for l in range(n) if k != l]
            y = m.addVars(keys, lb=0.0, name="y")

            for i in range(n):
                m.addConstr(gp.quicksum(z[i, k] for k in range(n)) == 1)
                for k in range(n):
                    if i != k:
                        m.addConstr(z[i, k] <= z[k, k])
            for k in range(n):
                m.addConstr(gp.quicksum(O[i] * z[i, k] for i in range(n))
                            <= cap[k] * z[k, k])
            m.addConstr(gp.quicksum(cap[k] * z[k, k] for k in range(n)) >= Dtot)
            for i in range(n):
                for k in range(n):
                    m.addConstr(
                        gp.quicksum(y[i, k, l] for l in range(n) if l != k)
                        - gp.quicksum(y[i, l, k] for l in range(n) if l != k)
                        == O[i] * z[i, k]
                        - gp.quicksum(w[i, j] * z[j, k] for j in range(n)
                                      if w[i, j] != 0.0))

            obj = gp.LinExpr()
            for k in range(n):
                obj += f[k] * z[k, k]
            for i in range(n):
                for k in range(n):
                    if Cas[i, k] != 0.0:
                        obj += Cas[i, k] * z[i, k]
            for (i, k, l) in keys:
                c = alpha * d[k, l]
                if c != 0.0:
                    obj += c * y[i, k, l]
            m.setObjective(obj, GRB.MINIMIZE)

            for i in range(n):
                for k in range(n):
                    z[i, k].Start = 0.0
                z[i, int(best_a[i])].Start = 1.0

            state = {"best": best, "best_a": best_a}
            cb_seen = [math.inf]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if objv < cb_seen[0] - 1e-6:
                        cb_seen[0] = objv
                        try:
                            vals = model.cbGetSolution(z)
                            asg = np.array(
                                [max(range(n), key=lambda k2: vals[i2, k2])
                                 for i2 in range(n)], dtype=np.int64)
                            ex = full_cost(asg)
                            if ex < state["best"] - 1e-7:
                                state["best"] = ex
                                state["best_a"] = asg
                                if logger:
                                    logger.log_solution(ex, sol_dict(asg, ex))
                        except Exception:
                            pass

            m.optimize(cb)

            if m.SolCount > 0:
                asg = np.array([max(range(n), key=lambda k2: z[i2, k2].X)
                                for i2 in range(n)], dtype=np.int64)
                ex = full_cost(asg)
                if ex < state["best"] - 1e-9:
                    state["best"] = ex
                    state["best_a"] = asg
                    if logger:
                        logger.log_solution(ex, sol_dict(asg, ex))
            if state["best"] < best - 1e-9:
                best = state["best"]
                best_a = np.asarray(state["best_a"], dtype=np.int64)
        except Exception:
            pass

    # continue SA if time remains
    if n > 1 and time.time() < deadline - 2:
        a3 = best_a.copy()
        load3 = np.zeros(n)
        for i in range(n):
            load3[a3[i]] += O[i]
        best, best_a, _, _, _ = anneal(a3, load3, best, best, best_a, deadline)

    result = sol_dict(best_a, best)
    with open(args.solution_path, "w") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    main()