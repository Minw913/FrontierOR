import argparse
import json
import time
import random
from bisect import bisect_right
from collections import deque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))
    reserve = 1.5

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    n_real = int(inst["num_real_activities"])
    N = int(inst.get("num_activities_total", n_real + 2))
    if N < n_real + 2:
        N = n_real + 2
    end = n_real + 1
    D = int(inst["max_project_duration"])

    p = [0] * N
    for a in inst.get("activities", []):
        aid = int(a["id"])
        if 0 <= aid < N:
            p[aid] = int(a["processing_time"])
    p[0] = 0
    p[end] = 0

    edges = []
    for c in inst.get("temporal_constraints", []):
        u = int(c["from_activity"]); v = int(c["to_activity"]); w = int(c["time_lag"])
        if 0 <= u < N and 0 <= v < N:
            edges.append((u, v, w))
    # deadline constraint (safe even if duplicated in the data)
    edges.append((end, 0, -D))

    succ = [[] for _ in range(N)]
    pred = [[] for _ in range(N)]
    radj = [[] for _ in range(N)]
    for (u, v, w) in edges:
        succ[u].append((v, w))
        pred[v].append((u, w))
        radj[v].append((u, w))

    NINF = float("-inf")

    def spfa_longest(adj, init):
        dist = list(init)
        inq = [False] * N
        cnt = [0] * N
        q = deque([i for i in range(N) if dist[i] > NINF])
        for i in q:
            inq[i] = True
        while q:
            u = q.popleft()
            inq[u] = False
            du = dist[u]
            for (v, w) in adj[u]:
                nd = du + w
                if nd > dist[v]:
                    dist[v] = nd
                    cnt[v] += 1
                    if cnt[v] >= N:
                        return None  # positive cycle -> infeasible
                    if not inq[v]:
                        inq[v] = True
                        q.append(v)
        return dist

    temporal_feasible = spfa_longest(succ, [0] * N) is not None
    init0 = [NINF] * N
    init0[0] = 0
    dist_from0 = spfa_longest(succ, init0)
    dist_to0 = spfa_longest(radj, init0)
    if dist_from0 is None or dist_to0 is None:
        temporal_feasible = False

    ES = [0] * N
    LS = [D] * N
    if dist_from0 is not None and dist_to0 is not None:
        for i in range(N):
            es = dist_from0[i]
            ES[i] = 0 if (es == NINF or es < 0) else es
            ls = dist_to0[i]
            LS[i] = D if ls == NINF else min(D, -ls)
    ES[0] = 0
    LS[0] = 0
    windows_ok = temporal_feasible and all(ES[i] <= LS[i] for i in range(N))

    # ---- resources ----
    resources = inst.get("resources", []) or []
    K = len(resources)
    Pi_sorted = []
    cap = []
    dem = []
    for res in resources:
        Pi = sorted(set(int(x) for x in res.get("Pi_k", []) if 1 <= int(x) <= D))
        Pi_sorted.append(Pi)
        cap.append(int(res.get("capacity", 0)))
        dd = [0] * N
        for kstr, val in (res.get("demands", {}) or {}).items():
            ai = int(kstr)
            if 0 <= ai < N:
                dd[ai] = int(val)
        dem.append(dd)

    cons_res = [[] for _ in range(N)]
    for i in range(N):
        if p[i] > 0:
            for k in range(K):
                r = dem[k][i]
                if r > 0:
                    cons_res[i].append((k, r))

    res_acts = [[] for _ in range(K)]
    for i in range(N):
        for (k, r) in cons_res[i]:
            res_acts[k].append(i)

    # breakpoints where coverage (and hence consumption) can change
    bps = [[] for _ in range(N)]
    for i in range(N):
        if cons_res[i]:
            pi = p[i]
            bs = set()
            for (k, r) in cons_res[i]:
                Pi = Pi_sorted[k]
                L = len(Pi)
                if L > 20000:
                    step = (L + 19999) // 20000
                    Pi = Pi[::step]
                for per in Pi:
                    bs.add(per - pi)
                    bs.add(per)
                if len(bs) > 300000:
                    break
            bps[i] = sorted(bs)

    rng = random.Random(0)

    def cov(k, i, t):
        Pi = Pi_sorted[k]
        return bisect_right(Pi, t + p[i]) - bisect_right(Pi, t)

    def envelope(i, s):
        lb = ES[i]
        for (u, w) in pred[i]:
            v = s[u] + w
            if v > lb:
                lb = v
        ub = LS[i]
        for (v, w) in succ[i]:
            b = s[v] - w
            if b < ub:
                ub = b
        if lb < 0:
            lb = 0
        if ub > D:
            ub = D
        return lb, ub

    MAXC = 4000

    def cand_times(i, lo, hi):
        if hi < lo:
            return []
        cands = [lo]
        bl = bps[i]
        if bl:
            a = bisect_right(bl, lo)
            b = bisect_right(bl, hi)
            n = b - a
            if 0 < n <= MAXC:
                cands.extend(bl[a:b])
            elif n > MAXC:
                step = n / MAXC
                for j in range(MAXC):
                    cands.append(bl[a + int(j * step)])
                cands.append(bl[b - 1])
        return cands

    def recompute_cons(s):
        cons = [0] * K
        for i in range(N):
            for (k, r) in cons_res[i]:
                cons[k] += r * cov(k, i, s[i])
        return cons

    def total_viol(cons):
        V = 0
        for k in range(K):
            o = cons[k] - cap[k]
            if o > 0:
                V += o
        return V

    def move_delta_viol(i, t0, t1, cons):
        dv = 0
        for (k, r) in cons_res[i]:
            d = r * (cov(k, i, t1) - cov(k, i, t0))
            o0 = cons[k] - cap[k]
            o1 = o0 + d
            if o1 > 0:
                dv += o1
            if o0 > 0:
                dv -= o0
        return dv

    def move_feasible(i, t0, t1, cons):
        for (k, r) in cons_res[i]:
            if cons[k] + r * (cov(k, i, t1) - cov(k, i, t0)) > cap[k]:
                return False
        return True

    def apply_move(i, t1, s, cons):
        t0 = s[i]
        if t1 == t0:
            return
        for (k, r) in cons_res[i]:
            cons[k] += r * (cov(k, i, t1) - cov(k, i, t0))
        s[i] = t1

    def temporal_closure(s):
        for _ in range(N):
            changed = False
            for (u, v, w) in edges:
                if v == 0:
                    continue
                if s[u] + w > s[v]:
                    s[v] = s[u] + w
                    changed = True
            if not changed:
                break
        s[0] = 0

    def serial_init(jitter, deadline):
        if jitter > 0:
            order = sorted(range(N), key=lambda i: ES[i] + rng.random() * jitter)
        else:
            order = sorted(range(N), key=lambda i: ES[i])
        s = [0] * N
        scheduled = [False] * N
        cons = [0] * K
        for idx, i in enumerate(order):
            if (idx & 31) == 0 and time.time() > deadline:
                break
            lb = ES[i]
            ub = LS[i]
            for (u, w) in pred[i]:
                if scheduled[u]:
                    v = s[u] + w
                    if v > lb:
                        lb = v
            for (v, w) in succ[i]:
                if scheduled[v]:
                    b = s[v] - w
                    if b < ub:
                        ub = b
            if lb < 0:
                lb = 0
            if ub > D:
                ub = D
            if lb > ub:
                t = lb
            elif not cons_res[i]:
                t = lb
            else:
                t = None
                for tt in cand_times(i, lb, ub):
                    ok = True
                    for (k, r) in cons_res[i]:
                        if cons[k] + r * cov(k, i, tt) > cap[k]:
                            ok = False
                            break
                    if ok:
                        t = tt
                        break
                if t is None:
                    besttt = lb
                    bestv = None
                    for tt in cand_times(i, lb, ub):
                        vv = 0
                        for (k, r) in cons_res[i]:
                            nc = cons[k] + r * cov(k, i, tt)
                            if nc > cap[k]:
                                vv += nc - cap[k]
                        if bestv is None or vv < bestv:
                            bestv = vv
                            besttt = tt
                    t = besttt
            s[i] = t
            scheduled[i] = True
            for (k, r) in cons_res[i]:
                cons[k] += r * cov(k, i, t)
        for j in range(N):
            if not scheduled[j]:
                s[j] = ES[j]
        temporal_closure(s)
        cons = recompute_cons(s)
        return s, cons

    def repair(s, cons, deadline):
        V = total_viol(cons)
        if V <= 0:
            return 0
        stuck = 0
        it = 0
        while V > 0 and it < 5000 and time.time() < deadline:
            it += 1
            kstar = -1
            worst = 0
            for k in range(K):
                o = cons[k] - cap[k]
                if o > worst:
                    worst = o
                    kstar = k
            if kstar < 0:
                break
            acts = res_acts[kstar]
            if not acts:
                break
            acts_sorted = sorted(acts, key=lambda i: dem[kstar][i] * cov(kstar, i, s[i]),
                                 reverse=True)
            moved = False
            for i in acts_sorted:
                if time.time() >= deadline:
                    break
                lb, ub = envelope(i, s)
                if lb > ub:
                    continue
                t0 = s[i]
                best_t = t0
                best_dv = 0
                for t in cand_times(i, lb, ub):
                    if t == t0:
                        continue
                    dv = move_delta_viol(i, t0, t, cons)
                    if dv < best_dv:
                        best_dv = dv
                        best_t = t
                if best_t != t0:
                    apply_move(i, best_t, s, cons)
                    V += best_dv
                    moved = True
                    break
            if not moved:
                stuck += 1
                if stuck > 30:
                    break
                i = acts_sorted[rng.randrange(len(acts_sorted))]
                lb, ub = envelope(i, s)
                if lb <= ub:
                    cs = cand_times(i, lb, ub)
                    if cs:
                        t = cs[rng.randrange(len(cs))]
                        dv = move_delta_viol(i, s[i], t, cons)
                        if dv <= 0 or rng.random() < 0.05:
                            apply_move(i, t, s, cons)
                            V += dv
        return V

    def descent(s, cons, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            order = sorted(range(N), key=lambda j: -s[j])
            for i in order:
                if time.time() >= deadline:
                    break
                t0 = s[i]
                lb, ub = envelope(i, s)
                if lb >= t0:
                    continue
                for t in cand_times(i, lb, t0 - 1):
                    if move_feasible(i, t0, t, cons):
                        apply_move(i, t, s, cons)
                        improved = True
                        break

    def full_check(s):
        if s[0] != 0:
            return False, None
        for i in range(N):
            if s[i] < 0:
                return False, None
        for (u, v, w) in edges:
            if s[v] - s[u] < w:
                return False, None
        cons = [0] * K
        for i in range(N):
            for (k, r) in cons_res[i]:
                cons[k] += r * cov(k, i, s[i])
        for k in range(K):
            if cons[k] > cap[k]:
                return False, cons
        return True, cons

    best = {"s": None, "obj": float("inf")}

    def solution_dict(s, obj):
        return {"objective_value": float(obj),
                "schedule": {str(i): int(s[i]) for i in range(N)}}

    def try_update(s):
        ok, _ = full_check(s)
        if ok and s[end] < best["obj"]:
            best["obj"] = s[end]
            best["s"] = list(s)
            if logger:
                logger.log_solution(float(s[end]), solution_dict(s, s[end]))
            return True
        return False

    def heur_pass(jitter, deadline):
        s, cons = serial_init(jitter, deadline)
        V = total_viol(cons)
        if V > 0:
            V = repair(s, cons, deadline)
        if V == 0 and time.time() < deadline:
            descent(s, cons, deadline)
        try_update(s)

    # ---- size estimates for the MIP ----
    consuming_idx = [i for i in range(N) if cons_res[i]]
    B_total = 0
    for i in consuming_idx:
        B_total += LS[i] - ES[i] + 1
    total_terms = 0
    for k in range(K):
        for i in res_acts[k]:
            total_terms += LS[i] - ES[i] + 1

    elapsed = time.time() - t_start
    rem_total = time_limit - elapsed - reserve
    lower_bound = ES[end]
    est_build = total_terms / 150000.0 + B_total / 80000.0 + 1.0
    can_mip = (windows_ok and K > 0 and B_total <= 300000
               and total_terms <= 6000000)

    heur_budget = max(0.5, min(6.0, 0.05 * time_limit))
    if can_mip:
        heur_deadline = time.time() + min(heur_budget, max(0.0, rem_total - 1.5))
    else:
        heur_deadline = time.time() + max(0.0, rem_total)

    # ---- phase 1: fast heuristics to secure a good incumbent ----
    try:
        if windows_ok and time.time() < heur_deadline:
            heur_pass(0.0, heur_deadline)
            if K > 0 and time.time() < heur_deadline:
                s = list(ES)
                cons = recompute_cons(s)
                V = total_viol(cons)
                if V > 0:
                    V = repair(s, cons, heur_deadline)
                if V == 0 and time.time() < heur_deadline:
                    descent(s, cons, heur_deadline)
                try_update(s)
            it = 0
            while K > 0 and time.time() < heur_deadline and it < 1000:
                heur_pass(1.0, heur_deadline)
                it += 1
    except Exception:
        pass

    rem_now = time_limit - (time.time() - t_start) - reserve
    use_mip = (can_mip and best["obj"] > lower_bound and rem_now > est_build + 2.0)

    # ---- phase 2: time-indexed MIP ----
    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("prcpsp")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.OutputFlag = 0

            yidx = []
            for i in consuming_idx:
                for t in range(ES[i], LS[i] + 1):
                    yidx.append((i, t))
            y = m.addVars(yidx, vtype=GRB.BINARY, name="y")
            s_ids = [i for i in range(N) if not cons_res[i]]
            svar = m.addVars(s_ids,
                             lb=[float(ES[i]) for i in s_ids],
                             ub=[float(LS[i]) for i in s_ids],
                             vtype=GRB.INTEGER, name="s")
            m.update()

            start_expr = {}
            for i in consuming_idx:
                trange = range(ES[i], LS[i] + 1)
                vs = [y[(i, t)] for t in trange]
                m.addConstr(gp.quicksum(vs) == 1.0)
                start_expr[i] = gp.LinExpr([float(t) for t in trange], vs)
            for i in s_ids:
                start_expr[i] = svar[i]

            for (u, v, w) in edges:
                m.addConstr(start_expr[v] - start_expr[u] >= float(w))

            for k in range(K):
                Pi = Pi_sorted[k]
                coefs = []
                vars_ = []
                for i in res_acts[k]:
                    r = dem[k][i]
                    pi = p[i]
                    for t in range(ES[i], LS[i] + 1):
                        c = bisect_right(Pi, t + pi) - bisect_right(Pi, t)
                        if c:
                            coefs.append(float(r * c))
                            vars_.append(y[(i, t)])
                if vars_:
                    m.addConstr(gp.LinExpr(coefs, vars_) <= float(cap[k]))

            m.setObjective(start_expr[end], GRB.MINIMIZE)

            if best["s"] is not None:
                bs = best["s"]
                varlist = []
                vallist = []
                for i in consuming_idx:
                    ti = bs[i]
                    if (i, ti) in y:
                        varlist.append(y[(i, ti)])
                        vallist.append(1.0)
                for i in s_ids:
                    varlist.append(svar[i])
                    vallist.append(float(bs[i]))
                if varlist:
                    m.setAttr("Start", varlist, vallist)

            y_keys = yidx
            y_vars = [y[k] for k in yidx]
            s_keys = s_ids
            s_vars = [svar[i] for i in s_ids]

            def reconstruct(yvals, svals):
                sched = [0] * N
                for j in range(len(y_keys)):
                    if yvals[j] > 0.5:
                        i, t = y_keys[j]
                        sched[i] = t
                for j in range(len(s_keys)):
                    sched[s_keys[j]] = int(round(svals[j]))
                return sched

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        yvals = model.cbGetSolution(y_vars) if y_vars else []
                        svals = model.cbGetSolution(s_vars) if s_vars else []
                    except Exception:
                        return
                    sched = reconstruct(yvals, svals)
                    obj = sched[end]
                    if obj < best["obj"]:
                        best["obj"] = obj
                        best["s"] = sched
                        if logger:
                            logger.log_solution(float(obj),
                                                solution_dict(sched, obj))

            remaining = time_limit - (time.time() - t_start) - reserve
            m.Params.TimeLimit = max(1.0, remaining)
            m.optimize(cb)

            if m.SolCount > 0:
                yvals = m.getAttr("X", y_vars) if y_vars else []
                svals = m.getAttr("X", s_vars) if s_vars else []
                sched = reconstruct(yvals, svals)
                ok, _ = full_check(sched)
                if ok and sched[end] < best["obj"]:
                    best["obj"] = sched[end]
                    best["s"] = sched
                    if logger:
                        logger.log_solution(float(sched[end]),
                                            solution_dict(sched, sched[end]))
        except Exception:
            pass
    else:
        # continue improving with heuristics for the remaining time
        try:
            dl = t_start + time_limit - reserve
            it = 0
            while windows_ok and K > 0 and time.time() < dl and it < 2000:
                heur_pass(1.0, dl)
                it += 1
        except Exception:
            pass

    # ---- final output ----
    if best["s"] is None:
        s = [0] * N
        for i in range(N):
            v = ES[i]
            if v < 0:
                v = 0
            elif v > D:
                v = D
            s[i] = v
        best["s"] = s
        best["obj"] = s[end]

    out = {"objective_value": float(best["obj"]),
           "schedule": {str(i): int(best["s"][i]) for i in range(N)}}
    with open(args.solution_path, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()