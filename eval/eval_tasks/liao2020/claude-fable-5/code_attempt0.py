import argparse
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(3, args.time_limit) - 2.0

    def tleft():
        return deadline - time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    P = int(inst["num_patients_P"])
    L = float(inst["provider_service_hours_L_minutes"])
    cs_all = inst["cost_structures"]
    cs_name = "Cost1" if "Cost1" in cs_all else next(iter(cs_all))
    cw = float(cs_all[cs_name]["c_w"])
    cg = float(cs_all[cs_name]["c_g"])
    co = float(cs_all[cs_name]["c_o"])

    pats = sorted(inst["patients"], key=lambda r: int(r["patient_index"]))
    pid = [int(r["patient_index"]) for r in pats]
    muA = [float(r["mean_duration_adequate_prep"]) for r in pats]
    muI = [float(r["mean_duration_inadequate_prep"]) for r in pats]
    AL = [float(r["lower_bound_duration_adequate_prep"]) for r in pats]
    AU = [float(r["upper_bound_duration_adequate_prep"]) for r in pats]
    IL = [float(r["lower_bound_duration_inadequate_prep"]) for r in pats]
    IU = [float(r["upper_bound_duration_inadequate_prep"]) for r in pats]
    muq = [min(1.0, max(0.0, float(r["mean_prep_adequacy"]))) for r in pats]
    muu = [float(r["mean_arrival_time_deviation"]) for r in pats]
    uLo = [float(r["lower_bound_arrival_time_deviation"]) for r in pats]
    uHi = [float(r["upper_bound_arrival_time_deviation"]) for r in pats]
    # clamp means into supports for numerical safety
    for p in range(P):
        muA[p] = min(max(muA[p], AL[p]), AU[p])
        muI[p] = min(max(muI[p], IL[p]), IU[p])
        muu[p] = min(max(muu[p], uLo[p]), uHi[p])
    meanD = [muq[p] * muA[p] + (1.0 - muq[p]) * muI[p] for p in range(P)]
    rng = random.Random(0)

    # ------------------------------------------------------------------
    # true recursion cost for one realization
    # ------------------------------------------------------------------
    def f_true(seq, t, D, u):
        C = 0.0
        cost = 0.0
        for i in range(P):
            p = seq[i]
            ti = t[i]
            w = u[p]
            c2 = C - ti
            if c2 > w:
                w = c2
            if w < 0.0:
                w = 0.0
            C = ti + w + D[p]
            cost += cw * w
            if i < P - 1:
                gap = t[i + 1] - C
                if gap > 0.0:
                    cost += cg * gap
        if C > L:
            cost += co * (C - L)
        return cost

    # ------------------------------------------------------------------
    # global scenario pool (patient-indexed realizations)
    # ------------------------------------------------------------------
    pool_q = []
    pool_dA = []
    pool_dI = []
    pool_u = []
    pool_D = []
    seen = {}

    def add_scen(qv, dAv, dIv, uv):
        qv = tuple(1.0 if x > 0.5 else 0.0 for x in qv)
        dAv = tuple(float(x) for x in dAv)
        dIv = tuple(float(x) for x in dIv)
        uv = tuple(float(x) for x in uv)
        key = (qv, tuple(round(x, 5) for x in dAv),
               tuple(round(x, 5) for x in dIv), tuple(round(x, 5) for x in uv))
        if key in seen:
            return seen[key], False
        idx = len(pool_q)
        seen[key] = idx
        pool_q.append(qv)
        pool_dA.append(dAv)
        pool_dI.append(dIv)
        pool_u.append(uv)
        pool_D.append(tuple(dAv[p] if qv[p] > 0.5 else dIv[p] for p in range(P)))
        return idx, True

    # comonotone Bernoulli coupling scenarios (guarantee moment-LP feasibility)
    order_q = sorted(range(P), key=lambda p: -muq[p])
    init_dist = []
    for k in range(P + 1):
        qv = [0.0] * P
        for j in range(k):
            qv[order_q[j]] = 1.0
        idx, _ = add_scen(qv, muA, muI, muu)
        if k == 0:
            pr = 1.0 - (muq[order_q[0]] if P > 0 else 0.0)
        elif k < P:
            pr = muq[order_q[k - 1]] - muq[order_q[k]]
        else:
            pr = muq[order_q[P - 1]]
        if pr > 1e-12:
            init_dist.append((idx, pr))
    # a few extreme vertices
    add_scen([1.0] * P, AU, IU, uHi)
    add_scen([0.0] * P, AU, IU, uHi)
    add_scen([1.0 if AU[p] >= IU[p] else 0.0 for p in range(P)], AU, IU, uHi)
    add_scen([1.0 if AL[p] <= IL[p] else 0.0 for p in range(P)], AL, IL, uLo)
    add_scen([0.0] * P, AL, IL, uLo)
    add_scen([1.0] * P, AL, IL, uLo)

    # ------------------------------------------------------------------
    # gurobi
    # ------------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    def new_model():
        m = gp.Model("m", env=env)
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        return m

    # ------------------------------------------------------------------
    # separation: max_xi f(xi) - rho - alpha'xi over box vertices (local search)
    # ------------------------------------------------------------------
    def separate(seq, t, rho, aq, adA, adI, au, scale):
        tol = 1e-6 * max(1.0, abs(scale))
        dA_ = [0.0] * P
        dI_ = [0.0] * P
        u_ = [0.0] * P
        D_ = [0.0] * P

        def evalh(qb, ab, bb, cb):
            pen = rho
            for p in range(P):
                dA = AU[p] if ab[p] else AL[p]
                dI = IU[p] if bb[p] else IL[p]
                uu = uHi[p] if cb[p] else uLo[p]
                dA_[p] = dA
                dI_[p] = dI
                u_[p] = uu
                D_[p] = dA if qb[p] else dI
                pen += aq[p] * qb[p] + adA[p] * dA + adI[p] * dI + au[p] * uu
            return f_true(seq, t, D_, u_) - pen

        def ls(state):
            qb, ab, bb, cb = state
            cur = evalh(qb, ab, bb, cb)
            for _ in range(30):
                improved = False
                for p in range(P):
                    for arr in (qb, ab, bb, cb):
                        arr[p] = 1 - arr[p]
                        v = evalh(qb, ab, bb, cb)
                        if v > cur + 1e-9:
                            cur = v
                            improved = True
                        else:
                            arr[p] = 1 - arr[p]
                if not improved:
                    break
            return cur, (qb, ab, bb, cb)

        starts = [
            ([1] * P, [1] * P, [1] * P, [1] * P),
            ([0] * P, [0] * P, [0] * P, [0] * P),
            ([1 if muq[p] >= 0.5 else 0 for p in range(P)], [1] * P, [1] * P, [1] * P),
            ([1 if adA[p] <= adI[p] else 0 for p in range(P)],
             [1 if adA[p] <= 0 else 0 for p in range(P)],
             [1 if adI[p] <= 0 else 0 for p in range(P)],
             [1 if au[p] <= 0 else 0 for p in range(P)]),
        ]
        for _ in range(2):
            starts.append(([rng.randint(0, 1) for _ in range(P)],
                           [rng.randint(0, 1) for _ in range(P)],
                           [rng.randint(0, 1) for _ in range(P)],
                           [rng.randint(0, 1) for _ in range(P)]))
        results = []
        seenk = set()
        for st in starts:
            v, (qb, ab, bb, cb) = ls(tuple(list(x) for x in st))
            if v > tol:
                qv = tuple(float(x) for x in qb)
                dAv = tuple(AU[p] if ab[p] else AL[p] for p in range(P))
                dIv = tuple(IU[p] if bb[p] else IL[p] for p in range(P))
                uv = tuple(uHi[p] if cb[p] else uLo[p] for p in range(P))
                k = (qv, dAv, dIv, uv)
                if k not in seenk:
                    seenk.add(k)
                    results.append((v, k))
        results.sort(key=lambda z: -z[0])
        return [k for _, k in results[:3]]

    # ------------------------------------------------------------------
    # inner worst-case expectation (moment problem, column generation)
    # ------------------------------------------------------------------
    def worst_case(seq, t, max_new, dl):
        m = new_model()
        m.ModelSense = GRB.MAXIMIZE
        c_norm = m.addLConstr(gp.LinExpr(), GRB.EQUAL, 1.0)
        c_q = [m.addLConstr(gp.LinExpr(), GRB.EQUAL, muq[p]) for p in range(P)]
        c_dA = [m.addLConstr(gp.LinExpr(), GRB.EQUAL, muA[p]) for p in range(P)]
        c_dI = [m.addLConstr(gp.LinExpr(), GRB.EQUAL, muI[p]) for p in range(P)]
        c_u = [m.addLConstr(gp.LinExpr(), GRB.EQUAL, muu[p]) for p in range(P)]
        m.update()
        vars_ = []

        def add_col(si):
            col = gp.Column()
            col.addTerms(1.0, c_norm)
            for p in range(P):
                if pool_q[si][p] != 0.0:
                    col.addTerms(pool_q[si][p], c_q[p])
                if pool_dA[si][p] != 0.0:
                    col.addTerms(pool_dA[si][p], c_dA[p])
                if pool_dI[si][p] != 0.0:
                    col.addTerms(pool_dI[si][p], c_dI[p])
                if pool_u[si][p] != 0.0:
                    col.addTerms(pool_u[si][p], c_u[p])
            v = m.addVar(lb=0.0, obj=f_true(seq, t, pool_D[si], pool_u[si]), column=col)
            vars_.append(v)

        for si in range(len(pool_q)):
            add_col(si)

        val = None
        duals = None
        added = 0
        while True:
            m.Params.TimeLimit = max(0.5, min(30.0, dl - time.time()))
            try:
                m.optimize()
            except gp.GurobiError:
                break
            if m.Status != GRB.OPTIMAL or m.SolCount == 0:
                break
            val = m.ObjVal
            try:
                rho = c_norm.Pi
                aq = [c.Pi for c in c_q]
                adA = [c.Pi for c in c_dA]
                adI = [c.Pi for c in c_dI]
                au = [c.Pi for c in c_u]
                duals = (rho, aq, adA, adI, au)
            except gp.GurobiError:
                break
            if added >= max_new or dl - time.time() < 0.5 or len(pool_q) > 4000:
                break
            news = separate(seq, t, rho, aq, adA, adI, au, val)
            grew = False
            for (qv, dAv, dIv, uv) in news:
                si, isnew = add_scen(qv, dAv, dIv, uv)
                if isnew:
                    add_col(si)
                    added += 1
                    grew = True
            if not grew:
                break
        dist = []
        if val is not None:
            for i, v in enumerate(vars_):
                try:
                    x = v.X
                except gp.GurobiError:
                    x = 0.0
                if x > 1e-10:
                    dist.append((i, x))
            s = sum(pr for _, pr in dist)
            if s > 0:
                dist = [(i, pr / s) for i, pr in dist]
        m.dispose()
        return val, dist, duals

    # ------------------------------------------------------------------
    # master schedule LP (CCG cuts = worst-case distributions)
    # ------------------------------------------------------------------
    def solve_master(seq, cuts, dl):
        ids = sorted({si for cut in cuts for si, _ in cut})
        m = new_model()
        tv = [m.addVar(lb=0.0, ub=L) for _ in range(P)]
        tv[0].UB = 0.0
        for i in range(1, P):
            m.addConstr(tv[i] >= tv[i - 1])
        cost = {}
        for si in ids:
            D = pool_D[si]
            uu = pool_u[si]
            w = [m.addVar(lb=0.0) for _ in range(P)]
            for i in range(P):
                p = seq[i]
                if uu[p] > 0:
                    m.addConstr(w[i] >= uu[p])
                if i > 0:
                    m.addConstr(w[i] >= w[i - 1] + D[seq[i - 1]] - tv[i] + tv[i - 1])
            expr = cw * gp.quicksum(w)
            if P > 1:
                g = [m.addVar(lb=0.0) for _ in range(P - 1)]
                for i in range(P - 1):
                    m.addConstr(g[i] >= tv[i + 1] - tv[i] - w[i] - D[seq[i]])
                expr = expr + cg * gp.quicksum(g)
            O = m.addVar(lb=0.0)
            m.addConstr(O >= tv[P - 1] + w[P - 1] + D[seq[P - 1]] - L)
            expr = expr + co * O
            cost[si] = expr
        th = m.addVar(lb=0.0)
        for cut in cuts:
            m.addConstr(th >= gp.quicksum(pr * cost[si] for si, pr in cut))
        m.setObjective(th, GRB.MINIMIZE)
        m.Params.TimeLimit = max(0.5, min(60.0, dl - time.time()))
        try:
            m.optimize()
        except gp.GurobiError:
            m.dispose()
            return None, None
        if m.SolCount == 0:
            m.dispose()
            return None, None
        res = [tv[i].X for i in range(P)]
        res[0] = 0.0
        for i in range(1, P):
            res[i] = min(L, max(res[i], res[i - 1]))
        lb = m.ObjVal
        m.dispose()
        return res, lb

    # ------------------------------------------------------------------
    # CCG for a fixed sequence
    # ------------------------------------------------------------------
    def ccg(seq, t0, iters, sep, dl):
        cuts = []
        t = list(t0)
        bval = math.inf
        bt = list(t0)
        bdist = None
        bduals = None
        for it in range(iters):
            if dl - time.time() < 1.0:
                break
            val, dist, duals = worst_case(seq, t, sep, dl)
            if val is None:
                break
            if val < bval:
                bval = val
                bt = list(t)
                bdist = dist
                bduals = duals
            if dist:
                cuts.append(dist)
            if it == iters - 1 or not cuts or dl - time.time() < 1.0:
                break
            tv, lb = solve_master(seq, cuts, dl)
            if tv is None:
                break
            if lb is not None and lb >= bval - 1e-6 * max(1.0, abs(bval)):
                break
            t = tv
        return (bval if math.isfinite(bval) else None), bt, bdist, bduals

    def exp_cost(seq, t, dist):
        return sum(pr * f_true(seq, t, pool_D[si], pool_u[si]) for si, pr in dist)

    def sched0(seq):
        t = [0.0] * P
        c = 0.0
        for i in range(1, P):
            c += meanD[seq[i - 1]]
            t[i] = min(L, max(t[i - 1], c))
        return t

    def polish_t(seq, t, dist):
        tv = list(t)
        cur = exp_cost(seq, tv, dist)
        for step in (8.0, 4.0, 2.0, 1.0, 0.5):
            improved = True
            while improved and tleft() > 2.0:
                improved = False
                for i in range(1, P):
                    base = tv[i]
                    lo = tv[i - 1]
                    hi = tv[i + 1] if i < P - 1 else L
                    for d in (step, -step):
                        cand = min(hi, max(lo, base + d))
                        if abs(cand - base) < 1e-9:
                            continue
                        tv[i] = cand
                        v = exp_cost(seq, tv, dist)
                        if v < cur - 1e-9:
                            cur = v
                            base = cand
                            improved = True
                        else:
                            tv[i] = base
        return tv

    # ------------------------------------------------------------------
    # solution assembly / incumbent management
    # ------------------------------------------------------------------
    def make_solution(val, seq, t, duals):
        assignment = {}
        pst = {}
        sched = {}
        for pos in range(P):
            p = seq[pos]
            assignment[str(pid[p])] = pos + 1
            pst[str(pid[p])] = float(t[pos])
            sched[str(pos + 1)] = float(t[pos])
        mv = {}
        for pos in range(P):
            mv["x_{}_{}".format(pid[seq[pos]], pos + 1)] = 1.0
        for pos in range(P):
            if abs(t[pos]) > 1e-12:
                mv["t_{}".format(pos + 1)] = float(t[pos])
        if duals:
            rho, aq, adA, adI, au = duals
            if abs(rho) > 1e-12:
                mv["rho"] = float(rho)
            for p in range(P):
                for nm, arr in (("alpha_q", aq), ("alpha_dA", adA),
                                ("alpha_dI", adI), ("alpha_u", au)):
                    if abs(arr[p]) > 1e-12:
                        mv["{}_{}".format(nm, pid[p])] = float(arr[p])
        return {
            "objective_value": float(val),
            "model_variables": mv,
            "cost_structure": cs_name,
            "c_w": cw,
            "c_g": cg,
            "c_o": co,
            "assignment": assignment,
            "schedule": sched,
            "patient_start_times": pst,
        }

    seq_init = list(range(P))
    inc = {"val": math.inf, "seq": seq_init, "t": sched0(seq_init),
           "dist": init_dist, "duals": None}
    logged_best = [math.inf]

    def write_solution(val, seq, t, duals):
        sol = make_solution(val, seq, t, duals)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return sol

    def try_update(val, seq, t, dist, duals):
        if val is None or not math.isfinite(val):
            return False
        if val < inc["val"] - 1e-9:
            inc["val"] = val
            inc["seq"] = list(seq)
            inc["t"] = list(t)
            if dist:
                inc["dist"] = dist
            if duals:
                inc["duals"] = duals
            sol = write_solution(val, inc["seq"], inc["t"], inc["duals"])
            if logger and val < logged_best[0] - 1e-9:
                logged_best[0] = val
                logger.log_solution(float(val), sol)
            return True
        return False

    # write an early fallback file (not logged; not a worst-case value)
    write_solution(exp_cost(inc["seq"], inc["t"], init_dist), inc["seq"], inc["t"], None)

    try:
        # ---------------- seed sequences ----------------
        proxyvar = [muq[p] * (1 - muq[p]) * (muA[p] - muI[p]) ** 2
                    + muq[p] * ((AU[p] - AL[p]) / 2.0) ** 2
                    + (1 - muq[p]) * ((IU[p] - IL[p]) / 2.0) ** 2
                    + ((uHi[p] - uLo[p]) / 2.0) ** 2 for p in range(P)]
        seeds = []

        def addseed(s):
            s = list(s)
            if s not in seeds:
                seeds.append(s)

        addseed(sorted(range(P), key=lambda p: proxyvar[p]))
        addseed(sorted(range(P), key=lambda p: -proxyvar[p]))
        addseed(sorted(range(P), key=lambda p: meanD[p]))
        addseed(seq_init)

        small = args.time_limit < 60
        seed_iters = 2 if small else 3
        seed_sep = 6 if small else 10

        for s in seeds:
            if tleft() < 4:
                break
            val, tt, dist, duals = ccg(s, sched0(s), seed_iters, seed_sep, deadline)
            try_update(val, s, tt, dist, duals)

        # ---------------- deep refinement of incumbent ----------------
        if tleft() > 8 and math.isfinite(inc["val"]):
            val, tt, dist, duals = ccg(inc["seq"], inc["t"], 8, 25, deadline)
            try_update(val, inc["seq"] if val is None else inc["seq"], tt, dist, duals)

        # ---------------- sequence local search ----------------
        while tleft() > 6 and P > 1 and math.isfinite(inc["val"]):
            base = inc["seq"]
            moves = []
            mv_seen = set()
            for i in range(P):
                for j in range(i + 1, P):
                    s2 = base[:]
                    s2[i], s2[j] = s2[j], s2[i]
                    k = tuple(s2)
                    if k not in mv_seen:
                        mv_seen.add(k)
                        moves.append(s2)
            for i in range(P):
                for j in range(P):
                    if j == i:
                        continue
                    s2 = base[:]
                    x = s2.pop(i)
                    s2.insert(j, x)
                    k = tuple(s2)
                    if k not in mv_seen and s2 != base:
                        mv_seen.add(k)
                        moves.append(s2)
            if len(moves) > 250:
                moves = rng.sample(moves, 250)
            dist_ref = inc["dist"] if inc["dist"] else init_dist
            scored = sorted(((exp_cost(s2, inc["t"], dist_ref), s2) for s2 in moves),
                            key=lambda z: z[0])
            improved = False
            for k in range(min(4, len(scored))):
                if tleft() < 4:
                    break
                _, s2 = scored[k]
                val, tt, dist, duals = ccg(s2, inc["t"], 3, 10, deadline)
                if try_update(val, s2, tt, dist, duals):
                    improved = True
                    break
            if not improved:
                break

        # ---------------- schedule polish ----------------
        if tleft() > 4 and math.isfinite(inc["val"]) and inc["dist"]:
            t2 = polish_t(inc["seq"], inc["t"], inc["dist"])
            if any(abs(a - b) > 1e-9 for a, b in zip(t2, inc["t"])):
                val, dist, duals = worst_case(inc["seq"], t2, 20, deadline)
                try_update(val, inc["seq"], t2, dist, duals)

        # ---------------- final honest re-evaluation ----------------
        final_val = inc["val"]
        final_duals = inc["duals"]
        if tleft() > 1.5:
            v, d, du = worst_case(inc["seq"], inc["t"], 15, deadline)
            if v is not None:
                if v < inc["val"] - 1e-9:
                    try_update(v, inc["seq"], inc["t"], d, du)
                    final_val = inc["val"]
                    final_duals = inc["duals"]
                else:
                    final_val = v
                    if du:
                        final_duals = du
        if not math.isfinite(final_val):
            final_val = exp_cost(inc["seq"], inc["t"], init_dist)
        write_solution(final_val, inc["seq"], inc["t"], final_duals)
    except Exception:
        # guarantee a valid output file no matter what
        fv = inc["val"] if math.isfinite(inc["val"]) else exp_cost(inc["seq"], inc["t"], init_dist)
        try:
            write_solution(fv, inc["seq"], inc["t"], inc["duals"])
        except Exception:
            pass


if __name__ == "__main__":
    main()