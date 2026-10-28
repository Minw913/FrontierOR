import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(5, int(args.time_limit)) - 2.0

    def left():
        return deadline - time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    P = int(data["num_patients_P"])
    L = float(data["provider_service_hours_L_minutes"])
    css = data["cost_structures"]
    cs_name = next(iter(css.keys()))
    cw = float(css[cs_name]["c_w"])
    cg = float(css[cs_name]["c_g"])
    co = float(css[cs_name]["c_o"])

    pats = sorted(data["patients"], key=lambda r: int(r["patient_index"]))
    pid = [int(r["patient_index"]) for r in pats]
    muA = [float(r["mean_duration_adequate_prep"]) for r in pats]
    muI = [float(r["mean_duration_inadequate_prep"]) for r in pats]
    dAL = [float(r["lower_bound_duration_adequate_prep"]) for r in pats]
    dAU = [float(r["upper_bound_duration_adequate_prep"]) for r in pats]
    dIL = [float(r["lower_bound_duration_inadequate_prep"]) for r in pats]
    dIU = [float(r["upper_bound_duration_inadequate_prep"]) for r in pats]
    muq = [float(r["mean_prep_adequacy"]) for r in pats]
    muu = [float(r["mean_arrival_time_deviation"]) for r in pats]
    uLo = [float(r["lower_bound_arrival_time_deviation"]) for r in pats]
    uHi = [float(r["upper_bound_arrival_time_deviation"]) for r in pats]

    BND = 10.0 * (P + 2) * (cw + cg + co) + 100.0

    # ---------------- true-cost recursion evaluator (fallback) ----------------
    def eval_cost(order, tv, dpos, upos):
        w0 = max(0.0, upos[0])
        tot = cw * w0
        C = tv[0] + w0 + dpos[0]
        for i in range(1, P):
            wi = max(0.0, upos[i], C - tv[i])
            gi = tv[i] + wi - C
            tot += cw * wi + cg * gi
            C = tv[i] + wi + dpos[i]
        tot += co * max(0.0, C - L)
        return tot

    # ---------------- master problem ----------------
    m = gp.Model("master")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    x = [[m.addVar(vtype=GRB.BINARY, name="x[%d,%d]" % (p, i)) for i in range(P)] for p in range(P)]
    t = [m.addVar(lb=0.0, ub=L, name="t[%d]" % i) for i in range(P)]
    m.addConstr(t[0] == 0.0)
    for i in range(P - 1):
        m.addConstr(t[i] <= t[i + 1])
    for p in range(P):
        m.addConstr(gp.quicksum(x[p][i] for i in range(P)) == 1)
    for i in range(P):
        m.addConstr(gp.quicksum(x[p][i] for p in range(P)) == 1)

    theta = m.addVar(lb=-GRB.INFINITY, name="theta")
    rho = [m.addVar(lb=-BND, ub=BND) for _ in range(P)]
    alp = [m.addVar(lb=-BND, ub=BND) for _ in range(P)]
    lam = [m.addVar(lb=-BND, ub=BND) for _ in range(P)]
    gam = [m.addVar(lb=-BND, ub=BND) for _ in range(P)]

    m.setObjective(
        theta + gp.quicksum(rho[p] * muA[p] + alp[p] * muI[p] + lam[p] * muq[p] + gam[p] * muu[p]
                            for p in range(P)),
        GRB.MINIMIZE)

    # symmetry breaking among identical patients
    groups = {}
    for p in range(P):
        key = (round(muA[p], 9), round(muI[p], 9), round(dAL[p], 9), round(dAU[p], 9),
               round(dIL[p], 9), round(dIU[p], 9), round(muq[p], 9), round(muu[p], 9),
               round(uLo[p], 9), round(uHi[p], 9))
        groups.setdefault(key, []).append(p)
    for key, plist in groups.items():
        for a, b in zip(plist, plist[1:]):
            posa = gp.quicksum((i + 1) * x[a][i] for i in range(P))
            posb = gp.quicksum((i + 1) * x[b][i] for i in range(P))
            m.addConstr(posa + 1 <= posb)

    seen = set()

    def scen_key(qv, dAv, dIv, uv):
        return tuple([int(qv[p]) for p in range(P)] +
                     [round(dAv[p], 4) for p in range(P)] +
                     [round(dIv[p], 4) for p in range(P)] +
                     [round(uv[p], 4) for p in range(P)])

    def add_scenario(qv, dAv, dIv, uv):
        key = scen_key(qv, dAv, dIv, uv)
        if key in seen:
            return False
        seen.add(key)
        d = [qv[p] * dAv[p] + (1.0 - qv[p]) * dIv[p] for p in range(P)]
        upos = [max(0.0, uv[p]) for p in range(P)]
        W1 = gp.quicksum(upos[p] * x[p][0] for p in range(P))
        D = [gp.quicksum(d[p] * x[p][i] for p in range(P)) for i in range(P)]
        U = [gp.quicksum(uv[p] * x[p][i] for p in range(P)) for i in range(P)]
        w = {}
        for i in range(1, P):
            w[i] = m.addVar(lb=0.0)
        o = m.addVar(lb=0.0)
        for i in range(1, P):
            m.addConstr(w[i] >= U[i])
            prev = W1 if i == 1 else w[i - 1]
            m.addConstr(w[i] >= prev + D[i - 1] - (t[i] - t[i - 1]))
        lastw = w[P - 1] if P > 1 else W1
        m.addConstr(o >= lastw + D[P - 1] - (L - t[P - 1]))
        sumw = W1 + gp.quicksum(w[i] for i in range(1, P))
        cost = cw * sumw + co * o
        if P > 1:
            cost = cost + cg * (t[P - 1] - t[0] + lastw - W1 -
                                gp.quicksum(D[i] for i in range(P - 1)))
        mom = gp.quicksum(rho[p] * dAv[p] + alp[p] * dIv[p] + lam[p] * qv[p] + gam[p] * uv[p]
                          for p in range(P))
        m.addConstr(theta + mom >= cost)
        return True

    # initial vertex scenarios
    init_patterns = [(1, 1, 1, 1), (0, 1, 1, 1), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 0, 1), (0, 0, 1, 0)]
    for (qq, hA, hI, hu) in init_patterns:
        qv = [float(qq)] * P
        dAv = [dAU[p] if hA else dAL[p] for p in range(P)]
        dIv = [dIU[p] if hI else dIL[p] for p in range(P)]
        uv = [uHi[p] if hu else uLo[p] for p in range(P)]
        add_scenario(qv, dAv, dIv, uv)

    # ---------------- separation problem ----------------
    def separate(order, tv, rp, av, lv, gv, th, tlim, tol):
        sm = gp.Model("sep")
        sm.Params.OutputFlag = 0
        sm.Params.Seed = 0
        sm.Params.MIPGap = 1e-4
        sm.Params.NumericFocus = 0
        sm.Params.Threads = 1
        sm.Params.TimeLimit = max(1.0, tlim)

        aL = [dAL[p] for p in order]
        aU = [dAU[p] for p in order]
        iL = [dIL[p] for p in order]
        iU = [dIU[p] for p in order]
        ul = [uLo[p] for p in order]
        uh = [uHi[p] for p in order]
        dmax = [max(aU[i], iU[i]) for i in range(P)]

        Wcap = [0.0] * P
        Wcap[0] = max(0.0, uh[0]) + 1.0
        for i in range(1, P):
            Wcap[i] = max(0.0, uh[i], Wcap[i - 1] + dmax[i - 1] - (tv[i] - tv[i - 1])) + 1.0
        Ocap = Wcap[P - 1] + dmax[P - 1] + L + 10.0

        q = [sm.addVar(vtype=GRB.BINARY) for _ in range(P)]
        dA = [sm.addVar(lb=aL[i], ub=aU[i]) for i in range(P)]
        dI = [sm.addVar(lb=iL[i], ub=iU[i]) for i in range(P)]
        u = [sm.addVar(lb=ul[i], ub=uh[i]) for i in range(P)]
        zA = [sm.addVar(lb=min(0.0, aL[i]), ub=max(0.0, aU[i])) for i in range(P)]
        zI = [sm.addVar(lb=min(0.0, iL[i]), ub=max(0.0, iU[i])) for i in range(P)]
        dexpr = []
        for i in range(P):
            sm.addConstr(zA[i] <= aU[i] * q[i])
            sm.addConstr(zA[i] >= aL[i] * q[i])
            sm.addConstr(zA[i] <= dA[i] - aL[i] * (1 - q[i]))
            sm.addConstr(zA[i] >= dA[i] - aU[i] * (1 - q[i]))
            sm.addConstr(zI[i] <= iU[i] * q[i])
            sm.addConstr(zI[i] >= iL[i] * q[i])
            sm.addConstr(zI[i] <= dI[i] - iL[i] * (1 - q[i]))
            sm.addConstr(zI[i] >= dI[i] - iU[i] * (1 - q[i]))
            dexpr.append(zA[i] + dI[i] - zI[i])

        w = [sm.addVar(lb=0.0, ub=Wcap[i]) for i in range(P)]
        o = sm.addVar(lb=0.0, ub=Ocap)
        for i in range(P):
            sm.addConstr(w[i] >= u[i])
            if i == 0:
                terms = [gp.LinExpr(0.0), gp.LinExpr(u[0])]
            else:
                prev = w[i - 1] + dexpr[i - 1] - (tv[i] - tv[i - 1])
                sm.addConstr(w[i] >= prev)
                terms = [gp.LinExpr(0.0), gp.LinExpr(u[i]), prev]
            bs = [sm.addVar(vtype=GRB.BINARY) for _ in terms]
            sm.addConstr(gp.quicksum(bs) == 1)
            for bvar, term in zip(bs, terms):
                sm.addGenConstrIndicator(bvar, True, w[i] <= term)
        oc = w[P - 1] + dexpr[P - 1] - (L - tv[P - 1])
        sm.addConstr(o >= oc)
        b0 = sm.addVar(vtype=GRB.BINARY)
        b1 = sm.addVar(vtype=GRB.BINARY)
        sm.addConstr(b0 + b1 == 1)
        sm.addGenConstrIndicator(b0, True, o <= 0.0)
        sm.addGenConstrIndicator(b1, True, o <= oc)

        cost = cw * gp.quicksum(w[i] for i in range(P)) + co * o
        if P > 1:
            cost = cost + cg * (tv[P - 1] - tv[0] + w[P - 1] - w[0] -
                                gp.quicksum(dexpr[i] for i in range(P - 1)))
        mom = gp.quicksum(rp[i] * dA[i] + av[i] * dI[i] + lv[i] * q[i] + gv[i] * u[i]
                          for i in range(P))
        sm.setObjective(cost - mom - th, GRB.MAXIMIZE)
        sm.optimize()
        if sm.SolCount == 0:
            return None, 0.0
        viol = sm.ObjVal
        qv = [0.0] * P
        dAv = [0.0] * P
        dIv = [0.0] * P
        uv = [0.0] * P
        for i in range(P):
            p = order[i]
            qv[p] = 1.0 if q[i].X > 0.5 else 0.0
            dAv[p] = min(dAU[p], max(dAL[p], dA[i].X))
            dIv[p] = min(dIU[p], max(dIL[p], dI[i].X))
            uv[p] = min(uHi[p], max(uLo[p], u[i].X))
        return (qv, dAv, dIv, uv), viol

    # ---------------- main loop ----------------
    def extract():
        order = [0] * P
        for i in range(P):
            for p in range(P):
                if x[p][i].X > 0.5:
                    order[i] = p
                    break
        tv = [t[i].X for i in range(P)]
        rp = [rho[order[i]].X for i in range(P)]
        av = [alp[order[i]].X for i in range(P)]
        lv = [lam[order[i]].X for i in range(P)]
        gv = [gam[order[i]].X for i in range(P)]
        return order, tv, rp, av, lv, gv, theta.X

    best_obj = None
    best_sched = None
    tol = 1e-4

    while left() > 5.0:
        m.Params.TimeLimit = max(1.0, left() * 0.5)
        m.optimize()
        if m.SolCount == 0:
            break
        order, tv, rp, av, lv, gv, th = extract()
        obj = m.ObjVal
        res, viol = separate(order, tv, rp, av, lv, gv, th,
                             min(max(1.0, left() * 0.5), 60.0), tol)
        cur = obj + max(0.0, viol)
        if best_obj is None or cur < best_obj:
            best_obj = cur
            best_sched = (order[:], tv[:])
            if logger is not None:
                try:
                    logger.log(best_obj)
                except Exception:
                    pass
        if res is None or viol <= tol:
            break
        add_scenario(*res)

    if best_sched is None:
        order = list(range(P))
        tv = []
        c = 0.0
        for i in range(P):
            tv.append(min(L, c))
            p = order[i]
            c += muq[p] * muA[p] + (1.0 - muq[p]) * muI[p]
        best_sched = (order, tv)
        best_obj = None

    sol = {
        "cost_structure": cs_name,
        "appointment_order": [pid[p] for p in best_sched[0]],
        "appointment_times": [float(v) for v in best_sched[1]],
        "objective": best_obj,
    }
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()