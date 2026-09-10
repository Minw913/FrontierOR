import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t0 = time.time()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    patients = data["patients"]
    hospitals = data["hospitals"]
    scenarios = sorted(data["scenarios"], key=lambda s: s["scenario_id"])
    P = len(patients)
    S = len(scenarios)
    dur = [list(map(float, sc["surgery_durations_minutes"])) for sc in scenarios]

    HD = []
    B, F, G, R = {}, {}, {}, {}
    daylabel = {}
    for h in hospitals:
        hid = h["hospital_id"]
        R[hid] = h.get("num_ors") or data["num_ors_per_hospital"]
        for dd in h["days"]:
            d = dd["day_id"]
            daylabel[d] = dd["day_label"]
            HD.append((hid, d))
            B[(hid, d)] = float(dd["B_hd"])
            F[(hid, d)] = float(dd["F_hd"])
            G[(hid, d)] = float(dd["G_hd"])

    pid = [p["patient_id"] for p in patients]
    mand = [bool(p["is_mandatory"]) for p in patients]
    cun = [float(p["c_unsched"]) for p in patients]
    ccan = [float(p["c_cancel"]) for p in patients]
    csched = []
    for p in patients:
        cs = p["c_sched_per_day"]
        row = {}
        for d, lab in daylabel.items():
            if str(lab) in cs:
                row[d] = float(cs[str(lab)])
            elif str(d) in cs:
                row[d] = float(cs[str(d)])
            else:
                row[d] = float(list(cs.values())[0])
        csched.append(row)

    ed = [sum(dur[s][i] for s in range(S)) / S for i in range(P)]

    # ---------------- Greedy initial feasible solution ----------------
    open_suite = set()
    open_count = {}
    rem = {}
    assign = {}

    def open_new_or(h, d):
        r = open_count.get((h, d), 0)
        open_count[(h, d)] = r + 1
        rem[(h, d, r)] = B[(h, d)]
        open_suite.add((h, d))
        return r

    def find_best(i):
        best = None
        for (h, d, r), cap in rem.items():
            if cap >= ed[i]:
                c = csched[i][d]
                if best is None or c < best[0]:
                    best = (c, "e", (h, d, r))
        for (h, d) in HD:
            if open_count.get((h, d), 0) >= R[h] or B[(h, d)] < ed[i]:
                continue
            c = F[(h, d)] + (0.0 if (h, d) in open_suite else G[(h, d)]) + csched[i][d]
            if best is None or c < best[0]:
                best = (c, "n", (h, d))
        return best

    for i in sorted([i for i in range(P) if mand[i]], key=lambda i: -ed[i]):
        best = find_best(i)
        if best is None:
            if rem:
                cand = max(rem.items(), key=lambda kv: kv[1])[0]
            else:
                hd = min(HD, key=lambda hd: F[hd] + (0.0 if hd in open_suite else G[hd]))
                r = open_new_or(*hd)
                cand = (hd[0], hd[1], r)
            best = (0.0, "e", cand)
        if best[1] == "n":
            h, d = best[2]
            r = open_new_or(h, d)
            slot = (h, d, r)
        else:
            slot = best[2]
        assign[i] = slot
        rem[slot] -= ed[i]

    for i in sorted([i for i in range(P) if not mand[i]], key=lambda i: -cun[i]):
        best = find_best(i)
        if best is not None and best[0] - cun[i] < -1e-9:
            if best[1] == "n":
                h, d = best[2]
                r = open_new_or(h, d)
                slot = (h, d, r)
            else:
                slot = best[2]
            assign[i] = slot
            rem[slot] -= ed[i]

    def cancellations(asg):
        ors = {}
        for i, slot in asg.items():
            ors.setdefault(slot, []).append(i)
        cost = 0.0
        zset = set()
        for (h, d, r), plist in ors.items():
            cap = B[(h, d)]
            for s in range(S):
                tot = sum(dur[s][i] for i in plist)
                if tot <= cap + 1e-9:
                    continue
                order = sorted(plist, key=lambda i: ccan[i] / max(dur[s][i], 1e-6))
                for i in order:
                    if tot <= cap + 1e-9:
                        break
                    tot -= dur[s][i]
                    cost += ccan[i]
                    zset.add((i, h, d, r, s))
        return cost / S, zset

    ors_open = set(rem.keys())
    canc_cost, zset = cancellations(assign)
    g_obj = (
        sum(G[hd] for hd in open_suite)
        + sum(F[(h, d)] for (h, d, r) in ors_open)
        + sum(csched[i][assign[i][1]] for i in assign)
        + sum(cun[i] for i in range(P) if not mand[i] and i not in assign)
        + canc_cost
    )

    def make_sol(objval, suites, ors_o, asg, postponed):
        u_d = {f"{h},{d}": (1 if (h, d) in suites else 0) for (h, d) in HD}
        y_d = {}
        for (h, d) in HD:
            for r in range(R[h]):
                y_d[f"{h},{d},{r}"] = 1 if (h, d, r) in ors_o else 0
        x_d = {}
        for i, (h, d, r) in asg.items():
            x_d[f"{h},{d},{pid[i]},{r}"] = 1
        w_d = {}
        for i in range(P):
            if not mand[i]:
                w_d[str(pid[i])] = 1 if i in postponed else 0
        return {"objective_value": float(objval), "u": u_d, "y": y_d, "x": x_d, "w": w_d}

    g_post = {i for i in range(P) if not mand[i] and i not in assign}
    greedy_sol = make_sol(g_obj, open_suite, ors_open, assign, g_post)
    best = [g_obj]
    best_sol = [greedy_sol]
    if logger:
        logger.log_solution(g_obj, greedy_sol)
    with open(args.solution_path, "w") as f:
        json.dump(greedy_sol, f)

    remaining = args.time_limit - (time.time() - t0) - 2.0
    if remaining < 3.0:
        return

    # ---------------- Extensive-form MIP ----------------
    m = gp.Model("or_scheduling")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    ykeys = [(h, d, r) for (h, d) in HD for r in range(R[h])]
    xkeys = [(i, h, d, r) for i in range(P) for (h, d, r) in ykeys]
    wkeys = [i for i in range(P) if not mand[i]]
    zkeys = [(i, h, d, r, s) for (i, h, d, r) in xkeys for s in range(S)]

    u = m.addVars(HD, vtype=GRB.BINARY, obj=[G[k] for k in HD], name="u")
    y = m.addVars(ykeys, vtype=GRB.BINARY, obj=[F[(h, d)] for (h, d, r) in ykeys], name="y")
    x = m.addVars(xkeys, vtype=GRB.BINARY, obj=[csched[i][d] for (i, h, d, r) in xkeys], name="x")
    w = m.addVars(wkeys, vtype=GRB.BINARY, obj=[cun[i] for i in wkeys], name="w")
    z = m.addVars(zkeys, vtype=GRB.BINARY, obj=[ccan[i] / S for (i, h, d, r, s) in zkeys], name="z")
    m.ModelSense = GRB.MINIMIZE

    for i in range(P):
        expr = gp.quicksum(x[i, h, d, r] for (h, d, r) in ykeys)
        if mand[i]:
            m.addConstr(expr == 1)
        else:
            m.addConstr(expr + w[i] == 1)

    m.addConstrs((x[i, h, d, r] <= y[h, d, r] for (i, h, d, r) in xkeys))
    m.addConstrs((y[h, d, r] <= u[h, d] for (h, d, r) in ykeys))
    m.addConstrs((z[i, h, d, r, s] <= x[i, h, d, r] for (i, h, d, r, s) in zkeys))

    negdur = [[-dur[s][i] for i in range(P)] for s in range(S)]
    for (h, d, r) in ykeys:
        xv = [x[i, h, d, r] for i in range(P)]
        for s in range(S):
            expr = gp.LinExpr(dur[s], xv)
            expr.addTerms(negdur[s], [z[i, h, d, r, s] for i in range(P)])
            m.addConstr(expr <= B[(h, d)] * y[h, d, r])

    for (h, d) in HD:
        for r in range(R[h] - 1):
            m.addConstr(y[h, d, r] >= y[h, d, r + 1])

    # MIP start from greedy
    for v in m.getVars():
        v.Start = 0.0
    for hd in open_suite:
        u[hd].Start = 1.0
    for k in ors_open:
        y[k].Start = 1.0
    for i, (h, d, r) in assign.items():
        x[i, h, d, r].Start = 1.0
    for i in g_post:
        w[i].Start = 1.0
    for k in zset:
        z[k].Start = 1.0

    ulist = [u[k] for k in HD]
    ylist = [y[k] for k in ykeys]
    xlist = [x[k] for k in xkeys]
    wlist = [w[k] for k in wkeys]

    def sol_from_vals(objv, uv, yv, xv, wv):
        suites = {HD[k] for k, v in enumerate(uv) if v > 0.5}
        ors_o = {ykeys[k] for k, v in enumerate(yv) if v > 0.5}
        asg = {}
        for k, v in enumerate(xv):
            if v > 0.5:
                i, h, d, r = xkeys[k]
                asg[i] = (h, d, r)
        postponed = {wkeys[k] for k, v in enumerate(wv) if v > 0.5}
        return make_sol(objv, suites, ors_o, asg, postponed)

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objv >= best[0] - 1e-6:
                return
            uv = model.cbGetSolution(ulist)
            yv = model.cbGetSolution(ylist)
            xv = model.cbGetSolution(xlist)
            wv = model.cbGetSolution(wlist)
            sol = sol_from_vals(objv, uv, yv, xv, wv)
            best[0] = objv
            best_sol[0] = sol
            if logger:
                logger.log_solution(objv, sol)

    m.Params.TimeLimit = max(1.0, args.time_limit - (time.time() - t0) - 1.5)
    try:
        m.optimize(cb)
    except Exception:
        pass

    try:
        if m.SolCount > 0:
            objv = m.ObjVal
            if objv < best[0] - 1e-9:
                uv = [v.X for v in ulist]
                yv = [v.X for v in ylist]
                xv = [v.X for v in xlist]
                wv = [v.X for v in wlist]
                sol = sol_from_vals(objv, uv, yv, xv, wv)
                best[0] = objv
                best_sol[0] = sol
                if logger:
                    logger.log_solution(objv, sol)
    except Exception:
        pass

    with open(args.solution_path, "w") as f:
        json.dump(best_sol[0], f)


if __name__ == "__main__":
    main()