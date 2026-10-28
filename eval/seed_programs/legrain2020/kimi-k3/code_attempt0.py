import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Nurse Scheduling Problem solver (MIP).")
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", required=True, help="Path to write solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None, help="Optional path to JSONL log file.")
    args = parser.parse_args()

    t_start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ----------------------------- instance -----------------------------
    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    N = int(inst["num_nurses"])
    W = int(inst["num_weeks"])
    D = int(inst["num_days"])
    S = int(inst["num_shifts"])
    K = int(inst["num_skills"])
    Wk = max(0, min(W, D // 7))

    skill_names = list(inst["skills"])
    kidx = {nm: i for i, nm in enumerate(skill_names)}
    shift_names = [sh["name"] for sh in inst["shifts"]]
    sidx = {nm: i for i, nm in enumerate(shift_names)}
    CSm = [int(sh["CS_minus"]) for sh in inst["shifts"]]
    CSp = [int(sh["CS_plus"]) for sh in inst["shifts"]]

    forbset = set()
    for pair in inst.get("forbidden_shift_successions", []) or []:
        if len(pair) >= 2 and pair[0] in sidx and pair[1] in sidx:
            forbset.add((sidx[pair[0]], sidx[pair[1]]))

    pw = inst["penalty_weights"]
    c1 = float(pw["c_S1"]);  c2a = float(pw["c_S2a"]); c2b = float(pw["c_S2b"])
    c3 = float(pw["c_S3"]);  c4 = float(pw["c_S4"]);   c5 = float(pw["c_S5"])
    c6 = float(pw["c_S6"]);  c7 = float(pw["c_S7"])

    nurses = inst["nurses"]
    nurse_ids = [nr["id"] for nr in nurses]
    nurse_names = [nr["name"] for nr in nurses]
    nskills, LM, LP, CDm, CDp, CRm, CRp, WEP, prefs = [], [], [], [], [], [], [], [], []
    for nr in nurses:
        nskills.append(sorted({kidx[k] for k in nr.get("skills", []) if k in kidx}))
        ct = nr["contract"]
        LM.append(int(ct["L_minus"])); LP.append(int(ct["L_plus"]))
        CDm.append(int(ct["CD_minus"])); CDp.append(int(ct["CD_plus"]))
        CRm.append(int(ct["CR_minus"])); CRp.append(int(ct["CR_plus"]))
        WEP.append(int(ct["WE_plus"]))
        ps = set()
        for p in nr.get("preferences", []) or []:
            dd = int(p["day"]); sh = p["shift"]
            if sh in sidx and 0 <= dd < D:
                ps.add((dd, sidx[sh]))
        prefs.append(ps)

    skill_nurses = [[] for _ in range(K)]
    for n in range(N):
        for k in nskills[n]:
            skill_nurses[k].append(n)

    minD = [[[0] * K for _ in range(S)] for _ in range(D)]
    optD = [[[0] * K for _ in range(S)] for _ in range(D)]
    demand = inst.get("demand", []) or []
    for d in range(min(D, len(demand))):
        entry = demand[d] or {}
        for sname, skmap in entry.items():
            if sname not in sidx:
                continue
            si = sidx[sname]
            for kname, vv in (skmap or {}).items():
                if kname not in kidx:
                    continue
                ki = kidx[kname]
                minD[d][si][ki] = int(vv.get("minimum", 0))
                optD[d][si][ki] = int(vv.get("optimal", 0))

    # ----------------------------- helpers -----------------------------
    def check_feasible(assign):
        cnt = [[[0] * K for _ in range(S)] for _ in range(D)]
        for n in range(N):
            for d, (s, k) in assign[n].items():
                cnt[d][s][k] += 1
        for d in range(D):
            for s in range(S):
                for k in range(K):
                    if cnt[d][s][k] < minD[d][s][k]:
                        return False
        for n in range(N):
            ad = assign[n]
            for d in range(D - 1):
                if d in ad and (d + 1) in ad and (ad[d][0], ad[d + 1][0]) in forbset:
                    return False
        return True

    def evaluate(assign):
        total = 0.0
        cnt = [[[0] * K for _ in range(S)] for _ in range(D)]
        for n in range(N):
            for d, (s, k) in assign[n].items():
                cnt[d][s][k] += 1
        if c1 > 0:
            acc = 0
            for d in range(D):
                for s in range(S):
                    for k in range(K):
                        df = optD[d][s][k] - cnt[d][s][k]
                        if df > 0:
                            acc += df
            total += c1 * acc
        for n in range(N):
            ad = assign[n]
            wseq = [1 if d in ad else 0 for d in range(D)]
            d = 0
            while d < D:
                if wseq[d] == 1:
                    a = d
                    while d < D and wseq[d] == 1:
                        d += 1
                    L = d - a
                    if L < CDm[n]:
                        total += c2a * (CDm[n] - L)
                    elif L > CDp[n]:
                        total += c2a * (L - CDp[n])
                else:
                    d += 1
            d = 0
            while d < D:
                if wseq[d] == 0:
                    a = d
                    while d < D and wseq[d] == 0:
                        d += 1
                    L = d - a
                    if L < CRm[n]:
                        total += c3 * (CRm[n] - L)
                    elif L > CRp[n]:
                        total += c3 * (L - CRp[n])
                else:
                    d += 1
            sseq = [ad[d][0] if d in ad else -1 for d in range(D)]
            d = 0
            while d < D:
                sv = sseq[d]
                if sv >= 0:
                    a = d
                    while d < D and sseq[d] == sv:
                        d += 1
                    L = d - a
                    if L < CSm[sv]:
                        total += c2b * (CSm[sv] - L)
                    elif L > CSp[sv]:
                        total += c2b * (L - CSp[sv])
                else:
                    d += 1
            if c4 > 0:
                for (dd, ss) in prefs[n]:
                    if sseq[dd] == ss:
                        total += c4
            wk_worked = 0
            for wk in range(Wk):
                sd = 7 * wk + 5
                ud = 7 * wk + 6
                cww = (1 if sd in ad else 0) + (1 if ud in ad else 0)
                if cww == 1:
                    total += c5
                if cww >= 1:
                    wk_worked += 1
            T = len(ad)
            if T < LM[n]:
                total += c6 * (LM[n] - T)
            elif T > LP[n]:
                total += c6 * (T - LP[n])
            if wk_worked > WEP[n]:
                total += c7 * (wk_worked - WEP[n])
        return total

    def assign_to_solution(assign, objv):
        schedule = []
        for n in range(N):
            arr = [{"day": d, "shift": shift_names[s], "skill": skill_names[k]}
                   for d, (s, k) in sorted(assign[n].items())]
            schedule.append({"nurse_id": nurse_ids[n],
                             "nurse_name": nurse_names[n],
                             "assignments": arr})
        return {"objective_value": float(objv), "schedule": schedule}

    # ------------------- constructive heuristic -------------------
    assign = [dict() for _ in range(N)]
    worked = [0] * N
    last = [-1] * N
    for d in range(D):
        dems = []
        for s in range(S):
            for k in range(K):
                m = minD[d][s][k]
                if m > 0:
                    dems.append((s, k, m))
        dems.sort(key=lambda t: (len(skill_nurses[t[1]]), t[0], t[1]))
        for (s, k, m) in dems:
            need = m
            cands = [n for n in skill_nurses[k]
                     if d not in assign[n] and (last[n] < 0 or (last[n], s) not in forbset)]
            cands.sort(key=lambda n: (worked[n] >= LM[n], worked[n], last[n] != s))
            for n in cands:
                if need <= 0:
                    break
                assign[n][d] = (s, k)
                worked[n] += 1
                need -= 1
            if need > 0:  # last resort (may violate a forbidden succession; MIP will repair)
                for n in skill_nurses[k]:
                    if need <= 0:
                        break
                    if d in assign[n]:
                        continue
                    assign[n][d] = (s, k)
                    worked[n] += 1
                    need -= 1
        for n in range(N):
            a = assign[n].get(d)
            last[n] = a[0] if a is not None else -1

    # fill towards total-day bounds while covering optimal demand
    cnt = [[[0] * K for _ in range(S)] for _ in range(D)]
    for n in range(N):
        for d, (s, k) in assign[n].items():
            cnt[d][s][k] += 1
    rem = [[[max(0, optD[d][s][k] - cnt[d][s][k]) for k in range(K)]
            for s in range(S)] for d in range(D)]
    for n in range(N):
        if not nskills[n]:
            continue
        while worked[n] < LP[n]:
            best = None
            bestscore = 0
            if c1 > 0:
                for d in range(D):
                    if d in assign[n]:
                        continue
                    prev = assign[n].get(d - 1) if d > 0 else None
                    nxt = assign[n].get(d + 1) if d < D - 1 else None
                    rday = rem[d]
                    for s in range(S):
                        if prev is not None and (prev[0], s) in forbset:
                            continue
                        if nxt is not None and (s, nxt[0]) in forbset:
                            continue
                        rs = rday[s]
                        for k in nskills[n]:
                            if rs[k] > 0:
                                score = (4 * rs[k]
                                         + (1 if prev is not None and prev[0] == s else 0)
                                         + (1 if nxt is not None and nxt[0] == s else 0))
                                if score > bestscore:
                                    bestscore = score
                                    best = (d, s, k)
            if best is not None:
                d, s, k = best
                assign[n][d] = (s, k)
                worked[n] += 1
                rem[d][s][k] -= 1
                continue
            if worked[n] < LM[n] and c6 > 0:
                placed = False
                for d in range(D):
                    if d in assign[n]:
                        continue
                    prev = assign[n].get(d - 1) if d > 0 else None
                    nxt = assign[n].get(d + 1) if d < D - 1 else None
                    for s in range(S):
                        if prev is not None and (prev[0], s) in forbset:
                            continue
                        if nxt is not None and (s, nxt[0]) in forbset:
                            continue
                        assign[n][d] = (s, nskills[n][0])
                        worked[n] += 1
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    continue
            break

    heur_obj = evaluate(assign)
    heur_feasible = check_feasible(assign)
    last_logged = {"obj": float("inf")}
    if logger is not None and heur_feasible:
        try:
            logger.log_solution(float(heur_obj), assign_to_solution(assign, heur_obj))
            last_logged["obj"] = float(heur_obj)
        except Exception:
            pass

    final_sol = None
    final_obj = None

    # ----------------------------- MIP -----------------------------
    if args.time_limit - (time.time() - t_start) > 3.0:
        try:
            model = gp.Model("nurse_scheduling")
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("MIPFocus", 1)

            x_index = [(n, d, s, k) for n in range(N) for d in range(D)
                       for s in range(S) for k in nskills[n]]
            x = model.addVars(x_index, vtype=GRB.BINARY, name="x")
            xvars = [x[t] for t in x_index]

            ws_idx = [(n, d, s) for n in range(N) for d in range(D) for s in range(S)]
            ws = model.addVars(ws_idx, lb=0.0, ub=1.0, name="ws")
            w_idx = [(n, d) for n in range(N) for d in range(D)]
            wv = model.addVars(w_idx, lb=0.0, ub=1.0, name="w")
            rv = model.addVars(w_idx, lb=0.0, ub=1.0, name="r")

            model.addConstrs((ws[n, d, s] == gp.quicksum(x[n, d, s, k] for k in nskills[n])
                              for (n, d, s) in ws_idx), name="ws_def")
            model.addConstrs((wv[n, d] == gp.quicksum(ws[n, d, s] for s in range(S))
                              for (n, d) in w_idx), name="w_def")
            model.addConstrs((rv[n, d] == 1.0 - wv[n, d] for (n, d) in w_idx), name="r_def")

            # demand: hard minimum coverage + soft optimal shortfall
            o_vars = []
            for d in range(D):
                for s in range(S):
                    for k in range(K):
                        mnq = minD[d][s][k]
                        opq = optD[d][s][k]
                        if mnq <= 0 and (c1 <= 0 or opq <= 0):
                            continue
                        expr = gp.quicksum(x[n, d, s, k] for n in skill_nurses[k])
                        if mnq > 0:
                            model.addConstr(expr >= mnq, name="dmin_%d_%d_%d" % (d, s, k))
                        if c1 > 0 and opq > 0:
                            ov = model.addVar(lb=0.0, name="o_%d_%d_%d" % (d, s, k))
                            model.addConstr(ov >= opq - expr, name="odef_%d_%d_%d" % (d, s, k))
                            o_vars.append(ov)

            # forbidden shift successions (hard)
            for (s1, s2) in sorted(forbset):
                model.addConstrs((ws[n, d, s1] + ws[n, d + 1, s2] <= 1.0
                                  for n in range(N) for d in range(D - 1)),
                                 name="forb_%d_%d" % (s1, s2))

            wL = [[wv[n, d] for d in range(D)] for n in range(N)]
            rL = [[rv[n, d] for d in range(D)] for n in range(N)]

            def add_stretch(seq, mn_list, mx_list, tag):
                """Linear per-day penalty for run lengths outside [mn, mx] (per nurse)."""
                objx = gp.LinExpr()
                exc_idx = []
                for n in range(N):
                    mx = mx_list[n]
                    if mx < D:
                        exc_idx.extend((n, d) for d in range(mx, D))
                if exc_idx:
                    exc = model.addVars(exc_idx, lb=0.0, ub=1.0, name=tag + "_x")
                    model.addConstrs(
                        (exc[n, d] >= gp.quicksum(seq[n][d - i] for i in range(mx_list[n] + 1)) - mx_list[n]
                         for (n, d) in exc_idx), name=tag + "_xc")
                    objx += exc.sum()
                need_min = [n for n in range(N) if mn_list[n] > 1]
                if need_min:
                    st_idx = [(n, d) for n in need_min for d in range(D)]
                    st = model.addVars(st_idx, lb=0.0, ub=1.0, name=tag + "_s")
                    en = model.addVars(st_idx, lb=0.0, ub=1.0, name=tag + "_e")
                    u_idx = [(n, d, i) for n in need_min for d in range(D)
                             for i in range(1, mn_list[n])]
                    u = model.addVars(u_idx, lb=0.0, ub=1.0, name=tag + "_u")
                    model.addConstrs((st[n, 0] >= seq[n][0] for n in need_min), name=tag + "_s0")
                    model.addConstrs((st[n, d] >= seq[n][d] - seq[n][d - 1]
                                      for n in need_min for d in range(1, D)), name=tag + "_sd")
                    model.addConstrs((en[n, D - 1] >= seq[n][D - 1] for n in need_min), name=tag + "_eL")
                    model.addConstrs((en[n, d] >= seq[n][d] - seq[n][d + 1]
                                      for n in need_min for d in range(D - 1)), name=tag + "_ed")
                    model.addConstrs(
                        (u[n, d, i] >= en[n, d]
                         + gp.quicksum(st[n, d - j] for j in range(i) if d - j >= 0) - 1.0
                         for (n, d, i) in u_idx), name=tag + "_uc")
                    objx += u.sum()
                return objx

            obj = gp.LinExpr()
            if o_vars:
                obj += c1 * gp.quicksum(o_vars)
            if c2a > 0:
                obj += c2a * add_stretch(wL, CDm, CDp, "wa")
            if c3 > 0:
                obj += c3 * add_stretch(rL, CRm, CRp, "re")
            if c2b > 0:
                for s in range(S):
                    if CSm[s] > 1 or CSp[s] < D:
                        seq_s = [[ws[n, d, s] for d in range(D)] for n in range(N)]
                        obj += c2b * add_stretch(seq_s, [CSm[s]] * N, [CSp[s]] * N, "sb%d" % s)
            if c4 > 0:
                for n in range(N):
                    for (dd, ss) in sorted(prefs[n]):
                        obj += c4 * ws[n, dd, ss]
            if c5 > 0 and Wk > 0:
                iw_idx = [(n, wk) for n in range(N) for wk in range(Wk)]
                iw = model.addVars(iw_idx, lb=0.0, ub=1.0, name="iw")
                model.addConstrs((iw[n, wk] >= wv[n, 7 * wk + 5] - wv[n, 7 * wk + 6]
                                  for (n, wk) in iw_idx), name="iw_a")
                model.addConstrs((iw[n, wk] >= wv[n, 7 * wk + 6] - wv[n, 7 * wk + 5]
                                  for (n, wk) in iw_idx), name="iw_b")
                obj += c5 * iw.sum()
            if c7 > 0 and Wk > 0:
                ww_idx = [(n, wk) for n in range(N) for wk in range(Wk)]
                ww = model.addVars(ww_idx, lb=0.0, ub=1.0, name="ww")
                model.addConstrs((ww[n, wk] >= wv[n, 7 * wk + 5] for (n, wk) in ww_idx), name="ww_a")
                model.addConstrs((ww[n, wk] >= wv[n, 7 * wk + 6] for (n, wk) in ww_idx), name="ww_b")
                ewe_idx = [n for n in range(N) if WEP[n] < Wk]
                if ewe_idx:
                    ewe = model.addVars(ewe_idx, lb=0.0, name="ewe")
                    model.addConstrs((ewe[n] >= gp.quicksum(ww[n, wk] for wk in range(Wk)) - WEP[n]
                                      for n in ewe_idx), name="ewe_c")
                    obj += c7 * ewe.sum()
            if c6 > 0:
                dl_idx = [n for n in range(N) if LM[n] > 0]
                du_idx = [n for n in range(N) if LP[n] < D]
                if dl_idx:
                    devL = model.addVars(dl_idx, lb=0.0, name="devL")
                    model.addConstrs((devL[n] >= LM[n] - gp.quicksum(wv[n, d] for d in range(D))
                                      for n in dl_idx), name="devL_c")
                    obj += c6 * devL.sum()
                if du_idx:
                    devU = model.addVars(du_idx, lb=0.0, name="devU")
                    model.addConstrs((devU[n] >= gp.quicksum(wv[n, d] for d in range(D)) - LP[n]
                                      for n in du_idx), name="devU_c")
                    obj += c6 * devU.sum()

            model.setObjective(obj, GRB.MINIMIZE)

            # MIP start from the heuristic schedule
            try:
                model.setAttr("Start", xvars,
                              [1.0 if assign[n].get(d) == (s, k) else 0.0
                               for (n, d, s, k) in x_index])
                model.setAttr("Start", [wv[n, d] for (n, d) in w_idx],
                              [1.0 if d in assign[n] else 0.0 for (n, d) in w_idx])
                model.setAttr("Start", [ws[n, d, s] for (n, d, s) in ws_idx],
                              [1.0 if assign[n].get(d, (-1, -1))[0] == s else 0.0
                               for (n, d, s) in ws_idx])
                model.setAttr("Start", [rv[n, d] for (n, d) in w_idx],
                              [0.0 if d in assign[n] else 1.0 for (n, d) in w_idx])
            except Exception:
                pass

            def vals_to_solution(vals, objv):
                per = [[] for _ in range(N)]
                for i, v in enumerate(vals):
                    if v > 0.5:
                        n, d, s, k = x_index[i]
                        per[n].append((d, s, k))
                schedule = []
                for n in range(N):
                    per[n].sort()
                    schedule.append({
                        "nurse_id": nurse_ids[n],
                        "nurse_name": nurse_names[n],
                        "assignments": [{"day": d, "shift": shift_names[s], "skill": skill_names[k]}
                                        for (d, s, k) in per[n]]
                    })
                return {"objective_value": float(objv), "schedule": schedule}

            def cb(mdl, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        objv = float(mdl.cbGet(GRB.Callback.MIPSOL_OBJ))
                        if logger is not None and objv < last_logged["obj"] - 1e-6:
                            vals = mdl.cbGetSolution(xvars)
                            logger.log_solution(objv, vals_to_solution(vals, objv))
                            last_logged["obj"] = objv
                    except Exception:
                        pass

            rem = args.time_limit - (time.time() - t_start) - 1.0
            if rem > 0.5:
                model.setParam("TimeLimit", max(1.0, rem))
                if logger is not None:
                    model.optimize(cb)
                else:
                    model.optimize()

            if model.SolCount > 0:
                vals = model.getAttr("X", xvars)
                final_obj = float(model.ObjVal)
                final_sol = vals_to_solution(vals, final_obj)
        except Exception:
            final_sol = None
            final_obj = None

    if final_sol is None or (heur_feasible and heur_obj < final_obj - 1e-9):
        final_obj = float(heur_obj)
        final_sol = assign_to_solution(assign, heur_obj)

    if logger is not None and final_obj < last_logged["obj"] - 1e-6:
        try:
            logger.log_solution(final_obj, final_sol)
            last_logged["obj"] = final_obj
        except Exception:
            pass

    final_sol["objective_value"] = float(final_obj)
    with open(args.solution_path, "w") as fh:
        json.dump(final_sol, fh)


if __name__ == "__main__":
    main()