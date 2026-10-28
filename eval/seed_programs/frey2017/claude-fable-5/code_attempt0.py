import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t_start = time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as fh:
        inst = json.load(fh)

    T = int(inst["planning_horizon"]["num_periods"])
    target = float(inst["objective"]["target_utilization"])
    Vs = [float(v) for v in inst["objective"]["threshold_values"]]
    pens = [float(p) for p in inst["objective"]["penalties"]]
    nL = len(Vs)
    cs = inst["central_storage"]
    STCAP = float(cs["capacity_bags"])
    DEPR = max(1, int(cs["depletion_rate_bags_per_period"]))
    DEPOFF = int(cs["depletion_offset_periods"])
    WRATE = max(1, int(inst["working_rate_bags_per_period"]))
    TRANS = int(inst.get("baggage_transfer_time_periods", 0) or 0)

    carousels = inst.get("carousels", [])
    nC = len(carousels)
    flights_raw = inst.get("flights", [])
    nF = len(flights_raw)

    def write_out(sol):
        with open(args.solution_path, "w") as fo:
            json.dump(sol, fo, indent=1)

    if nF == 0 or nC == 0:
        write_out({"objective_value": 0.0, "assignments": [], "utilization_violations": []})
        if logger:
            logger.log_solution(0.0, {"objective_value": 0.0, "assignments": [], "utilization_violations": []})
        return

    # ------------- horizon / peaks -------------
    max_end = max(int(f["end_period"]) for f in flights_raw)
    peaks0 = [(int(p["peak_id"]), int(p["start_period"]), int(p["end_period"]))
              for p in inst.get("peak_intervals", [])]
    max_pe = max([pe for _, _, pe in peaks0], default=0)
    T_arr = max(T, max_end + 2, max_pe + 2) + 2
    peaks = []
    for pid, ps, pe in peaks0:
        ps2 = max(0, ps)
        pe2 = min(T_arr - 1, pe)
        if ps2 <= pe2:
            peaks.append((pid, ps2, pe2))
    nP = len(peaks)
    peak_mask = np.zeros(T_arr, dtype=bool)
    for _, ps, pe in peaks:
        peak_mask[ps:pe + 1] = True

    # ------------- carousels -------------
    Kws_arr = np.array([max(1, int(c["Kws"])) for c in carousels], dtype=float)
    Kpp_arr = np.array([max(1, int(c["Kpp"])) for c in carousels], dtype=float)
    Kcb_arr = np.array([max(1, int(c["Kcb"])) for c in carousels], dtype=float)
    ctype_info = {}
    for key, v in (inst.get("carousel_types", {}) or {}).items():
        try:
            ctype_info[int(key)] = v
        except Exception:
            pass
    type_carousels = defaultdict(list)
    for ci, c in enumerate(carousels):
        r = int(c["type_r"])
        type_carousels[r].append(ci)
        if r not in ctype_info:
            ctype_info[r] = {"Kpp": c["Kpp"], "Kws": c["Kws"],
                             "Kppws": c["Kppws"], "Kcb": c["Kcb"]}

    def level_for(over):
        for l in range(nL):
            if Vs[l] >= over - 1e-9:
                return l
        return nL - 1

    def pen_of(over):
        if over <= 1e-9 or nL == 0:
            return 0.0
        return pens[level_for(over)]

    # ------------- flights preprocess -------------
    F = []
    for fl in flights_raw:
        end = int(fl["end_period"])
        es = int(fl["earliest_start_period"])
        ls = int(fl["latest_start_period"])
        if ls > end:
            ls = end
        if es < 0:
            es = 0
        if es > ls:
            es = ls
        prof = np.array(fl["arrival_profile"], dtype=np.int64)
        p0 = min(int(fl["earliest_start_period"]), es)
        if p0 < 0:
            p0 = 0
        nc = int(fl["num_containers"])
        bags = int(fl.get("expected_baggage", int(prof.sum())))
        tb = fl.get("working_station_bounds_by_carousel_type") or {}
        compat_types = {}
        for r in type_carousels:
            ti = ctype_info[r]
            b = tb.get(str(r)) or tb.get(r)
            if b:
                wlo, whi = int(b["Wmin"]), int(b["Wmax"])
            else:
                kpw = max(1, int(ti["Kppws"]))
                wlo = max(1, nc // kpw)
                rem = nc % kpw
                whi = int(math.ceil(nc / kpw)) + (1 if rem > 1 else 0)
            whi = min(max(whi, wlo), int(ti["Kws"]))
            if wlo > whi:
                continue
            compat_types[r] = (wlo, whi)
        if not compat_types:
            for r in type_carousels:
                ti = ctype_info[r]
                w_f = max(1, min(int(ti["Kws"]),
                                 int(math.ceil(nc / max(1, int(ti["Kppws"]))))))
                compat_types[r] = (w_f, w_f)
        # w candidates per type (at most 3 each)
        wset = set()
        for r, (wlo, whi) in compat_types.items():
            if whi - wlo <= 2:
                wset.update(range(wlo, whi + 1))
            else:
                wset.update({wlo, (wlo + whi) // 2, whi})
        compat = {}
        for w in sorted(wset):
            lst = []
            for r, (wlo, whi) in compat_types.items():
                if wlo <= w <= whi:
                    lst.extend(type_carousels[r])
            if lst:
                compat[w] = sorted(lst)
        if not compat:
            compat = {1: list(range(nC))}
        F.append({"fid": int(fl["flight_id"]), "end": end, "es": es, "ls": ls,
                  "prof": prof, "p0": p0, "nc": nc, "bags": bags,
                  "compat": compat, "wcands": sorted(compat.keys())})

    # ------------- simulation -------------
    def simulate(prof, p0, end, w, s, d):
        rate = w * WRATE
        S = 0
        B = 0
        npf = len(prof)
        u = np.zeros(end - s + 1, dtype=float)
        Sp = np.zeros(end - p0 + 1, dtype=float)
        dep_fut = {}
        dep_lim = end - TRANS
        for t in range(p0, end + 1):
            a = int(prof[t - p0]) if 0 <= t - p0 < npf else 0
            if t < s:
                S += a
                Sp[t - p0] = S
                continue
            if t >= d and t <= dep_lim and S > 0:
                dep = DEPR if S >= DEPR else S
                S -= dep
                ta = t + TRANS
                if ta <= end:
                    dep_fut[ta] = dep_fut.get(ta, 0) + dep
            inflow = a + dep_fut.pop(t, 0)
            B += inflow - rate
            if B < 0:
                B = 0
            u[t - s] = B
            Sp[t - p0] = S
        extra = int(prof[end - p0 + 1:].sum()) if end - p0 + 1 < npf else 0
        leftover = B + S + extra + sum(dep_fut.values())
        nz = np.nonzero(Sp)[0]
        if len(nz):
            st0 = p0 + int(nz[0])
            sarr = Sp[nz[0]:nz[-1] + 1].copy()
        else:
            st0, sarr = None, None
        return u, st0, sarr, float(leftover)

    def d_cands(prof, p0, s, end):
        k = max(0, min(s - p0, len(prof)))
        pre = int(prof[:k].sum())
        if pre <= 0:
            return [s]
        need = int(math.ceil(pre / DEPR))
        dl = min(end - DEPOFF, end - TRANS - need + 1)
        if dl <= s:
            return [s]
        return [s, dl]

    # ------------- option generation -------------
    sim_budget = 60000
    per_flight_allow = max(8, sim_budget // nF)
    for f in F:
        nw = max(1, len(f["wcands"]))
        K0 = max(4, per_flight_allow // (nw * 2))
        K0 = min(K0, 60)
        srange = list(range(f["es"], f["ls"] + 1))
        if len(srange) > K0:
            idx = np.unique(np.round(np.linspace(0, len(srange) - 1, K0)).astype(int))
            srange = [srange[i] for i in idx]
        opts_all = []
        seen = set()
        for w in f["wcands"]:
            for s in srange:
                for d in d_cands(f["prof"], f["p0"], s, f["end"]):
                    key = (w, s, d)
                    if key in seen:
                        continue
                    seen.add(key)
                    u, st0, sarr, lo = simulate(f["prof"], f["p0"], f["end"], w, s, d)
                    opts_all.append({"w": w, "s": s, "d": d, "u": u,
                                     "st0": st0, "sarr": sarr, "leftover": lo})
        feas = [o for o in opts_all if o["leftover"] <= 1e-9]
        if feas:
            opts = feas
        else:
            mn = min(o["leftover"] for o in opts_all)
            opts = [o for o in opts_all if o["leftover"] <= mn + 1e-9]
        opts.sort(key=lambda o: (o["s"], o["w"], o["d"]))
        f["opts"] = opts
        f["svals"] = sorted({o["s"] for o in opts})

    # ------------- budget-driven filtering -------------
    if args.time_limit <= 60:
        BUDGET = 25000
    elif args.time_limit <= 180:
        BUDGET = 50000
    else:
        BUDGET = 90000

    def sel_s(svals, K):
        if len(svals) <= K:
            return set(svals)
        idx = np.unique(np.round(np.linspace(0, len(svals) - 1, K)).astype(int))
        return {svals[i] for i in idx}

    def build_filtered(K, use_dl, wr):
        total = 0
        per = []
        for f in F:
            ss = sel_s(f["svals"], K)
            wall = f["wcands"]
            if wr >= 2 and len(wall) > 1:
                wsel = {max(wall)}
            elif wr == 1 and len(wall) > 2:
                wsel = {min(wall), max(wall)}
            else:
                wsel = set(wall)
            lst = [o for o in f["opts"]
                   if o["s"] in ss and o["w"] in wsel and (use_dl or o["d"] == o["s"])]
            if not lst:
                lst = [f["opts"][0]]
            per.append(lst)
            total += sum(len(f["compat"][o["w"]]) for o in lst)
        return per, total

    cfgs = [(60, True, 0), (40, True, 0), (28, True, 0), (20, True, 0),
            (14, True, 0), (10, True, 0), (8, True, 0), (6, True, 0),
            (6, True, 1), (5, True, 1), (4, True, 1), (4, False, 1),
            (3, False, 1), (2, False, 1), (2, False, 2), (1, False, 2)]
    chosen_per = None
    per = None
    for cfg in cfgs:
        per, tot = build_filtered(*cfg)
        if tot <= BUDGET:
            chosen_per = per
            break
    if chosen_per is None:
        chosen_per = per

    # ------------- variable descriptors -------------
    xinfo = []
    f_var_ids = []
    for fi, f in enumerate(F):
        ids = []
        for o in chosen_per[fi]:
            for ci in f["compat"][o["w"]]:
                ids.append(len(xinfo))
                xinfo.append((fi, o, ci))
        if not ids:
            u, st0, sarr, lo = simulate(f["prof"], f["p0"], f["end"], 1, f["es"], f["es"])
            o = {"w": 1, "s": f["es"], "d": f["es"], "u": u, "st0": st0,
                 "sarr": sarr, "leftover": lo}
            ids.append(len(xinfo))
            xinfo.append((fi, o, 0))
        f_var_ids.append(np.array(ids, dtype=np.int64))
    n_x = len(xinfo)

    # ------------- evaluation of a full assignment -------------
    def evaluate_full(assign_ids):
        belt = np.zeros((nC, T_arr))
        ws_u = np.zeros((nC, T_arr))
        pp_u = np.zeros((nC, T_arr))
        st_u = np.zeros(T_arr)
        for fi in range(nF):
            _, o, ci = xinfo[assign_ids[fi]]
            s, e = o["s"], F[fi]["end"]
            belt[ci, s:e + 1] += o["u"]
            ws_u[ci, s:e + 1] += o["w"]
            pp_u[ci, s:e + 1] += F[fi]["nc"]
            if o["sarr"] is not None:
                st_u[o["st0"]:o["st0"] + len(o["sarr"])] += o["sarr"]
        hard = float(np.maximum(ws_u - Kws_arr[:, None], 0).sum()
                     + np.maximum(pp_u - Kpp_arr[:, None], 0).sum()
                     + np.maximum(st_u - STCAP, 0).sum())
        tot = 0.0
        viols = []
        for ci in range(nC):
            util = belt[ci] / Kcb_arr[ci]
            for pi, (pid, ps, pe) in enumerate(peaks):
                mu = float(util[ps:pe + 1].max())
                over = mu - target
                if over > 1e-9 and nL > 0:
                    l = level_for(over)
                    tot += pens[l]
                    viols.append({"carousel_id": int(carousels[ci]["carousel_id"]),
                                  "peak_interval": int(pid),
                                  "threshold_level": int(l),
                                  "threshold_value": float(Vs[l])})
        return tot, viols, hard

    def build_solution(assign_ids, obj, viols):
        assignments = []
        for fi in range(nF):
            _, o, ci = xinfo[assign_ids[fi]]
            c = carousels[ci]
            assignments.append({"flight_id": int(F[fi]["fid"]),
                                "carousel_id": int(c["carousel_id"]),
                                "carousel_type": int(c["type_r"]),
                                "working_stations": int(o["w"]),
                                "handling_start_period": int(o["s"]),
                                "depletion_start_period": int(o["d"]),
                                "end_period": int(F[fi]["end"])})
        return {"objective_value": float(obj), "assignments": assignments,
                "utilization_violations": viols}

    # ------------- greedy construction -------------
    order = sorted(range(nF), key=lambda i: (F[i]["ls"] - F[i]["es"], -F[i]["bags"], i))
    ws_use = np.zeros((nC, T_arr))
    pp_use = np.zeros((nC, T_arr))
    st_use = np.zeros(T_arr)
    belt = np.zeros((nC, T_arr))
    maxu = np.zeros((nC, max(nP, 1)))
    greedy_choice = [None] * nF
    for fi in order:
        f = F[fi]
        best = None
        best_inf = None
        for vid in f_var_ids[fi]:
            _, o, ci = xinfo[vid]
            s, e, w = o["s"], f["end"], o["w"]
            vio = max(0.0, float((ws_use[ci, s:e + 1] + w).max()) - Kws_arr[ci])
            vio += max(0.0, float((pp_use[ci, s:e + 1] + f["nc"]).max()) - Kpp_arr[ci])
            if o["sarr"] is not None:
                st0 = o["st0"]
                L = len(o["sarr"])
                vio += max(0.0, float((st_use[st0:st0 + L] + o["sarr"]).max()) - STCAP)
            newu = (belt[ci, s:e + 1] + o["u"]) / Kcb_arr[ci]
            dpen = 0.0
            addu = 0.0
            for pi, (pid, ps, pe) in enumerate(peaks):
                lo = max(ps, s)
                hi = min(pe, e)
                if lo > hi:
                    continue
                seg = newu[lo - s:hi - s + 1]
                mu = float(seg.max())
                dpen += pen_of(max(mu, maxu[ci, pi]) - target) - pen_of(maxu[ci, pi] - target)
                addu += float(seg.sum())
            if vio <= 1e-9:
                key = (dpen, addu, o["leftover"], int(vid))
                if best is None or key < best[0]:
                    best = (key, int(vid))
            else:
                key2 = (vio, dpen, addu, int(vid))
                if best_inf is None or key2 < best_inf[0]:
                    best_inf = (key2, int(vid))
        chosen = best[1] if best is not None else best_inf[1]
        greedy_choice[fi] = chosen
        _, o, ci = xinfo[chosen]
        s, e = o["s"], f["end"]
        ws_use[ci, s:e + 1] += o["w"]
        pp_use[ci, s:e + 1] += f["nc"]
        if o["sarr"] is not None:
            st_use[o["st0"]:o["st0"] + len(o["sarr"])] += o["sarr"]
        belt[ci, s:e + 1] += o["u"]
        for pi, (pid, ps, pe) in enumerate(peaks):
            lo = max(ps, s)
            hi = min(pe, e)
            if lo > hi:
                continue
            mu = float((belt[ci, lo:hi + 1] / Kcb_arr[ci]).max())
            if mu > maxu[ci, pi]:
                maxu[ci, pi] = mu

    g_obj, g_viols, g_hard = evaluate_full(greedy_choice)
    best_state = {"key": (g_hard > 1e-6, g_obj), "assign": list(greedy_choice),
                  "obj": g_obj, "viols": g_viols}
    if logger:
        logger.log_solution(g_obj, build_solution(greedy_choice, g_obj, g_viols))

    # ------------- MIP -------------
    remaining = args.time_limit - (time.time() - t_start) - 6
    try:
        if remaining < 3:
            raise RuntimeError("no time for MIP")
        import gurobipy as gp
        from gurobipy import GRB
        import scipy.sparse as sp

        r_ws, c_ws, v_ws = [], [], []
        r_pp, c_pp, v_pp = [], [], []
        r_st, c_st, v_st = [], [], []
        r_u, c_u, v_u = [], [], []
        r_eq, c_eq = [], []
        for i, (fi, o, ci) in enumerate(xinfo):
            s, e = o["s"], F[fi]["end"]
            L = e - s + 1
            tarr = np.arange(s, e + 1)
            base = ci * T_arr
            r_ws.append(base + tarr)
            c_ws.append(np.full(L, i))
            v_ws.append(np.full(L, float(o["w"])))
            r_pp.append(base + tarr)
            c_pp.append(np.full(L, i))
            v_pp.append(np.full(L, float(F[fi]["nc"])))
            if o["sarr"] is not None:
                sa = o["sarr"]
                nz = sa > 0
                if nz.any():
                    ts = o["st0"] + np.nonzero(nz)[0]
                    r_st.append(ts)
                    c_st.append(np.full(len(ts), i))
                    v_st.append(sa[nz])
            m2 = (o["u"] > 0) & peak_mask[s:e + 1]
            if m2.any():
                r_u.append(base + tarr[m2])
                c_u.append(np.full(int(m2.sum()), i))
                v_u.append(o["u"][m2])
            r_eq.append(fi)
            c_eq.append(i)

        def mk(rows, cols, vals, nr):
            if rows:
                return sp.coo_matrix((np.concatenate(vals),
                                      (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(nr, n_x)).tocsr()
            return sp.csr_matrix((nr, n_x))

        A_ws = mk(r_ws, c_ws, v_ws, nC * T_arr)
        A_pp = mk(r_pp, c_pp, v_pp, nC * T_arr)
        A_st = mk(r_st, c_st, v_st, T_arr)
        A_u = mk(r_u, c_u, v_u, nC * T_arr)
        A_eq = sp.coo_matrix((np.ones(n_x), (np.array(r_eq), np.array(c_eq))),
                             shape=(nF, n_x)).tocsr()

        keep_ws = np.diff(A_ws.indptr) > 0
        keep_pp = np.diff(A_pp.indptr) > 0
        keep_st = np.diff(A_st.indptr) > 0
        keep_u = np.diff(A_u.indptr) > 0
        A_ws = A_ws[keep_ws]
        A_pp = A_pp[keep_pp]
        A_st = A_st[keep_st]
        b_ws = np.repeat(Kws_arr, T_arr)[keep_ws]
        b_pp = np.repeat(Kpp_arr, T_arr)[keep_pp]
        b_st = np.full(T_arr, STCAP)[keep_st]

        n_y = nC * nP * nL
        ry, cy, vy = [], [], []
        for ci in range(nC):
            for pi, (pid, ps, pe) in enumerate(peaks):
                rows0 = ci * T_arr + np.arange(ps, pe + 1)
                for l in range(nL):
                    ry.append(rows0)
                    cy.append(np.full(len(rows0), ci * nP * nL + pi * nL + l))
                    vy.append(np.full(len(rows0), -Kcb_arr[ci] * Vs[l]))
        if ry and n_y > 0:
            Y_full = sp.coo_matrix((np.concatenate(vy),
                                    (np.concatenate(ry), np.concatenate(cy))),
                                   shape=(nC * T_arr, n_y)).tocsr()
        else:
            Y_full = sp.csr_matrix((nC * T_arr, max(n_y, 1)))
        A_u2 = A_u[keep_u]
        Y_u = Y_full[keep_u]
        b_u = (target * np.repeat(Kcb_arr, T_arr))[keep_u]

        m = gp.Model("bhs")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.MIPFocus = 1
        m.Params.TimeLimit = max(3.0, args.time_limit - (time.time() - t_start) - 5)

        x = m.addMVar(n_x, vtype=GRB.BINARY, name="x")
        max_pen = max(pens) if pens else 1.0
        Mslack = max_pen * 10.0 + 1e6
        obj_expr = 0.0
        m.addConstr(A_eq @ x == np.ones(nF))
        if A_ws.shape[0] > 0:
            sl1 = m.addMVar(A_ws.shape[0], lb=0.0)
            m.addConstr(A_ws @ x - sl1 <= b_ws)
            obj_expr = obj_expr + Mslack * sl1.sum()
        if A_pp.shape[0] > 0:
            sl2 = m.addMVar(A_pp.shape[0], lb=0.0)
            m.addConstr(A_pp @ x - sl2 <= b_pp)
            obj_expr = obj_expr + Mslack * sl2.sum()
        if A_st.shape[0] > 0:
            sl3 = m.addMVar(A_st.shape[0], lb=0.0)
            m.addConstr(A_st @ x - sl3 <= b_st)
            obj_expr = obj_expr + Mslack * sl3.sum()
        y = None
        if n_y > 0:
            y = m.addMVar(n_y, vtype=GRB.BINARY, name="y")
            pen_vec = np.tile(np.array(pens, dtype=float), nC * nP)
            obj_expr = obj_expr + pen_vec @ y
            # at most one threshold per (carousel, peak)
            rys = np.repeat(np.arange(nC * nP), nL)
            cys = np.arange(n_y)
            A_ys = sp.coo_matrix((np.ones(n_y), (rys, cys)),
                                 shape=(nC * nP, n_y)).tocsr()
            m.addConstr(A_ys @ y <= np.ones(nC * nP))
            if A_u2.shape[0] > 0:
                sl4 = m.addMVar(A_u2.shape[0], lb=0.0)
                m.addConstr(A_u2 @ x + Y_u @ y - sl4 <= b_u)
                obj_expr = obj_expr + Mslack * sl4.sum()
        m.setObjective(obj_expr, GRB.MINIMIZE)

        # warm start
        xs = np.zeros(n_x)
        for fi in range(nF):
            xs[greedy_choice[fi]] = 1.0
        x.Start = xs
        if y is not None:
            ys = np.zeros(n_y)
            for ci in range(nC):
                for pi in range(nP):
                    over = maxu[ci, pi] - target
                    if over > 1e-9:
                        ys[ci * nP * nL + pi * nL + level_for(over)] = 1.0
            y.Start = ys

        xlist = x.tolist()

        def cb(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                vals = model.cbGetSolution(xlist)
            except Exception:
                return
            va = np.array(vals)
            assign = []
            for fi in range(nF):
                ids = f_var_ids[fi]
                assign.append(int(ids[int(np.argmax(va[ids]))]))
            obj, viols, hard = evaluate_full(assign)
            key = (hard > 1e-6, obj)
            if key < best_state["key"]:
                best_state["key"] = key
                best_state["assign"] = assign
                best_state["obj"] = obj
                best_state["viols"] = viols
                if logger:
                    logger.log_solution(obj, build_solution(assign, obj, viols))

        m.optimize(cb)

        if m.SolCount > 0:
            va = np.array(m.getAttr("X", xlist))
            assign = []
            for fi in range(nF):
                ids = f_var_ids[fi]
                assign.append(int(ids[int(np.argmax(va[ids]))]))
            obj, viols, hard = evaluate_full(assign)
            key = (hard > 1e-6, obj)
            if key < best_state["key"]:
                best_state["key"] = key
                best_state["assign"] = assign
                best_state["obj"] = obj
                best_state["viols"] = viols
                if logger:
                    logger.log_solution(obj, build_solution(assign, obj, viols))
    except Exception:
        pass

    sol = build_solution(best_state["assign"], best_state["obj"], best_state["viols"])
    write_out(sol)


if __name__ == "__main__":
    main()