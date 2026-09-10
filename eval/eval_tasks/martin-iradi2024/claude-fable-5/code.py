import argparse
import json
import time
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t_start = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    beta = float(inst["handling_time_deviation_factor_beta_per_meter"])
    cp = inst["cost_parameters"]
    FUEL_P = float(cp["fuel_price_usd_per_tonne"])
    HC = float(cp["handling_cost_usd_per_hour"])
    DC = float(cp["delay_cost_usd_per_hour"])
    WC = float(cp["waiting_cost_usd_per_hour"])
    LP = float(cp["lft_penalty_usd"])
    horizon = float(inst.get("planning_horizon_hours", 0))

    ports = {p["id"]: float(p["quay_length_meters"]) for p in inst["ports"]}
    dists = {k: float(v) for k, v in inst["distances_nautical_miles"].items()}
    speeds = sorted(float(s) for s in inst["speeds_knots"])
    tt_raw = {float(k): float(v) for k, v in inst["travel_time_hours_per_nm_by_speed"].items()}

    def tt(s):
        if s in tt_raw:
            return tt_raw[s]
        # nearest key fallback
        kk = min(tt_raw.keys(), key=lambda k: abs(k - s))
        return tt_raw[kk]

    ships = inst["ships"]
    nS = len(ships)
    vis_list = []
    ship_fuel = []
    for ship in ships:
        visits = sorted(ship["port_visits"], key=lambda v: v["port_call_index"])
        vis_list.append(visits)
        fm = {float(k): float(v) for k, v in ship["fuel_consumption_per_nm_by_speed"].items()}
        def mk(fm):
            def f(s):
                if s in fm:
                    return fm[s]
                kk = min(fm.keys(), key=lambda k: abs(k - s))
                return fm[kk]
            return f
        ship_fuel.append(mk(fm))

    ext_by_port = {}
    for grp in inst.get("external_ships", []) or []:
        pid = grp["port_id"]
        ext_by_port.setdefault(pid, [])
        for e in grp.get("ships", []):
            ext_by_port[pid].append(e)

    def dist(o, d):
        return dists.get("%s_%s" % (o, d), 0.0)

    # -------------------- greedy heuristic --------------------
    def greedy_schedule():
        occupied = {}
        for pid, els in ext_by_port.items():
            occupied.setdefault(pid, [])
            for e in els:
                occupied[pid].append((float(e["berthing_position"]), float(e["length_meters"]),
                                      float(e["start_time_hours"]), float(e["end_time_hours"])))
        xv, tv, sv = {}, {}, {}
        order = sorted(range(nS), key=lambda si: vis_list[si][0]["earliest_start_time_hours"])
        for si in order:
            ship = ships[si]
            slen = float(ship["length_meters"])
            visits = vis_list[si]
            prev_dep = None
            prev_port = None
            for vi, vis in enumerate(visits):
                pid = vis["port_id"]
                Lq = ports[pid]
                maxpos = max(0.0, Lq - slen)
                ideal = float(vis["ideal_berthing_position"])
                hmin = float(vis["min_handling_time_hours"])
                est = float(vis["earliest_start_time_hours"])
                if vi > 0:
                    dnm = dist(prev_port, pid)
                    chosen = None
                    for s in speeds:
                        if prev_dep + dnm * tt(s) <= est + 1e-9:
                            chosen = s
                            break
                    if chosen is None:
                        chosen = speeds[-1]
                    sv[(si, vi)] = chosen
                    arr = prev_dep + dnm * tt(chosen)
                else:
                    arr = est
                t = max(arr, est)
                occ = occupied.setdefault(pid, [])
                pos = None
                h = None
                max_iter = 2 * len(occ) + 10
                for _ in range(max_iter):
                    cands = set([min(max(ideal, 0.0), maxpos), 0.0, maxpos])
                    for (p, l, s0, e0) in occ:
                        cands.add(min(max(p + l, 0.0), maxpos))
                        cands.add(min(max(p - slen, 0.0), maxpos))
                    found = None
                    for c in sorted(cands, key=lambda c: abs(c - ideal)):
                        hh = hmin * (1.0 + beta * abs(c - ideal))
                        ok = True
                        for (p, l, s0, e0) in occ:
                            if p < c + slen - 1e-7 and p + l > c + 1e-7 and s0 < t + hh - 1e-7 and e0 > t + 1e-7:
                                ok = False
                                break
                        if ok:
                            found = (c, hh)
                            break
                    if found:
                        pos, h = found
                        break
                    ends = [e0 for (_, _, _, e0) in occ if e0 > t + 1e-7]
                    if not ends:
                        break
                    t = min(ends)
                if pos is None:
                    all_ends = [e0 for (_, _, _, e0) in occ]
                    t = max([t] + all_ends)
                    pos = min(max(ideal, 0.0), maxpos)
                    h = hmin * (1.0 + beta * abs(pos - ideal))
                xv[(si, vi)] = pos
                tv[(si, vi)] = t
                occ.append((pos, slen, t, t + h))
                prev_dep = t + h
                prev_port = pid
        return xv, tv, sv

    # -------------------- solution builder --------------------
    def compute_solution(xvals, tvals, svals):
        total = 0.0
        ships_out = []
        for si, ship in enumerate(ships):
            slen = float(ship["length_meters"])
            visits = vis_list[si]
            pv = []
            legs_out = []
            prev_t = prev_h = None
            prev_port = None
            for vi, vis in enumerate(visits):
                xx = float(xvals[(si, vi)])
                tS = float(tvals[(si, vi)])
                ideal = float(vis["ideal_berthing_position"])
                dev = abs(xx - ideal)
                h = float(vis["min_handling_time_hours"]) * (1.0 + beta * dev)
                if vi == 0:
                    arr = tS
                else:
                    sp = svals[(si, vi)]
                    dnm = dist(prev_port, vis["port_id"])
                    arr = prev_t + prev_h + dnm * tt(sp)
                    if tS < arr:
                        tS = arr
                    total += FUEL_P * ship_fuel[si](sp) * dnm
                    legs_out.append({
                        "from_call": int(visits[vi - 1]["port_call_index"]),
                        "to_call": int(vis["port_call_index"]),
                        "speed_knots": float(sp),
                    })
                end = tS + h
                delay = max(0.0, end - float(vis["expected_finish_time_hours"]))
                lftv = max(0.0, end - float(vis["latest_finish_time_hours"]))
                wait = max(0.0, tS - arr)
                total += WC * wait + HC * h + DC * delay + LP * lftv
                pv.append({
                    "port_call_index": int(vis["port_call_index"]),
                    "port_id": vis["port_id"],
                    "berthing_position": xx,
                    "start_time": tS,
                    "handling_time": h,
                    "arrival_time": arr,
                    "delay": delay,
                    "lft_violation": lftv,
                    "position_deviation": dev,
                })
                prev_t, prev_h, prev_port = tS, h, vis["port_id"]
            ships_out.append({"ship_id": int(ship["id"]), "port_visits": pv, "speed_selections": legs_out})
        return total, {"objective_value": total, "ships": ships_out}

    # greedy incumbent
    gx, gt, gs = greedy_schedule()
    g_obj, g_sol = compute_solution(gx, gt, gs)
    best = [g_obj, g_sol]
    if logger:
        logger.log_solution(g_obj, g_sol)
    with open(args.solution_path, "w") as f:
        json.dump(g_sol, f, indent=1)

    remaining = args.time_limit - (time.time() - t_start) - 2.0
    if remaining < 3.0 or nS == 0:
        return

    # -------------------- MIP model --------------------
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    # time upper bound
    max_lft = 0.0
    for si in range(nS):
        for vis in vis_list[si]:
            max_lft = max(max_lft, float(vis["latest_finish_time_hours"]))
    ext_max = 0.0
    for pid, els in ext_by_port.items():
        for e in els:
            ext_max = max(ext_max, float(e["end_time_hours"]))
    base = max(horizon, max_lft, ext_max)
    slow_tt = max(tt_raw.values())
    serial = 0.0
    hub = {}
    for si, ship in enumerate(ships):
        slen = float(ship["length_meters"])
        visits = vis_list[si]
        for vi, vis in enumerate(visits):
            Lq = ports[vis["port_id"]]
            maxpos = max(0.0, Lq - slen)
            ideal = float(vis["ideal_berthing_position"])
            maxdev = max(ideal, maxpos - ideal, 0.0)
            hub[(si, vi)] = float(vis["min_handling_time_hours"]) * (1.0 + beta * maxdev)
            serial += hub[(si, vi)]
            if vi > 0:
                serial += dist(visits[vi - 1]["port_id"], vis["port_id"]) * slow_tt
    greedy_max_end = 0.0
    for sh in g_sol["ships"]:
        for pv in sh["port_visits"]:
            greedy_max_end = max(greedy_max_end, pv["start_time"] + pv["handling_time"])
    TUB = max(base + serial + 10.0, greedy_max_end + 10.0)
    MT = TUB + max(hub.values()) + 1.0

    m = gp.Model("bap")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.MIPFocus = 1

    X, T, D, Z = {}, {}, {}, {}
    HEXP = {}
    obj_terms = gp.LinExpr()

    for si, ship in enumerate(ships):
        slen = float(ship["length_meters"])
        visits = vis_list[si]
        for vi, vis in enumerate(visits):
            pid = vis["port_id"]
            Lq = ports[pid]
            maxpos = max(0.0, Lq - slen)
            ideal = float(vis["ideal_berthing_position"])
            hmin = float(vis["min_handling_time_hours"])
            est = float(vis["earliest_start_time_hours"])
            eft = float(vis["expected_finish_time_hours"])
            lft = float(vis["latest_finish_time_hours"])
            maxdev = max(ideal, maxpos - ideal, 0.0)

            xv = m.addVar(lb=0.0, ub=maxpos, name="x_%d_%d" % (si, vi))
            tv = m.addVar(lb=est, ub=TUB, name="t_%d_%d" % (si, vi))
            dv = m.addVar(lb=0.0, ub=maxdev, name="d_%d_%d" % (si, vi))
            m.addConstr(dv >= xv - ideal)
            m.addConstr(dv >= ideal - xv)
            hx = hmin + hmin * beta * dv
            X[(si, vi)], T[(si, vi)], D[(si, vi)] = xv, tv, dv
            HEXP[(si, vi)] = hx

            dly = m.addVar(lb=0.0)
            lvv = m.addVar(lb=0.0)
            m.addConstr(dly >= tv + hx - eft)
            m.addConstr(lvv >= tv + hx - lft)
            obj_terms += HC * hx + DC * dly + LP * lvv

            if vi > 0:
                dnm = dist(visits[vi - 1]["port_id"], pid)
                zsum = gp.LinExpr()
                trav = gp.LinExpr()
                for k, s in enumerate(speeds):
                    zz = m.addVar(vtype=GRB.BINARY, name="z_%d_%d_%d" % (si, vi, k))
                    Z[(si, vi, k)] = zz
                    zsum += zz
                    trav += tt(s) * dnm * zz
                    obj_terms += FUEL_P * ship_fuel[si](s) * dnm * zz
                m.addConstr(zsum == 1)
                av = m.addVar(lb=0.0, ub=TUB)
                m.addConstr(av == T[(si, vi - 1)] + HEXP[(si, vi - 1)] + trav)
                wv = m.addVar(lb=0.0)
                m.addConstr(wv == tv - av)
                obj_terms += WC * wv

    # non-overlap: optimized pairs
    visits_at = {}
    for si in range(nS):
        for vi, vis in enumerate(vis_list[si]):
            visits_at.setdefault(vis["port_id"], []).append((si, vi))

    pair_bins = []
    for pid, lst in visits_at.items():
        MS = ports[pid]
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                (si, vi), (sj, vj) = lst[a], lst[b]
                if si == sj:
                    continue
                Lu = float(ships[si]["length_meters"])
                Lv = float(ships[sj]["length_meters"])
                s1 = m.addVar(vtype=GRB.BINARY)  # u left of v
                s2 = m.addVar(vtype=GRB.BINARY)  # v left of u
                d1 = m.addVar(vtype=GRB.BINARY)  # u before v
                d2 = m.addVar(vtype=GRB.BINARY)  # v before u
                m.addConstr(X[(si, vi)] + Lu <= X[(sj, vj)] + MS * (1 - s1))
                m.addConstr(X[(sj, vj)] + Lv <= X[(si, vi)] + MS * (1 - s2))
                m.addConstr(T[(si, vi)] + HEXP[(si, vi)] <= T[(sj, vj)] + MT * (1 - d1))
                m.addConstr(T[(sj, vj)] + HEXP[(sj, vj)] <= T[(si, vi)] + MT * (1 - d2))
                m.addConstr(s1 + s2 + d1 + d2 >= 1)
                pair_bins.append(((si, vi), (sj, vj), s1, s2, d1, d2, Lu, Lv))

    ext_bins = []
    for si in range(nS):
        slen = float(ships[si]["length_meters"])
        for vi, vis in enumerate(vis_list[si]):
            pid = vis["port_id"]
            est = float(vis["earliest_start_time_hours"])
            MS = ports[pid]
            for e in ext_by_port.get(pid, []):
                pe = float(e["berthing_position"])
                le = float(e["length_meters"])
                se = float(e["start_time_hours"])
                ee = float(e["end_time_hours"])
                if ee <= est + 1e-9:
                    continue  # ship always starts after external ends
                b1 = m.addVar(vtype=GRB.BINARY)  # v left of e
                b2 = m.addVar(vtype=GRB.BINARY)  # v right of e
                b3 = m.addVar(vtype=GRB.BINARY)  # v before e
                b4 = m.addVar(vtype=GRB.BINARY)  # v after e
                m.addConstr(X[(si, vi)] + slen <= pe + MS * (1 - b1))
                m.addConstr(pe + le <= X[(si, vi)] + MS * (1 - b2))
                m.addConstr(T[(si, vi)] + HEXP[(si, vi)] <= se + MT * (1 - b3))
                m.addConstr(ee <= T[(si, vi)] + MT * (1 - b4))
                m.addConstr(b1 + b2 + b3 + b4 >= 1)
                ext_bins.append(((si, vi), b1, b2, b3, b4, pe, le, se, ee, slen))

    m.setObjective(obj_terms, GRB.MINIMIZE)

    # -------------------- warm start from greedy --------------------
    ghs = {}
    for si in range(nS):
        for vi, vis in enumerate(vis_list[si]):
            ideal = float(vis["ideal_berthing_position"])
            ghs[(si, vi)] = float(vis["min_handling_time_hours"]) * (1.0 + beta * abs(gx[(si, vi)] - ideal))
            X[(si, vi)].Start = gx[(si, vi)]
            T[(si, vi)].Start = gt[(si, vi)]
            D[(si, vi)].Start = abs(gx[(si, vi)] - ideal)
            if vi > 0:
                for k, s in enumerate(speeds):
                    Z[(si, vi, k)].Start = 1.0 if abs(s - gs[(si, vi)]) < 1e-9 else 0.0

    for ((si, vi), (sj, vj), s1, s2, d1, d2, Lu, Lv) in pair_bins:
        xu, xvv = gx[(si, vi)], gx[(sj, vj)]
        tu, tvv = gt[(si, vi)], gt[(sj, vj)]
        hu, hv = ghs[(si, vi)], ghs[(sj, vj)]
        s1.Start = 1.0 if xu + Lu <= xvv + 1e-6 else 0.0
        s2.Start = 1.0 if xvv + Lv <= xu + 1e-6 else 0.0
        d1.Start = 1.0 if tu + hu <= tvv + 1e-6 else 0.0
        d2.Start = 1.0 if tvv + hv <= tu + 1e-6 else 0.0
    for ((si, vi), b1, b2, b3, b4, pe, le, se, ee, slen) in ext_bins:
        xu = gx[(si, vi)]
        tu = gt[(si, vi)]
        hu = ghs[(si, vi)]
        b1.Start = 1.0 if xu + slen <= pe + 1e-6 else 0.0
        b2.Start = 1.0 if pe + le <= xu + 1e-6 else 0.0
        b3.Start = 1.0 if tu + hu <= se + 1e-6 else 0.0
        b4.Start = 1.0 if ee <= tu + 1e-6 else 0.0

    # -------------------- callback logging --------------------
    xkeys = list(X.keys())
    xvars = [X[k] for k in xkeys]
    tvars = [T[k] for k in xkeys]
    zkeys = list(Z.keys())
    zvars = [Z[k] for k in zkeys]

    def extract(get_vals):
        xs = get_vals(xvars)
        ts = get_vals(tvars)
        zs = get_vals(zvars)
        xvals = {k: xs[i] for i, k in enumerate(xkeys)}
        tvals = {k: ts[i] for i, k in enumerate(xkeys)}
        zbest = {}
        for i, (si, vi, k) in enumerate(zkeys):
            cur = zbest.get((si, vi))
            if cur is None or zs[i] > cur[0]:
                zbest[(si, vi)] = (zs[i], speeds[k])
        svals = {kk: vv[1] for kk, vv in zbest.items()}
        return xvals, tvals, svals

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xvals, tvals, svals = extract(lambda vs: model.cbGetSolution(vs))
                obj, sol = compute_solution(xvals, tvals, svals)
                if obj < best[0] - 1e-6:
                    best[0] = obj
                    best[1] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    remaining = args.time_limit - (time.time() - t_start) - 2.0
    if remaining > 1.0:
        m.Params.TimeLimit = remaining
        try:
            m.optimize(cb)
        except Exception:
            pass
        if m.SolCount > 0:
            try:
                xvals, tvals, svals = extract(lambda vs: [v.X for v in vs])
                obj, sol = compute_solution(xvals, tvals, svals)
                if obj < best[0] - 1e-9:
                    best[0] = obj
                    best[1] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    with open(args.solution_path, "w") as f:
        json.dump(best[1], f, indent=1)


if __name__ == "__main__":
    main()