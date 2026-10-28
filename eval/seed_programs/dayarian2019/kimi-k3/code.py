#!/usr/bin/env python3
"""
Integrated production-staffing + multi-trip vehicle routing solver.

Approach:
  1) Regret / cheapest insertion builds multi-trip routes (time windows, vehicle
     capacity, trips start at period boundaries, route duration caps, fleet size).
  2) Local search (relocate / swap / 2-opt / trip move) minimizes routing cost
     (driver salary + vehicle operating cost).
  3) Given trip start periods, an aggregate production-staffing MIP (Gurobi)
     computes the cost-minimal production allocation, employee assignment,
     setup starts, paid workstations and shift starts.
  4) An iterated local search loop keeps improving the combined objective and
     logs every incumbent.
"""

import argparse
import bisect
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    TL = float(args.time_limit)
    deadline = t_start + TL
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    random.seed(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    g = inst["global_parameters"]
    K = int(g["num_periods"])
    S = int(g["num_subperiods"])
    spp = int(g["subperiods_per_period"])
    prep = float(g["preparation_time_minutes"])
    Tmax = int(g["max_trips_per_route"])
    max_min = float(g["max_shift_route_length_minutes"])
    max_per = int(g["max_shift_route_length_periods"])
    min_paid_min = float(g["min_paid_time_minutes"])
    min_paid_sub = int(g["min_paid_time_subperiods"])
    drv_sal = float(g["driver_salary_per_minute"])
    emp_sal = float(g["employee_salary_per_subperiod"])
    setup_cost = float(g["setup_cost"])
    veh_cost = float(g["vehicle_operating_cost_per_minute"])
    setup_len = int(g["setup_length_subperiods"])
    horizon_end = float(g["horizon_end_minutes"])

    cap = float(inst["vehicle_capacity"])
    E = int(inst["num_production_employees"])
    WS = int(inst["num_workstations"])
    fleet = int(min(inst["num_vehicles"], inst["num_drivers"]))

    periods = sorted(inst["periods"], key=lambda p: p["start_time"])
    period_start = [float(p["start_time"]) for p in periods]
    period_id = [p["period_id"] for p in periods]
    subs = sorted(inst["subperiods"], key=lambda s: s["start_time"])
    sub_start = [float(s["start_time"]) for s in subs]
    sub_end = [float(s["end_time"]) for s in subs]
    sub_id = [s["subperiod_id"] for s in subs]
    sid_to_idx = {sub_id[i]: i for i in range(S)}
    period_of_sub = [max(0, bisect.bisect_right(period_start, sub_start[t]) - 1)
                     for t in range(S)]

    # number of periods a started shift must be paid for (min paid time)
    cover = 0
    if min_paid_sub > 0:
        cover = max(1, int(math.ceil(float(min_paid_sub) / max(1, spp))))

    orders = inst["orders"]
    N = len(orders)
    oid = [o["order_id"] for o in orders]
    tws = [float(o["time_window_start"]) for o in orders]
    twe = [float(o["time_window_end"]) for o in orders]
    svc = [float(o["service_time"]) for o in orders]
    space = [float(o["space_requirement"]) for o in orders]
    dem = []
    for o in orders:
        d = {}
        for kk, vv in o.get("demands", {}).items():
            if vv is not None and float(vv) > 0:
                d[int(kk)] = float(vv)
        dem.append(d)

    rate = {p["product_id"]: float(p["production_rate_per_subperiod"])
            for p in inst["products"]}

    tm = inst["travel_time_matrix"]
    t0 = [float(tm[0][i + 1]) for i in range(N)]
    ti0 = [float(tm[i + 1][0]) for i in range(N)]
    tt = [[float(tm[i + 1][j + 1]) for j in range(N)] for i in range(N)]

    pw = inst.get("production_windows", {})
    valid = {}
    min_end = {}
    for o in range(N):
        ow = pw.get(str(oid[o]), {})
        for pid in dem[o]:
            w = ow.get(str(pid))
            vs = w.get("valid_subperiod_ids", []) if w else []
            idxs = sorted(sid_to_idx[v] for v in vs if v in sid_to_idx)
            valid[(o, pid)] = idxs
            min_end[(o, pid)] = min((sub_end[t] for t in idxs), default=math.inf)

    # production cutoff: production for an order must finish before its trip's
    # period start (conservative w.r.t. the trip's depot departure).
    cutoff_ok = [[True] * K for _ in range(N)]
    for o in range(N):
        for k in range(K):
            for pid in dem[o]:
                if min_end[(o, pid)] > period_start[k] + 1e-9:
                    cutoff_ok[o][k] = False
                    break

    feas_cache = {}
    route_cache = {}
    prod_cache = {}

    # ------------------------- routing primitives -------------------------
    def travel_of(seq):
        if not seq:
            return 0.0
        tot = t0[seq[0]] + ti0[seq[-1]]
        for a, b in zip(seq, seq[1:]):
            tot += tt[a][b]
        return tot

    def trip_time_r(seq, k, ignore_tw=False):
        t = period_start[k] + prep
        prev = -1
        for idx, o in enumerate(seq):
            t += t0[o] if idx == 0 else tt[prev][o]
            st = t if t > tws[o] else tws[o]
            if st > twe[o] + 1e-6:
                if not ignore_tw:
                    return None
                st = twe[o]
            t = st + svc[o]
            prev = o
        if seq:
            t += ti0[prev]
        return t

    def feasible_periods_r(seq, relax=0):
        key = (tuple(seq), relax)
        r = feas_cache.get(key)
        if r is not None:
            return r
        res = []
        if relax >= 3 or sum(space[o] for o in seq) <= cap + 1e-9:
            for k in range(K):
                if relax < 1:
                    okc = True
                    for o in seq:
                        if not cutoff_ok[o][k]:
                            okc = False
                            break
                    if not okc:
                        continue
                rt = trip_time_r(seq, k, ignore_tw=(relax >= 2))
                if rt is not None and (relax >= 4 or rt <= horizon_end + 1e-6):
                    res.append((k, rt))
        feas_cache[key] = res
        return res

    def _route_opt_impl(seqs, relax):
        m = len(seqs)
        if m == 0:
            return {'seqs': [], 'periods': [], 'cost': 0.0}
        if m > Tmax:
            return None
        feas = [feasible_periods_r(s, relax) for s in seqs]
        if any(len(f) == 0 for f in feas):
            return None
        order = sorted(range(m), key=lambda i: feas[i][0][0])
        fs = [seqs[i] for i in order]
        ff = [feas[i] for i in order]
        op = veh_cost * sum(travel_of(s) for s in fs)
        best = None
        for (p1, r1) in ff[0]:
            ps = [p1]
            rs = [r1]
            prev = p1
            ok = True
            for i in range(1, m):
                nxt = None
                for (p, r) in ff[i]:
                    if p > prev:
                        nxt = (p, r)
                        break
                if nxt is None:
                    ok = False
                    break
                ps.append(nxt[0])
                rs.append(nxt[1])
                prev = nxt[0]
            if not ok:
                continue
            mr = max(rs)
            dur = mr - period_start[p1]
            if relax < 4 and dur > max_min + 1e-6:
                continue
            ep = bisect.bisect_right(period_start, mr + 1e-9) - 1
            if relax < 4 and ep - p1 + 1 > max_per:
                continue
            drv = drv_sal * max(dur, min_paid_min)
            c = drv + op
            if best is None or c < best['cost'] - 1e-9:
                best = {'seqs': [list(s) for s in fs], 'periods': list(ps), 'cost': c}
        return best

    def _route_latest_impl(seqs, relax):
        m = len(seqs)
        if m == 0:
            return {'seqs': [], 'periods': [], 'cost': 0.0}
        if m > Tmax:
            return None
        feas = [feasible_periods_r(s, relax) for s in seqs]
        if any(len(f) == 0 for f in feas):
            return None
        order = sorted(range(m), key=lambda i: feas[i][0][0])
        fs = [seqs[i] for i in order]
        ff = [feas[i] for i in order]
        ps = [0] * m
        rs = [0.0] * m
        nxt = None
        for i in range(m - 1, -1, -1):
            cand = None
            for (p, r) in reversed(ff[i]):
                if nxt is None or p < nxt:
                    cand = (p, r)
                    break
            if cand is None:
                return None
            ps[i], rs[i] = cand
            nxt = cand[0]
        mr = max(rs)
        dur = mr - period_start[ps[0]]
        if relax < 4 and dur > max_min + 1e-6:
            return None
        ep = bisect.bisect_right(period_start, mr + 1e-9) - 1
        if relax < 4 and ep - ps[0] + 1 > max_per:
            return None
        op = veh_cost * sum(travel_of(s) for s in fs)
        drv = drv_sal * max(dur, min_paid_min)
        return {'seqs': [list(s) for s in fs], 'periods': ps, 'cost': drv + op}

    def route_opt(seqs, relax=0):
        key = (tuple(sorted(tuple(s) for s in seqs)), relax, 0)
        if key in route_cache:
            return route_cache[key]
        res = _route_opt_impl(seqs, relax)
        route_cache[key] = res
        return res

    def route_latest(seqs, relax=0):
        key = (tuple(sorted(tuple(s) for s in seqs)), relax, 1)
        if key in route_cache:
            return route_cache[key]
        res = _route_latest_impl(seqs, relax)
        route_cache[key] = res
        return res

    def latest_version(rts):
        out = []
        for r in rts:
            rl = route_latest(r['seqs'])
            if rl is None:
                return None
            out.append(rl)
        return out

    def copy_routes(rts):
        return [{'seqs': [list(s) for s in r['seqs']],
                 'periods': list(r['periods']),
                 'cost': r['cost']} for r in rts]

    def routing_cost(rts):
        return sum(r['cost'] for r in rts)

    # ------------------------- construction -------------------------
    def all_insert_options(o, routes, relax=0):
        opts = []
        for ri, r in enumerate(routes):
            base = r['cost']
            seqs = r['seqs']
            for i, seq in enumerate(seqs):
                for pos in range(len(seq) + 1):
                    ns = seq[:pos] + [o] + seq[pos:]
                    if relax < 3 and sum(space[x] for x in ns) > cap:
                        continue
                    nseqs = [list(s) for s in seqs]
                    nseqs[i] = ns
                    ro = route_opt(nseqs, relax)
                    if ro:
                        opts.append((ro['cost'] - base, ri, ro))
            if len(seqs) < Tmax:
                nseqs = [list(s) for s in seqs] + [[o]]
                ro = route_opt(nseqs, relax)
                if ro:
                    opts.append((ro['cost'] - base, ri, ro))
        if len(routes) < fleet:
            ro = route_opt([[o]], relax)
            if ro:
                opts.append((ro['cost'], -1, ro))
        opts.sort(key=lambda x: x[0])
        return opts

    def apply_opt(routes, o, opt):
        delta, ri, ro = opt
        if ri == -1:
            routes.append(ro)
        else:
            routes[ri] = ro

    def forced_insert(o, routes):
        for relax in (1, 2, 3, 4):
            opts = all_insert_options(o, routes, relax)
            if opts:
                apply_opt(routes, o, opts[0])
                return
        routes.append({'seqs': [[o]], 'periods': [K - 1], 'cost': 0.0})

    def construct():
        routes = []
        unassigned = list(range(N))
        unassigned.sort(key=lambda o: (twe[o], tws[o]))
        use_regret = N <= 45
        while unassigned:
            if time.time() > t_start + 0.15 * TL:
                use_regret = False
            if use_regret:
                best_key = None
                best_o = None
                best_opt = None
                for o in unassigned:
                    opts = all_insert_options(o, routes)
                    if not opts:
                        continue
                    b1 = opts[0][0]
                    b2 = opts[1][0] if len(opts) > 1 else float('inf')
                    key = (b2 - b1, -b1)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_o = o
                        best_opt = opts[0]
                if best_o is None:
                    o = unassigned[0]
                    forced_insert(o, routes)
                    unassigned.remove(o)
                    continue
                apply_opt(routes, best_o, best_opt)
                unassigned.remove(best_o)
            else:
                o = unassigned.pop(0)
                opts = all_insert_options(o, routes)
                if opts:
                    apply_opt(routes, o, opts[0])
                else:
                    forced_insert(o, routes)
        return routes

    # ------------------------- local search -------------------------
    def pass_relocate(routes, dl):
        ops = 0
        for a in range(len(routes)):
            Ra = routes[a]
            for i in range(len(Ra['seqs'])):
                seq = Ra['seqs'][i]
                for pos in range(len(seq)):
                    o = seq[pos]
                    rem = seq[:pos] + seq[pos + 1:]
                    # reposition within same trip
                    for q in range(len(rem) + 1):
                        if q == pos:
                            continue
                        ns = rem[:q] + [o] + rem[q:]
                        parts = [ns if idx == i else list(s0)
                                 for idx, s0 in enumerate(Ra['seqs'])]
                        ro = route_opt(parts)
                        ops += 1
                        if ro and ro['cost'] < Ra['cost'] - 1e-9:
                            routes[a] = ro
                            return True
                        if ops % 256 == 0 and time.time() > dl:
                            return False
                    for b in range(len(routes)):
                        Rb = routes[b]
                        for j in range(len(Rb['seqs'])):
                            if a == b and j == i:
                                continue
                            seqb = Rb['seqs'][j]
                            if sum(space[x] for x in seqb) + space[o] > cap:
                                continue
                            for q in range(len(seqb) + 1):
                                ops += 1
                                if ops % 256 == 0 and time.time() > dl:
                                    return False
                                if a == b:
                                    parts = []
                                    for idx, s0 in enumerate(Ra['seqs']):
                                        if idx == i:
                                            parts.append(rem)
                                        elif idx == j:
                                            parts.append(s0[:q] + [o] + s0[q:])
                                        else:
                                            parts.append(list(s0))
                                    parts = [p for p in parts if p]
                                    ro = route_opt(parts)
                                    if ro and ro['cost'] < Ra['cost'] - 1e-9:
                                        routes[a] = ro
                                        return True
                                else:
                                    seqsA = [rem if idx == i else list(s0)
                                             for idx, s0 in enumerate(Ra['seqs'])]
                                    seqsA = [p for p in seqsA if p]
                                    roA = route_opt(seqsA) if seqsA else None
                                    if seqsA and roA is None:
                                        continue
                                    seqsB = [(s0[:q] + [o] + s0[q:]) if idx == j else list(s0)
                                             for idx, s0 in enumerate(Rb['seqs'])]
                                    roB = route_opt(seqsB)
                                    if roB is None:
                                        continue
                                    delta = ((roA['cost'] if roA else 0.0) + roB['cost']
                                             - Ra['cost'] - Rb['cost'])
                                    if delta < -1e-9:
                                        routes[a] = roA if roA is not None else None
                                        routes[b] = roB
                                        routes[:] = [x for x in routes if x is not None]
                                        return True
                        # insert as new trip in route b
                        if len(Rb['seqs']) < Tmax:
                            ops += 1
                            if ops % 256 == 0 and time.time() > dl:
                                return False
                            if a == b:
                                parts = [rem if idx == i else list(s0)
                                         for idx, s0 in enumerate(Ra['seqs'])]
                                parts = [p for p in parts if p] + [[o]]
                                ro = route_opt(parts)
                                if ro and ro['cost'] < Ra['cost'] - 1e-9:
                                    routes[a] = ro
                                    return True
                            else:
                                seqsA = [rem if idx == i else list(s0)
                                         for idx, s0 in enumerate(Ra['seqs'])]
                                seqsA = [p for p in seqsA if p]
                                roA = route_opt(seqsA) if seqsA else None
                                if not (seqsA and roA is None):
                                    seqsB = [list(s0) for s0 in Rb['seqs']] + [[o]]
                                    roB = route_opt(seqsB)
                                    if roB is not None:
                                        delta = ((roA['cost'] if roA else 0.0) + roB['cost']
                                                 - Ra['cost'] - Rb['cost'])
                                        if delta < -1e-9:
                                            routes[a] = roA if roA is not None else None
                                            routes[b] = roB
                                            routes[:] = [x for x in routes if x is not None]
                                            return True
                    # create new route
                    if len(routes) < fleet:
                        ops += 1
                        seqsA = [rem if idx == i else list(s0)
                                 for idx, s0 in enumerate(Ra['seqs'])]
                        seqsA = [p for p in seqsA if p]
                        roA = route_opt(seqsA) if seqsA else None
                        roB = route_opt([[o]])
                        if (roA is not None or not seqsA) and roB is not None:
                            delta = ((roA['cost'] if roA else 0.0) + roB['cost'] - Ra['cost'])
                            if delta < -1e-9:
                                if roA is None:
                                    routes.pop(a)
                                else:
                                    routes[a] = roA
                                routes.append(roB)
                                return True
        return False

    def pass_swap(routes, dl):
        ops = 0
        for a in range(len(routes)):
            for b in range(a, len(routes)):
                Ra, Rb = routes[a], routes[b]
                for i in range(len(Ra['seqs'])):
                    jstart = i + 1 if a == b else 0
                    for j in range(jstart, len(Rb['seqs'])):
                        seqA, seqB = Ra['seqs'][i], Rb['seqs'][j]
                        for p1 in range(len(seqA)):
                            for p2 in range(len(seqB)):
                                ops += 1
                                if ops % 256 == 0 and time.time() > dl:
                                    return False
                                o1, o2 = seqA[p1], seqB[p2]
                                nA = seqA[:p1] + [o2] + seqA[p1 + 1:]
                                nB = seqB[:p2] + [o1] + seqB[p2 + 1:]
                                if sum(space[x] for x in nA) > cap:
                                    continue
                                if sum(space[x] for x in nB) > cap:
                                    continue
                                if a == b:
                                    parts = []
                                    for idx, s0 in enumerate(Ra['seqs']):
                                        if idx == i:
                                            parts.append(nA)
                                        elif idx == j:
                                            parts.append(nB)
                                        else:
                                            parts.append(list(s0))
                                    ro = route_opt(parts)
                                    if ro and ro['cost'] < Ra['cost'] - 1e-9:
                                        routes[a] = ro
                                        return True
                                else:
                                    seqsA = [nA if idx == i else list(s0)
                                             for idx, s0 in enumerate(Ra['seqs'])]
                                    seqsB = [nB if idx == j else list(s0)
                                             for idx, s0 in enumerate(Rb['seqs'])]
                                    roA = route_opt(seqsA)
                                    roB = route_opt(seqsB)
                                    if roA is None or roB is None:
                                        continue
                                    delta = (roA['cost'] + roB['cost']
                                             - Ra['cost'] - Rb['cost'])
                                    if delta < -1e-9:
                                        routes[a] = roA
                                        routes[b] = roB
                                        return True
        return False

    def pass_twoopt(routes, dl):
        ops = 0
        for a in range(len(routes)):
            Ra = routes[a]
            for i in range(len(Ra['seqs'])):
                seq = Ra['seqs'][i]
                L = len(seq)
                for x in range(L - 1):
                    for y in range(x + 1, L):
                        ops += 1
                        if ops % 256 == 0 and time.time() > dl:
                            return False
                        ns = seq[:x] + seq[x:y + 1][::-1] + seq[y + 1:]
                        parts = [ns if idx == i else list(s0)
                                 for idx, s0 in enumerate(Ra['seqs'])]
                        ro = route_opt(parts)
                        if ro and ro['cost'] < Ra['cost'] - 1e-9:
                            routes[a] = ro
                            return True
        return False

    def pass_tripmove(routes, dl):
        ops = 0
        for a in range(len(routes)):
            Ra = routes[a]
            for i in range(len(Ra['seqs'])):
                trip = Ra['seqs'][i]
                seqsA = [list(s0) for idx, s0 in enumerate(Ra['seqs']) if idx != i]
                roA = route_opt(seqsA) if seqsA else None
                if seqsA and roA is None:
                    continue
                costA = roA['cost'] if roA else 0.0
                for b in range(len(routes)):
                    if b == a:
                        continue
                    Rb = routes[b]
                    if len(Rb['seqs']) >= Tmax:
                        continue
                    ops += 1
                    if ops % 64 == 0 and time.time() > dl:
                        return False
                    seqsB = [list(s0) for s0 in Rb['seqs']] + [list(trip)]
                    roB = route_opt(seqsB)
                    if roB is None:
                        continue
                    delta = costA + roB['cost'] - Ra['cost'] - Rb['cost']
                    if delta < -1e-9:
                        routes[a] = roA if roA is not None else None
                        routes[b] = roB
                        routes[:] = [x for x in routes if x is not None]
                        return True
                if len(routes) < fleet:
                    roB = route_opt([list(trip)])
                    if roB:
                        delta = costA + roB['cost'] - Ra['cost']
                        if delta < -1e-9:
                            if roA is None:
                                routes.pop(a)
                            else:
                                routes[a] = roA
                            routes.append(roB)
                            return True
        return False

    def local_search(routes, dl):
        while time.time() < dl:
            if pass_relocate(routes, dl):
                continue
            if pass_swap(routes, dl):
                continue
            if pass_twoopt(routes, dl):
                continue
            if pass_tripmove(routes, dl):
                continue
            break
        return routes

    # ------------------------- production -------------------------
    def order_period_of(rts):
        opm = {}
        for r in rts:
            for per, seq in zip(r['periods'], r['seqs']):
                for o in seq:
                    opm[o] = per
        return opm

    def heuristic_staffing(order_period):
        if order_period is None or len(order_period) < N:
            return None
        loads = {}
        sched = []
        feasible = True
        for o in range(N):
            C = period_start[order_period[o]]
            for pid, d in dem[o].items():
                A = [t for t in valid[(o, pid)] if sub_end[t] <= C + 1e-9]
                if not A:
                    feasible = False
                    A = list(valid[(o, pid)])
                if not A:
                    A = list(range(S))
                bt = min(A, key=lambda t: loads.get((pid, t), 0.0))
                loads[(pid, bt)] = loads.get((pid, bt), 0.0) + d
                sched.append((o, pid, bt, d))
        byprod = {}
        for (pid, t), q in loads.items():
            d0 = byprod.setdefault(pid, {})
            d0[t] = d0.get(t, 0.0) + q
        e = {}
        s = {}
        workers = {}
        for pid, dct in byprod.items():
            prev_t = None
            prev_e = 0
            for t in sorted(dct):
                ev = int(math.ceil(dct[t] / rate[pid] - 1e-9))
                if prev_t is not None and t == prev_t + 1:
                    sv = max(0, ev - prev_e)
                else:
                    sv = ev
                prev_t = t
                prev_e = ev
                if ev > 0:
                    e[(pid, t)] = ev
                if sv > 0:
                    s[(pid, t)] = sv
                workers[t] = workers.get(t, 0) + ev
                if sv > 0:
                    for tp in range(t, min(S, t + setup_len)):
                        workers[tp] = workers.get(tp, 0) + sv
        Pk = [0] * K
        for t, w in workers.items():
            k = period_of_sub[t]
            if w > Pk[k]:
                Pk[k] = w
        if max(Pk) > E or max(Pk) > WS:
            feasible = False
        h = [0] * K
        cum = 0
        for k in range(K):
            hk = max(0, Pk[k] - cum)
            h[k] = hk
            cum += hk
        if cum > E:
            feasible = False
        W = [0] * K
        for k in range(K):
            lo = max(0, k - cover + 1)
            W[k] = max(Pk[k], sum(h[lo:k + 1]))
            if W[k] > WS:
                feasible = False
        cost = emp_sal * spp * sum(W) + setup_cost * sum(s.values())
        return {'cost': cost, 'e': e, 's': s, 'W': W, 'h': h,
                'sched': sched, 'feasible': feasible, 'order_period': order_period}

    def solve_production(vroutes, tl, relax_emp_cap=False, force=False):
        order_period = order_period_of(vroutes)
        if len(order_period) < N:
            return None
        key = (tuple(order_period.get(o, -1) for o in range(N)), relax_emp_cap)
        if not force and key in prod_cache:
            return prod_cache[key]
        allowed = {}
        for o in range(N):
            C = period_start[order_period[o]]
            for pid, d in dem[o].items():
                A = [t for t in valid[(o, pid)] if sub_end[t] <= C + 1e-9]
                if not A:
                    prod_cache[key] = None
                    return None
                allowed[(o, pid)] = A
        prods_used = sorted({pid for o in range(N) for pid in dem[o]})
        try:
            m = gp.Model("prod")
            m.Params.OutputFlag = 0
            m.Params.LogToConsole = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, tl)

            Tspan = {}
            for pid in prods_used:
                lo = min(min(allowed[(o, pid)]) for o in range(N) if pid in dem[o])
                hi = max(max(allowed[(o, pid)]) for o in range(N) if pid in dem[o])
                Tspan[pid] = list(range(lo, hi + 1))
            e = {}
            s = {}
            for pid in prods_used:
                for t in Tspan[pid]:
                    e[pid, t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=E)
                    s[pid, t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=E)
            x = {}
            for (o, pid), A in allowed.items():
                d = dem[o][pid]
                for t in A:
                    x[o, pid, t] = m.addVar(lb=0.0, ub=d, vtype=GRB.CONTINUOUS)
            W = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=WS) for _ in range(K)]
            h = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=E) for _ in range(K)]
            m.update()

            for (o, pid), A in allowed.items():
                m.addConstr(gp.quicksum(x[o, pid, t] for t in A) >= dem[o][pid])
            for pid in prods_used:
                olist = [o for o in range(N) if pid in dem[o]]
                for t in Tspan[pid]:
                    terms = [x[o, pid, t] for o in olist if (o, pid, t) in x]
                    if terms:
                        m.addConstr(gp.quicksum(terms) <= rate[pid] * e[pid, t])
                    else:
                        m.addConstr(e[pid, t] == 0)
            for pid in prods_used:
                prev = None
                for t in Tspan[pid]:
                    if prev is None:
                        m.addConstr(s[pid, t] >= e[pid, t])
                    else:
                        m.addConstr(s[pid, t] >= e[pid, t] - e[pid, prev])
                    prev = t
            if prods_used:
                loA = min(Tspan[pid][0] for pid in prods_used)
                hiA = max(Tspan[pid][-1] for pid in prods_used)
                all_t = list(range(loA, min(S - 1, hiA + setup_len - 1) + 1))
            else:
                all_t = []
            wexpr = {}
            for t in all_t:
                terms = []
                for pid in prods_used:
                    if (pid, t) in e:
                        terms.append(e[pid, t])
                    lo = max(0, t - setup_len + 1)
                    for tp in range(lo, t + 1):
                        if (pid, tp) in s:
                            terms.append(s[pid, tp])
                if not terms:
                    continue
                wexpr[t] = gp.quicksum(terms)
                k = period_of_sub[t]
                m.addConstr(wexpr[t] <= W[k])
                m.addConstr(wexpr[t] <= E)
            if cover >= 1:
                for k in range(K):
                    lo = max(0, k - cover + 1)
                    m.addConstr(W[k] >= gp.quicksum(h[kk] for kk in range(lo, k + 1)))
            for t, we in wexpr.items():
                k = period_of_sub[t]
                m.addConstr(gp.quicksum(h[kk] for kk in range(k + 1)) >= we)
            if not relax_emp_cap:
                m.addConstr(gp.quicksum(h) <= E)

            hs = heuristic_staffing(order_period)
            if hs is not None:
                for (pid, t), v in hs['e'].items():
                    if (pid, t) in e:
                        e[pid, t].Start = v
                for (pid, t), v in hs['s'].items():
                    if (pid, t) in s:
                        s[pid, t].Start = v
                for k in range(K):
                    W[k].Start = hs['W'][k]
                    h[k].Start = hs['h'][k]

            m.setObjective(emp_sal * spp * gp.quicksum(W)
                           + setup_cost * gp.quicksum(list(s.values())),
                           GRB.MINIMIZE)
            m.optimize()
            if m.SolCount == 0:
                prod_cache[key] = None
                return None
            e_v = {(p, t): int(round(v.X)) for (p, t), v in e.items()}
            s_v = {(p, t): int(round(v.X)) for (p, t), v in s.items()}
            W_v = [int(round(v.X)) for v in W]
            h_v = [int(round(v.X)) for v in h]
            sched = [(o, pid, t, float(v.X)) for (o, pid, t), v in x.items()
                     if v.X > 1e-6]
            cost = emp_sal * spp * sum(W_v) + setup_cost * sum(s_v.values())
            res = {'cost': cost, 'e': e_v, 's': s_v, 'W': W_v, 'h': h_v,
                   'sched': sched, 'order_period': order_period,
                   'feasible': True}
            prod_cache[key] = res
            return res
        except Exception:
            return None

    # ------------------------- solution assembly -------------------------
    def build_solution(vroutes, prod):
        rjson = []
        for vi, r in enumerate(vroutes):
            trips = []
            for ti, (seq, per) in enumerate(zip(r['seqs'], r['periods'])):
                trips.append({'trip': int(ti),
                              'customers': [int(oid[o]) for o in seq],
                              'start_period': int(period_id[per])})
            rjson.append({'vehicle': int(vi), 'trips': trips})
        sched = [{'order': int(oid[o]), 'product': int(pid),
                  'subperiod': int(sub_id[t]), 'quantity': float(q)}
                 for (o, pid, t, q) in prod['sched']]
        emp_list = [{'product': int(pid), 'subperiod': int(sub_id[t]),
                     'employees': int(v)}
                    for (pid, t), v in sorted(prod['e'].items()) if v > 0]
        set_list = [{'product': int(pid), 'subperiod': int(sub_id[t]),
                     'setups': int(v)}
                    for (pid, t), v in sorted(prod['s'].items()) if v > 0]
        obj = float(routing_cost(vroutes) + prod['cost'])
        sol = {'objective_value': obj,
               'routes': rjson,
               'production_schedule': sched,
               'production_staffing': {
                   'employees_by_product_subperiod': emp_list,
                   'setups_by_product_subperiod': set_list,
                   'workstations_by_period': [int(w) for w in prod['W']],
                   'shift_starts_by_period': [int(hh) for hh in prod['h']]}}
        return obj, sol

    # ------------------------- perturbation -------------------------
    def perturb(rts):
        r = copy_routes(rts)
        moves_target = random.randint(1, 3)
        moves_done = 0
        attempts = 0
        while moves_done < moves_target and attempts < 12:
            attempts += 1
            occ = [(a, i, pos) for a, ra in enumerate(r)
                   for i, sq in enumerate(ra['seqs']) for pos in range(len(sq))]
            if not occ:
                break
            a, i, pos = random.choice(occ)
            o = r[a]['seqs'][i][pos]
            rem = r[a]['seqs'][i][:pos] + r[a]['seqs'][i][pos + 1:]
            targets = []
            for b, rb in enumerate(r):
                for j, seqb in enumerate(rb['seqs']):
                    if a == b and j == i:
                        continue
                    if sum(space[x] for x in seqb) + space[o] <= cap:
                        targets.append(('trip', b, j, random.randint(0, len(seqb))))
                if len(rb['seqs']) < Tmax:
                    targets.append(('newtrip', b, None, None))
            if len(r) < fleet:
                targets.append(('newroute', None, None, None))
            random.shuffle(targets)
            done = False
            for kind, b, j, q in targets:
                if kind == 'trip':
                    if a == b:
                        parts = []
                        for idx, s0 in enumerate(r[a]['seqs']):
                            if idx == i:
                                parts.append(rem)
                            elif idx == j:
                                parts.append(s0[:q] + [o] + s0[q:])
                            else:
                                parts.append(list(s0))
                        parts = [p for p in parts if p]
                        ro = route_opt(parts)
                        if ro:
                            r[a] = ro
                            done = True
                    else:
                        seqsA = [rem if idx == i else list(s0)
                                 for idx, s0 in enumerate(r[a]['seqs'])]
                        seqsA = [p for p in seqsA if p]
                        roA = route_opt(seqsA) if seqsA else None
                        if seqsA and roA is None:
                            continue
                        seqsB = [(s0[:q] + [o] + s0[q:]) if idx == j else list(s0)
                                 for idx, s0 in enumerate(r[b]['seqs'])]
                        roB = route_opt(seqsB)
                        if roB is None:
                            continue
                        r[a] = roA if roA is not None else None
                        r[b] = roB
                        r[:] = [x for x in r if x is not None]
                        done = True
                elif kind == 'newtrip':
                    if a == b:
                        parts = [rem if idx == i else list(s0)
                                 for idx, s0 in enumerate(r[a]['seqs'])]
                        parts = [p for p in parts if p] + [[o]]
                        ro = route_opt(parts)
                        if ro:
                            r[a] = ro
                            done = True
                    else:
                        seqsA = [rem if idx == i else list(s0)
                                 for idx, s0 in enumerate(r[a]['seqs'])]
                        seqsA = [p for p in seqsA if p]
                        roA = route_opt(seqsA) if seqsA else None
                        if seqsA and roA is None:
                            continue
                        seqsB = [list(s0) for s0 in r[b]['seqs']] + [[o]]
                        roB = route_opt(seqsB)
                        if roB is None:
                            continue
                        r[a] = roA if roA is not None else None
                        r[b] = roB
                        r[:] = [x for x in r if x is not None]
                        done = True
                else:  # new route
                    roB = route_opt([[o]])
                    if roB is None:
                        continue
                    seqsA = [rem if idx == i else list(s0)
                             for idx, s0 in enumerate(r[a]['seqs'])]
                    seqsA = [p for p in seqsA if p]
                    roA = route_opt(seqsA) if seqsA else None
                    if seqsA and roA is None:
                        continue
                    if roA is None:
                        r.pop(a)
                    else:
                        r[a] = roA
                    r.append(roB)
                    done = True
                if done:
                    break
            if done:
                moves_done += 1
        return r

    # ------------------------- main flow -------------------------
    routes0 = []
    best = {'obj': math.inf, 'sol': None, 'routes': None}

    def consider(vroutes, tl):
        if vroutes is None:
            return
        if time.time() > deadline - 1.0:
            return
        res = solve_production(vroutes, tl)
        if res is None:
            res = solve_production(vroutes, max(1.0, tl), relax_emp_cap=True)
        if res is None:
            return
        obj = routing_cost(vroutes) + res['cost']
        if obj < best['obj'] - 1e-9:
            o, sol = build_solution(vroutes, res)
            best['obj'] = o
            best['sol'] = sol
            best['routes'] = copy_routes(vroutes)
            if logger:
                logger.log_solution(o, sol)

    try:
        routes0 = construct()
        local_search(routes0, min(deadline - 1.0, t_start + 0.45 * TL))

        tl0 = min(60.0, max(5.0, 0.25 * TL))
        consider(routes0, tl0)
        consider(latest_version(routes0), tl0)

        base = best['routes'] if best['routes'] is not None else copy_routes(routes0)
        while time.time() < deadline - 1.0:
            cand = perturb(base)
            local_search(cand, min(deadline - 1.0,
                                   time.time() + max(2.0, 0.06 * TL)))
            variants = [cand]
            lv = latest_version(cand)
            if lv is not None:
                variants.append(lv)
            prev_best = best['obj']
            for vv in variants:
                hs = heuristic_staffing(order_period_of(vv))
                if hs is None:
                    continue
                est = routing_cost(vv) + hs['cost']
                if hs['feasible'] and est < best['obj'] * 1.10:
                    base = copy_routes(vv)
                try_mip = (hs['feasible'] and est < best['obj'] * 1.03 + 1e-9) or \
                          (not hs['feasible'] and routing_cost(vv) < best['obj'] - 1e-9)
                if try_mip:
                    rem = deadline - 1.0 - time.time()
                    if rem < 2.0:
                        break
                    consider(vv, min(30.0, rem * 0.6))
            if best['obj'] < prev_best - 1e-9:
                base = best['routes']
    except Exception:
        pass

    if best['sol'] is None:
        try:
            vr = latest_version(routes0) or routes0
            hs = heuristic_staffing(order_period_of(vr))
            if hs is not None:
                o, sol = build_solution(vr, hs)
                best['obj'] = o
                best['sol'] = sol
                if logger:
                    logger.log_solution(o, sol)
        except Exception:
            best['sol'] = None

    if best['sol'] is None:
        best['sol'] = {'objective_value': 0.0,
                       'routes': [],
                       'production_schedule': [],
                       'production_staffing': {
                           'employees_by_product_subperiod': [],
                           'setups_by_product_subperiod': [],
                           'workstations_by_period': [0] * K,
                           'shift_starts_by_period': [0] * K}}

    with open(args.solution_path, 'w') as f:
        json.dump(best['sol'], f)


if __name__ == "__main__":
    main()