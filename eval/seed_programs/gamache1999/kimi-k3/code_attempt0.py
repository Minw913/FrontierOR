#!/usr/bin/env python3
"""
Personalized monthly cabin-crew rostering.

Approach
--------
1. Enumerate a pool of feasible rosters per crew member:
   randomized greedy construction guided by an exact rule checker
   (qualifications, non-overlap, flight-hour limits, early-departure rule,
   cross-pairing limit, reserve-block rule, weekly/monthly rest and the
   max-spread-between-breaks rule checked with a small DP).
2. Solve a set-partitioning MIP (Gurobi) choosing exactly one roster per crew
   member, minimizing total uncovered pairing minutes.
3. Improve the chosen solution with insertion / swap local search.
"""

import argparse
import bisect
import json
import random
import time

T = 1440          # minutes per day
EPS = 1e-6
RESTLIKE_TYPES = ('annual_leave', 'desiderata', 'day_off', 'off', 'leave',
                  'vacation')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    tl = max(2, int(args.time_limit))

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    # ---------------- global data ----------------
    D = int(inst.get('planning_horizon_days', 30))
    H = D * T
    wr = inst.get('work_rules', {}) or {}

    def gi(k, d):
        try:
            return int(wr.get(k, d))
        except Exception:
            return d

    def gf(k, d):
        try:
            return float(wr.get(k, d))
        except Exception:
            return d

    LW_H = gi('weekly_rest_period_hours', 48)
    MADD_H = gi('monthly_rest_additional_hours', 48)
    SPREAD = gi('max_spread_between_breaks_days', 8)
    MLIM = gf('monthly_flight_limit_hours', 77.0)
    RLIM = gf('rolling_flight_limit_hours', 85.0)
    MAXE = gi('max_consecutive_early_departures_before_8am', 2)
    MAXX = gi('max_cross_pairings_per_month', 1)
    SUNH = gi('weekly_rest_latest_start_on_sunday_hour', 6) * 60
    LW = LW_H * 60                       # weekly rest length in minutes
    NW = (D + 6) // 7                    # number of (partial) weeks
    EXT = 30 * 60                        # allowed rest overflow into next week

    # ---------------- pairings ----------------
    prs = inst.get('pairings', []) or []
    P = len(prs)
    PID = [int(p.get('id', i)) for i, p in enumerate(prs)]
    SD = [max(0, int(p.get('start_day', 0))) for p in prs]
    DEP = [float(p.get('departure_hour', 0.0)) for p in prs]
    FH = [float(p.get('flight_hours', 0.0)) for p in prs]
    DURM = [int(p.get('duration_minutes', 0)) for p in prs]
    AT = [int(p.get('aircraft_type', 0)) for p in prs]
    CROSS = [bool(p.get('is_cross_pairing', False)) for p in prs]
    REQ = [max(0, int(p.get('coverage_requirement', 1))) for p in prs]
    PS = []   # pairing start minute
    PE = []   # pairing end minute (clipped to horizon)
    for i, p in enumerate(prs):
        s = SD[i] * T + int(round(DEP[i] * 60.0))
        if DURM[i] > 0:
            e = s + DURM[i]
        else:
            dd = max(1, int(p.get('duration_days', 1)))
            e = (SD[i] + dd - 1) * T + int(round(float(p.get('return_hour', 0.0)) * 60.0))
            if e <= s:
                e = s + T
        PS.append(s)
        PE.append(min(e, H))

    # ---------------- crew data ----------------
    crews = inst.get('crew_members', []) or []
    C = len(crews)
    CID = [int(c.get('id', i)) for i, c in enumerate(crews)]
    CDATA = []
    for c in crews:
        acts_raw = c.get('pre_assigned_activities', []) or []
        acts = []
        covered_days = set()
        for a in acts_raw:
            try:
                sd = int(a.get('start_day', 0))
                ed = int(a.get('end_day', sd))
            except Exception:
                continue
            if ed < sd:
                sd, ed = ed, sd
            s = max(0, sd * T)
            e = min(H, (ed + 1) * T)
            if e <= s:
                continue
            tp = str(a.get('type', '')).lower()
            acts.append((s, e, tp))
            for d in range(sd, ed + 1):
                covered_days.add(d)
        for d in c.get('pre_assigned_days', []) or []:
            try:
                d = int(d)
            except Exception:
                continue
            if 0 <= d < D and d not in covered_days:
                acts.append((d * T, (d + 1) * T, 'other'))
                covered_days.add(d)
        act_intervals = sorted((s, e) for s, e, tp in acts)
        restlike = sorted((s, e) for s, e, tp in acts if tp in RESTLIKE_TYPES)
        reserves = sorted((s, e) for s, e, tp in acts if tp == 'reserve_block')
        inactive = 0
        for a in acts_raw:
            tp = str(a.get('type', '')).lower()
            if tp in RESTLIKE_TYPES:
                try:
                    inactive += int(a.get('duration_days',
                                         int(a.get('end_day', 0)) - int(a.get('start_day', 0)) + 1))
                except Exception:
                    pass
        extra = max(0, MADD_H - 12 * (inactive // 7)) * 60
        CDATA.append({
            'acts': act_intervals,
            'restlike': restlike,
            'reserves': reserves,
            'extra': extra,
            'quals': set(int(q) for q in c.get('qualifications', []) or []),
            'prev': float(c.get('flight_hours_previous_15_days', 0.0) or 0.0),
            'act_minutes': sum(e - s for s, e in act_intervals),
        })

    # ---------------- candidate pairings per crew ----------------
    CAND = []
    CANDSET = []
    for c in range(C):
        cd = CDATA[c]
        lst = []
        for p in range(P):
            if REQ[p] <= 0 or AT[p] not in cd['quals']:
                continue
            s, e = PS[p], PE[p]
            ok = True
            for (a, b) in cd['acts']:
                if b <= s:
                    continue
                if a >= e:
                    break
                ok = False
                break
            if ok:
                lst.append(p)
        CAND.append(lst)
        CANDSET.append(set(lst))

    # ---------------- rule-checking helpers ----------------
    def lim_minutes(cd, a, b):
        for rs, re_ in cd['reserves']:
            if rs < b and re_ > a:
                return (SPREAD + 1) * T
        return SPREAD * T

    def chain_ok(cd, t1, t2):
        """Check max-spread rule over the fixed leave breaks lying strictly
        between a previous break end t1 (None = month start) and the next
        break start t2 (None = month end)."""
        cur = t1
        for ls, le in cd['restlike']:
            if t2 is not None and ls >= t2:
                break
            if t1 is not None and le <= t1:
                continue
            if cur is not None and ls > cur and (ls - cur) > lim_minutes(cd, cur, ls):
                return False
            if cur is None or le > cur:
                cur = le
        if t2 is not None and cur is not None and t2 > cur:
            if (t2 - cur) > lim_minutes(cd, cur, t2):
                return False
        return True

    def rest_intervals(cd, pints):
        busy = sorted(cd['acts'] + pints)
        rints = []
        cur = 0
        for s, e in busy:
            if s > cur:
                rints.append((cur, s))
                cur = e
            elif e > cur:
                cur = e
        if cur < H:
            rints.append((cur, H))
        rints.extend(cd['restlike'])
        rints.sort()
        if rints and rints[-1][1] >= H:
            rints[-1] = (rints[-1][0], H + EXT)
        return rints

    def week_domains(rints, L):
        doms = []
        for w in range(NW):
            Ws = w * 7 * T
            sun06 = Ws + 6 * T + SUNH
            lst = []
            for a, b in rints:
                if b - a < L:
                    continue
                lo = a if a > Ws else Ws
                hb = b - L
                hi = hb if hb < sun06 else sun06
                if lo <= hi:
                    lst.append(lo)
                    mid = (lo + hi) // 2
                    if mid != lo and mid != hi:
                        lst.append(mid)
                    if hi != lo:
                        lst.append(hi)
            doms.append(lst)
        return doms

    def rest_ok(cd, pints):
        """Weekly rest in every week, one monthly rest (weekly + extra),
        and max spread between consecutive breaks.  Small DP over weeks."""
        rints = rest_intervals(cd, pints)
        E = cd['extra']
        domLw = week_domains(rints, LW)
        for w in range(NW):
            if not domLw[w]:
                return False
        domLm = week_domains(rints, LW + E) if E > 0 else None
        states = {0: [None]}     # monthly-rest-used flag -> list of (s, e)
        for w in range(NW):
            ns = {}
            dl = domLw[w]
            dm = domLm[w] if domLm is not None else ()
            for flag, prevs in states.items():
                for prev in prevs:
                    pe = prev[1] if prev is not None else None
                    lst0 = ns.setdefault(flag, [])
                    for s2 in dl:
                        if chain_ok(cd, pe, s2):
                            lst0.append((s2, s2 + LW))
                    if E > 0 and flag == 0:
                        lst1 = ns.setdefault(1, [])
                        for s2 in dm:
                            if chain_ok(cd, pe, s2):
                                lst1.append((s2, s2 + LW + E))
            if not ns:
                return False
            states = ns
        want = 1 if E > 0 else 0
        return any(chain_ok(cd, pe, None) for (_, pe) in states.get(want, ()))

    def cheap_ok(c, sel):
        cd = CDATA[c]
        tot = 0.0
        roll = cd['prev']
        cross = 0
        pts = []
        edays = set()
        for p in sel:
            tot += FH[p]
            if tot > MLIM + EPS:
                return False
            if SD[p] < 15:
                roll += FH[p]
            if CROSS[p]:
                cross += 1
                if cross > MAXX:
                    return False
            if DEP[p] < 8.0 - 1e-9:
                edays.add(SD[p])
            pts.append((PS[p], PE[p]))
        if roll > RLIM + EPS:
            return False
        pts.sort()
        for i in range(1, len(pts)):
            if pts[i][0] < pts[i - 1][1]:
                return False
        ed = sorted(edays)
        run = 1
        for i in range(1, len(ed)):
            if ed[i] == ed[i - 1] + 1:
                run += 1
                if run > MAXE:
                    return False
            else:
                run = 1
        for rs_, re_ in cd['reserves']:
            nxt = None
            nd = 24.0
            for p in sel:
                sp = PS[p]
                if sp >= re_ and (nxt is None or sp < nxt):
                    nxt = sp
                    nd = DEP[p]
            if nxt is not None and (nxt - re_) < LW and nd < 12.0 - 1e-9:
                return False
        return True

    def full_check(c, sel):
        cd = CDATA[c]
        tot = 0.0
        roll = cd['prev']
        cross = 0
        pints = []
        edays = []
        for p in sel:
            tot += FH[p]
            if SD[p] < 15:
                roll += FH[p]
            if CROSS[p]:
                cross += 1
            if DEP[p] < 8.0 - 1e-9:
                edays.append(SD[p])
            pints.append((PS[p], PE[p]))
        if tot > MLIM + EPS or roll > RLIM + EPS or cross > MAXX:
            return False
        pints.sort()
        for i in range(1, len(pints)):
            if pints[i][0] < pints[i - 1][1]:
                return False
        for s, e in pints:
            for a, b in cd['acts']:
                if b <= s:
                    continue
                if a >= e:
                    break
                return False
        edays = sorted(set(edays))
        run = 1
        for i in range(1, len(edays)):
            if edays[i] == edays[i - 1] + 1:
                run += 1
                if run > MAXE:
                    return False
            else:
                run = 1
        for rs_, re_ in cd['reserves']:
            nxt = None
            nxtp = -1
            for p in sel:
                sp = PS[p]
                if sp >= re_ and (nxt is None or sp < nxt):
                    nxt = sp
                    nxtp = p
            if nxtp >= 0 and (nxt - re_) < LW and DEP[nxtp] < 12.0 - 1e-9:
                return False
        return rest_ok(cd, pints)

    # ---------------- roster construction ----------------
    def build_roster(c, order):
        cd = CDATA[c]
        sel = []
        starts = []
        sints = []
        hours = 0.0
        roll = cd['prev']
        cross = 0
        edays = set()
        reserves = cd['reserves']
        for p in order:
            s = PS[p]
            e = PE[p]
            i = bisect.bisect_left(starts, s)
            if i > 0 and sints[i - 1][1] > s:
                continue
            if i < len(sints) and sints[i][0] < e:
                continue
            h = FH[p]
            if hours + h > MLIM + EPS:
                continue
            if SD[p] < 15 and roll + h > RLIM + EPS:
                continue
            if CROSS[p] and cross >= MAXX:
                continue
            dep = DEP[p]
            if dep < 8.0 - 1e-9:
                d = SD[p]
                viol = False
                for st in range(d - MAXE, d + 1):
                    good = True
                    for k in range(st, st + MAXE + 1):
                        if k != d and k not in edays:
                            good = False
                            break
                    if good:
                        viol = True
                        break
                if viol:
                    continue
            if dep < 12.0 - 1e-9 and reserves:
                badr = False
                for rs_, re_ in reserves:
                    if re_ <= s and s - re_ < LW:
                        j = bisect.bisect_left(starts, re_)
                        if j < i:
                            continue
                        badr = True
                        break
                if badr:
                    continue
            sints.insert(i, (s, e))
            starts.insert(i, s)
            sel.append(p)
            hours += h
            if SD[p] < 15:
                roll += h
            if CROSS[p]:
                cross += 1
            if dep < 8.0 - 1e-9:
                edays.add(SD[p])
        return sel

    def make_feasible(c, sel):
        while sel and not full_check(c, sel):
            cd = CDATA[c]
            pints = sorted((PS[p], PE[p]) for p in sel)
            rints = rest_intervals(cd, pints)
            dom = week_domains(rints, LW)
            badw = [w for w in range(NW) if not dom[w]]
            rem = -1
            if badw:
                bestd = None
                for p in sel:
                    hit = False
                    for w in badw:
                        if PS[p] < (w + 1) * 7 * T and PE[p] > w * 7 * T:
                            hit = True
                            break
                    if hit and (bestd is None or DURM[p] < bestd):
                        bestd = DURM[p]
                        rem = p
            if rem < 0:
                rem = min(sel, key=lambda p: DURM[p])
            sel = [p for p in sel if p != rem]
        return sel

    # ---------------- roster pools ----------------
    pools = [[] for _ in range(C)]
    poolidx = [dict() for _ in range(C)]

    def add_roster(c, sel):
        t = tuple(sorted(sel))
        d = poolidx[c]
        if t in d:
            return -1
        i = len(pools[c])
        d[t] = i
        pools[c].append(t)
        return i

    def roster_index_of(c, sel):
        t = tuple(sorted(sel))
        d = poolidx[c]
        v = d.get(t)
        if v is not None:
            return v
        i = len(pools[c])
        d[t] = i
        pools[c].append(t)
        return i

    for c in range(C):
        add_roster(c, [])          # always keep the empty roster (MIP feasibility)

    def build_solution(assign):
        rosters = {}
        cnt = [0] * P
        for c in range(C):
            lst = sorted(assign[c])
            ridx = roster_index_of(c, lst)
            pm = 0
            for p in lst:
                cnt[p] += 1
                pm += (PE[p] - PS[p])
            unprod = H - pm - CDATA[c]['act_minutes']
            if unprod < 0:
                unprod = 0
            rosters[str(CID[c])] = {
                'roster_index': int(ridx),
                'pairings': [int(PID[p]) for p in sorted(lst, key=lambda p: (SD[p], PS[p], PID[p]))],
                'unproductive_time': int(unprod),
            }
        unc = []
        obj = 0
        for p in range(P):
            u = REQ[p] - cnt[p]
            if u > 0:
                obj += DURM[p] * u
                unc.append({'pairing_id': int(PID[p]),
                            'uncovered_count': int(u),
                            'duration_minutes': int(DURM[p])})
        return float(obj), {'objective_value': float(obj),
                            'rosters': rosters,
                            'uncovered_pairings': unc}

    # ---------------- initial greedy solution ----------------
    init_assign = [[] for _ in range(C)]
    counts = [0] * P
    porder = sorted([p for p in range(P) if REQ[p] > 0], key=lambda p: -DURM[p])
    try:
        init_deadline = t0 + min(0.15 * tl, 90.0)
        for p in porder:
            if time.time() > init_deadline:
                break
            corder = sorted(range(C), key=lambda c: len(init_assign[c]))
            for c in corder:
                if counts[p] >= REQ[p]:
                    break
                if p not in CANDSET[c]:
                    continue
                nr = init_assign[c] + [p]
                if cheap_ok(c, nr) and full_check(c, nr):
                    init_assign[c] = nr
                    counts[p] += 1
    except Exception:
        init_assign = [[] for _ in range(C)]
        counts = [0] * P

    for c in range(C):
        add_roster(c, init_assign[c])

    best_obj, best_sol = build_solution(init_assign)
    best_assign = [list(a) for a in init_assign]
    if logger:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    # ---------------- roster pool generation ----------------
    try:
        rng = random.Random(0)
        gen_deadline = t0 + min(0.55 * tl, max(tl - 12.0, 1.0))
        target = 90

        # focused rosters for pairings left uncovered by the greedy solution
        hard = [p for p in range(P) if REQ[p] > 0 and counts[p] < REQ[p]]
        rng.shuffle(hard)
        for p in hard:
            if time.time() > gen_deadline:
                break
            qc = [c for c in range(C) if p in CANDSET[c]]
            rng.shuffle(qc)
            for c in qc[:3]:
                order = [p] + sorted(CAND[c], key=lambda q: (SD[q] + rng.random() * 2.0))
                sel = make_feasible(c, build_roster(c, order))
                add_roster(c, sel)

        stagnant = 0
        c = 0
        while time.time() < gen_deadline:
            added = False
            if CAND[c]:
                m = rng.randrange(3)
                if m == 0:
                    order = sorted(CAND[c], key=lambda p: (SD[p] + rng.random() * 4.0))
                elif m == 1:
                    order = sorted(CAND[c], key=lambda p: -DURM[p] * (0.4 + rng.random()))
                else:
                    order = CAND[c][:]
                    rng.shuffle(order)
                sel = make_feasible(c, build_roster(c, order))
                added = add_roster(c, sel) >= 0
            stagnant = 0 if added else stagnant + 1
            c += 1
            if c >= C:
                c = 0
                if all(len(pools[cc]) >= target or not CAND[cc] for cc in range(C)):
                    break
            if stagnant > 4 * C + 30:
                break
    except Exception:
        pass

    # ---------------- set-partitioning MIP ----------------
    pool_counts = [len(pools[c]) for c in range(C)]
    try:
        import gurobipy as gp
        from gurobipy import GRB

        mip_budget = (t0 + 0.90 * tl) - time.time()
        if mip_budget >= 2.0:
            model = gp.Model('rostering')
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.LogToConsole = 0
            model.Params.TimeLimit = max(1.0, mip_budget)

            ylist = []
            yindex = []
            crew_vars = []
            for c in range(C):
                vs = []
                for r in range(pool_counts[c]):
                    v = model.addVar(vtype=GRB.BINARY, name='y_%d_%d' % (c, r))
                    ylist.append(v)
                    yindex.append((c, r))
                    vs.append(v)
                crew_vars.append(vs)
            uvars = [model.addVar(vtype=GRB.INTEGER, lb=0, ub=REQ[p],
                                  name='u_%d' % p) for p in range(P)]
            for c in range(C):
                model.addConstr(gp.quicksum(crew_vars[c]) == 1)
            memb = [[] for _ in range(P)]
            for k, (c, r) in enumerate(yindex):
                for p in pools[c][r]:
                    memb[p].append(ylist[k])
            for p in range(P):
                if REQ[p] <= 0:
                    if memb[p]:
                        model.addConstr(gp.quicksum(memb[p]) == 0)
                else:
                    model.addConstr(gp.quicksum(memb[p]) + uvars[p] == REQ[p])
            model.setObjective(gp.quicksum(DURM[p] * uvars[p] for p in range(P)),
                               GRB.MINIMIZE)

            # warm start from the incumbent
            try:
                for c in range(C):
                    r = poolidx[c].get(tuple(sorted(best_assign[c])))
                    if r is not None and r < pool_counts[c]:
                        crew_vars[c][r].Start = 1.0
            except Exception:
                pass

            def cb(mdl, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = mdl.cbGetSolution(ylist)
                        assign = [[] for _ in range(C)]
                        for k, (c, r) in enumerate(yindex):
                            if vals[k] > 0.5:
                                assign[c] = list(pools[c][r])
                        o, sol = build_solution(assign)
                        nonlocal_best = None  # placeholder (edited below)
                        if o < cb.best[0] - 1e-9:
                            cb.best[0] = o
                            cb.best[1] = sol
                            cb.best[2] = [list(a) for a in assign]
                        if logger:
                            logger.log_solution(o, sol)
                    except Exception:
                        pass

            cb.best = [best_obj, best_sol, best_assign]
            model.optimize(cb)
            best_obj, best_sol, best_assign = cb.best[0], cb.best[1], cb.best[2]

            if model.SolCount > 0:
                try:
                    vals = [v.X for v in ylist]
                    assign = [[] for _ in range(C)]
                    for k, (c, r) in enumerate(yindex):
                        if vals[k] > 0.5:
                            assign[c] = list(pools[c][r])
                    o, sol = build_solution(assign)
                    if o < best_obj - 1e-9:
                        best_obj, best_sol = o, sol
                        best_assign = [list(a) for a in assign]
                        if logger:
                            logger.log_solution(o, sol)
                except Exception:
                    pass
    except Exception:
        pass

    # ---------------- local-search improvement ----------------
    try:
        pp_deadline = t0 + 0.97 * tl
        assign = [list(a) for a in best_assign]
        counts = [0] * P
        for c in range(C):
            for p in assign[c]:
                counts[p] += 1
        it = 0
        changed = True
        while changed and it < 8 and time.time() < pp_deadline:
            it += 1
            changed = False
            corder = sorted(range(C), key=lambda c: len(assign[c]))
            # insertion of uncovered copies
            for p in porder:
                if time.time() > pp_deadline:
                    break
                while counts[p] < REQ[p]:
                    placed = False
                    for c in corder:
                        if p in CANDSET[c]:
                            nr = assign[c] + [p]
                            if cheap_ok(c, nr) and full_check(c, nr):
                                assign[c] = nr
                                counts[p] += 1
                                placed = True
                                changed = True
                                break
                    if not placed:
                        break
            # swap: replace a shorter covered pairing by a longer uncovered one
            for p in porder:
                if time.time() > pp_deadline:
                    break
                while counts[p] < REQ[p]:
                    moved = False
                    for c in corder:
                        if p in assign[c] or p not in CANDSET[c]:
                            continue
                        for q in sorted(assign[c], key=lambda q: DURM[q]):
                            if DURM[q] >= DURM[p]:
                                break
                            base = [x for x in assign[c] if x != q]
                            newr = base + [p]
                            if not (cheap_ok(c, newr) and full_check(c, newr)):
                                continue
                            ins = -1
                            for c2 in corder:
                                if c2 == c or q not in CANDSET[c2]:
                                    continue
                                nr2 = assign[c2] + [q]
                                if cheap_ok(c2, nr2) and full_check(c2, nr2):
                                    ins = c2
                                    break
                            if ins >= 0:
                                assign[c] = newr
                                assign[ins] = assign[ins] + [q]
                                counts[p] += 1
                                moved = True
                                changed = True
                                break
                            elif DURM[p] > DURM[q]:
                                assign[c] = newr
                                counts[q] -= 1
                                counts[p] += 1
                                moved = True
                                changed = True
                                break
                        if moved:
                            break
                    if not moved:
                        break
            if changed:
                o, sol = build_solution(assign)
                if o < best_obj - 1e-9:
                    best_obj, best_sol = o, sol
                    best_assign = [list(a) for a in assign]
                    if logger:
                        try:
                            logger.log_solution(o, sol)
                        except Exception:
                            pass
    except Exception:
        pass

    # ---------------- final output ----------------
    try:
        best_obj, best_sol = build_solution(best_assign)
    except Exception:
        if best_sol is None:
            best_obj, best_sol = build_solution([[] for _ in range(C)])

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f)


if __name__ == '__main__':
    main()