import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# --------------------------------------------------------------------------- #
# Instance parsing
# --------------------------------------------------------------------------- #
def prepare(data):
    I = {}
    I['alpha'] = float(data.get('alpha', 1.0))
    patients = data.get('patients', [])
    surgeons = data.get('surgeons', [])
    hospitals = data.get('hospitals', [])

    I['P'] = [p['patient_id'] for p in patients]
    I['S'] = [s['surgeon_id'] for s in surgeons]
    I['H'] = [h['hospital_id'] for h in hospitals]

    days = set()
    for s in surgeons:
        for d in s.get('operating_days', []):
            days.add(int(d))
    I['days'] = sorted(days)

    avail, opdays = {}, {}
    for s in surgeons:
        sid = s['surgeon_id']
        av = {int(k): float(v) for k, v in s.get('availability_by_day', {}).items()}
        avail[sid] = av
        opdays[sid] = sorted({int(d) for d in s.get('operating_days', [])
                              if av.get(int(d), 0.0) > 0})
    I['avail'], I['opdays'] = avail, opdays

    ors, R_of_h = {}, {}
    for h in hospitals:
        hid = h['hospital_id']
        R_of_h[hid] = []
        for o in h.get('ORs', []):
            rid = o['or_id']
            ors[(hid, rid)] = {
                'reg_def': float(o.get('regular_time', 0.0)),
                'mot_def': float(o.get('max_overtime', 0.0)),
                'fcost_hr': float(o.get('fixed_cost_per_hour', 0.0)),
                'otcost_hr': float(o.get('overtime_cost_per_hour', 0.0)),
                'daily': {int(k): v for k, v in o.get('daily', {}).items()},
            }
            R_of_h[hid].append(rid)
    I['ors'], I['R_of_h'] = ors, R_of_h

    reg, mot, open_cost, ot_rate = {}, {}, {}, {}
    for (h, r), o in ors.items():
        ot_rate[(h, r)] = o['otcost_hr'] / 60.0
        for d in I['days']:
            dd = o['daily'].get(d)
            rg = float(dd['regular_time']) if dd and 'regular_time' in dd else o['reg_def']
            mo = float(dd['max_overtime']) if dd and 'max_overtime' in dd else o['mot_def']
            reg[(h, d, r)] = rg
            mot[(h, d, r)] = mo
            # Opening cost: per-hour rate times the regular session hours of that day.
            open_cost[(h, d, r)] = o['fcost_hr'] * rg / 60.0
    I['reg'], I['mot'], I['open_cost'], I['ot_rate'] = reg, mot, open_cost, ot_rate

    scost = {}
    for h in hospitals:
        hid = h['hospital_id']
        for k, v in h.get('surgeon_fixed_costs_per_day', {}).items():
            scost[(hid, int(k))] = float(v)
    I['scost'] = scost

    prep, clean, due, mand, reward = {}, {}, {}, {}, {}
    elig_s, elig_or, tsur = {}, {}, {}
    for p in patients:
        pid = p['patient_id']
        prep[pid] = float(p.get('preparation_time', 0.0))
        clean[pid] = float(p.get('cleaning_time', 0.0))
        due[pid] = int(p.get('due_date', I['days'][-1] if I['days'] else 0))
        mand[pid] = bool(p.get('is_mandatory', True))
        rw = float(p.get('reward', 0.0))
        reward[pid] = 0.0 if mand[pid] else rw
        elig_s[pid] = [int(s) for s in p.get('eligible_surgeons', [])]
        elig_or[pid] = {int(h): [int(r) for r in rs]
                        for h, rs in p.get('eligible_ORs_by_hospital', {}).items()}
        base = float(p.get('base_surgical_time', 0.0))
        tt = {int(k): float(v) for k, v in p.get('surgeon_specific_times', {}).items()}
        for s in elig_s[pid]:
            if s not in tt:
                tt[s] = base
        tsur[pid] = tt
    I.update({'prep': prep, 'clean': clean, 'due': due, 'mand': mand,
              'reward': reward, 'elig_s': elig_s, 'elig_or': elig_or, 'tsur': tsur})

    tuples_by_p = {}
    for pid in I['P']:
        lst = []
        for s in elig_s[pid]:
            if s not in tsur[pid] or s not in opdays:
                continue
            tps = tsur[pid][s]
            T = prep[pid] + tps + clean[pid]
            for d in opdays[s]:
                if mand[pid] and d > due[pid]:
                    continue
                if avail[s].get(d, 0.0) <= 0 or tps > avail[s][d] + 1e-6:
                    continue
                for h, rids in elig_or[pid].items():
                    for r in rids:
                        if (h, r) not in ors:
                            continue
                        if T > reg[(h, d, r)] + mot[(h, d, r)] + 1e-6:
                            continue
                        lst.append((pid, s, h, d, r))
        tuples_by_p[pid] = lst
    I['tuples_by_p'] = tuples_by_p
    return I


def T_of(I, t):
    p, s, _, _, _ = t
    return I['prep'][p] + I['tsur'][p][s] + I['clean'][p]


# --------------------------------------------------------------------------- #
# Schedule verification / left-shifting
# --------------------------------------------------------------------------- #
def verify(I, sched, eps=1e-5):
    rooms, surg = defaultdict(list), defaultdict(list)
    for pid, (s, h, d, r, fin, start, ext) in sched.items():
        rooms[(h, d, r)].append((start - I['prep'][pid], ext, fin, pid))
        surg[(s, h, d)].append((start, fin, pid))
    for (h, d, r), jobs in rooms.items():
        jobs.sort()
        prev_exit, tot = 0.0, 0.0
        for entry, ext, fin, pid in jobs:
            if entry < -eps or entry < prev_exit - eps:
                return False
            prev_exit = ext
            tot += I['prep'][pid] + I['tsur'][pid][sched[pid][0]] + I['clean'][pid]
        comp = prev_exit
        if comp > I['reg'][(h, d, r)] + I['mot'][(h, d, r)] + eps:
            return False
        if tot > max(comp, I['reg'][(h, d, r)]) + eps:
            return False
    for (s, h, d), jobs in surg.items():
        jobs.sort()
        prev_fin, first_start, last_fin, work = 0.0, None, 0.0, 0.0
        for start, fin, pid in jobs:
            if start < prev_fin - eps:
                return False
            if first_start is None:
                first_start = start
            prev_fin = fin
            last_fin = max(last_fin, fin)
            tps = I['tsur'][pid][s]
            work += I['alpha'] * tps + (1.0 - I['alpha']) * (
                I['prep'][pid] + tps + I['clean'][pid])
        if jobs and (last_fin - first_start > I['avail'][s][d] + eps
                     or work > I['avail'][s][d] + eps):
            return False
    for pid, (s, h, d, r, fin, start, ext) in sched.items():
        if fin < I['prep'][pid] + I['tsur'][pid][s] - eps:
            return False
    return True


def left_shift(I, sched):
    """Re-time each hospital-day as early as possible, preserving orders."""
    groups = defaultdict(list)
    for pid, rec in sched.items():
        groups[(rec[1], rec[2])].append(pid)
    new = dict(sched)
    for (h, d), pids in groups.items():
        pids.sort(key=lambda pid: (sched[pid][4], sched[pid][5], pid))
        room_free, surg_free = defaultdict(float), defaultdict(float)
        for pid in pids:
            s, _, _, r, _, _, _ = sched[pid]
            tps = I['tsur'][pid][s]
            start = max(room_free[r] + I['prep'][pid], surg_free[s])
            fin = start + tps
            ext = fin + I['clean'][pid]
            room_free[r] = ext
            surg_free[s] = fin
            new[pid] = (s, h, d, r, fin, start, ext)
    return new


# --------------------------------------------------------------------------- #
# Solution dictionary construction (projects onto solution_schema.json)
# --------------------------------------------------------------------------- #
def build_from_state(I, sched):
    prep, clean, tsur, mand, reward = (I['prep'], I['clean'], I['tsur'],
                                       I['mand'], I['reward'])
    reg, open_cost, ot_rate, scost = (I['reg'], I['open_cost'],
                                      I['ot_rate'], I['scost'])
    assignments = []
    room_jobs, surg_jobs, surg_hosp = defaultdict(list), defaultdict(list), {}
    for pid, (s, h, d, r, fin, start, ext) in sched.items():
        tps = tsur[pid][s]
        assignments.append({
            'patient_id': int(pid), 'surgeon_id': int(s), 'hospital_id': int(h),
            'day': int(d), 'or_id': int(r), 'is_mandatory': bool(mand[pid]),
            'T_ps': float(prep[pid] + tps + clean[pid]),
            'finish_time': float(fin), 'surgery_start_time': float(start),
            'room_entry_time': float(start - prep[pid]),
            'room_exit_time': float(ext)})
        room_jobs[(h, d, r)].append(pid)
        surg_jobs[(s, h, d)].append(pid)
        surg_hosp[(s, d)] = h

    opened, or_seq, obj = [], [], 0.0
    for (h, d, r), jobs in room_jobs.items():
        if not jobs:
            continue
        comp = max(sched[pid][6] for pid in jobs)
        otv = max(0.0, comp - reg[(h, d, r)])
        obj += open_cost[(h, d, r)] + otv * ot_rate[(h, r)]
        rg = reg[(h, d, r)]
        opened.append({
            'hospital_id': int(h), 'day': int(d), 'or_id': int(r),
            'regular_time': int(rg) if float(rg).is_integer() else float(rg),
            'completion_time': float(comp), 'overtime': float(otv)})
        js = sorted(jobs, key=lambda pid: (sched[pid][4], sched[pid][5], pid))
        for i in range(len(js)):
            for j in range(i + 1, len(js)):
                or_seq.append({'hospital_id': int(h), 'day': int(d),
                               'or_id': int(r), 'patient_p': int(js[i]),
                               'patient_k': int(js[j]), 'p_after_k': 0})

    sassign, surg_seq = [], []
    for (s, d), jobs in surg_jobs.items():
        if not jobs:
            continue
        h = surg_hosp[(s, d)]
        st = min(sched[pid][5] for pid in jobs)
        en = max(sched[pid][4] for pid in jobs)
        obj += scost.get((h, s), 0.0)
        sassign.append({'surgeon_id': int(s), 'hospital_id': int(h),
                        'day': int(d), 'start_time': float(st),
                        'end_time': float(en)})
        js = sorted(jobs, key=lambda pid: (sched[pid][5], sched[pid][4], pid))
        for i in range(len(js)):
            for j in range(i + 1, len(js)):
                surg_seq.append({'hospital_id': int(h), 'day': int(d),
                                 'surgeon_id': int(s), 'patient_p': int(js[i]),
                                 'patient_k': int(js[j]), 'p_after_k': 0})

    for pid in sched:
        if not mand[pid]:
            obj -= reward[pid]

    assignments.sort(key=lambda a: (a['day'], a['hospital_id'], a['or_id'],
                                    a['room_entry_time'], a['patient_id']))
    opened.sort(key=lambda o: (o['day'], o['hospital_id'], o['or_id']))
    sassign.sort(key=lambda a: (a['day'], a['hospital_id'], a['surgeon_id']))
    return {'objective_value': float(obj), 'assignments': assignments,
            'opened_ors': opened, 'surgeon_assignments': sassign,
            'or_sequences': or_seq, 'surgeon_sequences': surg_seq}


# --------------------------------------------------------------------------- #
# Greedy constructive heuristic (warm start / fallback)
# --------------------------------------------------------------------------- #
def greedy(I):
    P = I['P']
    alpha = I['alpha']
    room_end, room_load = {}, defaultdict(float)
    room_jobs = defaultdict(list)
    surg_hosp, surg_free, surg_first = {}, {}, {}
    surg_work = defaultdict(float)
    sched = {}

    def attempt(pid, s, h, d, r):
        tps = I['tsur'][pid][s]
        T = I['prep'][pid] + tps + I['clean'][pid]
        rf = room_end.get((h, d, r), 0.0)
        sf = surg_free.get((s, d), 0.0)
        start = max(rf + I['prep'][pid], sf)
        fin, ext = start + tps, start + tps + I['clean'][pid]
        rg, mo = I['reg'][(h, d, r)], I['mot'][(h, d, r)]
        if room_load[(h, d, r)] + T > rg + mo + 1e-6:
            return None
        if ext > rg + mo + 1e-6:
            return None
        av = I['avail'][s][d]
        first = surg_first.get((s, d))
        nf = first if first is not None else start
        if fin - nf > av + 1e-6:
            return None
        w = alpha * tps + (1.0 - alpha) * T
        if surg_work[(s, d)] + w > av + 1e-6:
            return None
        delta = 0.0
        if not room_jobs[(h, d, r)]:
            delta += I['open_cost'][(h, d, r)]
        if (s, d) not in surg_hosp:
            delta += I['scost'].get((h, s), 0.0)
        delta += (max(0.0, ext - rg) - max(0.0, rf - rg)) * I['ot_rate'][(h, r)]
        return delta, fin, start, ext, w

    def apply(pid, s, h, d, r, fin, start, ext, w):
        room_end[(h, d, r)] = ext
        room_load[(h, d, r)] += (I['prep'][pid] + I['tsur'][pid][s]
                                 + I['clean'][pid])
        room_jobs[(h, d, r)].append(pid)
        if (s, d) not in surg_hosp:
            surg_hosp[(s, d)] = h
            surg_first[(s, d)] = start
        surg_free[(s, d)] = fin
        surg_work[(s, d)] += w
        sched[pid] = (s, h, d, r, fin, start, ext)

    mlist = [pid for pid in P if I['mand'][pid]]
    mlist.sort(key=lambda pid: (I['due'][pid], pid))
    for pid in mlist:
        best = None
        for (p, s, h, d, r) in I['tuples_by_p'][pid]:
            if surg_hosp.get((s, d), h) != h:
                continue
            res = attempt(pid, s, h, d, r)
            if res is None:
                continue
            key = (round(res[0], 9), d, res[1])
            if best is None or key < best[0]:
                best = (key, (s, h, d, r) + tuple(res[1:]))
        if best is None:
            return None
        apply(pid, *best[1])

    olist = [pid for pid in P if not I['mand'][pid] and I['reward'][pid] > 0]
    olist.sort(key=lambda pid: (-I['reward'][pid], pid))
    for pid in olist:
        best = None
        for (p, s, h, d, r) in I['tuples_by_p'][pid]:
            if surg_hosp.get((s, d), h) != h:
                continue
            res = attempt(pid, s, h, d, r)
            if res is None:
                continue
            if res[0] >= I['reward'][pid] - 1e-9:
                continue
            if best is None or res[0] < best[0]:
                best = (res[0], (s, h, d, r) + tuple(res[1:]))
        if best is not None:
            apply(pid, *best[1])

    return build_from_state(I, sched)


# --------------------------------------------------------------------------- #
# MIP solution extraction
# --------------------------------------------------------------------------- #
def extract_mip(I, xvars, fvars, get):
    sched = {}
    for pid in I['P']:
        bt, bv = None, 0.5
        for t in I['tuples_by_p'][pid]:
            v = get(xvars[t])
            if v > bv:
                bv, bt = v, t
        if bt is None:
            continue
        p, s, h, d, r = bt
        tps = I['tsur'][p][s]
        fin = float(get(fvars[pid]))
        fin = max(fin, I['prep'][p] + tps)
        sched[pid] = (s, h, d, r, fin, fin - tps, fin + I['clean'][p])
    shifted = left_shift(I, sched)
    if verify(I, shifted):
        sched = shifted
    return build_from_state(I, sched)


# --------------------------------------------------------------------------- #
# Full MIP
# --------------------------------------------------------------------------- #
def run_mip(I, deadline, logger, warm):
    P, S, H, days = I['P'], I['S'], I['H'], I['days']
    tb = I['tuples_by_p']
    all_tuples = [t for pid in P for t in tb[pid]]
    if not all_tuples:
        return None

    by_hdr = defaultdict(list)
    by_hdr_p = defaultdict(lambda: defaultdict(list))
    by_shd = defaultdict(list)
    by_shd_p = defaultdict(lambda: defaultdict(list))
    by_sd_p = defaultdict(lambda: defaultdict(list))
    for t in all_tuples:
        p, s, h, d, r = t
        by_hdr[(h, d, r)].append(t)
        by_hdr_p[(h, d, r)][p].append(t)
        by_shd[(s, h, d)].append(t)
        by_shd_p[(s, h, d)][p].append(t)
        by_sd_p[(s, d)][p].append(t)

    pair_est = 0
    for dd in list(by_hdr_p.values()) + list(by_sd_p.values()):
        n = len(dd)
        pair_est += n * (n - 1) // 2
    if pair_est > 120000 or len(all_tuples) > 80000:
        return None  # too large; rely on greedy

    hdr_keys = [(h, d, r) for h in H for d in days for r in I['R_of_h'][h]]
    shd_keys = sorted(by_shd.keys())
    max_session = max((I['reg'][k] + I['mot'][k]) for k in hdr_keys) if hdr_keys else 0.0
    max_extra = max((I['prep'][p] + I['clean'][p] + max(I['tsur'][p].values())
                     for p in P), default=0.0)
    max_t = max((v for p in P for v in I['tsur'][p].values()), default=0.0)
    M1 = max_session + max_extra + 1.0
    M2 = max_session + max_t + 1.0
    alpha = I['alpha']

    model = gp.Model('coalition_surgical_scheduling')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)

    x = model.addVars(all_tuples, vtype=GRB.BINARY, name='x')
    openv = model.addVars(hdr_keys, vtype=GRB.BINARY, name='open')
    u = model.addVars(shd_keys, vtype=GRB.BINARY, name='u')
    ot = model.addVars(hdr_keys, lb=0.0, name='ot')
    for k in hdr_keys:
        ot[k].UB = max(0.0, I['mot'][k])
    comp = model.addVars(hdr_keys, lb=0.0, ub=max_session, name='comp')
    f = model.addVars(P, lb=0.0, ub=max_session, name='f')
    sst = model.addVars(shd_keys, lb=0.0, ub=max_session, name='sst')
    sen = model.addVars(shd_keys, lb=0.0, ub=max_session, name='sen')

    obj = gp.LinExpr()
    for k in hdr_keys:
        obj += I['open_cost'][k] * openv[k]
        obj += I['ot_rate'][(k[0], k[2])] * ot[k]
    for k in shd_keys:
        obj += I['scost'].get((k[1], k[0]), 0.0) * u[k]
    for pid in P:
        if I['reward'][pid] > 0:
            for t in tb[pid]:
                obj -= I['reward'][pid] * x[t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Patient assignment
    for pid in P:
        e = gp.quicksum(x[t] for t in tb[pid])
        if I['mand'][pid]:
            model.addConstr(e == 1)
        else:
            model.addConstr(e <= 1)

    # Linking: OR open / surgeon present
    for t in all_tuples:
        p, s, h, d, r = t
        model.addConstr(x[t] <= openv[(h, d, r)])
        model.addConstr(x[t] <= u[(s, h, d)])

    # Surgeon at at most one hospital per day
    by_sd_u = defaultdict(list)
    for (s, h, d) in shd_keys:
        by_sd_u[(s, d)].append(h)
    for (s, d), hs in by_sd_u.items():
        if len(hs) > 1:
            model.addConstr(gp.quicksum(u[(s, h, d)] for h in hs) <= 1)

    # OR aggregate capacity (regular + overtime)
    for k in hdr_keys:
        lst = by_hdr.get(k)
        if lst:
            model.addConstr(gp.quicksum(T_of(I, t) * x[t] for t in lst)
                            <= I['reg'][k] + ot[k])

    # Surgeon alpha-weighted workload capacity
    for k in shd_keys:
        lst = by_shd.get(k)
        if not lst:
            continue
        s, h, d = k
        e = gp.quicksum(
            (alpha * I['tsur'][t[0]][s] + (1.0 - alpha) * T_of(I, t)) * x[t]
            for t in lst)
        model.addConstr(e <= I['avail'][s][d] * u[k])

    # Finish-time lower bound: prep + surgical time
    for pid in P:
        if tb[pid]:
            model.addConstr(f[pid] >= gp.quicksum(
                (I['prep'][pid] + I['tsur'][pid][t[1]]) * x[t]
                for t in tb[pid]))

    # OR completion times and overtime
    for k in hdr_keys:
        for pid, lst in by_hdr_p[k].items():
            model.addConstr(comp[k] >= f[pid] + I['clean'][pid]
                            - M1 * (1 - gp.quicksum(x[t] for t in lst)))
        model.addConstr(ot[k] >= comp[k] - I['reg'][k])
        model.addConstr(comp[k] <= (I['reg'][k] + I['mot'][k]) * openv[k])

    # Surgeon start/end/span constraints
    for k in shd_keys:
        s, h, d = k
        for pid, lst in by_shd_p[k].items():
            model.addConstr(sen[k] >= f[pid]
                            - M1 * (1 - gp.quicksum(x[t] for t in lst)))
        model.addConstr(sen[k] - sst[k] <= I['avail'][s][d] * u[k])
    for t in all_tuples:
        p, s, h, d, r = t
        model.addConstr(sst[(s, h, d)] <= f[p] - I['tsur'][p][s]
                        + M1 * (1 - x[t]))

    # OR disjunctive sequencing constraints
    tsurg_expr = {pid: gp.quicksum(I['tsur'][pid][t[1]] * x[t]
                                   for t in tb[pid]) for pid in P}
    for k in hdr_keys:
        pids = sorted(by_hdr_p[k].keys())
        Xp = {pid: gp.quicksum(x[t] for t in by_hdr_p[k][pid]) for pid in pids}
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                p, q = pids[i], pids[j]
                dl = model.addVar(vtype=GRB.BINARY)
                model.addConstr(
                    f[p] >= f[q] + I['clean'][q] + I['prep'][p] + tsurg_expr[p]
                    - M1 * (3 - dl - Xp[p] - Xp[q]))
                model.addConstr(
                    f[q] >= f[p] + I['clean'][p] + I['prep'][q] + tsurg_expr[q]
                    - M1 * (2 + dl - Xp[p] - Xp[q]))

    # Surgeon disjunctive sequencing constraints
    for (s, d), dd in by_sd_p.items():
        pids = sorted(dd.keys())
        Y = {pid: gp.quicksum(x[t] for t in dd[pid]) for pid in pids}
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                p, q = pids[i], pids[j]
                mu = model.addVar(vtype=GRB.BINARY)
                model.addConstr(f[p] >= f[q] + I['tsur'][p][s]
                                - M2 * (3 - mu - Y[p] - Y[q]))
                model.addConstr(f[q] >= f[p] + I['tsur'][q][s]
                                - M2 * (2 + mu - Y[p] - Y[q]))

    # Warm start from greedy solution
    if warm:
        try:
            for a in warm.get('assignments', []):
                t = (a['patient_id'], a['surgeon_id'], a['hospital_id'],
                     a['day'], a['or_id'])
                if t in x:
                    x[t].Start = 1.0
                    f[a['patient_id']].Start = a['finish_time']
            for o in warm.get('opened_ors', []):
                k = (o['hospital_id'], o['day'], o['or_id'])
                if k in openv:
                    openv[k].Start = 1.0
                    comp[k].Start = o['completion_time']
                    ot[k].Start = o['overtime']
            for sa in warm.get('surgeon_assignments', []):
                k = (sa['surgeon_id'], sa['hospital_id'], sa['day'])
                if k in u:
                    u[k].Start = 1.0
                    sst[k].Start = sa['start_time']
                    sen[k].Start = sa['end_time']
        except Exception:
            pass

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                sol = extract_mip(I, x, f, lambda v: m.cbGetSolution(v))
                logger.log_solution(float(sol['objective_value']), sol)
            except Exception:
                try:
                    logger.log(float(m.cbGet(GRB.Callback.MIPSOL_OBJBST)))
                except Exception:
                    pass

    tl = deadline - time.time()
    if tl < 1.5:
        return None
    model.setParam('TimeLimit', max(1.0, tl - 1.0))
    model.optimize(cb)

    if model.SolCount > 0:
        try:
            return extract_mip(I, x, f, lambda v: v.X)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, required=True)
    ap.add_argument('--log_path', default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    logger = (SolutionLogger(args.log_path, sense='minimize')
              if args.log_path else None)
    t0 = time.time()
    with open(args.instance_path) as fh:
        data = json.load(fh)
    I = prepare(data)

    best = None
    try:
        best = greedy(I)
    except Exception:
        best = None
    if best is not None and logger:
        try:
            logger.log_solution(best['objective_value'], best)
        except Exception:
            pass

    deadline = t0 + float(args.time_limit)
    if time.time() < deadline - 2.0:
        try:
            sol = run_mip(I, deadline - 1.0, logger, best)
        except Exception:
            sol = None
        if sol is not None and (best is None or
                                sol['objective_value'] < best['objective_value'] - 1e-9):
            best = sol
            if logger:
                try:
                    logger.log_solution(best['objective_value'], best)
                except Exception:
                    pass

    if best is None:
        best = {'objective_value': 0.0, 'assignments': [], 'opened_ors': [],
                'surgeon_assignments': [], 'or_sequences': [],
                'surgeon_sequences': []}

    with open(args.solution_path, 'w') as fh:
        json.dump(best, fh, indent=2)


if __name__ == '__main__':
    main()