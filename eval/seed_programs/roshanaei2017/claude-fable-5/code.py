import argparse
import json
import time
import random
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    hard_deadline = t0 + max(5, args.time_limit) - 2.0
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    rng = random.Random(0)

    with open(args.instance_path) as f:
        inst = json.load(f)

    num_days = int(inst['num_days'])
    alpha = float(inst.get('alpha', 0.0))

    # ------------- derive day labels -------------
    dayset = set()
    for hd in inst['hospitals']:
        for orx in hd['ORs']:
            for k in (orx.get('daily') or {}):
                dayset.add(int(k))
    if not dayset:
        for sd in inst['surgeons']:
            dayset.update(int(d) for d in sd.get('operating_days', []))
    if not dayset:
        dayset = set(range(1, num_days + 1))
    days = sorted(dayset)
    if len(days) > num_days:
        days = days[:num_days]

    # ------------- parse data -------------
    pats = {}
    for pd in inst['patients']:
        pid = int(pd['patient_id'])
        pats[pid] = {
            'id': pid,
            'mand': bool(pd['is_mandatory']),
            'due': int(pd['due_date']),
            'prep': float(pd['preparation_time']),
            'clean': float(pd['cleaning_time']),
            'elig_s': [int(s) for s in pd.get('eligible_surgeons', [])],
            'el_or': {int(h): [int(r) for r in lst]
                      for h, lst in (pd.get('eligible_ORs_by_hospital') or {}).items()},
            'st': {int(s): float(v) for s, v in (pd.get('surgeon_specific_times') or {}).items()},
            'reward': float(pd.get('reward') or 0.0),
        }

    surg = {}
    for sd in inst['surgeons']:
        sid = int(sd['surgeon_id'])
        surg[sid] = {
            'av': {int(k): float(v) for k, v in (sd.get('availability_by_day') or {}).items()},
            'od': set(int(d) for d in sd.get('operating_days', [])),
        }

    reg = {}
    mot = {}
    opencost = {}
    otmin = {}
    sc = {}
    hosp_ids = []
    for hd in inst['hospitals']:
        h = int(hd['hospital_id'])
        hosp_ids.append(h)
        for orx in hd['ORs']:
            r = int(orx['or_id'])
            drt = float(orx.get('regular_time') or 0)
            dmo = float(orx.get('max_overtime') or 0)
            daily = orx.get('daily') or {}
            for d in days:
                dd = daily.get(str(d)) or {}
                rg = float(dd.get('regular_time', drt))
                mo = float(dd.get('max_overtime', dmo))
                reg[(h, d, r)] = rg
                mot[(h, d, r)] = mo
                opencost[(h, d, r)] = float(orx.get('fixed_cost_per_hour') or 0.0) * rg / 60.0
            otmin[(h, r)] = float(orx.get('overtime_cost_per_hour') or 0.0) / 60.0
        for s, v in (hd.get('surgeon_fixed_costs_per_day') or {}).items():
            sc[(h, int(s))] = float(v)

    def pdata(pid, s):
        p = pats[pid]
        dur = p['st'][s]
        T = p['prep'] + dur + p['clean']
        w = alpha * dur + (1.0 - alpha) * T
        return dur, T, w

    # ------------- enumerate feasible combos -------------
    combos = []
    for pid, p in pats.items():
        for s in p['elig_s']:
            if s not in surg or s not in p['st']:
                continue
            dur, T, w = pdata(pid, s)
            for d in days:
                if d not in surg[s]['od']:
                    continue
                if p['mand'] and d > p['due']:
                    continue
                av = surg[s]['av'].get(d, 0.0)
                if w > av + 1e-9:
                    continue
                for h in hosp_ids:
                    for r in p['el_or'].get(h, []):
                        key = (h, d, r)
                        if key not in reg:
                            continue
                        cap = reg[key] + mot[key]
                        if cap <= 1e-9 or T > cap + 1e-9:
                            continue
                        combos.append((pid, s, h, d, r))

    # ------------- sequencing helpers -------------
    def simulate(plist, lrng, it):
        room_free = {}
        surg_free = {}
        last_sr = {}
        rem = list(plist)
        out = {}
        slack = 0.0 if it == 0 else (0, 0, 5, 15, 30, 60)[lrng.randrange(6)]
        while rem:
            m_st = None
            scored = []
            for q in rem:
                st = room_free.get(q['r'], 0.0) + q['prep']
                sf = surg_free.get(q['s'], 0.0)
                if sf > st:
                    st = sf
                scored.append((st, q))
                if m_st is None or st < m_st:
                    m_st = st
            cands = [t for t in scored if t[0] <= m_st + slack]
            if it == 0:
                cands.sort(key=lambda t: (t[0],
                                          0 if last_sr.get(t[1]['r']) == t[1]['s'] else 1,
                                          -t[1]['dur'], t[1]['p']))
                st, q = cands[0]
            else:
                pref = [t for t in cands if last_sr.get(t[1]['r']) == t[1]['s']]
                if pref and lrng.random() < 0.6:
                    cands = pref
                st, q = cands[lrng.randrange(len(cands))]
                st = max(room_free.get(q['r'], 0.0) + q['prep'], surg_free.get(q['s'], 0.0))
            fin = st + q['dur']
            out[q['p']] = (st - q['prep'], st, fin, fin + q['clean'])
            room_free[q['r']] = fin + q['clean']
            surg_free[q['s']] = fin
            last_sr[q['r']] = q['s']
            rem.remove(q)
        return out

    def sequence_day(h, d, plist, tries, lrng):
        sums = defaultdict(float)
        for q in plist:
            sums[q['r']] += q['T']
        lb_ot = sum(max(0.0, sums[r] - reg[(h, d, r)]) for r in sums)
        n = len(plist)
        tries_eff = 1 if n <= 1 else min(tries, 20 + 10 * n)
        best = None
        for it in range(tries_eff):
            sch = simulate(plist, lrng, it)
            comp = {}
            smin = {}
            smax = {}
            for q in plist:
                e = sch[q['p']]
                if e[3] > comp.get(q['r'], 0.0):
                    comp[q['r']] = e[3]
                s = q['s']
                if s not in smin or e[1] < smin[s]:
                    smin[s] = e[1]
                if s not in smax or e[2] > smax[s]:
                    smax[s] = e[2]
            viol = 0.0
            otc = 0.0
            csum = 0.0
            ot = {}
            for r, c in comp.items():
                oo = max(0.0, c - reg[(h, d, r)])
                ot[r] = oo
                if oo > mot[(h, d, r)]:
                    viol += oo - mot[(h, d, r)]
                otc += oo * otmin[(h, r)]
                csum += c
            for s in smin:
                span = smax[s] - smin[s]
                av = surg[s]['av'].get(d, 0.0)
                if span > av:
                    viol += span - av
            scr = (viol, otc, csum)
            if best is None or scr < best['scr']:
                best = {'scr': scr, 'sch': sch, 'comp': comp, 'ot': ot,
                        'smin': smin, 'smax': smax}
            if scr[0] <= 1e-9 and scr[1] <= lb_ot + 1e-6:
                break
        return best

    def find_slot(pid, assign, forbid):
        p = pats[pid]
        cur = assign[pid]
        orl = defaultdict(float)
        sgw = defaultdict(float)
        sgh = defaultdict(set)
        for q2, (s2, h2, d2, r2) in assign.items():
            if q2 == pid:
                continue
            _, T2, w2 = pdata(q2, s2)
            orl[(h2, d2, r2)] += T2
            sgw[(s2, h2, d2)] += w2
            sgh[(s2, d2)].add(h2)
        for buf_or, buf_sg in ((0.9, 0.95), (1.0, 1.0)):
            best = None
            for s in p['elig_s']:
                if s not in surg or s not in p['st']:
                    continue
                dur, T, w = pdata(pid, s)
                for d in days:
                    if d not in surg[s]['od']:
                        continue
                    if p['mand'] and d > p['due']:
                        continue
                    av = surg[s]['av'].get(d, 0.0)
                    for h in hosp_ids:
                        hs = sgh.get((s, d))
                        if hs and h not in hs:
                            continue
                        if forbid[0] == 'sg' and (h, d, s) == (forbid[1], forbid[2], forbid[3]):
                            continue
                        if sgw[(s, h, d)] + w > buf_sg * av + 1e-9:
                            continue
                        for r in p['el_or'].get(h, []):
                            key = (h, d, r)
                            if key not in reg:
                                continue
                            if forbid[0] == 'or' and key == (forbid[1], forbid[2], forbid[3]):
                                continue
                            if (s, h, d, r) == cur:
                                continue
                            if orl[key] + T > reg[key] + buf_or * mot[key] + 1e-9:
                                continue
                            add = 0.0
                            if orl[key] <= 1e-9:
                                add += opencost[key]
                            if sgw[(s, h, d)] <= 1e-9:
                                add += sc.get((h, s), 0.0)
                            otm = otmin[(h, r)]
                            add += (max(0.0, orl[key] + T - reg[key]) -
                                    max(0.0, orl[key] - reg[key])) * otm
                            s0, h0, d0, r0 = cur
                            _, T0, _ = pdata(pid, s0)
                            k0 = (h0, d0, r0)
                            if orl[k0] <= 1e-9:
                                add -= opencost[k0]
                            if sgw[(s0, h0, d0)] <= 1e-9:
                                add -= sc.get((h0, s0), 0.0)
                            otm0 = otmin[(h0, r0)]
                            add -= (max(0.0, orl[k0] + T0 - reg[k0]) -
                                    max(0.0, orl[k0] - reg[k0])) * otm0
                            if best is None or add < best[0]:
                                best = (add, (s, h, d, r))
            if best is not None:
                return best[1]
        return None

    def make_daymap(assign):
        daymap = defaultdict(list)
        for pid, (s, h, d, r) in assign.items():
            dur, T, _ = pdata(pid, s)
            daymap[(h, d)].append({'p': pid, 's': s, 'r': r,
                                   'prep': pats[pid]['prep'],
                                   'dur': dur, 'clean': pats[pid]['clean'], 'T': T})
        return daymap

    def build_full(assign0, tries, bdeadline, seed=0):
        lrng = random.Random(seed)
        assign = dict(assign0)
        cache = {}
        escalated = set()

        def compute_results(daymap):
            results = {}
            for key, plist in daymap.items():
                ck = (key, frozenset(q['p'] for q in plist), key in escalated)
                if ck not in cache:
                    tt = tries * (4 if key in escalated else 1)
                    cache[ck] = sequence_day(key[0], key[1], plist, tt, lrng)
                results[key] = cache[ck]
            return results

        for _ in range(150):
            daymap = make_daymap(assign)
            results = compute_results(daymap)
            viols = []
            for key, res in results.items():
                h, d = key
                for r, oo in res['ot'].items():
                    if oo - mot[(h, d, r)] > 1e-6:
                        viols.append(('or', h, d, r, oo - mot[(h, d, r)]))
                for s in res['smin']:
                    span = res['smax'][s] - res['smin'][s]
                    av = surg[s]['av'].get(d, 0.0)
                    if span - av > 1e-6:
                        viols.append(('sg', h, d, s, span - av))
            if not viols or time.time() > bdeadline:
                return assign, results, (len(viols) == 0)
            vt, vh, vd, vk, _ = max(viols, key=lambda t: t[4])
            plist = daymap[(vh, vd)]
            if vt == 'or':
                cands = [q for q in plist if q['r'] == vk]
            else:
                cands = [q for q in plist if q['s'] == vk]
            opts = sorted([q for q in cands if not pats[q['p']]['mand']],
                          key=lambda q: pats[q['p']]['reward'])
            mands = sorted([q for q in cands if pats[q['p']]['mand']],
                           key=lambda q: -q['T'])
            moved = False
            forbid = (vt, vh, vd, vk)
            for q in opts + mands:
                slot = find_slot(q['p'], assign, forbid)
                if slot is not None:
                    assign[q['p']] = slot
                    moved = True
                    break
            if not moved and opts:
                del assign[opts[0]['p']]
                moved = True
            if not moved:
                if (vh, vd) not in escalated:
                    escalated.add((vh, vd))
                else:
                    return assign, results, False
        # loop exhausted: recompute results so they match final assignment
        daymap = make_daymap(assign)
        results = compute_results(daymap)
        return assign, results, False

    def finalize(assign, results):
        used_or = defaultdict(list)
        used_s = defaultdict(list)
        asg_out = []
        reward_sum = 0.0
        feas = True
        for pid, (s, h, d, r) in assign.items():
            res = results[(h, d)]
            entry, stt, fin, ext = res['sch'][pid]
            dur, T, _ = pdata(pid, s)
            asg_out.append({
                'patient_id': pid, 'surgeon_id': s, 'hospital_id': h, 'day': d, 'or_id': r,
                'is_mandatory': pats[pid]['mand'], 'T_ps': T, 'finish_time': fin,
                'surgery_start_time': stt, 'room_entry_time': entry, 'room_exit_time': ext})
            used_or[(h, d, r)].append((pid, fin, stt))
            used_s[(s, h, d)].append((pid, fin, stt))
            if not pats[pid]['mand']:
                reward_sum += pats[pid]['reward']
        opened = []
        or_cost = 0.0
        ot_cost = 0.0
        for (h, d, r), lst in used_or.items():
            res = results[(h, d)]
            comp = res['comp'][r]
            oo = max(0.0, comp - reg[(h, d, r)])
            if oo > mot[(h, d, r)] + 1e-6:
                feas = False
            opened.append({'hospital_id': h, 'day': d, 'or_id': r,
                           'regular_time': int(round(reg[(h, d, r)])),
                           'completion_time': comp, 'overtime': oo})
            or_cost += opencost[(h, d, r)]
            ot_cost += oo * otmin[(h, r)]
        sa = []
        s_cost = 0.0
        for (s, h, d), lst in used_s.items():
            st_min = min(t[2] for t in lst)
            f_max = max(t[1] for t in lst)
            if f_max - st_min > surg[s]['av'].get(d, 0.0) + 1e-6:
                feas = False
            sa.append({'surgeon_id': s, 'hospital_id': h, 'day': d,
                       'start_time': st_min, 'end_time': f_max})
            s_cost += sc.get((h, s), 0.0)
        orseq = []
        for (h, d, r), lst in used_or.items():
            lst2 = sorted(lst, key=lambda t: t[1])
            for i in range(len(lst2)):
                for j in range(i + 1, len(lst2)):
                    orseq.append({'hospital_id': h, 'day': d, 'or_id': r,
                                  'patient_p': lst2[j][0], 'patient_k': lst2[i][0],
                                  'p_after_k': 1})
        sgseq = []
        for (s, h, d), lst in used_s.items():
            lst2 = sorted(lst, key=lambda t: t[1])
            for i in range(len(lst2)):
                for j in range(i + 1, len(lst2)):
                    sgseq.append({'hospital_id': h, 'day': d, 'surgeon_id': s,
                                  'patient_p': lst2[j][0], 'patient_k': lst2[i][0],
                                  'p_after_k': 1})
        for pid, p in pats.items():
            if p['mand'] and pid not in assign:
                feas = False
        obj = or_cost + s_cost + ot_cost - reward_sum
        sol = {'objective_value': obj, 'assignments': asg_out, 'opened_ors': opened,
               'surgeon_assignments': sa, 'or_sequences': orseq, 'surgeon_sequences': sgseq}
        return sol, obj, feas

    best = None  # (feas, obj, sol, assign)

    def consider(assign, tries, bdl, seed):
        nonlocal best
        try:
            a2, results, _ = build_full(assign, tries, bdl, seed)
            sol, obj, feas = finalize(a2, results)
        except Exception:
            return
        key = (0 if feas else 1, obj)
        cur = (0 if best[0] else 1, best[1]) if best is not None else None
        if cur is None or key < cur:
            best = (feas, obj, sol, dict(a2))
            if feas and logger:
                logger.log_solution(obj, sol)

    # ------------- MIP -------------
    last_assign = None
    if combos:
        try:
            env = gp.Env(params={'OutputFlag': 0})
            m = gp.Model(env=env)
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.MIPFocus = 1

            Tc = {}
            Wc = {}
            for c in combos:
                pid, s, h, d, r = c
                dur, T, w = pdata(pid, s)
                Tc[c] = T
                Wc[c] = w

            x = {}
            for c in combos:
                pid = c[0]
                ob = -pats[pid]['reward'] if not pats[pid]['mand'] else 0.0
                x[c] = m.addVar(vtype=GRB.BINARY, obj=ob)
            ys = sorted(set((h, d, r) for _, _, h, d, r in combos))
            zs = sorted(set((s, h, d) for _, s, h, d, _ in combos))
            y = {k: m.addVar(vtype=GRB.BINARY, obj=opencost[k]) for k in ys}
            z = {k: m.addVar(vtype=GRB.BINARY, obj=sc.get((k[1], k[0]), 0.0)) for k in zs}
            o = {k: m.addVar(lb=0.0, ub=mot[k], obj=otmin[(k[0], k[2])]) for k in ys}

            by_p = defaultdict(list)
            by_or = defaultdict(list)
            by_psh = defaultdict(list)
            by_shd = defaultdict(list)
            for c in combos:
                pid, s, h, d, r = c
                by_p[pid].append(c)
                by_or[(h, d, r)].append(c)
                by_psh[(pid, s, h, d)].append(c)
                by_shd[(s, h, d)].append(c)

            PEN = 1e7
            for pid, p in pats.items():
                lst = by_p.get(pid, [])
                if p['mand']:
                    uv = m.addVar(vtype=GRB.BINARY, obj=PEN)
                    m.addConstr(gp.quicksum(x[c] for c in lst) + uv == 1)
                elif lst:
                    m.addConstr(gp.quicksum(x[c] for c in lst) <= 1)
            for k, lst in by_or.items():
                expr = gp.quicksum(Tc[c] * x[c] for c in lst)
                m.addConstr(expr <= reg[k] + o[k])
                m.addConstr(expr <= (reg[k] + mot[k]) * y[k])
                m.addConstr(o[k] <= mot[k] * y[k])
            for k, lst in by_psh.items():
                m.addConstr(gp.quicksum(x[c] for c in lst) <= z[(k[1], k[2], k[3])])
            for (s, h, d), lst in by_shd.items():
                m.addConstr(gp.quicksum(Wc[c] * x[c] for c in lst)
                            <= surg[s]['av'].get(d, 0.0) * z[(s, h, d)])
            zz = defaultdict(list)
            for (s, h, d) in zs:
                zz[(s, d)].append((s, h, d))
            for k, lst in zz.items():
                if len(lst) > 1:
                    m.addConstr(gp.quicksum(z[t] for t in lst) <= 1)
            m.ModelSense = GRB.MINIMIZE

            xlist = [x[c] for c in combos]

            def extract():
                vals = m.getAttr('X', xlist)
                a = {}
                for c, v in zip(combos, vals):
                    if v > 0.5:
                        a[c[0]] = (c[1], c[2], c[3], c[4])
                return a

            mip_deadline = min(hard_deadline - 2.0, t0 + 0.8 * args.time_limit)
            chunk = max(8.0, args.time_limit / 8.0)
            last_mip = float('inf')
            while time.time() < mip_deadline:
                m.Params.TimeLimit = max(1.0, min(chunk, mip_deadline - time.time()))
                try:
                    m.optimize()
                except gp.GurobiError:
                    break
                if m.SolCount > 0 and m.ObjVal < last_mip - 1e-6:
                    last_mip = m.ObjVal
                    last_assign = extract()
                    bdl = min(hard_deadline, time.time() + max(3.0, 0.05 * args.time_limit))
                    consider(last_assign, 60, bdl, seed=rng.randrange(10 ** 6))
                if m.Status != GRB.TIME_LIMIT:
                    break
        except Exception:
            pass

    # ------------- polish sequencing with remaining time -------------
    base_assign = None
    if last_assign is not None:
        base_assign = last_assign
    elif best is not None:
        base_assign = best[3]
    seed_i = 1
    feas_rounds = 0
    while base_assign is not None and time.time() < hard_deadline - 0.5:
        consider(base_assign, 150, hard_deadline - 0.2, seed=seed_i)
        seed_i += 1
        if best is not None and best[0]:
            feas_rounds += 1
            if feas_rounds >= 6:
                break
        if seed_i > 40:
            break

    # ------------- write output -------------
    if best is not None:
        sol = best[2]
        if best[0] and logger:
            logger.log_solution(best[1], sol)
    else:
        sol = {'objective_value': 0.0, 'assignments': [], 'opened_ors': [],
               'surgeon_assignments': [], 'or_sequences': [], 'surgeon_sequences': []}
        if not any(p['mand'] for p in pats.values()) and logger:
            logger.log_solution(0.0, sol)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=1)


if __name__ == '__main__':
    main()