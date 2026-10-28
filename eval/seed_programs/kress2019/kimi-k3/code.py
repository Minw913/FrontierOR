import argparse
import json
import time
import random

import numpy as np


def solve(inst, time_limit, logger, sol_path):
    t_start = time.time()
    budget = max(2.0, float(time_limit))
    t_hard = t_start + budget - 0.8

    S = int(inst['S']); Q = S + 1
    p = int(inst['p'])
    sigma_w = int(inst['sigma_w']); sigma_l = int(inst['sigma_l'])
    sea = sorted(inst.get('seaside_containers') or [], key=lambda c: c['id'])
    land = sorted(inst.get('landside_containers') or [], key=lambda c: (c['deadline'], c['id']))
    n = len(sea); m = len(land)
    A = [int(c['target_slot']) for c in sea]
    sea_id = [int(c['id']) for c in sea]
    land_id = [int(c['id']) for c in land]
    land_src = [int(c['source_slot']) for c in land]
    land_eft = [int(c['earliest_finish_time']) for c in land]
    land_ddl = [int(c['deadline']) for c in land]
    max_eft = max(land_eft) if m else 0

    def time_ok():
        return time.time() < t_hard

    # ---------- seaside crane trajectory (fixed pickup order at slot 0) ----------
    def build_W(drop_slots):
        w = [sigma_w]
        pos = sigma_w
        t = 0
        drop = []
        for d in drop_slots:
            while pos > 0:
                pos -= 1; t += 1; w.append(pos)
            for _ in range(p):                      # lift at slot 0
                t += 1; w.append(pos)
            while pos < d:
                pos += 1; t += 1; w.append(pos)
            drop.append(t)
            for _ in range(p):                      # drop at slot d
                t += 1; w.append(pos)
        while pos > 0:                              # park at 0 to free space for L
            pos -= 1; t += 1; w.append(pos)
        return w, drop, t

    # ---------- earliest start of a p-length stationary stay at slot s ----------
    # L may be at slot s during integer times [T, T+p]; T >= nb; obstacle is w_arr.
    def earliest_presence(w_arr, Hmax, t0, p0, s, nb):
        if t0 >= nb and p0 == s and int(w_arr[t0:t0 + p + 1].max()) < s:
            return t0, [p0]
        reach = np.zeros(Q + 1, dtype=bool)
        reach[p0] = True
        hist = [reach]
        pp1 = p + 1
        for T in range(t0 + 1, Hmax + 1):
            new = reach.copy()
            new[1:] |= reach[:-1]
            new[:-1] |= reach[1:]
            wt = int(w_arr[T])
            new[:wt + 1] = False                    # L must stay strictly right of W
            if T >= nb and new[s] and int(w_arr[T:T + pp1].max()) < s:
                L = T - t0
                path = [0] * (L + 1)
                path[L] = s
                cur = s
                for tau in range(L, 0, -1):
                    prev = hist[tau - 1]
                    if prev[cur]:
                        c = cur
                    elif cur > 1 and prev[cur - 1]:
                        c = cur - 1
                    elif cur < Q and prev[cur + 1]:
                        c = cur + 1
                    else:
                        c = cur
                    path[tau - 1] = c
                    cur = c
                return T, path
            hist.append(new)
            reach = new
        return None

    def run_schedule(w_arr, Hmax, order, strict_land=True):
        lpos = [sigma_l]
        t = 0; pos = sigma_l
        events = {}
        for task in order:
            nb = task.get('ready', 0)
            r1 = earliest_presence(w_arr, Hmax, t, pos, task['lift'], nb)
            if r1 is None:
                if not task['required']:
                    continue
                return None
            T1, path1 = r1
            t1 = T1 + p
            pos1 = task['lift']
            nb2 = t1
            if task['kind'] == 'land':
                nb2 = max(t1, task['eft'])
            r2 = earliest_presence(w_arr, Hmax, t1, pos1, task['drop'], nb2)
            if r2 is None:
                if not task['required']:
                    continue
                return None
            if task['kind'] == 'land':
                lim = task['deadline'] - (p if strict_land else 0)
                if r2[0] > lim:
                    if not task['required']:
                        continue
                    return None
            T2, path2 = r2
            lpos.extend(path1[1:])
            lpos.extend([task['lift']] * p)
            lpos.extend(path2[1:])
            lpos.extend([task['drop']] * p)
            t = T2 + p
            pos = task['drop']
            events[task['key']] = (T1, T2)
        return lpos, events, t

    def make_land_task(j, req):
        return {'kind': 'land', 'key': ('l', j), 'cid': j, 'lift': land_src[j],
                'drop': Q, 'eft': land_eft[j], 'deadline': land_ddl[j],
                'due': land_ddl[j], 'required': req}

    eval_cache = {}

    def evaluate(assignment):
        key = tuple(assignment)
        if key in eval_cache:
            return eval_cache[key]
        drop_slots = [assignment[i] if assignment[i] > 0 else A[i] for i in range(n)]
        w, drop_s, T_w = build_W(drop_slots)

        def compute_T(events):
            Tm = 0
            for i in range(n):
                if assignment[i] > 0:
                    f = events[('c', i)][1] + p
                else:
                    f = drop_s[i] + p
                if f > Tm:
                    Tm = f
            return Tm

        required = set()
        best = None
        w_arr = None
        Hmax = 0
        for _ in range(25):
            coop = []
            for i in range(n):
                if assignment[i] > 0:
                    x = assignment[i]
                    coop.append({'kind': 'coop', 'key': ('c', i), 'cid': i, 'lift': x,
                                 'drop': A[i], 'ready': drop_s[i],
                                 'due': drop_s[i] + p + (A[i] - x) + p,
                                 'ord': i, 'required': True})
            req_tasks = [make_land_task(j, True) for j in range(m) if j in required]
            opt_tasks = [make_land_task(j, False) for j in range(m) if j not in required]
            merge = coop + req_tasks
            coop_ord = sorted(coop, key=lambda tk: tk['ord'])
            req_ord = sorted(req_tasks, key=lambda tk: (tk['due'], tk['cid']))
            orders = [
                sorted(merge, key=lambda tk: (tk['due'], 0 if tk['kind'] == 'land' else 1, tk['cid'])) + opt_tasks,
                coop_ord + req_ord + opt_tasks,
                req_ord + coop_ord + opt_tasks,
            ]
            ntasks = len(merge) + len(opt_tasks)
            Hmax = T_w + ntasks * (2 * p + 2 * Q + 12) + max_eft + 130 + p
            w_arr = np.zeros(Hmax + p + 12, dtype=np.int32)
            w_arr[:len(w)] = w
            if len(w) < len(w_arr):
                w_arr[len(w):] = w[-1]
            cand = None
            for strict, skip_req in ((True, False), (False, False), (False, True)):
                for order in orders:
                    o = order
                    if skip_req:
                        o = [dict(tk) for tk in order]
                        for tk in o:
                            if tk['kind'] == 'land':
                                tk['required'] = False
                    r = run_schedule(w_arr, Hmax, o, strict_land=strict)
                    if r is None:
                        continue
                    lpos, events, _ = r
                    Tm = compute_T(events)
                    if cand is None or Tm < cand['T']:
                        cand = {'T': Tm, 'lpos': lpos, 'events': events,
                                'order': o, 'merge_len': len(merge)}
                if cand is not None or not time_ok():
                    break
            if cand is None:
                if best is not None:
                    break
                eval_cache[key] = None
                return None
            best = cand
            newreq = {j for j in range(m) if land_ddl[j] <= best['T']}
            if newreq <= required:
                break
            required |= newreq
            if not time_ok():
                break

        # ensure at least one container of the next deadline group after makespan
        Tm = best['T']
        later = [j for j in range(m) if land_ddl[j] > Tm]
        if later:
            Dstar = min(land_ddl[j] for j in later)
            group = [j for j in later if land_ddl[j] == Dstar]
            sched_land = {k[1] for k in best['events'] if k[0] == 'l'}
            if not any(j in sched_land for j in group):
                for j in sorted(group, key=lambda jj: (land_eft[jj], jj)):
                    task = make_land_task(j, False)
                    order = best['order']
                    ml = best['merge_len']
                    new_order = order[:ml] + [task] + [tk for tk in order[ml:]
                                                       if not (tk['kind'] == 'land' and tk['cid'] == j)]
                    r = run_schedule(w_arr, Hmax, new_order, strict_land=False)
                    if r is None:
                        continue
                    lpos2, events2, _ = r
                    if ('l', j) not in events2:
                        continue
                    best['lpos'] = lpos2
                    best['events'] = events2
                    best['order'] = new_order
                    break

        best['assignment'] = list(assignment)
        best['w'] = w
        best['drop_s'] = drop_s
        best['drop_slots'] = drop_slots
        eval_cache[key] = best
        return best

    def build_solution(res):
        assignment = res['assignment']
        w = res['w']; lpos = res['lpos']
        H = max(len(w), len(lpos)) - 1
        wp = w + [w[-1]] * (H + 1 - len(w))
        lp = lpos + [lpos[-1]] * (H + 1 - len(lpos))
        sea_events = []
        for i in range(n):
            e = {'container_id': sea_id[i], 'target_slot': A[i],
                 'seaside_drop_time': res['drop_s'][i],
                 'seaside_drop_slot': res['drop_slots'][i]}
            if assignment[i] > 0:
                lt, dt = res['events'][('c', i)]
                e['landside_lift_time'] = lt
                e['landside_lift_slot'] = assignment[i]
                e['landside_drop_time'] = dt
            sea_events.append(e)
        land_events = []
        for j in range(m):
            if ('l', j) in res['events']:
                lt, dt = res['events'][('l', j)]
                land_events.append({'container_id': land_id[j],
                                    'source_slot': land_src[j],
                                    'lift_time': lt, 'drop_time': dt})
        return {'objective_value': float(res['T']),
                'crane_w_positions': {str(t): wp[t] for t in range(H + 1)},
                'crane_l_positions': {str(t): lp[t] for t in range(H + 1)},
                'seaside_events': sea_events,
                'landside_events': land_events}

    best_res = None
    best_sol = None

    def consider(res):
        nonlocal best_res, best_sol
        if res is None:
            return
        if best_res is None or res['T'] < best_res['T']:
            best_res = res
            sol = build_solution(res)
            best_sol = sol
            if logger is not None:
                try:
                    logger.log_solution(float(res['T']), sol)
                except Exception:
                    pass

    def assign_from_theta(th):
        asg = []
        for i in range(n):
            a = A[i]
            if a == Q:
                asg.append(max(1, min(th, S)))
            elif a > th:
                asg.append(max(1, min(th, a - 1)))
            else:
                asg.append(0)
        return asg

    try:
        # ---- parameter sweep over handover threshold theta ----
        cand_thetas = set()
        for a in A:
            cand_thetas.add(max(1, min(a, S)))
            if a >= 2:
                cand_thetas.add(max(1, min(a - 1, S)))
        cand_thetas.add(max(1, S))
        thetas = sorted(t for t in cand_thetas if 1 <= t)
        cap = 12 if budget >= 20 else 7
        if len(thetas) > cap:
            idxs = sorted(set(np.linspace(0, len(thetas) - 1, cap).round().astype(int).tolist()))
            thetas = [thetas[i] for i in idxs]
        thetas = [Q] + [t for t in thetas if t != Q]  # all-direct baseline first

        for th in thetas:
            if not time_ok():
                break
            consider(evaluate(assign_from_theta(th)))

        if time_ok() and n:
            asg = [max(1, min(A[i] - 1, S)) if A[i] >= 2 else 0 for i in range(n)]
            consider(evaluate(asg))

        # ---- local search on per-container mode / handover slot ----
        mutable = [i for i in range(n) if min(A[i] - 1, S) >= 1]
        if best_res is not None and mutable:
            random.seed(0)
            cur = list(best_res['assignment'])
            stall = 0
            max_stall = 3 * len(mutable) + 15
            while time_ok() and stall < max_stall:
                i = random.choice(mutable)
                a = A[i]
                maxx = min(a - 1, S)
                opts = []
                if a <= S:
                    opts.append(0)
                base = cur[i] if cur[i] > 0 else maxx
                for dx in (0, -1, 1, -2, 2, -4, 4, -8, 8):
                    v = base + dx
                    if 1 <= v <= maxx:
                        opts.append(v)
                opts.extend([maxx, max(1, maxx // 2), 1])
                seen = set()
                cand_opts = []
                for v in opts:
                    if v != cur[i] and v not in seen:
                        seen.add(v)
                        cand_opts.append(v)
                random.shuffle(cand_opts)
                improved = False
                for v in cand_opts[:3]:
                    if not time_ok():
                        break
                    new = list(cur)
                    new[i] = v
                    r = evaluate(new)
                    if r is not None and r['T'] < best_res['T']:
                        consider(r)
                        cur = list(best_res['assignment'])
                        improved = True
                        break
                stall = 0 if improved else stall + 1
    except Exception:
        pass

    if best_sol is None:
        best_sol = {'objective_value': 0.0,
                    'crane_w_positions': {'0': sigma_w},
                    'crane_l_positions': {'0': sigma_l},
                    'seaside_events': [],
                    'landside_events': []}
    with open(sol_path, 'w') as f:
        json.dump(best_sol, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense='minimize')

    with open(args.instance_path) as f:
        inst = json.load(f)
    solve(inst, args.time_limit, logger, args.solution_path)


if __name__ == '__main__':
    main()