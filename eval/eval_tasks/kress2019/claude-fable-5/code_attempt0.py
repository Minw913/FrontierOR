import json
import argparse
import time
from collections import deque

from solution_logger import SolutionLogger


def simulate(inst, sea, lsc, Kset, hmap, eager, strict, buffer_, hard_deadline):
    """Step-by-step simulation of both cranes. Returns result dict or None on failure."""
    S = inst['S']; n = inst['n']; p = inst['p']
    SINK = S + 1
    targets = [c['target_slot'] for c in sea]
    coop = [hmap[i] is not None for i in range(n)]
    dropslot = [hmap[i] if coop[i] else targets[i] for i in range(n)]

    order = sorted(range(len(lsc)), key=lambda j: (lsc[j]['deadline'], j))
    ls_list = [j for j in order if j in Kset]

    maxeft = 0
    for j in ls_list:
        if lsc[j]['earliest_finish_time'] > maxeft:
            maxeft = lsc[j]['earliest_finish_time']
    tmax = maxeft + (n + len(ls_list) + 4) * (2 * (S + 2) + 2 * p + 10) + 8 * (S + 2) + 300

    posW = inst['sigma_w']; posL = inst['sigma_l']
    if not (0 <= posW < posL <= SINK):
        return None

    busyW = 0; busyL = 0
    wpend = None; lpend = None
    wi = 0
    wphase = 'topick' if n > 0 else 'done'
    ltask = None
    ls_ptr = 0
    coopQ = deque()
    announced = [False] * n
    dropStart = [None] * n
    dropFinish = [None] * n
    sea_drop_time = [None] * n
    l_lift_time = [None] * n
    l_drop_time = [None] * n
    ev_ls = {}
    coop_total = sum(1 for c in coop if c)
    coop_done = 0
    violated = False
    posWs = [posW]; posLs = [posL]
    makespan = 0
    t = 0
    steps = 0

    def lb_of(j):
        e = lsc[j]['earliest_finish_time']
        return e if strict else e - p

    def est_ls(j, q, t0):
        src = lsc[j]['source_slot']
        ds = t0 + abs(q - src) + p + (SINK - src)
        lb = lb_of(j)
        if ds < lb:
            ds = lb
        return ds + p

    def est_coop(i, q, t0):
        h = dropslot[i]; tg = targets[i]
        if dropFinish[i] is not None:
            df = dropFinish[i]
        elif dropStart[i] is not None:
            df = dropStart[i] + p
        else:
            df = max(t0, busyW) + abs(posW - h) + p
        st = max(t0 + abs(q - h), df + 1)
        end = st + p + abs(tg - h) + p
        return end, tg

    while True:
        # --- process completions at time t ---
        if wpend is not None and t >= busyW:
            tag = wpend; wpend = None
            if tag == 'lift':
                wphase = 'todrop'
            else:
                i = tag[1]
                dropFinish[i] = t
                if not coop[i]:
                    if t > makespan:
                        makespan = t
                wi += 1
                wphase = 'topick' if wi < n else 'done'
        if lpend is not None and t >= busyL:
            tag = lpend; lpend = None
            kind = tag[0]
            if kind == 'lsl':
                ltask = ('ls', tag[1], 'sink')
            elif kind == 'lsd':
                ltask = None
            elif kind == 'cl':
                ltask = ('coop', tag[1], 'target')
            elif kind == 'cd':
                coop_done += 1
                if t > makespan:
                    makespan = t
                ltask = None

        # --- termination check ---
        if (wi >= n) and (ltask is None) and (ls_ptr >= len(ls_list)) and (not coopQ) and (coop_done == coop_total):
            break
        if t > tmax:
            return None
        steps += 1
        if (steps & 4095) == 0 and time.time() > hard_deadline:
            return None

        # --- W crane: goal / action start ---
        lockedW = t < busyW
        gW = posW
        if not lockedW:
            if wphase == 'topick':
                gW = 0
                if posW == 0:
                    busyW = t + p; wpend = 'lift'; lockedW = True
                    if coop[wi] and not announced[wi]:
                        announced[wi] = True
                        coopQ.append(wi)
            elif wphase == 'todrop':
                gW = dropslot[wi]
                if posW == gW:
                    busyW = t + p; wpend = ('drop', wi); lockedW = True
                    dropStart[wi] = t
                    sea_drop_time[wi] = t
            else:
                gW = 0

        # --- L crane: dispatch ---
        lockedL = t < busyL
        if not lockedL and ltask is None:
            j0 = ls_list[ls_ptr] if ls_ptr < len(ls_list) else None
            chosen = None
            if coopQ:
                i = coopQ[0]
                if j0 is not None:
                    endt, q2 = est_coop(i, posL, t)
                    f = est_ls(j0, q2, endt)
                    if f > lsc[j0]['deadline'] - buffer_:
                        chosen = ('ls', j0)
                    else:
                        chosen = ('coop', i)
                else:
                    chosen = ('coop', i)
            elif j0 is not None:
                f = est_ls(j0, posL, t)
                slack = lsc[j0]['deadline'] - f
                src = lsc[j0]['source_slot']
                arr = t + abs(posL - src) + p + (SINK - src)
                waitt = max(0, lb_of(j0) - arr)
                more_coop = any(coop[i2] and not announced[i2] for i2 in range(wi, n))
                if (not more_coop) or slack <= buffer_ or (eager and waitt <= 2 * p):
                    chosen = ('ls', j0)
            if chosen is not None:
                if chosen[0] == 'ls':
                    ls_ptr += 1
                    ltask = ('ls', chosen[1], 'src')
                else:
                    coopQ.popleft()
                    ltask = ('coop', chosen[1], 'wait')

        # --- L crane: goal / action start ---
        gL = posL
        if not lockedL:
            if ltask is None:
                # park / yield to W
                if (not lockedW) and wphase in ('topick', 'todrop') and gW >= posL:
                    gL = min(SINK, gW + 1)
                else:
                    nxt = None
                    for i2 in range(wi, n):
                        if coop[i2]:
                            nxt = i2
                            break
                    gL = min(SINK, dropslot[nxt] + 1) if nxt is not None else SINK
                    if (not lockedW) and wphase in ('topick', 'todrop') and gW >= gL:
                        gL = min(SINK, gW + 1)
            else:
                kind, idx, ph = ltask
                if kind == 'ls':
                    if ph == 'src':
                        gL = lsc[idx]['source_slot']
                        if posL == gL:
                            busyL = t + p; lpend = ('lsl', idx); lockedL = True
                            ev_ls[idx] = {'lift': t}
                    else:
                        gL = SINK
                        if posL == SINK:
                            if t >= lb_of(idx):
                                busyL = t + p; lpend = ('lsd', idx); lockedL = True
                                ev_ls[idx]['drop'] = t
                                if t + p > lsc[idx]['deadline']:
                                    violated = True
                else:
                    h = dropslot[idx]
                    if ph == 'wait':
                        if dropFinish[idx] is not None:
                            gL = h
                            if posL == h and t >= dropFinish[idx]:
                                busyL = t + p; lpend = ('cl', idx); lockedL = True
                                l_lift_time[idx] = t
                        else:
                            gL = min(SINK, h + 1)
                    else:
                        gL = targets[idx]
                        if posL == gL:
                            busyL = t + p; lpend = ('cd', idx); lockedL = True
                            l_drop_time[idx] = t

        # --- movement resolution ---
        if lockedW:
            npW = posW
        else:
            npW = posW + (1 if gW > posW else (-1 if gW < posW else 0))
        if lockedL:
            npL = posL
        else:
            npL = posL + (1 if gL > posL else (-1 if gL < posL else 0))

        if lockedW and not lockedL:
            if npL <= posW:
                npL = posL
        elif lockedL and not lockedW:
            if npW >= posL:
                npW = posW
        elif (not lockedW) and (not lockedL):
            if npW >= npL:
                npW = npL - 1
                if npW < posW - 1:
                    npW = posW - 1
                if npW < 0:
                    npW = 0
                    if npL <= npW:
                        npL = posL
        if npW < 0:
            npW = 0
        if npL > SINK:
            npL = SINK
        if npW >= npL:
            # final safety: keep previous positions (should not happen)
            npW, npL = posW, posL

        posW, posL = npW, npL
        t += 1
        posWs.append(posW)
        posLs.append(posL)

    return {
        'violated': violated,
        'makespan': makespan,
        'posWs': posWs,
        'posLs': posLs,
        'sea_drop_time': sea_drop_time,
        'dropslot': dropslot,
        'coop': coop,
        'l_lift_time': l_lift_time,
        'l_drop_time': l_drop_time,
        'ev_ls': ev_ls,
        'T': t,
    }


def missing_required(lsc, M, K):
    add = set()
    rest = []
    for j, c in enumerate(lsc):
        if c['deadline'] <= M:
            if j not in K:
                add.add(j)
        else:
            rest.append(j)
    if rest:
        dmin = min(lsc[j]['deadline'] for j in rest)
        grp = [j for j in rest if lsc[j]['deadline'] == dmin]
        if not (set(grp) & K) and not (set(grp) & add):
            g = min(grp, key=lambda j: (lsc[j]['earliest_finish_time'], j))
            add.add(g)
    return add


def solve_config(inst, sea, lsc, hmap, eager, strict, buffer_, hard_deadline):
    K = set()
    for _ in range(len(lsc) + 2):
        if time.time() > hard_deadline:
            return None
        res = simulate(inst, sea, lsc, K, hmap, eager, strict, buffer_, hard_deadline)
        if res is None:
            return None
        add = missing_required(lsc, res['makespan'], K)
        if not add:
            res['Kset'] = set(K)
            return res
        K |= add
    return None


def build_sol(inst, sea, lsc, res):
    events = []
    for i, c in enumerate(sea):
        e = {
            'container_id': int(c['id']),
            'target_slot': int(c['target_slot']),
            'seaside_drop_time': int(res['sea_drop_time'][i]),
            'seaside_drop_slot': int(res['dropslot'][i]),
        }
        if res['coop'][i]:
            e['landside_lift_time'] = int(res['l_lift_time'][i])
            e['landside_lift_slot'] = int(res['dropslot'][i])
            e['landside_drop_time'] = int(res['l_drop_time'][i])
        events.append(e)
    lev = []
    for j in sorted(res['ev_ls'].keys()):
        d = res['ev_ls'][j]
        lev.append({
            'container_id': int(lsc[j]['id']),
            'source_slot': int(lsc[j]['source_slot']),
            'lift_time': int(d['lift']),
            'drop_time': int(d['drop']),
        })
    return {
        'objective_value': float(res['makespan']),
        'crane_w_positions': {str(i): int(v) for i, v in enumerate(res['posWs'])},
        'crane_l_positions': {str(i): int(v) for i, v in enumerate(res['posLs'])},
        'seaside_events': events,
        'landside_events': lev,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()
    t_start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)
    S = inst['S']; n = inst['n']; p = inst['p']
    sea = sorted(inst['seaside_containers'], key=lambda c: c['id'])
    lsc = list(inst['landside_containers'])

    hard_deadline = t_start + max(3, args.time_limit) - 1.0

    # candidate handover slots (ordered by expected quality)
    hcands = []
    for v in [max(1, S // 2), max(1, (3 * S) // 4), max(1, S // 4), max(1, S - 1), max(1, S), 1,
              max(1, (5 * S) // 8), max(1, (3 * S) // 8)]:
        if v not in hcands:
            hcands.append(v)
    # cooperation cutoffs: targets >= cutoff use cooperative handling (S+2 => only mandatory)
    cutoffs = []
    for v in [S + 2, max(2, (3 * (S + 1)) // 4), max(2, (S + 1) // 2), max(2, (7 * (S + 1)) // 8)]:
        if v not in cutoffs:
            cutoffs.append(v)

    combos = []
    seen = set()
    for co in cutoffs:
        for hp in hcands:
            hmap = []
            for c in sea:
                tg = c['target_slot']
                if tg == S + 1 or (tg >= co and tg >= 2):
                    hmap.append(max(1, min(hp, tg - 1, S)))
                else:
                    hmap.append(None)
            for eager in (True, False):
                sig = (tuple(hmap), eager)
                if sig in seen:
                    continue
                seen.add(sig)
                combos.append((hmap, eager))

    buffer_ = 2 * p + max(6, (S + 2) // 4)

    best = None; best_sol = None
    best_v = None; best_v_sol = None

    for strict in (True, False):
        for hmap, eager in combos:
            if time.time() > hard_deadline:
                break
            res = solve_config(inst, sea, lsc, hmap, eager, strict, buffer_, hard_deadline)
            if res is None:
                continue
            if not res['violated']:
                if best is None or res['makespan'] < best:
                    best = res['makespan']
                    best_sol = build_sol(inst, sea, lsc, res)
                    if logger:
                        logger.log_solution(float(best), best_sol)
            else:
                if best_v is None or res['makespan'] < best_v:
                    best_v = res['makespan']
                    best_v_sol = build_sol(inst, sea, lsc, res)
        if best is not None or time.time() > hard_deadline:
            break

    if best_sol is None:
        if best_v_sol is not None:
            best_sol = best_v_sol
            best = best_v
            if logger:
                logger.log_solution(float(best), best_sol)
        else:
            # emergency fallback (should not normally happen)
            best_sol = {
                'objective_value': 0.0,
                'crane_w_positions': {'0': int(inst['sigma_w'])},
                'crane_l_positions': {'0': int(inst['sigma_l'])},
                'seaside_events': [],
                'landside_events': [],
            }
            if logger:
                logger.log_solution(0.0, best_sol)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f)


if __name__ == '__main__':
    main()