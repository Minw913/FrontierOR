import argparse
import json
import math
import time
import random
import itertools
import sys
from collections import defaultdict


def ceil_div(a, b):
    return -(-a // b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()
    t_start = time.time()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    V = inst['num_vessels']
    K = inst['num_berths']
    Q = inst['total_qc']
    S = inst['time_steps_per_shift']
    T = inst['total_time_steps']
    c1 = inst['c1']
    c2 = inst['c2']
    vessels = inst['vessels']
    berths = inst['berths']

    # ---------- time indexing base detection ----------
    mins = min([v['a'] for v in vessels] + [bk['a_k'] for bk in berths])
    offset = 0 if mins <= 0 else 1
    horizon_end = offset + T - 1

    # ---------- deterministic profile generation (published rule) ----------
    seed = inst.get('seed', inst.get('random_seed', inst.get('profile_seed', 0)))

    def gen_sequences(cfg, n):
        seqs = []
        for m in range(cfg['min_handling_shifts'], cfg['max_handling_shifts'] + 1):
            for combo in itertools.product(range(cfg['min_qc'], cfg['max_qc'] + 1), repeat=m):
                seqs.append(list(combo))
        rng = random.Random(seed)
        if n >= len(seqs) or n <= 0:
            return seqs
        return rng.sample(seqs, n)

    big_cfg = inst['qc_profiles']['big']
    small_cfg = inst['qc_profiles']['small']
    sampled = {
        'big': gen_sequences(big_cfg, inst.get('num_big_profiles', 0)),
        'small': gen_sequences(small_cfg, inst.get('num_small_profiles', 0)),
    }
    cfg_of = {'big': big_cfg, 'small': small_cfg}

    steps_per_hour = T / max(1, inst['time_horizon_hours'])

    # ---------- per vessel: class, required duration, per-position profile ----------
    def classify(m_v):
        if small_cfg['min_handling_shifts'] <= m_v <= small_cfg['max_handling_shifts']:
            return 'small'
        if big_cfg['min_handling_shifts'] <= m_v <= big_cfg['max_handling_shifts']:
            return 'big'
        return 'big' if m_v > small_cfg['max_handling_shifts'] else 'small'

    def pick_seq(cls, m_star):
        seqs = sampled[cls]
        if not seqs:
            cfg = cfg_of[cls]
            return 0, [cfg['min_qc']] * m_star
        best = None
        for idx, seq in enumerate(seqs):
            key = (abs(len(seq) - m_star),
                   0 if len(seq) >= m_star else 1,
                   sum(seq), tuple(seq))
            if best is None or key < best[0]:
                best = (key, idx, seq)
        return best[1], best[2]

    def expand(seq, pos):
        steps = [seq[0]] * (S - pos)
        for q in seq[1:]:
            steps += [q] * S
        return steps

    posinfo = []  # posinfo[v][pos] = (prof_idx, D, qc_steps)
    for v in range(V):
        dv = max(1, int(round(vessels[v]['duration_hours'] * steps_per_hour)))
        m_v = ceil_div(dv, S)
        cls = classify(m_v)
        cfg = cfg_of[cls]
        info = []
        for pos in range(S):
            rem = S - pos
            m_req = 1 if dv <= rem else 1 + ceil_div(dv - rem, S)
            m_star = min(cfg['max_handling_shifts'], max(cfg['min_handling_shifts'], m_req))
            prof_idx, seq = pick_seq(cls, m_star)
            steps = expand(seq, pos)
            info.append((prof_idx, len(steps), steps))
        posinfo.append(info)

    # ---------- candidate enumeration ----------
    cand = []
    for v in range(V):
        vv = vessels[v]
        kbar, tbar = vv['k_bar'], vv['t_bar']
        lo = max(vv['a'], offset)
        hi = min(vv['b'], horizon_end)
        lst = []
        for t in range(lo, hi + 1):
            pos = (t - offset) % S
            prof_idx, D, _ = posinfo[v][pos]
            end = t + D - 1
            if end > horizon_end:
                continue
            for k in range(K):
                bk = berths[k]
                if t >= bk['a_k'] and end <= bk['b_k']:
                    cost = c1 * abs(k - kbar) + c2 * abs(t - tbar)
                    lst.append((cost, k, t, end, prof_idx, pos))
        if not lst:  # relaxation level 2: end may exceed berth window
            for t in range(lo, hi + 1):
                pos = (t - offset) % S
                prof_idx, D, _ = posinfo[v][pos]
                end = t + D - 1
                for k in range(K):
                    bk = berths[k]
                    if bk['a_k'] <= t <= bk['b_k'] and end <= horizon_end:
                        cost = c1 * abs(k - kbar) + c2 * abs(t - tbar)
                        lst.append((cost, k, t, end, prof_idx, pos))
        if not lst:  # relaxation level 3: any berth, any start in vessel window
            for t in range(lo, hi + 1):
                pos = (t - offset) % S
                prof_idx, D, _ = posinfo[v][pos]
                end = t + D - 1
                for k in range(K):
                    cost = c1 * abs(k - kbar) + c2 * abs(t - tbar)
                    lst.append((cost, k, t, end, prof_idx, pos))
        if not lst:
            pos = (lo - offset) % S
            prof_idx, D, _ = posinfo[v][pos]
            lst.append((0, min(kbar, K - 1), lo, lo + D - 1, prof_idx, pos))
        lst.sort()
        cand.append(lst)

    # cap total candidate count to keep the MIP tractable
    total = sum(len(c) for c in cand)
    if total > 400000:
        cap = max(200, 400000 // max(1, V))
        cand = [c[:cap] for c in cand]

    # ---------- greedy construction ----------
    def greedy():
        berth_occ = [[False] * T for _ in range(K)]
        qc_used = [0] * T
        assign = {}
        order = sorted(range(V), key=lambda v: (len(cand[v]), vessels[v]['b'] - vessels[v]['a']))
        for v in order:
            placed = None
            for c in cand[v]:
                cost, k, t, end, prof_idx, pos = c
                qsteps = posinfo[v][pos][2]
                ok = True
                for r, tau in enumerate(range(t, min(end, horizon_end) + 1)):
                    i = tau - offset
                    if berth_occ[k][i] or qc_used[i] + qsteps[r] > Q:
                        ok = False
                        break
                if ok:
                    placed = c
                    break
            if placed is None:  # ignore QC capacity
                for c in cand[v]:
                    cost, k, t, end, prof_idx, pos = c
                    ok = True
                    for tau in range(t, min(end, horizon_end) + 1):
                        if berth_occ[k][tau - offset]:
                            ok = False
                            break
                    if ok:
                        placed = c
                        break
            if placed is None:
                placed = cand[v][0]
            cost, k, t, end, prof_idx, pos = placed
            qsteps = posinfo[v][pos][2]
            for r, tau in enumerate(range(t, min(end, horizon_end) + 1)):
                i = tau - offset
                berth_occ[k][i] = True
                qc_used[i] += qsteps[r]
            assign[v] = placed
        return assign

    def build_solution(assign):
        obj = 0.0
        vs = []
        for v in range(V):
            cost, k, t, end, prof_idx, pos = assign[v]
            obj += cost
            vs.append({'id': v, 'berth': k, 'start_time': t, 'end_time': end,
                       'profile': prof_idx})
        return obj, {'objective_value': obj, 'vessels': vs}

    g_assign = greedy()
    g_obj, g_sol = build_solution(g_assign)
    if logger:
        logger.log_solution(g_obj, g_sol)

    best_obj, best_sol = g_obj, g_sol

    # ---------- MIP with Gurobi ----------
    try:
        import gurobipy as gp
        from gurobipy import GRB

        remaining = args.time_limit - (time.time() - t_start) - 3
        if remaining > 2:
            m = gp.Model('bacap')
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(2, remaining)

            xs = []
            meta = []
            vessel_vars = [[] for _ in range(V)]
            occ = defaultdict(list)
            qcc = defaultdict(list)
            startmap = {}
            for v in range(V):
                for c in cand[v]:
                    cost, k, t, end, prof_idx, pos = c
                    x = m.addVar(vtype=GRB.BINARY, obj=float(cost))
                    idx = len(xs)
                    xs.append(x)
                    meta.append((v, k, t, end, prof_idx, pos, cost))
                    vessel_vars[v].append(x)
                    startmap[(v, k, t)] = x
                    qsteps = posinfo[v][pos][2]
                    for r, tau in enumerate(range(t, min(end, horizon_end) + 1)):
                        occ[(k, tau)].append(x)
                        qcc[tau].append((x, qsteps[r]))
            for v in range(V):
                m.addConstr(gp.quicksum(vessel_vars[v]) == 1)
            for key, lst in occ.items():
                if len(lst) > 1:
                    m.addConstr(gp.quicksum(lst) <= 1)
            for tau, lst in qcc.items():
                if sum(q for _, q in lst) > Q:
                    m.addConstr(gp.quicksum(x * q for x, q in lst) <= Q)
            m.ModelSense = GRB.MINIMIZE

            # warm start from greedy
            for x in xs:
                x.Start = 0.0
            for v in range(V):
                cost, k, t, end, prof_idx, pos = g_assign[v]
                if (v, k, t) in startmap:
                    startmap[(v, k, t)].Start = 1.0

            m._vars = xs
            m._meta = meta
            m._best = best_obj + 1e-9
            m._logger = logger

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < model._best - 1e-6:
                        vals = model.cbGetSolution(model._vars)
                        sel = {}
                        for i, val in enumerate(vals):
                            if val > 0.5:
                                v, k, t, end, prof_idx, pos, cost = model._meta[i]
                                sel[v] = (cost, k, t, end, prof_idx, pos)
                        if len(sel) == V:
                            o, sol = build_solution(sel)
                            model._best = obj
                            if model._logger:
                                model._logger.log_solution(o, sol)

            m.optimize(cb)

            if m.SolCount > 0:
                sel = {}
                for i, x in enumerate(xs):
                    if x.X > 0.5:
                        v, k, t, end, prof_idx, pos, cost = meta[i]
                        sel[v] = (cost, k, t, end, prof_idx, pos)
                if len(sel) == V:
                    o, sol = build_solution(sel)
                    if o < best_obj - 1e-9:
                        best_obj, best_sol = o, sol
                        if logger:
                            logger.log_solution(o, sol)
    except Exception:
        pass

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f, indent=2)


if __name__ == '__main__':
    main()