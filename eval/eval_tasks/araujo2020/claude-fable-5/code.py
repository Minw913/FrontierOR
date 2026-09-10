import argparse
import json
import time
import random
import heapq
from collections import deque, defaultdict

import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(5, args.time_limit) - 1.0

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as f:
        data = json.load(f)

    projects = data['projects']
    res = data.get('resources', {}) or {}
    ren = res.get('renewable', []) or []
    non = res.get('nonrenewable', []) or []
    Rcap = [r['capacity'] for r in ren]
    Ncap = [r['capacity'] for r in non]
    K = len(Rcap)
    L = len(Ncap)

    jobs = data['jobs']
    n = len(jobs)
    jid_list = [jb['job_id'] for jb in jobs]
    idx = {jid: i for i, jid in enumerate(jid_list)}
    proj_index = {p['project_id']: i for i, p in enumerate(projects)}
    P = len(projects)
    release = [p['release_date'] for p in projects]
    sink_idx = [idx[p['artificial_sink_job_id']] for p in projects]
    pj = [proj_index[jb['project_id']] for jb in jobs]

    succ = [[idx[s] for s in jb.get('successors', [])] for jb in jobs]
    preds = [[] for _ in range(n)]
    for i in range(n):
        for s in succ[i]:
            preds[s].append(i)

    modes = []
    for jb in jobs:
        ms = []
        for m in jb['modes']:
            rc = list(m.get('renewable_consumption') or [])
            nc = list(m.get('nonrenewable_consumption') or [])
            rc = (rc + [0] * K)[:K]
            nc = (nc + [0] * L)[:L]
            ms.append((m['mode_id'], int(m['duration']), rc, nc))
        modes.append(ms)

    # ---------- CPM ----------
    mindur = [min(m[1] for m in modes[j]) for j in range(n)]
    indeg0 = [len(preds[j]) for j in range(n)]
    q = deque(j for j in range(n) if indeg0[j] == 0)
    topo = []
    dtmp = indeg0[:]
    while q:
        j = q.popleft()
        topo.append(j)
        for s in succ[j]:
            dtmp[s] -= 1
            if dtmp[s] == 0:
                q.append(s)

    ES = [0] * n
    for j in topo:
        e = release[pj[j]]
        for i in preds[j]:
            v = ES[i] + mindur[i]
            if v > e:
                e = v
        ES[j] = e
    CP = [ES[sink_idx[p]] - release[p] for p in range(P)]
    tail = [0] * n
    for j in reversed(topo):
        t = 0
        for s in succ[j]:
            v = mindur[s] + tail[s]
            if v > t:
                t = v
        tail[j] = t
    lst_lb = [release[pj[j]] + CP[pj[j]] - tail[j] - mindur[j] for j in range(n)]

    max_rel = max(release) if release else 0

    # ---------- mode selection / repair ----------
    def nr_totals(sel):
        tot = [0] * L
        for j in range(n):
            nc = modes[j][sel[j]][3]
            for l in range(L):
                tot[l] += nc[l]
        return tot

    def repair(sel, rng):
        if L == 0:
            return sel
        tot = nr_totals(sel)
        exc = sum(max(0, tot[l] - Ncap[l]) for l in range(L))
        guard = 0
        while exc > 0 and guard < 5000:
            guard += 1
            best = None
            for j in range(n):
                cur = modes[j][sel[j]]
                for mi in range(len(modes[j])):
                    if mi == sel[j]:
                        continue
                    alt = modes[j][mi]
                    newexc = 0
                    for l in range(L):
                        newexc += max(0, tot[l] - cur[3][l] + alt[3][l] - Ncap[l])
                    red = exc - newexc
                    if red > 0:
                        key = (red, -(alt[1] - cur[1]))
                        if best is None or key > best[0]:
                            best = (key, j, mi)
            if best is None:
                return None
            _, j, mi = best
            cur = modes[j][sel[j]]
            alt = modes[j][mi]
            for l in range(L):
                tot[l] += alt[3][l] - cur[3][l]
            sel[j] = mi
            exc = sum(max(0, tot[l] - Ncap[l]) for l in range(L))
        return sel if exc == 0 else None

    def base_mode_sel():
        sel = []
        for j in range(n):
            best_mi = 0
            best_key = None
            for mi, (mid, d, rc, nc) in enumerate(modes[j]):
                w = sum((nc[l] / max(1, Ncap[l])) for l in range(L))
                key = (d, w)
                if best_key is None or key < best_key:
                    best_key = key
                    best_mi = mi
            sel.append(best_mi)
        return sel

    rng = random.Random(0)
    sel0 = base_mode_sel()
    r = repair(sel0[:], rng)
    if r is None:
        # fallback: minimize nonrenewable load per job
        sel0 = []
        for j in range(n):
            best_mi = min(range(len(modes[j])),
                          key=lambda mi: (sum(modes[j][mi][3][l] / max(1, Ncap[l]) for l in range(L)), modes[j][mi][1]))
            sel0.append(best_mi)
        r2 = repair(sel0[:], rng)
        base_sel = r2 if r2 is not None else sel0
    else:
        base_sel = r

    # ---------- serial SGS ----------
    def sgs(sel, prio):
        H = max_rel + sum(modes[j][sel[j]][1] for j in range(n)) + 2
        usage = np.zeros((K, H), dtype=np.int64) if K > 0 else None
        indeg = [len(preds[j]) for j in range(n)]
        heap = [(prio[j], j) for j in range(n) if indeg[j] == 0]
        heapq.heapify(heap)
        start = [0] * n
        fin = [0] * n
        cnt = 0
        while heap:
            _, j = heapq.heappop(heap)
            mid, d, rc, nc = modes[j][sel[j]]
            t = release[pj[j]]
            for i in preds[j]:
                if fin[i] > t:
                    t = fin[i]
            if d > 0 and K > 0:
                ks = [k for k in range(K) if rc[k] > 0]
                while ks:
                    bad = -1
                    for k in ks:
                        seg = usage[k, t:t + d]
                        v = np.nonzero(seg + rc[k] > Rcap[k])[0]
                        if v.size:
                            b = t + int(v[-1]) + 1
                            if b > bad:
                                bad = b
                    if bad < 0:
                        break
                    t = bad
                for k in ks:
                    usage[k, t:t + d] += rc[k]
            start[j] = t
            fin[j] = t + d
            cnt += 1
            for s in succ[j]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    heapq.heappush(heap, (prio[s], s))
        if cnt < n:
            return None
        comp = [start[sink_idx[p]] for p in range(P)]
        delay = sum(comp[p] - release[p] - CP[p] for p in range(P))
        mk = max(comp) if comp else 0
        return delay, mk, start

    best = None  # (delay, mk, start, sel)

    def build_sol(delay, mk, start, sel):
        sched = [{'job_id': jid_list[j],
                  'mode_id': modes[j][sel[j]][0],
                  'start_time': int(start[j])} for j in range(n)]
        return {'objective_value': float(delay), 'schedule': sched, 'makespan': int(mk)}

    def write_best():
        if best is None:
            return
        sol = build_sol(best[0], best[1], best[2], best[3])
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)

    def register(res, sel):
        nonlocal best
        if res is None:
            return False
        delay, mk, start = res
        if best is None or (delay, mk) < (best[0], best[1]):
            best = (delay, mk, start[:], sel[:])
            if logger:
                logger.log_solution(float(delay), build_sol(delay, mk, start, sel))
            write_best()
            return True
        return False

    # deterministic first pass
    register(sgs(base_sel, lst_lb), base_sel)

    spread = max(1.0, (max(lst_lb) - min(lst_lb)) if n > 0 else 1.0)

    def sample_iteration():
        # modes
        if best is not None and rng.random() < 0.7:
            sel = best[3][:]
        else:
            sel = base_sel[:]
        if rng.random() < 0.6:
            kmut = rng.randint(1, max(1, n // 15))
            for _ in range(kmut):
                j = rng.randrange(n)
                sel[j] = rng.randrange(len(modes[j]))
            r2 = repair(sel, rng)
            if r2 is None:
                sel = best[3][:] if best is not None else base_sel[:]
            else:
                sel = r2
        # priorities
        rule = rng.random()
        if rule < 0.55:
            bpri = lst_lb
        elif rule < 0.8:
            bpri = ES
        else:
            bpri = [rng.uniform(0, spread) for _ in range(n)]
        alpha = rng.choice([0.1, 0.3, 0.6, 1.0])
        prio = [bpri[j] + rng.uniform(0, alpha * spread) for j in range(n)]
        register(sgs(sel, prio), sel)

    # ---------- heuristic phase ----------
    heur_budget = min(0.25 * max(5, args.time_limit), 25.0)
    heur_end = min(t0 + heur_budget, deadline)
    while time.time() < heur_end:
        sample_iteration()

    # ---------- decide on MIP ----------
    use_mip = False
    if best is not None and time.time() < deadline - 5:
        D = best[0]
        total_vars = 0
        total_coefs = 0
        for j in range(n):
            for mi, (mid, d, rc, nc) in enumerate(modes[j]):
                ls = release[pj[j]] + CP[pj[j]] + D - tail[j] - d
                es = ES[j]
                if ls >= es:
                    w = ls - es + 1
                    total_vars += w
                    total_coefs += w * d * sum(1 for k in range(K) if rc[k] > 0)
        if total_vars <= 250000 and total_coefs <= 6000000:
            use_mip = True

    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            D = best[0]
            Hm = max(release[p] + CP[p] for p in range(P)) + D
            m = gp.Model('mrcmpsp')
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time() - 1.0)

            x = {}
            for j in range(n):
                for mi, (mid, d, rc, nc) in enumerate(modes[j]):
                    ls = release[pj[j]] + CP[pj[j]] + D - tail[j] - d
                    es = ES[j]
                    for t in range(es, ls + 1):
                        x[(j, mi, t)] = m.addVar(vtype=GRB.BINARY)

            # assignment + expressions
            start_expr = [None] * n
            end_expr = [None] * n
            byjob = defaultdict(list)
            for key, v in x.items():
                byjob[key[0]].append((key[1], key[2], v))
            feasible_model = True
            for j in range(n):
                lst = byjob.get(j, [])
                if not lst:
                    feasible_model = False
                    break
                m.addConstr(gp.quicksum(v for _, _, v in lst) == 1)
                start_expr[j] = gp.quicksum(t * v for _, t, v in lst)
                end_expr[j] = gp.quicksum((t + modes[j][mi][1]) * v for mi, t, v in lst)

            if feasible_model:
                for i in range(n):
                    for s in succ[i]:
                        m.addConstr(end_expr[i] <= start_expr[s])

                # renewable
                racc = defaultdict(list)
                for (j, mi, t), v in x.items():
                    d = modes[j][mi][1]
                    if d == 0:
                        continue
                    rc = modes[j][mi][2]
                    for k in range(K):
                        c = rc[k]
                        if c > 0:
                            for tau in range(t, t + d):
                                racc[(k, tau)].append((c, v))
                for (k, tau), lstv in racc.items():
                    m.addConstr(gp.quicksum(c * v for c, v in lstv) <= Rcap[k])

                # nonrenewable
                for l in range(L):
                    terms = []
                    for (j, mi, t), v in x.items():
                        c = modes[j][mi][3][l]
                        if c > 0:
                            terms.append((c, v))
                    if terms:
                        m.addConstr(gp.quicksum(c * v for c, v in terms) <= Ncap[l])

                MS = m.addVar(lb=0.0, ub=Hm + 1)
                for p in range(P):
                    m.addConstr(MS >= start_expr[sink_idx[p]])
                eps = 0.5 / (Hm + 2.0)
                const = sum(release[p] + CP[p] for p in range(P))
                m.setObjective(gp.quicksum(start_expr[sink_idx[p]] for p in range(P)) - const + eps * MS,
                               GRB.MINIMIZE)

                # warm start
                for v in x.values():
                    v.Start = 0.0
                _, _, bst, bsel = best
                for j in range(n):
                    key = (j, bsel[j], bst[j])
                    if key in x:
                        x[key].Start = 1.0

                keys = list(x.keys())
                varlist = [x[k] for k in keys]

                def cb(model, where):
                    if where == GRB.Callback.MIPSOL:
                        try:
                            vals = model.cbGetSolution(varlist)
                        except Exception:
                            return
                        sel2 = [0] * n
                        st2 = [0] * n
                        for (jj, mi, t), val in zip(keys, vals):
                            if val > 0.5:
                                sel2[jj] = mi
                                st2[jj] = t
                        comp = [st2[sink_idx[p]] for p in range(P)]
                        delay = sum(comp[p] - release[p] - CP[p] for p in range(P))
                        mk = max(comp) if comp else 0
                        register((delay, mk, st2), sel2)

                m.optimize(cb)

                if m.SolCount > 0:
                    sel2 = [0] * n
                    st2 = [0] * n
                    for (jj, mi, t), v in x.items():
                        if v.X > 0.5:
                            sel2[jj] = mi
                            st2[jj] = t
                    comp = [st2[sink_idx[p]] for p in range(P)]
                    delay = sum(comp[p] - release[p] - CP[p] for p in range(P))
                    mk = max(comp) if comp else 0
                    register((delay, mk, st2), sel2)
        except Exception:
            pass

    # ---------- use remaining time with heuristic sampling ----------
    while time.time() < deadline - 0.2:
        sample_iteration()

    if best is None:
        # last resort: emit serial schedule ignoring nothing scheduled case
        prio = list(range(n))
        res = sgs(base_sel, prio)
        if res:
            register(res, base_sel)

    write_best()
    if best is None:
        with open(args.solution_path, 'w') as f:
            json.dump({'objective_value': float('inf'), 'schedule': [], 'makespan': 0}, f)


if __name__ == '__main__':
    main()