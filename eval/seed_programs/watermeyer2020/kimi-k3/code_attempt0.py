import argparse
import json
import math
import random
import time


def main():
    parser = argparse.ArgumentParser(description="RCPSP with partially renewable resources")
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    # ----------------------------- load instance -----------------------------
    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    n = int(inst.get('n', 0))
    N = int(inst.get('num_activities', n + 2))
    if N < n + 2:
        N = n + 2
    end = N - 1
    d_bar = int(inst['d_bar'])
    D = d_bar

    acts = inst.get('activities', {}) or {}
    p = [0] * N
    for aid, ad in acts.items():
        try:
            i = int(aid)
        except Exception:
            continue
        if 0 <= i < N:
            p[i] = int(ad.get('processing_time', 0) or 0)

    res = inst.get('resources', {}) or {}

    def _rid_key(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    res_ids = sorted(res.keys(), key=_rid_key)
    R = int(inst.get('num_resources', len(res_ids)) or 0)
    if R > len(res_ids):
        R = len(res_ids)
    res_ids = res_ids[:R]
    ridx = {rid: k for k, rid in enumerate(res_ids)}
    cap = [int(res[rid].get('capacity', 0)) for rid in res_ids]

    # prefix sums: pref[k][t] = number of designated periods of resource k <= t
    pref = []
    for rid in res_ids:
        mark = [0] * (D + 2)
        for per in res[rid].get('periods', []) or []:
            per = int(per)
            if 1 <= per <= D:
                mark[per] = 1
        ps = [0] * (D + 2)
        c = 0
        for tt in range(D + 2):
            c += mark[tt]
            ps[tt] = c
        pref.append(ps)

    dem = [[0] * R for _ in range(N)]
    for aid, ad in acts.items():
        try:
            i = int(aid)
        except Exception:
            continue
        if 0 <= i < N:
            for rid, v in (ad.get('resource_demands', {}) or {}).items():
                k = ridx.get(rid)
                if k is not None:
                    dem[i][k] = int(v or 0)

    cons = []
    have_e0 = False
    for c in inst.get('temporal_constraints', []) or []:
        i = int(c['i']); j = int(c['j']); w = int(c['delta'])
        if 0 <= i < N and 0 <= j < N:
            cons.append((i, j, w))
            if i == end and j == 0:
                have_e0 = True
    if not have_e0:
        cons.append((end, 0, -d_bar))

    # add S_i >= S_0 (=0) edges so every node is reachable from the source
    edges = cons + [(0, i, 0) for i in range(N)]
    NEG = float('-inf')

    def longest_from(src, elist):
        dist = [NEG] * N
        dist[src] = 0
        for _ in range(N - 1):
            upd = False
            for (u, v, w) in elist:
                du = dist[u]
                if du != NEG:
                    val = du + w
                    if val > dist[v]:
                        dist[v] = val
                        upd = True
            if not upd:
                break
        return dist

    dist0 = longest_from(0, edges)
    redges = [(v, u, w) for (u, v, w) in edges]
    dist_to0 = longest_from(0, redges)  # longest path i -> 0 in original graph

    ES = [0] * N
    LS = [0] * N
    for i in range(N):
        e = dist0[i]
        ES[i] = 0 if e == NEG else int(e)
        dv = dist_to0[i]
        l = d_bar if dv == NEG else int(-dv)
        if l > d_bar:
            l = d_bar
        if l < ES[i]:
            l = ES[i]
        LS[i] = l

    def ov(k, t, dur):
        # number of designated periods of resource k in (t, t+dur]
        hi = t + dur
        if hi > D:
            hi = D
        lo = t if t <= D else D
        if hi < lo:
            hi = lo
        return pref[k][hi] - pref[k][lo]

    cons_in = [[] for _ in range(N)]
    cons_out = [[] for _ in range(N)]
    for (i, j, w) in cons:
        cons_out[i].append((j, w))
        cons_in[j].append((i, w))

    def check(full):
        if full[0] != 0 or full[end] > d_bar:
            return False
        for i in range(N):
            if full[i] < 0:
                return False
        for (i, j, w) in cons:
            if full[j] - full[i] < w:
                return False
        usage = [0] * R
        for i in range(1, end):
            t = full[i]
            dur = p[i]
            if dur:
                di = dem[i]
                for k in range(R):
                    r = di[k]
                    if r:
                        usage[k] += r * ov(k, t, dur)
                        if usage[k] > cap[k]:
                            return False
        return True

    def make_sol(full):
        return {
            "objective_value": int(full[end]),
            "start_times": {str(i): int(full[i]) for i in range(N)},
        }

    def write_sol(full):
        with open(args.solution_path, 'w') as f:
            json.dump(make_sol(full), f)

    reals = list(range(1, end))

    # ---------------- fast path: earliest schedule is a valid lower bound ----------------
    es_full = [ES[i] for i in range(N)]
    best_full = None
    best_obj = math.inf
    if check(es_full):
        # ES[end] equals the longest-path lower bound -> optimal
        best_full = es_full
        best_obj = es_full[end]
        if logger:
            logger.log_solution(int(best_obj), make_sol(best_full))
        write_sol(best_full)
        return

    # ---------------- heuristic: serial schedule generation scheme ----------------
    def sgs(order):
        S = [None] * N
        S[0] = 0
        used = [0] * R
        for i in order:
            low = ES[i]
            high = LS[i]
            for (a, w) in cons_in[i]:
                sa = S[a]
                if sa is not None:
                    v = sa + w
                    if v > low:
                        low = v
            for (b, w) in cons_out[i]:
                sb = S[b]
                if sb is not None:
                    v = sb - w
                    if v < high:
                        high = v
            if low > high:
                return None
            dur = p[i]
            di = dem[i]
            span = high - low + 1
            if span * max(1, R) <= 250000:
                cand = range(low, high + 1)
            else:
                step = max(1, span // 2000)
                cand = list(range(low, high + 1, step))
                if cand[-1] != high:
                    cand.append(high)
            best_t = low
            best_cost = None
            for t in cand:
                cost = 0
                for k in range(R):
                    r = di[k]
                    if r:
                        over = used[k] + r * ov(k, t, dur) - cap[k]
                        if over > 0:
                            cost += over
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_t = t
                    if cost == 0:
                        break
            S[i] = best_t
            if dur:
                for k in range(R):
                    r = di[k]
                    if r:
                        used[k] += r * ov(k, best_t, dur)
        send = ES[end]
        for (a, w) in cons_in[end]:
            sa = S[a]
            if sa is not None:
                v = sa + w
                if v > send:
                    send = v
        if send > d_bar:
            return None
        S[end] = send
        for (b, w) in cons_out[end]:
            sb = S[b]
            if sb is not None and sb - send < w:
                return None
        full = [S[i] if S[i] is not None else 0 for i in range(N)]
        if not check(full):
            return None
        return (send, full)

    h_budget = min(4.0, max(0.3, 0.05 * time_limit))
    h_start = time.time()
    det_orders = [
        sorted(reals, key=lambda i: (LS[i], ES[i])),
        sorted(reals, key=lambda i: (ES[i], LS[i])),
        sorted(reals, key=lambda i: (LS[i] - ES[i], LS[i])),
        sorted(reals, key=lambda i: (-p[i], LS[i])),
    ]
    for order in det_orders:
        res_ = sgs(order)
        if res_ is not None and res_[0] < best_obj:
            best_obj, best_full = res_[0], res_[1]
    rng = random.Random(0)
    while time.time() - h_start < h_budget:
        order = reals[:]
        order.sort(key=lambda i: LS[i] + rng.uniform(-4.0, 4.0))
        res_ = sgs(order)
        if res_ is not None and res_[0] < best_obj:
            best_obj, best_full = res_[0], res_[1]

    if best_full is not None:
        if logger:
            logger.log_solution(int(best_obj), make_sol(best_full))
        if best_obj <= ES[end]:  # matches lower bound -> optimal
            write_sol(best_full)
            return

    # ---------------- time-indexed MIP ----------------
    final_full = best_full if best_full is not None else es_full
    elapsed = time.time() - t_start
    remaining = time_limit - elapsed - 0.75
    if remaining >= 0.5:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            lo = ES[:]
            hi = LS[:]
            total = sum(hi[i] - lo[i] + 1 for i in reals)
            MAXV = 600000
            if total > MAXV and reals:
                center = best_full if best_full is not None else es_full
                width = max(8, MAXV // (2 * len(reals)))
                for i in reals:
                    c = center[i]
                    a = max(ES[i], c - width)
                    b = min(LS[i], c + width)
                    lo[i] = a
                    hi[i] = max(a, b)

            m = gp.Model("rcpsp_partially_renewable")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.LogToConsole = 0
            m.Params.TimeLimit = max(1.0, remaining)
            if best_full is None:
                m.Params.MIPFocus = 1

            keys = [(i, t) for i in reals for t in range(lo[i], hi[i] + 1)]
            x = m.addVars(keys, vtype=GRB.BINARY, name="x")
            ub_end = min(LS[end], d_bar)
            if ub_end < ES[end]:
                ub_end = ES[end]
            s_end = m.addVar(lb=float(ES[end]), ub=float(ub_end), name="s_end")
            m.update()

            sexpr = {}
            for i in reals:
                vs = []
                le = gp.LinExpr()
                for t in range(lo[i], hi[i] + 1):
                    v = x[(i, t)]
                    vs.append(v)
                    le.addTerm(float(t), v)
                sexpr[i] = le
                m.addConstr(gp.quicksum(vs) == 1.0, name="assign_%d" % i)

            def sexpr_of(i):
                if i == 0:
                    return None
                if i == end:
                    return s_end
                return sexpr[i]

            for (i, j, w) in cons:
                if i == j:
                    continue
                ei = sexpr_of(i)
                ej = sexpr_of(j)
                if i == 0:
                    m.addConstr(ej >= float(w))
                elif j == 0:
                    m.addConstr(ei <= float(-w))
                else:
                    m.addConstr(ej - ei >= float(w))

            for k in range(R):
                le = gp.LinExpr()
                ck = cap[k]
                for i in reals:
                    r = dem[i][k]
                    if r:
                        dur = p[i]
                        if dur == 0:
                            continue
                        for t in range(lo[i], hi[i] + 1):
                            o = ov(k, t, dur)
                            if o:
                                le.addTerm(float(r * o), x[(i, t)])
                if le.size() > 0:
                    m.addConstr(le <= float(ck), name="res_%d" % k)

            m.setObjective(s_end, GRB.MINIMIZE)
            m.update()

            if best_full is not None:
                for (i, t) in keys:
                    x[(i, t)].Start = 1.0 if t == best_full[i] else 0.0
                s_end.Start = float(best_full[end])

            var_list = [x[kk] for kk in keys]
            state = {'best': best_obj}

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    except Exception:
                        return
                    if obj < state['best'] - 1e-6:
                        state['best'] = obj
                        if logger:
                            try:
                                vals = model.cbGetSolution(var_list)
                                full = [0] * N
                                for idx, (i, t) in enumerate(keys):
                                    if vals[idx] > 0.5:
                                        full[i] = t
                                se = model.cbGetSolution(s_end)
                                full[end] = int(round(se))
                                full[0] = 0
                                logger.log_solution(int(round(obj)), make_sol(full))
                            except Exception:
                                try:
                                    logger.log(int(round(obj)))
                                except Exception:
                                    pass

            m.optimize(cb)

            if m.SolCount > 0:
                vals = m.getAttr('X', var_list)
                full = [0] * N
                for idx, (i, t) in enumerate(keys):
                    if vals[idx] > 0.5:
                        full[i] = t
                full[end] = int(round(s_end.X))
                full[0] = 0
                if check(full):
                    if best_full is None or full[end] < best_full[end]:
                        final_full = full
                    else:
                        final_full = best_full
                else:
                    final_full = best_full if best_full is not None else full
        except Exception:
            pass

    write_sol(final_full)


if __name__ == '__main__':
    main()