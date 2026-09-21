import argparse
import json
import time
import random
import bisect

INF = float('inf')


# ---------------------------------------------------------------------------
# Weighted completion time variant
# ---------------------------------------------------------------------------

def wct_solution(assign, n, m, w, p):
    """Build full solution dict from assignment, sequencing by WSPT per machine."""
    schedule = {}
    comp = {}
    total = 0
    for i in range(m):
        jobs = [j for j in range(n) if assign[j] == i]
        jobs.sort(key=lambda j: (p[j][i] / w[j] if w[j] > 0 else INF, j))
        t = 0
        for j in jobs:
            t += p[j][i]
            comp[str(j)] = float(t)
            total += w[j] * t
        schedule[str(i)] = jobs
    sol = {
        'objective_value': float(total),
        'schedule': schedule,
        'completion_times': comp,
        'tardy_jobs': [],
        'on_time_jobs': list(range(n)),
    }
    return total, sol


def solve_wct(inst, deadline, logger):
    n = inst['num_jobs']
    m = inst['num_machines']
    w = inst['jobs']['weights']
    p = inst['jobs']['processing_times']

    if n == 0:
        sol = {'objective_value': 0.0,
               'schedule': {str(i): [] for i in range(m)},
               'completion_times': {}, 'tardy_jobs': [], 'on_time_jobs': []}
        if logger:
            logger.log_solution(0.0, sol)
        return 0.0, sol

    def ratio(j, i):
        return p[j][i] / w[j] if w[j] > 0 else INF

    def build(jobs, i):
        seq = sorted(jobs, key=lambda j: (ratio(j, i), j))
        keys = [ratio(j, i) for j in seq]
        pp = [0]
        for j in seq:
            pp.append(pp[-1] + p[j][i])
        L = len(seq)
        sw = [0] * (L + 1)
        for t in range(L - 1, -1, -1):
            sw[t] = sw[t + 1] + w[seq[t]]
        cost = 0
        for t in range(L):
            cost += w[seq[t]] * pp[t + 1]
        return [seq, keys, pp, sw, cost]

    def ins_delta(st, i, j):
        pos = bisect.bisect_right(st[1], ratio(j, i))
        return w[j] * (st[2][pos] + p[j][i]) + p[j][i] * st[3][pos]

    def rem_delta(st, i, j):
        pos = st[0].index(j)
        return -(w[j] * st[2][pos + 1] + p[j][i] * st[3][pos + 1])

    # ---- greedy construction (global WSPT order, cheapest insertion) ----
    order = sorted(range(n), key=lambda j: min(ratio(j, i) for i in range(m)))
    states = [build([], i) for i in range(m)]
    assign = [-1] * n
    total = 0
    for j in order:
        bd = None
        bi = 0
        for i in range(m):
            dd = ins_delta(states[i], i, j)
            if bd is None or dd < bd:
                bd = dd
                bi = i
        assign[j] = bi
        states[bi] = build(states[bi][0] + [j], bi)
        total += bd

    best_total, best_sol = wct_solution(assign, n, m, w, p)
    best_assign = list(assign)
    if logger:
        logger.log_solution(best_total, best_sol)

    rng = random.Random(0)

    def local_search(assign, states, total):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # move neighborhood
            for j in range(n):
                if time.time() > deadline:
                    break
                i = assign[j]
                rd = rem_delta(states[i], i, j)
                bd = -1e-9
                bi = -1
                for i2 in range(m):
                    if i2 == i:
                        continue
                    dd = rd + ins_delta(states[i2], i2, j)
                    if dd < bd:
                        bd = dd
                        bi = i2
                if bi >= 0:
                    s = list(states[i][0])
                    s.remove(j)
                    states[i] = build(s, i)
                    states[bi] = build(states[bi][0] + [j], bi)
                    assign[j] = bi
                    total += bd
                    improved = True
            # sampled swap neighborhood
            if m > 1 and 1 < n <= 1500:
                for j in range(n):
                    if time.time() > deadline:
                        break
                    i = assign[j]
                    cands = rng.sample(range(n), min(n, 8))
                    bd = -1e-9
                    bk = -1
                    for k in cands:
                        if k == j:
                            continue
                        i2 = assign[k]
                        if i2 == i:
                            continue
                        c1 = build([x for x in states[i][0] if x != j] + [k], i)[4]
                        c2 = build([x for x in states[i2][0] if x != k] + [j], i2)[4]
                        dd = c1 + c2 - states[i][4] - states[i2][4]
                        if dd < bd:
                            bd = dd
                            bk = k
                    if bk >= 0:
                        k = bk
                        i2 = assign[k]
                        states[i] = build([x for x in states[i][0] if x != j] + [k], i)
                        states[i2] = build([x for x in states[i2][0] if x != k] + [j], i2)
                        assign[j], assign[k] = i2, i
                        total += bd
                        improved = True
        return total

    total = local_search(assign, states, total)
    if total < best_total - 1e-9:
        best_total, best_sol = wct_solution(assign, n, m, w, p)
        best_assign = list(assign)
        if logger:
            logger.log_solution(best_total, best_sol)

    # ---- exact MILP (linearized pairwise Smith-rule formulation) ----
    qterms = m * n * (n - 1) // 2
    use_mip = (m > 1) and qterms <= 150000 and (deadline - time.time()) > 10
    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            mod = gp.Model('wct')
            mod.Params.OutputFlag = 0
            mod.Params.Seed = 0
            mod.Params.MIPGap = 1e-4
            mod.Params.NumericFocus = 0
            mod.Params.Threads = 1
            mod.Params.TimeLimit = max(1.0, deadline - time.time() - 1.0)
            x = mod.addVars(n, m, vtype=GRB.BINARY)
            mod.addConstrs((x.sum(j, '*') == 1 for j in range(n)))
            pairs = []
            coeffs = []
            for i in range(m):
                for j in range(n):
                    wj = w[j]
                    pji = p[j][i]
                    for k in range(j + 1, n):
                        c = wj * p[k][i]
                        c2 = w[k] * pji
                        if c2 < c:
                            c = c2
                        if c > 0:
                            pairs.append((i, j, k))
                            coeffs.append(c)
            y = mod.addVars(len(pairs), lb=0.0, ub=1.0)
            mod.addConstrs((y[t] - x[pairs[t][1], pairs[t][0]] - x[pairs[t][2], pairs[t][0]] >= -1
                            for t in range(len(pairs))))
            lin = gp.LinExpr()
            lin.addTerms([w[j] * p[j][i] for j in range(n) for i in range(m)],
                         [x[j, i] for j in range(n) for i in range(m)])
            if pairs:
                lin.addTerms(coeffs, [y[t] for t in range(len(pairs))])
            mod.setObjective(lin, GRB.MINIMIZE)
            for j in range(n):
                for i in range(m):
                    x[j, i].Start = 1.0 if best_assign[j] == i else 0.0
            xlist = [x[j, i] for j in range(n) for i in range(m)]
            holder = [best_total, list(best_assign)]

            def cb(mm, where):
                if where == GRB.Callback.MIPSOL:
                    vals = mm.cbGetSolution(xlist)
                    a = [0] * n
                    for j in range(n):
                        bi = 0
                        bv = -1.0
                        base = j * m
                        for i in range(m):
                            v = vals[base + i]
                            if v > bv:
                                bv = v
                                bi = i
                        a[j] = bi
                    tot, sol = wct_solution(a, n, m, w, p)
                    if tot < holder[0] - 1e-9:
                        holder[0] = tot
                        holder[1] = list(a)
                        if logger:
                            logger.log_solution(tot, sol)

            mod.optimize(cb)
            if holder[0] < best_total - 1e-9:
                best_assign = holder[1]
                best_total, best_sol = wct_solution(best_assign, n, m, w, p)
        except Exception:
            pass
    else:
        # ---- ILS: perturb + reinsert + local search until deadline ----
        while m > 1 and time.time() < deadline - 0.3:
            a = list(best_assign)
            k = min(n, max(2, n // 8))
            removed = rng.sample(range(n), k)
            remset = set(removed)
            states2 = [build([j for j in range(n) if a[j] == i and j not in remset], i)
                       for i in range(m)]
            removed.sort(key=lambda j: min(ratio(j, i) for i in range(m)))
            for j in removed:
                bd = None
                bi = 0
                for i in range(m):
                    dd = ins_delta(states2[i], i, j)
                    if bd is None or dd < bd:
                        bd = dd
                        bi = i
                a[j] = bi
                states2[bi] = build(states2[bi][0] + [j], bi)
            tot = sum(st[4] for st in states2)
            tot = local_search(a, states2, tot)
            if tot < best_total - 1e-9:
                best_total, best_sol = wct_solution(a, n, m, w, p)
                best_assign = list(a)
                if logger:
                    logger.log_solution(best_total, best_sol)

    return best_total, best_sol


# ---------------------------------------------------------------------------
# Weighted tardy jobs variant
# ---------------------------------------------------------------------------

def tardy_solution(sets, n, m, w, p, d):
    """Build solution dict from per-machine on-time sets (EDD sequencing).
    Any job that would end up tardy in the sequence is dropped to tardy list."""
    schedule = {}
    comp = {}
    on = []
    for i in range(m):
        seq = sorted(sets[i], key=lambda j: (d[j], j))
        t = 0
        kept = []
        for j in seq:
            if t + p[j][i] <= d[j]:
                t += p[j][i]
                comp[str(j)] = float(t)
                kept.append(j)
        schedule[str(i)] = kept
        on.extend(kept)
    onset = set(on)
    tardy = [j for j in range(n) if j not in onset]
    obj = float(sum(w[j] for j in tardy))
    sol = {'objective_value': obj, 'schedule': schedule, 'completion_times': comp,
           'tardy_jobs': tardy, 'on_time_jobs': sorted(on)}
    return obj, sol


def solve_tardy(inst, deadline, logger):
    n = inst['num_jobs']
    m = inst['num_machines']
    w = inst['jobs']['weights']
    p = inst['jobs']['processing_times']
    d = inst['jobs']['due_dates']

    if n == 0:
        sol = {'objective_value': 0.0,
               'schedule': {str(i): [] for i in range(m)},
               'completion_times': {}, 'tardy_jobs': [], 'on_time_jobs': []}
        if logger:
            logger.log_solution(0.0, sol)
        return 0.0, sol

    def build_t(jobs, i):
        seq = sorted(jobs, key=lambda j: (d[j], j))
        keys = [(d[j], j) for j in seq]
        ct = []
        t = 0
        for j in seq:
            t += p[j][i]
            ct.append(t)
        L = len(seq)
        suf = [INF] * (L + 1)
        for s in range(L - 1, -1, -1):
            sl = d[seq[s]] - ct[s]
            suf[s] = sl if sl < suf[s + 1] else suf[s + 1]
        return [seq, keys, ct, suf]

    def can_insert(st, i, j):
        pji = p[j][i]
        if pji > d[j]:
            return False
        pos = bisect.bisect_left(st[1], (d[j], j))
        before = st[2][pos - 1] if pos > 0 else 0
        if before + pji > d[j]:
            return False
        if st[3][pos] < pji:
            return False
        return True

    def feasible_set(jobs, i):
        t = 0
        for j in sorted(jobs, key=lambda jj: (d[jj], jj)):
            t += p[j][i]
            if t > d[j]:
                return False
        return True

    def construct(order):
        sts = [build_t([], i) for i in range(m)]
        sets = [[] for _ in range(m)]
        for j in order:
            best = None
            for i in range(m):
                if can_insert(sts[i], i, j):
                    key = p[j][i]
                    if best is None or key < best[0]:
                        best = (key, i)
            if best is not None:
                i = best[1]
                sets[i].append(j)
                sts[i] = build_t(sets[i], i)
        return sets, sts

    def improve(sets, sts):
        assign = [-1] * n
        for i in range(m):
            for j in sets[i]:
                assign[j] = i
        changed = True
        while changed and time.time() < deadline:
            changed = False
            tardy = sorted([j for j in range(n) if assign[j] == -1], key=lambda j: -w[j])
            for j in tardy:
                if time.time() > deadline:
                    break
                done = False
                for i in range(m):
                    if can_insert(sts[i], i, j):
                        sets[i].append(j)
                        assign[j] = i
                        sts[i] = build_t(sets[i], i)
                        done = True
                        changed = True
                        break
                if done:
                    continue
                bestopt = None
                for i in range(m):
                    if p[j][i] > d[j]:
                        continue
                    for k in sets[i]:
                        if w[k] >= w[j]:
                            continue
                        cand = [x for x in sets[i] if x != k] + [j]
                        if feasible_set(cand, i):
                            g = w[j] - w[k]
                            if bestopt is None or g > bestopt[0]:
                                bestopt = (g, i, k)
                if bestopt is not None:
                    g, i, k = bestopt
                    sets[i] = [x for x in sets[i] if x != k] + [j]
                    sts[i] = build_t(sets[i], i)
                    assign[j] = i
                    assign[k] = -1
                    changed = True
        return sets, sts

    # ---- multi-rule greedy construction ----
    edd = sorted(range(n), key=lambda j: (d[j], j))
    orders = [
        edd,
        sorted(range(n), key=lambda j: (-w[j], d[j], j)),
        sorted(range(n), key=lambda j: (-(w[j] / max(1.0, min(p[j][i] for i in range(m)))), d[j], j)),
        sorted(range(n), key=lambda j: (d[j], -w[j], j)),
    ]
    best_sets = None
    best_sts = None
    best_onw = -1
    for od in orders:
        sets, sts = construct(od)
        onw = sum(w[j] for s in sets for j in s)
        if onw > best_onw:
            best_onw = onw
            best_sets, best_sts = sets, sts
        if time.time() > deadline:
            break

    best_obj, best_sol = tardy_solution(best_sets, n, m, w, p, d)
    if logger:
        logger.log_solution(best_obj, best_sol)

    best_sets, best_sts = improve(best_sets, best_sts)
    obj2, sol2 = tardy_solution(best_sets, n, m, w, p, d)
    if obj2 < best_obj - 1e-9:
        best_obj, best_sol = obj2, sol2
        if logger:
            logger.log_solution(best_obj, best_sol)

    holder = [best_obj, [list(s) for s in best_sets]]

    # ---- exact MIP (EDD prefix formulation) ----
    est = m * n * n / 2.0
    use_mip = est <= 3e6 and (deadline - time.time()) > 5
    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            mod = gp.Model('tardy')
            mod.Params.OutputFlag = 0
            mod.Params.Seed = 0
            mod.Params.MIPGap = 1e-4
            mod.Params.NumericFocus = 0
            mod.Params.Threads = 1
            mod.Params.TimeLimit = max(1.0, deadline - time.time() - 1.0)
            x = {}
            for i in range(m):
                for j in range(n):
                    if p[j][i] <= d[j]:
                        x[(i, j)] = mod.addVar(vtype=GRB.BINARY)
            for j in range(n):
                vs = [x[(i, j)] for i in range(m) if (i, j) in x]
                if vs:
                    mod.addConstr(gp.quicksum(vs) <= 1)
            for i in range(m):
                coeffs = []
                vlist = []
                tot = 0
                for j in edd:
                    if (i, j) not in x:
                        continue
                    coeffs.append(p[j][i])
                    vlist.append(x[(i, j)])
                    tot += p[j][i]
                    M = tot - d[j]
                    if M > 0:
                        expr = gp.LinExpr(coeffs, vlist)
                        expr.add(x[(i, j)], M)
                        mod.addConstr(expr <= d[j] + M)
            mod.setObjective(gp.quicksum(w[j] * x[(i, j)] for (i, j) in x), GRB.MAXIMIZE)
            assign_best = [-1] * n
            for i in range(m):
                for j in holder[1][i]:
                    assign_best[j] = i
            for (i, j), v in x.items():
                v.Start = 1.0 if assign_best[j] == i else 0.0
            keys = list(x.keys())
            xvars = [x[k] for k in keys]

            def cb(mm, where):
                if where == GRB.Callback.MIPSOL:
                    vals = mm.cbGetSolution(xvars)
                    sets2 = [[] for _ in range(m)]
                    for t2 in range(len(keys)):
                        if vals[t2] > 0.5:
                            i2, j2 = keys[t2]
                            sets2[i2].append(j2)
                    ob, so = tardy_solution(sets2, n, m, w, p, d)
                    if ob < holder[0] - 1e-9:
                        holder[0] = ob
                        holder[1] = [list(s) for s in sets2]
                        if logger:
                            logger.log_solution(ob, so)

            mod.optimize(cb)
            if mod.SolCount > 0:
                sets2 = [[] for _ in range(m)]
                for (i, j), v in x.items():
                    if v.X > 0.5:
                        sets2[i].append(j)
                ob, _ = tardy_solution(sets2, n, m, w, p, d)
                if ob < holder[0] - 1e-9:
                    holder[0] = ob
                    holder[1] = [list(s) for s in sets2]
        except Exception:
            pass
    else:
        # ---- ILS fallback for very large instances ----
        rng = random.Random(0)
        while time.time() < deadline - 0.3:
            base = holder[1]
            allon = [(i, j) for i in range(m) for j in base[i]]
            newsets = [list(s) for s in base]
            if allon:
                k = max(1, len(allon) // 10)
                rem = set(rng.sample(range(len(allon)), min(k, len(allon))))
                newsets = [[] for _ in range(m)]
                for idx, (i, j) in enumerate(allon):
                    if idx not in rem:
                        newsets[i].append(j)
            sts2 = [build_t(newsets[i], i) for i in range(m)]
            newsets, sts2 = improve(newsets, sts2)
            ob, so = tardy_solution(newsets, n, m, w, p, d)
            if ob < holder[0] - 1e-9:
                holder[0] = ob
                holder[1] = [list(s) for s in newsets]
                if logger:
                    logger.log_solution(ob, so)

    best_obj, best_sol = tardy_solution(holder[1], n, m, w, p, d)
    return best_obj, best_sol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start = time.time()
    with open(args.instance_path) as f:
        inst = json.load(f)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense='minimize')
        except Exception:
            logger = None

    deadline = start + max(1.0, args.time_limit - 2.0)

    ptype = inst.get('problem_type', '')
    is_tardy = (ptype == 'weighted_tardy_jobs' or
                inst.get('objective', '') == 'minimize_weighted_number_of_tardy_jobs')

    n = inst['num_jobs']
    m = inst['num_machines']
    w = inst['jobs']['weights']
    p = inst['jobs']['processing_times']

    try:
        if is_tardy:
            obj, sol = solve_tardy(inst, deadline, logger)
        else:
            obj, sol = solve_wct(inst, deadline, logger)
    except Exception:
        # fallback trivial feasible solution
        if is_tardy:
            d = inst['jobs']['due_dates']
            obj, sol = tardy_solution([[] for _ in range(m)], n, m, w, p, d)
        else:
            obj, sol = wct_solution([0] * n, n, m, w, p)
        if logger:
            try:
                logger.log_solution(obj, sol)
            except Exception:
                pass

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()