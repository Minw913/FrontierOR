import argparse
import json
import time
import random
import bisect
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 1.0

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)
    J = data['num_jobs']
    M = data['num_machines']
    jobs = [None] * J
    for job in data['jobs']:
        ops_sorted = sorted(job['operations'], key=lambda x: x['operation_id'])
        ops = []
        for op in ops_sorted:
            pt = {int(k): int(v) for k, v in op['processing_times'].items()}
            elig = {int(m): pt[int(m)] for m in op['eligible_machines']}
            ops.append(elig)
        jobs[job['job_id']] = ops

    ops_list = [(j, o) for j in range(J) for o in range(len(jobs[j]))]
    N = len(ops_list)

    if N == 0:
        sol = {"objective_value": 0.0, "assignment": {}, "schedule": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        if logger:
            logger.log_solution(0.0, sol)
        return

    rng = random.Random(0)

    # ---------- helpers ----------
    def build_sol(assign, starts):
        a = {}
        sched = {}
        cmax = 0
        for (j, o), m in assign.items():
            p = jobs[j][o][m]
            s = int(starts[(j, o)])
            a["({},{})".format(j, o)] = int(m)
            sched["({},{})".format(j, o)] = list(range(s + 1, s + p + 1))
            if s + p > cmax:
                cmax = s + p
        sol = {"objective_value": float(cmax + 1), "assignment": a, "schedule": sched}
        return sol, cmax

    def greedy(noise):
        mach_avail = [0] * M
        job_ready = [0] * J
        nxt = [0] * J
        assign = {}
        starts = {}
        seq = []
        for _ in range(N):
            cands = []
            for j in range(J):
                o = nxt[j]
                if o >= len(jobs[j]):
                    continue
                r = job_ready[j]
                for m, p in jobs[j][o].items():
                    ma = mach_avail[m]
                    est = r if r > ma else ma
                    cands.append((est + p, est, rng.random(), j, o, m, p))
            cands.sort()
            if noise > 0 and len(cands) > 1 and rng.random() < noise:
                idx = rng.randrange(min(3, len(cands)))
            else:
                idx = 0
            fin, est, _, j, o, m, p = cands[idx]
            assign[(j, o)] = m
            starts[(j, o)] = est
            seq.append((j, o))
            mach_avail[m] = fin
            job_ready[j] = fin
            nxt[j] += 1
        cmax = max((starts[k] + jobs[k[0]][k[1]][assign[k]]) for k in starts)
        return cmax, assign, starts, seq

    def decode(seq, assign):
        # active schedule generation with gap insertion
        mach = [[] for _ in range(M)]  # sorted list of (start, end)
        job_ready = [0] * J
        starts = {}
        cmax = 0
        for (j, o) in seq:
            m = assign[(j, o)]
            p = jobs[j][o][m]
            r = job_ready[j]
            ints = mach[m]
            prev = 0
            s = None
            for (a, b) in ints:
                c = prev if prev > r else r
                if c + p <= a:
                    s = c
                    break
                if b > prev:
                    prev = b
            if s is None:
                s = prev if prev > r else r
            bisect.insort(ints, (s, s + p))
            starts[(j, o)] = s
            e = s + p
            job_ready[j] = e
            if e > cmax:
                cmax = e
        return cmax, starts

    # ---------- phase 1: randomized greedy restarts ----------
    best_cmax = None
    best_sol = None
    best_assign = None
    best_starts = None
    best_seq = None

    def register(cmax, assign, starts, seq):
        nonlocal best_cmax, best_sol, best_assign, best_starts, best_seq
        if best_cmax is None or cmax < best_cmax:
            best_cmax = cmax
            best_assign = dict(assign)
            best_starts = dict(starts)
            best_seq = list(seq)
            best_sol, _ = build_sol(best_assign, best_starts)
            if logger:
                logger.log_solution(best_sol["objective_value"], best_sol)

    # first deterministic pass (always executed)
    c0, a0, s0, q0 = greedy(0.0)
    c1, st1 = decode(q0, a0)
    if c1 <= c0:
        register(c1, a0, st1, q0)
    else:
        register(c0, a0, s0, q0)

    # decide whether MIP is applicable
    use_mip = False
    pairs = []
    if N <= 120:
        try:
            import gurobipy as gp  # noqa
            paircost = 0
            msets = {op: set(jobs[op[0]][op[1]].keys()) for op in ops_list}
            for i in range(N):
                a = ops_list[i]
                for k in range(i + 1, N):
                    b = ops_list[k]
                    if a[0] == b[0]:
                        continue
                    common = msets[a] & msets[b]
                    if common:
                        pairs.append((a, b, common))
                        paircost += len(common)
                        if paircost > 40000:
                            break
                if paircost > 40000:
                    break
            if paircost <= 40000 and (deadline - time.time()) > 5.0:
                use_mip = True
        except Exception:
            use_mip = False

    phase1_frac = 0.15 if use_mip else 0.35
    phase1_end = min(deadline, t0 + max(2.0, phase1_frac * args.time_limit))
    while time.time() < phase1_end:
        c, a, s, q = greedy(0.35)
        c2, st2 = decode(q, a)
        if c2 <= c:
            register(c2, a, st2, q)
        else:
            register(c, a, s, q)

    # ---------- phase 2 ----------
    if use_mip and time.time() < deadline - 2.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            H = float(best_cmax)
            mdl = gp.Model("fjsp")
            mdl.Params.OutputFlag = 0
            mdl.Params.Seed = 0
            mdl.Params.MIPGap = 1e-4
            mdl.Params.NumericFocus = 0
            mdl.Params.Threads = 1
            mdl.Params.TimeLimit = max(1.0, deadline - time.time())

            y = {}
            s = {}
            for (j, o) in ops_list:
                s[(j, o)] = mdl.addVar(vtype=GRB.INTEGER, lb=0, ub=H, name="s_%d_%d" % (j, o))
                for m in jobs[j][o]:
                    y[(j, o, m)] = mdl.addVar(vtype=GRB.BINARY, name="y_%d_%d_%d" % (j, o, m))
            cm = mdl.addVar(vtype=GRB.INTEGER, lb=0, ub=H, name="cmax")

            for (j, o) in ops_list:
                mdl.addConstr(gp.quicksum(y[(j, o, m)] for m in jobs[j][o]) == 1)
                pexpr = gp.quicksum(jobs[j][o][m] * y[(j, o, m)] for m in jobs[j][o])
                mdl.addConstr(cm >= s[(j, o)] + pexpr)
                if o + 1 < len(jobs[j]):
                    mdl.addConstr(s[(j, o + 1)] >= s[(j, o)] + pexpr)

            z = {}
            for (a, b, common) in pairs:
                zz = mdl.addVar(vtype=GRB.BINARY)
                z[(a, b)] = zz
                for m in common:
                    pa = jobs[a[0]][a[1]][m]
                    pb = jobs[b[0]][b[1]][m]
                    ya = y[(a[0], a[1], m)]
                    yb = y[(b[0], b[1], m)]
                    mdl.addConstr(s[b] >= s[a] + pa - H * (3 - zz - ya - yb))
                    mdl.addConstr(s[a] >= s[b] + pb - H * (2 + zz - ya - yb))

            mdl.setObjective(cm, GRB.MINIMIZE)

            # warm start
            for key, var in y.items():
                var.Start = 1.0 if best_assign[(key[0], key[1])] == key[2] else 0.0
            for op, var in s.items():
                var.Start = float(best_starts[op])
            cm.Start = float(best_cmax)
            for (a, b), zz in z.items():
                sa = best_starts[a]
                sb = best_starts[b]
                zz.Start = 1.0 if (sa, a) < (sb, b) else 0.0

            svars = [s[k] for k in ops_list]
            ykeys = list(y.keys())
            yvars = [y[k] for k in ykeys]

            mdl._bestc = best_cmax
            mdl._bestsol = None
            mdl._bestass = None
            mdl._bestst = None

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < model._bestc - 0.5:
                        sval = model.cbGetSolution(svars)
                        yval = model.cbGetSolution(yvars)
                        assign = {}
                        bval = {}
                        for k, v in zip(ykeys, yval):
                            op = (k[0], k[1])
                            if op not in bval or v > bval[op]:
                                bval[op] = v
                                assign[op] = k[2]
                        starts = {op: int(round(sv)) for op, sv in zip(ops_list, sval)}
                        sol, c = build_sol(assign, starts)
                        if c < model._bestc:
                            model._bestc = c
                            model._bestsol = sol
                            model._bestass = assign
                            model._bestst = starts
                            if logger:
                                logger.log_solution(sol["objective_value"], sol)

            mdl.optimize(cb)

            if mdl._bestsol is not None and mdl._bestc < best_cmax:
                best_cmax = mdl._bestc
                best_sol = mdl._bestsol
                best_assign = mdl._bestass
                best_starts = mdl._bestst
        except Exception:
            pass

    # ---------- SA phase with remaining time (also runs if MIP finished early) ----------
    if time.time() < deadline - 0.5:
        flex = [op for op in ops_list if len(jobs[op[0]][op[1]]) > 1]
        cur_seq = list(best_seq)
        cur_assign = dict(best_assign)
        cur_c, cur_st = decode(cur_seq, cur_assign)
        T0s = max(2.0, 0.04 * best_cmax)
        T = T0s
        while True:
            now = time.time()
            if now >= deadline:
                break
            r = rng.random()
            moved = False
            c = cur_c
            st = None
            if flex and r < 0.4:
                op = flex[rng.randrange(len(flex))]
                old = cur_assign[op]
                choices = [m for m in jobs[op[0]][op[1]] if m != old]
                if choices:
                    newm = choices[rng.randrange(len(choices))]
                    cur_assign[op] = newm
                    c, st = decode(cur_seq, cur_assign)
                    if c <= cur_c or rng.random() < math.exp((cur_c - c) / max(T, 1e-9)):
                        cur_c = c
                        moved = True
                    else:
                        cur_assign[op] = old
            else:
                i = rng.randrange(N)
                k = rng.randrange(N)
                if cur_seq[i][0] != cur_seq[k][0]:
                    cur_seq[i], cur_seq[k] = cur_seq[k], cur_seq[i]
                    c, st = decode(cur_seq, cur_assign)
                    if c <= cur_c or rng.random() < math.exp((cur_c - c) / max(T, 1e-9)):
                        cur_c = c
                        moved = True
                    else:
                        cur_seq[i], cur_seq[k] = cur_seq[k], cur_seq[i]
            if moved and st is not None and c < best_cmax:
                best_cmax = c
                best_assign = dict(cur_assign)
                best_starts = dict(st)
                best_seq = list(cur_seq)
                best_sol, _ = build_sol(best_assign, best_starts)
                if logger:
                    logger.log_solution(best_sol["objective_value"], best_sol)
            # occasional reset to best if drifted too far
            if cur_c > best_cmax * 1.25:
                cur_seq = list(best_seq)
                cur_assign = dict(best_assign)
                cur_c, cur_st = decode(cur_seq, cur_assign)
            T *= 0.9995
            if T < 0.3:
                T = T0s * rng.uniform(0.3, 1.0)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()