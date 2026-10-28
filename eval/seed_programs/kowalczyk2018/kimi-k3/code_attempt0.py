import argparse
import bisect
import functools
import heapq
import json
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    T = max(1, int(args.time_limit))
    reserve = max(0.3, min(2.0, 0.03 * T))
    hard = t0 + T - reserve  # hard deadline for all computation

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        inst = json.load(f)
    jobs = inst["jobs"]
    n = int(inst.get("n", len(jobs)))
    m = max(1, int(inst.get("m", 1)))

    # ---- WSPT order: nonincreasing w/p (exact integer comparison, stable) ----
    order = list(range(n))
    if n > 1:
        def _cmp(a, b):
            lhs = jobs[a]["w"] * jobs[b]["p"]
            rhs = jobs[b]["w"] * jobs[a]["p"]
            if lhs > rhs:
                return -1
            if lhs < rhs:
                return 1
            return 0
        order.sort(key=functools.cmp_to_key(_cmp))

    p = [0] * n
    w = [0] * n
    orig = [0] * n
    for pos, i in enumerate(order):
        p[pos] = jobs[i]["p"]
        w[pos] = jobs[i]["w"]
        orig[pos] = jobs[i].get("j", i + 1)

    def build_solution(seqs):
        total = 0
        machines_out = []
        for seq in seqs:
            t = 0
            c = 0
            jl = []
            for g in seq:
                t += p[g]
                c += w[g] * t
                jl.append({"j_orig": orig[g], "p": p[g], "w": w[g], "C_j": t})
            machines_out.append({"schedule_cost": c, "jobs": jl})
            total += c
        return total, {"objective_value": float(total), "machines": machines_out}

    # ---- incumbent state used by heuristics ----
    seqs = [[] for _ in range(m)]   # sorted global job indices per machine
    assign = [-1] * n
    prefP = [[] for _ in range(m)]
    prefW = [[] for _ in range(m)]
    Wtot = [0] * m
    mcost = [0] * m
    cur = {"total": 0}
    best = {"obj": None, "seqs": None}

    def recompute(mm):
        seq = seqs[mm]
        k = len(seq)
        pp = [0] * k
        pw = [0] * k
        sp = 0
        sw = 0
        c = 0
        for t in range(k):
            g = seq[t]
            sp += p[g]
            sw += w[g]
            pp[t] = sp
            pw[t] = sw
            c += w[g] * sp
        prefP[mm] = pp
        prefW[mm] = pw
        Wtot[mm] = sw
        cur["total"] += c - mcost[mm]
        mcost[mm] = c

    def full_recompute():
        cur["total"] = 0
        for mm in range(m):
            mcost[mm] = 0
        for mm in range(m):
            recompute(mm)

    def reset_state():
        cur["total"] = 0
        for mm in range(m):
            seqs[mm] = []
            prefP[mm] = []
            prefW[mm] = []
            Wtot[mm] = 0
            mcost[mm] = 0

    def finalize_assign():
        for mm in range(m):
            for g in seqs[mm]:
                assign[g] = mm

    def snapshot_best():
        if best["obj"] is None or cur["total"] < best["obj"]:
            best["obj"] = cur["total"]
            best["seqs"] = [list(s) for s in seqs]
            _, sol = build_solution(best["seqs"])
            if logger:
                logger.log_solution(float(best["obj"]), sol)

    def restore_best():
        for mm in range(m):
            seqs[mm] = list(best["seqs"][mm])
        finalize_assign()
        full_recompute()

    def ins_delta(mm, g):
        # cost increase of inserting job g into machine mm (WSPT position)
        seq = seqs[mm]
        t = bisect.bisect_left(seq, g)
        if t > 0:
            Pb = prefP[mm][t - 1]
            Wa = Wtot[mm] - prefW[mm][t - 1]
        else:
            Pb = 0
            Wa = Wtot[mm]
        return w[g] * (Pb + p[g]) + p[g] * Wa

    def apply_relocate(g, B):
        A = assign[g]
        ta = bisect.bisect_left(seqs[A], g)
        seqs[A].pop(ta)
        recompute(A)
        pos = bisect.bisect_left(seqs[B], g)
        seqs[B].insert(pos, g)
        recompute(B)
        assign[g] = B

    def swap_delta(A, ta, B, tb):
        a = seqs[A][ta]
        b = seqs[B][tb]
        remA = w[a] * prefP[A][ta] + p[a] * (Wtot[A] - prefW[A][ta])
        remB = w[b] * prefP[B][tb] + p[b] * (Wtot[B] - prefW[B][tb])
        # insert b into A without a
        t = bisect.bisect_left(seqs[A], b)
        if t > 0:
            Pb = prefP[A][t - 1]
            Wa = Wtot[A] - prefW[A][t - 1]
        else:
            Pb = 0
            Wa = Wtot[A]
        if ta < t:
            Pb -= p[a]
        else:
            Wa -= w[a]
        val = w[b] * (Pb + p[b]) + p[b] * Wa - remA
        # insert a into B without b
        t2 = bisect.bisect_left(seqs[B], a)
        if t2 > 0:
            Pb2 = prefP[B][t2 - 1]
            Wa2 = Wtot[B] - prefW[B][t2 - 1]
        else:
            Pb2 = 0
            Wa2 = Wtot[B]
        if tb < t2:
            Pb2 -= p[b]
        else:
            Wa2 -= w[b]
        val += w[a] * (Pb2 + p[a]) + p[a] * Wa2 - remB
        return val

    def apply_swap(A, ta, B, tb):
        a = seqs[A].pop(ta)
        b = seqs[B].pop(tb)
        ia = bisect.bisect_left(seqs[A], b)
        seqs[A].insert(ia, b)
        ib = bisect.bisect_left(seqs[B], a)
        seqs[B].insert(ib, a)
        assign[a] = B
        assign[b] = A
        recompute(A)
        recompute(B)

    def relocate_pass(deadline):
        improved = False
        for g in range(n):
            if time.time() > deadline:
                return improved
            A = assign[g]
            ta = bisect.bisect_left(seqs[A], g)
            remA = w[g] * prefP[A][ta] + p[g] * (Wtot[A] - prefW[A][ta])
            bestB = -1
            bestd = 0
            for B in range(m):
                if B == A:
                    continue
                d = ins_delta(B, g) - remA
                if d < bestd:
                    bestd = d
                    bestB = B
            if bestB >= 0:
                apply_relocate(g, bestB)
                improved = True
        return improved

    def swap_pass(deadline):
        improved_any = False
        counter = 0
        for A in range(m):
            for B in range(A + 1, m):
                while seqs[A] and seqs[B]:
                    applied = False
                    la = len(seqs[A])
                    lb = len(seqs[B])
                    br = False
                    for ta in range(la):
                        for tb in range(lb):
                            counter += 1
                            if counter >= 4096:
                                counter = 0
                                if time.time() > deadline:
                                    return improved_any
                            if swap_delta(A, ta, B, tb) < 0:
                                apply_swap(A, ta, B, tb)
                                improved_any = True
                                applied = True
                                br = True
                                break
                        if br:
                            break
                    if not applied:
                        break
        return improved_any

    def local_search(deadline, do_swaps=True):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if relocate_pass(deadline):
                improved = True
                snapshot_best()
            if do_swaps and time.time() < deadline and swap_pass(deadline):
                improved = True
                snapshot_best()

    def construct_ratio():
        # WSPT list scheduling: jobs in ratio order to least-loaded machine
        reset_state()
        heap = [(0, mm) for mm in range(m)]
        heapq.heapify(heap)
        for g in range(n):
            load, mm = heapq.heappop(heap)
            seqs[mm].append(g)
            assign[g] = mm
            heapq.heappush(heap, (load + p[g], mm))
        full_recompute()

    def construct_order(ord_list):
        # insertion heuristic in a given job order
        reset_state()
        for g in ord_list:
            bestB = 0
            bestd = None
            for B in range(m):
                d = ins_delta(B, g)
                if bestd is None or d < bestd:
                    bestd = d
                    bestB = B
            pos = bisect.bisect_left(seqs[bestB], g)
            seqs[bestB].insert(pos, g)
            assign[g] = bestB
            recompute(bestB)

    def ils(deadline, do_swaps):
        rng = random.Random(1234567)
        max_strength = max(3, min(15, n // 10 + 2))
        while time.time() < deadline:
            restore_best()
            strength = rng.randint(2, max_strength)
            if rng.random() < 0.5:
                for _ in range(strength):
                    g = rng.randrange(n)
                    B = rng.randrange(m)
                    if B != assign[g]:
                        apply_relocate(g, B)
            else:
                for _ in range(strength):
                    A, B = rng.sample(range(m), 2)
                    if seqs[A] and seqs[B]:
                        apply_swap(A, rng.randrange(len(seqs[A])),
                                   B, rng.randrange(len(seqs[B])))
            local_search(deadline, do_swaps)
            if cur["total"] < best["obj"]:
                snapshot_best()

    def dp_two_machines():
        # exact pseudo-polynomial DP for m == 2
        from array import array
        P = sum(p)
        INF = 1 << 62
        dp = array('q', [INF]) * (P + 1)
        dp[0] = 0
        prefix = 0
        choices = []
        for g in range(n):
            if g % 64 == 0 and time.time() > hard:
                return None
            pg = p[g]
            wg = w[g]
            new_prefix = prefix + pg
            ndp = array('q', [INF]) * (P + 1)
            ch = bytearray(P + 1)
            for L in range(0, new_prefix + 1):
                bestv = INF
                c0 = 0
                if L <= prefix:
                    v = dp[L]
                    if v < INF:
                        bestv = v + wg * (new_prefix - L)
                        c0 = 0
                if L >= pg:
                    v = dp[L - pg]
                    if v < INF:
                        v2 = v + wg * L
                        if v2 < bestv:
                            bestv = v2
                            c0 = 1
                ndp[L] = bestv
                ch[L] = c0
            dp = ndp
            choices.append(ch)
            prefix = new_prefix
        L = 0
        bv = dp[0]
        for x in range(1, P + 1):
            if dp[x] < bv:
                bv = dp[x]
                L = x
        on0 = [False] * n
        for g in range(n - 1, -1, -1):
            if choices[g][L]:
                on0[g] = True
                L -= p[g]
        s0 = [g for g in range(n) if on0[g]]
        s1 = [g for g in range(n) if not on0[g]]
        return [s0, s1]

    def mip_solve():
        import gurobipy as gp
        from gurobipy import GRB
        model = gp.Model("pm_twct")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LogToConsole = 0
        x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        u = model.addVars(pairs, lb=0.0, ub=1.0, name="u")
        one = model.addVar(lb=1.0, ub=1.0, name="one")
        const = 0
        for j in range(n):
            const += w[j] * p[j]
        model.setObjective(
            gp.quicksum((w[j] * p[i]) * u[i, j] for (i, j) in pairs) + const * one,
            GRB.MINIMIZE)
        model.addConstrs((x.sum(j, "*") == 1 for j in range(n)), name="assign")
        for mm in range(m):
            model.addConstrs(
                (u[i, j] >= x[i, mm] + x[j, mm] - 1 for (i, j) in pairs),
                name="link_%d" % mm)
        # symmetry breaking: nonincreasing machine loads
        for mm in range(m - 1):
            model.addConstr(
                gp.quicksum(p[j] * x[j, mm] for j in range(n)) >=
                gp.quicksum(p[j] * x[j, mm + 1] for j in range(n)),
                name="loadord_%d" % mm)
        model.update()
        # warm start from best heuristic solution (loads sorted descending)
        order_m = sorted(range(m), key=lambda mm: -sum(p[g] for g in best["seqs"][mm]))
        assign_b = [0] * n
        for pos, mm in enumerate(order_m):
            for g in best["seqs"][mm]:
                assign_b[g] = pos
        for j in range(n):
            aj = assign_b[j]
            for mm in range(m):
                x[j, mm].Start = 1.0 if mm == aj else 0.0
        for (i, j) in pairs:
            u[i, j].Start = 1.0 if assign_b[i] == assign_b[j] else 0.0
        one.Start = 1.0

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                xv = model.cbGetSolution(x)
                new_seqs = [[] for _ in range(m)]
                for j in range(n):
                    for mm in range(m):
                        if xv[j, mm] > 0.5:
                            new_seqs[mm].append(j)
                            break
                total = 0
                for seq in new_seqs:
                    tt = 0
                    for g in seq:
                        tt += p[g]
                        total += w[g] * tt
                if total < best["obj"]:
                    best["obj"] = total
                    best["seqs"] = new_seqs
                    _, sol = build_solution(new_seqs)
                    if logger:
                        logger.log_solution(float(total), sol)

        model.Params.TimeLimit = max(0.5, hard - time.time())
        model.optimize(cb)
        return model.Status == GRB.OPTIMAL

    def finish():
        total, sol = build_solution(best["seqs"])
        sol["objective_value"] = float(total)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    # ---- fallback feasible solution: all jobs on machine 0 (WSPT) ----
    if n > 0:
        seqs[0] = list(range(n))
        finalize_assign()
        full_recompute()
        snapshot_best()
    else:
        best["obj"] = 0
        best["seqs"] = [[] for _ in range(m)]

    # ---- trivial / exactly solvable cases ----
    if m == 1 or n == 0:
        finish()
        return
    if m >= n:
        reset_state()
        for g in range(n):
            seqs[g] = [g]
        finalize_assign()
        full_recompute()
        snapshot_best()
        finish()
        return

    try:
        if m == 2:
            Pall = sum(p)
            sumw = sum(w)
            if (Pall + 1) <= 1500000 and n * (Pall + 1) <= 3000000 \
                    and sumw * max(1, Pall) < (1 << 62):
                res = dp_two_machines()
                if res is not None:
                    reset_state()
                    seqs[0] = res[0]
                    seqs[1] = res[1]
                    finalize_assign()
                    full_recompute()
                    snapshot_best()
                    finish()
                    return

        do_swaps = n <= 1800

        # ---- constructive phase ----
        construct_ratio()
        saved = [list(s) for s in seqs]
        saved_total = cur["total"]
        if n <= 4000 and time.time() < hard:
            ord2 = sorted(range(n), key=lambda g: (-w[g], g))
            construct_order(ord2)
            if cur["total"] >= saved_total:
                for mm in range(m):
                    seqs[mm] = list(saved[mm])
                finalize_assign()
                full_recompute()
        snapshot_best()

        # ---- local search phase ----
        est_con = m * n * (n - 1) // 2
        mip_ok = (n <= 450) and (est_con <= 200000)
        if mip_ok:
            ls_end = min(hard, time.time() + max(0.5, 0.10 * T))
        else:
            ls_end = min(hard, time.time() + min(0.45 * T, 90.0))
        local_search(ls_end, do_swaps)
        snapshot_best()

        # ---- exact MIP (small/medium) or iterated local search (large) ----
        solved_opt = False
        if mip_ok and time.time() < hard - 2.0:
            try:
                solved_opt = mip_solve()
            except Exception:
                solved_opt = False
        if (not solved_opt) and time.time() < hard:
            ils(hard, do_swaps)
    except Exception:
        pass

    finish()


if __name__ == "__main__":
    main()