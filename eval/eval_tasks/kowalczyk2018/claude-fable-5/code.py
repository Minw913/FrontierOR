import argparse
import json
import time
import bisect
import random
import functools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.5

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    m = int(data["m"])
    jobs_in = data.get("jobs", [])

    # Trivial cases
    if n == 0 or m == 0:
        sol = {"objective_value": 0.0,
               "machines": [{"schedule_cost": 0, "jobs": []} for _ in range(m)]}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # Sort jobs by WSPT (nonincreasing w/p). Instance claims sorted already,
    # but re-sort defensively using exact integer comparison.
    def cmp(a, b):
        l = a["w"] * b["p"]
        r = b["w"] * a["p"]
        if l > r:
            return -1
        if l < r:
            return 1
        # tie -> keep input order by index
        if a["j"] < b["j"]:
            return -1
        if a["j"] > b["j"]:
            return 1
        return 0

    order = sorted(jobs_in, key=functools.cmp_to_key(cmp))
    p = [int(jb["p"]) for jb in order]
    w = [int(jb["w"]) for jb in order]
    orig = [int(jb["j"]) for jb in order]

    # ------------------------------------------------------------------
    # State: partition of ranks (0..n-1) into m machines. On each machine
    # jobs are processed in increasing rank order (WSPT-optimal).
    # ------------------------------------------------------------------
    ranks = [[] for _ in range(m)]      # sorted lists of ranks
    mach_of = [0] * n
    pp = [[] for _ in range(m)]         # prefix processing time (inclusive) = C_j
    pw = [[] for _ in range(m)]         # prefix weight (inclusive)
    tw = [0] * m                        # total weight on machine
    cost = [0] * m                      # weighted completion cost on machine

    def recompute(k):
        lst = ranks[k]
        t = 0
        cw = 0
        c = 0
        ap_ = []
        aw_ = []
        for r in lst:
            t += p[r]
            cw += w[r]
            c += w[r] * t
            ap_.append(t)
            aw_.append(cw)
        pp[k] = ap_
        pw[k] = aw_
        tw[k] = cw
        cost[k] = c

    def seq_cost(lst):
        t = 0
        c = 0
        for r in lst:
            t += p[r]
            c += w[r] * t
        return c

    # ---------------- greedy WSPT list scheduling ----------------
    import heapq
    heap = [(0, k) for k in range(m)]
    heapq.heapify(heap)
    for t in range(n):
        load, k = heapq.heappop(heap)
        ranks[k].append(t)
        mach_of[t] = k
        heapq.heappush(heap, (load + p[t], k))
    for k in range(m):
        recompute(k)

    def build_solution(rl):
        machines = []
        total = 0
        for k in range(m):
            t = 0
            sc = 0
            jl = []
            for r in rl[k]:
                t += p[r]
                sc += w[r] * t
                jl.append({"j_orig": orig[r], "p": p[r], "w": w[r], "C_j": t})
            total += sc
            machines.append({"schedule_cost": sc, "jobs": jl})
        return {"objective_value": float(total), "machines": machines}, total

    best_total = sum(cost)
    best_ranks = [list(r) for r in ranks]
    if logger:
        sol, _ = build_solution(best_ranks)
        logger.log_solution(float(best_total), sol)

    def snapshot_and_log():
        nonlocal best_total, best_ranks
        tot = sum(cost)
        if tot < best_total:
            best_total = tot
            best_ranks = [list(r) for r in ranks]
            if logger:
                sol, _ = build_solution(best_ranks)
                logger.log_solution(float(tot), sol)

    def set_state(rl):
        for k in range(m):
            ranks[k] = list(rl[k])
            for r in ranks[k]:
                mach_of[r] = k
            recompute(k)

    # ---------------- move / swap machinery ----------------
    def eval_move(j, a, b):
        i = bisect.bisect_left(ranks[a], j)
        Cj = pp[a][i]
        suf_w = tw[a] - pw[a][i]
        gain = w[j] * Cj + p[j] * suf_w
        pos = bisect.bisect_left(ranks[b], j)
        pre = pp[b][pos - 1] if pos > 0 else 0
        prew = pw[b][pos - 1] if pos > 0 else 0
        sufw_b = tw[b] - prew
        add = w[j] * (pre + p[j]) + p[j] * sufw_b
        return add - gain

    def apply_move(j, a, b):
        i = bisect.bisect_left(ranks[a], j)
        ranks[a].pop(i)
        bisect.insort(ranks[b], j)
        mach_of[j] = b
        recompute(a)
        recompute(b)

    def eval_swap(j, k):
        a = mach_of[j]
        b = mach_of[k]
        if a == b:
            return 0
        la = [r for r in ranks[a] if r != j]
        bisect.insort(la, k)
        lb = [r for r in ranks[b] if r != k]
        bisect.insort(lb, j)
        return seq_cost(la) + seq_cost(lb) - cost[a] - cost[b]

    def apply_swap(j, k):
        a = mach_of[j]
        b = mach_of[k]
        ia = bisect.bisect_left(ranks[a], j)
        ranks[a].pop(ia)
        ib = bisect.bisect_left(ranks[b], k)
        ranks[b].pop(ib)
        bisect.insort(ranks[a], k)
        bisect.insort(ranks[b], j)
        mach_of[j] = b
        mach_of[k] = a
        recompute(a)
        recompute(b)

    rng = random.Random(0)

    def move_pass(dl):
        improved = False
        for j in range(n):
            if (j & 63) == 0 and time.time() > dl:
                return improved
            a = mach_of[j]
            bestd = 0
            bestb = -1
            for b in range(m):
                if b == a:
                    continue
                d = eval_move(j, a, b)
                if d < bestd:
                    bestd = d
                    bestb = b
            if bestb >= 0:
                apply_move(j, a, bestb)
                improved = True
        return improved

    def swap_pass(dl):
        improved = False
        cnt = 0
        if n <= 250:
            for j in range(n):
                for k in range(j + 1, n):
                    cnt += 1
                    if (cnt & 255) == 0 and time.time() > dl:
                        return improved
                    if mach_of[j] == mach_of[k]:
                        continue
                    d = eval_swap(j, k)
                    if d < 0:
                        apply_swap(j, k)
                        improved = True
        else:
            trials = min(40000, 8 * n)
            for _ in range(trials):
                cnt += 1
                if (cnt & 255) == 0 and time.time() > dl:
                    return improved
                j = rng.randrange(n)
                k = rng.randrange(n)
                if j == k or mach_of[j] == mach_of[k]:
                    continue
                d = eval_swap(j, k)
                if d < 0:
                    apply_swap(j, k)
                    improved = True
        return improved

    def local_search(dl):
        while time.time() < dl:
            imp1 = move_pass(dl)
            snapshot_and_log()
            if time.time() > dl:
                break
            imp2 = swap_pass(dl)
            snapshot_and_log()
            if not (imp1 or imp2):
                break

    # ---------------- initial local search ----------------
    if m > 1 and n > m:
        local_search(deadline)
    snapshot_and_log()

    # ---------------- exact MIP for small instances ----------------
    solved_optimal = False
    trivial = (m == 1) or (m >= n)
    use_mip = (not trivial) and (n <= 200) and (n * n * m <= 600000) and \
              (deadline - time.time() > 5.0)
    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("pwc")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1.0, deadline - time.time() - 1.0)

            x = {}
            for t in range(n):
                for k in range(min(t + 1, m)):
                    x[t, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{k}")

            const = sum(w[t] * p[t] for t in range(n))
            obj = gp.LinExpr()
            obj.addConstant(const)
            y = {}
            for i in range(n):
                if p[i] == 0:
                    continue
                for jj in range(i + 1, n):
                    if w[jj] == 0:
                        continue
                    v = model.addVar(lb=0.0, ub=1.0, name=f"y_{i}_{jj}")
                    y[i, jj] = v
                    obj.add(v, p[i] * w[jj])

            for t in range(n):
                model.addConstr(
                    gp.quicksum(x[t, k] for k in range(min(t + 1, m))) == 1)

            for (i, jj), v in y.items():
                for k in range(min(i + 1, m)):
                    model.addConstr(v >= x[i, k] + x[jj, k] - 1)

            model.setObjective(obj, GRB.MINIMIZE)

            # Warm start with canonical machine relabeling so that job t
            # (in WSPT order) uses machine index <= t.
            assign = [0] * n
            for k in range(m):
                for r in best_ranks[k]:
                    assign[r] = k
            relabel = {}
            nxt = 0
            new_assign = [0] * n
            for t in range(n):
                a = assign[t]
                if a not in relabel:
                    relabel[a] = nxt
                    nxt += 1
                new_assign[t] = relabel[a]
            for t in range(n):
                for k in range(min(t + 1, m)):
                    x[t, k].Start = 1.0 if new_assign[t] == k else 0.0

            xvars = list(x.values())
            xkeys = list(x.keys())
            model._loginc = float("inf")

            def cb(mdl, where):
                if where == GRB.Callback.MIPSOL:
                    objv = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if objv < mdl._loginc - 1e-6:
                        mdl._loginc = objv
                        if logger:
                            try:
                                vals = mdl.cbGetSolution(xvars)
                                asg = [0] * n
                                for (t, k), val in zip(xkeys, vals):
                                    if val > 0.5:
                                        asg[t] = k
                                rl = [[] for _ in range(m)]
                                for t in range(n):
                                    rl[asg[t]].append(t)
                                sol, tot = build_solution(rl)
                                logger.log_solution(float(tot), sol)
                            except Exception:
                                logger.log(float(objv))

            model.optimize(cb)

            if model.SolCount > 0:
                asg = [0] * n
                for (t, k), var in x.items():
                    if var.X > 0.5:
                        asg[t] = k
                rl = [[] for _ in range(m)]
                for t in range(n):
                    rl[asg[t]].append(t)
                tot = sum(seq_cost(rl[k]) for k in range(m))
                if tot < best_total:
                    best_total = tot
                    best_ranks = [list(r) for r in rl]
                    if logger:
                        sol, _ = build_solution(best_ranks)
                        logger.log_solution(float(tot), sol)
                if model.Status == GRB.OPTIMAL:
                    solved_optimal = True
        except Exception:
            pass

    # ---------------- iterated local search with remaining time ----------------
    if not solved_optimal and not trivial:
        set_state(best_ranks)
        while time.time() < deadline - 0.2:
            # perturbation
            q = rng.randint(2, min(6, n))
            for _ in range(q):
                j = rng.randrange(n)
                a = mach_of[j]
                if m <= 1:
                    break
                b = rng.randrange(m)
                while b == a:
                    b = rng.randrange(m)
                apply_move(j, a, b)
            local_search(deadline)
            cur = sum(cost)
            if cur < best_total:
                best_total = cur
                best_ranks = [list(r) for r in ranks]
                if logger:
                    sol, _ = build_solution(best_ranks)
                    logger.log_solution(float(cur), sol)
            else:
                set_state(best_ranks)

    # ---------------- write final solution ----------------
    sol, tot = build_solution(best_ranks)
    sol["objective_value"] = float(tot)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()