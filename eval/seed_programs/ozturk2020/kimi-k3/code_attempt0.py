import argparse
import json
import random
import time


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    time_limit = max(1, int(args.time_limit))
    deadline = t0 + float(time_limit)
    margin = 2.0

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    jobs_in = inst["jobs"]
    n = int(inst.get("num_jobs", len(jobs_in)))
    M = int(inst.get("num_machines", 1))
    Cap = int(inst["batch_capacity"])
    job_ids = [int(jj["job_id"]) for jj in jobs_in]
    r = [int(jj["release_date"]) for jj in jobs_in]
    p = [int(jj["processing_time"]) for jj in jobs_in]
    s = [int(jj["size"]) for jj in jobs_in]

    if n == 0:
        out = {"objective_value": 0.0, "batches": [], "job_assignments": {}}
        with open(args.solution_path, "w") as f:
            json.dump(out, f)
        if logger is not None:
            try:
                logger.log_solution(0.0, out)
            except Exception:
                pass
        return

    M = max(1, min(M, n))

    # ----------------- basic helpers -----------------
    def schedule_machine(mb):
        prev = 0
        tot = 0
        for b in mb:
            mr = b["mr"]
            st = prev if prev > mr else mr
            fin = st + b["mp"]
            b["start"] = st
            b["finish"] = fin
            prev = fin
            tot += len(b["jobs"]) * fin
        return tot

    def total_obj(machines):
        tot = 0
        for mb in machines:
            for b in mb:
                tot += len(b["jobs"]) * b["finish"]
        return tot

    def simulate(mb, start_idx, prev_finish, mods):
        total = 0
        L = len(mb)
        for k in range(start_idx, L):
            v = mods.get(k) if mods else None
            if v is None:
                b = mb[k]
                cnt = len(b["jobs"])
                mr = b["mr"]
                mp = b["mp"]
            else:
                cnt, mr, mp = v
            st = prev_finish if prev_finish > mr else mr
            fin = st + mp
            total += cnt * fin
            prev_finish = fin
        return total, prev_finish

    def index_of(lst, obj):
        for k in range(len(lst)):
            if lst[k] is obj:
                return k
        return -1

    def snapshot(machines):
        return [[{"jobs": b["jobs"][:], "size": b["size"], "mr": b["mr"],
                  "mp": b["mp"], "start": b["start"], "finish": b["finish"]}
                 for b in mb] for mb in machines]

    def build_output(machines):
        batches_out = []
        assigns = {}
        total = 0
        for m in range(M):
            bi = 0
            for b in machines[m]:
                if not b["jobs"]:
                    continue
                bi += 1
                fin = b["finish"]
                batches_out.append({
                    "batch_id": bi,
                    "machine": m,
                    "start_time": b["start"],
                    "processing_time": b["mp"],
                    "jobs": [job_ids[j] for j in b["jobs"]],
                })
                for j in b["jobs"]:
                    assigns[str(job_ids[j])] = {
                        "batch": bi,
                        "machine": m,
                        "completion_time": fin,
                    }
                total += len(b["jobs"]) * fin
        return {"objective_value": float(total),
                "batches": batches_out,
                "job_assignments": assigns}

    state = {"obj": None, "snap": None}

    def register_incumbent(obj, machines):
        o = int(round(obj))
        if state["obj"] is None or o < state["obj"]:
            state["obj"] = o
            state["snap"] = snapshot(machines)
            if logger is not None:
                try:
                    logger.log_solution(float(o), build_output(machines))
                except Exception:
                    pass

    # ----------------- greedy constructive heuristic -----------------
    def greedy(order, theta):
        machines = [[] for _ in range(M)]
        finishes = [0] * M
        for j in order:
            rj = r[j]; pj = p[j]; sj = s[j]
            best_eff = None
            best_kind = 0  # 0: new batch, 1: add to last batch
            best_m = 0
            for m in range(M):
                mb = machines[m]
                if mb:
                    last = mb[-1]
                    if last["size"] + sj <= Cap:
                        prev_finish = mb[-2]["finish"] if len(mb) >= 2 else 0
                        k = len(last["jobs"])
                        mr = last["mr"] if last["mr"] > rj else rj
                        mp = last["mp"] if last["mp"] > pj else pj
                        st = prev_finish if prev_finish > mr else mr
                        comp = st + mp
                        delta = (k + 1) * comp - k * last["finish"]
                        eff = delta * theta
                        if best_eff is None or eff < best_eff:
                            best_eff = eff; best_kind = 1; best_m = m
                st = finishes[m] if finishes[m] > rj else rj
                comp = st + pj
                if best_eff is None or comp < best_eff:
                    best_eff = comp; best_kind = 0; best_m = m
            m = best_m
            if best_kind == 1:
                last = machines[m][-1]
                last["jobs"].append(j)
                last["size"] += sj
                if rj > last["mr"]: last["mr"] = rj
                if pj > last["mp"]: last["mp"] = pj
                prev_finish = machines[m][-2]["finish"] if len(machines[m]) >= 2 else 0
                st = prev_finish if prev_finish > last["mr"] else last["mr"]
                last["start"] = st
                last["finish"] = st + last["mp"]
                finishes[m] = last["finish"]
            else:
                st = finishes[m] if finishes[m] > rj else rj
                machines[m].append({"jobs": [j], "size": sj, "mr": rj, "mp": pj,
                                    "start": st, "finish": st + pj})
                finishes[m] = st + pj
        return machines

    # ----------------- move evaluations (exact delta, no mutation) -----------------
    def eval_relocate(machines, m1, a, j, m2, B, bidx):
        mb1 = machines[m1]
        A = mb1[a]
        rj = r[j]; pj = p[j]
        if len(A["jobs"]) == 1:
            modA = (0, 0, 0)
        else:
            mrA = 0; mpA = 0
            for x in A["jobs"]:
                if x == j:
                    continue
                if r[x] > mrA: mrA = r[x]
                if p[x] > mpA: mpA = p[x]
            modA = (len(A["jobs"]) - 1, mrA, mpA)
        prev1 = mb1[a - 1]["finish"] if a > 0 else 0
        if B is None:  # append j as a singleton batch at end of machine m2
            if m2 == m1:
                old_val, _ = simulate(mb1, a, prev1, None)
                new_val, prev_after = simulate(mb1, a, prev1, {a: modA})
                st = prev_after if prev_after > rj else rj
                new_val += st + pj
                return new_val - old_val
            old1, _ = simulate(mb1, a, prev1, None)
            new1, _ = simulate(mb1, a, prev1, {a: modA})
            mb2 = machines[m2]
            lastfin = mb2[-1]["finish"] if mb2 else 0
            st = lastfin if lastfin > rj else rj
            return (new1 - old1) + (st + pj)
        modB = (len(B["jobs"]) + 1,
                B["mr"] if B["mr"] > rj else rj,
                B["mp"] if B["mp"] > pj else pj)
        if m2 == m1:
            if bidx == a:
                return None
            start_idx = a if a < bidx else bidx
            prev = mb1[start_idx - 1]["finish"] if start_idx > 0 else 0
            old_val, _ = simulate(mb1, start_idx, prev, None)
            new_val, _ = simulate(mb1, start_idx, prev, {a: modA, bidx: modB})
            return new_val - old_val
        old1, _ = simulate(mb1, a, prev1, None)
        new1, _ = simulate(mb1, a, prev1, {a: modA})
        mb2 = machines[m2]
        prev2 = mb2[bidx - 1]["finish"] if bidx > 0 else 0
        old2, _ = simulate(mb2, bidx, prev2, None)
        new2, _ = simulate(mb2, bidx, prev2, {bidx: modB})
        return (new1 - old1) + (new2 - old2)

    def apply_relocate(machines, loc, m1, a, j, target):
        mb1 = machines[m1]
        A = mb1[a]
        A["jobs"].remove(j)
        if A["jobs"]:
            A["size"] -= s[j]
            mr = 0; mp = 0
            for x in A["jobs"]:
                if r[x] > mr: mr = r[x]
                if p[x] > mp: mp = p[x]
            A["mr"] = mr; A["mp"] = mp
        else:
            mb1.pop(a)
        m2 = target[1]
        if target[0] == "batch":
            B = target[2]
            B["jobs"].append(j)
            B["size"] += s[j]
            if r[j] > B["mr"]: B["mr"] = r[j]
            if p[j] > B["mp"]: B["mp"] = p[j]
            loc[j] = (m2, B)
        else:
            B = {"jobs": [j], "size": s[j], "mr": r[j], "mp": p[j],
                 "start": 0, "finish": 0}
            machines[m2].append(B)
            loc[j] = (m2, B)
        schedule_machine(machines[m1])
        if m2 != m1:
            schedule_machine(machines[m2])

    def eval_batch_move(machines, m1, a, m2):
        mb1 = machines[m1]
        b = mb1[a]
        prev1 = mb1[a - 1]["finish"] if a > 0 else 0
        old1, _ = simulate(mb1, a, prev1, None)
        new1, _ = simulate(mb1, a, prev1, {a: (0, 0, 0)})
        mb2 = machines[m2]
        lastfin = mb2[-1]["finish"] if mb2 else 0
        st = lastfin if lastfin > b["mr"] else b["mr"]
        fin = st + b["mp"]
        return (new1 - old1) + len(b["jobs"]) * fin

    def eval_swap(machines, m1, a, j1, m2, bidx, j2):
        mb1 = machines[m1]
        A = mb1[a]
        r1 = r[j1]; p1 = p[j1]
        r2 = r[j2]; p2 = p[j2]
        mrA = 0; mpA = 0
        for x in A["jobs"]:
            if x == j1: continue
            if r[x] > mrA: mrA = r[x]
            if p[x] > mpA: mpA = p[x]
        if r2 > mrA: mrA = r2
        if p2 > mpA: mpA = p2
        modA = (len(A["jobs"]), mrA, mpA)
        mb2 = machines[m2]
        B = mb2[bidx]
        mrB = 0; mpB = 0
        for x in B["jobs"]:
            if x == j2: continue
            if r[x] > mrB: mrB = r[x]
            if p[x] > mpB: mpB = p[x]
        if r1 > mrB: mrB = r1
        if p1 > mpB: mpB = p1
        modB = (len(B["jobs"]), mrB, mpB)
        if m1 == m2:
            if a == bidx:
                return None
            start_idx = a if a < bidx else bidx
            prev = mb1[start_idx - 1]["finish"] if start_idx > 0 else 0
            old_val, _ = simulate(mb1, start_idx, prev, None)
            new_val, _ = simulate(mb1, start_idx, prev, {a: modA, bidx: modB})
            return new_val - old_val
        prev1 = mb1[a - 1]["finish"] if a > 0 else 0
        old1, _ = simulate(mb1, a, prev1, None)
        new1, _ = simulate(mb1, a, prev1, {a: modA})
        prev2 = mb2[bidx - 1]["finish"] if bidx > 0 else 0
        old2, _ = simulate(mb2, bidx, prev2, None)
        new2, _ = simulate(mb2, bidx, prev2, {bidx: modB})
        return (new1 - old1) + (new2 - old2)

    def apply_swap(machines, loc, m1, a, j1, m2, bidx, j2):
        A = machines[m1][a]
        B = machines[m2][bidx]
        A["jobs"].remove(j1)
        A["jobs"].append(j2)
        B["jobs"].remove(j2)
        B["jobs"].append(j1)
        szA = 0; mrA = 0; mpA = 0
        for x in A["jobs"]:
            szA += s[x]
            if r[x] > mrA: mrA = r[x]
            if p[x] > mpA: mpA = p[x]
        A["size"] = szA; A["mr"] = mrA; A["mp"] = mpA
        szB = 0; mrB = 0; mpB = 0
        for x in B["jobs"]:
            szB += s[x]
            if r[x] > mrB: mrB = r[x]
            if p[x] > mpB: mpB = p[x]
        B["size"] = szB; B["mr"] = mrB; B["mp"] = mpB
        schedule_machine(machines[m1])
        if m2 != m1:
            schedule_machine(machines[m2])
        loc[j1] = (m2, B)
        loc[j2] = (m1, A)

    def polish_orders(machines):
        improved = False
        for m in range(M):
            mb = machines[m]
            if len(mb) < 2:
                continue
            best_o = simulate(mb, 0, 0, None)[0]
            best_order = None
            for key in ((lambda b: (b["mr"], b["mp"])),
                        (lambda b: (b["mr"], -b["mp"])),
                        (lambda b: (b["mp"], b["mr"])),
                        (lambda b: (b["mr"], b["mp"], -len(b["jobs"])))):
                cand = sorted(mb, key=key)
                o = simulate(cand, 0, 0, None)[0]
                if o < best_o:
                    best_o = o
                    best_order = cand
            if best_order is not None:
                machines[m] = best_order
                schedule_machine(best_order)
                improved = True
        return improved

    # ----------------- iterated local search -----------------
    def local_search(machines, ls_deadline):
        if n <= 1:
            return
        loc = [None] * n
        for m in range(M):
            for b in machines[m]:
                for j in b["jobs"]:
                    loc[j] = (m, b)
        obj = total_obj(machines)
        rng = random.Random(987654321)
        best_local = obj
        best_snap = snapshot(machines)
        patience = max(500, 3 * n)
        it = 0
        since_improve = 0
        perturb_count = 0
        while True:
            it += 1
            if (it & 31) == 0 and time.time() > ls_deadline:
                break

            # occasional swap move (helps when batches are capacity-tight)
            if rng.random() < 0.12:
                j1 = rng.randrange(n)
                m1s, A = loc[j1]
                mb1s = machines[m1s]
                a_s = index_of(mb1s, A)
                m2s = rng.randrange(M)
                if machines[m2s]:
                    B = machines[m2s][rng.randrange(len(machines[m2s]))]
                    if B is not A:
                        j2 = B["jobs"][rng.randrange(len(B["jobs"]))]
                        if (A["size"] - s[j1] + s[j2] <= Cap
                                and B["size"] - s[j2] + s[j1] <= Cap):
                            b_s = index_of(machines[m2s], B)
                            d = eval_swap(machines, m1s, a_s, j1, m2s, b_s, j2)
                            if d is not None and d < 0:
                                apply_swap(machines, loc, m1s, a_s, j1, m2s, b_s, j2)
                                obj += d
                                since_improve = 0
                                if obj < best_local:
                                    best_local = obj
                                    best_snap = snapshot(machines)
                                    register_incumbent(obj, machines)
                                    perturb_count = 0

            # relocation move: pick a job, evaluate best target
            j = rng.randrange(n)
            m1, A = loc[j]
            mb1 = machines[m1]
            a = index_of(mb1, A)
            sj = s[j]
            best_delta = 0
            best_target = None
            total_batches = 0
            for mb in machines:
                total_batches += len(mb)
            if total_batches <= 25:
                for m2 in range(M):
                    for bidx, B in enumerate(machines[m2]):
                        if B is A or B["size"] + sj > Cap:
                            continue
                        d = eval_relocate(machines, m1, a, j, m2, B, bidx)
                        if d is not None and d < best_delta:
                            best_delta = d
                            best_target = ("batch", m2, B)
            else:
                for _ in range(16):
                    m2 = rng.randrange(M)
                    mb2 = machines[m2]
                    if not mb2:
                        continue
                    bidx = rng.randrange(len(mb2))
                    B = mb2[bidx]
                    if B is A or B["size"] + sj > Cap:
                        continue
                    d = eval_relocate(machines, m1, a, j, m2, B, bidx)
                    if d is not None and d < best_delta:
                        best_delta = d
                        best_target = ("batch", m2, B)
            # append-as-new-batch targets: min-finish machine, own machine, one random
            mf = 0
            mv = None
            for m2 in range(M):
                fin = machines[m2][-1]["finish"] if machines[m2] else 0
                if mv is None or fin < mv:
                    mv = fin
                    mf = m2
            for m2 in (mf, m1, rng.randrange(M)):
                d = eval_relocate(machines, m1, a, j, m2, None, None)
                if d is not None and d < best_delta:
                    best_delta = d
                    best_target = ("new", m2)
            if best_target is not None:
                apply_relocate(machines, loc, m1, a, j, best_target)
                obj += best_delta
                since_improve = 0
                if obj < best_local:
                    best_local = obj
                    best_snap = snapshot(machines)
                    register_incumbent(obj, machines)
                    perturb_count = 0
            else:
                since_improve += 1
                if since_improve >= patience:
                    # perturbation (iterated local search)
                    perturb_count += 1
                    for _ in range(3):
                        jj = rng.randrange(n)
                        mm1, AA = loc[jj]
                        aa = index_of(machines[mm1], AA)
                        m2 = rng.randrange(M)
                        if machines[m2] and rng.random() < 0.7:
                            B = machines[m2][rng.randrange(len(machines[m2]))]
                            if B is not AA and B["size"] + s[jj] <= Cap:
                                apply_relocate(machines, loc, mm1, aa, jj, ("batch", m2, B))
                                continue
                        apply_relocate(machines, loc, mm1, aa, jj, ("new", m2))
                    obj = total_obj(machines)
                    since_improve = 0
                    if perturb_count >= 10:
                        # restart from best found
                        machines[:] = snapshot(best_snap)
                        for m in range(M):
                            schedule_machine(machines[m])
                        for m in range(M):
                            for b in machines[m]:
                                for jj in b["jobs"]:
                                    loc[jj] = (m, b)
                        obj = best_local
                        perturb_count = 0

            # occasional whole-batch relocation to the best other machine
            if (it % 40) == 0:
                non_empty = [m for m in range(M) if machines[m]]
                if non_empty:
                    m1b = non_empty[rng.randrange(len(non_empty))]
                    ab = rng.randrange(len(machines[m1b]))
                    bestd = 0
                    bestm2 = None
                    for m2 in range(M):
                        if m2 == m1b:
                            continue
                        d = eval_batch_move(machines, m1b, ab, m2)
                        if d < bestd:
                            bestd = d
                            bestm2 = m2
                    if bestm2 is not None:
                        b = machines[m1b].pop(ab)
                        machines[bestm2].append(b)
                        schedule_machine(machines[m1b])
                        schedule_machine(machines[bestm2])
                        for jj in b["jobs"]:
                            loc[jj] = (bestm2, b)
                        obj += bestd
                        if obj < best_local:
                            best_local = obj
                            best_snap = snapshot(machines)
                            register_incumbent(obj, machines)
                            perturb_count = 0

            # occasional re-sequencing of batches within each machine
            if (it % 300) == 0:
                if polish_orders(machines):
                    obj = total_obj(machines)
                    if obj < best_local:
                        best_local = obj
                        best_snap = snapshot(machines)
                        register_incumbent(obj, machines)
                        perturb_count = 0

        polish_orders(machines)
        obj = total_obj(machines)
        if state["obj"] is None or obj < state["obj"]:
            register_incumbent(obj, machines)

    # ----------------- MIP (for small/medium instances) -----------------
    def run_mip(warm_machines):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return
        try:
            K = n
            max_r = max(r)
            sum_p = sum(p)
            max_p = max(p)
            horizon = max_r + sum_p
            bigM = float(horizon + max_p + 1)

            model = gp.Model("batch_sched")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.LogToConsole = 0

            x = model.addVars(n, M, K, vtype=GRB.BINARY, name="x")
            S = model.addVars(M, K, lb=0.0, ub=float(horizon + max_p), name="S")
            P = model.addVars(M, K, lb=0.0, ub=float(max_p), name="P")
            C = model.addVars(n, lb=0.0, ub=float(horizon + max_p), name="C")

            model.setObjective(gp.quicksum(C[j] for j in range(n)), GRB.MINIMIZE)

            for j in range(n):
                model.addConstr(gp.quicksum(x[j, m, k] for m in range(M) for k in range(K)) == 1.0)
            for m in range(M):
                for k in range(K):
                    model.addConstr(gp.quicksum(s[j] * x[j, m, k] for j in range(n)) <= Cap)
            for j in range(n):
                rj = r[j]; pj = p[j]
                if rj == 0 and pj == 0:
                    continue
                for m in range(M):
                    for k in range(K):
                        if rj > 0:
                            model.addConstr(S[m, k] >= rj * x[j, m, k])
                        if pj > 0:
                            model.addConstr(P[m, k] >= pj * x[j, m, k])
            for m in range(M):
                for k in range(K - 1):
                    model.addConstr(S[m, k + 1] >= S[m, k] + P[m, k])
            for j in range(n):
                for m in range(M):
                    for k in range(K):
                        model.addConstr(C[j] >= S[m, k] + P[m, k] - bigM * (1.0 - x[j, m, k]))
            # symmetry breaking: non-increasing number of jobs per machine index
            for m in range(M - 1):
                model.addConstr(
                    gp.quicksum(x[j, m, k] for j in range(n) for k in range(K))
                    >= gp.quicksum(x[j, m + 1, k] for j in range(n) for k in range(K)))
            # slot compaction: slot k+1 can be used only if slot k is used
            for m in range(M):
                for k in range(K - 1):
                    model.addConstr(
                        gp.quicksum(x[j, m, k + 1] for j in range(n))
                        <= n * gp.quicksum(x[j, m, k] for j in range(n)))

            model.update()

            x_flat = [x[j, m, k] for j in range(n) for m in range(M) for k in range(K)]
            start_vals = [0.0] * (n * M * K)
            warm_sorted = sorted(warm_machines,
                                 key=lambda mb: sum(len(b["jobs"]) for b in mb),
                                 reverse=True)
            for m in range(M):
                mb = warm_sorted[m] if m < len(warm_sorted) else []
                for k in range(min(len(mb), K)):
                    b = mb[k]
                    for j in b["jobs"]:
                        start_vals[(j * M + m) * K + k] = 1.0
                    S[m, k].Start = float(b["start"])
                    P[m, k].Start = float(b["mp"])
                    for j in b["jobs"]:
                        C[j].Start = float(b["finish"])
            model.setAttr("Start", x_flat, start_vals)

            def extract_from(vals):
                machines2 = [[] for _ in range(M)]
                seen = [False] * n
                for m in range(M):
                    for k in range(K):
                        jl = []
                        for j in range(n):
                            if vals[(j * M + m) * K + k] > 0.5 and not seen[j]:
                                jl.append(j)
                                seen[j] = True
                        if jl:
                            sz = 0; mr = 0; mp = 0
                            for j in jl:
                                sz += s[j]
                                if r[j] > mr: mr = r[j]
                                if p[j] > mp: mp = p[j]
                            machines2[m].append({"jobs": jl, "size": sz, "mr": mr,
                                                 "mp": mp, "start": 0, "finish": 0})
                for j in range(n):
                    if not seen[j]:
                        bestm = 0; bestc = None
                        for m in range(M):
                            fin = machines2[m][-1]["finish"] if machines2[m] else 0
                            st = fin if fin > r[j] else r[j]
                            c = st + p[j]
                            if bestc is None or c < bestc:
                                bestc = c; bestm = m
                        machines2[bestm].append({"jobs": [j], "size": s[j], "mr": r[j],
                                                 "mp": p[j], "start": 0, "finish": 0})
                for mb in machines2:
                    schedule_machine(mb)
                return machines2, total_obj(machines2)

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = model.cbGetSolution(x_flat)
                        machines2, o2 = extract_from(vals)
                        register_incumbent(o2, machines2)
                    except Exception:
                        pass

            tl = deadline - time.time() - margin
            if tl <= 1.0:
                return
            model.Params.TimeLimit = tl
            model.optimize(cb)

            if model.SolCount > 0:
                try:
                    vals = [v.X for v in x_flat]
                    machines2, o2 = extract_from(vals)
                    register_incumbent(o2, machines2)
                except Exception:
                    pass
        except Exception:
            return

    # ----------------- fallbacks -----------------
    def trivial_solution():
        machines = [[] for _ in range(M)]
        finishes = [0] * M
        for j in sorted(range(n), key=lambda j: (r[j], p[j])):
            bestm = 0; bestc = None
            for m in range(M):
                st = finishes[m] if finishes[m] > r[j] else r[j]
                c = st + p[j]
                if bestc is None or c < bestc:
                    bestc = c; bestm = m
            st = finishes[bestm] if finishes[bestm] > r[j] else r[j]
            machines[bestm].append({"jobs": [j], "size": s[j], "mr": r[j], "mp": p[j],
                                    "start": st, "finish": st + p[j]})
            finishes[bestm] = st + p[j]
        return machines

    def verify(machines):
        seen = 0
        for mb in machines:
            prev = 0
            for b in mb:
                jl = b["jobs"]
                if not jl:
                    continue
                sz = 0; mr = 0; mp = 0
                for j in jl:
                    sz += s[j]
                    if r[j] > mr: mr = r[j]
                    if p[j] > mp: mp = p[j]
                if sz > Cap:
                    return False
                if b["start"] < prev or b["start"] < mr:
                    return False
                if b["mp"] != mp:
                    return False
                if b["finish"] != b["start"] + mp:
                    return False
                prev = b["finish"]
                seen += len(jl)
        return seen == n

    # ----------------- main flow -----------------
    try:
        idx = list(range(n))
        orders = [
            sorted(idx, key=lambda j: (r[j], p[j])),
            sorted(idx, key=lambda j: (r[j], -s[j], p[j])),
            sorted(idx, key=lambda j: (r[j] + p[j], p[j])),
            sorted(idx, key=lambda j: (p[j], r[j])),
            sorted(idx, key=lambda j: (r[j], -p[j], -s[j])),
            sorted(idx, key=lambda j: (r[j], p[j], -s[j])),
        ]
        starts = []
        for order in orders:
            for theta in (1.0, 0.9, 1.1):
                mach = greedy(order, theta)
                starts.append((total_obj(mach), mach))
        starts.sort(key=lambda t: t[0])
        best_machines = starts[0][1]
        register_incumbent(starts[0][0], best_machines)

        mip_ok = (n * n * M) <= 9000
        if mip_ok:
            ls_budget = min(max(2.0, 0.12 * time_limit), 25.0)
            remaining = deadline - time.time() - margin
            ls_budget = min(ls_budget, max(0.5, remaining - 8.0))
        else:
            ls_budget = max(0.5, time_limit - margin - 0.2)
        local_search(best_machines, time.time() + ls_budget)

        ran_mip = False
        if mip_ok:
            remaining = deadline - time.time() - margin
            if remaining >= 12.0:
                ran_mip = True
                run_mip(state["snap"])
        if not ran_mip:
            cont_deadline = deadline - margin
            if time.time() < cont_deadline:
                mach = snapshot(state["snap"])
                local_search(mach, cont_deadline)
    except Exception:
        pass

    final_machines = state["snap"]
    if final_machines is None or not verify(final_machines):
        final_machines = trivial_solution()
    out = build_output(final_machines)
    if logger is not None and state["obj"] is None:
        try:
            logger.log_solution(out["objective_value"], out)
        except Exception:
            pass
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    solve()