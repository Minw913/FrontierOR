import json
import argparse
import time
import math
import random

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit) - 0.7

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    random.seed(0)

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = inst["n"]
    m = inst["m"]
    jobs = inst["jobs"]

    ids = [jb["id"] for jb in jobs]
    p = [jb["p"] for jb in jobs]
    d = [jb["d"] for jb in jobs]
    w = [jb["w"] for jb in jobs]

    if n == 0:
        with open(args.solution_path, "w") as f:
            json.dump({"objective_value": 0.0, "schedule": {}}, f)
        return

    m = max(1, m)

    # ---------- helpers ----------
    def seq_cost(seq):
        t = 0
        c = 0
        for j in seq:
            t += p[j]
            lt = t - d[j]
            if lt > 0:
                c += w[j] * lt
        return c

    def total_cost(machines):
        return sum(seq_cost(s) for s in machines)

    def best_insertion(seq, j):
        """Return (cost_of_seq_after_best_insert, best_position)."""
        pj = p[j]
        wj = w[j]
        dj = d[j]
        L = len(seq)
        pref_cost = [0] * (L + 1)
        pref_C = [0] * (L + 1)
        c = 0
        cost = 0
        for i in range(L):
            x = seq[i]
            c += p[x]
            lt = c - d[x]
            if lt > 0:
                cost += w[x] * lt
            pref_C[i + 1] = c
            pref_cost[i + 1] = cost
        S = [0] * (L + 1)
        for i in range(L - 1, -1, -1):
            x = seq[i]
            lt = pref_C[i + 1] + pj - d[x]
            S[i] = S[i + 1] + (w[x] * lt if lt > 0 else 0)
        best = None
        bq = 0
        for q in range(L + 1):
            lt = pref_C[q] + pj - dj
            cand = pref_cost[q] + (wj * lt if lt > 0 else 0) + S[q]
            if best is None or cand < best - 1e-12:
                best = cand
                bq = q
        return best, bq

    def build_solution(machines, obj):
        schedule = {}
        for seq in machines:
            t = 0
            for j in seq:
                schedule[str(ids[j])] = t
                t += p[j]
        return {"objective_value": float(obj), "schedule": schedule}

    # ---------- initial solution: ATC dispatching ----------
    def atc_schedule(k):
        machines = [[] for _ in range(m)]
        mt = [0] * m
        remaining = list(range(n))
        rem_p_sum = sum(p)
        for _ in range(n):
            mi = min(range(m), key=lambda i: mt[i])
            t = mt[mi]
            pavg = rem_p_sum / max(1, len(remaining))
            best_j = -1
            best_pr = -1.0
            best_pos = -1
            for pos, j in enumerate(remaining):
                slack = d[j] - t - p[j]
                if slack < 0:
                    slack = 0
                pr = (w[j] / p[j]) * math.exp(-slack / (k * pavg))
                if pr > best_pr:
                    best_pr = pr
                    best_j = j
                    best_pos = pos
            machines[mi].append(best_j)
            mt[mi] += p[best_j]
            rem_p_sum -= p[best_j]
            remaining.pop(best_pos)
        return machines

    best_machines = None
    best_total = None
    for k in (0.5, 1.0, 2.0, 3.0, 5.0):
        if time.time() > deadline:
            break
        mach = atc_schedule(k)
        c = total_cost(mach)
        if best_total is None or c < best_total:
            best_total = c
            best_machines = mach

    if best_machines is None:
        # trivial fallback
        best_machines = [[] for _ in range(m)]
        for i in range(n):
            best_machines[i % m].append(i)
        best_total = total_cost(best_machines)

    machines = [list(s) for s in best_machines]
    mcost = [seq_cost(s) for s in machines]
    total = sum(mcost)
    best_total = total
    best_snap = [list(s) for s in machines]

    if logger:
        logger.log_solution(float(best_total), build_solution(best_snap, best_total))

    def flush_best():
        if logger:
            logger.log_solution(float(best_total), build_solution(best_snap, best_total))

    # ---------- local descent: best re-insertion of each job ----------
    def descent(machines, mcost, dl):
        nonlocal best_total, best_snap
        total_loc = sum(mcost)
        loc = {}
        for mi, seq in enumerate(machines):
            for j in seq:
                loc[j] = mi
        order = list(range(n))
        improved = True
        while improved and time.time() < dl:
            improved = False
            random.shuffle(order)
            cnt = 0
            for j in order:
                cnt += 1
                if (cnt & 31) == 0 and time.time() > dl:
                    break
                mi = loc[j]
                seq = machines[mi]
                idx = seq.index(j)
                seq_wo = seq[:idx] + seq[idx + 1:]
                cost_wo = seq_cost(seq_wo)
                best_delta = -1e-9
                best_move = None
                for mk in range(m):
                    base = seq_wo if mk == mi else machines[mk]
                    newc, q = best_insertion(base, j)
                    if mk == mi:
                        delta = newc - mcost[mi]
                    else:
                        delta = (cost_wo - mcost[mi]) + (newc - mcost[mk])
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (mk, q, newc)
                if best_move is not None:
                    mk, q, newc = best_move
                    if mk == mi:
                        machines[mi] = seq_wo[:q] + [j] + seq_wo[q:]
                        mcost[mi] = newc
                    else:
                        machines[mi] = seq_wo
                        mcost[mi] = cost_wo
                        tgt = machines[mk]
                        machines[mk] = tgt[:q] + [j] + tgt[q:]
                        mcost[mk] = newc
                        loc[j] = mk
                    total_loc += best_delta
                    improved = True
            if total_loc < best_total - 1e-9:
                best_total = total_loc
                best_snap = [list(s) for s in machines]
                flush_best()
        return total_loc

    descent_budget = min(deadline, time.time() + 0.30 * max(1, args.time_limit))
    total = descent(machines, mcost, descent_budget)

    if best_total <= 1e-12:
        with open(args.solution_path, "w") as f:
            json.dump(build_solution(best_snap, best_total), f)
        return

    # ---------- simulated annealing ----------
    # estimate temperature from sampled move deltas
    sample_deltas = []
    for _ in range(60):
        mi = random.randrange(m)
        if not machines[mi]:
            continue
        mk = random.randrange(m)
        if not machines[mk]:
            continue
        a = random.randrange(len(machines[mi]))
        b = random.randrange(len(machines[mk]))
        if mi == mk and a == b:
            continue
        s1 = list(machines[mi])
        if mi == mk:
            s1[a], s1[b] = s1[b], s1[a]
            dlt = seq_cost(s1) - mcost[mi]
        else:
            s2 = list(machines[mk])
            s1[a], s2[b] = s2[b], s1[a]
            dlt = (seq_cost(s1) - mcost[mi]) + (seq_cost(s2) - mcost[mk])
        if abs(dlt) > 1e-9:
            sample_deltas.append(abs(dlt))
    T0 = max(1.0, (sum(sample_deltas) / len(sample_deltas)) if sample_deltas else 1.0)
    Tend = max(1e-3, T0 * 1e-3)

    sa_start = time.time()
    sa_span = max(0.1, deadline - sa_start)
    it = 0
    T = T0
    nonempty_tries = 4 * m + 8

    while True:
        it += 1
        if (it & 127) == 0:
            now = time.time()
            if now > deadline:
                break
            frac = (now - sa_start) / sa_span
            if frac > 1.0:
                frac = 1.0
            T = T0 * ((Tend / T0) ** frac)

        r = random.random()
        if r < 0.55:
            # relocate a random job
            mi = -1
            for _ in range(nonempty_tries):
                cand = random.randrange(m)
                if machines[cand]:
                    mi = cand
                    break
            if mi < 0:
                continue
            seq = machines[mi]
            idx = random.randrange(len(seq))
            j = seq[idx]
            seq_wo = seq[:idx] + seq[idx + 1:]
            cost_wo = seq_cost(seq_wo)
            mk = random.randrange(m)
            base = seq_wo if mk == mi else machines[mk]
            if random.random() < 0.7:
                newc, q = best_insertion(base, j)
            else:
                q = random.randrange(len(base) + 1)
                newseq = base[:q] + [j] + base[q:]
                newc = seq_cost(newseq)
            if mk == mi:
                delta = newc - mcost[mi]
            else:
                delta = (cost_wo - mcost[mi]) + (newc - mcost[mk])
            if delta < 1e-12 or random.random() < math.exp(-delta / T):
                if mk == mi:
                    machines[mi] = seq_wo[:q] + [j] + seq_wo[q:]
                    mcost[mi] = newc
                else:
                    machines[mi] = seq_wo
                    mcost[mi] = cost_wo
                    tgt = machines[mk]
                    machines[mk] = tgt[:q] + [j] + tgt[q:]
                    mcost[mk] = newc
                total += delta
                if total < best_total - 1e-9:
                    best_total = total
                    best_snap = [list(s) for s in machines]
                    flush_best()
        else:
            # swap two jobs
            mi = mk = -1
            for _ in range(nonempty_tries):
                cand = random.randrange(m)
                if machines[cand]:
                    mi = cand
                    break
            for _ in range(nonempty_tries):
                cand = random.randrange(m)
                if machines[cand]:
                    mk = cand
                    break
            if mi < 0 or mk < 0:
                continue
            if mi == mk and len(machines[mi]) < 2:
                continue
            a = random.randrange(len(machines[mi]))
            b = random.randrange(len(machines[mk]))
            if mi == mk:
                if a == b:
                    continue
                s1 = list(machines[mi])
                s1[a], s1[b] = s1[b], s1[a]
                c1 = seq_cost(s1)
                delta = c1 - mcost[mi]
                if delta < 1e-12 or random.random() < math.exp(-delta / T):
                    machines[mi] = s1
                    mcost[mi] = c1
                    total += delta
                    if total < best_total - 1e-9:
                        best_total = total
                        best_snap = [list(s) for s in machines]
                        flush_best()
            else:
                s1 = list(machines[mi])
                s2 = list(machines[mk])
                s1[a], s2[b] = s2[b], s1[a]
                c1 = seq_cost(s1)
                c2 = seq_cost(s2)
                delta = (c1 - mcost[mi]) + (c2 - mcost[mk])
                if delta < 1e-12 or random.random() < math.exp(-delta / T):
                    machines[mi] = s1
                    machines[mk] = s2
                    mcost[mi] = c1
                    mcost[mk] = c2
                    total += delta
                    if total < best_total - 1e-9:
                        best_total = total
                        best_snap = [list(s) for s in machines]
                        flush_best()

    # final polish on best solution if a sliver of time remains
    machines = [list(s) for s in best_snap]
    mcost = [seq_cost(s) for s in machines]
    tl = time.time() + 0.3
    if tl < start_time + max(1, args.time_limit) - 0.3:
        descent(machines, mcost, tl)

    sol = build_solution(best_snap, best_total)
    if logger:
        logger.log_solution(float(best_total), sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()