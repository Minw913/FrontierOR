import json
import argparse
import time
import math
import random
import sys

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.perf_counter()
    deadline = start_time + max(1.0, args.time_limit - 1.0)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    random.seed(0)

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    jobs = inst["jobs"]
    n = inst["parameters"]["n"]
    m = inst["parameters"]["m"]
    horizon = inst.get("time_horizon", None)

    IDS = [j["id"] for j in jobs]
    P = [int(j["processing_time"]) for j in jobs]
    W = [int(j["weight"]) for j in jobs]
    D = [int(j["due_date"]) for j in jobs]
    n = len(jobs)

    if horizon is None:
        total_p = sum(P)
        pmax = max(P) if P else 0
        horizon = (total_p - pmax) // max(1, m) + pmax

    # ----- trivial edge case -----
    if n == 0:
        sol = {"objective_value": 0.0, "schedule": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        if logger:
            logger.log_solution(0.0, sol)
        return

    def seq_cost(seq):
        t = 0
        c = 0
        for j in seq:
            t += P[j]
            lt = t - D[j]
            if lt > 0:
                c += W[j] * lt
        return c

    def total_cost(seqs):
        return sum(seq_cost(s) for s in seqs)

    def best_insertion(seq, px, wx, dx):
        """Best position to insert a job (px, wx, dx) into seq.
        Returns (new_total_cost_of_seq, position)."""
        L = len(seq)
        pref = [0] * (L + 1)
        comp = [0] * (L + 1)
        t = 0
        for i in range(L):
            j = seq[i]
            t += P[j]
            comp[i + 1] = t
            lt = t - D[j]
            pref[i + 1] = pref[i] + (W[j] * lt if lt > 0 else 0)
        suf = [0] * (L + 1)
        for i in range(L - 1, -1, -1):
            j = seq[i]
            lt = comp[i + 1] + px - D[j]
            suf[i] = suf[i + 1] + (W[j] * lt if lt > 0 else 0)
        best_c = None
        best_pos = 0
        for i in range(L + 1):
            lt = comp[i] + px - dx
            c = pref[i] + (wx * lt if lt > 0 else 0) + suf[i]
            if best_c is None or c < best_c:
                best_c = c
                best_pos = i
        return best_c, best_pos

    # ----- construction: ATC dispatching (list scheduling on min-available machine) -----
    pbar = sum(P) / n

    def atc_construct(kpar):
        avail = [0] * m
        remaining = list(range(n))
        seqs = [[] for _ in range(m)]
        while remaining:
            a = min(range(m), key=lambda i: avail[i])
            t = avail[a]
            best_j = -1
            best_pr = -1.0
            best_pos = -1
            for pos, j in enumerate(remaining):
                slack = D[j] - P[j] - t
                if slack < 0:
                    slack = 0
                pr = (W[j] / P[j]) * math.exp(-slack / (kpar * pbar))
                if pr > best_pr:
                    best_pr = pr
                    best_j = j
                    best_pos = pos
            remaining.pop(best_pos)
            seqs[a].append(best_j)
            avail[a] = t + P[best_j]
        return seqs

    def edd_construct():
        avail = [0] * m
        order = sorted(range(n), key=lambda j: (D[j], -W[j] / P[j]))
        seqs = [[] for _ in range(m)]
        for j in order:
            a = min(range(m), key=lambda i: avail[i])
            seqs[a].append(j)
            avail[a] += P[j]
        return seqs

    def wspt_construct():
        avail = [0] * m
        order = sorted(range(n), key=lambda j: (P[j] / W[j], D[j]))
        seqs = [[] for _ in range(m)]
        for j in order:
            a = min(range(m), key=lambda i: avail[i])
            seqs[a].append(j)
            avail[a] += P[j]
        return seqs

    candidates = []
    for k in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        if time.perf_counter() > deadline:
            break
        candidates.append(atc_construct(k))
    candidates.append(edd_construct())
    candidates.append(wspt_construct())

    best_init = None
    best_init_cost = None
    for s in candidates:
        c = total_cost(s)
        if best_init_cost is None or c < best_init_cost:
            best_init_cost = c
            best_init = s

    seqs = [list(s) for s in best_init]

    def build_solution(seqs_in):
        schedule = []
        total = 0
        for mach in range(m):
            t = 0
            prev = 0
            for j in seqs_in[mach]:
                s = t
                t += P[j]
                lt = t - D[j]
                wt = W[j] * lt if lt > 0 else 0
                total += wt
                schedule.append({
                    "machine_id": mach,
                    "prev_job": prev,
                    "job": IDS[j],
                    "start_time": s,
                    "completion_time": t,
                    "weighted_tardiness": wt,
                })
                prev = IDS[j]
        schedule.sort(key=lambda e: e["job"])
        return {"objective_value": float(total), "schedule": schedule}

    best_seqs = [list(s) for s in seqs]
    best_cost = total_cost(best_seqs)
    if logger:
        logger.log_solution(best_cost, build_solution(best_seqs))

    # ----- local search -----
    def intra_opt(seq, cost):
        """Reinsertion neighborhood within one machine."""
        improved_any = False
        improved = True
        while improved:
            improved = False
            L = len(seq)
            for idx in range(L):
                if time.perf_counter() > deadline:
                    return seq, cost, improved_any
                x = seq[idx]
                rem = seq[:idx] + seq[idx + 1:]
                newc, pos = best_insertion(rem, P[x], W[x], D[x])
                if newc < cost - 1e-9:
                    seq = rem[:pos] + [x] + rem[pos:]
                    cost = newc
                    improved = True
                    improved_any = True
                    break
        return seq, cost, improved_any

    def cross_opt(seqs_l, costs, loads):
        """Move jobs between machines (best target per job, first improvement applied)."""
        improved_any = False
        improved = True
        while improved:
            improved = False
            for a in range(m):
                idx = 0
                while idx < len(seqs_l[a]):
                    if time.perf_counter() > deadline:
                        return improved_any
                    x = seqs_l[a][idx]
                    remseq = seqs_l[a][:idx] + seqs_l[a][idx + 1:]
                    remcost = seq_cost(remseq)
                    gain = costs[a] - remcost
                    best_net = -1e-9
                    best_b = -1
                    best_bc = None
                    best_pos = None
                    for b in range(m):
                        if b == a:
                            continue
                        if loads[b] + P[x] > horizon:
                            continue
                        nc, pos = best_insertion(seqs_l[b], P[x], W[x], D[x])
                        inc = nc - costs[b]
                        net = gain - inc
                        if net > best_net:
                            best_net = net
                            best_b = b
                            best_bc = nc
                            best_pos = pos
                    if best_b >= 0:
                        b = best_b
                        seqs_l[a] = remseq
                        loads[a] -= P[x]
                        costs[a] = remcost
                        seqs_l[b] = seqs_l[b][:best_pos] + [x] + seqs_l[b][best_pos:]
                        loads[b] += P[x]
                        costs[b] = best_bc
                        improved = True
                        improved_any = True
                    else:
                        idx += 1
        return improved_any

    def local_search(seqs_l):
        costs = [seq_cost(s) for s in seqs_l]
        loads = [sum(P[j] for j in s) for s in seqs_l]
        while True:
            if time.perf_counter() > deadline:
                break
            imp = False
            for a in range(m):
                seqs_l[a], costs[a], ia = intra_opt(seqs_l[a], costs[a])
                if ia:
                    imp = True
            if time.perf_counter() > deadline:
                break
            if m > 1:
                if cross_opt(seqs_l, costs, loads):
                    imp = True
            if not imp:
                break
        return seqs_l, sum(costs)

    seqs, cur_cost = local_search(seqs)
    if cur_cost < best_cost - 1e-9:
        best_cost = cur_cost
        best_seqs = [list(s) for s in seqs]
        if logger:
            logger.log_solution(best_cost, build_solution(best_seqs))

    # ----- iterated local search -----
    def perturb(seqs_l, strength):
        loads = [sum(P[j] for j in s) for s in seqs_l]
        for _ in range(strength):
            nonempty = [a for a in range(m) if seqs_l[a]]
            if not nonempty:
                return
            a = random.choice(nonempty)
            idx = random.randrange(len(seqs_l[a]))
            x = seqs_l[a].pop(idx)
            loads[a] -= P[x]
            feas = [b for b in range(m) if loads[b] + P[x] <= horizon]
            if not feas:
                feas = [a]
            b = random.choice(feas)
            pos = random.randrange(len(seqs_l[b]) + 1)
            seqs_l[b].insert(pos, x)
            loads[b] += P[x]

    while time.perf_counter() < deadline:
        cur = [list(s) for s in best_seqs]
        strength = random.randint(2, max(2, min(6, n // 4 if n >= 8 else 2)))
        perturb(cur, strength)
        cur, cur_cost = local_search(cur)
        if cur_cost < best_cost - 1e-9:
            best_cost = cur_cost
            best_seqs = [list(s) for s in cur]
            if logger:
                logger.log_solution(best_cost, build_solution(best_seqs))
        if best_cost <= 0:
            break

    sol = build_solution(best_seqs)
    if logger:
        logger.log_solution(sol["objective_value"], sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()