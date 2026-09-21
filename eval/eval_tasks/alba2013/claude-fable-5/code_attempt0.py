import argparse
import json
import random
import time
from math import exp

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(1, args.time_limit) - 0.8

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    par = inst["parameters"]
    s = int(par["num_stacks_s"])
    l = int(par["stack_height_l"])
    Dp = inst["pickup_region"]["distance_matrix"]
    pdep = inst["pickup_region"]["depot_index"]
    Dd = inst["delivery_region"]["distance_matrix"]
    ddep = inst["delivery_region"]["depot_index"]
    reqs = inst["requests"]
    n = len(reqs)
    pv = [r["pickup_vertex"] for r in reqs]
    dv = [r["delivery_vertex"] for r in reqs]

    def write_solution(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    if n == 0:
        sol = {"objective_value": 0.0,
               "pickup_tour": [pdep, pdep],
               "delivery_tour": [ddep, ddep]}
        if logger:
            logger.log_solution(0.0, sol)
        write_solution(sol)
        return

    # ---------- helpers ----------
    def tour_cost(order, nd, D, dep):
        c = 0
        prev = dep
        for r in order:
            v = nd[r]
            c += D[prev][v]
            prev = v
        return c + D[prev][dep]

    def is_sym(D):
        m = len(D)
        for i in range(m):
            Di = D[i]
            for j in range(i + 1, m):
                if Di[j] != D[j][i]:
                    return False
        return True

    sym_p = is_sym(Dp)
    sym_d = is_sym(Dd)

    def check_mode(pick, dpos, worst):
        tops = [n] * s
        sizes = [0] * s
        for r in pick:
            d = dpos[r]
            best = -1
            bt = -1 if worst else n + 1
            for k in range(s):
                if sizes[k] < l:
                    t = tops[k]
                    if t > d:
                        if worst:
                            if t > bt:
                                bt = t
                                best = k
                        else:
                            if t < bt:
                                bt = t
                                best = k
            if best < 0:
                return False
            tops[best] = d
            sizes[best] += 1
        return True

    def check(pick, dpos):
        if check_mode(pick, dpos, False):
            return True
        return check_mode(pick, dpos, True)

    def swap_delta(order, i, j, nd, D, dep):
        # positions i < j
        A = nd[order[i]]
        B = nd[order[j]]
        p = dep if i == 0 else nd[order[i - 1]]
        q = dep if j == len(order) - 1 else nd[order[j + 1]]
        if j == i + 1:
            return (D[p][B] + D[B][A] + D[A][q]) - (D[p][A] + D[A][B] + D[B][q])
        ni = nd[order[i + 1]]
        pj = nd[order[j - 1]]
        return (D[p][B] + D[B][ni] + D[pj][A] + D[A][q]) - \
               (D[p][A] + D[A][ni] + D[pj][B] + D[B][q])

    def two_opt_delta(order, i, j, nd, D, dep):
        a = dep if i == 0 else nd[order[i - 1]]
        b = nd[order[i]]
        c = nd[order[j]]
        d = dep if j == len(order) - 1 else nd[order[j + 1]]
        return D[a][c] + D[b][d] - D[a][b] - D[c][d]

    # ---------- initial solution ----------
    unv = set(range(n))
    cur = pdep
    pick = []
    while unv:
        best_r = min(unv, key=lambda r: Dp[cur][pv[r]])
        pick.append(best_r)
        unv.discard(best_r)
        cur = pv[best_r]

    if sym_p:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = pdep if i == 0 else pv[pick[i - 1]]
                    b = pv[pick[i]]
                    c2 = pv[pick[j]]
                    d2 = pdep if j == n - 1 else pv[pick[j + 1]]
                    if Dp[a][c2] + Dp[b][d2] < Dp[a][b] + Dp[c2][d2]:
                        pick[i:j + 1] = pick[i:j + 1][::-1]
                        improved = True

    # assign to stacks balanced, in pickup order
    sizes = [0] * s
    stacks = [[] for _ in range(s)]
    for r in pick:
        k = min(range(s), key=lambda q: (sizes[q] if sizes[q] < l else 10 ** 9))
        stacks[k].append(r)
        sizes[k] += 1

    # greedy delivery from stack tops (LIFO feasible by construction)
    ptr = [len(st) - 1 for st in stacks]
    cur = ddep
    dl = []
    for _ in range(n):
        cand = [(Dd[cur][dv[stacks[k][ptr[k]]]], k) for k in range(s) if ptr[k] >= 0]
        _, k = min(cand)
        r = stacks[k][ptr[k]]
        ptr[k] -= 1
        dl.append(r)
        cur = dv[r]

    # ---------- polish (deterministic local search) ----------
    def polish(pick0, dl0, dl_deadline):
        pk = pick0[:]
        de = dl0[:]
        ppos = [0] * n
        dpos = [0] * n
        for i, r in enumerate(pk):
            ppos[r] = i
        for i, r in enumerate(de):
            dpos[r] = i
        pc = tour_cost(pk, pv, Dp, pdep)
        dc = tour_cost(de, dv, Dd, ddep)
        improved = True
        while improved:
            improved = False
            if time.time() >= dl_deadline:
                break
            # coupled swaps (always feasibility preserving)
            for a in range(n - 1):
                if time.time() >= dl_deadline:
                    break
                for b in range(a + 1, n):
                    ia, ib = ppos[a], ppos[b]
                    if ia > ib:
                        ia, ib = ib, ia
                    d1 = swap_delta(pk, ia, ib, pv, Dp, pdep)
                    ja, jb = dpos[a], dpos[b]
                    if ja > jb:
                        ja, jb = jb, ja
                    d2 = swap_delta(de, ja, jb, dv, Dd, ddep)
                    if d1 + d2 < 0:
                        pk[ppos[a]], pk[ppos[b]] = pk[ppos[b]], pk[ppos[a]]
                        ppos[a], ppos[b] = ppos[b], ppos[a]
                        de[dpos[a]], de[dpos[b]] = de[dpos[b]], de[dpos[a]]
                        dpos[a], dpos[b] = dpos[b], dpos[a]
                        pc += d1
                        dc += d2
                        improved = True
            if sym_p:
                for i in range(n - 1):
                    if time.time() >= dl_deadline:
                        break
                    for j in range(i + 1, n):
                        d1 = two_opt_delta(pk, i, j, pv, Dp, pdep)
                        if d1 < 0:
                            newp = pk[:i] + pk[i:j + 1][::-1] + pk[j + 1:]
                            if check(newp, dpos):
                                pk = newp
                                pc += d1
                                improved = True
                                for idx in range(i, j + 1):
                                    ppos[pk[idx]] = idx
            if sym_d:
                for i in range(n - 1):
                    if time.time() >= dl_deadline:
                        break
                    for j in range(i + 1, n):
                        d1 = two_opt_delta(de, i, j, dv, Dd, ddep)
                        if d1 < 0:
                            newd = de[:i] + de[i:j + 1][::-1] + de[j + 1:]
                            ndpos = dpos[:]
                            for idx in range(i, j + 1):
                                ndpos[newd[idx]] = idx
                            if check(pk, ndpos):
                                de = newd
                                dpos = ndpos
                                dc += d1
                                improved = True
        return pk, de, pc, dc

    pick, dl, pc0, dc0 = polish(pick, dl, min(deadline, time.time() + 5.0))

    best_pick = pick[:]
    best_dl = dl[:]
    best_cost = pc0 + dc0

    def make_sol():
        return {"objective_value": float(best_cost),
                "pickup_tour": [pdep] + [pv[r] for r in best_pick] + [pdep],
                "delivery_tour": [ddep] + [dv[r] for r in best_dl] + [ddep]}

    if logger:
        logger.log_solution(float(best_cost), make_sol())

    # ---------- simulated annealing ----------
    if n >= 2:
        tot = 0
        cnt = 0
        for D in (Dp, Dd):
            m = len(D)
            for i in range(m):
                Di = D[i]
                for j in range(m):
                    if i != j:
                        tot += Di[j]
                        cnt += 1
        avg = tot / max(1, cnt)
        T0 = max(1.0, 0.4 * avg)
        Tend = max(0.05, 0.008 * avg)
        ratio = Tend / T0

        rng = random.Random(0)
        total_sa = deadline - time.time()
        round_len = max(2.0, total_sa / 8.0)

        while time.time() < deadline:
            round_start = time.time()
            round_end = min(deadline, round_start + round_len)
            cur_pick = best_pick[:]
            cur_dl = best_dl[:]
            ppos = [0] * n
            dpos = [0] * n
            for i, r in enumerate(cur_pick):
                ppos[r] = i
            for i, r in enumerate(cur_dl):
                dpos[r] = i
            pc = tour_cost(cur_pick, pv, Dp, pdep)
            dc = tour_cost(cur_dl, dv, Dd, ddep)
            T = T0
            it = 0
            while True:
                it += 1
                if it & 63 == 0:
                    now = time.time()
                    if now >= round_end:
                        break
                    frac = (now - round_start) / round_len
                    if frac > 1.0:
                        frac = 1.0
                    T = T0 * (ratio ** frac)
                u = rng.random()
                if u < 0.25:
                    # coupled swap: exchange (pickup-slot, delivery-slot) of two requests
                    a = rng.randrange(n)
                    b = rng.randrange(n)
                    if a == b:
                        continue
                    ia, ib = ppos[a], ppos[b]
                    if ia > ib:
                        ia, ib = ib, ia
                    d1 = swap_delta(cur_pick, ia, ib, pv, Dp, pdep)
                    ja, jb = dpos[a], dpos[b]
                    if ja > jb:
                        ja, jb = jb, ja
                    d2 = swap_delta(cur_dl, ja, jb, dv, Dd, ddep)
                    delta = d1 + d2
                    if delta <= 0 or rng.random() < exp(-delta / T):
                        cur_pick[ppos[a]], cur_pick[ppos[b]] = cur_pick[ppos[b]], cur_pick[ppos[a]]
                        ppos[a], ppos[b] = ppos[b], ppos[a]
                        cur_dl[dpos[a]], cur_dl[dpos[b]] = cur_dl[dpos[b]], cur_dl[dpos[a]]
                        dpos[a], dpos[b] = dpos[b], dpos[a]
                        pc += d1
                        dc += d2
                        tot_c = pc + dc
                        if tot_c < best_cost:
                            best_cost = tot_c
                            best_pick = cur_pick[:]
                            best_dl = cur_dl[:]
                            if logger:
                                logger.log_solution(float(best_cost), make_sol())
                else:
                    on_pickup = rng.random() < 0.5
                    if on_pickup:
                        order, nd, D, dep, c0 = cur_pick, pv, Dp, pdep, pc
                    else:
                        order, nd, D, dep, c0 = cur_dl, dv, Dd, ddep, dc
                    mv = rng.random()
                    if mv < 0.45:
                        i = rng.randrange(n - 1)
                        j = rng.randrange(i + 1, n)
                        new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    elif mv < 0.8:
                        i = rng.randrange(n)
                        j = rng.randrange(n)
                        if i == j:
                            continue
                        x = order[i]
                        rem = order[:i] + order[i + 1:]
                        new = rem[:j] + [x] + rem[j:]
                    else:
                        i = rng.randrange(n)
                        j = rng.randrange(n)
                        if i == j:
                            continue
                        new = order[:]
                        new[i], new[j] = new[j], new[i]
                    newc = tour_cost(new, nd, D, dep)
                    delta = newc - c0
                    if delta <= 0 or rng.random() < exp(-delta / T):
                        if on_pickup:
                            if check(new, dpos):
                                cur_pick = new
                                pc = newc
                                for idx, r in enumerate(new):
                                    ppos[r] = idx
                            else:
                                continue
                        else:
                            ndpos = [0] * n
                            for idx, r in enumerate(new):
                                ndpos[r] = idx
                            if check(cur_pick, ndpos):
                                cur_dl = new
                                dpos = ndpos
                                dc = newc
                            else:
                                continue
                        tot_c = pc + dc
                        if tot_c < best_cost:
                            best_cost = tot_c
                            best_pick = cur_pick[:]
                            best_dl = cur_dl[:]
                            if logger:
                                logger.log_solution(float(best_cost), make_sol())
            # polish best at end of round
            if time.time() < deadline:
                bp, bd, bpc, bdc = polish(best_pick, best_dl, deadline)
                if bpc + bdc < best_cost:
                    best_pick, best_dl, best_cost = bp, bd, bpc + bdc
                    if logger:
                        logger.log_solution(float(best_cost), make_sol())

    # exact recompute and final output
    best_cost = tour_cost(best_pick, pv, Dp, pdep) + tour_cost(best_dl, dv, Dd, ddep)
    sol = make_sol()
    if logger:
        logger.log_solution(float(best_cost), sol)
    write_solution(sol)


if __name__ == "__main__":
    main()