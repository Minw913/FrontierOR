import argparse
import json
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit) - 0.8
    random.seed(0)

    logger = None
    try:
        from solution_logger import SolutionLogger
        if args.log_path:
            logger = SolutionLogger(args.log_path, sense="minimize")
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    C = inst["distance_matrix"]
    Q = inst["vehicle_capacity"]
    K = inst["num_vehicles"]
    D = inst["depot"]["index"]
    Lc = inst["linehaul_customers"]
    Bc = inst["backhaul_customers"]
    L = [c["index"] for c in Lc]
    B = [c["index"] for c in Bc]
    dem = {}
    for c in Lc:
        dem[c["index"]] = c["demand"]
    for c in Bc:
        dem[c["index"]] = c["demand"]
    is_lh = set(L)
    allow_bh_only = (len(L) == 0)
    all_cust = L + B
    n = len(all_cust)

    lh = [[] for _ in range(K)]
    bh = [[] for _ in range(K)]
    lload = [0] * K
    bload = [0] * K
    rcost = [0.0] * K
    croute = {}

    def route_cost(lhl, bhl):
        s = 0
        prev = D
        for x in lhl:
            s += C[prev][x]
            prev = x
        for x in bhl:
            s += C[prev][x]
            prev = x
        s += C[prev][D]
        return s

    def recompute(k):
        rcost[k] = route_cost(lh[k], bh[k])

    def total():
        return sum(rcost)

    def best_ins_lh(k, cust):
        if lload[k] + dem[cust] > Q:
            return None
        seq = [D] + lh[k] + bh[k] + [D]
        m = len(lh[k])
        best = None
        for j in range(m + 1):
            dl = C[seq[j]][cust] + C[cust][seq[j + 1]] - C[seq[j]][seq[j + 1]]
            if best is None or dl < best[0]:
                best = (dl, j)
        return best

    def best_ins_bh(k, cust):
        if bload[k] + dem[cust] > Q:
            return None
        if not lh[k] and not allow_bh_only:
            return None
        seq = [D] + lh[k] + bh[k] + [D]
        m = len(lh[k])
        kb = len(bh[k])
        best = None
        for j in range(m, m + kb + 1):
            dl = C[seq[j]][cust] + C[cust][seq[j + 1]] - C[seq[j]][seq[j + 1]]
            if best is None or dl < best[0]:
                best = (dl, j - m)
        return best

    def rem_delta(k, cust):
        seq = [D] + lh[k] + bh[k] + [D]
        i = seq.index(cust, 1)
        return C[seq[i - 1]][seq[i + 1]] - C[seq[i - 1]][seq[i]] - C[seq[i]][seq[i + 1]]

    # ---------------- Construction ----------------
    order = sorted(L, key=lambda i: -(C[D][i] + C[i][D]))
    for cust in order:
        best = None
        for k in range(K):
            r = best_ins_lh(k, cust)
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], k, r[1])
        if best is None:
            k = min(range(K), key=lambda kk: lload[kk])
            lh[k].append(cust)
            lload[k] += dem[cust]
        else:
            _, k, pos = best
            lh[k].insert(pos, cust)
            lload[k] += dem[cust]

    def bh_fits():
        nb = sum(1 for k in range(K) if lh[k]) if not allow_bh_only else K
        if nb == 0:
            return len(B) == 0
        bins = [0] * nb
        for dv in sorted((dem[b] for b in B), reverse=True):
            placed = False
            for i in range(nb):
                if bins[i] + dv <= Q:
                    bins[i] += dv
                    placed = True
                    break
            if not placed:
                return False
        return True

    def try_open_route():
        empties = [k for k in range(K) if not lh[k] and not bh[k]]
        if not empties:
            return False
        e = empties[0]
        best = None
        for k in range(K):
            if len(lh[k]) >= 2:
                for cust in lh[k]:
                    dl = rem_delta(k, cust) + C[D][cust] + C[cust][D] - C[D][D]
                    if best is None or dl < best[0]:
                        best = (dl, k, cust)
        if best is None:
            return False
        _, k, cust = best
        lh[k].remove(cust)
        lload[k] -= dem[cust]
        lh[e].append(cust)
        lload[e] += dem[cust]
        return True

    guard = 0
    while B and not bh_fits() and guard < K + 5:
        if not try_open_route():
            break
        guard += 1

    order_b = sorted(B, key=lambda i: -dem[i])
    for cust in order_b:
        best = None
        for k in range(K):
            r = best_ins_bh(k, cust)
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], k, r[1])
        if best is None:
            if try_open_route():
                for k in range(K):
                    r = best_ins_bh(k, cust)
                    if r is not None and (best is None or r[0] < best[0]):
                        best = (r[0], k, r[1])
        if best is None:
            # last resort: force into route with linehauls and minimal backhaul load
            cands = [k for k in range(K) if lh[k]] or list(range(K))
            k = min(cands, key=lambda kk: bload[kk])
            bh[k].append(cust)
            bload[k] += dem[cust]
        else:
            _, k, pos = best
            bh[k].insert(pos, cust)
            bload[k] += dem[cust]

    for k in range(K):
        recompute(k)
        for c in lh[k]:
            croute[c] = k
        for c in bh[k]:
            croute[c] = k

    # ---------------- Local search ----------------
    def local_search():
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            # relocations
            for cust in all_cust:
                if time.time() > deadline:
                    return
                a = croute[cust]
                typ = cust in is_lh
                can_leave = True
                if typ and len(lh[a]) == 1 and bh[a]:
                    can_leave = False
                moved = False
                if can_leave:
                    rd = rem_delta(a, cust)
                    best = None
                    for b in range(K):
                        if b == a:
                            continue
                        if typ:
                            if lload[b] + dem[cust] > Q:
                                continue
                            r = best_ins_lh(b, cust)
                        else:
                            r = best_ins_bh(b, cust)
                        if r is not None and (best is None or r[0] < best[0]):
                            best = (r[0], b, r[1])
                    if best is not None and rd + best[0] < -1e-9:
                        _, b, pos = best
                        if typ:
                            lh[a].remove(cust)
                            lload[a] -= dem[cust]
                            lh[b].insert(pos, cust)
                            lload[b] += dem[cust]
                        else:
                            bh[a].remove(cust)
                            bload[a] -= dem[cust]
                            bh[b].insert(pos, cust)
                            bload[b] += dem[cust]
                        croute[cust] = b
                        recompute(a)
                        recompute(b)
                        improved = True
                        moved = True
                if not moved:
                    lst = lh[a] if typ else bh[a]
                    if len(lst) >= 2:
                        i = lst.index(cust)
                        tmp = lst[:i] + lst[i + 1:]
                        bestc = rcost[a]
                        bestlst = None
                        for j in range(len(tmp) + 1):
                            if j == i:
                                continue
                            cand = tmp[:j] + [cust] + tmp[j:]
                            c2 = route_cost(cand, bh[a]) if typ else route_cost(lh[a], cand)
                            if c2 < bestc - 1e-9:
                                bestc = c2
                                bestlst = cand
                        if bestlst is not None:
                            if typ:
                                lh[a] = bestlst
                            else:
                                bh[a] = bestlst
                            rcost[a] = bestc
                            improved = True
            # swaps (same type, different routes)
            for group, loads, lists in ((L, lload, lh), (B, bload, bh)):
                pn = {}
                for k in range(K):
                    seq = [D] + lh[k] + bh[k] + [D]
                    for i in range(1, len(seq) - 1):
                        pn[seq[i]] = (seq[i - 1], seq[i + 1])
                ng = len(group)
                for ii in range(ng):
                    if time.time() > deadline:
                        return
                    u = group[ii]
                    for jj in range(ii + 1, ng):
                        v = group[jj]
                        a = croute[u]
                        b = croute[v]
                        if a == b:
                            continue
                        if loads[a] - dem[u] + dem[v] > Q or loads[b] - dem[v] + dem[u] > Q:
                            continue
                        pu, nu = pn[u]
                        pv, nv = pn[v]
                        dl = (C[pu][v] + C[v][nu] - C[pu][u] - C[u][nu]
                              + C[pv][u] + C[u][nv] - C[pv][v] - C[v][nv])
                        if dl < -1e-9:
                            ia = lists[a].index(u)
                            ib = lists[b].index(v)
                            lists[a][ia] = v
                            lists[b][ib] = u
                            loads[a] += dem[v] - dem[u]
                            loads[b] += dem[u] - dem[v]
                            croute[u] = b
                            croute[v] = a
                            recompute(a)
                            recompute(b)
                            improved = True
                            for k in (a, b):
                                seq = [D] + lh[k] + bh[k] + [D]
                                for i in range(1, len(seq) - 1):
                                    pn[seq[i]] = (seq[i - 1], seq[i + 1])

    def snapshot():
        return ([x[:] for x in lh], [x[:] for x in bh], lload[:], bload[:],
                dict(croute), rcost[:])

    def restore(s):
        for k in range(K):
            lh[k] = s[0][k][:]
            bh[k] = s[1][k][:]
        lload[:] = s[2]
        bload[:] = s[3]
        croute.clear()
        croute.update(s[4])
        rcost[:] = s[5]

    def perturb():
        q = random.randint(2, max(2, min(10, max(2, n // 3))))
        removed = []
        cands = all_cust[:]
        random.shuffle(cands)
        for cust in cands:
            if len(removed) >= q:
                break
            a = croute[cust]
            if cust in is_lh:
                if len(lh[a]) == 1 and bh[a]:
                    continue
                lh[a].remove(cust)
                lload[a] -= dem[cust]
            else:
                bh[a].remove(cust)
                bload[a] -= dem[cust]
            recompute(a)
            del croute[cust]
            removed.append(cust)
        lhr = [c for c in removed if c in is_lh]
        bhr = [c for c in removed if c not in is_lh]
        random.shuffle(lhr)
        random.shuffle(bhr)
        for cust in lhr:
            best = None
            for k in range(K):
                r = best_ins_lh(k, cust)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], k, r[1])
            if best is None:
                return False
            _, k, pos = best
            lh[k].insert(pos, cust)
            lload[k] += dem[cust]
            croute[cust] = k
            recompute(k)
        for cust in bhr:
            best = None
            for k in range(K):
                r = best_ins_bh(k, cust)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], k, r[1])
            if best is None:
                return False
            _, k, pos = best
            bh[k].insert(pos, cust)
            bload[k] += dem[cust]
            croute[cust] = k
            recompute(k)
        return True

    def sol_dict(obj):
        return {"objective_value": float(obj),
                "routes": [[D] + lh[k] + bh[k] + [D] for k in range(K)]}

    # initial improvement
    local_search()
    best_cost = total()
    best = snapshot()
    if logger:
        logger.log_solution(best_cost, sol_dict(best_cost))

    cur_cost = best_cost
    no_imp = 0
    while n > 0 and time.time() < deadline:
        s = snapshot()
        ok = perturb()
        if not ok:
            restore(s)
            continue
        local_search()
        c = total()
        if c < cur_cost - 1e-9:
            cur_cost = c
        elif c < cur_cost * 1.02 and random.random() < 0.05:
            cur_cost = c
        else:
            restore(s)
            c = cur_cost
        if c < best_cost - 1e-9:
            best_cost = c
            best = snapshot()
            no_imp = 0
            if logger:
                logger.log_solution(best_cost, sol_dict(best_cost))
        else:
            no_imp += 1
            if no_imp >= 60:
                restore(best)
                cur_cost = best_cost
                no_imp = 0

    restore(best)
    for k in range(K):
        recompute(k)
    final_cost = total()
    out = sol_dict(final_cost)
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()