import argparse
import json
import time
import random
import itertools

from solution_logger import SolutionLogger

BIG = 1e7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    reserve = min(5.0, max(1.0, args.time_limit * 0.05))
    deadline = t_start + args.time_limit - reserve
    hard_deadline = t_start + args.time_limit - 0.3

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    H = data["vehicle"]["H"]
    W = data["vehicle"]["W"]
    Q = data["vehicle"]["Q"]
    K = data["vehicle"]["K"]
    cf = data["parameters"]["recourse_cost_cf"]
    n = data["parameters"]["n_customers"]
    D = data["distance_matrix"]
    HW = H * W

    det_items = {}   # cust -> list[(h,w,wt)]
    sto_items = {}   # cust -> list[ list[(h,w,wt,p)] ]
    exp_area = {}
    exp_wt = {}
    cust_ids = []
    for cu in data["customers"]:
        cid = cu["id"]
        cust_ids.append(cid)
        di, si = [], []
        ea = 0.0
        ew = 0.0
        for it in cu["items"]:
            reals = it["realizations"]
            for r in reals:
                ea += r["probability"] * r["height"] * r["width"]
                ew += r["probability"] * r["weight"]
            if len(reals) == 1:
                r = reals[0]
                di.append((r["height"], r["width"], r["weight"]))
            else:
                si.append([(r["height"], r["width"], r["weight"], r["probability"]) for r in reals])
        det_items[cid] = di
        sto_items[cid] = si
        exp_area[cid] = ea
        exp_wt[cid] = ew

    # ------------------ packing (skyline, LIFO by construction) ------------------
    KEYS = [lambda t: (-t[1], -t[0]),
            lambda t: (-t[0], -t[1]),
            lambda t: (-t[0] * t[1], -t[0])]

    def pack_once(seqs, key):
        # seqs: list (in reverse service order) of item lists [(h,w)]
        sky = [[0, W, 0]]  # segments [x0, x1, height]
        for items in seqs:
            for hw in sorted(items, key=key):
                h, w = hw
                besty = None
                bestx = 0
                besti = 0
                bestj = 0
                m = len(sky)
                for i in range(m):
                    x = sky[i][0]
                    if x + w > W:
                        break
                    xe = x + w
                    y = sky[i][2]
                    j = i
                    while sky[j][1] < xe:
                        j += 1
                        if sky[j][2] > y:
                            y = sky[j][2]
                    if y + h <= H and (besty is None or y < besty or (y == besty and x < bestx)):
                        besty = y
                        bestx = x
                        besti = i
                        bestj = j
                if besty is None:
                    return False
                xe = bestx + w
                ynew = besty + h
                tail = []
                if sky[bestj][1] > xe:
                    tail = [[xe, sky[bestj][1], sky[bestj][2]]]
                sky[besti:bestj + 1] = [[bestx, xe, ynew]] + tail
                k = 0
                while k + 1 < len(sky):
                    if sky[k][2] == sky[k + 1][2]:
                        sky[k][1] = sky[k + 1][1]
                        del sky[k + 1]
                    else:
                        k += 1
        return True

    def try_pack(seqs):
        for key in KEYS:
            if pack_once(seqs, key):
                return True
        return False

    # ------------------ scenario handling ------------------
    SCEN_ENUM = 120
    SCEN_SAMPLE = 100
    FINAL_ENUM = 3000
    FINAL_SAMPLE = 1500

    def route_seed(rt):
        s = 0
        for c in rt:
            s = (s * 1000003 + c + 1) % 2147483647
        return s

    def gen_scenarios(rt, enum_cap, nsamp):
        slots = []
        for c in rt:
            for reals in sto_items[c]:
                slots.append((c, reals))
        base_items = {c: [(h, w) for (h, w, wt) in det_items[c]] for c in rt}
        base_wt = {c: sum(wt for _, _, wt in det_items[c]) for c in rt}
        if not slots:
            return [(1.0, base_items, base_wt)]
        count = 1
        for _, r in slots:
            count *= len(r)
            if count > enum_cap:
                break
        combos = []
        if count <= enum_cap:
            for combo in itertools.product(*[range(len(r)) for _, r in slots]):
                p = 1.0
                for (slot, ci) in zip(slots, combo):
                    p *= slot[1][ci][3]
                if p > 0:
                    combos.append((combo, p))
        else:
            rng = random.Random(route_seed(rt))
            cnt = {}
            for _ in range(nsamp):
                combo = []
                for _, reals in slots:
                    u = rng.random()
                    acc = 0.0
                    pick = len(reals) - 1
                    for ri, r in enumerate(reals):
                        acc += r[3]
                        if u < acc:
                            pick = ri
                            break
                    combo.append(pick)
                combo = tuple(combo)
                cnt[combo] = cnt.get(combo, 0) + 1
            combos = [(cb, m / float(nsamp)) for cb, m in cnt.items()]
        out = []
        for combo, p in combos:
            items = {c: list(base_items[c]) for c in rt}
            wt = dict(base_wt)
            for (slot, ci) in zip(slots, combo):
                c, reals = slot
                r = reals[ci]
                items[c].append((r[0], r[1]))
                wt[c] += r[2]
            out.append((p, items, wt))
        return out

    def scenario_drops(rt, items, wt):
        def feas(seq):
            if not seq:
                return True
            tw = 0
            for c in seq:
                tw += wt[c]
            if tw > Q:
                return False
            return try_pack([items[c] for c in reversed(seq)])

        if feas(rt):
            return 0
        rem = list(rt)
        drops = 0
        while rem:
            order = sorted(rem, key=lambda c: -(sum(h * w for h, w in items[c]) / float(HW) + wt[c] / float(Q)))
            fixed = False
            for c in order:
                r2 = [x for x in rem if x != c]
                if feas(r2):
                    drops += 1
                    rem = r2
                    fixed = True
                    break
            if fixed:
                break
            rem = [x for x in rem if x != order[0]]
            drops += 1
        return drops

    def eval_route_generic(rt, enum_cap, nsamp, cache):
        v = cache.get(rt)
        if v is not None:
            return v
        ea = sum(exp_area[c] for c in rt)
        ew = sum(exp_wt[c] for c in rt)
        if ea > HW + 1e-9 or ew > Q + 1e-9:
            v = (False, BIG * (1.0 + max(ea / HW - 1.0, 0.0) + max(ew / float(Q) - 1.0, 0.0)))
            cache[rt] = v
            return v
        pen = 0.0
        psum = 0.0
        anyf = False
        for p, items, wt in gen_scenarios(rt, enum_cap, nsamp):
            d = scenario_drops(rt, items, wt)
            if d == 0:
                anyf = True
            pen += p * d
            psum += p
            if time.time() > hard_deadline:
                break
        if psum > 1e-12:
            pen /= psum
        if not anyf:
            v = (False, BIG)
        else:
            v = (True, cf * pen)
        cache[rt] = v
        return v

    eval_cache = {}
    final_cache = {}

    def eval_route(rt):
        return eval_route_generic(rt, SCEN_ENUM, SCEN_SAMPLE, eval_cache)

    # ------------------ travel / route helpers ------------------
    def travel(rt):
        if not rt:
            return 0.0
        t = D[0][rt[0]]
        prev = rt[0]
        for b in rt[1:]:
            t += D[prev][b]
            prev = b
        t += D[prev][0]
        return float(t)

    def mintrav(rt):
        if not rt:
            return 0.0
        t1 = travel(rt)
        t2 = travel(rt[::-1])
        return t1 if t1 < t2 else t2

    def route_solution(rt):
        # returns (oriented route list, cost, ok)
        if not rt:
            return [], 0.0, True
        r1 = list(rt)
        r2 = r1[::-1]
        t1 = travel(r1)
        t2 = travel(r2)
        if t1 <= t2:
            first, second, tf, ts = r1, r2, t1, t2
        else:
            first, second, tf, ts = r2, r1, t2, t1
        ok1, p1 = eval_route(tuple(first))
        c1 = tf + p1
        if ok1 and p1 == 0.0 and ts >= tf:
            return first, c1, ok1
        ok2, p2 = eval_route(tuple(second))
        c2 = ts + p2
        if c2 < c1 - 1e-12:
            return second, c2, ok2
        return first, c1, ok1

    # ------------------ initial construction (Clarke-Wright) ------------------
    ids = list(cust_ids)
    if K >= len(ids):
        routes0 = [[c] for c in ids] + [[] for _ in range(K - len(ids))]
    else:
        routes = [[c] for c in ids]
        route_of = {c: i for i, c in enumerate(ids)}
        areas = [exp_area[c] for c in ids]
        wts = [exp_wt[c] for c in ids]
        savings = []
        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                i, j = ids[ii], ids[jj]
                s = D[0][i] + D[0][j] - D[i][j]
                savings.append((s, i, j))
        savings.sort(key=lambda x: -x[0])
        nroutes = len(ids)
        for s, i, j in savings:
            if nroutes <= K:
                break
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue
            A = routes[ri]
            B = routes[rj]
            if A is None or B is None:
                continue
            if not (A[0] == i or A[-1] == i):
                continue
            if not (B[0] == j or B[-1] == j):
                continue
            if areas[ri] + areas[rj] > HW + 1e-9 or wts[ri] + wts[rj] > Q + 1e-9:
                continue
            if A[-1] != i:
                A = A[::-1]
            if B[0] != j:
                B = B[::-1]
            newr = A + B
            routes[ri] = newr
            areas[ri] += areas[rj]
            wts[ri] += wts[rj]
            for c in B:
                route_of[c] = ri
            routes[rj] = None
            nroutes -= 1
        rts = [r for r in routes if r]
        while len(rts) > K:
            rts.sort(key=lambda r: sum(exp_area[c] for c in r))
            a = rts.pop(0)
            b = rts.pop(0)
            rts.append(a + b)
        routes0 = rts

    # ------------------ solution bookkeeping ------------------
    best_feas = None
    best_feas_obj = float("inf")

    def record(sol_, total_, sok_):
        nonlocal best_feas, best_feas_obj
        if all(sok_) and total_ < best_feas_obj - 1e-9:
            best_feas_obj = total_
            best_feas = [list(r) for r in sol_]
            out = [list(r) for r in sol_]
            while len(out) < K:
                out.append([])
            if logger:
                try:
                    logger.log_solution(float(total_), {"objective_value": float(total_), "routes": out})
                except Exception:
                    pass

    rngls = random.Random(12345)

    # ------------------ local search ------------------
    def try_moves(sol, scost, sok):
        any_impr = False
        Kr = len(sol)

        # Relocate (inter + intra)
        idxs = [(a, i) for a in range(Kr) for i in range(len(sol[a]))]
        rngls.shuffle(idxs)
        for (a, i) in idxs:
            if time.time() > deadline:
                return any_impr
            if i >= len(sol[a]):
                continue
            c = sol[a][i]
            A2 = sol[a][:i] + sol[a][i + 1:]
            if not A2 and n >= Kr:
                continue
            tA2 = mintrav(A2)
            base_a = scost[a]
            done = False
            for b in range(Kr):
                if done:
                    break
                Bb = A2 if b == a else sol[b]
                for p in range(len(Bb) + 1):
                    B2 = Bb[:p] + [c] + Bb[p:]
                    if b == a:
                        if B2 == sol[a]:
                            continue
                        lb = mintrav(B2) - base_a
                        if lb >= -1e-9:
                            continue
                        rB, cB, okB = route_solution(B2)
                        if cB < base_a - 1e-9:
                            sol[a] = rB
                            scost[a] = cB
                            sok[a] = okB
                            any_impr = True
                            done = True
                            break
                    else:
                        lb = tA2 + mintrav(B2) - base_a - scost[b]
                        if lb >= -1e-9:
                            continue
                        rA, cA, okA = route_solution(A2)
                        rB, cB, okB = route_solution(B2)
                        if cA + cB < base_a + scost[b] - 1e-9:
                            sol[a] = rA
                            scost[a] = cA
                            sok[a] = okA
                            sol[b] = rB
                            scost[b] = cB
                            sok[b] = okB
                            any_impr = True
                            done = True
                            break

        # Swap (inter-route)
        positions = [(a, i) for a in range(Kr) for i in range(len(sol[a]))]
        rngls.shuffle(positions)
        P = len(positions)
        for x in range(P):
            if time.time() > deadline:
                return any_impr
            a, i = positions[x]
            if i >= len(sol[a]):
                continue
            for y in range(x + 1, P):
                b, j = positions[y]
                if a == b:
                    continue
                if i >= len(sol[a]) or j >= len(sol[b]):
                    continue
                c = sol[a][i]
                d = sol[b][j]
                A2 = sol[a][:i] + [d] + sol[a][i + 1:]
                B2 = sol[b][:j] + [c] + sol[b][j + 1:]
                lb = mintrav(A2) + mintrav(B2) - scost[a] - scost[b]
                if lb >= -1e-9:
                    continue
                rA, cA, okA = route_solution(A2)
                rB, cB, okB = route_solution(B2)
                if cA + cB < scost[a] + scost[b] - 1e-9:
                    sol[a] = rA
                    scost[a] = cA
                    sok[a] = okA
                    sol[b] = rB
                    scost[b] = cB
                    sok[b] = okB
                    any_impr = True
                    break

        # 2-opt intra-route
        for a in range(Kr):
            if time.time() > deadline:
                return any_impr
            changed = True
            while changed:
                changed = False
                L = len(sol[a])
                for i in range(L - 1):
                    for j in range(i + 1, L):
                        R2 = sol[a][:i] + sol[a][i:j + 1][::-1] + sol[a][j + 1:]
                        lb = mintrav(R2) - scost[a]
                        if lb >= -1e-9:
                            continue
                        rA, cA, okA = route_solution(R2)
                        if cA < scost[a] - 1e-9:
                            sol[a] = rA
                            scost[a] = cA
                            sok[a] = okA
                            any_impr = True
                            changed = True
                            break
                    if changed:
                        break
                if time.time() > deadline:
                    return any_impr
        return any_impr

    def local_search(sol, scost, sok):
        while time.time() < deadline:
            if not try_moves(sol, scost, sok):
                break

    def perturb(cur, rngp):
        Kr = len(cur)
        nm = rngp.randint(2, 4)
        for _ in range(nm):
            minlen = 1 if n >= Kr else 0
            cand = [a for a in range(Kr) if len(cur[a]) > minlen]
            if not cand:
                break
            a = rngp.choice(cand)
            i = rngp.randrange(len(cur[a]))
            c = cur[a].pop(i)
            b = rngp.randrange(Kr)
            p = rngp.randrange(len(cur[b]) + 1)
            cur[b].insert(p, c)
        # occasional segment reversal
        if rngp.random() < 0.5:
            cand = [a for a in range(Kr) if len(cur[a]) >= 3]
            if cand:
                a = rngp.choice(cand)
                L = len(cur[a])
                i = rngp.randrange(L - 1)
                j = rngp.randrange(i + 1, L)
                cur[a][i:j + 1] = cur[a][i:j + 1][::-1]

    # ------------------ main ILS ------------------
    sol, scost, sok = [], [], []
    for r in routes0:
        if r:
            rr, cc, oo = route_solution(r)
        else:
            rr, cc, oo = [], 0.0, True
        sol.append(rr)
        scost.append(cc)
        sok.append(oo)
    record(sol, sum(scost), sok)

    local_search(sol, scost, sok)
    total = sum(scost)
    record(sol, total, sok)

    best = [list(r) for r in sol]
    best_total = total
    rngp = random.Random(777)

    while time.time() < deadline:
        cur = [list(r) for r in best]
        perturb(cur, rngp)
        cur2, ccost, cok = [], [], []
        for r in cur:
            if r:
                rr, cc, oo = route_solution(r)
            else:
                rr, cc, oo = [], 0.0, True
            cur2.append(rr)
            ccost.append(cc)
            cok.append(oo)
        cur = cur2
        local_search(cur, ccost, cok)
        t = sum(ccost)
        if t < best_total - 1e-9:
            best = [list(r) for r in cur]
            best_total = t
            record(cur, t, cok)

    # ------------------ finalize ------------------
    out_routes = best_feas if best_feas is not None else best
    out_routes = [list(r) for r in out_routes]
    while len(out_routes) < K:
        out_routes.append([])
    out_routes = out_routes[:max(K, len(out_routes))]

    obj = 0.0
    for r in out_routes:
        obj += travel(r)
    for r in out_routes:
        if not r:
            continue
        if time.time() < hard_deadline - 0.5:
            ok, pen = eval_route_generic(tuple(r), FINAL_ENUM, FINAL_SAMPLE, final_cache)
        else:
            ok, pen = eval_route(tuple(r))
        obj += pen

    sol_dict = {"objective_value": float(obj), "routes": [[int(c) for c in r] for r in out_routes]}
    with open(args.solution_path, "w") as f:
        json.dump(sol_dict, f)
    if logger:
        try:
            logger.log_solution(float(obj), sol_dict)
        except Exception:
            pass


if __name__ == "__main__":
    main()