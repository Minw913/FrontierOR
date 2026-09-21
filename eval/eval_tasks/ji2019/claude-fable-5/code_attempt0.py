import argparse
import json
import time
import random

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

EPS = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(1, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None
    random.seed(0)

    with open(args.instance_path) as f:
        data = json.load(f)

    n = int(data['num_ships'])
    nb = int(data['num_berths'])
    W = float(data['chamber_width_W'])
    L = float(data['chamber_length_L'])
    Du = float(data['lockage_duration_Du'])
    MT = float(data['min_time_between_lockages_MT'])
    sc = float(data['lock_start_availability_sc'])
    ek = float(data['berth_end_availability_ek'])

    ships = data['ships']
    berths = data['berths']
    Hmat = data['handling_times']

    sid = [s['id'] for s in ships]
    id2i = {s['id']: i for i, s in enumerate(ships)}
    bid = [b['id'] for b in berths]
    bid2i = {b['id']: i for i, b in enumerate(berths)}

    wid = [float(s['width']) for s in ships]
    ln = [float(s['length']) for s in ships]
    arr = [float(s['arrival_time']) for s in ships]
    pen = [float(s['transshipment_penalty']) for s in ships]
    MR = [set(id2i[x] for x in (s.get('mooring_set_MR') or []) if x in id2i) for s in ships]
    EB = [[bid2i[x] for x in (s.get('eligible_berths_B') or []) if x in bid2i] for s in ships]
    sk = [float(b['start_availability_sk']) for b in berths]

    H = [[float(Hmat[i][k]) for k in range(nb)] for i in range(n)] if n > 0 and nb > 0 else [[0.0] * nb for _ in range(n)]

    fits = [wid[i] <= W + EPS and ln[i] <= L + EPS for i in range(n)]
    has_berth = [len(EB[i]) > 0 for i in range(n)]

    # forced modes: 0=lock, 1=trans, None=free
    forced = [None] * n
    for i in range(n):
        if not has_berth[i]:
            forced[i] = 0
        elif not fits[i]:
            forced[i] = 1

    # ---------------- packing feasibility ----------------
    pack_cache = {}

    def pack_feasible(fs):
        r = pack_cache.get(fs)
        if r is not None:
            return r
        idxs = sorted(fs, key=lambda i: (-ln[i], -wid[i], i))
        ok = True
        rows = []  # [height, used_width, last_ship]
        tot_h = 0.0
        for s in idxs:
            if wid[s] > W + EPS or ln[s] > L + EPS:
                ok = False
                break
            placed = False
            for r0 in rows:
                if r0[1] + wid[s] <= W + EPS and r0[2] in MR[s]:
                    r0[1] += wid[s]
                    r0[2] = s
                    placed = True
                    break
            if not placed:
                if tot_h + ln[s] <= L + EPS:
                    rows.append([ln[s], wid[s], s])
                    tot_h += ln[s]
                    placed = True
            if not placed:
                ok = False
                break
        pack_cache[fs] = ok
        return ok

    # ---------------- lock evaluation (DP over consecutive groups) ----------------
    lock_cache = {}

    def eval_lock(lockset):
        key = frozenset(lockset)
        if key in lock_cache:
            return lock_cache[key]
        if not lockset:
            res = (0.0, [], [])
            lock_cache[key] = res
            return res
        order = sorted(lockset, key=lambda i: (arr[i], i))
        m = len(order)
        area = W * L
        prefA = [0.0] * (m + 1)
        prefR = [0.0] * (m + 1)
        for t, i in enumerate(order):
            prefA[t + 1] = prefA[t] + wid[i] * ln[i]
            prefR[t + 1] = prefR[t] + arr[i]
        # dp[i]: list of (delay, T, prev_j, prev_ptr)
        dp = [[] for _ in range(m + 1)]
        dp[0] = [(0.0, None, -1, -1)]
        for i in range(1, m + 1):
            cands = []
            j = i - 1
            while j >= 0:
                if prefA[i] - prefA[j] > area + EPS:
                    break
                g = order[j:i]
                feas = (len(g) == 1) or pack_feasible(frozenset(g))
                if feas:
                    amax = max(arr[x] for x in g)
                    sa = prefR[i] - prefR[j]
                    gl = len(g)
                    for pi, ent in enumerate(dp[j]):
                        d, T = ent[0], ent[1]
                        Tn = max(sc + Du, amax + Du)
                        if T is not None and T + MT > Tn:
                            Tn = T + MT
                        cands.append((d + gl * Tn - sa, Tn, j, pi))
                j -= 1
            # Pareto prune: sort by T asc, keep strictly decreasing delay
            cands.sort(key=lambda c: (c[1], c[0]))
            pf = []
            bestd = float('inf')
            for c in cands:
                if c[0] < bestd - 1e-12:
                    pf.append(c)
                    bestd = c[0]
            if len(pf) > 40:
                idxsel = set()
                for t in range(40):
                    idxsel.add(int(round(t * (len(pf) - 1) / 39.0)))
                pf = [pf[t] for t in sorted(idxsel)]
            dp[i] = pf
            if not dp[i]:
                lock_cache[key] = None
                return None
        # best = min delay at dp[m]
        best = min(dp[m], key=lambda c: (c[0], c[1]))
        groups = []
        Ts = []
        level = m
        ent = best
        while level > 0:
            j = ent[2]
            groups.append(order[j:level])
            Ts.append(ent[1])
            pent = dp[j][ent[3]]
            ent = pent
            level = j
        groups.reverse()
        Ts.reverse()
        total = best[0]
        res = (total, groups, Ts)
        lock_cache[key] = res
        return res

    # ---------------- transshipment evaluation ----------------
    trans_cache = {}

    def berth_cost(lst, k, end_cap):
        t = sk[k]
        tot = 0.0
        for s in lst:
            st = max(t, arr[s])
            c = st + H[s][k]
            if c > end_cap + EPS:
                return None
            tot += c - arr[s]
            t = c
        return tot

    def eval_trans(transset, relax=False):
        key = (frozenset(transset), relax)
        if key in trans_cache:
            return trans_cache[key]
        end_cap = float('inf') if relax else ek
        if not transset:
            res = (0.0, {})
            trans_cache[key] = res
            return res
        order = sorted(transset, key=lambda i: (arr[i], i))
        assign = {}
        seq = {k: [] for k in range(nb)}
        free = [sk[k] for k in range(nb)]
        ok = True
        for s in order:
            best = None
            for k in EB[s]:
                st = max(free[k], arr[s])
                c = st + H[s][k]
                if c <= end_cap + EPS and (best is None or c < best[0] - 1e-9):
                    best = (c, k)
            if best is None:
                ok = False
                break
            c, k = best
            free[k] = c
            seq[k].append(s)
            assign[s] = k
        if not ok:
            trans_cache[key] = None
            return None
        # local improvement: move ships between berths
        for _ in range(3):
            improved = False
            for s in order:
                k = assign[s]
                cur = seq[k]
                base_k = berth_cost(cur, k, end_cap)
                rem = [x for x in cur if x != s]
                base_rem = berth_cost(rem, k, end_cap)
                if base_k is None or base_rem is None:
                    continue
                moved = False
                for k2 in EB[s]:
                    if k2 == k:
                        continue
                    l2 = seq[k2]
                    base2 = berth_cost(l2, k2, end_cap)
                    if base2 is None:
                        continue
                    nl2 = sorted(l2 + [s], key=lambda i: (arr[i], i))
                    c2 = berth_cost(nl2, k2, end_cap)
                    if c2 is None:
                        continue
                    if base_rem + c2 < base_k + base2 - 1e-9:
                        seq[k] = rem
                        seq[k2] = nl2
                        assign[s] = k2
                        improved = True
                        moved = True
                        break
                if moved:
                    continue
            if not improved:
                break
        total = 0.0
        for k in range(nb):
            if seq[k]:
                total += berth_cost(seq[k], k, end_cap)
        total += sum(pen[s] for s in transset)
        out = {s: (assign[s], seq[assign[s]].index(s)) for s in transset}
        res = (total, out)
        trans_cache[key] = res
        return res

    # ---------------- full evaluation ----------------
    eval_cache = {}

    def evaluate(mv):
        if mv in eval_cache:
            return eval_cache[mv]
        lockset = frozenset(i for i in range(n) if mv[i] == 0)
        transset = frozenset(i for i in range(n) if mv[i] == 1)
        rl = eval_lock(lockset)
        rt = eval_trans(transset)
        c = float('inf') if (rl is None or rt is None) else rl[0] + rt[0]
        eval_cache[mv] = c
        return c

    def build_solution(mv):
        lockset = frozenset(i for i in range(n) if mv[i] == 0)
        transset = frozenset(i for i in range(n) if mv[i] == 1)
        rl = eval_lock(lockset)
        rt = eval_trans(transset)
        if rt is None:
            rt = eval_trans(transset, relax=True)
        if rl is None:
            # fallback: singletons per lockage in FCFS order
            order = sorted(lockset, key=lambda i: (arr[i], i))
            groups = [[s] for s in order]
            Ts = []
            prev = None
            tot = 0.0
            for g in groups:
                T = max(sc + Du, arr[g[0]] + Du)
                if prev is not None and prev + MT > T:
                    T = prev + MT
                Ts.append(T)
                tot += T - arr[g[0]]
                prev = T
            rl = (tot, groups, Ts)
        if rt is None:
            rt = (0.0, {})
        obj = rl[0] + rt[0]
        modes = {str(sid[i]): int(mv[i]) for i in range(n)}
        completion = {}
        lockassign = {}
        lct = {}
        for li, (g, T) in enumerate(zip(rl[1], rl[2])):
            lct[str(li)] = float(T)
            for s in g:
                lockassign[str(sid[s])] = int(li)
                completion[str(sid[s])] = float(T)
        ba = {}
        for s, (k, o) in rt[1].items():
            ba[str(sid[s])] = {"berth": int(bid[k]), "order": int(o)}
            completion[str(sid[s])] = float(arr[s])
        sol = {
            "objective_value": float(obj),
            "modes": modes,
            "completion_times": completion,
            "lockage_assignments": lockassign,
            "lockage_completion_times": lct,
            "berth_assignments": ba,
        }
        return obj, sol

    # ---------------- initial solution ----------------
    if n == 0:
        _, sol = 0.0, {
            "objective_value": 0.0, "modes": {}, "completion_times": {},
            "lockage_assignments": {}, "lockage_completion_times": {}, "berth_assignments": {}
        }
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    mv0 = []
    for i in range(n):
        if forced[i] is not None:
            mv0.append(forced[i])
        else:
            mv0.append(0)  # default: lock mode
    mv0 = tuple(mv0)
    c0 = evaluate(mv0)

    # try alternative starts if infeasible or to diversify
    free = [i for i in range(n) if forced[i] is None]
    candidates = [mv0]
    mv_all_trans = tuple(1 if (forced[i] is None or forced[i] == 1) and has_berth[i] else 0 for i in range(n))
    candidates.append(mv_all_trans)
    # greedy per-ship: pick trans if solo lock delay heuristic worse than penalty
    mv_greedy = list(mv0)
    for i in free:
        # rough solo comparison
        best_tr = float('inf')
        for k in EB[i]:
            st = max(sk[k], arr[i])
            c = st + H[i][k]
            if c <= ek + EPS:
                best_tr = min(best_tr, c - arr[i] + pen[i])
        solo_lock = max(sc + Du, arr[i] + Du) - arr[i]
        if best_tr < solo_lock:
            mv_greedy[i] = 1
    candidates.append(tuple(mv_greedy))

    best_mv = None
    best_c = float('inf')
    for mv in candidates:
        c = evaluate(mv)
        if c < best_c:
            best_c = c
            best_mv = mv

    if best_mv is None or best_c == float('inf'):
        best_mv = mv0
        best_c = c0

    obj, sol = build_solution(best_mv)
    if logger:
        logger.log_solution(obj, sol)
    best_obj, best_sol = obj, sol

    # ---------------- local search ----------------
    def climb(mv):
        mv = list(mv)
        c = evaluate(tuple(mv))
        while time.time() < deadline:
            best_i = None
            best_c2 = c
            for i in free:
                if time.time() >= deadline:
                    break
                mv[i] ^= 1
                c2 = evaluate(tuple(mv))
                mv[i] ^= 1
                if c2 < best_c2 - 1e-9:
                    best_c2 = c2
                    best_i = i
            if best_i is None:
                break
            mv[best_i] ^= 1
            c = best_c2
        return tuple(mv), c

    if free:
        mv2, c2 = climb(best_mv)
        if c2 < best_c - 1e-9:
            best_mv, best_c = mv2, c2
            obj, sol = build_solution(best_mv)
            if obj < best_obj - 1e-9:
                best_obj, best_sol = obj, sol
                if logger:
                    logger.log_solution(obj, sol)

        # perturbation loop
        while time.time() < deadline:
            mv = list(best_mv)
            k = random.randint(1, min(3, len(free)))
            for i in random.sample(free, k):
                mv[i] ^= 1
            mv2, c2 = climb(mv)
            if c2 < best_c - 1e-9:
                best_mv, best_c = mv2, c2
                obj, sol = build_solution(best_mv)
                if obj < best_obj - 1e-9:
                    best_obj, best_sol = obj, sol
                    if logger:
                        logger.log_solution(obj, sol)

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f, indent=2)


if __name__ == '__main__':
    main()