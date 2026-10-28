import argparse
import heapq
import json
import random
import sys
import time
from collections import deque

from solution_logger import SolutionLogger

if hasattr(int, "bit_count"):
    popcount = int.bit_count
else:
    def popcount(x):
        return bin(x).count("1")


class SearchTimeout(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    sys.setrecursionlimit(20000)

    t0 = time.time()
    T = max(1, int(args.time_limit))
    hard_end = t0 + T - max(0.3, 0.02 * T)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------------- read instance ----------------
    with open(args.instance_path, "r") as f:
        inst = json.load(f)
    g = inst["graph"] if isinstance(inst, dict) and "graph" in inst else inst
    n = int(g.get("n", 0) or 0)
    adjset = [set() for _ in range(n)]
    raw_edges = g.get("edges")
    if raw_edges:
        for e in raw_edges:
            u = int(e[0]); v = int(e[1])
            if u != v and 0 <= u < n and 0 <= v < n:
                adjset[u].add(v)
                adjset[v].add(u)
    else:
        for u, nbrs in enumerate(g.get("adjacency_list", [])):
            if u >= n:
                break
            for v in nbrs:
                v = int(v)
                if v != u and 0 <= v < n:
                    adjset[u].add(v)
                    adjset[v].add(u)

    adj_list = [sorted(s) for s in adjset]
    deg = [len(a) for a in adj_list]
    m = sum(deg) // 2
    del adjset

    # ---------------- helpers ----------------
    def build_solution(colors):
        remap = {}
        norm = [0] * n
        nxt = 0
        for v in range(n):
            c = colors[v]
            r = remap.get(c)
            if r is None:
                r = nxt
                remap[c] = nxt
                nxt += 1
            norm[v] = r
        classes = [[] for _ in range(nxt)]
        for v in range(n):
            classes[norm[v]].append(v)
        sol = {
            "objective_value": nxt,
            "coloring": {str(v): norm[v] for v in range(n)},
            "color_classes": {str(i): classes[i] for i in range(nxt)},
        }
        return sol, norm, nxt

    def verify(colors):
        for v in range(n):
            cv = colors[v]
            for u in adj_list[v]:
                if u > v and colors[u] == cv:
                    return False
        return True

    def emit(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    # trivial cases
    if n == 0:
        sol = {"objective_value": 0, "coloring": {}, "color_classes": {}}
        if logger:
            logger.log_solution(0, sol)
        emit(sol)
        return
    if m == 0:
        sol, _, _ = build_solution([0] * n)
        if logger:
            logger.log_solution(1, sol)
        emit(sol)
        return

    # bitmasks (only for reasonably sized graphs)
    use_bits = n <= 6000
    adjm = None
    if use_bits:
        adjm = [0] * n
        for v in range(n):
            b = 0
            for u in adj_list[v]:
                b |= (1 << u)
            adjm[v] = b

    # ---------------- DSATUR (heap based) ----------------
    def dsatur(rng=None):
        colors = [-1] * n
        satmask = [0] * n
        satc = [0] * n
        if rng is not None:
            tie = [rng.random() for _ in range(n)]
        else:
            tie = [0.0] * n
        heap = [(0, -deg[v], tie[v], v) for v in range(n)]
        heapq.heapify(heap)
        nc = 0
        remaining = n
        while remaining > 0:
            while True:
                ns, nd, tv, v = heapq.heappop(heap)
                if colors[v] < 0 and -ns == satc[v]:
                    break
            fm = satmask[v]
            c = 0
            while (fm >> c) & 1:
                c += 1
            colors[v] = c
            if c + 1 > nc:
                nc = c + 1
            bit = 1 << c
            for u in adj_list[v]:
                if colors[u] < 0 and not (satmask[u] >> c) & 1:
                    satmask[u] |= bit
                    satc[u] += 1
                    heapq.heappush(heap, (-satc[u], -deg[u], tie[u], u))
            remaining -= 1
        return colors, nc

    # ---------------- max clique (Tomita, bitset) for lower bound ----------------
    def max_clique(deadline):
        best = [0]
        bestcl = [[]]
        calls = [0]

        def color_sort(P):
            order = []
            bounds = []
            U = P
            cnum = 0
            while U:
                cnum += 1
                A = U
                while A:
                    lsb = A & -A
                    v = lsb.bit_length() - 1
                    A ^= lsb
                    U ^= lsb
                    order.append(v)
                    bounds.append(cnum)
                    A &= ~adjm[v]
            return order, bounds

        def expand(size, cl, P):
            calls[0] += 1
            if (calls[0] & 255) == 0 and time.time() > deadline:
                raise SearchTimeout()
            order, bounds = color_sort(P)
            for i in range(len(order) - 1, -1, -1):
                if size + bounds[i] <= best[0]:
                    return
                v = order[i]
                NP = P & adjm[v]
                if NP:
                    cl.append(v)
                    expand(size + 1, cl, NP)
                    cl.pop()
                elif size + 1 > best[0]:
                    best[0] = size + 1
                    bestcl[0] = cl + [v]
                P &= ~(1 << v)

        try:
            expand(0, [], (1 << n) - 1)
        except SearchTimeout:
            pass
        return best[0], bestcl[0]

    # ---------------- Tabucol local search for fixed k ----------------
    def tabucol(init_colors, k, rng, deadline):
        color = list(init_colors)
        counts = [[0] * k for _ in range(n)]
        for v in range(n):
            cv = color[v]
            for u in adj_list[v]:
                counts[u][cv] += 1
        conf_list = []
        pos = [-1] * n
        F = 0
        for v in range(n):
            if counts[v][color[v]] > 0:
                pos[v] = len(conf_list)
                conf_list.append(v)
                F += counts[v][color[v]]
        F //= 2
        tabu = [[0] * k for _ in range(n)]
        bestF = F
        it = 0
        while F > 0:
            it += 1
            if (it & 127) == 0 and time.time() >= deadline:
                return None, F
            L = len(conf_list)
            trials = 10 if L >= 10 else L
            best = None; bestd = 1 << 30
            fb = None; fbd = 1 << 30
            for _ in range(trials):
                v = conf_list[rng.randrange(L)]
                cv = color[v]
                row = counts[v]
                base = row[cv]
                trow = tabu[v]
                for d in range(k):
                    if d == cv:
                        continue
                    delta = row[d] - base
                    if delta < fbd:
                        fbd = delta; fb = (v, d)
                    if trow[d] <= it or F + delta < bestF:
                        if delta < bestd or (delta == bestd and rng.random() < 0.25):
                            bestd = delta; best = (v, d)
            if best is None:
                best = fb; bestd = fbd
            v, d = best
            cv = color[v]
            F += bestd
            for u in adj_list[v]:
                counts[u][cv] -= 1
                counts[u][d] += 1
                cu = color[u]
                if counts[u][cu] > 0:
                    if pos[u] < 0:
                        pos[u] = len(conf_list); conf_list.append(u)
                else:
                    i = pos[u]
                    if i >= 0:
                        last = conf_list[-1]
                        conf_list[i] = last; pos[last] = i
                        conf_list.pop(); pos[u] = -1
            color[v] = d
            if counts[v][d] > 0:
                if pos[v] < 0:
                    pos[v] = len(conf_list); conf_list.append(v)
            else:
                i = pos[v]
                if i >= 0:
                    last = conf_list[-1]
                    conf_list[i] = last; pos[last] = i
                    conf_list.pop(); pos[v] = -1
            tabu[v][cv] = it + 7 + rng.randrange(10) + (F >> 3)
            if F < bestF:
                bestF = F
        return color, 0

    # ---------------- exact k-colorability (DSATUR backtracking + kernelization) ----------------
    def exact_colorable(k, clique, deadline):
        if k <= 0:
            return ("no", None)
        # kernelize: iteratively remove vertices with current degree < k
        rem = (1 << n) - 1
        kdeg = list(deg)
        dq = deque([v for v in range(n) if kdeg[v] < k])
        removed = []
        while dq:
            v = dq.popleft()
            if not (rem >> v) & 1:
                continue
            rem &= ~(1 << v)
            removed.append(v)
            mm = adjm[v] & rem
            while mm:
                lsb = mm & -mm
                u = lsb.bit_length() - 1
                mm ^= lsb
                kdeg[u] -= 1
                if kdeg[u] == k - 1:
                    dq.append(u)

        def finish(pairs):
            coloring = [-1] * n
            for v, c in pairs:
                coloring[v] = c
            for v in reversed(removed):
                forb = 0
                mm = adjm[v]
                while mm:
                    lsb = mm & -mm
                    u = lsb.bit_length() - 1
                    mm ^= lsb
                    cu = coloring[u]
                    if cu >= 0:
                        forb |= (1 << cu)
                c = 0
                while (forb >> c) & 1:
                    c += 1
                coloring[v] = c
            return coloring

        if rem == 0:
            return ("yes", finish([]))

        verts = []
        mm = rem
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            mm ^= lsb
            verts.append(v)
        nk = len(verts)
        if k >= nk:
            return ("yes", finish([(v, i) for i, v in enumerate(verts)]))
        if nk > 900:
            return ("unknown", None)

        idx = {v: i for i, v in enumerate(verts)}
        kadj = [0] * nk
        for i, v in enumerate(verts):
            b = 0
            mm = adjm[v] & rem
            while mm:
                lsb = mm & -mm
                u = lsb.bit_length() - 1
                mm ^= lsb
                b |= (1 << idx[u])
            kadj[i] = b
        kdegi = [popcount(kadj[i]) for i in range(nk)]

        cnt = [[0] * k for _ in range(nk)]
        sat = [0] * nk
        assign = [-1] * nk
        clsize = [0] * k
        uncol = (1 << nk) - 1
        trail = []
        nodes = [0]

        def assign_vc(v, c):
            nonlocal uncol
            mark = len(trail)
            nb = kadj[v] & uncol
            while nb:
                lsb = nb & -nb
                u = lsb.bit_length() - 1
                nb ^= lsb
                row_u = cnt[u]
                if row_u[c] == 0:
                    if sat[u] + 1 >= k:
                        while len(trail) > mark:
                            uu, inc, cc = trail.pop()
                            cnt[uu][cc] -= 1
                            if inc:
                                sat[uu] -= 1
                        return None
                    sat[u] += 1
                    row_u[c] = 1
                    trail.append((u, True, c))
                else:
                    row_u[c] += 1
                    trail.append((u, False, c))
            assign[v] = c
            clsize[c] += 1
            uncol &= ~(1 << v)
            return mark

        def unassign_vc(v, c, mark):
            nonlocal uncol
            uncol |= (1 << v)
            assign[v] = -1
            clsize[c] -= 1
            while len(trail) > mark:
                uu, inc, cc = trail.pop()
                cnt[uu][cc] -= 1
                if inc:
                    sat[uu] -= 1

        # fix a clique to distinct colors (valid by color symmetry)
        cl = [idx[v] for v in clique if v in idx]
        if len(cl) > k:
            return ("no", None)
        for c, v in enumerate(cl):
            if cnt[v][c] != 0:
                return ("no", None)
            if assign_vc(v, c) is None:
                return ("no", None)

        def search():
            nonlocal uncol
            if uncol == 0:
                return True
            nodes[0] += 1
            if (nodes[0] & 1023) == 0 and time.time() > deadline:
                raise SearchTimeout()
            mm = uncol
            bv = -1; bs = -1; bd = -1
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                mm ^= lsb
                s = sat[v]
                if s > bs or (s == bs and kdegi[v] > bd):
                    bs = s; bd = kdegi[v]; bv = v
            if bs >= k:
                return False
            v = bv
            row = cnt[v]
            corder = sorted(range(k), key=lambda c: -clsize[c])
            for c in corder:
                if row[c] == 0:
                    mark = assign_vc(v, c)
                    if mark is not None:
                        if search():
                            return True
                        unassign_vc(v, c, mark)
            return False

        try:
            found = search()
        except SearchTimeout:
            return ("unknown", None)
        if not found:
            return ("no", None)
        return ("yes", finish([(verts[i], assign[i]) for i in range(nk)]))

    # ---------------- driver ----------------
    rng = random.Random(0)

    best_colors, best_k = dsatur()
    sol, best_colors, best_k = build_solution(best_colors)
    if logger:
        logger.log_solution(best_k, sol)

    # quick randomized DSATUR improvements
    rds_end = min(t0 + 0.05 * T, t0 + 2.0, hard_end)
    runs = 0
    while runs < 500 and best_k > 2 and time.time() < rds_end:
        runs += 1
        c_r, k_r = dsatur(rng)
        if k_r < best_k and verify(c_r):
            sol, best_colors, best_k = build_solution(c_r)
            if logger:
                logger.log_solution(best_k, sol)

    # lower bound via maximum clique
    LB = 2
    clique = []
    if use_bits and best_k > 2:
        cend = min(t0 + max(1.0, 0.12 * T), hard_end)
        if time.time() < cend:
            cs, clique = max_clique(cend)
            if cs > LB:
                LB = cs
            if LB > best_k:
                LB = best_k

    def tabucol_attempt(k, deadline):
        freq = [0] * best_k
        for c in best_colors:
            freq[c] += 1
        cmin = min(range(best_k), key=lambda c: freq[c])
        init = []
        for c in best_colors:
            if c == cmin:
                init.append(rng.randrange(k))
            elif c == k:
                init.append(cmin)
            else:
                init.append(c)
        return tabucol(init, k, rng, deadline)

    # heuristic improvement phase
    heur_end = t0 + (0.55 * T if n <= 1500 else 0.97 * T)
    if heur_end > hard_end:
        heur_end = hard_end
    while best_k > LB:
        now = time.time()
        if now >= heur_end:
            break
        k = best_k - 1
        attempt_deadline = min(heur_end, now + max(1.5, 0.10 * T))
        res, F = tabucol_attempt(k, attempt_deadline)
        if res is not None:
            sol_n, norm_n, kk = build_solution(res)
            if kk < best_k and verify(norm_n):
                sol = sol_n; best_colors = norm_n; best_k = kk
                if logger:
                    logger.log_solution(best_k, sol)

    # exact phase: try to prove optimality or find better colorings
    proven = (best_k == LB)
    while (not proven) and use_bits and best_k > LB:
        if time.time() >= hard_end - 0.05:
            break
        status, coloring = exact_colorable(best_k - 1, clique, hard_end)
        if status == "yes" and coloring is not None and verify(coloring):
            sol_n, norm_n, kk = build_solution(coloring)
            if kk < best_k:
                sol = sol_n; best_colors = norm_n; best_k = kk
                if logger:
                    logger.log_solution(best_k, sol)
                if best_k == LB:
                    proven = True
                    break
            else:
                break
        elif status == "no":
            proven = True
            break
        else:
            break

    # leftover time: keep improving heuristically
    while (not proven) and best_k > LB:
        now = time.time()
        if now >= hard_end:
            break
        k = best_k - 1
        res, F = tabucol_attempt(k, hard_end)
        if res is not None:
            sol_n, norm_n, kk = build_solution(res)
            if kk < best_k and verify(norm_n):
                sol = sol_n; best_colors = norm_n; best_k = kk
                if logger:
                    logger.log_solution(best_k, sol)
                continue
        break

    # ---------------- final output ----------------
    sol, norm, kk = build_solution(best_colors)
    if not verify(norm):
        c0, _ = dsatur()
        sol, norm, kk = build_solution(c0)
    emit(sol)


if __name__ == "__main__":
    main()