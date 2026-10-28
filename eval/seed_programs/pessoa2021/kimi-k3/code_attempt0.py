import argparse
import heapq
import json
import math
import random
import time

from solution_logger import SolutionLogger

# ------------------------- global instance data -------------------------
n = 0
K = 0
Q = 0
Gamma = 0
D = []      # nominal demands
DEV = []    # demand deviations (clamped >= 0)
C0 = []     # depot -> customer cost
CR = []     # customer -> depot cost
CM = []     # customer -> customer cost matrix
IDS = []    # customer identifiers (node indices)
DEPOT = 0


class Route:
    __slots__ = ("seq", "cost", "nom", "sl", "top", "revp")


def build_route(seq):
    r = Route()
    s = list(seq)
    L = len(s)
    cost = 0
    nom = 0
    if L:
        cost = C0[s[0]] + CR[s[-1]]
        cm = CM
        for t in range(L - 1):
            cost += cm[s[t]][s[t + 1]]
        devs = []
        dd = D
        dv = DEV
        for x in s:
            nom += dd[x]
            devs.append(dv[x])
        devs.sort()
        top = sum(devs[-Gamma:]) if Gamma > 0 else 0
    else:
        devs = []
        top = 0
    # prefix sums of (reverse arc - forward arc) for fast directed 2-opt
    revp = [0] * L
    for k in range(1, L):
        revp[k] = revp[k - 1] + CM[s[k]][s[k - 1]] - CM[s[k - 1]][s[k]]
    r.seq = s
    r.cost = cost
    r.nom = nom
    r.sl = devs
    r.top = top
    r.revp = revp
    return r


def route_cost_of_seq(s):
    L = len(s)
    if L == 0:
        return 0
    c = C0[s[0]]
    for t in range(L - 1):
        c += CM[s[t]][s[t + 1]]
    c += CR[s[-1]]
    return c


def worst_of_seq(s):
    nom = 0
    devs = []
    for x in s:
        nom += D[x]
        devs.append(DEV[x])
    if Gamma > 0 and len(devs) > Gamma:
        return nom + sum(heapq.nlargest(Gamma, devs))
    return nom + sum(devs)


# ------------- O(1)/small robust-demand delta helpers -------------
def top_add(sl, top, v):
    """New top-Gamma sum after adding one deviation value v."""
    if Gamma <= 0:
        return top
    L = len(sl)
    if L < Gamma:
        return top + v
    th = sl[-Gamma]
    if v > th:
        return top + v - th
    return top


def top_add_multi(sl, top, vals):
    """New top-Gamma sum after adding several deviation values."""
    if Gamma <= 0:
        return top
    L = len(sl)
    base = list(sl[-Gamma:]) if L > Gamma else list(sl)
    base.extend(vals)
    tot = L + len(vals)
    take = Gamma if Gamma < tot else tot
    if len(base) <= take:
        return sum(base)
    return sum(heapq.nlargest(take, base))


def top_swap(sl, top, out_v, in_v):
    """New top-Gamma sum when out_v is removed and in_v added (size unchanged)."""
    if Gamma <= 0:
        return 0
    L = len(sl)
    if L > Gamma:
        base = list(sl[-Gamma:])
        th = sl[-Gamma]
        if out_v >= th:
            base.remove(out_v)
            base.append(sl[-Gamma - 1])
    else:
        base = list(sl)
        base.remove(out_v)
    base.append(in_v)
    take = Gamma if Gamma < L else L
    if len(base) <= take:
        return sum(base)
    return sum(heapq.nlargest(take, base))


# ------------------------- arc delta helpers -------------------------
def removal_delta(s, i):
    x = s[i]
    a = s[i - 1] if i > 0 else -1
    b = s[i + 1] if i + 1 < len(s) else -1
    old = (CM[a][x] if a >= 0 else C0[x]) + (CM[x][b] if b >= 0 else CR[x])
    if a >= 0 and b >= 0:
        new = CM[a][b]
    elif a >= 0:
        new = CR[a]
    elif b >= 0:
        new = C0[b]
    else:
        new = 0
    return new - old


def insertion_delta(s, j, x):
    u = s[j - 1] if j > 0 else -1
    v = s[j] if j < len(s) else -1
    if u >= 0 and v >= 0:
        old = CM[u][v]
    elif u >= 0:
        old = CR[u]
    elif v >= 0:
        old = C0[v]
    else:
        old = 0
    new = (CM[u][x] if u >= 0 else C0[x]) + (CM[x][v] if v >= 0 else CR[x])
    return new - old


def replace_delta(s, i, y):
    x = s[i]
    a = s[i - 1] if i > 0 else -1
    b = s[i + 1] if i + 1 < len(s) else -1
    old = (CM[a][x] if a >= 0 else C0[x]) + (CM[x][b] if b >= 0 else CR[x])
    new = (CM[a][y] if a >= 0 else C0[y]) + (CM[y][b] if b >= 0 else CR[y])
    return new - old


def two_opt_delta(s, revp, i, j):
    a = s[i - 1] if i > 0 else -1
    b = s[j + 1] if j + 1 < len(s) else -1
    si = s[i]
    sj = s[j]
    old_l = CM[a][si] if a >= 0 else C0[si]
    new_l = CM[a][sj] if a >= 0 else C0[sj]
    old_r = CM[sj][b] if b >= 0 else CR[sj]
    new_r = CM[si][b] if b >= 0 else CR[si]
    return (new_l - old_l) + (new_r - old_r) + (revp[j] - revp[i])


# ------------------------- local search passes -------------------------
def pass_two_opt_intra(routes, end_time):
    improved = False
    for idx in range(len(routes)):
        if time.time() > end_time:
            break
        r = routes[idx]
        s = r.seq
        L = len(s)
        if L < 2:
            continue
        revp = r.revp
        bestd = 0
        bi = -1
        bj = -1
        for i in range(L - 1):
            for j in range(i + 1, L):
                d = two_opt_delta(s, revp, i, j)
                if d < bestd:
                    bestd = d
                    bi = i
                    bj = j
        if bi >= 0:
            ns = s[:bi] + s[bi:bj + 1][::-1] + s[bj + 1:]
            routes[idx] = build_route(ns)
            improved = True
    return improved


def pass_relocate(routes, end_time):
    R = len(routes)
    best = None
    bestd = 0
    for ai in range(R):
        if time.time() > end_time:
            break
        A = routes[ai]
        sa = A.seq
        La = len(sa)
        if La < 2:
            continue
        for i in range(La):
            x = sa[i]
            rem = removal_delta(sa, i)
            # intra-route reinsertion (set unchanged -> still feasible)
            for j in range(La + 1):
                if j == i or j == i + 1:
                    continue
                d = rem + insertion_delta(sa, j, x)
                if d < bestd:
                    bestd = d
                    best = (0, ai, i, j, -1)
            dx = D[x]
            dvx = DEV[x]
            for bi in range(R):
                if bi == ai:
                    continue
                B = routes[bi]
                nt = top_add(B.sl, B.top, dvx)
                if B.nom + dx + nt > Q:
                    continue
                sb = B.seq
                for j in range(len(sb) + 1):
                    d = rem + insertion_delta(sb, j, x)
                    if d < bestd:
                        bestd = d
                        best = (1, ai, i, bi, j)
    if best is None:
        return False
    if best[0] == 0:
        _, ai, i, j, _ = best
        s = routes[ai].seq[:]
        x = s.pop(i)
        s.insert(j - 1 if j > i else j, x)
        routes[ai] = build_route(s)
    else:
        _, ai, i, bi, j = best
        s = routes[ai].seq[:]
        x = s.pop(i)
        routes[ai] = build_route(s)
        sb = routes[bi].seq[:]
        sb.insert(j, x)
        routes[bi] = build_route(sb)
    return True


def pass_oropt(routes, end_time):
    R = len(routes)
    best = None
    bestd = 0
    for ai in range(R):
        if time.time() > end_time:
            break
        A = routes[ai]
        sa = A.seq
        La = len(sa)
        for l in (2, 3):
            if La < l + 1:
                continue
            for i in range(La - l + 1):
                f = sa[i]
                ll = sa[i + l - 1]
                a = sa[i - 1] if i > 0 else -1
                b = sa[i + l] if i + l < La else -1
                old = (CM[a][f] if a >= 0 else C0[f]) + (CM[ll][b] if b >= 0 else CR[ll])
                if a >= 0 and b >= 0:
                    new = CM[a][b]
                elif a >= 0:
                    new = CR[a]
                elif b >= 0:
                    new = C0[b]
                else:
                    new = 0
                rem = new - old
                segnom = 0
                segdev = []
                for t in range(l):
                    segnom += D[sa[i + t]]
                    segdev.append(DEV[sa[i + t]])
                # intra-route
                for j in range(La + 1):
                    if i <= j <= i + l:
                        continue
                    u = sa[j - 1] if j > 0 else -1
                    v = sa[j] if j < La else -1
                    if u >= 0 and v >= 0:
                        old2 = CM[u][v]
                    elif u >= 0:
                        old2 = CR[u]
                    elif v >= 0:
                        old2 = C0[v]
                    else:
                        old2 = 0
                    new2 = (CM[u][f] if u >= 0 else C0[f]) + (CM[ll][v] if v >= 0 else CR[ll])
                    d = rem + new2 - old2
                    if d < bestd:
                        bestd = d
                        best = (0, ai, i, l, j, -1)
                # inter-route
                for bi in range(R):
                    if bi == ai:
                        continue
                    B = routes[bi]
                    nt = top_add_multi(B.sl, B.top, segdev)
                    if B.nom + segnom + nt > Q:
                        continue
                    sb = B.seq
                    Lb = len(sb)
                    for j in range(Lb + 1):
                        u = sb[j - 1] if j > 0 else -1
                        v = sb[j] if j < Lb else -1
                        if u >= 0 and v >= 0:
                            old2 = CM[u][v]
                        elif u >= 0:
                            old2 = CR[u]
                        elif v >= 0:
                            old2 = C0[v]
                        else:
                            old2 = 0
                        new2 = (CM[u][f] if u >= 0 else C0[f]) + (CM[ll][v] if v >= 0 else CR[ll])
                        d = rem + new2 - old2
                        if d < bestd:
                            bestd = d
                            best = (1, ai, i, l, bi, j)
    if best is None:
        return False
    if best[0] == 0:
        _, ai, i, l, j, _ = best
        s = routes[ai].seq[:]
        seg = s[i:i + l]
        del s[i:i + l]
        j2 = j if j < i else j - l
        s[j2:j2] = seg
        routes[ai] = build_route(s)
    else:
        _, ai, i, l, bi, j = best
        s = routes[ai].seq[:]
        seg = s[i:i + l]
        del s[i:i + l]
        routes[ai] = build_route(s)
        sb = routes[bi].seq[:]
        sb[j:j] = seg
        routes[bi] = build_route(sb)
    return True


def pass_swap(routes, end_time):
    R = len(routes)
    best = None
    bestd = 0
    for ai in range(R):
        if time.time() > end_time:
            break
        A = routes[ai]
        sa = A.seq
        La = len(sa)
        if La == 0:
            continue
        for bi in range(ai + 1, R):
            B = routes[bi]
            sb = B.seq
            Lb = len(sb)
            if Lb == 0:
                continue
            for i in range(La):
                x = sa[i]
                dx = D[x]
                dvx = DEV[x]
                for j in range(Lb):
                    y = sb[j]
                    d = replace_delta(sa, i, y) + replace_delta(sb, j, x)
                    if d < bestd:
                        dy = D[y]
                        dvy = DEV[y]
                        tA = top_swap(A.sl, A.top, dvx, dvy)
                        if A.nom - dx + dy + tA > Q:
                            continue
                        tB = top_swap(B.sl, B.top, dvy, dvx)
                        if B.nom - dy + dx + tB > Q:
                            continue
                        bestd = d
                        best = (ai, i, bi, j)
    if best is None:
        return False
    ai, i, bi, j = best
    sa = routes[ai].seq[:]
    sb = routes[bi].seq[:]
    sa[i], sb[j] = sb[j], sa[i]
    routes[ai] = build_route(sa)
    routes[bi] = build_route(sb)
    return True


def pass_cross(routes, end_time):
    """2-opt* tail exchange between pairs of routes."""
    R = len(routes)
    best = None
    bestd = 0
    for ai in range(R):
        if time.time() > end_time:
            break
        A = routes[ai]
        sa = A.seq
        La = len(sa)
        if La == 0:
            continue
        for bi in range(ai + 1, R):
            B = routes[bi]
            sb = B.seq
            Lb = len(sb)
            if Lb == 0:
                continue
            for i in range(La):
                anode = sa[i]
                an = sa[i + 1] if i + 1 < La else -1
                old_a = CM[anode][an] if an >= 0 else CR[anode]
                for j in range(Lb):
                    if an < 0 and j == Lb - 1:
                        continue  # identity
                    bnode = sb[j]
                    bn = sb[j + 1] if j + 1 < Lb else -1
                    old_b = CM[bnode][bn] if bn >= 0 else CR[bnode]
                    new_a = CM[anode][bn] if bn >= 0 else CR[anode]
                    new_b = CM[bnode][an] if an >= 0 else CR[bnode]
                    d = (new_a - old_a) + (new_b - old_b)
                    if d < bestd:
                        nsa = sa[:i + 1] + sb[j + 1:]
                        nsb = sb[:j + 1] + sa[i + 1:]
                        if not nsa or not nsb:
                            continue
                        if worst_of_seq(nsa) <= Q and worst_of_seq(nsb) <= Q:
                            bestd = d
                            best = (ai, i, bi, j)
    if best is None:
        return False
    ai, i, bi, j = best
    sa = routes[ai].seq
    sb = routes[bi].seq
    nsa = sa[:i + 1] + sb[j + 1:]
    nsb = sb[:j + 1] + sa[i + 1:]
    routes[ai] = build_route(nsa)
    routes[bi] = build_route(nsb)
    return True


def local_search(routes, end_time):
    passes = (pass_two_opt_intra, pass_relocate, pass_oropt, pass_swap, pass_cross)
    while time.time() < end_time:
        any_imp = False
        for fn in passes:
            if time.time() >= end_time:
                break
            if fn(routes, end_time):
                any_imp = True
        if not any_imp:
            break
    return routes


def perturb(routes, rng, strength):
    R = len(routes)
    nonempty = [i for i in range(R) if len(routes[i].seq) > 0]
    for _ in range(strength):
        if not nonempty:
            break
        u = rng.random()
        if u < 0.45:
            ai = rng.choice(nonempty)
            A = routes[ai]
            if len(A.seq) <= 1:
                continue
            i = rng.randrange(len(A.seq))
            x = A.seq[i]
            bi = rng.randrange(R)
            if bi == ai:
                continue
            B = routes[bi]
            nt = top_add(B.sl, B.top, DEV[x])
            if B.nom + D[x] + nt > Q:
                continue
            j = rng.randrange(len(B.seq) + 1)
            sa = A.seq[:]
            sa.pop(i)
            routes[ai] = build_route(sa)
            sb = B.seq[:]
            sb.insert(j, x)
            routes[bi] = build_route(sb)
        elif u < 0.9:
            if len(nonempty) < 2:
                continue
            ai, bi = rng.sample(nonempty, 2)
            A = routes[ai]
            B = routes[bi]
            i = rng.randrange(len(A.seq))
            j = rng.randrange(len(B.seq))
            x = A.seq[i]
            y = B.seq[j]
            tA = top_swap(A.sl, A.top, DEV[x], DEV[y])
            if A.nom - D[x] + D[y] + tA > Q:
                continue
            tB = top_swap(B.sl, B.top, DEV[y], DEV[x])
            if B.nom - D[y] + D[x] + tB > Q:
                continue
            sa = A.seq[:]
            sb = B.seq[:]
            sa[i] = y
            sb[j] = x
            routes[ai] = build_route(sa)
            routes[bi] = build_route(sb)
        else:
            ai = rng.choice(nonempty)
            A = routes[ai]
            L = len(A.seq)
            if L < 2:
                continue
            i = rng.randrange(L - 1)
            j = rng.randrange(i + 1, L)
            s = A.seq
            routes[ai] = build_route(s[:i] + s[i:j + 1][::-1] + s[j + 1:])
        nonempty = [i2 for i2 in range(R) if len(routes[i2].seq) > 0]
    return routes


# ------------------------- constructors -------------------------
def nn_tour(start_node):
    unvis = set(range(n))
    unvis.discard(start_node)
    tour = [start_node]
    row = CM[start_node]
    while unvis:
        nxt = min(unvis, key=row.__getitem__)
        tour.append(nxt)
        unvis.remove(nxt)
        row = CM[nxt]
    return tour


def split_tour(tour, end_time):
    """Optimal split of a giant tour into exactly K robust-feasible routes (DP)."""
    nn = len(tour)
    if nn == 0 or K > nn:
        return None
    seg_cost = [None] * nn
    jmax = [-1] * nn
    for i in range(nn):
        if (i & 31) == 0 and time.time() > end_time:
            return None
        costs = []
        nom = 0
        top = 0
        big = []
        rest = []
        cost = 0
        jm = i - 1
        for j in range(i, nn):
            x = tour[j]
            nom += D[x]
            if Gamma > 0:
                v = DEV[x]
                heapq.heappush(big, v)
                top += v
                if len(big) > Gamma:
                    m = heapq.heappop(big)
                    top -= m
                    heapq.heappush(rest, -m)
                w = nom + top
            else:
                w = nom
            if w > Q:
                break  # infeasibility is monotone in segment length
            if j == i:
                cost = C0[x] + CR[x]
            else:
                p = tour[j - 1]
                cost += CM[p][x] + CR[x] - CR[p]
            costs.append(cost)
            jm = j
        if jm < i:
            return None  # a customer that cannot even be alone -> infeasible
        seg_cost[i] = costs
        jmax[i] = jm
    INF = float("inf")
    Dp = [[INF] * (nn + 1) for _ in range(K + 1)]
    par = [[-1] * (nn + 1) for _ in range(K + 1)]
    Dp[0][0] = 0
    for k in range(1, K + 1):
        Dk = Dp[k]
        Dkm = Dp[k - 1]
        pk = par[k]
        hi = nn - (K - k)
        for j in range(k, hi + 1):
            best = INF
            bi = -1
            i = j - 1
            while i >= k - 1 and j - 1 <= jmax[i]:
                c = Dkm[i] + seg_cost[i][j - 1 - i]
                if c < best:
                    best = c
                    bi = i
                i -= 1
            Dk[j] = best
            pk[j] = bi
    if Dp[K][nn] == INF:
        return None
    seqs = []
    j = nn
    for k in range(K, 0, -1):
        i = par[k][j]
        seqs.append(tour[i:j])
        j = i
    seqs.reverse()
    return [build_route(s) for s in seqs]


def clarke_wright(end_time):
    if n == 0:
        return []
    sav = []
    for i in range(n):
        ri = CR[i]
        row = CM[i]
        for j in range(n):
            if i != j:
                sav.append((-(ri + C0[j] - row[j]), i, j))
    heapq.heapify(sav)
    seqs = [[i] for i in range(n)]
    last = list(range(n))
    rid_of_first = list(range(n))
    rid_of_last = list(range(n))
    nom = [D[i] for i in range(n)]
    sl = [[DEV[i]] for i in range(n)]
    active = [True] * n
    pops = 0
    while sav:
        _, i, j = heapq.heappop(sav)
        pops += 1
        if (pops & 8191) == 0 and time.time() > end_time:
            break
        ra = rid_of_last[i]
        rb = rid_of_first[j]
        if ra < 0 or rb < 0 or ra == rb:
            continue
        wnom = nom[ra] + nom[rb]
        base = sl[ra] + sl[rb]
        if Gamma > 0 and len(base) > Gamma:
            mt = sum(heapq.nlargest(Gamma, base))
        else:
            mt = sum(base)
        if wnom + mt > Q:
            continue
        seqs[ra] = seqs[ra] + seqs[rb]
        sl[ra] = base
        nom[ra] = wnom
        rid_of_last[i] = -1
        rid_of_first[j] = -1
        nl = last[rb]
        last[ra] = nl
        rid_of_last[nl] = ra
        active[rb] = False
    return [build_route(seqs[r]) for r in range(n) if active[r]]


def dissolve_smallest(routes):
    idx = min(range(len(routes)), key=lambda t: len(routes[t].seq))
    src = routes[idx]
    routes.pop(idx)
    for x in src.seq:
        bestd = None
        bestwhere = None
        for bi in range(len(routes)):
            B = routes[bi]
            nt = top_add(B.sl, B.top, DEV[x])
            if B.nom + D[x] + nt > Q:
                continue
            for j in range(len(B.seq) + 1):
                d = insertion_delta(B.seq, j, x)
                if bestd is None or d < bestd:
                    bestd = d
                    bestwhere = (bi, j)
        if bestwhere is None:
            return False
        bi, j = bestwhere
        s = routes[bi].seq[:]
        s.insert(j, x)
        routes[bi] = build_route(s)
    return True


def adjust_route_count(routes, end_time):
    # reduce to K routes by best feasible merges
    while len(routes) > K:
        if time.time() > end_time:
            return None
        bestsav = None
        bestpair = None
        R = len(routes)
        for a in range(R):
            A = routes[a]
            for b in range(R):
                if a == b:
                    continue
                B = routes[b]
                wnom = A.nom + B.nom
                base = A.sl + B.sl
                if Gamma > 0 and len(base) > Gamma:
                    mt = sum(heapq.nlargest(Gamma, base))
                else:
                    mt = sum(base)
                if wnom + mt > Q:
                    continue
                savg = CR[A.seq[-1]] + C0[B.seq[0]] - CM[A.seq[-1]][B.seq[0]]
                if bestsav is None or savg > bestsav:
                    bestsav = savg
                    bestpair = (a, b)
        if bestpair is None:
            if not dissolve_smallest(routes):
                return None
        else:
            a, b = bestpair
            nr = build_route(routes[a].seq + routes[b].seq)
            for idx in sorted((a, b), reverse=True):
                routes.pop(idx)
            routes.append(nr)
    # increase to K routes by cheapest splits (subsets stay feasible)
    while len(routes) < K:
        best = None
        for idx in range(len(routes)):
            r = routes[idx]
            s = r.seq
            L = len(s)
            if L < 2:
                continue
            for t in range(1, L):
                ca = route_cost_of_seq(s[:t])
                cb = route_cost_of_seq(s[t:])
                inc = ca + cb - r.cost
                if best is None or inc < best[0]:
                    best = (inc, idx, t)
        if best is None:
            return None
        _, idx, t = best
        s = routes.pop(idx).seq
        routes.append(build_route(s[:t]))
        routes.append(build_route(s[t:]))
    return routes


def greedy_k_routes():
    routes = [build_route([]) for _ in range(K)]
    order = sorted(range(n), key=lambda i: -(D[i] + DEV[i]))
    for x in order:
        bestd = None
        bestwhere = None
        for bi in range(K):
            B = routes[bi]
            nt = top_add(B.sl, B.top, DEV[x])
            if B.nom + D[x] + nt > Q:
                continue
            sb = B.seq
            for j in range(len(sb) + 1):
                d = insertion_delta(sb, j, x)
                if bestd is None or d < bestd:
                    bestd = d
                    bestwhere = (bi, j)
        if bestwhere is None:
            bi = min(range(K), key=lambda b: routes[b].nom + routes[b].top)
            s = routes[bi].seq[:]
            s.append(x)
            routes[bi] = build_route(s)
        else:
            bi, j = bestwhere
            s = routes[bi].seq[:]
            s.insert(j, x)
            routes[bi] = build_route(s)
    return routes


# ------------------------- set partitioning polish -------------------------
def gurobi_polish(pool, time_left):
    import gurobipy as gp
    from gurobipy import GRB

    if len(pool) > 30000:
        items = sorted(pool.items(), key=lambda kv: kv[1])[:30000]
        pool = dict(items)
    keys = list(pool.keys())
    costs = [pool[k] for k in keys]
    m = gp.Model("robust_cvrp_spp")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1.0, time_left)
    y = m.addVars(len(keys), vtype=GRB.BINARY, obj=costs, name="y")
    cust = [[] for _ in range(n)]
    for t, key in enumerate(keys):
        for x in key:
            cust[x].append(t)
    for i in range(n):
        m.addConstr(gp.quicksum(y[t] for t in cust[i]) == 1)
    m.addConstr(y.sum() == K)
    m.optimize()
    if m.SolCount > 0:
        sel = [keys[t] for t in range(len(keys)) if y[t].X > 0.5]
        if len(sel) == K:
            return [build_route(list(s)) for s in sel]
    return None


# ------------------------- output -------------------------
def make_output(seqs, obj):
    routes_out = []
    for s in seqs:
        r = build_route(s)
        c = r.cost
        if isinstance(c, float) and c.is_integer():
            c = int(c)
        w = r.nom + r.top
        if isinstance(w, float) and w.is_integer():
            w = int(w)
        routes_out.append({
            "customers": [IDS[x] for x in s],
            "cost": c,
            "worst_case_demand": w,
        })
    return {"objective_value": float(obj), "routes": routes_out}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    global n, K, Q, Gamma, D, DEV, C0, CR, CM, IDS, DEPOT

    args = parse_args()
    with open(args.instance_path) as f:
        inst = json.load(f)

    customers = inst["customers"]
    n = len(customers)
    K = int(inst["num_vehicles"])
    Q = inst["vehicle_capacity"]
    Gamma = int(inst["uncertainty_set"]["Gamma"])
    depot = inst["depot"]
    DEPOT = depot.get("id", 0)
    dist = inst["distance_matrix"]

    IDS = [c["id"] for c in customers]
    D = [c["nominal_demand"] for c in customers]
    DEV = [max(0, c.get("demand_deviation", 0)) for c in customers]
    C0 = [dist[DEPOT][cid] for cid in IDS]
    CR = [dist[cid][DEPOT] for cid in IDS]
    CM = [[dist[a][b] for b in IDS] for a in IDS]
    XS = [c["x"] for c in customers]
    YS = [c["y"] for c in customers]
    XD, YD = depot["x"], depot["y"]

    start = time.time()
    total = max(1, args.time_limit)
    margin = 0.4
    hard_end = start + total - margin
    if total >= 25:
        polish_time = 10.0
    elif total >= 12:
        polish_time = 4.0
    else:
        polish_time = 0.0
    ls_end = hard_end - polish_time

    rng = random.Random(0)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    best_obj = float("inf")
    best_seqs = None
    pool = {}
    POOL_CAP = 60000

    def consider(routes):
        nonlocal best_obj, best_seqs
        if routes is None or len(routes) != K:
            return
        obj = 0
        for r in routes:
            obj += r.cost
            if len(pool) < POOL_CAP:
                key = tuple(r.seq)
                if key and key not in pool:
                    pool[key] = r.cost
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_seqs = [r.seq[:] for r in routes]
            if logger:
                logger.log_solution(float(obj), make_output(best_seqs, obj))

    # ---- 1) guaranteed baseline: greedy feasible insertion + local search ----
    base = greedy_k_routes()
    consider(base)
    if time.time() < ls_end:
        local_search(base, ls_end)
        consider(base)

    # ---- 2) Clarke-Wright savings ----
    if n > 0 and time.time() < ls_end:
        try:
            cw = clarke_wright(ls_end)
            if cw:
                cw = adjust_route_count(cw, ls_end)
                if cw and len(cw) == K:
                    consider(cw)
                    local_search(cw, ls_end)
                    consider(cw)
        except Exception:
            pass

    # ---- 3) giant tours + exact split ----
    if n > 0 and K <= n and time.time() < ls_end:
        try:
            tours = []
            seen = set()

            def add_tour(t):
                key = tuple(t)
                if key not in seen:
                    seen.add(key)
                    tours.append(list(t))

            add_tour(nn_tour(0))
            far = max(range(n), key=lambda i: C0[i] + CR[i])
            add_tour(nn_tour(far))
            if n >= 2:
                ang = [math.atan2(YS[i] - YD, XS[i] - XD) for i in range(n)]
                order = sorted(range(n), key=lambda i: ang[i])
                for o in sorted({0, n // 4, n // 2, (3 * n) // 4}):
                    t = order[o:] + order[:o]
                    add_tour(t)
                    add_tour(list(reversed(t)))
            for _ in range(2):
                t = list(range(n))
                rng.shuffle(t)
                add_tour(t)

            scored = []
            for t in tours:
                if time.time() >= ls_end:
                    break
                r = split_tour(t, ls_end)
                if r:
                    scored.append((sum(x.cost for x in r), r))
            scored.sort(key=lambda z: z[0])
            for _, r in scored[:4]:
                consider(r)
                if time.time() >= ls_end:
                    break
                local_search(r, ls_end)
                consider(r)
        except Exception:
            pass

    # ---- 4) iterated local search ----
    it = 0
    while time.time() < ls_end and best_seqs is not None:
        try:
            cand = [build_route(s) for s in best_seqs]
            strength = max(6, n // 6) if it % 5 == 4 else max(3, n // 12)
            perturb(cand, rng, strength)
            local_search(cand, ls_end)
            consider(cand)
        except Exception:
            break
        it += 1

    # ---- 5) Gurobi set-partitioning polish over the collected route pool ----
    tl = hard_end - time.time()
    if pool and tl > 3.0 and n >= K:
        try:
            pr = gurobi_polish(pool, max(1.0, tl - 0.3))
            if pr:
                consider(pr)
        except Exception:
            pass

    sol = make_output(best_seqs, best_obj)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()