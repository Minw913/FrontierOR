import argparse
import json
import math
import random
import time

from solution_logger import SolutionLogger

EPS = 1e-7
IMPR = 1e-7

# ------------------------- global instance data -------------------------
D = None          # distance matrix, node 0 = depot, 1..n = customers
DEM = None
READY = None
DUE = None
SVC = None
CAP = 0
DEP_RDY = 0.0
DEP_DUE = 0.0
N = 0             # number of nodes incl. depot
CUST_IDS = []
DEPOT_ID = 0
DMAX = 1.0
TSPAN = 1.0
DEMAX = 1.0


class Route:
    __slots__ = ("nodes", "cost", "load", "arr", "start", "dep", "slack")

    def __init__(self, nodes=None):
        self.nodes = list(nodes) if nodes else []
        self.cost = 0.0
        self.load = 0
        self.arr = []
        self.start = []
        self.dep = []
        self.slack = []

    def recalc(self):
        nodes = self.nodes
        L = len(nodes)
        t = DEP_RDY
        prev = 0
        cost = 0.0
        load = 0
        arr = [0.0] * L
        start = [0.0] * L
        dep = [0.0] * L
        for i in range(L):
            c = nodes[i]
            a = t + D[prev][c]
            rc = READY[c]
            st = a if a > rc else rc
            dp = st + SVC[c]
            arr[i] = a
            start[i] = st
            dep[i] = dp
            cost += D[prev][c]
            load += DEM[c]
            t = dp
            prev = c
        ret = t + D[prev][0]
        cost += D[prev][0]
        self.arr = arr
        self.start = start
        self.dep = dep
        self.cost = cost
        self.load = load
        # forward time slack: max delay at each position keeping route feasible
        slack = [0.0] * (L + 1)
        slack[L] = DEP_DUE - ret
        for i in range(L - 1, -1, -1):
            c = nodes[i]
            lim = DUE[c] - start[i]
            if i + 1 < L:
                s2 = (start[i + 1] - arr[i + 1]) + slack[i + 1]
            else:
                s2 = slack[i + 1]
            slack[i] = lim if lim < s2 else s2
        self.slack = slack


def eval_list(nodes):
    """Simulate a node sequence; return travel cost or None if infeasible."""
    t = DEP_RDY
    prev = 0
    load = 0
    cost = 0.0
    for c in nodes:
        a = t + D[prev][c]
        rc = READY[c]
        st = a if a > rc else rc
        if st > DUE[c] + EPS:
            return None
        load += DEM[c]
        if load > CAP:
            return None
        t = st + SVC[c]
        cost += D[prev][c]
        prev = c
    a = t + D[prev][0]
    if a > DEP_DUE + EPS:
        return None
    return cost + D[prev][0]


def single_route_cost(u):
    a = DEP_RDY + D[0][u]
    st = a if a > READY[u] else READY[u]
    if st > DUE[u] + EPS:
        return None
    ret = st + SVC[u] + D[u][0]
    if ret > DEP_DUE + EPS:
        return None
    return D[0][u] + D[u][0]


def ins_eval(rt, u, p):
    """Cost delta of inserting customer u at position p of route rt, or None."""
    if rt.load + DEM[u] > CAP:
        return None
    nodes = rt.nodes
    if p > 0:
        prev = nodes[p - 1]
        t = rt.dep[p - 1]
    else:
        prev = 0
        t = DEP_RDY
    a = t + D[prev][u]
    ru = READY[u]
    st = a if a > ru else ru
    if st > DUE[u] + EPS:
        return None
    dp = st + SVC[u]
    if p < len(nodes):
        nxt = nodes[p]
        an = dp + D[u][nxt]
        rn = READY[nxt]
        stn = an if an > rn else rn
        if stn > rt.start[p] + rt.slack[p] + EPS:
            return None
        return D[prev][u] + D[u][nxt] - D[prev][nxt]
    else:
        an = dp + D[u][0]
        if an > DEP_DUE + EPS:
            return None
        return D[prev][u] + D[u][0] - D[prev][0]


def best_ins_in_route(rt, u):
    best = None
    for p in range(len(rt.nodes) + 1):
        d = ins_eval(rt, u, p)
        if d is not None and (best is None or d < best[0]):
            best = (d, p)
    return best


def clone_routes(routes):
    out = []
    for rt in routes:
        r2 = Route(rt.nodes)
        r2.recalc()
        out.append(r2)
    return out


def total_cost(routes):
    return math.fsum(rt.cost for rt in routes)


def build_pos(routes):
    pos = {}
    for ri, rt in enumerate(routes):
        for i, u in enumerate(rt.nodes):
            pos[u] = (ri, i)
    return pos


# ----------------------------- destroy ops ------------------------------
def remove_nodes(routes, picks):
    by_route = {}
    for ri, i in picks:
        by_route.setdefault(ri, []).append(i)
    removed = []
    for ri, idxs in by_route.items():
        rt = routes[ri]
        nodes = rt.nodes
        for i in sorted(idxs, reverse=True):
            removed.append(nodes.pop(i))
        rt.recalc()
    routes[:] = [rt for rt in routes if rt.nodes]
    return removed


def all_positions(routes):
    out = []
    for ri, rt in enumerate(routes):
        for i in range(len(rt.nodes)):
            out.append((ri, i))
    return out


def random_removal(routes, q, rng):
    pos = all_positions(routes)
    if not pos:
        return []
    q = min(q, len(pos))
    return remove_nodes(routes, rng.sample(pos, q))


def worst_removal(routes, q, rng):
    arr = []
    for ri, rt in enumerate(routes):
        nodes = rt.nodes
        L = len(nodes)
        for i, u in enumerate(nodes):
            prev = nodes[i - 1] if i > 0 else 0
            nxt = nodes[i + 1] if i + 1 < L else 0
            c = D[prev][u] + D[u][nxt] - D[prev][nxt]
            arr.append((c + rng.random() * 1e-6, ri, i))
    arr.sort(reverse=True)
    picks = [(ri, i) for _, ri, i in arr[:q]]
    return remove_nodes(routes, picks)


def shaw_removal(routes, q, rng):
    pos = all_positions(routes)
    if not pos:
        return []
    sri, si = pos[rng.randrange(len(pos))]
    seed = routes[sri].nodes[si]
    rs = READY[seed]
    ds = DEM[seed]
    Ds = D[seed]
    arr = []
    for ri, rt in enumerate(routes):
        for i, u in enumerate(rt.nodes):
            r = (0.6 * Ds[u] / DMAX
                 + 0.3 * abs(rs - READY[u]) / TSPAN
                 + 0.1 * abs(ds - DEM[u]) / DEMAX
                 + rng.random() * 1e-6)
            arr.append((r, ri, i))
    arr.sort()
    picks = [(ri, i) for _, ri, i in arr[:q]]
    return remove_nodes(routes, picks)


# ----------------------------- repair ops -------------------------------
def repair(routes, removed, k, rng, deadline):
    if not removed:
        return
    ins = {}
    for u in removed:
        ins[u] = [best_ins_in_route(rt, u) for rt in routes]
    remaining = list(removed)
    while remaining:
        if time.time() > deadline:
            for u in remaining:
                rt = Route([u])
                rt.recalc()
                routes.append(rt)
            return
        best_key = None
        best_u = None
        best_b = None
        for u in remaining:
            entries = ins[u]
            opts = []
            for ri in range(len(entries)):
                e = entries[ri]
                if e is not None:
                    opts.append((e[0], ri, e[1]))
            opts.sort()
            nc = single_route_cost(u)
            if nc is None:
                nc = D[0][u] + D[u][0] + 1e6
            if opts:
                b0 = opts[0]
                bc = b0[0]
            else:
                b0 = None
                bc = nc
            if k <= 1:
                metric = -bc
            else:
                m = 0.0
                for t in range(1, k):
                    if t < len(opts):
                        m += opts[t][0] - bc
                    else:
                        m += nc - bc
                metric = m
            key = (metric, -bc)
            if best_key is None or key > best_key:
                best_key = key
                best_u = u
                best_b = b0
        u = best_u
        if best_b is None:
            rt = Route([u])
            rt.recalc()
            routes.append(rt)
            for u2 in remaining:
                if u2 != u:
                    ins[u2].append(best_ins_in_route(rt, u2))
        else:
            ri = best_b[1]
            p = best_b[2]
            rt = routes[ri]
            rt.nodes.insert(p, u)
            rt.recalc()
            for u2 in remaining:
                if u2 != u:
                    ins[u2][ri] = best_ins_in_route(rt, u2)
        remaining.remove(u)
        del ins[u]


# ----------------------------- local search -----------------------------
def two_opt_route(rt, candset):
    if len(rt.nodes) < 3:
        return False
    improved_any = False
    for _ in range(200):
        nodes = rt.nodes
        L = len(nodes)
        best_gain = -IMPR
        best = None
        use_filter = L > 120
        for i in range(L - 1):
            pi = nodes[i - 1] if i > 0 else 0
            ni = nodes[i]
            dpi_ni = D[pi][ni]
            cs_i = candset[ni] if use_filter else None
            Dni = D[ni]
            for j in range(i + 1, L):
                nj = nodes[j]
                if use_filter and nj not in cs_i:
                    continue
                nx = nodes[j + 1] if j + 1 < L else 0
                gain = D[pi][nj] + Dni[nx] - dpi_ni - D[nj][nx]
                if gain < best_gain:
                    newn = nodes[:i] + nodes[i:j + 1][::-1] + nodes[j + 1:]
                    if eval_list(newn) is not None:
                        best_gain = gain
                        best = newn
        if best is None:
            break
        rt.nodes = best
        rt.recalc()
        improved_any = True
    return improved_any


def oropt_route(rt):
    nodes = rt.nodes
    L = len(nodes)
    if L < 3:
        return False
    best_gain = -IMPR
    best = None
    for i in range(L):
        u = nodes[i]
        prev = nodes[i - 1] if i > 0 else 0
        nxt = nodes[i + 1] if i + 1 < L else 0
        rem_gain = D[prev][u] + D[u][nxt] - D[prev][nxt]
        short = nodes[:i] + nodes[i + 1:]
        for p in range(L):
            if p == i:
                continue
            a = short[p - 1] if p > 0 else 0
            b = short[p] if p < L - 1 else 0
            gain = D[a][u] + D[u][b] - D[a][b] - rem_gain
            if gain < best_gain:
                newn = short[:p] + [u] + short[p:]
                if eval_list(newn) is not None:
                    best_gain = gain
                    best = newn
    if best is not None:
        rt.nodes = best
        rt.recalc()
        return True
    return False


def relocate_pass(routes, cand, deadline):
    pos = build_pos(routes)
    for u in range(1, N):
        if time.time() > deadline:
            return False
        pu = pos.get(u)
        if pu is None:
            continue
        ri, i = pu
        A = routes[ri]
        nodes = A.nodes
        prev = nodes[i - 1] if i > 0 else 0
        nxt = nodes[i + 1] if i + 1 < len(nodes) else 0
        rem_gain = D[prev][u] + D[u][nxt] - D[prev][nxt]
        best = None
        best_delta = -IMPR
        for v in cand[u]:
            pv = pos.get(v)
            if pv is None:
                continue
            rj, j = pv
            if rj == ri:
                continue
            B = routes[rj]
            e1 = ins_eval(B, u, j)
            if e1 is not None and e1 - rem_gain < best_delta:
                best_delta = e1 - rem_gain
                best = (rj, j)
            e2 = ins_eval(B, u, j + 1)
            if e2 is not None and e2 - rem_gain < best_delta:
                best_delta = e2 - rem_gain
                best = (rj, j + 1)
        if best is not None:
            rj, p = best
            B = routes[rj]
            A.nodes.pop(i)
            A.recalc()
            B.nodes.insert(p, u)
            B.recalc()
            if not A.nodes:
                routes.remove(A)
            return True
    return False


def swap_pass(routes, cand, deadline):
    pos = build_pos(routes)
    for u in range(1, N):
        if time.time() > deadline:
            return False
        pu = pos.get(u)
        if pu is None:
            continue
        ri, i = pu
        A = routes[ri]
        for v in cand[u]:
            if v == u:
                continue
            pv = pos.get(v)
            if pv is None:
                continue
            rj, j = pv
            if rj == ri:
                continue
            B = routes[rj]
            if A.load - DEM[u] + DEM[v] > CAP:
                continue
            if B.load - DEM[v] + DEM[u] > CAP:
                continue
            newA = A.nodes[:i] + [v] + A.nodes[i + 1:]
            ca = eval_list(newA)
            if ca is None:
                continue
            newB = B.nodes[:j] + [u] + B.nodes[j + 1:]
            cb = eval_list(newB)
            if cb is None:
                continue
            if ca + cb < A.cost + B.cost - IMPR:
                A.nodes = newA
                A.recalc()
                B.nodes = newB
                B.recalc()
                return True
    return False


def two_opt_star_pass(routes, cand, deadline):
    pos = build_pos(routes)
    for u in range(1, N):
        if time.time() > deadline:
            return False
        pu = pos.get(u)
        if pu is None:
            continue
        ri, i = pu
        A = routes[ri]
        for v in cand[u]:
            pv = pos.get(v)
            if pv is None:
                continue
            rj, j = pv
            if rj == ri:
                continue
            B = routes[rj]
            if i + 1 == len(A.nodes) and j + 1 == len(B.nodes):
                continue
            newA = A.nodes[:i + 1] + B.nodes[j + 1:]
            ca = eval_list(newA)
            if ca is None:
                continue
            newB = B.nodes[:j + 1] + A.nodes[i + 1:]
            cb = eval_list(newB)
            if cb is None:
                continue
            if ca + cb < A.cost + B.cost - IMPR:
                A.nodes = newA
                A.recalc()
                B.nodes = newB
                B.recalc()
                return True
    return False


def local_search(routes, cand, candset, deadline):
    while time.time() < deadline:
        improved = False
        for rt in list(routes):
            if time.time() > deadline:
                return
            if two_opt_route(rt, candset):
                improved = True
            if oropt_route(rt):
                improved = True
        if time.time() > deadline:
            return
        if relocate_pass(routes, cand, deadline):
            improved = True
        if two_opt_star_pass(routes, cand, deadline):
            improved = True
        if swap_pass(routes, cand, deadline):
            improved = True
        if not improved:
            return


# -------------------------------- ALNS ----------------------------------
def roulette(w, rng):
    s = w[0] + w[1] + w[2]
    r = rng.random() * s
    acc = 0.0
    for i, x in enumerate(w):
        acc += x
        if r <= acc:
            return i
    return len(w) - 1


def alns(routes0, cand, candset, deadline, rng, logger, t0, last_logged):
    n = N - 1
    best = clone_routes(routes0)
    best_obj = total_cost(best)
    current = clone_routes(routes0)
    cur_obj = best_obj
    destroys = [random_removal, worst_removal, shaw_removal]
    regk = [1, 2, 3]
    wd = [1.0, 1.0, 1.0]
    wr = [1.0, 1.0, 1.0]
    sd = [0.0, 0.0, 0.0]
    sr = [0.0, 0.0, 0.0]
    ud = [0, 0, 0]
    ur = [0, 0, 0]
    T0 = max(1e-6, 0.03 * best_obj)
    Tend = T0 * 0.001
    qmin = max(2, int(0.05 * n))
    qmax = max(3, min(int(0.30 * n), 150))
    it = 0
    while True:
        now = time.time()
        if now >= deadline:
            break
        it += 1
        frac = (now - t0) / max(1e-9, deadline - t0)
        if frac > 1.0:
            frac = 1.0
        T = T0 * (Tend / T0) ** frac
        di = roulette(wd, rng)
        ki = roulette(wr, rng)
        trial = clone_routes(current)
        lo = qmin if qmin < n else n
        hi = qmax if qmax < n else n
        if lo > hi:
            lo = hi
        q = rng.randint(lo, hi)
        removed = destroys[di](trial, q, rng)
        if removed:
            repair(trial, removed, regk[ki], rng, deadline)
        obj = total_cost(trial)
        score = 0.0
        if obj < cur_obj - 1e-6:
            current = trial
            cur_obj = obj
            score = 2.0
            if obj < best_obj - 1e-6:
                local_search(trial, cand, candset, min(deadline, time.time() + 0.5))
                obj = total_cost(trial)
                current = trial
                cur_obj = obj
                if obj < best_obj - 1e-6:
                    best = clone_routes(trial)
                    best_obj = obj
                    score = 6.0
                    if logger and obj < last_logged - 1e-9:
                        logger.log_solution(obj, build_solution(best))
                        last_logged = obj
        elif obj < cur_obj + 1e-9:
            current = trial
            cur_obj = obj
            score = 0.5
        elif rng.random() < math.exp(-(obj - cur_obj) / T):
            current = trial
            cur_obj = obj
            score = 0.5
        sd[di] += score
        sr[ki] += score
        ud[di] += 1
        ur[ki] += 1
        if it % 50 == 0:
            for a in range(3):
                if ud[a]:
                    wd[a] = wd[a] * 0.8 + 0.2 * (sd[a] / ud[a])
                if ur[a]:
                    wr[a] = wr[a] * 0.8 + 0.2 * (sr[a] / ur[a])
            sd = [0.0, 0.0, 0.0]
            sr = [0.0, 0.0, 0.0]
            ud = [0, 0, 0]
            ur = [0, 0, 0]
        if it % 25 == 0:
            local_search(current, cand, candset, min(deadline, time.time() + 0.3))
            cur_obj = total_cost(current)
            if cur_obj < best_obj - 1e-6:
                best = clone_routes(current)
                best_obj = cur_obj
                if logger and cur_obj < last_logged - 1e-9:
                    logger.log_solution(cur_obj, build_solution(best))
                    last_logged = cur_obj
    return best, best_obj, last_logged


# ------------------------------ output ----------------------------------
def build_solution(routes):
    out = []
    for rt in routes:
        if not rt.nodes:
            continue
        cids = [CUST_IDS[u - 1] for u in rt.nodes]
        out.append({"customer_ids": cids,
                    "path": [DEPOT_ID] + cids + [DEPOT_ID]})
    return {"objective_value": float(total_cost(routes)), "routes": out}


# -------------------------------- main -----------------------------------
def main():
    global D, DEM, READY, DUE, SVC, CAP, DEP_RDY, DEP_DUE, N
    global CUST_IDS, DEPOT_ID, DMAX, TSPAN, DEMAX

    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    total = float(args.time_limit)
    margin = min(0.5, 0.05 * total + 0.05)
    end = t0 + total - margin

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    depot = inst["depot"]
    custs = inst.get("customers", [])
    n = len(custs)
    N = n + 1
    DEPOT_ID = depot["id"]
    CAP = inst["vehicle_capacity"]
    DEP_RDY = float(depot["ready_time"])
    DEP_DUE = float(depot["due_date"])

    xs = [float(depot["x"])]
    ys = [float(depot["y"])]
    DEM = [0]
    READY = [DEP_RDY]
    DUE = [DEP_DUE]
    SVC = [0.0]
    CUST_IDS = []
    for c in custs:
        xs.append(float(c["x"]))
        ys.append(float(c["y"]))
        DEM.append(c["demand"])
        READY.append(float(c["ready_time"]))
        DUE.append(float(c["due_date"]))
        SVC.append(float(c["service_time"]))
        CUST_IDS.append(c["id"])

    if n == 0:
        sol = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    D = [[0.0] * N for _ in range(N)]
    for i in range(N):
        xi = xs[i]
        yi = ys[i]
        Di = D[i]
        for j in range(N):
            Di[j] = math.hypot(xi - xs[j], yi - ys[j])

    DMAX = 0.0
    for i in range(1, N):
        row = D[i]
        for j in range(1, N):
            if row[j] > DMAX:
                DMAX = row[j]
    if DMAX <= 0:
        DMAX = 1.0
    rmin = min(READY[1:])
    rmax = max(READY[1:])
    TSPAN = rmax - rmin if rmax > rmin else 1.0
    DEMAX = float(max(DEM[1:])) if max(DEM[1:]) > 0 else 1.0

    K = min(20, n - 1)
    cand = [[] for _ in range(N)]
    for u in range(1, N):
        Du = D[u]
        order = sorted(range(1, N), key=lambda v: Du[v])
        cand[u] = [v for v in order if v != u][:K]
    candset = [set(c) for c in cand]

    rng = random.Random(0)

    # ---- initial construction: regret-3 insertion ----
    routes = []
    order = list(range(1, N))
    rng.shuffle(order)
    repair(routes, order, 3, rng, end)
    local_search(routes, cand, candset,
                 min(end, time.time() + max(0.5, 0.15 * total)))

    best = clone_routes(routes)
    best_obj = total_cost(best)
    last_logged = best_obj
    if logger:
        logger.log_solution(best_obj, build_solution(best))

    # ---- ALNS ----
    if total > 5:
        alns_end = min(end, t0 + (total - margin) * 0.90)
    else:
        alns_end = end
    if time.time() < alns_end:
        best, best_obj, last_logged = alns(best, cand, candset, alns_end,
                                           rng, logger, t0, last_logged)

    # ---- final intensification ----
    if time.time() < end:
        local_search(best, cand, candset, end)
        o = total_cost(best)
        if o < best_obj - 1e-9:
            best_obj = o
            if logger and o < last_logged - 1e-9:
                logger.log_solution(o, build_solution(best))
                last_logged = o

    sol = build_solution(best)
    sol["objective_value"] = float(best_obj)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()