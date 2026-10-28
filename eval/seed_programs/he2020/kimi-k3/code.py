"""VRPTW with quadratic inconvenience (deviation from preferred service time).

Strategy
--------
* Exact per-route schedule evaluation: for a fixed route sequence the optimal
  service-start schedule minimising sum rho*(t_i - p_i)^2 under chain
  precedence constraints and time windows is reduced (via the substitution
  u_i = t_i - cumulative_gap) to a uniform-weight isotonic regression with
  box constraints. This is solved exactly with PAVA followed by clipping
  (depot bounds are folded into customer bounds, depots carry no penalty).
* A fast greedy/regret construction, local search (relocate, swap, 2-opt,
  or-opt) and an ALNS (random/worst/Shaw removal + regret-k insertion,
  simulated-annealing acceptance) drive the heuristic search.
* For small instances the heuristic incumbent is used as a MIP start for an
  exact MIQP model in Gurobi (per-vehicle arc formulation with big-M timing),
  which can close the optimality gap. The exact schedule evaluator is reused
  to score Gurobi incumbents consistently.
"""

import argparse
import json
import math
import random
import time


INF = float("inf")


# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self, data):
        self.n = int(data["num_customers"])
        self.N = self.n + 1  # return depot index
        self.K = int(data["num_vehicles"])
        self.Q = float(data["vehicle_capacity"])

        n, N = self.n, self.N
        e = [0.0] * (N + 1)
        l = [0.0] * (N + 1)
        s = [0.0] * (N + 1)
        p = [0.0] * (N + 1)
        dem = [0.0] * (N + 1)

        for node in [data["depot"]] + list(data["customers"]) + [data["depot_end"]]:
            i = int(node["id"])
            tw = node["time_window"]
            e[i] = float(tw[0])
            l[i] = float(tw[1])
            s[i] = float(node.get("service_time", 0) or 0)
            pt = node.get("preferred_time", None)
            p[i] = float(pt) if pt is not None else 0.5 * (e[i] + l[i])
            dem[i] = float(node.get("demand", 0) or 0)

        self.e = e
        self.l = l
        self.s = s
        self.p = p
        self.dem = dem
        self.dist = [row for row in data["distance_matrix"]]
        icf = data.get("inconvenience_cost_function", {}) or {}
        self.rho = float(icf.get("rho", 1.0))

        # constants for Shaw relatedness
        md = 1.0
        ts = 1.0
        for a in range(N + 1):
            for b in range(N + 1):
                if self.dist[a][b] > md:
                    md = self.dist[a][b]
        for a in range(N + 1):
            v = abs(self.l[a] - self.e[a])
            if v > ts:
                ts = v
        self.max_d = md
        self.max_t = ts


# ------------------------------------------- exact schedule on a route
def _pava_isotonic(q):
    """Unweighted isotonic regression (nondecreasing) via PAVA."""
    out = []
    for i, qi in enumerate(q):
        out.append([1.0, qi, qi, i, i])  # count, sum, value, start, end
        while len(out) >= 2 and out[-2][2] > out[-1][2] + 1e-10:
            a = out[-2]
            b = out[-1]
            cnt = a[0] + b[0]
            ssum = a[1] + b[1]
            out[-2] = [cnt, ssum, ssum / cnt, a[3], b[4]]
            out.pop()
    res = [0.0] * len(q)
    for _, _, val, s_, e_ in out:
        for i in range(s_, e_ + 1):
            res[i] = val
    return res


def make_evaluators(ctx):
    dist = ctx.dist
    e = ctx.e
    l = ctx.l
    s = ctx.s
    p = ctx.p
    rho = ctx.rho
    N = ctx.N
    e0 = e[0]
    lN = l[N]
    cache = {}

    def compute(seq, want_times):
        """Return (total, inc, times) or None if infeasible."""
        m = len(seq)
        if m == 0:
            return (0.0, 0.0, []) if want_times else (0.0, 0.0)
        qs = [0.0] * m
        lo = [0.0] * m
        up = [0.0] * m
        D = [0.0] * m
        Dcum = 0.0
        prev = 0
        first = seq[0]
        arc = dist[0][first]
        for i in range(m):
            node = seq[i]
            Dcum += s[prev] + dist[prev][node]
            D[i] = Dcum
            a = e[node] - Dcum
            if a < e0:
                a = e0
            b = l[node] - Dcum
            if b > lN:
                b = lN
            if a > b + 1e-9:
                return None
            lo[i] = a
            up[i] = b
            qs[i] = p[node] - Dcum
            if i + 1 < m:
                arc += dist[node][seq[i + 1]]
            prev = node
        arc += dist[seq[-1]][N]
        # feasibility of monotone + box
        cur = lo[0]
        for i in range(1, m):
            if lo[i] > cur:
                cur = lo[i]
            if cur > up[i] + 1e-9:
                return None
        u = _pava_isotonic(qs)
        inc = 0.0
        times = []
        for i in range(m):
            v = u[i]
            if v < lo[i]:
                v = lo[i]
            elif v > up[i]:
                v = up[i]
            dv = v - qs[i]
            inc += rho * dv * dv
            if want_times:
                times.append(v + D[i])
        total = arc + inc
        return (total, inc, times) if want_times else (total, inc)

    def eval_cost(seq):
        key = tuple(seq)
        if key in cache:
            return cache[key]
        if len(cache) > 400000:
            cache.clear()
        res = compute(seq, False)
        val = res[0] if res is not None else INF
        cache[key] = val
        return val

    def detail(seq):
        res = compute(seq, True)
        if res is None:
            return None
        return res[2], res[0]

    return eval_cost, detail


def routes_total(eval_cost, routes):
    tot = 0.0
    for r in routes:
        v = eval_cost(r)
        if v == INF:
            return INF
        tot += v
    return tot


# ---------------------------------------------------------------- output
def build_solution(ctx, detail, routes):
    total = 0.0
    sched = {}
    routes_out = []
    for seq in routes:
        if not seq:
            continue
        res = detail(seq)
        if res is None:
            return None
        times, rc = res
        total += rc
        for node, tt in zip(seq, times):
            sched[str(node)] = float(tt)
        routes_out.append([0] + list(seq) + [ctx.N])
    return {
        "objective_value": float(total),
        "routes": routes_out,
        "schedule": sched,
    }


# ------------------------------------------------------- local operators
def op_relocate(ctx, eval_cost, routes):
    R = len(routes)
    costs = [eval_cost(r) for r in routes]
    for i in range(R):
        si = routes[i]
        ci = costs[i]
        for pos in range(len(si)):
            node = si[pos]
            rest = si[:pos] + si[pos + 1:]
            crest = eval_cost(rest) if rest else 0.0
            if crest == INF:
                continue
            for j in range(R):
                if j == i:
                    continue
                sj = routes[j]
                cj = costs[j]
                for pp in range(len(sj) + 1):
                    ns = sj[:pp] + [node] + sj[pp:]
                    v = eval_cost(ns)
                    if v == INF:
                        continue
                    if v + crest < ci + cj - 1e-9:
                        newr = []
                        for k in range(R):
                            if k == i:
                                if rest:
                                    newr.append(rest)
                            elif k == j:
                                newr.append(ns)
                            else:
                                newr.append(routes[k])
                        if len(newr) <= ctx.K:
                            routes[:] = newr
                            return True
            if R < ctx.K:
                v = eval_cost([node])
                if v < ci - 1e-9:
                    newr = [r for k, r in enumerate(routes) if k != i and r]
                    newr.append([node])
                    routes[:] = newr
                    return True
    return False


def op_swap(ctx, eval_cost, routes):
    R = len(routes)
    costs = [eval_cost(r) for r in routes]
    for i in range(R):
        si = routes[i]
        ci = costs[i]
        for a in range(len(si)):
            na = si[a]
            for j in range(i + 1, R):
                sj = routes[j]
                cj = costs[j]
                for b in range(len(sj)):
                    ni = si[:a] + [sj[b]] + si[a + 1:]
                    nj = sj[:b] + [na] + sj[b + 1:]
                    vi = eval_cost(ni)
                    if vi == INF:
                        continue
                    vj = eval_cost(nj)
                    if vj == INF:
                        continue
                    if vi + vj < ci + cj - 1e-9:
                        routes[i] = ni
                        routes[j] = nj
                        return True
    return False


def op_twoopt(ctx, eval_cost, routes):
    for i in range(len(routes)):
        seq = routes[i]
        base = eval_cost(seq)
        m = len(seq)
        for a in range(m - 1):
            for b in range(a + 1, m):
                cand = seq[:a] + seq[a:b + 1][::-1] + seq[b + 1:]
                v = eval_cost(cand)
                if v < base - 1e-9:
                    routes[i] = cand
                    return True
    return False


def op_or2(ctx, eval_cost, routes):
    R = len(routes)
    costs = [eval_cost(r) for r in routes]
    for i in range(R):
        si = routes[i]
        ci = costs[i]
        for pos in range(len(si) - 1):
            seg = si[pos:pos + 2]
            rest = si[:pos] + si[pos + 2:]
            crest = eval_cost(rest) if rest else 0.0
            if crest == INF:
                continue
            for j in range(R):
                if j == i:
                    continue
                sj = routes[j]
                cj = costs[j]
                for pp in range(len(sj) + 1):
                    ns = sj[:pp] + seg + sj[pp:]
                    v = eval_cost(ns)
                    if v == INF:
                        continue
                    if v + crest < ci + cj - 1e-9:
                        newr = []
                        for k in range(R):
                            if k == i:
                                if rest:
                                    newr.append(rest)
                            elif k == j:
                                newr.append(ns)
                            else:
                                newr.append(routes[k])
                        routes[:] = newr
                        return True
    return False


def local_search(ctx, eval_cost, routes, deadline):
    routes[:] = [r for r in routes if r]
    while time.time() < deadline:
        if op_relocate(ctx, eval_cost, routes):
            continue
        if op_swap(ctx, eval_cost, routes):
            continue
        if op_twoopt(ctx, eval_cost, routes):
            continue
        if op_or2(ctx, eval_cost, routes):
            continue
        break
    return routes


# ---------------------------------------------------- destroy / repair
def destroy_random(ctx, rng, routes, r):
    pos = [(i, j) for i, seq in enumerate(routes) for j in range(len(seq))]
    if not pos:
        return [x[:] for x in routes], []
    r = min(r, len(pos))
    pick = rng.sample(pos, r)
    removed_set = set()
    removed = []
    for i, j in pick:
        removed.append(routes[i][j])
        removed_set.add(routes[i][j])
    newroutes = [[c for c in seq if c not in removed_set] for seq in routes]
    return [s for s in newroutes if s], removed


def destroy_worst(ctx, eval_cost, rng, routes, r):
    cands = []
    for seq in routes:
        base = eval_cost(seq)
        if len(seq) == 1:
            cands.append((base, seq[0]))
        else:
            for pos, node in enumerate(seq):
                rest = seq[:pos] + seq[pos + 1:]
                delta = base - eval_cost(rest)
                cands.append((delta, node))
    cands.sort(key=lambda x: x[0], reverse=True)
    removed = [node for _, node in cands[:r]]
    removed_set = set(removed)
    newroutes = [[c for c in seq if c not in removed_set] for seq in routes]
    return [s for s in newroutes if s], removed


def destroy_shaw(ctx, rng, routes, r):
    allc = [c for seq in routes for c in seq]
    if not allc:
        return [x[:] for x in routes], []
    seed = rng.choice(allc)
    md = ctx.max_d
    mt = ctx.max_t
    dist = ctx.dist
    e = ctx.e
    scored = sorted(allc, key=lambda c: dist[seed][c] / md + abs(e[c] - e[seed]) / mt)
    removed = scored[:r]
    removed_set = set(removed)
    newroutes = [[c for c in seq if c not in removed_set] for seq in routes]
    return [s for s in newroutes if s], removed


def repair_regret(ctx, eval_cost, rng, routes_in, removed, kmax):
    routes = [list(r) for r in routes_in]
    unplaced = list(removed)
    rng.shuffle(unplaced)
    K = ctx.K
    BIG = 1e12
    while unplaced:
        best_c = None
        best_act = None
        best_regret = -INF
        best_d1 = INF
        for c in unplaced:
            dels = []
            for ri in range(len(routes)):
                seq = routes[ri]
                base = eval_cost(seq)
                if base == INF:
                    continue
                for pp in range(len(seq) + 1):
                    ns = seq[:pp] + [c] + seq[pp:]
                    v = eval_cost(ns)
                    if v < INF:
                        dels.append((v - base, ri, pp, False))
            if len(routes) < K:
                v = eval_cost([c])
                if v < INF:
                    dels.append((v, len(routes), 0, True))
            if not dels:
                return None
            dels.sort(key=lambda x: x[0])
            d1 = dels[0][0]
            if len(dels) >= kmax:
                regret = dels[kmax - 1][0] - d1
            elif len(dels) > 1:
                regret = BIG + (dels[-1][0] - d1)
            else:
                regret = BIG - d1
            if regret > best_regret + 1e-9 or (
                abs(regret - best_regret) <= 1e-9 and d1 < best_d1 - 1e-9
            ):
                best_regret = regret
                best_d1 = d1
                best_c = c
                best_act = dels[0]
        if best_c is None:
            return None
        _, ri, pp, isnew = best_act
        if isnew:
            routes.append([best_c])
        else:
            routes[ri] = routes[ri][:pp] + [best_c] + routes[ri][pp:]
        unplaced.remove(best_c)
    return [r for r in routes if r]


# ---------------------------------------------------------------- ALNS
def alns(ctx, rng, eval_cost, detail, start_routes, best, logger, deadline):
    routes = [r[:] for r in start_routes if r]
    cur = routes_total(eval_cost, routes)
    if cur == INF:
        return
    if cur < best["cost"] - 1e-9:
        best["cost"] = cur
        best["routes"] = [r[:] for r in routes]
        sol = build_solution(ctx, detail, best["routes"])
        if logger and sol:
            logger.log_solution(cur, sol)

    init_cost = max(best["cost"], 1.0)
    T0 = 0.04 * init_cost

    remove_ops = ["random", "worst", "shaw"]
    repair_ops = [("reg2", 2), ("reg3", 3), ("gr", 1)]
    w_rm = {"random": 1.0, "worst": 1.0, "shaw": 1.0}
    w_rp = {"reg2": 1.0, "reg3": 1.0, "gr": 1.0}
    stat_rm = {k: [0.0, 0] for k in remove_ops}
    stat_rp = {k: [0.0, 0] for k, _ in repair_ops}

    t_start = time.time()
    budget = max(deadline - t_start, 1e-6)
    it = 0
    since_best = 0

    while time.time() < deadline:
        it += 1
        # choose operators
        tot = sum(w_rm[k] for k in remove_ops)
        pick = rng.random() * tot
        acc = 0.0
        rop = remove_ops[-1]
        for k in remove_ops:
            acc += w_rm[k]
            if pick <= acc:
                rop = k
                break
        names = [n for n, _ in repair_ops]
        tot = sum(w_rp[n] for n in names)
        pick = rng.random() * tot
        acc = 0.0
        rpair = repair_ops[-1]
        for pair in repair_ops:
            acc += w_rp[pair[0]]
            if pick <= acc:
                rpair = pair
                break

        n_all = sum(len(r) for r in routes)
        r = rng.randint(1, max(2, min(max(4, int(0.25 * n_all)), 40)))
        if rop == "random":
            cand, removed = destroy_random(ctx, rng, routes, r)
        elif rop == "worst":
            cand, removed = destroy_worst(ctx, eval_cost, rng, routes, r)
        else:
            cand, removed = destroy_shaw(ctx, rng, routes, r)

        repaired = repair_regret(ctx, eval_cost, rng, cand, removed, rpair[1])
        if repaired is None:
            stat_rm[rop][1] += 1
            stat_rp[rpair[0]][1] += 1
            continue

        ncost = routes_total(eval_cost, repaired)
        if ncost == INF:
            continue
        delta = ncost - cur
        frac = min(1.0, (time.time() - t_start) / budget)
        temp = max(T0 * (1.0 - frac), 1e-6)
        accept = delta <= 0 or rng.random() < math.exp(-delta / temp)

        scr = 0.0
        if accept:
            routes = [x[:] for x in repaired]
            cur = ncost
            scr = 3.0 if delta < -1e-9 else 1.0
            if ncost < best["cost"] - 1e-9:
                best["cost"] = ncost
                best["routes"] = [x[:] for x in repaired]
                since_best = 0
                scr = 10.0
                sol = build_solution(ctx, detail, best["routes"])
                if logger and sol:
                    logger.log_solution(ncost, sol)
        stat_rm[rop][0] += scr
        stat_rm[rop][1] += 1
        stat_rp[rpair[0]][0] += scr
        stat_rp[rpair[0]][1] += 1

        since_best += 1
        if it % 60 == 0:
            for d, w in ((stat_rm, w_rm), (stat_rp, w_rp)):
                for k, (s_, c_) in d.items():
                    if c_ > 0:
                        w[k] = 0.6 * w[k] + 0.4 * (s_ / c_)
                    d[k] = [0.0, 0]
        if since_best >= 400:
            local_search(ctx, eval_cost, routes, time.time() + 0.25)
            cur = routes_total(eval_cost, routes)
            since_best = 0
            if cur < best["cost"] - 1e-9:
                best["cost"] = cur
                best["routes"] = [x[:] for x in routes]
                sol = build_solution(ctx, detail, best["routes"])
                if logger and sol:
                    logger.log_solution(cur, sol)


# -------------------------------------------------------------- Gurobi
def gurobi_phase(ctx, eval_cost, detail, best, logger, deadline):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return
    rem = deadline - time.time() - 0.4
    if rem < 2.0:
        return

    n, N, K = ctx.n, ctx.N, ctx.K
    Keff = min(K, n)
    dist = ctx.dist
    e = ctx.e
    l = ctx.l
    s = ctx.s
    p = ctx.p
    rho = ctx.rho
    dem = ctx.dem

    arcs = [(i, j) for i in range(N) for j in range(1, N + 1)
            if i != j and not (i == 0 and j == N)]

    try:
        model = gp.Model("vrptw")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = rem

        x = {}
        for k in range(Keff):
            for (i, j) in arcs:
                x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")
        t = [model.addVar(lb=e[i], ub=l[i], name=f"t_{i}") for i in range(N + 1)]

        for j in range(1, n + 1):
            model.addConstr(
                gp.quicksum(x[k, i, j] for k in range(Keff) for (i, jj) in arcs if jj == j) == 1)
        for k in range(Keff):
            outk = gp.quicksum(x[k, 0, j] for (i, j) in arcs if i == 0)
            ink = gp.quicksum(x[k, i, N] for (i, j) in arcs if j == N)
            model.addConstr(outk == ink)
            if k + 1 < Keff:
                outk2 = gp.quicksum(x[k + 1, 0, j] for (i, j) in arcs if i == 0)
                model.addConstr(outk2 <= outk)
            for j in range(1, n + 1):
                inj = gp.quicksum(x[k, i, jj] for (i, jj) in arcs if jj == j)
                outj = gp.quicksum(x[k, jj, i] for (jj, i) in arcs if jj == j)
                model.addConstr(inj == outj)
            cap = gp.quicksum(dem[j] * x[k, i, j] for (i, j) in arcs)
            model.addConstr(cap <= ctx.Q)

        for (i, j) in arcs:
            M = max(0.0, l[i] + s[i] + dist[i][j] - e[j])
            dij = dist[i][j] + s[i]
            for k in range(Keff):
                model.addConstr(t[j] >= t[i] + dij - M * (1 - x[k, i, j]))

        obj = gp.quicksum(dist[i][j] * x[k, i, j]
                        for k in range(Keff) for (i, j) in arcs)
        for i in range(1, n + 1):
            d = t[i] - p[i]
            obj += rho * d * d
        model.setObjective(obj, GRB.MINIMIZE)

        # MIP start from incumbent
        if best["routes"] and len(best["routes"]) <= Keff:
            for k, route in enumerate(best["routes"]):
                nodes = [0] + route + [N]
                for a in range(len(nodes) - 1):
                    i, j = nodes[a], nodes[a + 1]
                    var = x.get((k, i, j))
                    if var is not None:
                        var.Start = 1.0
            for seq in best["routes"]:
                res = detail(seq)
                if res is None:
                    continue
                times, _ = res
                for node, tt in zip(seq, times):
                    t[node].Start = tt
            t[0].Start = e[0]
            t[N].Start = min(l[N], e[N] if e[N] <= l[N] else l[N])

        def cb(mod, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                xs = mod.cbGetSolution(x)
                routes = []
                for k in range(Keff):
                    succ = {}
                    for (i, j) in arcs:
                        if xs.get(x[k, i, j], 0.0) > 0.5:
                            succ[i] = j
                    if 0 not in succ:
                        continue
                    cur = 0
                    seq = []
                    seen = set()
                    while cur != N and cur in succ and cur not in seen:
                        seen.add(cur)
                        cur = succ[cur]
                        if cur != N:
                            seq.append(cur)
                    if cur == N and seq:
                        routes.append(seq)
                cost = routes_total(eval_cost, routes)
                if cost < best["cost"] - 1e-7:
                    best["cost"] = cost
                    best["routes"] = [r[:] for r in routes]
                    sol = build_solution(ctx, detail, routes)
                    if logger and sol:
                        logger.log_solution(cost, sol)
            except Exception:
                pass

        model.optimize(cb)
    except Exception:
        return


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()
    deadline = t0 + max(1, args.time_limit)

    with open(args.instance_path) as f:
        data = json.load(f)
    ctx = Ctx(data)
    rng = random.Random(0)
    eval_cost, detail = make_evaluators(ctx)

    best = {"cost": INF, "routes": []}

    if ctx.n == 0:
        sol = {"objective_value": 0.0, "routes": [], "schedule": {}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # -------- construction (with restarts)
    init = None
    for attempt in range(12):
        customers = list(range(1, ctx.n + 1))
        if attempt > 0:
            rng.shuffle(customers)
        r = repair_regret(ctx, eval_cost, rng, [], customers, 2 if attempt % 2 == 0 else 3)
        if r is not None:
            init = r
            break
    if init is None:
        init = [[i] for i in range(1, ctx.n + 1)]

    ls_budget = min(deadline, time.time() + max(1.0, args.time_limit * 0.06))
    local_search(ctx, eval_cost, init, ls_budget)
    c = routes_total(eval_cost, init)
    if c < INF:
        best["cost"] = c
        best["routes"] = [r[:] for r in init]
        sol = build_solution(ctx, detail, best["routes"])
        if logger and sol:
            logger.log_solution(c, sol)

    # -------- choose MIQP (small instances) or full heuristic
    n_arcs = (ctx.N) * (ctx.N + 1)
    use_mip = ctx.n <= 30 and min(ctx.K, ctx.n) * n_arcs <= 9000

    if use_mip and deadline - time.time() > 8:
        warm_end = min(deadline, t0 + min(max(args.time_limit * 0.2, 2.0), 20.0))
        alns(ctx, rng, eval_cost, detail, best["routes"] or init, best, logger, warm_end)
        gurobi_phase(ctx, eval_cost, detail, best, logger, deadline)
        if time.time() < deadline - 0.3:
            alns(ctx, rng, eval_cost, detail, best["routes"] or init, best, logger,
                 deadline - 0.1)
    else:
        alns(ctx, rng, eval_cost, detail, best["routes"] or init, best, logger,
             deadline - 0.1)

    # -------- final output
    if best["cost"] == INF and init is not None:
        best["routes"] = [r[:] for r in init]
    sol = build_solution(ctx, detail, best["routes"])
    if sol is None:
        sol = {"objective_value": 0.0, "routes": [], "schedule": {}}
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()