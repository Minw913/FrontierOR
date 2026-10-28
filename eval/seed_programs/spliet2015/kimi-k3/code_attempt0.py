import argparse
import json
import math
import random
import time

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

EPS = 1e-9
TOL = 1e-7


# --------------------------------------------------------------------------- #
# Instance / context
# --------------------------------------------------------------------------- #
class Inst:
    def __init__(self, data):
        self.n = int(data["num_customers"])
        dep = data["depot"]
        self.E0 = float(dep["exogenous_time_window"][0])
        self.L0 = float(dep["exogenous_time_window"][1])
        self.Q = float(data["vehicle_capacity"])
        default_w = data.get("endogenous_time_window_width", 0.0)
        self.d = data["distance_matrix"]
        self.ids = []
        self.e = [0.0] * (self.n + 2)
        self.L = [0.0] * (self.n + 2)
        self.w = [0.0] * (self.n + 2)
        for c in data["customers"]:
            i = int(c["id"])
            self.ids.append(i)
            self.e[i] = float(c["exogenous_time_window"][0])
            self.L[i] = float(c["exogenous_time_window"][1])
            cw = c.get("endogenous_time_window_width", default_w)
            self.w[i] = float(cw)
        self.ids.sort()
        self.scenarios = []
        for s in data["scenarios"]:
            dem_list = s["demands"]
            dem = [0.0] * (self.n + 2)
            for k, cid in enumerate(self.ids):  # demands ordered by customer id
                dem[cid] = float(dem_list[k])
            self.scenarios.append({
                "id": s["scenario_id"],
                "p": float(s["probability"]),
                "dem": dem,
            })


class Ctx:
    def __init__(self, inst, a):
        self.inst = inst
        self.d = inst.d
        self.n = inst.n
        self.end = inst.n + 1
        self.E0 = inst.E0
        self.L0 = inst.L0
        self.Q = inst.Q
        self.ids = inst.ids
        self.a = a
        self.b = [0.0] * (self.n + 2)
        self.refresh_b()

    def refresh_b(self):
        for i in self.ids:
            self.b[i] = self.a[i] + self.inst.w[i]


# --------------------------------------------------------------------------- #
# Basic route helpers
# --------------------------------------------------------------------------- #
def route_cost(r, d, end):
    if not r:
        return 0.0
    c = d[0][r[0]]
    for k in range(len(r) - 1):
        c += d[r[k]][r[k + 1]]
    c += d[r[-1]][end]
    return c


def total_cost(routes, d, end):
    return sum(route_cost(r, d, end) for r in routes)


def compute_times(ctx, r):
    """Earliest feasible schedule; list aligned with [0]+r+[end], or None."""
    d = ctx.d
    a = ctx.a
    b = ctx.b
    t = ctx.E0
    prev = 0
    times = [0.0] * (len(r) + 2)
    times[0] = t
    for k, u in enumerate(r):
        arr = t + d[prev][u]
        au = a[u]
        s = arr if arr > au else au
        if s > b[u] + TOL:
            return None
        times[k + 1] = s
        t = s
        prev = u
    arr = t + d[prev][ctx.end]
    if arr > ctx.L0 + TOL:
        return None
    times[-1] = arr
    return times


def route_arrays(ctx, r):
    """serv[k]: current service time at position k; lt[k]: latest service time
    at position k that keeps the suffix (incl. depot return) feasible."""
    d = ctx.d
    a = ctx.a
    b = ctx.b
    end = ctx.end
    L = len(r)
    serv = [0.0] * L
    t = ctx.E0
    prev = 0
    for idx in range(L):
        u = r[idx]
        arr = t + d[prev][u]
        au = a[u]
        s = arr if arr > au else au
        serv[idx] = s
        t = s
        prev = u
    lt = [0.0] * L
    if L:
        u = r[-1]
        v1 = ctx.L0 - d[u][end]
        lt[-1] = b[u] if b[u] < v1 else v1
        for idx in range(L - 2, -1, -1):
            u = r[idx]
            v = r[idx + 1]
            x = lt[idx + 1] - d[u][v]
            lt[idx] = b[u] if b[u] < x else x
    return serv, lt


def best_insert(ctx, r, serv, lt, u):
    """Best (delta, position) insertion of u into (feasible) route r, or None."""
    d = ctx.d
    a = ctx.a
    b = ctx.b
    end = ctx.end
    E0 = ctx.E0
    L = len(r)
    au = a[u]
    bu = b[u]
    best = None
    for p in range(L + 1):
        if p == 0:
            prev_node = 0
            prev_time = E0
        else:
            prev_node = r[p - 1]
            prev_time = serv[p - 1]
        arr = prev_time + d[prev_node][u]
        s = arr if arr > au else au
        if s > bu + TOL:
            continue
        if p < L:
            v = r[p]
            if s + d[u][v] > lt[p] + TOL:
                continue
            nxt = v
        else:
            if s + d[u][end] > ctx.L0 + TOL:
                continue
            nxt = end
        delta = d[prev_node][u] + d[u][nxt] - d[prev_node][nxt]
        if best is None or delta < best[0]:
            best = (delta, p)
    return best


def loads_of(routes, dem):
    return [sum(dem[u] for u in r) for r in routes]


# --------------------------------------------------------------------------- #
# Construction (regret-2 insertion) and repair
# --------------------------------------------------------------------------- #
def _regret_loop(ctx, dem, routes, loads, unrouted, cache, rng, deadline):
    d = ctx.d
    end = ctx.end
    Q = ctx.Q

    def recompute(ri):
        r = routes[ri]
        serv, lt = route_arrays(ctx, r)
        for u in unrouted:
            if loads[ri] + dem[u] > Q + EPS:
                cache[u][ri] = None
            else:
                cache[u][ri] = best_insert(ctx, r, serv, lt, u)

    while unrouted:
        if deadline is not None and time.time() > deadline:
            for u in unrouted:
                routes.append([u])
                loads.append(dem[u])
            break
        best = None
        for u in unrouted:
            d1 = None
            d2 = None
            bri = -1
            bp = -1
            for ri, v in cache[u].items():
                if v is None:
                    continue
                delta, p = v
                if d1 is None or delta < d1:
                    d2 = d1
                    d1 = delta
                    bri = ri
                    bp = p
                elif d2 is None or delta < d2:
                    d2 = delta
            nd = d[0][u] + d[u][end]
            if d1 is None or nd < d1:
                if d1 is not None and (d2 is None or d1 < d2):
                    d2 = d1
                d1 = nd
                bri = -1
                bp = -1
            elif d2 is None or nd < d2:
                d2 = nd
            regret = (d2 if d2 is not None else 1e15) - d1
            key = regret + rng.random() * 1e-9
            if best is None or key > best[0]:
                best = (key, u, bri, bp)
        _, u, ri, p = best
        if ri < 0:
            routes.append([u])
            loads.append(dem[u])
            unrouted.remove(u)
            recompute(len(routes) - 1)
        else:
            routes[ri].insert(p, u)
            loads[ri] += dem[u]
            unrouted.remove(u)
            recompute(ri)
    return routes, loads


def construct(ctx, dem, rng, deadline=None):
    routes = []
    loads = []
    unrouted = [i for i in ctx.ids if dem[i] > EPS]
    cache = {u: {} for u in unrouted}
    return _regret_loop(ctx, dem, routes, loads, unrouted, cache, rng, deadline)


def regret_repair(ctx, dem, routes, loads, removed, rng, deadline=None):
    routes = [r[:] for r in routes]
    loads = loads[:]
    unrouted = list(removed)
    cache = {u: {} for u in unrouted}
    for ri in range(len(routes)):
        serv, lt = route_arrays(ctx, routes[ri])
        for u in unrouted:
            if loads[ri] + dem[u] > ctx.Q + EPS:
                cache[u][ri] = None
            else:
                cache[u][ri] = best_insert(ctx, routes[ri], serv, lt, u)
    return _regret_loop(ctx, dem, routes, loads, unrouted, cache, rng, deadline)


# --------------------------------------------------------------------------- #
# Local search
# --------------------------------------------------------------------------- #
def local_search(ctx, dem, routes, loads, rng, deadline, max_passes=30):
    d = ctx.d
    end = ctx.end
    Q = ctx.Q
    passes = 0
    while passes < max_passes and time.time() < deadline:
        passes += 1
        arrays = [route_arrays(ctx, r) for r in routes]
        # ---- relocate (best cross-route move) ----
        best = None
        for ri, r in enumerate(routes):
            L = len(r)
            for idx in range(L):
                u = r[idx]
                prev = r[idx - 1] if idx > 0 else 0
                nxt = r[idx + 1] if idx + 1 < L else end
                rem = d[prev][nxt] - d[prev][u] - d[u][nxt]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    if loads[rj] + dem[u] > Q + EPS:
                        continue
                    bi = best_insert(ctx, routes[rj], arrays[rj][0], arrays[rj][1], u)
                    if bi is None:
                        continue
                    tot = rem + bi[0]
                    if tot < -TOL and (best is None or tot < best[0]):
                        best = (tot, u, ri, rj, bi[1])
                nd = d[0][u] + d[u][end]
                tot = rem + nd
                if tot < -TOL and (best is None or tot < best[0]):
                    best = (tot, u, ri, -1, -1)
        if best is not None:
            _, u, ri, rj, p = best
            if rj >= 0:
                routes[rj].insert(p, u)
                loads[rj] += dem[u]
            else:
                routes.append([u])
                loads.append(dem[u])
            routes[ri].remove(u)
            loads[ri] -= dem[u]
            if not routes[ri]:
                del routes[ri]
                del loads[ri]
            continue
        # ---- swap ----
        swapped = False
        for ri in range(len(routes)):
            if swapped or time.time() > deadline:
                break
            for rj in range(ri + 1, len(routes)):
                if swapped:
                    break
                r1, r2 = routes[ri], routes[rj]
                c12 = route_cost(r1, d, end) + route_cost(r2, d, end)
                for i1, u in enumerate(r1):
                    if swapped:
                        break
                    for i2, v in enumerate(r2):
                        if loads[ri] - dem[u] + dem[v] > Q + EPS:
                            continue
                        if loads[rj] - dem[v] + dem[u] > Q + EPS:
                            continue
                        nr1 = r1[:]
                        nr1[i1] = v
                        nr2 = r2[:]
                        nr2[i2] = u
                        if route_cost(nr1, d, end) + route_cost(nr2, d, end) < c12 - TOL:
                            if compute_times(ctx, nr1) is not None and compute_times(ctx, nr2) is not None:
                                routes[ri] = nr1
                                routes[rj] = nr2
                                loads[ri] = loads[ri] - dem[u] + dem[v]
                                loads[rj] = loads[rj] - dem[v] + dem[u]
                                swapped = True
                                break
        if swapped:
            continue
        # ---- 2-opt within a route ----
        opted = False
        for ri in range(len(routes)):
            if time.time() > deadline:
                break
            r = routes[ri]
            L = len(r)
            if L < 3:
                continue
            base = route_cost(r, d, end)
            found = False
            for i in range(L - 1):
                if found:
                    break
                for j in range(i + 1, L):
                    nr = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    if route_cost(nr, d, end) < base - TOL and compute_times(ctx, nr) is not None:
                        routes[ri] = nr
                        found = True
                        opted = True
                        break
            if found:
                break
        if opted:
            continue
        break
    return routes, loads


# --------------------------------------------------------------------------- #
# ALNS per scenario
# --------------------------------------------------------------------------- #
def destroy(ctx, dem, routes, loads, k, rng):
    routes = [r[:] for r in routes]
    positions = [(ri, idx) for ri, r in enumerate(routes) for idx in range(len(r))]
    if not positions:
        return routes, loads[:], []
    op = rng.random()
    if op < 0.4:
        chosen = rng.sample(positions, min(k, len(positions)))
        removed = [routes[ri][idx] for ri, idx in chosen]
    elif op < 0.8:
        contrib = []
        for ri, r in enumerate(routes):
            L = len(r)
            for idx, u in enumerate(r):
                prev = r[idx - 1] if idx > 0 else 0
                nxt = r[idx + 1] if idx + 1 < L else ctx.end
                sav = ctx.d[prev][u] + ctx.d[u][nxt] - ctx.d[prev][nxt]
                contrib.append((sav, u))
        contrib.sort(reverse=True)
        removed = [u for _, u in contrib[:k]]
    else:
        ri, idx = positions[rng.randrange(len(positions))]
        seed = routes[ri][idx]
        others = []
        for r in routes:
            for u in r:
                if u == seed:
                    continue
                metric = ctx.d[seed][u] + abs(ctx.a[seed] - ctx.a[u])
                others.append((metric, u))
        others.sort()
        removed = [seed] + [u for _, u in others[:k - 1]]
    remset = set(removed)
    new_routes = []
    new_loads = []
    for r in routes:
        nr = [u for u in r if u not in remset]
        if nr:
            new_routes.append(nr)
            new_loads.append(sum(dem[u] for u in nr))
    return new_routes, new_loads, removed


def alns(ctx, dem, routes, loads, slice_end, rng):
    n = len(ctx.ids)
    cur = [r[:] for r in routes]
    curl = loads[:]
    curc = total_cost(cur, ctx.d, ctx.end)
    best = [r[:] for r in routes]
    bestl = loads[:]
    bestc = curc
    T = 0.03 * max(1.0, curc)
    it = 0
    while time.time() < slice_end and it < 2000:
        it += 1
        k = rng.randint(1, max(2, min(15, (n + 1) // 3)))
        rr, ll, removed = destroy(ctx, dem, cur, curl, k, rng)
        rr, ll = regret_repair(ctx, dem, rr, ll, removed, rng, slice_end)
        c = total_cost(rr, ctx.d, ctx.end)
        dc = c - curc
        if dc < -TOL or rng.random() < math.exp(-dc / T):
            cur, curl, curc = rr, ll, c
        if c < bestc - TOL:
            best = [r[:] for r in rr]
            bestl = ll[:]
            bestc = c
        T *= 0.97
    polish_end = min(slice_end, time.time() + 0.1)
    local_search(ctx, dem, best, bestl, rng, polish_end, max_passes=6)
    bestc = total_cost(best, ctx.d, ctx.end)
    return best, bestl, bestc


# --------------------------------------------------------------------------- #
# Time-window assignment heuristics
# --------------------------------------------------------------------------- #
def place_window(inst, i, t):
    """Place the width-w_i endogenous window inside the exogenous one so that
    it covers time t (t clamped into the exogenous window first)."""
    lo = inst.e[i]
    hi = inst.L[i] - inst.w[i]
    if hi < lo:
        hi = lo
    t = min(max(t, inst.e[i]), inst.L[i])
    ai = t - 0.5 * inst.w[i]
    if ai < lo:
        ai = lo
    if ai > hi:
        ai = hi
    if ai > t:
        ai = lo
    if ai + inst.w[i] < t:
        ai = hi
    return ai


def centered_windows(inst):
    a = [0.0] * (inst.n + 2)
    end = inst.n + 1
    for i in inst.ids:
        flo = max(inst.e[i], inst.E0 + inst.d[0][i])
        fhi = min(inst.L[i], inst.L0 - inst.d[i][end])
        mid = 0.5 * (inst.e[i] + inst.L[i])
        t = min(max(mid, flo), fhi) if fhi >= flo else mid
        a[i] = place_window(inst, i, t)
    return a


def reference_windows(inst, rng, deadline):
    """Route expected demands under exogenous windows, then place each narrow
    window around the realized service time."""
    exp_dem = [0.0] * (inst.n + 2)
    for i in inst.ids:
        e = 0.0
        for sc in inst.scenarios:
            e += sc["p"] * sc["dem"][i]
        v = int(math.ceil(e - 1e-9))
        if v < 1:
            v = 1
        if v > inst.Q:
            v = int(inst.Q)
        exp_dem[i] = v
    a0 = [0.0] * (inst.n + 2)
    for i in inst.ids:
        a0[i] = inst.e[i]
    ctx = Ctx(inst, a0)
    for i in inst.ids:  # use full exogenous windows for the reference routing
        ctx.b[i] = inst.L[i]
    routes, loads = construct(ctx, exp_dem, rng, deadline)
    local_search(ctx, exp_dem, routes, loads, rng, deadline, max_passes=8)
    t_i = {}
    for r in routes:
        times = compute_times(ctx, r)
        if times is None:
            continue
        for k, u in enumerate(r):
            t_i[u] = times[k + 1]
    a = [0.0] * (inst.n + 2)
    for i in inst.ids:
        ti = t_i.get(i, 0.5 * (inst.e[i] + inst.L[i]))
        a[i] = place_window(inst, i, ti)
    return a


def service_times_per_scenario(ctx, group_routes):
    out = []
    for routes in group_routes:
        st = {}
        for r in routes:
            times = compute_times(ctx, r)
            if times is None:
                return None
            for k, u in enumerate(r):
                st[u] = times[k + 1]
        out.append(st)
    return out


def recenter_windows(inst, ctx, scen_st):
    changed = False
    for i in inst.ids:
        vals = [st[i] for st in scen_st]
        m_i = min(vals)
        M_i = max(vals)
        lo = max(inst.e[i], M_i - inst.w[i])
        hi = min(inst.L[i] - inst.w[i], m_i)
        if lo > hi + 1e-9:
            continue
        if lo > hi:
            lo = hi
        target = 0.5 * (m_i + M_i) - 0.5 * inst.w[i]
        new = min(max(target, lo), hi)
        if abs(new - ctx.a[i]) > 1e-9:
            ctx.a[i] = new
            changed = True
    if changed:
        ctx.refresh_b()
    return changed


def perturb_windows(inst, ctx, scen_st, rng, frac=0.25):
    cand = list(inst.ids)
    rng.shuffle(cand)
    cnt = max(1, int(len(cand) * frac))
    changed = False
    for i in cand[:cnt]:
        vals = [st[i] for st in scen_st]
        m_i = min(vals)
        M_i = max(vals)
        lo = max(inst.e[i], M_i - inst.w[i])
        hi = min(inst.L[i] - inst.w[i], m_i)
        if hi - lo > 1e-9:
            new = lo + rng.random() * (hi - lo)
            if abs(new - ctx.a[i]) > 1e-9:
                ctx.a[i] = new
                changed = True
    if changed:
        ctx.refresh_b()
    return changed


def verify(ctx, group_routes):
    for routes in group_routes:
        for r in routes:
            if compute_times(ctx, r) is None:
                return False
    return True


def sanitize(ctx, groups, group_routes):
    out = []
    for g, routes in zip(groups, group_routes):
        dem = g["dem"]
        newr = []
        seen = set()
        for r in routes:
            rr = [u for u in r if u not in seen]
            for u in rr:
                seen.add(u)
            load = sum(dem[u] for u in rr)
            if rr and load <= ctx.Q + 1e-6 and compute_times(ctx, rr) is not None:
                newr.append(rr)
            else:
                for u in rr:
                    newr.append([u])
        for i in ctx.ids:
            if i not in seen:
                newr.append([i])
        out.append(newr)
    return out


# --------------------------------------------------------------------------- #
# Scenario grouping (identical demand vectors routed once)
# --------------------------------------------------------------------------- #
def build_groups(inst):
    groups = []
    mapping = {}
    for sc in inst.scenarios:
        key = tuple(sc["dem"][i] for i in inst.ids)
        if key in mapping:
            g = groups[mapping[key]]
            g["p"] += sc["p"]
            g["ids"].append(sc["id"])
        else:
            mapping[key] = len(groups)
            groups.append({"p": sc["p"], "ids": [sc["id"]], "dem": sc["dem"]})
    return groups


def build_solution(inst, ctx, group_routes, groups):
    tw = {str(i): [ctx.a[i], ctx.a[i] + inst.w[i]] for i in inst.ids}
    routes_out = {}
    obj = 0.0
    for g, routes in zip(groups, group_routes):
        entries = []
        gcost = 0.0
        for r in routes:
            times = compute_times(ctx, r)
            c = route_cost(r, ctx.d, ctx.end)
            at = {"0": times[0]}
            for k, u in enumerate(r):
                at[str(u)] = times[k + 1]
            at[str(ctx.end)] = times[-1]
            entries.append({
                "route": [0] + list(r) + [ctx.end],
                "arrival_times": at,
                "cost": c,
            })
            gcost += c
        for sid in g["ids"]:
            routes_out[str(sid)] = entries
        obj += g["p"] * gcost
    return {"objective_value": obj, "time_windows": tw, "routes": routes_out}, obj


# --------------------------------------------------------------------------- #
# Exact MIP (used on small/medium instances)
# --------------------------------------------------------------------------- #
def solve_mip(inst, ctx, group_routes, groups, time_budget):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None
    try:
        n = inst.n
        end = inst.end
        d = inst.d
        N = inst.ids
        S = len(groups)
        m = gp.Model("stochastic_tw_vrp")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, time_budget)
        m.Params.OutputFlag = 0

        a = {i: m.addVar(lb=inst.e[i], ub=max(inst.e[i], inst.L[i] - inst.w[i]), name=f"a_{i}") for i in N}
        x = {}
        s = {}
        q = {}
        maxt = 0.0
        for row in d:
            rm = max(row)
            if rm > maxt:
                maxt = rm
        M = (inst.L0 - inst.E0) + maxt + 1.0
        nodes_out = [0] + N
        nodes_in = N + [end]
        obj = gp.LinExpr()
        for si in range(S):
            p = groups[si]["p"]
            for i in nodes_out:
                for j in nodes_in:
                    if i == j or (i == 0 and j == end):
                        continue
                    var = m.addVar(vtype=GRB.BINARY, name=f"x_{si}_{i}_{j}")
                    x[si, i, j] = var
                    obj += p * d[i][j] * var
            for i in N:
                s[si, i] = m.addVar(lb=inst.e[i], ub=inst.L[i], name=f"s_{si}_{i}")
                q[si, i] = m.addVar(lb=0.0, ub=float(inst.Q), name=f"q_{si}_{i}")
            s[si, end] = m.addVar(lb=0.0, ub=float(inst.L0), name=f"s_{si}_{end}")
        m.setObjective(obj, GRB.MINIMIZE)
        for si in range(S):
            dem = groups[si]["dem"]
            for j in N:
                m.addConstr(gp.quicksum(x[si, i, j] for i in nodes_out if i != j) == 1.0)
                m.addConstr(gp.quicksum(x[si, j, k] for k in nodes_in if k != j) == 1.0)
                m.addConstr(s[si, j] >= a[j])
                m.addConstr(s[si, j] <= a[j] + inst.w[j])
                m.addConstr(q[si, j] >= dem[j])
            m.addConstr(gp.quicksum(x[si, 0, j] for j in N) ==
                        gp.quicksum(x[si, i, end] for i in N))
            for i in nodes_out:
                ti = inst.E0 if i == 0 else s[si, i]
                qi = 0.0 if i == 0 else q[si, i]
                for j in N:
                    if i == j:
                        continue
                    m.addConstr(s[si, j] >= ti + d[i][j] - M * (1 - x[si, i, j]))
                    m.addConstr(q[si, j] >= qi + dem[j] - inst.Q * (1 - x[si, i, j]))
            for i in N:
                m.addConstr(s[si, end] >= s[si, i] + d[i][end] - M * (1 - x[si, i, end]))
        # warm start from current incumbent
        for i in N:
            a[i].Start = ctx.a[i]
        for si in range(S):
            dem = groups[si]["dem"]
            for r in group_routes[si]:
                times = compute_times(ctx, r)
                nodes = [0] + list(r) + [end]
                for u, v in zip(nodes[:-1], nodes[1:]):
                    key = (si, u, v)
                    if key in x:
                        x[key].Start = 1.0
                if times is not None:
                    for k, u in enumerate(r):
                        s[si, u].Start = times[k + 1]
                    s[si, end].Start = times[-1]
                cum = 0.0
                for u in r:
                    cum += dem[u]
                    q[si, u].Start = cum
        m.optimize()
        if m.SolCount == 0:
            return None
        a_new = [0.0] * (n + 2)
        for i in N:
            a_new[i] = a[i].X
        routes_out = []
        for si in range(S):
            succ = {}
            starts = []
            for i in nodes_out:
                for j in nodes_in:
                    key = (si, i, j)
                    if key in x and x[key].X > 0.5:
                        if i == 0:
                            starts.append(j)
                        else:
                            succ[i] = j
            rs = []
            covered = set()
            for st in starts:
                r = []
                cur = st
                while cur != end and cur not in covered and len(r) <= n:
                    r.append(cur)
                    covered.add(cur)
                    cur = succ.get(cur, end)
                if r:
                    rs.append(r)
            for i in N:
                if i not in covered:
                    rs.append([i])
            routes_out.append(rs)
        tot = 0.0
        for si in range(S):
            tot += groups[si]["p"] * total_cost(routes_out[si], d, end)
        return a_new, routes_out, tot
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.instance_path) as f:
        data = json.load(f)
    inst = Inst(data)
    rng = random.Random(0)
    t0 = time.time()
    total = max(1, args.time_limit)
    deadline = t0 + total - max(0.3, min(2.0, 0.05 * total))

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    groups = build_groups(inst)
    G = len(groups)

    # ---------------- Phase 1: initial window assignments ---------------- #
    candidates = [centered_windows(inst)]
    if time.time() < deadline - 1.0 and inst.n <= 400:
        try:
            ref_budget = min(deadline, time.time() + max(1.0, 0.15 * total))
            candidates.append(reference_windows(inst, rng, ref_budget))
        except Exception:
            pass

    best = None
    last_logged = None
    for a_c in candidates:
        ctx_c = Ctx(inst, a_c)
        groutes = []
        gloads = []
        for g in groups:
            r, l = construct(ctx_c, g["dem"], rng, deadline)
            ls_slice = max(0.03, min(0.5, (deadline - time.time()) / (2 * G + 1)))
            local_search(ctx_c, g["dem"], r, l, rng, time.time() + ls_slice, max_passes=8)
            groutes.append(r)
            gloads.append(l)
        sol, obj = build_solution(inst, ctx_c, groutes, groups)
        if best is None or obj < best[0]:
            best = (obj, list(ctx_c.a), groutes, gloads)
            if logger:
                logger.log_solution(obj, sol)
            last_logged = obj
        if time.time() > deadline - 0.5:
            break

    best_obj, a_best, group_routes, group_loads = best
    ctx = Ctx(inst, a_best)

    # ---------------- Phase 2: exact MIP on small instances -------------- #
    remaining = deadline - time.time()
    binvars = G * (inst.n + 1) * (inst.n + 2)
    if remaining > 6.0 and inst.n <= 28 and binvars <= 15000:
        budget = min(remaining * 0.55, remaining - 3.0)
        res = solve_mip(inst, ctx, group_routes, groups, budget)
        if res is not None:
            a_m, routes_m, obj_m = res
            ctx_m = Ctx(inst, list(a_m))
            if obj_m < best_obj - 1e-6 and verify(ctx_m, routes_m):
                ctx = ctx_m
                group_routes = routes_m
                group_loads = [loads_of(rts, groups[si]["dem"]) for si, rts in enumerate(group_routes)]
                best_obj = obj_m
                sol, _ = build_solution(inst, ctx, group_routes, groups)
                if logger:
                    logger.log_solution(best_obj, sol)
                last_logged = best_obj

    # ---------------- Phase 3: heuristic improvement loop ---------------- #
    group_costs = [total_cost(rts, ctx.d, ctx.end) for rts in group_routes]
    while time.time() < deadline - 0.2:
        rem = deadline - time.time()
        slice_s = max(0.05, min(2.0, rem / (2.5 * G)))
        improved = False
        order = list(range(G))
        rng.shuffle(order)
        for si in order:
            now = time.time()
            if now > deadline - 0.2:
                break
            g = groups[si]
            try:
                nr, nl, nc = alns(ctx, g["dem"], group_routes[si], group_loads[si],
                                  now + slice_s, rng)
            except Exception:
                continue
            if nc < group_costs[si] - 1e-9:
                group_routes[si] = nr
                group_loads[si] = nl
                group_costs[si] = nc
                improved = True
                obj = sum(groups[k]["p"] * group_costs[k] for k in range(G))
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    sol, _ = build_solution(inst, ctx, group_routes, groups)
                    if logger:
                        logger.log_solution(best_obj, sol)
                    last_logged = best_obj
        if time.time() > deadline - 0.2:
            break
        scen_st = service_times_per_scenario(ctx, group_routes)
        if scen_st is None:
            continue
        old_a = ctx.a[:]
        old_b = ctx.b[:]
        changed = False
        try:
            changed = recenter_windows(inst, ctx, scen_st)
            if changed and not verify(ctx, group_routes):
                ctx.a = old_a
                ctx.b = old_b
                changed = False
        except Exception:
            ctx.a = old_a
            ctx.b = old_b
            changed = False
        if not improved and not changed:
            old_a = ctx.a[:]
            old_b = ctx.b[:]
            try:
                perturbed = perturb_windows(inst, ctx, scen_st, rng)
                if perturbed and not verify(ctx, group_routes):
                    ctx.a = old_a
                    ctx.b = old_b
            except Exception:
                ctx.a = old_a
                ctx.b = old_b

    # ---------------- Finalize ------------------------------------------- #
    group_routes = sanitize(ctx, groups, group_routes)
    sol, obj = build_solution(inst, ctx, group_routes, groups)
    if logger and (last_logged is None or obj < last_logged - 1e-9):
        logger.log_solution(obj, sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()