import argparse
import json
import math
import random
import time
import traceback

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# --------------------------------------------------------------------------
# Problem data
# --------------------------------------------------------------------------
class Problem:
    def __init__(self, inst):
        self.n = inst["num_customers"]
        self.V = max(1, inst.get("num_vehicles", self.n))
        self.Q = inst["vehicle_capacity"]
        self.horizon = inst.get("schedule_horizon", None)
        size = self.n + 1
        self.dem = [0] * size
        self.a = [0.0] * size
        self.b = [0.0] * size
        self.svc = [0.0] * size
        self.xy = [(0.0, 0.0)] * size
        for node in inst["nodes"]:
            i = node["id"]
            self.dem[i] = node.get("demand", 0)
            self.a[i] = float(node.get("time_window_start", 0))
            self.b[i] = float(node.get("time_window_end", 0))
            self.svc[i] = float(node.get("service_time", 0))
            self.xy[i] = (float(node.get("x", 0)), float(node.get("y", 0)))
        if inst.get("distance_matrix"):
            self.D = [[float(v) for v in row] for row in inst["distance_matrix"]]
        else:
            self.D = [
                [
                    math.hypot(self.xy[i][0] - self.xy[j][0], self.xy[i][1] - self.xy[j][1])
                    for j in range(size)
                ]
                for i in range(size)
            ]
        self.edge_sets = inst.get("edge_sets", []) or []
        self.S = len(self.edge_sets)
        self.set_ids = [s.get("id", idx) for idx, s in enumerate(self.edge_sets)]
        self.acc = [float(s.get("access_cost", 0.0)) for s in self.edge_sets]
        self.edge2idx = {}
        for idx, s in enumerate(self.edge_sets):
            for e in s.get("edges", []):
                u, v = e[0], e[1]
                if u > v:
                    u, v = v, u
                self.edge2idx[(u, v)] = idx
        for e in inst.get("free_edges", []) or []:
            u, v = e[0], e[1]
            if u > v:
                u, v = v, u
            if (u, v) not in self.edge2idx:
                self.edge2idx[(u, v)] = -1

    def set_idx(self, u, v):
        if u > v:
            u, v = v, u
        return self.edge2idx.get((u, v), -1)

    def d(self, u, v):
        return self.D[u][v]


# --------------------------------------------------------------------------
# Route timing helpers
# --------------------------------------------------------------------------
def route_arrays(P, route):
    nodes = [0] + list(route) + [0]
    L = len(nodes)
    fwd = [0.0] * L
    times = [0.0] * L
    ok = True
    t = P.a[0]
    times[0] = t
    fwd[0] = t + P.svc[0]
    eps = 1e-9
    for i in range(1, L):
        node = nodes[i]
        arr = fwd[i - 1] + P.d(nodes[i - 1], node)
        st_ = max(arr, P.a[node])
        if st_ > P.b[node] + eps:
            ok = False
        times[i] = st_
        fwd[i] = st_ + (P.svc[node] if i < L - 1 else 0.0)
    bwd = [0.0] * L
    bwd[L - 1] = P.b[0]
    for i in range(L - 2, -1, -1):
        node = nodes[i]
        bwd[i] = min(P.b[node], bwd[i + 1] - P.svc[node] - P.d(node, nodes[i + 1]))
    return ok, fwd, bwd, times


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class State:
    def __init__(self, P, routes):
        self.P = P
        self.routes = [list(r) for r in routes if r]
        self.recompute()

    def copy(self):
        s = State.__new__(State)
        s.P = self.P
        s.routes = [list(r) for r in self.routes]
        s.loads = list(self.loads)
        s.counts = list(self.counts)
        s.dist = self.dist
        try:
            s.obj = self.obj
        except AttributeError:
            s.obj = None
        s.arrays = [(list(f), list(b)) for f, b in getattr(self, "arrays", [])]
        return s

    def recompute(self):
        P = self.P
        self.routes = [r for r in self.routes if r]
        counts = [0] * P.S
        loads = []
        arrays = []
        dist = 0.0
        for r in self.routes:
            loads.append(sum(P.dem[v] for v in r))
            prev = 0
            for node in r:
                dist += P.d(prev, node)
                idx = P.set_idx(prev, node)
                if idx >= 0:
                    counts[idx] += 1
                prev = node
            dist += P.d(prev, 0)
            idx = P.set_idx(prev, 0)
            if idx >= 0:
                counts[idx] += 1
            ok, fwd, bwd, _ = route_arrays(P, r)
            arrays.append((fwd, bwd))
        self.loads = loads
        self.counts = counts
        self.arrays = arrays
        self.dist = dist
        self.obj = dist + sum(P.acc[i] for i in range(P.S) if counts[i] > 0)


# --------------------------------------------------------------------------
# Access-cost delta helper
# --------------------------------------------------------------------------
def access_delta(st, removed_pairs, added_pairs):
    P = st.P
    rem = {}
    add = {}
    affected = set()
    for (u, v) in removed_pairs:
        idx = P.set_idx(u, v)
        if idx >= 0:
            rem[idx] = rem.get(idx, 0) + 1
            affected.add(idx)
    for (u, v) in added_pairs:
        idx = P.set_idx(u, v)
        if idx >= 0:
            add[idx] = add.get(idx, 0) + 1
            affected.add(idx)
    delta = 0.0
    for idx in affected:
        old = st.counts[idx]
        new = old - rem.get(idx, 0) + add.get(idx, 0)
        if old <= 0 and new > 0:
            delta += P.acc[idx]
        elif old > 0 and new <= 0:
            delta -= P.acc[idx]
    return delta


# --------------------------------------------------------------------------
# Insertion evaluation
# --------------------------------------------------------------------------
def eval_insert(st, r_idx, v, pos):
    P = st.P
    route = st.routes[r_idx]
    if st.loads[r_idx] + P.dem[v] > P.Q + 1e-9:
        return None
    if len(route) == 0:
        pos = 0
        u, w = 0, 0
    else:
        if pos < 0 or pos > len(route):
            return None
        fwd, bwd = st.arrays[r_idx]
        u = 0 if pos == 0 else route[pos - 1]
        w = 0 if pos == len(route) else route[pos]
        dep_u = fwd[pos]
        arr_v = dep_u + P.d(u, v)
        st_v = max(arr_v, P.a[v])
        if st_v > P.b[v] + 1e-9:
            return None
        dep_v = st_v + P.svc[v]
        arr_w = dep_v + P.d(v, w)
        if arr_w > bwd[pos + 1] + 1e-9:
            return None
    d_dist = P.d(u, v) + P.d(v, w) - P.d(u, w)
    delta = d_dist + access_delta(st, [(u, w)], [(u, v), (v, w)])
    return delta


# --------------------------------------------------------------------------
# Initial construction
# --------------------------------------------------------------------------
def initial_solution(P):
    base_order = sorted(range(1, P.n + 1), key=lambda v: (P.b[v], P.a[v]))
    for attempt in range(4):
        unrouted = list(base_order)
        if attempt > 0:
            random.shuffle(unrouted)
        st = State(P, [])
        leftover = build_with_regret(P, st, unrouted)
        if not leftover:
            return st
    # fallback: return best partial construction
    return st


def build_with_regret(P, st, unrouted):
    unrouted = list(unrouted)
    while unrouted:
        best_choice = None
        chosen_v = None
        best_regret = -1.0
        for v in unrouted:
            opts = []
            for r_idx in range(len(st.routes)):
                route = st.routes[r_idx]
                for pos in range(len(route) + 1):
                    delta = eval_insert(st, r_idx, v, pos)
                    if delta is not None:
                        opts.append((delta, r_idx, pos))
            if len(st.routes) < max(1, P.V):
                ok, _, _, _ = route_arrays(P, [v])
                if ok and P.dem[v] <= P.Q + 1e-9:
                    base = 2.0 * P.d(0, v)
                    idx = P.set_idx(0, v)
                    delta = base + (P.acc[idx] if idx >= 0 and st.counts[idx] == 0 else 0.0)
                    opts.append((delta, -1, 0))
            if not opts:
                chosen_v = v
                best_choice = None
                best_regret = float("inf")
                break
            opts.sort(key=lambda x: x[0])
            regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else opts[0][0] + 1e6
            if regret > best_regret:
                best_regret = regret
                chosen_v = v
                best_choice = opts[0]
        if best_choice is None:
            return unrouted
        delta, r_idx, pos = best_choice
        if r_idx == -1:
            st.routes.append([chosen_v])
        else:
            st.routes[r_idx].insert(pos, chosen_v)
        unrouted.remove(chosen_v)
        st.recompute()
    return []


# --------------------------------------------------------------------------
# Solution dictionary
# --------------------------------------------------------------------------
def build_solution(P, st):
    try:
        st.recompute()
    except Exception:
        pass
    routes_out = []
    used = [P.set_ids[i] for i in range(P.S) if st.counts[i] > 0]
    for vid, r in enumerate(st.routes, start=1):
        ok, fwd, bwd, times = route_arrays(P, r)
        seq = [0] + list(r) + [0]
        arcs = [[seq[i], seq[i + 1]] for i in range(len(seq) - 1)]
        vt = {}
        for i in range(len(seq) - 1):
            vt[str(seq[i])] = float(times[i])
        routes_out.append(
            {"vehicle": vid, "sequence": seq, "arcs": arcs, "visit_times": vt}
        )
    obj = float(st.obj) if st.obj is not None else 0.0
    return {
        "objective_value": obj,
        "routes": routes_out,
        "edge_sets_used": sorted(used),
    }


# --------------------------------------------------------------------------
# Local search
# --------------------------------------------------------------------------
def relocate_best(P, st, deadline):
    if time.time() > deadline:
        return False
    best = None
    eps = 1e-9
    for ra in range(len(st.routes)):
        route_a = st.routes[ra]
        for idx in range(len(route_a)):
            if time.time() > deadline:
                return False
            v = route_a[idx]
            u = 0 if idx == 0 else route_a[idx - 1]
            w = 0 if idx + 1 == len(route_a) else route_a[idx + 1]
            base_removed = [(u, v), (v, w)]
            base_added = [(u, w)]
            for rb in range(len(st.routes)):
                if rb == ra:
                    continue
                if st.loads[rb] + P.dem[v] > P.Q + eps:
                    continue
                route_b = st.routes[rb]
                if len(route_b) == 0:
                    continue
                fwd, bwd = st.arrays[rb]
                for pos in range(len(route_b) + 1):
                    pu = 0 if pos == 0 else route_b[pos - 1]
                    pw = 0 if pos == len(route_b) else route_b[pos]
                    dep_u = fwd[pos]
                    arr_v = dep_u + P.d(pu, v)
                    st_v = max(arr_v, P.a[v])
                    if st_v > P.b[v] + eps:
                        continue
                    dep_v = st_v + P.svc[v]
                    arr_w = dep_v + P.d(v, pw)
                    if arr_w > bwd[pos + 1] + eps:
                        continue
                    d_dist = (
                        P.d(pu, v) + P.d(v, pw) - P.d(pu, pw)
                        + P.d(u, w) - P.d(u, v) - P.d(v, w)
                    )
                    delta = d_dist + access_delta(
                        st, base_removed + [(pu, pw)],
                        base_added + [(pu, v), (v, pw)],
                    )
                    if best is None or delta < best[0] - eps:
                        best = (delta, ra, idx, rb, pos)
    if best is not None and best[0] < -1e-9:
        _, ra, idx, rb, pos = best
        v = st.routes[ra].pop(idx)
        st.routes[rb].insert(pos, v)
        st.recompute()
        return True
    return False


def two_opt_best(P, st, deadline):
    eps = 1e-9
    for ra in range(len(st.routes)):
        if time.time() > deadline:
            return False
        route = st.routes[ra]
        L = len(route)
        if L < 2:
            continue
        best = None
        for i in range(L):
            for j in range(i + 1, L):
                a_prev = 0 if i == 0 else route[i - 1]
                a_first = route[i]
                a_last = route[j]
                b_next = 0 if j + 1 == L else route[j + 1]
                d_dist = (
                    P.d(a_prev, a_last) + P.d(a_first, b_next)
                    - P.d(a_prev, a_first) - P.d(a_last, b_next)
                )
                delta = d_dist + access_delta(
                    st, [(a_prev, a_first), (a_last, b_next)],
                    [(a_prev, a_last), (a_first, b_next)],
                )
                if best is None or delta < best[0] - eps:
                    best = (delta, i, j)
        if best is not None and best[0] < -1e-9:
            delta, i, j = best
            new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            ok, _, _, _ = route_arrays(P, new_route)
            if ok:
                st.routes[ra] = new_route
                st.recompute()
                return True
    return False


def local_search(P, st, deadline):
    while time.time() < deadline:
        moved = False
        if relocate_best(P, st, deadline):
            moved = True
        if two_opt_best(P, st, deadline):
            moved = True
        if not moved:
            break
    return st


# --------------------------------------------------------------------------
# Destroy / repair
# --------------------------------------------------------------------------
def destroy(P, st, k, op):
    positions = [(r, i) for r in range(len(st.routes)) for i in range(len(st.routes[r]))]
    if not positions:
        return []
    if op == "worst":
        scored = []
        for r, i in positions:
            route = st.routes[r]
            v = route[i]
            u = 0 if i == 0 else route[i - 1]
            w = 0 if i + 1 == len(route) else route[i + 1]
            d_dist = P.d(u, v) + P.d(v, w) - P.d(u, w)
            saving = d_dist + access_delta(st, [(u, v), (v, w)], [(u, w)])
            scored.append((saving, (r, i)))
        scored.sort(key=lambda x: -x[0])
        top = scored[: max(1, min(len(scored), max(k * 3, k)))]
        take = min(k, len(top))
        chosen = [t[1] for t in random.sample(top, take)] if take > 0 else []
    else:
        take = min(k, len(positions))
        chosen = random.sample(positions, take) if take > 0 else []
    removed = []
    by_route = {}
    for r, i in chosen:
        by_route.setdefault(r, []).append(i)
        if r < len(st.routes) and i < len(st.routes[r]):
            removed.append(st.routes[r][i])
    for r, idxs in by_route.items():
        for i in sorted(idxs, reverse=True):
            if r < len(st.routes) and i < len(st.routes[r]):
                st.routes[r].pop(i)
    st.recompute()
    return removed


def repair(P, st, removed):
    unrouted = list(removed)
    while unrouted:
        best_choice = None
        chosen_v = None
        best_regret = -1.0
        failed = False
        for v in unrouted:
            opts = []
            for r_idx in range(len(st.routes)):
                route = st.routes[r_idx]
                for pos in range(len(route) + 1):
                    delta = eval_insert(st, r_idx, v, pos)
                    if delta is not None:
                        opts.append((delta, r_idx, pos))
            if len(st.routes) < max(1, P.V):
                ok, _, _, _ = route_arrays(P, [v])
                if ok and P.dem[v] <= P.Q + 1e-9:
                    base = 2.0 * P.d(0, v)
                    idx = P.set_idx(0, v)
                    delta = base + (P.acc[idx] if idx >= 0 and st.counts[idx] == 0 else 0.0)
                    opts.append((delta, -1, 0))
            if not opts:
                failed = True
                break
            opts.sort(key=lambda x: x[0])
            regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else opts[0][0] + 1e6
            if regret > best_regret:
                best_regret = regret
                chosen_v = v
                best_choice = opts[0]
        if failed:
            return False
        delta, r_idx, pos = best_choice
        if r_idx == -1:
            st.routes.append([chosen_v])
        else:
            st.routes[r_idx].insert(pos, chosen_v)
        unrouted.remove(chosen_v)
        st.recompute()
    return True


# --------------------------------------------------------------------------
# Safe logging
# --------------------------------------------------------------------------
def safe_log(logger, obj, P, st):
    if logger is None:
        return
    try:
        logger.log_solution(float(obj), build_solution(P, st))
    except Exception:
        try:
            logger.log(float(obj))
        except Exception:
            pass


# --------------------------------------------------------------------------
# LNS
# --------------------------------------------------------------------------
def run_lns(P, st, deadline, logger, log_record):
    current = st.copy()
    best = st.copy()
    best_obj = best.obj
    if best_obj is not None and best_obj < log_record[0] - 1e-9:
        log_record[0] = best_obj
        safe_log(logger, best_obj, P, best)
    n = max(1, P.n)
    T0 = max(1e-3, 0.02 * max(abs(best_obj or 1.0), 1.0))
    start = time.time()
    it = 0
    max_iters = 200000
    while time.time() < deadline and it < max_iters:
        it += 1
        try:
            k_min = max(1, int(0.10 * n))
            k_max = max(k_min, max(2, int(0.30 * n)))
            if k_max < k_min:
                k_max = k_min
            k = random.randint(k_min, k_max)
            op = "worst" if random.random() < 0.5 else "random"
            cand = current.copy()
            removed = destroy(P, cand, k, op)
            if not removed:
                continue
            if not repair(P, cand, removed):
                continue
            span = max(1e-9, deadline - start)
            frac = max(0.0, min(1.0, (time.time() - start) / span))
            T = T0 * max(0.02, 1.0 - frac)
            delta = cand.obj - current.obj
            if delta <= 0:
                accept = True
            else:
                exponent = max(-700.0, -delta / max(T, 1e-12))
                accept = random.random() < math.exp(exponent)
            if accept:
                current = cand
            if cand.obj is not None and (best_obj is None or cand.obj < best_obj - 1e-9):
                best_obj = cand.obj
                best = cand.copy()
                if best_obj < log_record[0] - 1e-9:
                    log_record[0] = best_obj
                    safe_log(logger, best_obj, P, best)
        except Exception:
            # never let a single bad iteration kill the search
            continue
    return best


# --------------------------------------------------------------------------
# Gurobi MIP refinement (small instances)
# --------------------------------------------------------------------------
def run_mip(P, warm_state, deadline, logger, log_record):
    if time.time() > deadline - 0.5:
        return warm_state
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return warm_state
    try:
        n = P.n
        nodes = list(range(n + 1))
        K = max(1, min(P.V, n))
        arcs = [(i, j) for i in nodes for j in nodes if i != j]
        apos = {a: k for k, a in enumerate(arcs)}
        A = len(arcs)
        m = gp.Model("fleet_routing")
        m.setParam("Seed", 0)
        m.setParam("MIPGap", 1e-4)
        m.setParam("NumericFocus", 0)
        m.setParam("Threads", 1)
        remaining = max(1.0, deadline - time.time())
        m.setParam("TimeLimit", remaining)
        m.setParam("OutputFlag", 0)

        x = m.addVars(A, K, vtype=GRB.BINARY, name="x")
        y = m.addVars(P.S, vtype=GRB.BINARY, name="y")
        t = {}
        for i in nodes:
            for k in range(K):
                t[(i, k)] = m.addVar(lb=P.a[i], ub=P.b[i], name="t_%d_%d" % (i, k))
        u = {}
        cust = list(range(1, n + 1))
        for i in cust:
            for k in range(K):
                u[(i, k)] = m.addVar(lb=1, ub=max(1, n), name="u_%d_%d" % (i, k))

        for j in cust:
            m.addConstr(
                gp.quicksum(x[apos[(i, j)], k] for i in nodes if i != j for k in range(K)) == 1
            )
        for k in range(K):
            m.addConstr(
                gp.quicksum(x[apos[(0, j)], k] for j in cust)
                == gp.quicksum(x[apos[(i, 0)], k] for i in cust)
            )
            m.addConstr(gp.quicksum(x[apos[(0, j)], k] for j in cust) <= 1)
            for j in cust:
                inn = gp.quicksum(x[apos[(i, j)], k] for i in nodes if i != j)
                out = gp.quicksum(x[apos[(j, i)], k] for i in nodes if i != j)
                m.addConstr(inn - out == 0)
            m.addConstr(
                gp.quicksum(P.dem[j] * x[apos[(i, j)], k] for j in cust for i in nodes if i != j)
                <= P.Q
            )
        for (i, j) in arcs:
            a_idx = apos[(i, j)]
            travel = P.svc[i] + P.d(i, j)
            M = max(0.0, P.b[i] + travel - P.a[j])
            for k in range(K):
                m.addConstr(t[(j, k)] >= t[(i, k)] + travel - M * (1 - x[a_idx, k]))
        nbig = max(1, n)
        for (i, j) in arcs:
            if i >= 1 and j >= 1:
                a_idx = apos[(i, j)]
                for k in range(K):
                    m.addConstr(u[(j, k)] >= u[(i, k)] + 1 - nbig * (1 - x[a_idx, k]))
        for sidx in range(P.S):
            for e in P.edge_sets[sidx]["edges"]:
                uu, vv = e[0], e[1]
                if (uu, vv) in apos and (vv, uu) in apos:
                    a1 = apos[(uu, vv)]
                    a2 = apos[(vv, uu)]
                    m.addConstr(
                        gp.quicksum(x[a1, k] + x[a2, k] for k in range(K)) <= 2 * K * y[sidx]
                    )
        m.setObjective(
            gp.quicksum(P.d(i, j) * x[apos[(i, j)], k] for (i, j) in arcs for k in range(K))
            + gp.quicksum((P.acc[s] if s < len(P.acc) else 0.0) * y[s] for s in range(P.S)),
            GRB.MINIMIZE,
        )

        try:
            for sidx in range(P.S):
                y[sidx].Start = 1.0 if warm_state.counts[sidx] > 0 else 0.0
            for k, route in enumerate(warm_state.routes[:K]):
                seq = [0] + list(route) + [0]
                ok_, _, _, times = route_arrays(P, route)
                for i in range(len(seq) - 1):
                    x[apos[(seq[i], seq[i + 1])], k].Start = 1.0
                for i, node in enumerate(seq[:-1]):
                    t[(node, k)].Start = times[i]
        except Exception:
            pass

        xref = [x[a, k] for a in range(A) for k in range(K)]

        def extract(values):
            routes_v = []
            for k in range(K):
                adj = {}
                for a in range(A):
                    try:
                        if values[a * K + k] > 0.5:
                            i, j = arcs[a]
                            adj[i] = j
                    except Exception:
                        pass
                if 0 not in adj:
                    continue
                route = []
                cur = 0
                guard = 0
                while guard < n + 2:
                    nxt = adj.get(cur)
                    if nxt is None or nxt == 0:
                        break
                    route.append(nxt)
                    cur = nxt
                    guard += 1
                if route:
                    routes_v.append(route)
            return routes_v

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < log_record[0] - 1e-9:
                        values = model.cbGetSolution(xref)
                        routes_v = extract(values)
                        stmp = State(P, routes_v)
                        log_record[0] = stmp.obj
                        safe_log(logger, stmp.obj, P, stmp)
                except Exception:
                    pass

        m.optimize(cb)

        if m.SolCount > 0:
            values = [v.X for v in xref]
            routes_v = extract(values)
            new_state = State(P, routes_v)
            if new_state.obj < (warm_state.obj if warm_state.obj is not None else float("inf")) - 1e-9:
                return new_state
    except Exception:
        return warm_state
    return warm_state


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    random.seed(0)
    start = time.time()
    time_limit = max(1, int(args.time_limit))
    deadline = start + time_limit
    reserve = 0.6
    hard_deadline = deadline - reserve
    log_record = [float("inf")]

    best = None
    try:
        P = Problem(inst)

        # initial solution
        best = initial_solution(P)
        try:
            best = local_search(P, best, hard_deadline)
        except Exception:
            pass
        if best.obj is not None and best.obj < log_record[0] - 1e-9:
            log_record[0] = best.obj
            safe_log(logger, best.obj, P, best)

        use_mip = 0 < P.n <= 16
        if use_mip:
            try:
                heuristic_budget = max(1.0, min(0.2 * time_limit, 20.0))
                h_deadline = min(hard_deadline, start + heuristic_budget)
                best = run_lns(P, best, h_deadline, logger, log_record)
                best = local_search(P, best, min(hard_deadline, time.time() + 0.5))
            except Exception:
                pass
            try:
                mip_state = run_mip(P, best, hard_deadline, logger, log_record)
                if mip_state.obj < best.obj - 1e-9:
                    best = mip_state
            except Exception:
                pass
        else:
            try:
                best = run_lns(P, best, hard_deadline, logger, log_record)
            except Exception:
                pass

        if time.time() < hard_deadline:
            try:
                best = local_search(P, best, hard_deadline)
            except Exception:
                pass
        if best.obj is not None and best.obj < log_record[0] - 1e-9:
            log_record[0] = best.obj
    except Exception:
        traceback.print_exc()
        if best is None:
            try:
                P = Problem(inst)
                best = State(P, [])
            except Exception:
                with open(args.solution_path, "w") as f:
                    json.dump({"objective_value": 0.0, "routes": [], "edge_sets_used": []}, f)
                return

    sol = build_solution(P, best)
    if logger is not None:
        try:
            logger.log_solution(sol["objective_value"], sol)
        except Exception:
            try:
                logger.log(sol["objective_value"])
            except Exception:
                pass

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()