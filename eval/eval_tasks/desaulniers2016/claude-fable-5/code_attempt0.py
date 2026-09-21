import argparse
import json
import math
import time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    start_time = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data["num_customers"]
    T = data["num_periods"]
    K = data["num_vehicles"]
    C = float(data["vehicle_capacity"])
    N = data["num_nodes"]
    dist = data["distance_matrix"]
    h = data["holding_costs"]
    d = data["customer_demands"]          # d[i-1][t]
    p = data["supplier_production"]
    U = [float(u) for u in data["inventory_capacities"]]
    I0 = [float(v) for v in data["initial_inventories"]]
    cust = list(range(1, N))
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # ---------------- solution finalization ----------------
    def finalize(period_routes):
        routing = 0.0
        deliv = [[0.0] * N for _ in range(T)]
        routes_out = {}
        for t in range(T):
            pr = {}
            for k, (rt, dl) in enumerate(period_routes[t]):
                for a, b in zip(rt[:-1], rt[1:]):
                    routing += dist[a][b]
                pr[str(k)] = {"route": [int(v) for v in rt],
                              "deliveries": {str(i): float(v) for i, v in dl.items()}}
                for i, v in dl.items():
                    deliv[t][i] += v
            routes_out[str(t)] = pr
        holding = 0.0
        invout = {str(i): {} for i in range(N)}
        inv = list(I0)
        for t in range(T):
            inv[0] += p[t] - sum(deliv[t][1:])
            if abs(inv[0]) < 1e-9:
                inv[0] = 0.0
            holding += h[0] * inv[0]
            invout["0"][str(t)] = inv[0]
            for i in range(1, N):
                inv[i] += deliv[t][i] - d[i - 1][t]
                if abs(inv[i]) < 1e-9:
                    inv[i] = 0.0
                holding += h[i] * inv[i]
                invout[str(i)][str(t)] = inv[i]
        obj = routing + holding
        sol = {"objective_value": obj, "routes": routes_out, "inventories": invout}
        return obj, sol

    # ---------------- heuristic construction ----------------
    def get_state(q):
        prev = [[0.0] * N for _ in range(T)]
        after = [[0.0] * N for _ in range(T)]
        inv = list(I0)
        dep = I0[0]
        dep_end = [0.0] * T
        for t in range(T):
            dep += p[t]
            ship = 0.0
            for i in range(1, N):
                prev[t][i] = inv[i]
                a = inv[i] + q[t][i]
                after[t][i] = a
                inv[i] = a - d[i - 1][t]
                ship += q[t][i]
            dep -= ship
            dep_end[t] = dep
        return prev, after, dep_end

    def check(q):
        prev, after, dep_end = get_state(q)
        for t in range(T):
            for i in range(1, N):
                if q[t][i] > C + 1e-6:
                    return False, ('qcap', t, i)
                if after[t][i] > U[i] + 1e-6:
                    return False, ('cap', t, i)
                if after[t][i] - d[i - 1][t] < -1e-6:
                    return False, ('stock', t, i)
            s = sum(q[t][1:])
            if s > K * C + 1e-6:
                return False, ('veh', t, -1)
            if dep_end[t] < -1e-6:
                return False, ('dep_lo', t, -1)
            if dep_end[t] > U[0] + 1e-6:
                return False, ('dep_hi', t, -1)
        return True, None

    def move_earlier(q, t, i, amount):
        """Try to move up to `amount` of q[t][i] to earlier periods. Returns moved amount."""
        moved = 0.0
        for tp in range(t - 1, -1, -1):
            if moved >= amount - 1e-9:
                break
            prev, after, dep_end = get_state(q)
            veh_room = K * C - sum(q[tp][1:])
            if veh_room <= 1e-9:
                continue
            dslack = min(dep_end[s] for s in range(tp, t))
            room = min(U[i] - after[s][i] for s in range(tp, t))
            qcap_room = C - q[tp][i]
            delta = min(amount - moved, q[t][i], veh_room, dslack, room, qcap_room)
            if delta > 1e-9:
                q[tp][i] += delta
                q[t][i] -= delta
                moved += delta
        return moved

    def apply_fix(q, viol):
        typ, t, ii = viol
        if typ == 'qcap':
            excess = q[t][ii] - C
            return move_earlier(q, t, ii, excess) >= excess - 1e-7
        if typ == 'veh':
            excess = sum(q[t][1:]) - K * C
            cands = sorted([i for i in cust if q[t][i] > 1e-9], key=lambda i: -q[t][i])
            for i in cands:
                if excess <= 1e-7:
                    return True
                excess -= move_earlier(q, t, i, min(excess, q[t][i]))
            return excess <= 1e-7
        if typ == 'dep_hi':
            prev, after, dep_end = get_state(q)
            overflow = dep_end[t] - U[0]
            veh_room = K * C - sum(q[t][1:])
            cands = sorted(cust, key=lambda i: (0 if q[t][i] > 1e-9 else 1, dist[0][i]))
            for i in cands:
                if overflow <= 1e-7:
                    break
                room = min(U[i] - after[t][i], C - q[t][i], veh_room)
                delta = min(overflow, room)
                if delta <= 1e-9:
                    continue
                q[t][i] += delta
                e = delta
                for tp in range(t + 1, T):
                    r = min(q[tp][i], e)
                    q[tp][i] -= r
                    e -= r
                    if e <= 1e-9:
                        break
                overflow -= delta
                veh_room -= delta
            return overflow <= 1e-7
        return False

    def ffd_pack(items):
        items = sorted(items, key=lambda x: -x[1])
        bins = []
        for i, qt in items:
            placed = False
            for b in bins:
                if b[0] + qt <= C + 1e-6:
                    b[0] += qt
                    b[1][i] = qt
                    placed = True
                    break
            if not placed:
                if len(bins) >= K:
                    return None
                bins.append([qt, {i: qt}])
        return bins

    def tsp_route(custs):
        if not custs:
            return [0, 0]
        rem = set(custs)
        order = []
        cur = 0
        while rem:
            nxt = min(rem, key=lambda j: dist[cur][j])
            order.append(nxt)
            rem.remove(nxt)
            cur = nxt
        route = [0] + order + [0]
        improved = True
        while improved:
            improved = False
            for a in range(1, len(route) - 2):
                for b in range(a + 1, len(route) - 1):
                    delta = (dist[route[a - 1]][route[b]] + dist[route[a]][route[b + 1]]
                             - dist[route[a - 1]][route[a]] - dist[route[b]][route[b + 1]])
                    if delta < -1e-9:
                        route[a:b + 1] = reversed(route[a:b + 1])
                        improved = True
        return route

    def heuristic():
        q = [[0.0] * N for _ in range(T)]
        inv = list(I0)
        for t in range(T):
            for i in range(1, N):
                need = d[i - 1][t] - inv[i]
                if need > 1e-9:
                    q[t][i] = need
                    inv[i] = 0.0
                else:
                    inv[i] = inv[i] - d[i - 1][t]
        ok = False
        for _ in range(400):
            ok, viol = check(q)
            if ok:
                break
            if not apply_fix(q, viol):
                return None
        if not ok:
            ok, _ = check(q)
            if not ok:
                return None
        period_routes = []
        for t in range(T):
            items = [(i, q[t][i]) for i in range(1, N) if q[t][i] > 1e-9]
            bins = ffd_pack(items)
            if bins is None:
                return None
            rts = []
            for b in bins:
                rt = tsp_route(list(b[1].keys()))
                rts.append((rt, dict(b[1])))
            rts.sort(key=lambda r: -len(r[1]))
            period_routes.append(rts)
        return period_routes

    best = [float('inf'), None]

    heur_routes = None
    try:
        heur_routes = heuristic()
    except Exception:
        heur_routes = None

    if heur_routes is not None:
        obj, sol = finalize(heur_routes)
        best[0], best[1] = obj, sol
        if logger:
            logger.log_solution(obj, sol)

    # ---------------- MIP ----------------
    remaining = args.time_limit - (time.time() - start_time) - 3.0
    if remaining > 5.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("irp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.LazyConstraints = 1
            m.Params.TimeLimit = max(5.0, remaining)

            x = {}
            y = {}
            z = {}
            qv = {}
            for t in range(T):
                for k in range(K):
                    for (i, j) in edges:
                        ub = 2 if i == 0 else 1
                        x[t, k, i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub,
                                                 obj=dist[i][j], name=f"x_{t}_{k}_{i}_{j}")
                    for i in cust:
                        y[t, k, i] = m.addVar(vtype=GRB.BINARY, name=f"y_{t}_{k}_{i}")
                        qv[t, k, i] = m.addVar(lb=0.0, ub=min(U[i], C), name=f"q_{t}_{k}_{i}")
                    z[t, k] = m.addVar(vtype=GRB.BINARY, name=f"z_{t}_{k}")
            Iv = {}
            for t in range(T):
                for i in range(N):
                    Iv[t, i] = m.addVar(lb=0.0, ub=U[i], obj=h[i], name=f"I_{t}_{i}")
            m.ModelSense = GRB.MINIMIZE
            m.update()

            for t in range(T):
                for k in range(K):
                    m.addConstr(gp.quicksum(x[t, k, 0, j] for j in cust) == 2 * z[t, k])
                    for i in cust:
                        inc = [x[t, k, min(i, j), max(i, j)] for j in range(N) if j != i]
                        m.addConstr(gp.quicksum(inc) == 2 * y[t, k, i])
                        m.addConstr(y[t, k, i] <= z[t, k])
                        m.addConstr(qv[t, k, i] <= min(U[i], C) * y[t, k, i])
                        m.addConstr(x[t, k, 0, i] <= 2 * y[t, k, i])
                    for (i, j) in edges:
                        if i != 0:
                            m.addConstr(x[t, k, i, j] <= y[t, k, i])
                            m.addConstr(x[t, k, i, j] <= y[t, k, j])
                    m.addConstr(gp.quicksum(y[t, k, i] for i in cust) >= z[t, k])
                    m.addConstr(gp.quicksum(qv[t, k, i] for i in cust) <= C * z[t, k])
                for k in range(K - 1):
                    m.addConstr(z[t, k] >= z[t, k + 1])
                    m.addConstr(gp.quicksum(y[t, k, i] for i in cust) >=
                                gp.quicksum(y[t, k + 1, i] for i in cust))
                for i in cust:
                    m.addConstr(gp.quicksum(y[t, k, i] for k in range(K)) <= 1)
                    prevI = I0[i] if t == 0 else Iv[t - 1, i]
                    tot = gp.quicksum(qv[t, k, i] for k in range(K))
                    m.addConstr(Iv[t, i] == prevI + tot - d[i - 1][t])
                    m.addConstr(prevI + tot <= U[i])
                prev0 = I0[0] if t == 0 else Iv[t - 1, 0]
                m.addConstr(Iv[t, 0] == prev0 + p[t] -
                            gp.quicksum(qv[t, k, i] for k in range(K) for i in cust))

            # branching priorities
            for key, var in z.items():
                var.BranchPriority = 3
            for key, var in y.items():
                var.BranchPriority = 2

            # warm start
            if heur_routes is not None:
                for var in x.values():
                    var.Start = 0
                for var in y.values():
                    var.Start = 0
                for var in z.values():
                    var.Start = 0
                for var in qv.values():
                    var.Start = 0.0
                for t in range(T):
                    for k, (rt, dl) in enumerate(heur_routes[t]):
                        if k >= K:
                            continue
                        z[t, k].Start = 1
                        ecount = defaultdict(int)
                        for a, b in zip(rt[:-1], rt[1:]):
                            ecount[(min(a, b), max(a, b))] += 1
                        for (a, b), c in ecount.items():
                            x[t, k, a, b].Start = c
                        for i, v in dl.items():
                            y[t, k, i].Start = 1
                            qv[t, k, i].Start = v

            # flattened lists for fast callback retrieval
            xflat, xmeta = [], []
            xblock = {}
            for t in range(T):
                for k in range(K):
                    s0 = len(xflat)
                    for (i, j) in edges:
                        xflat.append(x[t, k, i, j])
                        xmeta.append((i, j))
                    xblock[(t, k)] = (s0, len(xflat))
            yflat, yblock = [], {}
            for t in range(T):
                for k in range(K):
                    s0 = len(yflat)
                    for i in cust:
                        yflat.append(y[t, k, i])
                    yblock[(t, k)] = (s0, len(yflat))
            qflat = []
            qindex = {}
            for t in range(T):
                for k in range(K):
                    for i in cust:
                        qindex[(t, k, i)] = len(qflat)
                        qflat.append(qv[t, k, i])

            def build_period_routes(xval, qval):
                period_routes = [[] for _ in range(T)]
                for t in range(T):
                    rts = []
                    for k in range(K):
                        s0, s1 = xblock[(t, k)]
                        adj = defaultdict(list)
                        for idx in range(s0, s1):
                            c = int(round(xval[idx]))
                            if c > 0:
                                i, j = xmeta[idx - s0] if False else edges[idx - s0]
                                for _ in range(c):
                                    adj[i].append(j)
                                    adj[j].append(i)
                        if not adj.get(0):
                            continue
                        route = [0]
                        cur = 0
                        guard = 0
                        while guard < 5 * N + 10:
                            guard += 1
                            nxt = adj[cur].pop()
                            adj[nxt].remove(cur)
                            route.append(nxt)
                            cur = nxt
                            if cur == 0:
                                break
                        dl = {}
                        for i in set(route) - {0}:
                            v = qval[qindex[(t, k, i)]]
                            dl[i] = max(0.0, v)
                        rts.append((route, dl))
                    period_routes[t] = rts
                return period_routes

            def callback(model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                xvals = model.cbGetSolution(xflat)
                yvals = model.cbGetSolution(yflat)
                added = False
                for t in range(T):
                    for k in range(K):
                        s0, s1 = xblock[(t, k)]
                        adj = defaultdict(list)
                        for idx in range(s0, s1):
                            if xvals[idx] > 0.5:
                                i, j = edges[idx - s0]
                                adj[i].append(j)
                                adj[j].append(i)
                        ys0, ys1 = yblock[(t, k)]
                        visited = [cust[idx - ys0] for idx in range(ys0, ys1)
                                   if yvals[idx] > 0.5]
                        if not visited:
                            continue
                        seen = {0}
                        stack = [0]
                        while stack:
                            u = stack.pop()
                            for v in adj[u]:
                                if v not in seen:
                                    seen.add(v)
                                    stack.append(v)
                        missing = [i for i in visited if i not in seen]
                        handled = set()
                        for start in missing:
                            if start in handled:
                                continue
                            comp = {start}
                            stack = [start]
                            while stack:
                                u = stack.pop()
                                for v in adj[u]:
                                    if v != 0 and v not in comp:
                                        comp.add(v)
                                        stack.append(v)
                            handled |= comp
                            S = sorted(comp)
                            if len(S) < 2:
                                continue
                            Spairs = [(a, b) for ai, a in enumerate(S) for b in S[ai + 1:]]
                            budget = len(S) * T * K
                            targets = ([(t2, k2) for t2 in range(T) for k2 in range(K)]
                                       if budget <= 2000 else [(t, k)])
                            for (t2, k2) in targets:
                                xs = gp.quicksum(x[t2, k2, a, b] for (a, b) in Spairs)
                                for mm in S:
                                    model.cbLazy(xs <= gp.quicksum(
                                        y[t2, k2, i] for i in S if i != mm))
                            added = True
                if not added:
                    qvals = model.cbGetSolution(qflat)
                    try:
                        pr = build_period_routes(xvals, qvals)
                        obj, sol = finalize(pr)
                        if obj < best[0] - 1e-6:
                            best[0], best[1] = obj, sol
                            if logger:
                                logger.log_solution(obj, sol)
                    except Exception:
                        pass

            m.optimize(callback)

            if m.SolCount > 0:
                xvals = [var.X for var in xflat]
                qvals = [var.X for var in qflat]
                pr = build_period_routes(xvals, qvals)
                obj, sol = finalize(pr)
                if obj < best[0] - 1e-6:
                    best[0], best[1] = obj, sol
                    if logger:
                        logger.log_solution(obj, sol)
        except Exception:
            pass

    if best[1] is None:
        # last resort: empty plan (schema-valid output)
        _, sol = finalize([[] for _ in range(T)])
        best[1] = sol

    with open(args.solution_path, "w") as f:
        json.dump(best[1], f, indent=2)


if __name__ == "__main__":
    main()