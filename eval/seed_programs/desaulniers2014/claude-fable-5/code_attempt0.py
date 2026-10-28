#!/usr/bin/env python3
import argparse
import json
import time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()
    t_start = time.time()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data["num_customers"]
    T = data["num_periods"]
    K = data["num_vehicles"]
    Q = float(data["vehicle_capacity"])
    dep = data["depot"]
    I00 = float(dep["initial_inventory"])
    U0 = float(dep["max_inventory"])
    prod = float(dep["production_per_period"])
    h0 = float(dep["holding_cost"])
    Iinit = [0.0] * (n + 1)
    U = [0.0] * (n + 1)
    L = [0.0] * (n + 1)
    dem = [0.0] * (n + 1)
    hc = [0.0] * (n + 1)
    for cd in data["customers"]:
        i = cd["id"]
        Iinit[i] = float(cd["initial_inventory"])
        U[i] = float(cd["max_inventory"])
        L[i] = float(cd.get("min_inventory", 0))
        dem[i] = float(cd["demand_per_period"])
        hc[i] = float(cd["holding_cost"])
    c = data["distance_matrix"]

    # ---------------- helpers ----------------
    def tsp_route(nodes):
        if not nodes:
            return [0, 0]
        rem = set(nodes)
        tour = [0]
        cur = 0
        while rem:
            nxt = min(rem, key=lambda j: c[cur][j])
            tour.append(nxt)
            rem.discard(nxt)
            cur = nxt
        tour.append(0)
        improved = True
        it = 0
        while improved and it < 60:
            improved = False
            it += 1
            m = len(tour)
            for a in range(m - 3):
                for b in range(a + 2, m - 1):
                    i1, i2 = tour[a], tour[a + 1]
                    j1, j2 = tour[b], tour[b + 1]
                    if c[i1][j1] + c[i2][j2] < c[i1][i2] + c[j1][j2] - 1e-9:
                        tour[a + 1:b + 1] = tour[a + 1:b + 1][::-1]
                        improved = True
        return tour

    def ffd(q):
        items = sorted(q.items(), key=lambda kv: -kv[1])
        bins = []
        for i, v in items:
            if v > Q + 1e-9:
                return None
            placed = False
            for b in bins:
                if b[0] + v <= Q + 1e-9:
                    b[0] += v
                    b[1].append(i)
                    placed = True
                    break
            if not placed:
                bins.append([v, [i]])
        return [b[1] for b in bins]

    def trace_routes(ec):
        adj = defaultdict(lambda: defaultdict(int))
        for (i, j), v in ec.items():
            adj[i][j] += v
            adj[j][i] += v
        routes = []
        guard = 0
        while guard < 100000:
            guard += 1
            start = None
            for j, v in adj[0].items():
                if v > 0:
                    start = j
                    break
            if start is None:
                break
            route = [0]
            cur = 0
            steps = 0
            while steps < 100000:
                steps += 1
                nb = None
                for j, v in adj[cur].items():
                    if v > 0:
                        nb = j
                        break
                if nb is None:
                    break
                adj[cur][nb] -= 1
                adj[nb][cur] -= 1
                route.append(nb)
                cur = nb
                if cur == 0:
                    break
            if route[-1] != 0:
                route.append(0)
            routes.append(route)
        return routes

    def build_solution(qall, routes_all):
        periods = {}
        I = Iinit[:]
        I0 = I00
        travel = 0.0
        hold = 0.0
        for t in range(T):
            routes = [list(r) for r in routes_all[t] if len(r) > 2]
            for r in routes:
                for a in range(len(r) - 1):
                    travel += c[r[a]][r[a + 1]]
            q = qall[t]
            tot = sum(q.values())
            I0 = I0 + prod - tot
            if abs(I0) < 1e-9:
                I0 = 0.0
            hold += h0 * I0
            inv = {"depot": round(I0, 6)}
            deliv = {}
            for i in range(1, n + 1):
                I[i] = I[i] + q.get(i, 0.0) - dem[i]
                if abs(I[i]) < 1e-9:
                    I[i] = 0.0
                hold += hc[i] * I[i]
                inv[str(i)] = round(I[i], 6)
                if q.get(i, 0.0) > 1e-9:
                    deliv[str(i)] = round(q[i], 6)
            periods[str(t + 1)] = {"routes": routes, "deliveries": deliv,
                                   "inventories": inv}
        obj = travel + hold
        return obj, {"objective_value": obj,
                     "solution_details": {"periods": periods}}

    # ---------------- greedy heuristic ----------------
    def greedy():
        I = Iinit[:]
        I0 = I00
        qall = []
        routes_all = []
        for t in range(T):
            supply = I0 + prod
            need = {}
            for i in range(1, n + 1):
                sh = dem[i] + L[i] - I[i]
                if sh > 1e-9:
                    if sh > Q + 1e-9:
                        return None
                    need[i] = sh
            total = sum(need.values())
            if total > supply + 1e-6 or total > K * Q + 1e-6:
                return None

            def finalize(q):
                tot = sum(q.values())
                excess = (I0 + prod - tot) - U0
                if excess > 1e-9:
                    for i in sorted(range(1, n + 1), key=lambda j: c[0][j]):
                        if excess <= 1e-9:
                            break
                        cur = q.get(i, 0.0)
                        room = min(U[i] - I[i], Q) - cur
                        room = min(room, K * Q - tot)
                        add = min(max(room, 0.0), excess)
                        if add > 1e-9:
                            q[i] = cur + add
                            tot += add
                            excess -= add
                    if excess > 1e-9:
                        return None
                q = {i: v for i, v in q.items() if v > 1e-9}
                bins = ffd(q)
                if bins is None or len(bins) > K:
                    return None
                return q, bins

            q = dict(need)
            rem = min(K * Q, supply) - total
            for i in sorted(need, key=lambda j: hc[j]):
                maxdel = max(0.0, dem[i] * (T - t) + L[i] - I[i])
                room = min(U[i] - I[i], Q, maxdel) - q[i]
                add = min(max(room, 0.0), rem)
                if add > 1e-9:
                    q[i] += add
                    rem -= add
            res = finalize(q)
            if res is None:
                res = finalize(dict(need))
            if res is None:
                return None
            q, bins = res
            routes = [tsp_route(b) for b in bins]
            tot = sum(q.values())
            I0 = I0 + prod - tot
            if I0 < -1e-6:
                return None
            for i in range(1, n + 1):
                I[i] = I[i] + q.get(i, 0.0) - dem[i]
            qall.append(q)
            routes_all.append(routes)
        return qall, routes_all

    best_obj = float("inf")
    best_payload = None
    heur = greedy()
    if heur is not None:
        h_qall, h_routes = heur
        obj, payload = build_solution(h_qall, h_routes)
        best_obj, best_payload = obj, payload
        if logger:
            logger.log_solution(obj, payload)

    # ---------------- MIP (branch-and-cut) ----------------
    rem_time = args.time_limit - (time.time() - t_start) - 1.5
    if n >= 1 and rem_time > 3:
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
            m.Params.TimeLimit = max(1.0, rem_time)

            edges = [(i, j) for i in range(n + 1) for j in range(i + 1, n + 1)]
            E = len(edges)
            x = {}
            for t in range(T):
                for (i, j) in edges:
                    if i == 0:
                        x[t, i, j] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=2,
                                              name=f"x_{t}_{i}_{j}")
                    else:
                        x[t, i, j] = m.addVar(vtype=GRB.BINARY,
                                              name=f"x_{t}_{i}_{j}")
            z = m.addVars(range(T), range(1, n + 1), vtype=GRB.BINARY, name="z")
            w = m.addVars(range(T), vtype=GRB.INTEGER, lb=0, ub=K, name="w")
            qv = {}
            for t in range(T):
                for i in range(1, n + 1):
                    ub = min(Q, U[i])
                    if t == 0:
                        ub = min(ub, max(0.0, U[i] - Iinit[i]))
                    qv[t, i] = m.addVar(lb=0.0, ub=ub, name=f"q_{t}_{i}")
            I0v = m.addVars(range(T), lb=0.0, ub=U0, name="I0")
            Iv = {}
            for t in range(T):
                for i in range(1, n + 1):
                    Iv[t, i] = m.addVar(lb=L[i], ub=U[i], name=f"I_{t}_{i}")

            inc = defaultdict(list)
            for (i, j) in edges:
                inc[i].append((i, j))
                inc[j].append((i, j))

            for t in range(T):
                for i in range(1, n + 1):
                    m.addConstr(gp.quicksum(x[t, a, b] for (a, b) in inc[i])
                                == 2 * z[t, i])
                m.addConstr(gp.quicksum(x[t, a, b] for (a, b) in inc[0])
                            == 2 * w[t])
                for i in range(1, n + 1):
                    m.addConstr(qv[t, i] <= min(Q, U[i]) * z[t, i])
                    prev = Iinit[i] if t == 0 else Iv[t - 1, i]
                    m.addConstr(Iv[t, i] == prev + qv[t, i] - dem[i])
                    if t > 0:
                        m.addConstr(Iv[t - 1, i] + qv[t, i] <= U[i])
                prev0 = I00 if t == 0 else I0v[t - 1]
                m.addConstr(I0v[t] == prev0 + prod
                            - gp.quicksum(qv[t, i] for i in range(1, n + 1)))
                m.addConstr(gp.quicksum(qv[t, i] for i in range(1, n + 1))
                            <= Q * w[t])
                for (i, j) in edges:
                    if i >= 1:
                        m.addConstr(x[t, i, j] <= z[t, i])
                        m.addConstr(x[t, i, j] <= z[t, j])

            obj_expr = gp.quicksum(c[i][j] * x[t, i, j]
                                   for t in range(T) for (i, j) in edges)
            obj_expr += gp.quicksum(h0 * I0v[t] for t in range(T))
            obj_expr += gp.quicksum(hc[i] * Iv[t, i]
                                    for t in range(T) for i in range(1, n + 1))
            m.setObjective(obj_expr, GRB.MINIMIZE)

            # warm start from heuristic
            if heur is not None:
                for t in range(T):
                    counts = defaultdict(int)
                    for r in h_routes[t]:
                        for a in range(len(r) - 1):
                            u, v2 = r[a], r[a + 1]
                            if u != v2:
                                counts[(min(u, v2), max(u, v2))] += 1
                    for (i, j) in edges:
                        x[t, i, j].Start = counts.get((i, j), 0)
                    for i in range(1, n + 1):
                        qval = h_qall[t].get(i, 0.0)
                        qv[t, i].Start = qval
                        z[t, i].Start = 1 if qval > 1e-9 else 0
                    w[t].Start = len([r for r in h_routes[t] if len(r) > 2])

            xlist = [x[t, i, j] for t in range(T) for (i, j) in edges]
            qlist = [qv[t, i] for t in range(T) for i in range(1, n + 1)]
            holder = {"best": best_obj, "payload": None}

            def delta_expr(t, S):
                return gp.quicksum(x[t, i, j] for (i, j) in edges
                                   if (i in S) != (j in S))

            def cb(model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                xs = model.cbGetSolution(xlist)
                qs = model.cbGetSolution(qlist)
                ncuts = 0
                routes_per_t = []
                for t in range(T):
                    ec = {}
                    base = t * E
                    for e in range(E):
                        v = int(round(xs[base + e]))
                        if v > 0:
                            ec[edges[e]] = v
                    parent = list(range(n + 1))

                    def find(a):
                        while parent[a] != a:
                            parent[a] = parent[parent[a]]
                            a = parent[a]
                        return a

                    nodes = set()
                    for (i, j) in ec:
                        nodes.add(i)
                        nodes.add(j)
                        ra, rb = find(i), find(j)
                        if ra != rb:
                            parent[ra] = rb
                    comps = defaultdict(set)
                    for i in nodes:
                        comps[find(i)].add(i)
                    for r, S in comps.items():
                        if 0 in S:
                            continue
                        expr = delta_expr(t, S)
                        for k in S:
                            model.cbLazy(expr >= 2 * z[t, k])
                            ncuts += 1
                    routes = trace_routes(ec)
                    for r in routes:
                        S = set(r) - {0}
                        if not S:
                            continue
                        load = sum(qs[t * n + i - 1] for i in S)
                        if load > Q + 1e-6:
                            expr = delta_expr(t, S)
                            model.cbLazy(
                                Q * expr >= 2 * gp.quicksum(qv[t, i] for i in S))
                            ncuts += 1
                    routes_per_t.append(routes)
                if ncuts == 0:
                    objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if objv < holder["best"] - 1e-6:
                        qall = []
                        for t in range(T):
                            qd = {}
                            for i in range(1, n + 1):
                                val = qs[t * n + i - 1]
                                if val > 1e-6:
                                    qd[i] = round(val, 6)
                            qall.append(qd)
                        robj, payload = build_solution(qall, routes_per_t)
                        holder["best"] = objv
                        holder["payload"] = (robj, payload)
                        if logger:
                            logger.log_solution(robj, payload)

            m.optimize(cb)

            if m.SolCount > 0:
                qall2 = []
                routes2 = []
                for t in range(T):
                    ec = {}
                    for (i, j) in edges:
                        v = int(round(x[t, i, j].X))
                        if v > 0:
                            ec[(i, j)] = v
                    routes2.append(trace_routes(ec))
                    qd = {}
                    for i in range(1, n + 1):
                        val = qv[t, i].X
                        if val > 1e-6:
                            qd[i] = round(val, 6)
                    qall2.append(qd)
                obj2, payload2 = build_solution(qall2, routes2)
                if obj2 < best_obj - 1e-9:
                    best_obj, best_payload = obj2, payload2
                    if logger:
                        logger.log_solution(obj2, payload2)
            if holder["payload"] is not None:
                robj, payload = holder["payload"]
                if robj < best_obj - 1e-9:
                    best_obj, best_payload = robj, payload
        except Exception:
            pass

    if best_payload is None:
        # last-resort trivial output (no deliveries)
        qall = [dict() for _ in range(T)]
        routes_all = [[] for _ in range(T)]
        best_obj, best_payload = build_solution(qall, routes_all)
        if logger:
            logger.log_solution(best_obj, best_payload)

    with open(args.solution_path, "w") as f:
        json.dump(best_payload, f)


if __name__ == "__main__":
    main()