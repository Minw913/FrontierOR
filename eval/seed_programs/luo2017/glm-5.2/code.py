#!/usr/bin/env python3
import argparse
import json
import math
import random
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    random.seed(0)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if (args.log_path and SolutionLogger is not None)
        else None
    )
    state = {"best_sol": None, "best_obj": float("inf"), "logged_obj": float("inf")}

    def update_best(obj, sol):
        if obj < state["best_obj"] - 1e-9:
            state["best_sol"] = sol
            state["best_obj"] = obj
            if logger is not None and obj < state["logged_obj"] - 1e-9:
                logger.log_solution(obj, sol)
                state["logged_obj"] = obj

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["num_customers"])
    Q = float(data["vehicle_capacity"])
    cf = data["cost_function"]
    a = float(cf["a"])
    b = float(cf["b"])

    depot = data["depot"]
    dist = [[float(x) for x in row] for row in data["distance_matrix"]]
    if "time_matrix" in data:
        time_mat = [[float(x) for x in row] for row in data["time_matrix"]]
    else:
        time_mat = dist

    demand = [0.0] * (n + 1)
    tw = [None] * (n + 2)
    service = [0.0] * (n + 2)
    coords = {}
    tw[0] = depot["time_window"]
    service[0] = float(depot.get("service_time", 0.0))
    coords[0] = (depot["x"], depot["y"])
    for c in data["customers"]:
        cid = int(c["id"])
        demand[cid] = float(c["demand"])
        tw[cid] = c["time_window"]
        service[cid] = float(c["service_time"])
        coords[cid] = (c["x"], c["y"])
    sink = n + 1
    tw[sink] = tw[0]
    service[sink] = 0.0
    e0, l0 = tw[0]

    full_trucks = []
    full_cost = 0.0
    rem = [0.0] * (n + 1)
    for i in range(1, n + 1):
        d = demand[i]
        if Q > 0 and d > 0:
            f = int(d // Q)
            r = d - f * Q
        else:
            f = 0
            r = d
        for _ in range(f):
            full_trucks.append(i)
            full_cost += dist[0][i] * (a * Q + b) + dist[i][0] * b
        rem[i] = r

    active = [i for i in range(1, n + 1) if rem[i] > 1e-9]
    m = len(active)

    def route_info(seq):
        if not seq:
            return None
        t = e0
        prev = 0
        s_prev = 0.0
        total = 0.0
        D = {}
        for c in seq:
            travel = time_mat[prev][c]
            arr = t + s_prev + travel
            start = arr if arr > tw[c][0] else tw[c][0]
            if start > tw[c][1] + 1e-6:
                return None
            total += dist[prev][c]
            D[c] = total
            t = start
            prev = c
            s_prev = service[c]
        arr_dep = t + s_prev + time_mat[prev][0]
        if arr_dep > l0 + 1e-6:
            return None
        total += dist[prev][0]
        return {
            "seq": seq,
            "dist": total,
            "F": b * total,
            "coeff": {c: a * D[c] for c in seq},
        }

    def route_cost(seq, q):
        load = sum(q.values())
        prev = 0
        cost = 0.0
        for c in seq:
            cost += dist[prev][c] * (a * load + b)
            load -= q[c]
            prev = c
        cost += dist[prev][0] * b
        return cost

    def build_solution_dict(full_trucks, reduced):
        routes_out = []
        obj = 0.0
        vid = 0
        for i in full_trucks:
            q = Q
            c = dist[0][i] * (a * q + b) + dist[i][0] * b
            obj += c
            routes_out.append(
                {
                    "vehicle": vid,
                    "route": [0, i, sink],
                    "deliveries": {str(i): float(q)},
                    "edges": [
                        {"from": 0, "to": i, "load_weight": float(q)},
                        {"from": i, "to": sink, "load_weight": 0.0},
                    ],
                }
            )
            vid += 1
        for veh in reduced:
            seq = veh["seq"]
            q = veh["q"]
            seq = [c for c in seq if q.get(c, 0.0) > 1e-9]
            q = {c: q[c] for c in seq}
            if not seq:
                continue
            c = route_cost(seq, q)
            obj += c
            route = [0] + seq + [sink]
            edges = []
            load = sum(q.values())
            prev = 0
            for c in seq:
                edges.append({"from": prev, "to": c, "load_weight": float(load)})
                load -= q[c]
                prev = c
            edges.append({"from": prev, "to": sink, "load_weight": 0.0})
            routes_out.append(
                {
                    "vehicle": vid,
                    "route": route,
                    "deliveries": {str(c): float(q[c]) for c in seq},
                    "edges": edges,
                }
            )
            vid += 1
        return {"objective_value": float(obj), "routes": routes_out}

    direct_reduced = [{"seq": [i], "q": {i: rem[i]}} for i in active]
    direct_sol = build_solution_dict(full_trucks, direct_reduced)
    update_best(direct_sol["objective_value"], direct_sol)

    if m == 0:
        if state["best_sol"] is None:
            state["best_sol"] = build_solution_dict(full_trucks, [])
        with open(args.solution_path, "w") as f:
            json.dump(state["best_sol"], f, indent=2)
        return

    def sol_from_seqs(seqs):
        veh = []
        for seq in seqs:
            if route_info(seq):
                veh.append({"seq": list(seq), "q": {i: rem[i] for i in seq}})
        return veh

    def clarke_wright(randomized=False):
        routes = {}
        rdemand = {}
        for i in active:
            routes[i] = [i]
            rdemand[i] = rem[i]
        cust_route = {i: i for i in active}
        savings = []
        for ii in range(m):
            i = active[ii]
            for jj in range(ii + 1, m):
                j = active[jj]
                best = None
                for seq in ((i, j), (j, i)):
                    info = route_info(seq)
                    if info:
                        if best is None or info["dist"] < best:
                            best = info["dist"]
                if best is not None:
                    save = (
                        dist[0][i] + dist[i][0] + dist[0][j] + dist[j][0]
                    ) - best
                    savings.append((save, i, j))
        if randomized:
            random.shuffle(savings)
        else:
            savings.sort(key=lambda x: x[0], reverse=True)
        for save, i, j in savings:
            ri = cust_route[i]
            rj = cust_route[j]
            if ri == rj:
                continue
            ri_seq = routes[ri]
            rj_seq = routes[rj]
            if i not in (ri_seq[0], ri_seq[-1]) or j not in (rj_seq[0], rj_seq[-1]):
                continue
            if rdemand[ri] + rdemand[rj] > Q + 1e-6:
                continue
            si = ri_seq if ri_seq[-1] == i else ri_seq[::-1]
            sj = rj_seq if rj_seq[0] == j else rj_seq[::-1]
            merged = si + sj
            if route_info(merged) is None:
                si2 = ri_seq if ri_seq[0] == i else ri_seq[::-1]
                sj2 = rj_seq if rj_seq[-1] == j else rj_seq[::-1]
                merged = sj2 + si2
                if route_info(merged) is None:
                    continue
            routes[ri] = merged
            rdemand[ri] = rdemand[ri] + rdemand[rj]
            for c in merged:
                cust_route[c] = ri
            del routes[rj]
            del rdemand[rj]
        return list(routes.values())

    def sweep_routes():
        if m == 0:
            return []
        angles = sorted(
            active,
            key=lambda i: math.atan2(
                coords[i][1] - coords[0][1], coords[i][0] - coords[0][0]
            ),
        )
        routes = []
        seq = []
        load = 0.0
        for i in angles:
            if seq and (load + rem[i] > Q + 1e-6 or route_info(seq + [i]) is None):
                routes.append(seq)
                seq = [i]
                load = rem[i]
            else:
                if route_info(seq + [i]) is None and seq:
                    routes.append(seq)
                    seq = [i]
                    load = rem[i]
                else:
                    seq.append(i)
                    load += rem[i]
        if seq:
            routes.append(seq)
        return [s for s in routes if route_info(s)]

    def nn_routes_from(start):
        unvisited = set(active)
        seq = [start]
        load = rem[start]
        unvisited.remove(start)
        while True:
            last = seq[-1]
            cand = [
                i
                for i in unvisited
                if load + rem[i] <= Q + 1e-6 and route_info(seq + [i])
            ]
            if not cand:
                break
            i = min(cand, key=lambda i: dist[last][i])
            seq.append(i)
            load += rem[i]
            unvisited.remove(i)
        return seq

    def nn_routes():
        unvisited = set(active)
        routes = []
        while unvisited:
            start = min(unvisited)
            seq = nn_routes_from(start)
            for c in seq:
                unvisited.discard(c)
            routes.append(seq)
        return routes

    heur_seqs_list = []
    heur_seqs_list.append(clarke_wright(False))
    for _ in range(3):
        heur_seqs_list.append(clarke_wright(True))
    heur_seqs_list.append(sweep_routes())
    heur_seqs_list.append(nn_routes())
    for s in list(active)[: min(m, 10)]:
        heur_seqs_list.append([nn_routes_from(s)])

    best_heur_vehicle = direct_reduced
    best_heur_cost = direct_sol["objective_value"]
    for seqs in heur_seqs_list:
        veh = sol_from_seqs(seqs)
        sol = build_solution_dict(full_trucks, veh)
        update_best(sol["objective_value"], sol)
        if sol["objective_value"] < best_heur_cost:
            best_heur_cost = sol["objective_value"]
            best_heur_vehicle = veh

    if gp is None:
        if state["best_sol"] is None:
            state["best_sol"] = build_solution_dict(full_trucks, direct_reduced)
        with open(args.solution_path, "w") as f:
            json.dump(state["best_sol"], f, indent=2)
        return

    def partition_route(seq, agg_q, z):
        vehicles = []
        remaining = {i: agg_q[i] for i in seq if agg_q.get(i, 0.0) > 1e-9}
        if not remaining:
            return vehicles
        for _ in range(z):
            cap = Q
            veh_q = {}
            for i in seq:
                if remaining.get(i, 0.0) <= 1e-9:
                    continue
                d = min(remaining[i], cap)
                if d > 1e-9:
                    veh_q[i] = d
                    remaining[i] -= d
                    cap -= d
            if veh_q:
                subseq = [i for i in seq if veh_q.get(i, 0.0) > 1e-9]
                vehicles.append({"seq": subseq, "q": veh_q})
        return vehicles

    if m <= 40:
        K = m
        model = gp.Model("compact")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        edges = []
        for j in active:
            edges.append((0, j))
        for i in active:
            for j in active:
                if i != j:
                    edges.append((i, j))
        for i in active:
            edges.append((i, sink))
        load_edges = [(0, j) for j in active]
        for i in active:
            for j in active:
                if i != j:
                    load_edges.append((i, j))

        H = max(l0, max(tw[i][1] for i in active))
        max_time = max(max(row) for row in time_mat)
        max_service = max(service)
        Mval = H + max_time + max_service + 1.0

        x = model.addVars(K, edges, vtype=GRB.BINARY)
        y = model.addVars(K, active, vtype=GRB.BINARY)
        q = model.addVars(K, active, lb=0.0)
        for i in active:
            for k in range(K):
                q[k, i].ub = rem[i]
        L = model.addVars(K, load_edges, lb=0.0, ub=Q)
        t = model.addVars(K, active, lb=0.0, ub=H)
        t0 = model.addVars(K, lb=e0, ub=l0)
        tsink = model.addVars(K, lb=0.0, ub=l0)

        for k in range(K):
            model.addConstr(gp.quicksum(x[k, 0, j] for j in active) <= 1)
            model.addConstr(
                gp.quicksum(x[k, i, sink] for i in active)
                == gp.quicksum(x[k, 0, j] for j in active)
            )
            for i in active:
                model.addConstr(
                    gp.quicksum(x[k, h, i] for h in active if h != i) + x[k, 0, i]
                    == y[k, i]
                )
                model.addConstr(
                    gp.quicksum(x[k, i, j] for j in active if j != i) + x[k, i, sink]
                    == y[k, i]
                )

        for i in active:
            model.addConstr(gp.quicksum(q[k, i] for k in range(K)) == rem[i])

        for k in range(K):
            model.addConstr(
                gp.quicksum(q[k, i] for i in active)
                <= Q * gp.quicksum(x[k, 0, j] for j in active)
            )
            for i in active:
                model.addConstr(q[k, i] <= Q * y[k, i])

        for k in range(K):
            model.addConstr(
                gp.quicksum(L[k, 0, j] for j in active)
                == gp.quicksum(q[k, i] for i in active)
            )
            for i in active:
                model.addConstr(
                    L[k, 0, i]
                    + gp.quicksum(L[k, h, i] for h in active if h != i)
                    - q[k, i]
                    == gp.quicksum(L[k, i, j] for j in active if j != i)
                )
            for j in active:
                model.addConstr(L[k, 0, j] <= Q * x[k, 0, j])
            for i in active:
                for j in active:
                    if i != j:
                        model.addConstr(L[k, i, j] <= Q * x[k, i, j])

        for k in range(K):
            for j in active:
                model.addConstr(
                    t[k, j] >= t0[k] + time_mat[0][j] - Mval * (1 - x[k, 0, j])
                )
            for i in active:
                for j in active:
                    if i != j:
                        model.addConstr(
                            t[k, j]
                            >= t[k, i]
                            + service[i]
                            + time_mat[i][j]
                            - Mval * (1 - x[k, i, j])
                        )
                model.addConstr(
                    tsink[k]
                    >= t[k, i]
                    + service[i]
                    + time_mat[i][0]
                    - Mval * (1 - x[k, i, sink])
                )
                model.addConstr(t[k, i] >= tw[i][0] * y[k, i])
                model.addConstr(t[k, i] <= tw[i][1])
            model.addConstr(tsink[k] <= l0)

        for k in range(1, K):
            model.addConstr(
                gp.quicksum(q[k, i] for i in active)
                <= gp.quicksum(q[k - 1, i] for i in active)
            )
            model.addConstr(
                gp.quicksum(x[k, 0, j] for j in active)
                <= gp.quicksum(x[k - 1, 0, j] for j in active)
            )

        obj = gp.LinExpr()
        for k in range(K):
            for j in active:
                obj += dist[0][j] * (a * L[k, 0, j] + b * x[k, 0, j])
            for i in active:
                for j in active:
                    if i != j:
                        obj += dist[i][j] * (a * L[k, i, j] + b * x[k, i, j])
            for i in active:
                obj += dist[i][0] * b * x[k, i, sink]
        model.setObjective(obj, GRB.MINIMIZE)

        warm_sorted = sorted(
            best_heur_vehicle, key=lambda v: sum(v["q"].values()), reverse=True
        )
        for k in range(K):
            t0[k].Start = e0
        for k, veh in enumerate(warm_sorted):
            seq = veh["seq"]
            qd = {c: veh["q"][c] for c in seq if veh["q"][c] > 1e-9}
            if not seq or not qd:
                continue
            x[k, 0, seq[0]].Start = 1
            y[k, seq[0]].Start = 1
            q[k, seq[0]].Start = qd[seq[0]]
            load = sum(qd.values())
            L[k, 0, seq[0]].Start = load
            tt = e0
            prev = 0
            s_prev = 0.0
            for p, c in enumerate(seq):
                if p > 0:
                    pc = seq[p - 1]
                    x[k, pc, c].Start = 1
                    y[k, c].Start = 1
                    q[k, c].Start = qd[c]
                    load -= qd[pc]
                    L[k, pc, c].Start = load
                arr = tt + s_prev + time_mat[prev][c]
                st = arr if arr > tw[c][0] else tw[c][0]
                t[k, c].Start = st
                tt = st
                prev = c
                s_prev = service[c]
            x[k, seq[-1], sink].Start = 1
            tsink[k].Start = tt + s_prev + time_mat[prev][0]

        succ = {0: active}
        for i in active:
            succ[i] = [j for j in active if j != i] + [sink]

        def compact_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                xvals = model.cbGetSolution(x)
                qvals = model.cbGetSolution(q)
                reduced = []
                for k in range(K):
                    seq = []
                    curr = 0
                    while True:
                        nxt = None
                        for j in succ[curr]:
                            if xvals[k, curr, j] > 0.5:
                                nxt = j
                                break
                        if nxt is None or nxt == sink:
                            break
                        seq.append(nxt)
                        curr = nxt
                        if len(seq) > m + 2:
                            break
                    if not seq:
                        continue
                    qd = {}
                    for c in seq:
                        val = qvals[k, c]
                        if val > 1e-6:
                            qd[c] = val
                    if not qd:
                        continue
                    seq2 = [c for c in seq if c in qd]
                    reduced.append({"seq": seq2, "q": qd})
                sol = build_solution_dict(full_trucks, reduced)
                update_best(sol["objective_value"], sol)

        elapsed = time.time() - start_time
        model.Params.TimeLimit = max(1.0, args.time_limit - elapsed - 1.0)
        model.optimize(compact_callback)

        if model.SolCount > 0:
            reduced = []
            for k in range(K):
                seq = []
                curr = 0
                while True:
                    nxt = None
                    for j in succ[curr]:
                        if x[k, curr, j].X > 0.5:
                            nxt = j
                            break
                    if nxt is None or nxt == sink:
                        break
                    seq.append(nxt)
                    curr = nxt
                    if len(seq) > m + 2:
                        break
                if not seq:
                    continue
                qd = {}
                for c in seq:
                    val = q[k, c].X
                    if val > 1e-6:
                        qd[c] = val
                if not qd:
                    continue
                seq2 = [c for c in seq if c in qd]
                reduced.append({"seq": seq2, "q": qd})
            sol = build_solution_dict(full_trucks, reduced)
            update_best(sol["objective_value"], sol)

    else:
        routes_dict = {}
        MAX_ROUTES = 15000
        deadline_gen = start_time + max(1.0, args.time_limit * 0.45)

        def add_route(seq):
            if len(routes_dict) >= MAX_ROUTES:
                return
            if seq in routes_dict:
                return
            info = route_info(seq)
            if info is not None:
                routes_dict[seq] = info

        for i in active:
            add_route((i,))
        for i in active:
            for j in active:
                if i != j:
                    add_route((i, j))

        if m <= 80:
            stack = []
            for s in active:
                if len(routes_dict) >= MAX_ROUTES or time.time() > deadline_gen:
                    break
                stack.append((s,))
            while stack:
                seq = stack.pop()
                info = route_info(seq)
                if info is not None:
                    routes_dict[seq] = info
                if len(seq) >= 3 or len(routes_dict) >= MAX_ROUTES or time.time() > deadline_gen:
                    continue
                for i in active:
                    if i not in seq:
                        newseq = seq + (i,)
                        if route_info(newseq) is not None:
                            stack.append(newseq)

        for seqs in heur_seqs_list:
            for seq in seqs:
                if len(seq) >= 2:
                    add_route(tuple(seq))

        def long_routes():
            for start in active:
                if time.time() > deadline_gen or len(routes_dict) >= MAX_ROUTES:
                    break
                seq = [start]
                unvisited = set(active) - {start}
                while True:
                    last = seq[-1]
                    cand = [i for i in unvisited if route_info(seq + [i])]
                    if not cand:
                        break
                    i = min(cand, key=lambda i: time_mat[last][i])
                    seq.append(i)
                    unvisited.remove(i)
                if len(seq) >= 2:
                    add_route(tuple(seq))

        if len(routes_dict) < MAX_ROUTES:
            long_routes()

        direct_fixed = {i: b * (dist[0][i] + dist[i][0]) for i in active}
        multi = [seq for seq in routes_dict if len(seq) > 1]
        multi.sort(
            key=lambda s: sum(direct_fixed[i] for i in s) - routes_dict[s]["F"],
            reverse=True,
        )
        single = [seq for seq in routes_dict if len(seq) == 1]
        keep = single + multi[: max(0, MAX_ROUTES - len(single))]
        route_list = []
        for seq in keep:
            if seq in routes_dict:
                route_list.append(seq)

        R = len(route_list)
        cust_routes = {i: [] for i in active}
        for idx, seq in enumerate(route_list):
            for i in seq:
                cust_routes[i].append(idx)

        model = gp.Model("routebased")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        z = model.addVars(R, vtype=GRB.INTEGER, lb=0, name="z")
        q = {}
        for idx, seq in enumerate(route_list):
            ub = max(1, int(math.ceil(sum(rem[i] for i in seq) / Q)))
            z[idx].ub = ub
            for i in seq:
                q[(idx, i)] = model.addVar(lb=0.0, ub=rem[i], name=f"q_{idx}_{i}")

        for idx, seq in enumerate(route_list):
            model.addConstr(gp.quicksum(q[(idx, i)] for i in seq) <= Q * z[idx])

        for i in active:
            model.addConstr(
                gp.quicksum(q[(idx, i)] for idx in cust_routes[i]) == rem[i]
            )

        obj = gp.LinExpr()
        for idx, seq in enumerate(route_list):
            info = routes_dict[seq]
            obj += info["F"] * z[idx]
            for i in seq:
                obj += info["coeff"][i] * q[(idx, i)]
        model.setObjective(obj, GRB.MINIMIZE)

        direct_idx = {}
        for idx, seq in enumerate(route_list):
            if len(seq) == 1:
                direct_idx[seq[0]] = idx
        for idx in range(R):
            z[idx].Start = 0
        for key in q:
            q[key].Start = 0
        for i in active:
            if i in direct_idx:
                idx = direct_idx[i]
                z[idx].Start = 1
                q[(idx, i)].Start = rem[i]

        def route_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                zvals = model.cbGetSolution(z)
                qvals = model.cbGetSolution(q)
                reduced = []
                for idx, seq in enumerate(route_list):
                    zv = zvals[idx]
                    if zv < 0.5:
                        continue
                    zc = int(round(zv))
                    agg = {}
                    for i in seq:
                        val = qvals[(idx, i)]
                        if val > 1e-6:
                            agg[i] = val
                    if not agg:
                        continue
                    reduced.extend(partition_route(list(seq), agg, zc))
                sol = build_solution_dict(full_trucks, reduced)
                update_best(sol["objective_value"], sol)

        elapsed = time.time() - start_time
        model.Params.TimeLimit = max(1.0, args.time_limit - elapsed - 1.0)
        model.optimize(route_callback)

        if model.SolCount > 0:
            reduced = []
            for idx, seq in enumerate(route_list):
                zv = z[idx].X
                if zv < 0.5:
                    continue
                zc = int(round(zv))
                agg = {}
                for i in seq:
                    val = q[(idx, i)].X
                    if val > 1e-6:
                        agg[i] = val
                if not agg:
                    continue
                reduced.extend(partition_route(list(seq), agg, zc))
            sol = build_solution_dict(full_trucks, reduced)
            update_best(sol["objective_value"], sol)

    if state["best_sol"] is None:
        state["best_sol"] = build_solution_dict(full_trucks, direct_reduced)
    with open(args.solution_path, "w") as f:
        json.dump(state["best_sol"], f, indent=2)


if __name__ == "__main__":
    main()