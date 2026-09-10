import argparse
import json
import time
import itertools

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data["num_retailers"]
    H = data["time_horizon"]
    sup = data["supplier"]
    sup_id = sup["id"]
    B0 = float(sup["starting_inventory"])
    prod = float(sup["production_rate"])
    h0 = float(sup["inventory_cost"])
    C = float(data["vehicle_capacity"])
    c = data["distance_matrix"]

    R = []
    d = {}
    U = {}
    I0 = {}
    h = {}
    for rt in data["retailers"]:
        i = rt["id"]
        R.append(i)
        d[i] = float(rt["demand"])
        U[i] = float(rt["max_inventory"])
        I0[i] = float(rt["starting_inventory"])
        h[i] = float(rt["inventory_cost"])
    R = sorted(R)

    T = list(range(1, H + 1))
    Text = list(range(1, H + 2))

    # ------------------------------------------------------------------ helpers
    def compute_inv(deliveries):
        I = {}
        B = {}
        for i in R:
            I[(i, 1)] = I0[i]
        B[1] = B0
        for tau in range(2, H + 2):
            shipped = sum(deliveries.get(tau - 1, {}).values())
            B[tau] = B[tau - 1] + prod - shipped
            for i in R:
                I[(i, tau)] = I[(i, tau - 1)] + deliveries.get(tau - 1, {}).get(i, 0.0) - d[i]
        return I, B

    def make_solution(deliveries, routes):
        I, B = compute_inv(deliveries)
        transport = 0.0
        for t, segs in routes.items():
            for a, b, k in segs:
                transport += c[a][b] * k
        obj = h0 * sum(B[tau] for tau in Text)
        obj += sum(h[i] * I[(i, tau)] for i in R for tau in Text)
        obj += transport
        sol = {
            "objective_value": float(round(obj, 6)),
            "deliveries": {
                str(t): {str(i): float(round(q, 6)) for i, q in dv.items()}
                for t, dv in sorted(deliveries.items()) if dv
            },
            "routes": {
                str(t): [[int(a), int(b), int(k)] for a, b, k in segs]
                for t, segs in sorted(routes.items()) if segs
            },
            "supplier_inventory": {
                str(tau): float(round(B[tau] if abs(B[tau]) > 1e-6 else 0.0, 6)) for tau in Text
            },
            "retailer_inventory": {
                str(i): {
                    str(tau): float(round(I[(i, tau)] if abs(I[(i, tau)]) > 1e-6 else 0.0, 6))
                    for tau in Text
                }
                for i in R
            },
        }
        return obj, sol

    def nn_route(visited):
        visited = list(visited)
        if not visited:
            return []
        if len(visited) == 1:
            return [[sup_id, visited[0], 2]]
        segs = []
        cur = sup_id
        unv = set(visited)
        while unv:
            nxt = min(unv, key=lambda j: c[cur][j])
            segs.append([cur, nxt, 1])
            cur = nxt
            unv.remove(nxt)
        segs.append([cur, sup_id, 1])
        return segs

    # ------------------------------------------------------------------ greedy heuristic
    def greedy(m_factor):
        Icur = {i: I0[i] for i in R}
        Bcur = B0
        deliveries = {}
        for t in T:
            must = [i for i in R if Icur[i] < d[i] - 1e-9]
            need = {i: U[i] - Icur[i] for i in R}
            total = sum(need[i] for i in must)
            cap = min(C, Bcur)
            if total > cap + 1e-9:
                return None
            visited = set(must)
            rem = cap - total
            extras = sorted(
                [i for i in R if i not in visited and Icur[i] < m_factor * d[i] and need[i] > 1e-9],
                key=lambda i: Icur[i] / max(d[i], 1e-9),
            )
            for i in extras:
                if need[i] <= rem + 1e-9:
                    visited.add(i)
                    rem -= need[i]
            dv = {}
            for i in R:
                if i in visited:
                    dv[i] = need[i]
                    Icur[i] = U[i] - d[i]
                    if Icur[i] < -1e-9:
                        return None
                else:
                    Icur[i] = Icur[i] - d[i]
                    if Icur[i] < -1e-9:
                        return None
            deliveries[t] = dv
            Bcur = Bcur + prod - sum(dv.values())
            if Bcur < -1e-9:
                return None
        return deliveries

    best_greedy = None  # (obj, sol, deliveries, routes)
    for mf in [1.0, 1.5, 2.0, 3.0]:
        dv = greedy(mf)
        if dv is None:
            continue
        routes = {t: nn_route(list(dv[t].keys())) for t in T if dv.get(t)}
        obj, sol = make_solution(dv, routes)
        if best_greedy is None or obj < best_greedy[0] - 1e-9:
            best_greedy = (obj, sol, dv, routes)

    if best_greedy is not None and logger:
        logger.log_solution(best_greedy[0], best_greedy[1])

    # ------------------------------------------------------------------ MIP model
    remaining = args.time_limit - (time.time() - t_start) - 2.0
    final_sol = best_greedy[1] if best_greedy is not None else None
    incumbent_holder = [final_sol]

    if remaining > 3.0:
        model = gp.Model("irp")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = max(3.0, remaining)

        z = model.addVars(R, T, vtype=GRB.BINARY, name="z")
        z0 = model.addVars(T, vtype=GRB.BINARY, name="z0")
        x = model.addVars(R, T, lb=0.0, name="x")
        Iv = model.addVars(R, Text, lb=0.0, name="I")
        Bv = model.addVars(Text, lb=0.0, name="B")

        Epairs = [(i, j) for idx, i in enumerate(R) for j in R[idx + 1:]]
        ye = model.addVars(Epairs, T, vtype=GRB.BINARY, name="ye")
        y0 = model.addVars(R, T, vtype=GRB.INTEGER, lb=0, ub=2, name="y0")

        # inventory dynamics
        for i in R:
            model.addConstr(Iv[i, 1] == I0[i])
            for tau in range(2, H + 2):
                model.addConstr(Iv[i, tau] == Iv[i, tau - 1] + x[i, tau - 1] - d[i])
        model.addConstr(Bv[1] == B0)
        for tau in range(2, H + 2):
            model.addConstr(Bv[tau] == Bv[tau - 1] + prod - gp.quicksum(x[i, tau - 1] for i in R))
        for t in T:
            model.addConstr(Bv[t] >= gp.quicksum(x[i, t] for i in R))
            model.addConstr(gp.quicksum(x[i, t] for i in R) <= C * z0[t])

        # order-up-to policy
        for i in R:
            for t in T:
                model.addConstr(x[i, t] <= U[i] * z[i, t])
                model.addConstr(x[i, t] <= U[i] - Iv[i, t])
                model.addConstr(x[i, t] >= U[i] * z[i, t] - Iv[i, t])
                model.addConstr(z[i, t] <= z0[t])

        # routing degree constraints
        for t in T:
            for i in R:
                model.addConstr(
                    gp.quicksum(ye[a, b, t] for (a, b) in Epairs if a == i or b == i) + y0[i, t]
                    == 2 * z[i, t]
                )
                model.addConstr(y0[i, t] <= 2 * z[i, t])
            model.addConstr(gp.quicksum(y0[i, t] for i in R) == 2 * z0[t])
            for (a, b) in Epairs:
                model.addConstr(ye[a, b, t] <= z[a, t])
                model.addConstr(ye[a, b, t] <= z[b, t])

        obj_expr = h0 * gp.quicksum(Bv[tau] for tau in Text)
        obj_expr += gp.quicksum(h[i] * Iv[i, tau] for i in R for tau in Text)
        obj_expr += gp.quicksum(c[a][b] * ye[a, b, t] for (a, b) in Epairs for t in T)
        obj_expr += gp.quicksum(c[sup_id][i] * y0[i, t] for i in R for t in T)
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # MIP start from greedy (track start values in local dicts, never read .Start)
        if best_greedy is not None:
            _, _, gdv, groutes = best_greedy
            for t in T:
                dv = gdv.get(t, {})
                z0[t].Start = 1 if dv else 0
                for i in R:
                    z[i, t].Start = 1 if i in dv else 0
                    x[i, t].Start = dv.get(i, 0.0)
                y0_start = {i: 0 for i in R}
                ye_start = {(a, b): 0 for (a, b) in Epairs}
                for seg in groutes.get(t, []):
                    a, b, k = seg
                    if a == sup_id or b == sup_id:
                        i = b if a == sup_id else a
                        y0_start[i] += k
                    else:
                        aa, bb = (a, b) if a < b else (b, a)
                        ye_start[(aa, bb)] = 1
                for i in R:
                    y0[i, t].Start = y0_start[i]
                for (a, b) in Epairs:
                    ye[a, b, t].Start = ye_start[(a, b)]

        best_logged = [best_greedy[0] if best_greedy is not None else float("inf")]

        def get_vals(model, vd, keys):
            vals = model.cbGetSolution([vd[k] for k in keys])
            return dict(zip(keys, vals))

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            zkeys = [(i, t) for i in R for t in T]
            zv = get_vals(model, z, zkeys)
            yekeys = [(a, b, t) for (a, b) in Epairs for t in T]
            yev = get_vals(model, ye, yekeys)
            y0keys = [(i, t) for i in R for t in T]
            y0v = get_vals(model, y0, y0keys)

            violated = False
            for t in T:
                vis = [i for i in R if zv[(i, t)] > 0.5]
                if not vis:
                    continue
                parent = {sup_id: sup_id}
                for i in vis:
                    parent[i] = i

                def find(a):
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    return a

                def union(a, b):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb

                for (a, b) in Epairs:
                    if a in parent and b in parent and yev[(a, b, t)] > 0.5:
                        union(a, b)
                for i in vis:
                    if y0v[(i, t)] > 0.5:
                        union(sup_id, i)
                root0 = find(sup_id)
                comps = {}
                for i in vis:
                    rt_ = find(i)
                    comps.setdefault(rt_, []).append(i)
                for rt_, S in comps.items():
                    if rt_ == root0:
                        continue
                    violated = True
                    S = sorted(S)
                    inner = gp.quicksum(
                        ye[a, b, t] for (a, b) in itertools.combinations(S, 2)
                    )
                    zsum = gp.quicksum(z[i, t] for i in S)
                    for k in S:
                        model.cbLazy(inner <= zsum - z[k, t])

            if not violated:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_logged[0] - 1e-6:
                    xkeys = [(i, t) for i in R for t in T]
                    xv = get_vals(model, x, xkeys)
                    deliveries = {}
                    routes = {}
                    for t in T:
                        dvv = {i: max(0.0, xv[(i, t)]) for i in R if zv[(i, t)] > 0.5}
                        if dvv:
                            deliveries[t] = dvv
                        segs = []
                        for (a, b) in Epairs:
                            if yev[(a, b, t)] > 0.5:
                                segs.append([a, b, 1])
                        for i in R:
                            k = int(round(y0v[(i, t)]))
                            if k >= 1:
                                segs.append([sup_id, i, k])
                        if segs:
                            routes[t] = segs
                    tobj, sol = make_solution(deliveries, routes)
                    best_logged[0] = obj
                    incumbent_holder[0] = sol
                    if logger:
                        logger.log_solution(tobj, sol)

        try:
            model.optimize(callback)
        except gp.GurobiError:
            pass

        if model.SolCount > 0:
            deliveries = {}
            routes = {}
            for t in T:
                dvv = {i: max(0.0, x[i, t].X) for i in R if z[i, t].X > 0.5}
                if dvv:
                    deliveries[t] = dvv
                segs = []
                for (a, b) in Epairs:
                    if ye[a, b, t].X > 0.5:
                        segs.append([a, b, 1])
                for i in R:
                    k = int(round(y0[i, t].X))
                    if k >= 1:
                        segs.append([sup_id, i, k])
                if segs:
                    routes[t] = segs
            obj, sol = make_solution(deliveries, routes)
            if final_sol is None or obj < final_sol["objective_value"] - 1e-9:
                final_sol = sol
                if logger and obj < best_logged[0] - 1e-6:
                    logger.log_solution(obj, sol)

        if incumbent_holder[0] is not None:
            if final_sol is None or incumbent_holder[0]["objective_value"] < final_sol["objective_value"] - 1e-9:
                final_sol = incumbent_holder[0]

    # ------------------------------------------------------------------ fallback
    if final_sol is None:
        obj, final_sol = make_solution({}, {})
        if logger:
            logger.log_solution(obj, final_sol)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()