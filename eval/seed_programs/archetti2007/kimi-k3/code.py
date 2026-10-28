import argparse
import json
import time
import itertools

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def clean(x):
    r = round(x)
    if abs(x - r) < 1e-6:
        return int(r)
    return float(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t_start = time.time()
    budget = max(1.0, float(args.time_limit))

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["num_retailers"])
    H = int(inst["time_horizon"])
    sup = inst["supplier"]
    rets = inst["retailers"]
    Q = float(inst["vehicle_capacity"])
    C = inst["distance_matrix"]

    sup_lbl = sup["id"]
    ret_lbl = [r["id"] for r in rets]
    labels = [sup_lbl] + ret_lbl
    if sorted(labels) != list(range(n + 1)):
        # fallback to positional indexing if ids do not match matrix indices
        sup_lbl = 0
        ret_lbl = list(range(1, n + 1))

    S0 = float(sup["starting_inventory"])
    prod = float(sup["production_rate"])
    h0 = float(sup["inventory_cost"])

    U = [float(r["max_inventory"]) for r in rets]
    d = [float(r["demand"]) for r in rets]
    s0 = [float(r["starting_inventory"]) for r in rets]
    hc = [float(r["inventory_cost"]) for r in rets]

    def dist(a, b):
        return float(C[a][b])

    RI = list(range(n))
    TI = list(range(1, H + 1))          # delivery periods
    TE = list(range(1, H + 2))          # extended horizon
    pairs = [(i, j) for i in RI for j in RI if j > i]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ------------------------- heuristic -------------------------
    def simulate_retailer(i, vset):
        I = [0.0] * (H + 2)
        I[1] = s0[i]
        q = {}
        viol = 0
        for t in TI:
            if t in vset:
                qq = U[i] - I[t]
                q[t] = qq
                if qq < -1e-9:
                    viol += 1
            nxt = I[t] + q.get(t, 0.0) - d[i]
            if nxt < -1e-9:
                viol += 1
            I[t + 1] = nxt
        return I, q, viol

    def simulate_all(visits):
        Iret, qmap = {}, {}
        viol = 0
        ship = {t: 0.0 for t in TI}
        for i in RI:
            Iv, qv, vv = simulate_retailer(i, visits[i])
            Iret[i] = Iv
            viol += vv
            for t, val in qv.items():
                qmap[(i, t)] = val
                ship[t] += val
        I0 = {1: S0}
        for t in TI:
            if ship[t] > Q + 1e-9:
                viol += 1
            if ship[t] > I0[t] + 1e-9:
                viol += 1
            I0[t + 1] = I0[t] + prod - ship[t]
        return Iret, qmap, I0, ship, viol

    # initial visits: as late as possible (on stockout)
    visits = {i: set() for i in RI}
    for i in RI:
        I = s0[i]
        for t in TI:
            if I - d[i] < -1e-9:
                visits[i].add(t)
                I = U[i]
            I = I - d[i]

    # repair capacity/supplier violations by moving or adding earlier visits
    heur_time_cap = min(0.25 * budget, 25.0)
    for _ in range(400):
        Iret, qmap, I0h, ship, viol = simulate_all(visits)
        if viol == 0 or time.time() - t_start > heur_time_cap:
            break
        tviol = None
        for t in TI:
            if ship[t] > Q + 1e-9 or ship[t] > I0h[t] + 1e-9:
                tviol = t
                break
        if tviol is None:
            fixed = False
            for i in RI:
                for t in TE:
                    if Iret[i][t] < -1e-9 and t - 1 >= 1:
                        visits[i].add(t - 1)
                        fixed = True
                        break
                if fixed:
                    break
            if not fixed:
                break
            continue
        t = tviol
        cand = [i for i in RI if qmap.get((i, t), 0.0) > 1e-9]
        cand.sort(key=lambda i: -qmap.get((i, t), 0.0))
        improved = False
        for i in cand:
            orig = set(visits[i])
            for tp in range(t - 1, 0, -1):
                if tp in orig:
                    continue
                visits[i] = (orig - {t}) | {tp}
                _, _, _, _, v1 = simulate_all(visits)
                if v1 < viol:
                    improved = True
                    break
                visits[i] = orig | {tp}
                _, _, _, _, v2 = simulate_all(visits)
                if v2 < viol:
                    improved = True
                    break
                visits[i] = orig
            if improved:
                break
        if not improved:
            break

    Iret, qmap, I0h, ship, viol = simulate_all(visits)
    heur_feasible = (viol == 0)

    def build_route(nodes):
        if not nodes:
            return []
        if len(nodes) == 1:
            return [[sup_lbl, nodes[0], 2]]
        unv = set(nodes)
        tour = []
        cur = sup_lbl
        while unv:
            nx = min(unv, key=lambda j: dist(cur, j))
            tour.append(nx)
            unv.discard(nx)
            cur = nx
        improved = True
        while improved:
            improved = False
            m_ = len(tour)
            for a in range(m_):
                for b in range(a + 1, m_):
                    A = tour[a - 1] if a > 0 else sup_lbl
                    B = tour[a]
                    Cc = tour[b]
                    D = tour[b + 1] if b < m_ - 1 else sup_lbl
                    delta = (dist(A, B) + dist(Cc, D)) - (dist(A, Cc) + dist(B, D))
                    if delta > 1e-9:
                        tour[a:b + 1] = list(reversed(tour[a:b + 1]))
                        improved = True
        segs = []
        prev = sup_lbl
        for node in tour:
            segs.append([prev, node, 1])
            prev = node
        segs.append([prev, sup_lbl, 1])
        return segs

    def build_output(qmap, I0map, Iret, routes_map):
        hold = h0 * sum(I0map[t] for t in TE)
        for i in RI:
            hold += hc[i] * sum(Iret[i][t] for t in TE)
        trans = 0.0
        deliveries, routes = {}, {}
        for t in TI:
            dd = {}
            for i in RI:
                val = qmap.get((i, t), 0.0)
                if val > 1e-7:
                    dd[str(ret_lbl[i])] = clean(val)
            deliveries[str(t)] = dd
            segs = []
            for seg in routes_map.get(t, []):
                a, b, k = seg
                if k > 0:
                    segs.append([int(a), int(b), int(k)])
                    trans += dist(a, b) * k
            routes[str(t)] = segs
        supinv = {str(t): clean(I0map[t]) for t in TE}
        retinv = {str(ret_lbl[i]): {str(t): clean(Iret[i][t]) for t in TE} for i in RI}
        obj = float(hold + trans)
        sol = {
            "objective_value": obj,
            "deliveries": deliveries,
            "routes": routes,
            "supplier_inventory": supinv,
            "retailer_inventory": retinv,
        }
        return obj, sol

    routes_h = {t: build_route([ret_lbl[i] for i in RI
                                if qmap.get((i, t), 0.0) > 1e-7]) for t in TI}
    obj_h, sol_h = build_output(qmap, I0h, Iret, routes_h)

    best = {"obj": obj_h if heur_feasible else float("inf"),
            "sol": sol_h if heur_feasible else None}
    fallback_sol = sol_h
    if logger and heur_feasible:
        logger.log_solution(obj_h, sol_h)

    # --------------------------- MIP ---------------------------
    mip_obj, mip_sol = None, None
    remaining = budget - (time.time() - t_start)

    if n > 0 and H > 0 and remaining > 2.0:
        try:
            mip = gp.Model("irp")
            mip.setParam("Seed", 0)
            mip.setParam("MIPGap", 1e-4)
            mip.setParam("NumericFocus", 0)
            mip.setParam("Threads", 1)
            mip.setParam("LazyConstraints", 1)
            mip.setParam("OutputFlag", 0)

            y = mip.addVars(RI, TI, vtype=GRB.BINARY, name="y")
            q = mip.addVars(RI, TI, lb=0.0, name="q")
            Ii = mip.addVars(RI, TE, lb=0.0, name="Ii")
            I0v = mip.addVars(TE, lb=0.0, ub=max(0.0, S0 + prod * H), name="I0")
            z0 = mip.addVars(RI, TI, lb=0, ub=2, vtype=GRB.INTEGER, name="z0")
            zz = mip.addVars([(i, j, t) for (i, j) in pairs for t in TI],
                             vtype=GRB.BINARY, name="zz")
            v = mip.addVars(TI, vtype=GRB.BINARY, name="v")

            for i in RI:
                for t in TE:
                    Ii[i, t].setAttr("ub", U[i])
                for t in TI:
                    q[i, t].setAttr("ub", U[i])

            mip.addConstr(I0v[1] == S0)
            for i in RI:
                mip.addConstr(Ii[i, 1] == s0[i])

            incident = {i: [] for i in RI}
            for (a, b) in pairs:
                incident[a].append((a, b))
                incident[b].append((a, b))

            for t in TI:
                sumq = gp.quicksum(q[i, t] for i in RI)
                mip.addConstr(I0v[t + 1] == I0v[t] + prod - sumq)
                mip.addConstr(sumq <= Q)
                mip.addConstr(sumq <= I0v[t])
                mip.addConstr(gp.quicksum(z0[i, t] for i in RI) == 2 * v[t])
                for i in RI:
                    Ui = U[i]
                    mip.addConstr(Ii[i, t + 1] == Ii[i, t] + q[i, t] - d[i])
                    # order-up-to: if visited, ship exactly U_i - I_i(t)
                    mip.addConstr(q[i, t] >= Ui * y[i, t] - Ii[i, t])
                    mip.addConstr(q[i, t] <= 2 * Ui - Ii[i, t] - Ui * y[i, t])
                    mip.addConstr(q[i, t] <= Ui * y[i, t])
                    mip.addConstr(y[i, t] <= v[t])
                    mip.addConstr(
                        z0[i, t] + gp.quicksum(zz[a, b, t] for (a, b) in incident[i])
                        == 2 * y[i, t]
                    )

            obj_expr = gp.quicksum(h0 * I0v[t] for t in TE)
            obj_expr += gp.quicksum(hc[i] * Ii[i, t] for i in RI for t in TE)
            obj_expr += gp.quicksum(dist(sup_lbl, ret_lbl[i]) * z0[i, t]
                                    for i in RI for t in TI)
            obj_expr += gp.quicksum(dist(ret_lbl[i], ret_lbl[j]) * zz[i, j, t]
                                    for (i, j) in pairs for t in TI)
            mip.setObjective(obj_expr, GRB.MINIMIZE)

            # warm start from heuristic
            if heur_feasible:
                label_to_i = {ret_lbl[i]: i for i in RI}
                for t in TI:
                    v[t].Start = 1.0 if ship.get(t, 0.0) > 1e-7 else 0.0
                    for i in RI:
                        z0[i, t].Start = 0.0
                    for (a, b) in pairs:
                        zz[a, b, t].Start = 0.0
                    seg_sup = {}
                    for seg in routes_h.get(t, []):
                        a, b, k = seg
                        if a == sup_lbl or b == sup_lbl:
                            other = b if a == sup_lbl else a
                            ii = label_to_i.get(other)
                            if ii is not None:
                                seg_sup[ii] = seg_sup.get(ii, 0) + int(k)
                        else:
                            ia, ib = label_to_i.get(a), label_to_i.get(b)
                            if ia is not None and ib is not None:
                                ii, jj = (ia, ib) if ia < ib else (ib, ia)
                                zz[ii, jj, t].Start = 1.0
                    for ii, k in seg_sup.items():
                        z0[ii, t].Start = float(min(2, k))
                for i in RI:
                    for t in TI:
                        y[i, t].Start = 1.0 if t in visits[i] else 0.0
                        q[i, t].Start = qmap.get((i, t), 0.0)
                    for t in TE:
                        Ii[i, t].Start = Iret[i][t]
                for t in TE:
                    I0v[t].Start = I0h.get(t, S0)

            # prebuilt variable lists for the callback
            y_list_t = {t: [y[i, t] for i in RI] for t in TI}
            z0_list_t = {t: [z0[i, t] for i in RI] for t in TI}
            zz_list_t = {t: [zz[i, j, t] for (i, j) in pairs] for t in TI}
            q_all = [q[i, t] for i in RI for t in TI]
            Ii_all = [Ii[i, t] for i in RI for t in TE]
            I0_all = [I0v[t] for t in TE]
            z0_all = [z0[i, t] for i in RI for t in TI]
            zz_all = [zz[i, j, t] for (i, j) in pairs for t in TI]

            def cb(model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    cuts = 0
                    for t in TI:
                        yvars = y_list_t[t]
                        yv = model.cbGetSolution(yvars)
                        visited = [i for i in RI if yv[i] > 0.5]
                        if len(visited) < 2:
                            continue
                        z0v = model.cbGetSolution(z0_list_t[t])
                        zzv = model.cbGetSolution(zz_list_t[t])
                        parent = {i: i for i in visited}

                        def find(x):
                            while parent[x] != x:
                                parent[x] = parent[parent[x]]
                                x = parent[x]
                            return x

                        for idx, (a, b) in enumerate(pairs):
                            if zzv[idx] > 0.5 and yv[a] > 0.5 and yv[b] > 0.5:
                                ra, rb = find(a), find(b)
                                if ra != rb:
                                    parent[ra] = rb
                        comps = {}
                        for i in visited:
                            comps.setdefault(find(i), []).append(i)
                        for members in comps.values():
                            if len(members) < 2:
                                continue
                            if any(z0v[i] > 0.5 for i in members):
                                continue
                            # retailer-only subtour disconnected from supplier
                            sub = [(a, b) for a, b in
                                   itertools.combinations(sorted(members), 2)]
                            lhs = gp.quicksum(zz[a, b, t] for (a, b) in sub)
                            rhs = (gp.quicksum(y[i, t] for i in members)
                                   - y[members[0], t])
                            model.cbLazy(lhs <= rhs)
                            cuts += 1
                    if cuts == 0:
                        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if obj_val < best["obj"] - 1e-7:
                            qv = model.cbGetSolution(q_all)
                            Iiv = model.cbGetSolution(Ii_all)
                            I0vv = model.cbGetSolution(I0_all)
                            z0vv = model.cbGetSolution(z0_all)
                            zzvv = model.cbGetSolution(zz_all)
                            qmap2, idx = {}, 0
                            for i in RI:
                                for t in TI:
                                    qmap2[(i, t)] = qv[idx]
                                    idx += 1
                            Iret2, idx = {}, 0
                            for i in RI:
                                Iret2[i] = [0.0] * (H + 2)
                                for t in TE:
                                    Iret2[i][t] = Iiv[idx]
                                    idx += 1
                            I0m = {t: I0vv[kk] for kk, t in enumerate(TE)}
                            routes_m = {t: [] for t in TI}
                            idx = 0
                            for i in RI:
                                for t in TI:
                                    k = int(round(z0vv[idx]))
                                    idx += 1
                                    if k > 0:
                                        routes_m[t].append([sup_lbl, ret_lbl[i], k])
                            idx = 0
                            for (a, b) in pairs:
                                for t in TI:
                                    if zzvv[idx] > 0.5:
                                        routes_m[t].append([ret_lbl[a], ret_lbl[b], 1])
                                    idx += 1
                            obj2, sol2 = build_output(qmap2, I0m, Iret2, routes_m)
                            best["obj"] = obj2
                            best["sol"] = sol2
                            if logger:
                                logger.log_solution(obj2, sol2)
                except Exception:
                    pass

            remaining2 = budget - (time.time() - t_start)
            mip.setParam("TimeLimit", max(0.2, remaining2 - 1.0))
            mip.optimize(cb)

            if mip.SolCount > 0:
                qmap2 = {(i, t): q[i, t].X for i in RI for t in TI}
                Iret2 = {i: [0.0] * (H + 2) for i in RI}
                for i in RI:
                    for t in TE:
                        Iret2[i][t] = Ii[i, t].X
                I0m = {t: I0v[t].X for t in TE}
                routes_m = {t: [] for t in TI}
                for i in RI:
                    for t in TI:
                        k = int(round(z0[i, t].X))
                        if k > 0:
                            routes_m[t].append([sup_lbl, ret_lbl[i], k])
                for (a, b) in pairs:
                    for t in TI:
                        if zz[a, b, t].X > 0.5:
                            routes_m[t].append([ret_lbl[a], ret_lbl[b], 1])
                mip_obj, mip_sol = build_output(qmap2, I0m, Iret2, routes_m)
        except Exception:
            mip_obj, mip_sol = None, None

    if mip_sol is not None and mip_obj < best["obj"] - 1e-9:
        best["obj"] = mip_obj
        best["sol"] = mip_sol

    final_sol = best["sol"] if best["sol"] is not None else fallback_sol
    final_obj = best["obj"] if best["sol"] is not None else fallback_sol["objective_value"]

    if logger and best["sol"] is not None:
        logger.log_solution(final_obj, final_sol)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()