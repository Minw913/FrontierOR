import argparse
import json
import math
import time
from collections import defaultdict


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def snap(x, tol=1e-6):
    r = round(x)
    if abs(x - r) <= tol:
        return int(r)
    return float(x)


# ----------------------------------------------------------------------
# Route ordering: nearest neighbour + 2-opt
# ----------------------------------------------------------------------
def order_route(nodes, dist):
    if not nodes:
        return []
    remaining = set(nodes)
    route = []
    cur = 0
    while remaining:
        nxt = min(remaining, key=lambda i: (dist[cur][i], i))
        route.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    improved = True
    while improved:
        improved = False
        for a in range(len(route)):
            for b in range(a + 1, len(route)):
                i0 = route[a - 1] if a > 0 else 0
                i1 = route[a]
                j0 = route[b]
                j1 = route[b + 1] if b + 1 < len(route) else 0
                if dist[i0][i1] + dist[j0][j1] > dist[i0][j0] + dist[i1][j1] + 1e-12:
                    route[a:b + 1] = route[a:b + 1][::-1]
                    improved = True
    return route


# ----------------------------------------------------------------------
# Clarke-Wright savings (parallel) for one period's delivery set
# ----------------------------------------------------------------------
def clarke_wright(loads, dist, C):
    routes = {i: [i] for i in loads}
    load = {i: loads[i] for i in loads}
    route_of = {i: i for i in loads}
    ks = list(loads.keys())
    sav = []
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            i, j = ks[a], ks[b]
            sav.append((dist[0][i] + dist[0][j] - dist[i][j], i, j))
    sav.sort(reverse=True)
    for s, i, j in sav:
        ri = route_of.get(i)
        rj = route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        if load[ri] + load[rj] > C + 1e-9:
            continue
        Ri = routes[ri]
        Rj = routes[rj]
        if Ri[-1] == i and Rj[0] == j:
            new = Ri + Rj
        elif Ri[0] == i and Rj[-1] == j:
            new = Rj + Ri
        elif Ri[0] == i and Rj[0] == j:
            new = Ri[::-1] + Rj
        elif Ri[-1] == i and Rj[-1] == j:
            new = Ri + Rj[::-1]
        else:
            continue
        routes[ri] = new
        load[ri] = load[ri] + load[rj]
        for c in Rj:
            route_of[c] = ri
        del routes[rj]
        del load[rj]
    return list(routes.values())


def route_cost(routes, dist):
    c = 0.0
    for rt in routes:
        full = [0] + rt + [0]
        for a, b in zip(full[:-1], full[1:]):
            c += dist[a][b]
    return c


def route_period(loads, dist, V, C):
    """Try several constructions; return best feasible routing or None."""
    items = {i: q for i, q in loads.items() if q > 1e-9}
    if not items:
        return []
    for q in items.values():
        if q > C + 1e-6:
            return None
    candidates = []
    # Clarke-Wright savings
    try:
        cw = clarke_wright(items, dist, C)
        if len(cw) <= V:
            candidates.append(cw)
    except Exception:
        pass
    # Greedy nearest-neighbour insertion
    un = dict(items)
    routes = []
    ok = True
    while un:
        rn = []
        rl = 0.0
        cur = 0
        while True:
            best = None
            bc = None
            for i, q in un.items():
                if rl + q <= C + 1e-9:
                    cc = dist[cur][i]
                    if bc is None or cc < bc - 1e-12:
                        best = i
                        bc = cc
            if best is None:
                break
            rn.append(best)
            rl += un.pop(best)
            cur = best
        routes.append(rn)
        if len(routes) > V:
            ok = False
            break
    if ok and not un and len(routes) <= V:
        candidates.append(routes)
    # Best-fit decreasing bin packing
    routes = []
    ok = True
    for i, q in sorted(items.items(), key=lambda kv: (-kv[1], kv[0])):
        br = None
        bres = None
        for rr in routes:
            if rr[1] + q <= C + 1e-9:
                res = C - (rr[1] + q)
                if bres is None or res < bres - 1e-12:
                    br = rr
                    bres = res
        if br is None:
            routes.append([[i], q])
            if len(routes) > V:
                ok = False
                break
        else:
            br[0].append(i)
            br[1] += q
    if ok and len(routes) <= V:
        candidates.append([rr[0] for rr in routes])
    if not candidates:
        return None
    best_routes = None
    best_cost = None
    for cand in candidates:
        cand = [order_route(rt, dist) for rt in cand]
        c = route_cost(cand, dist)
        if best_cost is None or c < best_cost - 1e-9:
            best_cost = c
            best_routes = cand
    return best_routes


# ----------------------------------------------------------------------
# Constructive delivery patterns
# ----------------------------------------------------------------------
def build_deliveries(inst, mode):
    n = inst["num_customers"]
    T = inst["num_periods"]
    N = inst["num_nodes"]
    dem = inst["customer_demands"]
    prod = inst["supplier_production"]
    cap = inst["inventory_capacities"]
    init = inst["initial_inventories"]
    customers = list(range(1, N))
    deliv = [[0.0] * T for _ in range(N)]
    inv = [float(init[i]) for i in range(N)]
    cum_dem = [[0.0] * (T + 1) for _ in range(n)]
    for ci in range(n):
        for t in range(T - 1, -1, -1):
            cum_dem[ci][t] = cum_dem[ci][t + 1] + dem[ci][t]
    for t in range(T):
        for i in customers:
            ci = i - 1
            di = dem[ci][t]
            if mode == "jit":
                dd = max(0.0, di - inv[i])
            elif mode == "fillcap":
                dd = (cap[i] - inv[i]) if inv[i] < di - 1e-9 else 0.0
            else:  # fillrem: cover all remaining demand, limited by capacity
                if inv[i] < di - 1e-9:
                    dd = min(cap[i] - inv[i], cum_dem[ci][t] - inv[i])
                else:
                    dd = 0.0
            dd = min(dd, cap[i] - inv[i])
            if dd < -1e-9:
                return None
            deliv[i][t] = max(0.0, dd)
        ship = sum(deliv[i][t] for i in customers)
        inv0 = inv[0] + prod[t]
        # resolve depot capacity overflow by pushing extra product to customers
        extra = inv0 - ship - cap[0]
        if extra > 1e-9:
            order = sorted(customers,
                           key=lambda i: (deliv[i][t] <= 0.0,
                                          -(cap[i] - inv[i] - deliv[i][t])))
            for i in order:
                if extra <= 1e-9:
                    break
                free = cap[i] - inv[i] - deliv[i][t]
                add = min(free, extra)
                if add > 1e-9:
                    deliv[i][t] += add
                    ship += add
                    extra -= add
            if extra > 1e-6:
                return None
        if inv0 - ship < -1e-6:
            return None
        inv[0] = inv0 - ship
        for i in customers:
            inv[i] = inv[i] + deliv[i][t] - dem[i - 1][t]
            if inv[i] < -1e-6:
                return None
            if inv[i] < 0:
                inv[i] = 0.0
    return deliv


def route_all_periods(inst, deliv):
    T = inst["num_periods"]
    N = inst["num_nodes"]
    V = inst["num_vehicles"]
    C = float(inst["vehicle_capacity"])
    dist = inst["distance_matrix"]
    customers = list(range(1, N))
    routes_by_t = []
    for t in range(T):
        loads = {i: deliv[i][t] for i in customers if deliv[i][t] > 1e-9}
        rts = route_period(loads, dist, V, C)
        if rts is None:
            return None
        routes_by_t.append(rts)
    return routes_by_t


# ----------------------------------------------------------------------
# Solution assembly / evaluation
# ----------------------------------------------------------------------
def build_solution_dict(inst, deliv, routes_by_t):
    T = inst["num_periods"]
    V = inst["num_vehicles"]
    N = inst["num_nodes"]
    dist = inst["distance_matrix"]
    h = inst["holding_costs"]
    dem = inst["customer_demands"]
    prod = inst["supplier_production"]
    init = inst["initial_inventories"]
    customers = list(range(1, N))
    sd = [[snap(deliv[i][t]) for t in range(T)] for i in range(N)]
    inv = [float(init[i]) for i in range(N)]
    routes_json = {}
    inventories = {str(i): {} for i in range(N)}
    travel = 0.0
    holding = 0.0
    for t in range(T):
        per = {}
        rts = routes_by_t[t] if t < len(routes_by_t) else []
        for v in range(V):
            if v < len(rts):
                rt = rts[v]
                full = [0] + list(rt) + [0]
                dmap = {str(i): sd[i][t] for i in rt}
                per[str(v)] = {"route": full, "deliveries": dmap}
                for a, b in zip(full[:-1], full[1:]):
                    travel += dist[a][b]
            else:
                per[str(v)] = {"route": [0], "deliveries": {}}
        routes_json[str(t)] = per
        ship = 0.0
        for i in customers:
            ship += sd[i][t]
        inv[0] = inv[0] + prod[t] - ship
        for i in customers:
            inv[i] = inv[i] + sd[i][t] - dem[i - 1][t]
        for i in range(N):
            val = inv[i]
            if -1e-6 < val < 0:
                val = 0.0
            inventories[str(i)][str(t)] = snap(val)
            holding += h[i] * val
    obj = travel + holding
    return {"objective_value": float(obj),
            "routes": routes_json,
            "inventories": inventories}, float(obj)


# ----------------------------------------------------------------------
# Decompose a period's selected undirected edges into depot routes
# ----------------------------------------------------------------------
def decompose_routes(sel, N):
    adj = defaultdict(list)
    for (i, j) in sel:
        adj[i].append(j)
        adj[j].append(i)
    routes = []
    while adj[0]:
        nb = adj[0].pop()
        adj[nb].remove(0)
        route = []
        cur = nb
        while cur != 0:
            route.append(cur)
            if not adj[cur]:
                break
            nxt = adj[cur].pop()
            adj[nxt].remove(cur)
            cur = nxt
        routes.append(route)
    leftover = []
    seen = set()
    for u in list(adj.keys()):
        if u == 0 or not adj[u] or u in seen:
            continue
        stack = [u]
        seen.add(u)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for w in adj[x]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        leftover.append(comp)
    return routes, leftover


# ----------------------------------------------------------------------
# Gurobi branch-and-cut (two-index formulation + lazy capacity/SEC cuts)
# ----------------------------------------------------------------------
def solve_with_gurobi(inst, best, deadline, logger):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return best
    try:
        T = inst["num_periods"]
        V = inst["num_vehicles"]
        C = float(inst["vehicle_capacity"])
        N = inst["num_nodes"]
        dist = inst["distance_matrix"]
        h = inst["holding_costs"]
        dem = inst["customer_demands"]
        prod = inst["supplier_production"]
        cap = inst["inventory_capacities"]
        init = inst["initial_inventories"]
        customers = list(range(1, N))
        edge_list = [(i, j) for i in range(N) for j in range(i + 1, N)]

        m = gp.Model("irp")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.LazyConstraints = 1
        m.Params.OutputFlag = 0
        remaining = deadline - time.time() - 1.0
        if remaining < 0.5:
            return best
        m.Params.TimeLimit = max(0.5, remaining)

        e = {}
        for (i, j) in edge_list:
            cij = float(dist[i][j])
            for t in range(T):
                e[(i, j, t)] = m.addVar(vtype=GRB.BINARY, obj=cij,
                                        name="x_%d_%d_%d" % (i, j, t))
        z = {}
        dvar = {}
        for i in customers:
            ub = min(C, float(cap[i]))
            for t in range(T):
                z[(i, t)] = m.addVar(vtype=GRB.BINARY, name="z_%d_%d" % (i, t))
                dvar[(i, t)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=ub,
                                        name="d_%d_%d" % (i, t))
        invv = {}
        for i in range(N):
            for t in range(T):
                invv[(i, t)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                        ub=float(cap[i]), obj=float(h[i]),
                                        name="inv_%d_%d" % (i, t))
        rvar = {}
        for t in range(T):
            rvar[t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=V, name="r_%d" % t)
        m.ModelSense = GRB.MINIMIZE
        m.update()

        incident = [[] for _ in range(N)]
        for (i, j) in edge_list:
            incident[i].append((i, j))
            incident[j].append((i, j))

        for i in customers:
            ci = i - 1
            for t in range(T):
                prev = invv[(i, t - 1)] if t > 0 else float(init[i])
                m.addConstr(invv[(i, t)] == prev + dvar[(i, t)] - dem[ci][t])
                m.addConstr(prev + dvar[(i, t)] <= float(cap[i]))
                m.addConstr(dvar[(i, t)] <= min(C, float(cap[i])) * z[(i, t)])
                m.addConstr(gp.quicksum(e[(a, b, t)] for (a, b) in incident[i])
                            == 2 * z[(i, t)])
        for t in range(T):
            prev0 = invv[(0, t - 1)] if t > 0 else float(init[0])
            m.addConstr(invv[(0, t)] == prev0 + float(prod[t])
                        - gp.quicksum(dvar[(i, t)] for i in customers))
            m.addConstr(gp.quicksum(e[(a, b, t)] for (a, b) in incident[0])
                        == 2 * rvar[t])
            m.addConstr(gp.quicksum(dvar[(i, t)] for i in customers)
                        <= C * rvar[t])
        m.update()

        # MIP start from best heuristic solution
        if best is not None and best[1] is not None:
            try:
                deliv0 = best[1]
                routes0 = best[2]
                eset = set()
                for t in range(T):
                    for rt in routes0[t]:
                        full = [0] + list(rt) + [0]
                        for a, b in zip(full[:-1], full[1:]):
                            eset.add((min(a, b), max(a, b), t))
                for key, var in e.items():
                    var.Start = 1.0 if key in eset else 0.0
                for i in customers:
                    for t in range(T):
                        q = deliv0[i][t]
                        dvar[(i, t)].Start = float(q)
                        z[(i, t)].Start = 1.0 if q > 1e-9 else 0.0
                invs = [float(init[i]) for i in range(N)]
                for t in range(T):
                    ship = sum(deliv0[i][t] for i in customers)
                    invs[0] = invs[0] + float(prod[t]) - ship
                    invv[(0, t)].Start = invs[0]
                    rvar[t].Start = float(len(routes0[t]))
                    for i in customers:
                        invs[i] = invs[i] + deliv0[i][t] - dem[i - 1][t]
                        invv[(i, t)].Start = invs[i]
            except Exception:
                pass

        state = {"last_logged": best[0] if best is not None else float("inf")}

        def crossing(S, t):
            return gp.quicksum(e[(i, j, t)] for (i, j) in edge_list
                               if ((i in S) != (j in S)))

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            if time.time() > deadline - 0.8:
                model.terminate()
                return
            try:
                ev = model.cbGetSolution(e)
                dv = model.cbGetSolution(dvar)
            except Exception:
                return
            cuts = []
            for t in range(T):
                sel = [(i, j) for (i, j) in edge_list if ev[(i, j, t)] > 0.5]
                if not sel:
                    continue
                routes, leftover = decompose_routes(sel, N)
                for comp in leftover:
                    S = set(comp)
                    D = sum(dv[(i, t)] for i in comp)
                    if D > 1e-6:
                        kk = int(math.ceil(D / C - 1e-9))
                        cuts.append(crossing(S, t) >= 2 * kk)
                    else:
                        cuts.append(crossing(S, t) >= 2 * z[(comp[0], t)])
                for R in routes:
                    D = sum(dv[(i, t)] for i in R)
                    if D > C + 1e-6:
                        kk = int(math.ceil(D / C - 1e-9))
                        cuts.append(crossing(set(R), t) >= 2 * kk)
            if cuts:
                for cc in cuts:
                    try:
                        model.cbLazy(cc)
                    except Exception:
                        pass
                return
            # Fully feasible incumbent: build and log it
            try:
                deliv = [[0.0] * T for _ in range(N)]
                for i in customers:
                    for t in range(T):
                        deliv[i][t] = dv[(i, t)]
                routes_by_t = []
                for t in range(T):
                    sel = [(i, j) for (i, j) in edge_list if ev[(i, j, t)] > 0.5]
                    rts, _ = decompose_routes(sel, N)
                    routes_by_t.append(rts)
                sdict, obj = build_solution_dict(inst, deliv, routes_by_t)
                if obj < state["last_logged"] - 1e-6:
                    state["last_logged"] = obj
                    if logger is not None:
                        logger.log_solution(obj, sdict)
            except Exception:
                pass

        try:
            m.optimize(callback)
        except Exception:
            return best

        try:
            if m.SolCount > 0:
                ev = m.getAttr("X", e)
                dv = m.getAttr("X", dvar)
                deliv = [[0.0] * T for _ in range(N)]
                for i in customers:
                    for t in range(T):
                        deliv[i][t] = dv[(i, t)]
                routes_by_t = []
                feasible = True
                for t in range(T):
                    sel = [(i, j) for (i, j) in edge_list if ev[(i, j, t)] > 0.5]
                    rts, leftover = decompose_routes(sel, N)
                    if len(rts) > V:
                        feasible = False
                    for comp in leftover:
                        for i in comp:
                            if deliv[i][t] > 1e-6:
                                feasible = False
                    for rt in rts:
                        if sum(deliv[i][t] for i in rt) > C + 1e-5:
                            feasible = False
                    routes_by_t.append(rts)
                if feasible:
                    sdict, obj = build_solution_dict(inst, deliv, routes_by_t)
                    if best is None or obj < best[0] - 1e-9:
                        best = (obj, deliv, routes_by_t, sdict)
        except Exception:
            pass
        return best
    except Exception:
        return best


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    deadline = t0 + max(1, args.time_limit)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    N = inst["num_nodes"]
    T = inst["num_periods"]

    # Constructive heuristics (several delivery patterns)
    best = None
    for mode in ("jit", "fillrem", "fillcap"):
        try:
            deliv = build_deliveries(inst, mode)
            if deliv is None:
                continue
            routes_by_t = route_all_periods(inst, deliv)
            if routes_by_t is None:
                continue
            sdict, obj = build_solution_dict(inst, deliv, routes_by_t)
            if best is None or obj < best[0] - 1e-9:
                best = (obj, deliv, routes_by_t, sdict)
        except Exception:
            continue

    if logger is not None and best is not None:
        try:
            logger.log_solution(best[0], best[3])
        except Exception:
            pass

    # Branch-and-cut with remaining time
    if deadline - time.time() > 2.0:
        best = solve_with_gurobi(inst, best, deadline, logger)

    # Last-resort fallback (schema-valid empty plan)
    if best is None:
        zero_deliv = [[0.0] * T for _ in range(N)]
        empty_routes = [[] for _ in range(T)]
        sdict, obj = build_solution_dict(inst, zero_deliv, empty_routes)
        best = (obj, zero_deliv, empty_routes, sdict)

    if logger is not None:
        try:
            logger.log_solution(best[0], best[3])
        except Exception:
            pass

    with open(args.solution_path, "w") as f:
        json.dump(best[3], f)


if __name__ == "__main__":
    main()