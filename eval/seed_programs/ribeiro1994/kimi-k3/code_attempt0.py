import argparse
import json
import math
import sys
import time
from collections import deque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    TLIMIT = float(args.time_limit)

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    # ---------------- Parse instance ----------------
    trips = inst.get("trips", [])
    trip_ids = [t["trip_id"] for t in trips]
    id_set = set(trip_ids)
    n = len(trip_ids)
    params = inst.get("parameters", {})
    depots = inst.get("depots", [])
    m = int(params.get("m_depots", len(depots)))
    depot_ids = list(range(1, m + 1))
    cap_list = inst.get("vehicle_capacities", [0] * m)
    caps = {d: int(cap_list[d - 1]) for d in depot_ids}
    relief = inst.get("relief_points", [])

    st = {t["trip_id"]: t["start_time"] for t in trips}
    et = {t["trip_id"]: t["end_time"] for t in trips}
    srp = {t["trip_id"]: t["start_relief_point_index"] for t in trips}
    erp = {t["trip_id"]: t["end_relief_point_index"] for t in trips}
    tkey = {i: (st[i], et[i]) for i in trip_ids}

    def pdist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    arc_cost = {}
    out_arcs = {i: [] for i in trip_ids}
    for a in inst.get("compatible_arcs", []):
        i = a["from_trip"]
        j = a["to_trip"]
        if i == j or i not in id_set or j not in id_set:
            continue
        if (i, j) not in arc_cost:
            arc_cost[(i, j)] = a["cost"]
            out_arcs[i].append(j)

    po_cost = {}
    for e in inst.get("depot_to_trip_costs", []):
        po_cost[(e["depot"], e["trip"])] = e["cost"]
    pi_cost = {}
    for e in inst.get("trip_to_depot_costs", []):
        pi_cost[(e["trip"], e["depot"])] = e["cost"]

    def get_po(d, i):
        c = po_cost.get((d, i))
        if c is None:
            c = int(math.floor(10.0 * pdist(depots[d - 1], relief[srp[i]]))) + 5000
            po_cost[(d, i)] = c
        return c

    def get_pi(i, d):
        c = pi_cost.get((i, d))
        if c is None:
            c = int(math.floor(10.0 * pdist(relief[erp[i]], depots[d - 1]))) + 5000
            pi_cost[(i, d)] = c
        return c

    def routes_objective(routes):
        obj = 0
        for r in routes:
            d = r["depot"]
            ts = r["trips"]
            obj += get_po(d, ts[0])
            for a, b in zip(ts[:-1], ts[1:]):
                obj += arc_cost[(a, b)]
            obj += get_pi(ts[-1], d)
        return obj

    def verify_routes(routes):
        flat = [t for r in routes for t in r["trips"]]
        return len(flat) == n and set(flat) == id_set

    # ---------------- Initial heuristic solution ----------------
    # Min path cover on the compatibility DAG via maximum bipartite matching
    # (Hopcroft-Karp with greedy seeding), then capacity-aware depot assignment.
    def initial_solution():
        matchL = {}
        matchR = {}
        order = sorted(trip_ids, key=lambda i: (st[i], et[i], i))
        for i in order:
            for j in out_arcs[i]:
                if j not in matchR:
                    matchR[j] = i
                    matchL[i] = j
                    break

        sys.setrecursionlimit(max(100000, 4 * n + 1000))
        left = trip_ids

        def bfs():
            dist = {}
            dq = deque()
            for i in left:
                if i not in matchL:
                    dist[i] = 0
                    dq.append(i)
            found = False
            while dq:
                i = dq.popleft()
                for j in out_arcs[i]:
                    k = matchR.get(j)
                    if k is None:
                        found = True
                    elif k not in dist:
                        dist[k] = dist[i] + 1
                        dq.append(k)
            return dist if found else None

        def dfs(i, dist):
            for j in out_arcs[i]:
                k = matchR.get(j)
                if k is None or (dist.get(k, -1) == dist[i] + 1 and dfs(k, dist)):
                    matchL[i] = j
                    matchR[j] = i
                    return True
            dist[i] = -1
            return False

        while time.time() - t0 < 0.5 * TLIMIT:
            dist = bfs()
            if dist is None:
                break
            for i in left:
                if i not in matchL:
                    try:
                        dfs(i, dist)
                    except RecursionError:
                        pass

        starts = [i for i in trip_ids if i not in matchR]
        chains = []
        for s in starts:
            chain = [s]
            cur = s
            while cur in matchL:
                cur = matchL[cur]
                chain.append(cur)
            chains.append(chain)

        remaining = dict(caps)
        routes = []
        for chain in chains:
            first, last = chain[0], chain[-1]
            bestd, bestc = None, None
            for d in depot_ids:
                if remaining.get(d, 0) > 0:
                    c = get_po(d, first) + get_pi(last, d)
                    if bestc is None or c < bestc:
                        bestd, bestc = d, c
            if bestd is None:
                for d in depot_ids:
                    c = get_po(d, first) + get_pi(last, d)
                    if bestc is None or c < bestc:
                        bestd, bestc = d, c
            else:
                remaining[bestd] -= 1
            routes.append({"depot": int(bestd), "trips": [int(t) for t in chain]})
        return routes

    if n == 0:
        best_sol = {"objective_value": 0.0, "routes": [], "num_vehicles": 0}
        if logger:
            logger.log_solution(0.0, best_sol)
        with open(args.solution_path, "w") as fh:
            json.dump(best_sol, fh)
        return

    init_routes = initial_solution()
    init_obj = float(routes_objective(init_routes))
    best_obj = init_obj
    best_sol = {
        "objective_value": init_obj,
        "routes": init_routes,
        "num_vehicles": len(init_routes),
    }
    if logger:
        logger.log_solution(init_obj, best_sol)

    # ---------------- Gurobi MIP ----------------
    gp = None
    try:
        import gurobipy as gp_mod
        from gurobipy import GRB
        gp = gp_mod
    except Exception:
        gp = None

    if gp is not None and (time.time() - t0) < TLIMIT - 1.0:
        try:
            arc_list = list(arc_cost.keys())
            po_list = [(d, i) for d in depot_ids for i in trip_ids]
            pi_list = [(i, d) for i in trip_ids for d in depot_ids]
            for e in po_list:
                get_po(*e)
            for e in pi_list:
                get_pi(*e)

            model = gp.Model("MDVSP")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1.0, TLIMIT - (time.time() - t0) - 0.5)

            x = model.addVars(arc_list, vtype=GRB.BINARY, name="x")
            u = model.addVars(po_list, vtype=GRB.BINARY, name="u")
            v = model.addVars(pi_list, vtype=GRB.BINARY, name="v")

            model.setObjective(
                gp.quicksum(arc_cost[a] * x[a] for a in arc_list)
                + gp.quicksum(po_cost[e] * u[e] for e in po_list)
                + gp.quicksum(pi_cost[e] * v[e] for e in pi_list),
                GRB.MINIMIZE,
            )

            # Each trip has exactly one predecessor and one successor.
            model.addConstrs(
                (x.sum("*", j) + u.sum("*", j) == 1 for j in trip_ids), name="pred"
            )
            model.addConstrs(
                (x.sum(i, "*") + v.sum(i, "*") == 1 for i in trip_ids), name="succ"
            )
            # Depot capacities and depot flow balance.
            model.addConstrs(
                (u.sum(d, "*") <= caps[d] for d in depot_ids), name="cap"
            )
            model.addConstrs(
                (u.sum(d, "*") == v.sum("*", d) for d in depot_ids), name="bal"
            )

            # Same-depot round trip enforcement via depot-label propagation.
            if m > 1:
                M = float(m)
                p = model.addVars(trip_ids, lb=1.0, ub=float(m), name="p")
                model.addConstrs(
                    (p[j] - p[i] <= M * (1 - x[i, j]) for (i, j) in arc_list),
                    name="lblx1",
                )
                model.addConstrs(
                    (p[i] - p[j] <= M * (1 - x[i, j]) for (i, j) in arc_list),
                    name="lblx2",
                )
                model.addConstrs(
                    (p[i] - d <= M * (1 - u[d, i]) for (d, i) in po_list),
                    name="lblu1",
                )
                model.addConstrs(
                    (d - p[i] <= M * (1 - u[d, i]) for (d, i) in po_list),
                    name="lblu2",
                )
                model.addConstrs(
                    (p[i] - d <= M * (1 - v[i, d]) for (i, d) in pi_list),
                    name="lblv1",
                )
                model.addConstrs(
                    (d - p[i] <= M * (1 - v[i, d]) for (i, d) in pi_list),
                    name="lblv2",
                )

            # Guard against degenerate depot-less cycles among trips with
            # identical (start, end) time keys (only possible cycles in a DAG).
            tie_arcs = [(i, j) for (i, j) in arc_list if tkey[i] == tkey[j]]
            has_tie_cycle_risk = False
            if tie_arcs:
                # check if tie-arc subgraph has any cycle (usually it is empty)
                indeg = {}
                tadj = {}
                for (i, j) in tie_arcs:
                    tadj.setdefault(i, []).append(j)
                    indeg[j] = indeg.get(j, 0) + 1
                    indeg.setdefault(i, indeg.get(i, 0))
                dq = deque([i for i in indeg if indeg[i] == 0])
                seen = 0
                while dq:
                    a = dq.popleft()
                    seen += 1
                    for b in tadj.get(a, ()):
                        indeg[b] -= 1
                        if indeg[b] == 0:
                            dq.append(b)
                has_tie_cycle_risk = seen < len(indeg)
            if has_tie_cycle_risk:
                S = float(n)
                o = model.addVars(trip_ids, lb=0.0, ub=S, name="o")
                model.addConstrs(
                    (o[j] >= o[i] + 1.0 - S * (1 - x[i, j]) for (i, j) in tie_arcs),
                    name="mtz",
                )

            model.update()

            # MIP start from the heuristic solution.
            arc_index = {a: k for k, a in enumerate(arc_list)}
            po_index = {e: k for k, e in enumerate(po_list)}
            pi_index = {e: k for k, e in enumerate(pi_list)}
            trip_pos = {i: k for k, i in enumerate(trip_ids)}
            x_start = [0.0] * len(arc_list)
            u_start = [0.0] * len(po_list)
            v_start = [0.0] * len(pi_list)
            p_start = [1.0] * n
            for r in init_routes:
                d = r["depot"]
                ts = r["trips"]
                u_start[po_index[(d, ts[0])]] = 1.0
                v_start[pi_index[(ts[-1], d)]] = 1.0
                for i in ts:
                    p_start[trip_pos[i]] = float(d)
                for a, b in zip(ts[:-1], ts[1:]):
                    if (a, b) in arc_index:
                        x_start[arc_index[(a, b)]] = 1.0
            x_vars = [x[a] for a in arc_list]
            u_vars = [u[e] for e in po_list]
            v_vars = [v[e] for e in pi_list]
            model.setAttr("Start", x_vars, x_start)
            model.setAttr("Start", u_vars, u_start)
            model.setAttr("Start", v_vars, v_start)
            if m > 1:
                model.setAttr("Start", [p[i] for i in trip_ids], p_start)

            def build_routes(xv, uv):
                succ = {}
                for k, (i, j) in enumerate(arc_list):
                    if xv[k] > 0.5:
                        succ[i] = j
                routes = []
                for k, (d, i) in enumerate(po_list):
                    if uv[k] > 0.5:
                        ts = [i]
                        cur = i
                        while cur in succ and len(ts) <= n:
                            cur = succ[cur]
                            ts.append(cur)
                        routes.append(
                            {"depot": int(d), "trips": [int(t) for t in ts]}
                        )
                return routes

            last_logged = [best_obj]

            def callback(mdl, where):
                try:
                    if time.time() - t0 > TLIMIT - 0.2:
                        mdl.cbTerminate()
                        return
                except Exception:
                    return
                if where == GRB.Callback.MIPSOL:
                    try:
                        obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    except Exception:
                        return
                    if logger and obj < last_logged[0] - 1e-6:
                        try:
                            xv = mdl.cbGetSolution(x_vars)
                            uv = mdl.cbGetSolution(u_vars)
                            routes = build_routes(xv, uv)
                            sol = {
                                "objective_value": float(obj),
                                "routes": routes,
                                "num_vehicles": len(routes),
                            }
                            logger.log_solution(float(obj), sol)
                        except Exception:
                            logger.log(float(obj))
                        last_logged[0] = obj

            model.optimize(callback)

            if model.SolCount > 0:
                xv = model.getAttr("X", x_vars)
                uv = model.getAttr("X", u_vars)
                routes = build_routes(xv, uv)
                if verify_routes(routes):
                    obj = float(routes_objective(routes))
                    if obj <= best_obj + 1e-6:
                        best_obj = obj
                        best_sol = {
                            "objective_value": obj,
                            "routes": routes,
                            "num_vehicles": len(routes),
                        }
        except Exception:
            # Any solver failure: keep the heuristic solution.
            pass

    if logger and best_sol["objective_value"] < init_obj - 1e-6:
        try:
            logger.log_solution(best_sol["objective_value"], best_sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as fh:
        json.dump(best_sol, fh)


if __name__ == "__main__":
    main()