import argparse
import heapq
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:  # pragma: no cover
    SolutionLogger = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Multicommodity service network design with integer fleet "
                    "assignment and vehicle circulation balance."
    )
    p.add_argument("--instance_path", required=True, help="Path to instance JSON file.")
    p.add_argument("--solution_path", required=True, help="Path to output solution JSON file.")
    p.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    p.add_argument("--log_path", default=None, help="Optional JSONL path for incumbent logging.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    # ----------------------------- data preparation -----------------------------
    nodes = list(inst["nodes"])
    n = len(nodes)
    nidx = {v: i for i, v in enumerate(nodes)}

    arcs = inst["arcs"]
    A = len(arcs)
    a_from = [nidx[a["from"]] for a in arcs]
    a_to = [nidx[a["to"]] for a in arcs]
    a_cost = [float(a["flow_cost"]) for a in arcs]
    a_from_id = [a["from"] for a in arcs]
    a_to_id = [a["to"] for a in arcs]

    out_arcs = [[] for _ in range(n)]
    in_arcs = [[] for _ in range(n)]
    for a in range(A):
        out_arcs[a_from[a]].append(a)
        in_arcs[a_to[a]].append(a)

    commodities = inst["commodities"]
    K = len(commodities)
    c_id = [c["id"] for c in commodities]
    c_org = [nidx[c["origin"]] for c in commodities]
    c_dst = [nidx[c["destination"]] for c in commodities]
    c_dem = [float(c["demand"]) for c in commodities]

    fleets = inst["fleet_types"]
    T = len(fleets)
    f_id = [ft["id"] for ft in fleets]
    f_cap = [float(ft["capacity"]) for ft in fleets]
    f_cost = [float(ft["fixed_cost"]) for ft in fleets]

    total_dem = sum(c_dem)
    best_ratio = min(f_cost[t] / f_cap[t] for t in range(T)) if T > 0 else 0.0

    # ----------------------------- helpers -----------------------------
    def dijkstra(s, t, w):
        if s == t:
            return []
        INF = float("inf")
        dist = [INF] * n
        prev = [-1] * n
        done = [False] * n
        dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if done[u]:
                continue
            done[u] = True
            if u == t:
                break
            for a in out_arcs[u]:
                v = a_to[a]
                nd = d + w[a]
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = a
                    heapq.heappush(pq, (nd, v))
        if dist[t] == INF:
            return None
        path = []
        cur = t
        while cur != s:
            a = prev[cur]
            if a < 0:
                return None
            path.append(a)
            cur = a_from[a]
        path.reverse()
        return path

    def route_all(w):
        flows = [0.0] * A
        paths = []
        for ki in range(K):
            p = dijkstra(c_org[ki], c_dst[ki], w)
            if p is None:
                return None
            paths.append(p)
            d = c_dem[ki]
            for a in p:
                flows[a] += d
        return paths, flows

    def cover_counts(flows, mode):
        """Return dict type -> list of vehicle counts per arc (ignores circulation)."""
        if mode[0] == "single":
            t = mode[1]
            cap = f_cap[t]
            lst = [0] * A
            for a in range(A):
                f = flows[a]
                if f > 0:
                    lst[a] = int(math.ceil(f / cap - 1e-12))
            return {t: lst}
        # mixed: bulk with best cost-per-capacity type, tail with cheapest cover
        b = min(range(T), key=lambda t: f_cost[t] / f_cap[t])
        counts = {t: [0] * A for t in range(T)}
        capB, costB = f_cap[b], f_cost[b]
        for a in range(A):
            f = flows[a]
            if f <= 0:
                continue
            best_cost = None
            best_assign = None
            for t in range(T):
                nn = int(math.ceil(f / f_cap[t] - 1e-12))
                cost = nn * f_cost[t]
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_assign = ((t, nn),)
            nB = int(f // capB)
            rem = f - nB * capB
            if rem > 1e-9:
                for t in range(T):
                    nn = int(math.ceil(rem / f_cap[t] - 1e-12))
                    cost = nB * costB + nn * f_cost[t]
                    if cost < best_cost - 1e-9:
                        best_cost = cost
                        best_assign = ((b, nB), (t, nn))
            for t, nn in best_assign:
                if nn:
                    counts[t][a] = nn
        return counts

    def repair_circulation(counts):
        """Add deadhead vehicles so each type forms a circulation.

        counts: dict type -> list per arc. Returns (y_dict, vehicle_cost) or None.
        Solved as a min-cost flow LP (dual simplex -> integral basic optimum).
        """
        y = {}
        cost = 0.0
        for t, lst in counts.items():
            if not any(lst):
                continue
            bvec = [0] * n
            for a in range(A):
                ca = lst[a]
                if ca:
                    bvec[a_from[a]] += ca
                    bvec[a_to[a]] -= ca
            m = gp.Model("repair")
            m.setParam(GRB.Param.OutputFlag, 0)
            m.setParam(GRB.Param.Threads, 1)
            m.setParam(GRB.Param.Seed, 0)
            m.setParam(GRB.Param.NumericFocus, 0)
            m.setParam(GRB.Param.Method, 1)  # dual simplex -> basic (integral) optimum
            z = m.addVars(A, lb=0.0)
            for v in range(n):
                m.addConstr(
                    gp.quicksum(z[a] for a in out_arcs[v])
                    - gp.quicksum(z[a] for a in in_arcs[v]) == -bvec[v]
                )
            m.setObjective(gp.quicksum(z[a] for a in range(A)), GRB.MINIMIZE)
            m.optimize()
            if m.Status != GRB.OPTIMAL:
                m.dispose()
                return None
            zv = [int(round(z[a].X)) for a in range(A)]
            y[t] = [lst[a] + zv[a] for a in range(A)]
            cost += f_cost[t] * sum(y[t])
            m.dispose()
        return y, cost

    def heuristic():
        weight_sets = [a_cost, [a_cost[a] + best_ratio for a in range(A)]]
        modes = [("single", t) for t in range(T)] + [("mixed", -1)]
        best = None  # (x_list, y_list, obj)
        for w in weight_sets:
            if time.time() - t0 > 0.5 * time_limit:
                break
            routed = route_all(w)
            if routed is None:
                continue
            paths, flows = routed
            flow_cost = sum(a_cost[a] * flows[a] for a in range(A))
            for mode in modes:
                counts = cover_counts(flows, mode)
                res = repair_circulation(counts)
                if res is None:
                    continue
                y, vcost = res
                obj = flow_cost + vcost
                if best is None or obj < best[2]:
                    x_list = [0.0] * (K * A)
                    for ki, p in enumerate(paths):
                        base = ki * A
                        for a in p:
                            x_list[base + a] += c_dem[ki]
                    y_list = [0.0] * (T * A)
                    for t, lst in y.items():
                        base = t * A
                        for a in range(A):
                            if lst[a]:
                                y_list[base + a] = float(lst[a])
                    best = (x_list, y_list, obj)
        return best

    # ----------------------------- output assembly -----------------------------
    x_keys = [
        "x_{}_{}_{}".format(c_id[ki], a_from_id[a], a_to_id[a])
        for ki in range(K) for a in range(A)
    ]
    x_key_cost = [a_cost[a] for ki in range(K) for a in range(A)]
    y_keys = [
        "y_{}_{}_{}".format(f_id[t], a_from_id[a], a_to_id[a])
        for t in range(T) for a in range(A)
    ]
    y_key_cost = [f_cost[t] for t in range(T) for a in range(A)]

    def build_output(xv, yv):
        flows = {}
        obj = 0.0
        for i in range(len(x_keys)):
            v = xv[i]
            if v is None or abs(v) < 1e-9:
                v = 0.0
            flows[x_keys[i]] = float(v)
            obj += x_key_cost[i] * v
        veh = {}
        for j in range(len(y_keys)):
            vv = yv[j]
            iv = int(round(vv)) if vv is not None else 0
            if iv < 0:
                iv = 0
            veh[y_keys[j]] = iv
            obj += y_key_cost[j] * iv
        return {"objective_value": float(obj), "flows": flows, "vehicles": veh}, obj

    # ----------------------------- heuristic incumbent -----------------------------
    best_sol = None
    best_obj = float("inf")
    heur = None
    try:
        heur = heuristic()
    except Exception:
        heur = None
    if heur is not None:
        best_sol, best_obj = build_output(heur[0], heur[1])
        if logger is not None:
            try:
                logger.log_solution(best_obj, best_sol)
            except Exception:
                pass

    # ----------------------------- MIP -----------------------------
    try:
        remaining = time_limit - (time.time() - t0)
        if remaining > 0.5 and K > 0 and A > 0 and T > 0:
            model = gp.Model("service_network_design")
            model.setParam(GRB.Param.Seed, 0)
            model.setParam(GRB.Param.MIPGap, 1e-4)
            model.setParam(GRB.Param.NumericFocus, 0)
            model.setParam(GRB.Param.Threads, 1)
            model.setParam(GRB.Param.OutputFlag, 0)

            # decision variables
            x_ub = {}
            for ki in range(K):
                dem = 0.0 if c_org[ki] == c_dst[ki] else c_dem[ki]
                for a in range(A):
                    x_ub[(ki, a)] = dem
            x = model.addVars(K, A, lb=0.0, ub=x_ub, vtype=GRB.CONTINUOUS, name="x")
            y_ub = {
                (t, a): int(math.ceil(total_dem / f_cap[t]))
                for t in range(T) for a in range(A)
            }
            y = model.addVars(T, A, lb=0, ub=y_ub, vtype=GRB.INTEGER, name="y")

            # commodity flow conservation
            for ki in range(K):
                o, d, dem = c_org[ki], c_dst[ki], c_dem[ki]
                if o == d:
                    continue
                for v in range(n):
                    if v == o:
                        rhs = dem
                    elif v == d:
                        rhs = -dem
                    else:
                        rhs = 0.0
                    model.addConstr(
                        gp.quicksum(x[ki, a] for a in out_arcs[v])
                        - gp.quicksum(x[ki, a] for a in in_arcs[v]) == rhs,
                        name="flow_{}_{}".format(ki, v),
                    )

            # arc capacity from assigned vehicles
            for a in range(A):
                model.addConstr(
                    gp.quicksum(x[ki, a] for ki in range(K))
                    <= gp.quicksum(f_cap[t] * y[t, a] for t in range(T)),
                    name="cap_{}".format(a),
                )

            # vehicle circulation balance per fleet type
            for t in range(T):
                for v in range(n):
                    model.addConstr(
                        gp.quicksum(y[t, a] for a in out_arcs[v])
                        - gp.quicksum(y[t, a] for a in in_arcs[v]) == 0,
                        name="circ_{}_{}".format(t, v),
                    )

            # objective
            obj_expr = gp.LinExpr()
            for a in range(A):
                ca = a_cost[a]
                for ki in range(K):
                    obj_expr.addTerm(ca, x[ki, a])
            for t in range(T):
                fc = f_cost[t]
                for a in range(A):
                    obj_expr.addTerm(fc, y[t, a])
            model.setObjective(obj_expr, GRB.MINIMIZE)

            # MIP start from heuristic
            if heur is not None:
                hx, hy, _ = heur
                idx = 0
                for ki in range(K):
                    base = ki * A
                    for a in range(A):
                        v = hx[base + a]
                        if v:
                            x[ki, a].Start = v
                        idx += 1
                for t in range(T):
                    base = t * A
                    for a in range(A):
                        v = hy[base + a]
                        if v:
                            y[t, a].Start = v

            x_var_list = [x[ki, a] for ki in range(K) for a in range(A)]
            y_var_list = [y[t, a] for t in range(T) for a in range(A)]

            def callback(m, where):
                if where == GRB.Callback.MIPSOL:
                    if time.time() - t0 > time_limit:
                        m.terminate()
                        return
                    if logger is None:
                        return
                    try:
                        xv = m.cbGetSolution(x_var_list)
                        yv = m.cbGetSolution(y_var_list)
                        sol, obj = build_output(xv, yv)
                        logger.log_solution(obj, sol)
                    except Exception:
                        pass

            model.setParam(GRB.Param.TimeLimit, max(0.1, time_limit - (time.time() - t0)))
            model.optimize(callback)

            if model.SolCount > 0:
                xv = [v.X for v in x_var_list]
                yv = [v.X for v in y_var_list]
                sol, obj = build_output(xv, yv)
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_sol = sol
            model.dispose()
    except Exception:
        pass

    # ----------------------------- fallback & output -----------------------------
    if best_sol is None:
        best_sol, best_obj = build_output([0.0] * (K * A), [0.0] * (T * A))

    if logger is not None:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as fh:
        json.dump(best_sol, fh)


if __name__ == "__main__":
    main()