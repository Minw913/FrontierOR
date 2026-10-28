import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# Skip the MIP (and rely on the heuristic) when the arc-based model would be
# too large to build/solve meaningfully within a typical time budget.
ARC_GUARD = 120000


def load_instance(path):
    with open(path, "r") as fh:
        return json.load(fh)


class IRPData:
    def __init__(self, raw):
        self.N = int(raw["num_customers"])
        self.T = int(raw["num_periods"])
        self.K = int(raw["num_vehicles"])
        self.Q = float(raw["vehicle_capacity"])
        dep = raw["depot"]
        self.p = float(dep["production_per_period"])
        self.init0 = float(dep["initial_inventory"])
        self.max0 = float(dep["max_inventory"])
        self.h0 = float(dep["holding_cost"])
        self.dist = raw["distance_matrix"]
        self.C = list(range(1, self.N + 1))
        self.d = {}
        self.minv = {}
        self.maxv = {}
        self.init = {}
        self.h = {}
        for c in raw["customers"]:
            i = int(c["id"])
            self.d[i] = float(c["demand_per_period"])
            self.minv[i] = float(c.get("min_inventory", 0))
            self.maxv[i] = float(c["max_inventory"])
            self.init[i] = float(c["initial_inventory"])
            self.h[i] = float(c["holding_cost"])


# ---------------------------------------------------------------------------
# Heuristic (warm start / fallback)
# ---------------------------------------------------------------------------

def two_opt(route, dist):
    if len(route) <= 4:
        return route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                a, b, c, d = route[i - 1], route[i], route[j], route[j + 1]
                if dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d] - 1e-9:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True
    return route


def build_routes(visits, q, Q, K, dist):
    """Best-fit-decreasing packing into capacitated routes + NN/2-opt ordering."""
    if not visits:
        return []
    items = sorted(visits, key=lambda i: -q[i])
    routes = []
    loads = []
    for i in items:
        best = -1
        best_rem = None
        for r in range(len(routes)):
            rem = Q - loads[r]
            if q[i] <= rem + 1e-9 and (best_rem is None or rem < best_rem):
                best = r
                best_rem = rem
        if best < 0:
            routes.append([i])
            loads.append(q[i])
        else:
            routes[best].append(i)
            loads[best] += q[i]
    if len(routes) > K:
        return None
    out = []
    for custs in routes:
        remaining = list(custs)
        cur = 0
        ordered = []
        while remaining:
            nxt = min(remaining, key=lambda i: dist[cur][i])
            ordered.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        route = [0] + ordered + [0]
        route = two_opt(route, dist)
        out.append(route)
    return out


def heuristic(data, fill_up=False):
    """Greedy period-by-period feasible solution, or None if it fails."""
    N, T, K, Q = data.N, data.T, data.K, data.Q
    C = data.C
    dist = data.dist
    I0_cur = data.init0
    Ii_cur = {i: data.init[i] for i in C}
    heur = []
    for t in range(T):
        avail = I0_cur + data.p
        q = {}
        for i in C:
            need = data.d[i] + data.minv[i] - Ii_cur[i]
            if need > 1e-9:
                cap = min(Q, data.maxv[i] - Ii_cur[i])
                if need > cap + 1e-9:
                    return None
                q[i] = need
        total_q = sum(q.values())
        if total_q > avail + 1e-9:
            return None
        if fill_up:
            extra = avail - total_q
            for i in list(q.keys()):
                if extra <= 1e-9:
                    break
                cap = min(Q, data.maxv[i] - Ii_cur[i])
                add = min(cap - q[i], extra)
                if add > 1e-9:
                    q[i] += add
                    extra -= add
            total_q = sum(q.values())
        # Push product out if the depot would otherwise exceed its capacity.
        excess = avail - total_q - data.max0
        if excess > 1e-9:
            order = sorted(C, key=lambda i: (0 if i in q else 1, data.h[i]))
            for i in order:
                if excess <= 1e-9:
                    break
                spare = min(Q, data.maxv[i] - Ii_cur[i]) - q.get(i, 0.0)
                if spare > 1e-9:
                    add = min(spare, excess)
                    q[i] = q.get(i, 0.0) + add
                    excess -= add
            if excess > 1e-9:
                return None
        visits = [i for i in q if q[i] > 1e-9]
        routes = build_routes(visits, q, Q, K, dist)
        if routes is None:
            return None
        total_q = sum(q.values())
        I0_new = avail - total_q
        if I0_new < -1e-6 or I0_new > data.max0 + 1e-6:
            return None
        I0_cur = I0_new
        for i in C:
            Ii_cur[i] = Ii_cur[i] + q.get(i, 0.0) - data.d[i]
            if Ii_cur[i] < data.minv[i] - 1e-6 or Ii_cur[i] > data.maxv[i] + 1e-6:
                return None
        heur.append((routes, q))
    return heur


# ---------------------------------------------------------------------------
# Solution construction / objective evaluation
# ---------------------------------------------------------------------------

def extract_solution(data, xbt, qbt):
    """Build the schema-compliant solution dict from per-period arc/delivery
    information. Inventories are recomputed from deliveries for consistency."""
    T, C, dist, N = data.T, data.C, data.dist, data.N
    I0_cur = data.init0
    Ii_cur = {i: data.init[i] for i in C}
    periods = {}
    total = 0.0
    for t in range(T):
        xd = xbt[t]
        qd = qbt[t]
        starts = [j for (i, j) in xd.keys() if i == 0]
        nxt = {i: j for (i, j) in xd.keys() if i != 0}
        routes = []
        for s in starts:
            route = [0]
            cur = s
            guard = 0
            while cur != 0 and guard < N + 3:
                route.append(cur)
                cur = nxt.get(cur, 0)
                guard += 1
            route.append(0)
            routes.append(route)
        # Remove zero-delivery stops and drop empty routes (keeps routes and
        # deliveries consistent; can only reduce travel cost).
        clean = []
        for r in routes:
            kept = [c for c in r[1:-1] if qd.get(c, 0.0) > 1e-6]
            if kept:
                clean.append([0] + kept + [0])
        routes = clean
        travel = 0.0
        for r in routes:
            for a, b in zip(r[:-1], r[1:]):
                travel += dist[a][b]
        I0_cur = I0_cur + data.p - sum(qd.values())
        hold = data.h0 * I0_cur
        inv = {"depot": float(I0_cur)}
        for i in C:
            Ii_cur[i] = Ii_cur[i] + qd.get(i, 0.0) - data.d[i]
            hold += data.h[i] * Ii_cur[i]
            inv[str(i)] = float(Ii_cur[i])
        total += travel + hold
        periods[str(t)] = {
            "routes": [[int(v) for v in r] for r in routes],
            "deliveries": {str(i): float(qd[i]) for i in qd},
            "inventories": inv,
        }
    sol = {"objective_value": float(total),
           "solution_details": {"periods": periods}}
    return float(total), sol


def heuristic_to_solution(data, heur):
    xbt = []
    qbt = []
    for routes, q in heur:
        xd = {}
        for r in routes:
            for a, b in zip(r[:-1], r[1:]):
                xd[(a, b)] = 1.0
        xbt.append(xd)
        qbt.append({i: v for i, v in q.items() if v > 1e-9})
    return extract_solution(data, xbt, qbt)


def build_solution(data, x_vals, q_vals, x_keys, q_keys):
    T = data.T
    xbt = [dict() for _ in range(T)]
    for (t, i, j), v in zip(x_keys, x_vals):
        if v > 0.5:
            xbt[t][(i, j)] = 1.0
    qbt = [dict() for _ in range(T)]
    for (t, i), v in zip(q_keys, q_vals):
        if v > 1e-6:
            qbt[t][i] = v
    return extract_solution(data, xbt, qbt)


def empty_solution(data):
    return extract_solution(data,
                            [dict() for _ in range(data.T)],
                            [dict() for _ in range(data.T)])[1]


# ---------------------------------------------------------------------------
# MIP model
# ---------------------------------------------------------------------------

def solve_mip(data, heur, time_limit, tracker, logger):
    N, T, K, Q = data.N, data.T, data.K, data.Q
    C = data.C
    dist = data.dist
    V = range(N + 1)

    m = gp.Model("IRP")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, float(time_limit))

    arcs = [(t, i, j) for t in range(T) for i in V for j in V if i != j]
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    f = m.addVars(arcs, lb=0.0, ub=Q, name="f")
    q = m.addVars([(t, i) for t in range(T) for i in C], lb=0.0, name="q")
    for t in range(T):
        for i in C:
            q[t, i].ub = min(Q, data.maxv[i])
            f[t, i, 0].ub = 0.0  # vehicles return empty: all load is delivered
    I0 = m.addVars(range(T), lb=0.0, ub=data.max0, name="I0")
    Ii = {}
    for t in range(T):
        for i in C:
            Ii[t, i] = m.addVar(lb=data.minv[i], ub=data.maxv[i],
                                name="Ii_%d_%d" % (t, i))

    for t in range(T):
        m.addConstr(gp.quicksum(x[t, 0, j] for j in C) <= K, "veh_%d" % t)
        m.addConstr(gp.quicksum(x[t, 0, j] for j in C)
                    == gp.quicksum(x[t, j, 0] for j in C), "depotbal_%d" % t)
        for i in C:
            indeg = gp.quicksum(x[t, j, i] for j in V if j != i)
            outdeg = gp.quicksum(x[t, i, j] for j in V if j != i)
            m.addConstr(indeg == outdeg, "flow_%d_%d" % (t, i))
            m.addConstr(indeg <= 1, "once_%d_%d" % (t, i))
            Mi = min(Q, data.maxv[i])
            m.addConstr(q[t, i] <= Mi * indeg, "link_%d_%d" % (t, i))
            m.addConstr(
                gp.quicksum(f[t, j, i] for j in V if j != i)
                - gp.quicksum(f[t, i, j] for j in V if j != i)
                == q[t, i], "load_%d_%d" % (t, i))
        prev0 = data.init0 if t == 0 else I0[t - 1]
        m.addConstr(I0[t] == prev0 + data.p
                    - gp.quicksum(q[t, i] for i in C), "inv0_%d" % t)
        for i in C:
            previ = data.init[i] if t == 0 else Ii[t - 1, i]
            m.addConstr(Ii[t, i] == previ + q[t, i] - data.d[i],
                        "inv_%d_%d" % (t, i))
            m.addConstr(previ + q[t, i] <= data.maxv[i],
                        "cap_%d_%d" % (t, i))
        m.addConstrs((f[t, i, j] <= Q * x[t, i, j]
                      for i in V for j in V if i != j), "fcap_%d" % t)

    travel = gp.quicksum(dist[i][j] * x[t, i, j] for (t, i, j) in arcs)
    hold = data.h0 * gp.quicksum(I0[t] for t in range(T))
    for i in C:
        hold = hold + data.h[i] * gp.quicksum(Ii[t, i] for t in range(T))
    m.setObjective(travel + hold, GRB.MINIMIZE)

    # Warm start from the heuristic solution.
    if heur is not None:
        I0_cur = data.init0
        Ii_cur = {i: data.init[i] for i in C}
        for t in range(T):
            routes, qd = heur[t]
            for r in routes:
                load = sum(qd.get(i, 0.0) for i in r if i != 0)
                for a, b in zip(r[:-1], r[1:]):
                    x[t, a, b].Start = 1.0
                    f[t, a, b].Start = load
                    if b != 0:
                        load -= qd.get(b, 0.0)
            for i, v in qd.items():
                q[t, i].Start = v
            I0_cur = I0_cur + data.p - sum(qd.values())
            I0[t].Start = I0_cur
            for i in C:
                Ii_cur[i] = Ii_cur[i] + qd.get(i, 0.0) - data.d[i]
                Ii[t, i].Start = Ii_cur[i]

    x_keys = arcs
    x_vars = [x[k] for k in x_keys]
    q_keys = [(t, i) for t in range(T) for i in C]
    q_vars = [q[k] for k in q_keys]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objv < tracker["best"] - 1e-9:
                    xv = model.cbGetSolution(x_vars)
                    qv = model.cbGetSolution(q_vars)
                    obj, sol = build_solution(data, xv, qv, x_keys, q_keys)
                    if obj < tracker["best"] - 1e-6:
                        tracker["best"] = obj
                        tracker["sol"] = sol
                        if logger and obj < tracker["logged"] - 1e-6:
                            logger.log_solution(obj, sol)
                            tracker["logged"] = obj
            except Exception:
                pass

    m.optimize(callback)

    if m.SolCount > 0:
        try:
            xv = [v.X for v in x_vars]
            qv = [v.X for v in q_vars]
            obj, sol = build_solution(data, xv, qv, x_keys, q_keys)
            if obj < tracker["best"] - 1e-6:
                tracker["best"] = obj
                tracker["sol"] = sol
                if logger and obj < tracker["logged"] - 1e-6:
                    logger.log_solution(obj, sol)
                    tracker["logged"] = obj
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    raw = load_instance(args.instance_path)
    data = IRPData(raw)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start = time.time()
    tracker = {"best": float("inf"), "sol": None, "logged": float("inf")}

    # Heuristic (two variants; keep the better feasible one).
    heur = None
    sol_h = None
    best_h = float("inf")
    for fill in (False, True):
        try:
            h = heuristic(data, fill)
        except Exception:
            h = None
        if h is not None:
            try:
                obj, sol = heuristic_to_solution(data, h)
                if obj < best_h - 1e-9:
                    best_h = obj
                    heur = h
                    sol_h = sol
            except Exception:
                pass
    if sol_h is not None:
        tracker["best"] = best_h
        tracker["sol"] = sol_h
        if logger:
            try:
                logger.log_solution(best_h, sol_h)
                tracker["logged"] = best_h
            except Exception:
                pass

    # MIP for small/medium instances, within the remaining time budget.
    total_arcs = data.T * (data.N + 1) * data.N
    run_mip = total_arcs <= ARC_GUARD
    elapsed = time.time() - start
    remaining = args.time_limit - elapsed - 1.0
    if run_mip and remaining > 1.0:
        try:
            solve_mip(data, heur, remaining, tracker, logger)
        except Exception:
            pass

    if tracker["sol"] is None:
        tracker["sol"] = empty_solution(data)
        tracker["best"] = tracker["sol"]["objective_value"]

    with open(args.solution_path, "w") as fh:
        json.dump(tracker["sol"], fh, indent=2)


if __name__ == "__main__":
    main()