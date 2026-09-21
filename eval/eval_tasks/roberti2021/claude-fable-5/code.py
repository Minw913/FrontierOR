import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_arrivals(troute, droute, tt, dt):
    """Compute earliest (tight) arrival times given truck route and drone route.

    Routes are lists of internal node ids (0 = depot origin, 1..n customers,
    n+1 = depot destination). Returns dict node -> arrival time.
    """
    arr = {}
    for nd in set(troute) | set(droute):
        arr[nd] = 0.0
    ta = list(zip(troute[:-1], troute[1:]))
    da = list(zip(droute[:-1], droute[1:]))
    for _ in range(len(ta) + len(da) + 3):
        changed = False
        for i, j in ta:
            nv = arr[i] + tt(i, j)
            if nv > arr[j] + 1e-9:
                arr[j] = nv
                changed = True
        for i, j in da:
            nv = arr[i] + dt(i, j)
            if nv > arr[j] + 1e-9:
                arr[j] = nv
                changed = True
        if not changed:
            break
    return arr


def build_solution(troute, droute, n, tt, dt):
    """Build the output solution dictionary; returns (objective, dict)."""
    e = n + 1
    arr = compute_arrivals(troute, droute, tt, dt)
    obj = arr.get(e, 0.0)
    tset = set(troute)
    dset = set(droute)
    types = {}
    for c in range(1, n + 1):
        if c in tset and c in dset:
            types[str(c)] = "combined"
        elif c in tset:
            types[str(c)] = "truck"
        else:
            types[str(c)] = "drone"

    def project(route):
        out = []
        for nd in route:
            out.append(0 if nd == e else nd)
        return out

    arrival_times = {}
    for i in range(e + 1):
        arrival_times[str(i)] = float(arr.get(i, 0.0))

    sol = {
        "objective_value": float(obj),
        "truck_route": project(troute),
        "drone_route": project(droute),
        "customer_types": types,
        "arrival_times": arrival_times,
    }
    return float(obj), sol


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["num_customers"])
    T = data["truck_travel_times"]
    D = data["drone_travel_times"]
    e = n + 1
    C = list(range(1, n + 1))

    def tt(i, j):
        return T[0 if i == e else i][0 if j == e else j]

    def dt(i, j):
        return D[0 if i == e else i][0 if j == e else j]

    # ---------------- Heuristic construction ----------------
    # Nearest neighbor TSP for the truck.
    order = []
    cur = 0
    unv = set(C)
    while unv:
        nxt = min(unv, key=lambda c: T[cur][c])
        order.append(nxt)
        unv.remove(nxt)
        cur = nxt

    # 2-opt improvement (time-capped).
    deadline2 = t0 + min(max(args.time_limit * 0.15, 1.0), 8.0)
    improved = True
    while improved and time.time() < deadline2:
        improved = False
        L = len(order)
        for i in range(L - 1):
            for j in range(i + 1, L):
                p = order[i - 1] if i > 0 else 0
                q = order[j + 1] if j < L - 1 else 0
                delta = (T[p][order[j]] + T[order[i]][q]
                         - T[p][order[i]] - T[order[j]][q])
                if delta < -1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
            if time.time() > deadline2:
                break

    # Candidate heuristic solutions.
    cands = []
    tro = [0] + order + [e]
    for k in range(len(order) - 1):
        # Two consecutive truck customers become combined; drone piggybacks.
        cands.append((tro, [0, order[k], order[k + 1], e]))
    if len(order) >= 3:
        for k in range(len(order) - 2):
            # Drone sortie: launch at order[k], serve order[k+1] alone, land order[k+2].
            tr = [0] + order[:k + 1] + order[k + 2:] + [e]
            cands.append((tr, [0, order[k], order[k + 1], order[k + 2], e]))
        # Drone serves the first customer alone, meets truck at second.
        tr = [0] + order[1:] + [e]
        cands.append((tr, [0, order[0], order[1], e]))
        # Drone launches from second-to-last customer, serves last alone.
        tr = [0] + order[:-1] + [e]
        cands.append((tr, [0, order[-2], order[-1], e]))

    best_obj = float("inf")
    best_sol = None
    best_routes = None
    for tr, dr in cands:
        obj, sol = build_solution(tr, dr, n, tt, dt)
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_sol = sol
            best_routes = (tr, dr)

    if logger and best_sol is not None:
        logger.log_solution(best_obj, best_sol)

    # ---------------- MIP model ----------------
    remaining = args.time_limit - (time.time() - t0) - 2.0
    state = {"best_obj": best_obj, "best_sol": best_sol}

    if remaining > 3.0:
        try:
            UB = best_obj + 1.0
            m = gp.Model("tsp_d")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, remaining)

            arcs = ([(0, c) for c in C]
                    + [(i, j) for i in C for j in C if i != j]
                    + [(c, e) for c in C])

            x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
            y = m.addVars(arcs, vtype=GRB.BINARY, name="y")
            u = m.addVars(C, vtype=GRB.BINARY, name="u")  # truck visits
            v = m.addVars(C, vtype=GRB.BINARY, name="v")  # drone visits
            w = m.addVars(C, vtype=GRB.BINARY, name="w")  # combined
            a = m.addVars(range(e + 1), lb=0.0, ub=UB, name="a")
            a[0].UB = 0.0

            m.setObjective(a[e], GRB.MINIMIZE)

            # Depot degree constraints.
            m.addConstr(gp.quicksum(x[0, c] for c in C) == 1)
            m.addConstr(gp.quicksum(x[c, e] for c in C) == 1)
            m.addConstr(gp.quicksum(y[0, c] for c in C) == 1)
            m.addConstr(gp.quicksum(y[c, e] for c in C) == 1)

            # Customer degree constraints.
            out_t = {c: [] for c in C}
            in_t = {c: [] for c in C}
            for (i, j) in arcs:
                if i in out_t:
                    out_t[i].append((i, j))
                if j in in_t:
                    in_t[j].append((i, j))
            for c in C:
                m.addConstr(gp.quicksum(x[ar] for ar in out_t[c]) == u[c])
                m.addConstr(gp.quicksum(x[ar] for ar in in_t[c]) == u[c])
                m.addConstr(gp.quicksum(y[ar] for ar in out_t[c]) == v[c])
                m.addConstr(gp.quicksum(y[ar] for ar in in_t[c]) == v[c])
                m.addConstr(u[c] + v[c] >= 1)
                m.addConstr(w[c] <= u[c])
                m.addConstr(w[c] <= v[c])
                m.addConstr(w[c] >= u[c] + v[c] - 1)
                # No single-customer drone round trip.
                m.addConstr(y[0, c] + y[c, e] <= 1)

            # Pairwise drone arc / combined constraints.
            for idx, i in enumerate(C):
                for j in C[idx + 1:]:
                    m.addConstr(y[i, j] + y[j, i] <= w[i] + w[j])

            # Timing (also eliminates subtours).
            for (i, j) in arcs:
                tij = tt(i, j)
                dij = dt(i, j)
                m.addConstr(a[j] >= a[i] + tij - (UB + tij) * (1 - x[i, j]))
                m.addConstr(a[j] >= a[i] + dij - (UB + dij) * (1 - y[i, j]))

            # Warm start from heuristic.
            if best_routes is not None:
                tr, dr = best_routes
                for (i, j) in arcs:
                    x[i, j].Start = 0.0
                    y[i, j].Start = 0.0
                for i, j in zip(tr[:-1], tr[1:]):
                    x[i, j].Start = 1.0
                for i, j in zip(dr[:-1], dr[1:]):
                    y[i, j].Start = 1.0
                tset = set(tr)
                dset = set(dr)
                arr0 = compute_arrivals(tr, dr, tt, dt)
                for c in C:
                    uc = 1.0 if c in tset else 0.0
                    vc = 1.0 if c in dset else 0.0
                    u[c].Start = uc
                    v[c].Start = vc
                    w[c].Start = 1.0 if (uc > 0.5 and vc > 0.5) else 0.0
                for nd in range(e + 1):
                    a[nd].Start = float(arr0.get(nd, 0.0))

            def extract_routes(xv, yv):
                def follow(vals):
                    succ = {}
                    for (i, j) in arcs:
                        if vals[i, j] > 0.5:
                            succ[i] = j
                    route = [0]
                    guard = 0
                    while route[-1] != e and guard <= e + 2:
                        route.append(succ[route[-1]])
                        guard += 1
                    return route
                return follow(xv), follow(yv)

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        xv = model.cbGetSolution(x)
                        yv = model.cbGetSolution(y)
                        tr, dr = extract_routes(xv, yv)
                        obj, sol = build_solution(tr, dr, n, tt, dt)
                        if obj < state["best_obj"] - 1e-9:
                            state["best_obj"] = obj
                            state["best_sol"] = sol
                            if logger:
                                logger.log_solution(obj, sol)
                    except Exception:
                        pass

            m.optimize(cb)

            if m.SolCount > 0:
                try:
                    xv = {ar: x[ar].X for ar in arcs}
                    yv = {ar: y[ar].X for ar in arcs}
                    tr, dr = extract_routes(xv, yv)
                    obj, sol = build_solution(tr, dr, n, tt, dt)
                    if obj < state["best_obj"] - 1e-9:
                        state["best_obj"] = obj
                        state["best_sol"] = sol
                        if logger:
                            logger.log_solution(obj, sol)
                except Exception:
                    pass
        except Exception:
            pass

    final_sol = state["best_sol"]
    if final_sol is None:
        # Extremely defensive fallback (should never happen).
        tr = [0] + order + [e]
        dr = [0, order[0], order[min(1, len(order) - 1)], e]
        _, final_sol = build_solution(tr, dr, n, tt, dt)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)


if __name__ == "__main__":
    main()