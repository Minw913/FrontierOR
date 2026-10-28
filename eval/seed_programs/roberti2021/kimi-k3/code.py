import argparse
import json
import random
import time

INF = float("inf")

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


# ---------------------------------------------------------------------------
# The travel-time matrices cover only the physical locations 0..n (depot +
# customers). The model also uses a distinct depot-destination node n+1 whose
# travel times equal those of the depot (node 0).  We therefore build extended
# (n+2)x(n+2) matrices once and use them everywhere internally.
# ---------------------------------------------------------------------------
def extend_matrix(mat, n):
    N = n + 2
    dest = n + 1
    ext = [[0.0] * N for _ in range(N)]
    for i in range(n + 1):
        row = mat[i]
        erow = ext[i]
        for j in range(n + 1):
            erow[j] = float(row[j])
        erow[dest] = float(row[0])      # travel customer/depot -> depot destination
    # row 'dest' is never used (no arcs leave the destination); fill with depot row
    for j in range(n + 1):
        ext[dest][j] = float(mat[0][j])
    ext[dest][dest] = 0.0
    ext[0][dest] = INF                  # no direct origin->destination arc
    return ext


# ---------------------------------------------------------------------------
# Feasibility check + exact arrival-time computation (longest path over the
# arcs used by both vehicles; shared/combined nodes synchronize the vehicles).
# Returns (objective, arrival_list) or (INF, None) if infeasible.
# ---------------------------------------------------------------------------
def evaluate(n, tt, td, truck_order, drone_order):
    dest = n + 1
    if truck_order is None or drone_order is None:
        return INF, None
    if len(truck_order) < 1 or len(drone_order) < 2:
        return INF, None
    if len(set(truck_order)) != len(truck_order):
        return INF, None
    if len(set(drone_order)) != len(drone_order):
        return INF, None
    N = n + 2
    inT = [False] * N
    inD = [False] * N
    for v in truck_order:
        if v < 1 or v > n:
            return INF, None
        inT[v] = True
    for v in drone_order:
        if v < 1 or v > n:
            return INF, None
        inD[v] = True
    for v in range(1, n + 1):
        if not inT[v] and not inD[v]:
            return INF, None

    arcs = []
    seq = [0] + list(truck_order) + [dest]
    for k in range(len(seq) - 1):
        a, b = seq[k], seq[k + 1]
        arcs.append((a, b, tt[a][b]))
    dseq = [0] + list(drone_order) + [dest]
    for k in range(len(dseq) - 1):
        a, b = dseq[k], dseq[k + 1]
        if a != 0 and b != dest:
            # drone arc between two customers needs >= one combined endpoint
            if not inT[a] and not inT[b]:
                return INF, None
        arcs.append((a, b, td[a][b]))

    arr = [0.0] * N
    for _ in range(N):
        updated = False
        for (a, b, t) in arcs:
            v_ = arr[a] + t
            if arr[b] < v_:
                arr[b] = v_
                updated = True
        if not updated:
            break
    # if any arc can still be relaxed there is a positive cycle -> infeasible
    for (a, b, t) in arcs:
        if arr[b] < arr[a] + t - 1e-9:
            return INF, None
    return arr[dest], arr


def build_solution(n, truck_order, drone_order, arr, obj):
    dest = n + 1
    Tset = set(truck_order)
    Dset = set(drone_order)
    ctypes = {}
    for v in range(1, n + 1):
        if v in Tset and v in Dset:
            ctypes[str(v)] = "combined"
        elif v in Tset:
            ctypes[str(v)] = "truck"
        else:
            ctypes[str(v)] = "drone"
    return {
        "objective_value": float(obj),
        "truck_route": [0] + [int(v) for v in truck_order] + [dest],
        "drone_route": [0] + [int(v) for v in drone_order] + [dest],
        "customer_types": ctypes,
        "arrival_times": {str(i): float(arr[i]) for i in range(n + 2)},
    }


def record(n, T, D, obj, arr, logger, log_state):
    if obj < log_state["best_obj"] - 1e-9:
        log_state["best_obj"] = obj
        log_state["best_T"] = list(T)
        log_state["best_D"] = list(D)
        log_state["best_arr"] = list(arr)
    if logger is not None and obj < log_state["last"] - 1e-6:
        sol = build_solution(n, T, D, arr, obj)
        logger.log_solution(float(obj), sol)
        log_state["last"] = obj


# ---------------------------------------------------------------------------
# Construction heuristic
# ---------------------------------------------------------------------------
def tour_cost(order, tt, dest):
    if not order:
        return 0.0
    c = tt[0][order[0]]
    for a, b in zip(order, order[1:]):
        c += tt[a][b]
    c += tt[order[-1]][dest]
    return c


def nn_tour(n, tt):
    unvisited = set(range(1, n + 1))
    order = []
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: (tt[cur][j], j))
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def two_opt(order, tt, dest, deadline):
    order = list(order)
    m = len(order)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(m - 1):
            a = order[i - 1] if i > 0 else 0
            b = order[i]
            found = False
            for j in range(i + 1, m):
                c = order[j]
                d = order[j + 1] if j + 1 < m else dest
                if tt[a][c] + tt[b][d] < tt[a][b] + tt[c][d] - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
                    found = True
                    break
            if found:
                break
    return order


def build_initial(n, tt, td, deadline):
    dest = n + 1
    C = list(range(1, n + 1))
    order = two_opt(nn_tour(n, tt), tt, dest, deadline)
    base = tour_cost(order, tt, dest)
    pos = {v: k for k, v in enumerate(order)}

    best = None

    # Option A: truck serves everyone; drone visits two combined customers
    # that are consecutive (and order-consistent) on the truck tour.
    candA = []
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        trip = td[0][i] + td[i][j] + td[j][dest]
        candA.append((max(base, trip), i, j))
    candA.sort(key=lambda c: c[0])
    for c in candA[:10]:
        _, i, j = c
        T = list(order)
        D = [i, j]
        obj, arr = evaluate(n, tt, td, T, D)
        if obj < INF and (best is None or obj < best[0]):
            best = (obj, T, D, arr)

    # Option B: one customer becomes drone-only (removed from the truck
    # tour) and one remaining customer is the combined rendezvous.
    candB = []
    for jj in C:
        p = pos[jj]
        prev = order[p - 1] if p > 0 else 0
        nxt = order[p + 1] if p + 1 < len(order) else dest
        tcost = base - tt[prev][jj] - tt[jj][nxt] + tt[prev][nxt]
        local = []
        for i in C:
            if i == jj:
                continue
            t1 = td[0][i] + td[i][jj] + td[jj][dest]
            local.append((max(tcost, t1), jj, i, 0))
            t2 = td[0][jj] + td[jj][i] + td[i][dest]
            local.append((max(tcost, t2), jj, i, 1))
        local.sort(key=lambda c: c[0])
        candB.extend(local[:3])
    candB.sort(key=lambda c: c[0])
    for c in candB[:20]:
        _, jj, i, flip = c
        T = [u for u in order if u != jj]
        D = [i, jj] if flip == 0 else [jj, i]
        obj, arr = evaluate(n, tt, td, T, D)
        if obj < INF and (best is None or obj < best[0]):
            best = (obj, T, D, arr)

    if best is None:
        # always-feasible fallback: truck serves all, drone meets it at the
        # first two customers of the truck tour (both combined).
        T = list(order)
        D = [order[0], order[1 if len(order) > 1 else 0]]
        obj, arr = evaluate(n, tt, td, T, D)
        best = (obj, T, D, arr)
    return best[1], best[2], best[3], best[0]


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
def find_move(n, tt, td, T, D, cur_obj, deadline):
    Tset = set(T)
    Dset = set(D)
    for v in range(1, n + 1):
        if time.time() > deadline:
            return None
        inT = v in Tset
        inD = v in Dset
        T_wo = [u for u in T if u != v] if inT else None
        D_wo = [u for u in D if u != v] if inD else None
        cands = []
        if inT and inD:
            if len(D) > 2:
                cands.append((T, D_wo))              # combined -> truck only
            if len(T) > 1:
                cands.append((T_wo, D))              # combined -> drone only
        elif inT:
            if len(T) > 1:
                for p in range(len(D) + 1):          # truck -> drone only
                    cands.append((T_wo, D[:p] + [v] + D[p:]))
            for p in range(len(D) + 1):              # truck -> combined
                cands.append((T, D[:p] + [v] + D[p:]))
        else:  # drone only
            if len(D) > 2:
                for p in range(len(T) + 1):          # drone -> truck only
                    cands.append((T[:p] + [v] + T[p:], D_wo))
            for p in range(len(T) + 1):              # drone -> combined
                cands.append((T[:p] + [v] + T[p:], D))
        # relocation inside each visited route
        if inT:
            for p in range(len(T_wo) + 1):
                nt = T_wo[:p] + [v] + T_wo[p:]
                if nt != T:
                    cands.append((nt, D))
        if inD:
            for p in range(len(D_wo) + 1):
                nd = D_wo[:p] + [v] + D_wo[p:]
                if nd != D:
                    cands.append((T, nd))
        for (nt, nd) in cands:
            obj, _ = evaluate(n, tt, td, nt, nd)
            if obj < cur_obj - 1e-9:
                return (nt, nd, obj)
    # 2-opt segment reversals inside each route
    for which in (0, 1):
        R = T if which == 0 else D
        m = len(R)
        for i in range(m - 1):
            if time.time() > deadline:
                return None
            for j in range(i + 1, m):
                nR = R[:i] + R[i:j + 1][::-1] + R[j + 1:]
                if which == 0:
                    obj, _ = evaluate(n, tt, td, nR, D)
                    if obj < cur_obj - 1e-9:
                        return (nR, D, obj)
                else:
                    obj, _ = evaluate(n, tt, td, T, nR)
                    if obj < cur_obj - 1e-9:
                        return (T, nR, obj)
    return None


def ls_improve(n, tt, td, T, D, deadline):
    obj, _ = evaluate(n, tt, td, T, D)
    while time.time() < deadline:
        mv = find_move(n, tt, td, T, D, obj, deadline)
        if mv is None:
            break
        T, D, obj = mv
    return T, D, obj


def perturb(n, tt, td, T, D, rng):
    for _ in range(20):
        T2 = list(T)
        D2 = list(D)
        for _ in range(rng.randint(1, 3)):
            v = rng.randint(1, n)
            if v in T2:
                T2.remove(v)
            if v in D2:
                D2.remove(v)
            newtype = rng.choice(("truck", "drone", "combined"))
            if newtype in ("truck", "combined"):
                T2.insert(rng.randint(0, len(T2)), v)
            if newtype in ("drone", "combined"):
                D2.insert(rng.randint(0, len(D2)), v)
        if len(T2) >= 1 and len(D2) >= 2:
            obj, _ = evaluate(n, tt, td, T2, D2)
            if obj < INF:
                return T2, D2
    return list(T), list(D)


def heuristic_optimize(n, tt, td, T, D, deadline, logger, log_state, rng):
    T, D, obj = ls_improve(n, tt, td, T, D, deadline)
    obj, arr = evaluate(n, tt, td, T, D)
    record(n, T, D, obj, arr, logger, log_state)
    best_T, best_D, best_obj = list(T), list(D), obj
    while time.time() < deadline:
        T2, D2 = perturb(n, tt, td, best_T, best_D, rng)
        obj2, _ = evaluate(n, tt, td, T2, D2)
        if obj2 >= INF:
            continue
        T2, D2, obj2 = ls_improve(n, tt, td, T2, D2, deadline)
        if obj2 < best_obj - 1e-9:
            obj2, arr2 = evaluate(n, tt, td, T2, D2)
            best_T, best_D, best_obj = list(T2), list(D2), obj2
            record(n, T2, D2, obj2, arr2, logger, log_state)
    return best_T, best_D, best_obj


# ---------------------------------------------------------------------------
# Exact MIP via Gurobi (used when the instance size is manageable)
# ---------------------------------------------------------------------------
def run_gurobi(n, tt, td, start_T, start_D, start_arr, time_remaining,
               logger, log_state):
    import gurobipy as gp
    from gurobipy import GRB

    dest = n + 1
    C = list(range(1, n + 1))
    arcs = [(0, j) for j in C]
    arcs += [(i, j) for i in C for j in C if i != j]
    arcs += [(i, dest) for i in C]
    succ = {i: [] for i in range(n + 2)}
    pred = {i: [] for i in range(n + 2)}
    for (i, j) in arcs:
        succ[i].append(j)
        pred[j].append(i)
    max_t = 1.0
    for (i, j) in arcs:
        if tt[i][j] > max_t:
            max_t = tt[i][j]
        if td[i][j] > max_t:
            max_t = td[i][j]
    UB = (n + 1) * max_t

    model = gp.Model("truck_drone")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, float(time_remaining))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    y = model.addVars(arcs, vtype=GRB.BINARY, name="y")
    zt = model.addVars(C, vtype=GRB.BINARY, name="zt")
    zd = model.addVars(C, vtype=GRB.BINARY, name="zd")
    zc = model.addVars(C, vtype=GRB.BINARY, name="zc")
    a = model.addVars(range(n + 2), lb=0.0, name="a")
    fx = model.addVars(arcs, lb=0.0, name="fx")
    fy = model.addVars(arcs, lb=0.0, name="fy")
    model.update()

    tv = {i: zt[i] + zc[i] for i in C}
    dv = {i: zd[i] + zc[i] for i in C}

    model.addConstr(a[0] == 0.0, "a0")
    for i in C:
        model.addConstr(zt[i] + zd[i] + zc[i] == 1, "type_%d" % i)
    model.addConstr(gp.quicksum(x[0, j] for j in C) == 1, "t_src")
    model.addConstr(gp.quicksum(x[i, dest] for i in C) == 1, "t_snk")
    model.addConstr(gp.quicksum(y[0, j] for j in C) == 1, "d_src")
    model.addConstr(gp.quicksum(y[i, dest] for i in C) == 1, "d_snk")
    for i in C:
        model.addConstr(gp.quicksum(x[j, i] for j in pred[i]) == tv[i], "t_in_%d" % i)
        model.addConstr(gp.quicksum(x[i, j] for j in succ[i]) == tv[i], "t_out_%d" % i)
        model.addConstr(gp.quicksum(y[j, i] for j in pred[i]) == dv[i], "d_in_%d" % i)
        model.addConstr(gp.quicksum(y[i, j] for j in succ[i]) == dv[i], "d_out_%d" % i)
    for (i, j) in arcs:
        model.addConstr(a[j] >= a[i] + tt[i][j] - (UB + tt[i][j]) * (1 - x[i, j]),
                        "tt_%d_%d" % (i, j))
        model.addConstr(a[j] >= a[i] + td[i][j] - (UB + td[i][j]) * (1 - y[i, j]),
                        "td_%d_%d" % (i, j))
    for ii in range(len(C)):
        for jj in range(ii + 1, len(C)):
            i, j = C[ii], C[jj]
            model.addConstr(y[i, j] + y[j, i] <= zc[i] + zc[j], "pair_%d_%d" % (i, j))
    for j in C:
        model.addConstr(y[0, j] + y[j, dest] <= 1, "nosingle_%d" % j)
    model.addConstr(gp.quicksum(dv[i] for i in C) >= 2, "drone_min")
    # single-commodity flows: connectivity / subtour elimination (robust even
    # if some travel times are zero)
    model.addConstr(gp.quicksum(fx[0, j] for j in C) == gp.quicksum(tv[i] for i in C), "fx_src")
    model.addConstr(gp.quicksum(fy[0, j] for j in C) == gp.quicksum(dv[i] for i in C), "fy_src")
    for i in C:
        model.addConstr(gp.quicksum(fx[j, i] for j in pred[i])
                        - gp.quicksum(fx[i, j] for j in succ[i]) == tv[i], "fx_bal_%d" % i)
        model.addConstr(gp.quicksum(fy[j, i] for j in pred[i])
                        - gp.quicksum(fy[i, j] for j in succ[i]) == dv[i], "fy_bal_%d" % i)
    for (i, j) in arcs:
        model.addConstr(fx[i, j] <= n * x[i, j], "fx_cap_%d_%d" % (i, j))
        model.addConstr(fy[i, j] <= n * y[i, j], "fy_cap_%d_%d" % (i, j))

    model.setObjective(a[dest], GRB.MINIMIZE)

    # ---- complete feasible MIP start from the heuristic solution ----
    Tset = set(start_T)
    Dset = set(start_D)
    Rt = [0] + list(start_T) + [dest]
    Rd = [0] + list(start_D) + [dest]
    tarc = set((Rt[k], Rt[k + 1]) for k in range(len(Rt) - 1))
    darc = set((Rd[k], Rd[k + 1]) for k in range(len(Rd) - 1))
    for (i, j) in arcs:
        x[i, j].Start = 1.0 if (i, j) in tarc else 0.0
        y[i, j].Start = 1.0 if (i, j) in darc else 0.0
        fx[i, j].Start = 0.0
        fy[i, j].Start = 0.0
    kT = len(start_T)
    for m in range(len(start_T)):
        fx[Rt[m], Rt[m + 1]].Start = float(kT - m)
    kD = len(start_D)
    for m in range(len(start_D)):
        fy[Rd[m], Rd[m + 1]].Start = float(kD - m)
    for i in C:
        zt[i].Start = 1.0 if (i in Tset and i not in Dset) else 0.0
        zd[i].Start = 1.0 if (i in Dset and i not in Tset) else 0.0
        zc[i].Start = 1.0 if (i in Tset and i in Dset) else 0.0
    for i in range(n + 2):
        a[i].Start = float(start_arr[i])

    x_list = [x[e] for e in arcs]
    y_list = [y[e] for e in arcs]

    def routes_from(xv, yv):
        succT = {}
        succD = {}
        for (e, v) in zip(arcs, xv):
            if v > 0.5:
                succT[e[0]] = e[1]
        for (e, v) in zip(arcs, yv):
            if v > 0.5:
                succD[e[0]] = e[1]
        T = []
        cur = 0
        seen = {0}
        while True:
            cur = succT.get(cur)
            if cur is None:
                return None
            if cur == dest:
                break
            if cur in seen:
                return None
            seen.add(cur)
            T.append(cur)
        D = []
        cur = 0
        seen = {0}
        while True:
            cur = succD.get(cur)
            if cur is None:
                return None
            if cur == dest:
                break
            if cur in seen:
                return None
            seen.add(cur)
            D.append(cur)
        return T, D

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = m.cbGetSolution(x_list)
                yv = m.cbGetSolution(y_list)
                r = routes_from(xv, yv)
                if r is None:
                    return
                T, D = r
                obj, arr = evaluate(n, tt, td, T, D)
                if obj >= INF:
                    return
                record(n, T, D, obj, arr, logger, log_state)
            except Exception:
                pass

    model.optimize(cb)

    best = None
    try:
        if model.SolCount > 0:
            xv = model.getAttr("X", x_list)
            yv = model.getAttr("X", y_list)
            r = routes_from(xv, yv)
            if r is not None:
                T, D = r
                obj, arr = evaluate(n, tt, td, T, D)
                if obj < INF:
                    best = (obj, T, D, arr)
    except Exception:
        best = None
    return best


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

    t_start = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        inst = json.load(f)
    n = int(inst["num_customers"])

    # Extended matrices including the artificial depot-destination node n+1.
    tt = extend_matrix(inst["truck_travel_times"], n)
    td = extend_matrix(inst["drone_travel_times"], n)

    rng = random.Random(0)
    log_state = {"last": INF, "best_obj": INF,
                 "best_T": None, "best_D": None, "best_arr": None}

    # ---- construction ----
    constr_deadline = t_start + min(max(2.0, 0.2 * time_limit), 30.0)
    T, D, arr, obj = build_initial(n, tt, td, constr_deadline)
    record(n, T, D, obj, arr, logger, log_state)

    # ---- local-search improvement ----
    if n <= 60:
        h_budget = min(max(1.0, 0.10 * time_limit), 15.0)
    else:
        h_budget = float(time_limit)  # large instance: heuristic only
    deadline_h = t_start + min(h_budget, max(0.0, time_limit - 0.5))
    if time.time() < deadline_h:
        heuristic_optimize(n, tt, td, log_state["best_T"], log_state["best_D"],
                           deadline_h, logger, log_state, rng)

    # ---- Gurobi exact/branch-and-cut with remaining time ----
    gurobi_ok = False
    if n <= 60:
        remaining = time_limit - (time.time() - t_start) - 0.6
        if remaining > 1.0:
            try:
                res = run_gurobi(n, tt, td,
                                 log_state["best_T"], log_state["best_D"],
                                 log_state["best_arr"], remaining,
                                 logger, log_state)
                gurobi_ok = True
                if res is not None:
                    obj2, T2, D2, arr2 = res
                    record(n, T2, D2, obj2, arr2, logger, log_state)
            except Exception:
                gurobi_ok = False

    # ---- if Gurobi was unavailable, keep improving heuristically ----
    if not gurobi_ok:
        extra_deadline = t_start + time_limit - 0.4
        if time.time() < extra_deadline:
            heuristic_optimize(n, tt, td, log_state["best_T"], log_state["best_D"],
                               extra_deadline, logger, log_state, rng)

    # ---- safety fallback (should not be needed) ----
    if log_state["best_T"] is None:
        T = list(range(1, n + 1))
        D = [1, 2 if n >= 2 else 1]
        obj, arr = evaluate(n, tt, td, T, D)
        record(n, T, D, obj, arr, logger, log_state)

    # ---- write final solution ----
    sol = build_solution(n, log_state["best_T"], log_state["best_D"],
                         log_state["best_arr"], log_state["best_obj"])
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()