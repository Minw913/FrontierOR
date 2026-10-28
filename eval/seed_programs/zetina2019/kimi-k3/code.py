import argparse
import json
import time
import heapq
from collections import deque


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    time_limit = max(1, int(args.time_limit))

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    # ---------------- Read instance ----------------
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    arcs = [(int(a[0]), int(a[1])) for a in inst["arcs"]]
    A = len(arcs)
    N = int(inst.get("num_nodes", 0))
    if arcs:
        mx = max(max(u, v) for u, v in arcs) + 1
        if mx > N:
            N = mx
    commodities = inst["commodities"]
    K = len(commodities)
    fixed_costs = inst["fixed_costs"]
    variable_costs = inst["variable_costs"]
    keys = ["{}_{}".format(u, v) for (u, v) in arcs]
    fixed = [float(fixed_costs[keys[i]]) for i in range(A)]
    vc = [[float(x) for x in variable_costs[keys[i]]] for i in range(A)]
    orig = [int(c["origin"]) for c in commodities]
    dest = [int(c["destination"]) for c in commodities]
    dem = [float(c["demand"]) for c in commodities]

    out_adj = [[] for _ in range(N)]
    rev_adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(arcs):
        out_adj[u].append((v, i))
        rev_adj[v].append((u, i))

    INF = float("inf")

    # ---------------- Utilities ----------------
    def dijkstra_k(s, t, k, open_mask, banned=-1):
        """Shortest path for commodity k (per-unit costs) over open arcs."""
        if s == t:
            return [], 0.0
        dist = [INF] * N
        prev = [-1] * N
        dist[s] = 0.0
        pq = [(0.0, s)]
        push = heapq.heappush
        pop = heapq.heappop
        while pq:
            d, u = pop(pq)
            if d > dist[u] + 1e-9:
                continue
            if u == t:
                break
            for (v, ai) in out_adj[u]:
                if ai == banned or not open_mask[ai]:
                    continue
                nd = d + vc[ai][k]
                if nd < dist[v] - 1e-9:
                    dist[v] = nd
                    prev[v] = ai
                    push(pq, (nd, v))
        if dist[t] == INF:
            return None, INF
        path = []
        cur = t
        while cur != s:
            ai = prev[cur]
            if ai < 0:
                return None, INF
            path.append(ai)
            cur = arcs[ai][0]
        path.reverse()
        return path, dist[t]

    def dijkstra_all(s, k, adj):
        dist = [INF] * N
        dist[s] = 0.0
        pq = [(0.0, s)]
        push = heapq.heappush
        pop = heapq.heappop
        while pq:
            d, u = pop(pq)
            if d > dist[u] + 1e-9:
                continue
            for (v, ai) in adj[u]:
                nd = d + vc[ai][k]
                if nd < dist[v] - 1e-9:
                    dist[v] = nd
                    push(pq, (nd, v))
        return dist

    def reachable(start, adj):
        seen = [False] * N
        dq = deque([start])
        seen[start] = True
        while dq:
            u = dq.popleft()
            for (v, _) in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    dq.append(v)
        return seen

    def make_sol(obj, open_mask, path_list=None, flow_list=None):
        open_arcs = {keys[i]: 1 for i in range(A) if open_mask[i]}
        routings = {}
        for k in range(K):
            d = {}
            if flow_list is not None:
                fk = flow_list.get(k)
                if fk:
                    for ai, val in fk.items():
                        if val > 1e-7:
                            d[keys[ai]] = float(val)
            elif path_list is not None:
                pk = path_list[k]
                if pk:
                    for ai in pk:
                        d[keys[ai]] = 1.0
            routings[str(k)] = d
        return {"objective_value": float(obj),
                "open_arcs": open_arcs,
                "routings": routings}

    last_logged = [INF]

    def log_if_better(obj, sol):
        if logger is not None and obj < last_logged[0] - 1e-9:
            last_logged[0] = float(obj)
            try:
                logger.log_solution(float(obj), sol)
            except Exception:
                pass

    def polish(open_mask):
        """Recompute exact shortest-path routing per commodity on open_mask."""
        fl = {}
        tot = 0.0
        for i in range(A):
            if open_mask[i]:
                tot += fixed[i]
        for k in range(K):
            p, c = dijkstra_k(orig[k], dest[k], k, open_mask)
            if p is None:
                return None, INF
            tot += dem[k] * c
            fl[k] = {ai: 1.0 for ai in p} if p else {}
        return fl, tot

    # ---------------- Heuristic: union of shortest paths ----------------
    all_open = [True] * A
    best_mask = [False] * A
    paths = [None] * K
    path_sets = [set() for _ in range(K)]
    pcost = [0.0] * K
    for k in range(K):
        p, c = dijkstra_k(orig[k], dest[k], k, all_open)
        if p is None:
            p = []
            c = 0.0
        paths[k] = p
        path_sets[k] = set(p)
        pcost[k] = c
        for ai in p:
            best_mask[ai] = True

    def total_cost(mask, pc):
        fsum = 0.0
        for i in range(A):
            if mask[i]:
                fsum += fixed[i]
        vsum = 0.0
        for k in range(K):
            vsum += dem[k] * pc[k]
        return fsum + vsum

    best_obj = total_cost(best_mask, pcost)
    best_sol = make_sol(best_obj, best_mask, path_list=paths)
    log_if_better(best_obj, best_sol)

    # ---------------- Drop heuristic (single-arc closure local search) ----------------
    heur_budget = min(10.0, max(0.3, 0.15 * time_limit))
    heur_deadline = time.time() + heur_budget
    improved = True
    while improved and time.time() < heur_deadline:
        improved = False
        open_list = [i for i in range(A) if best_mask[i]]
        open_list.sort(key=lambda i: -fixed[i])
        for a in open_list:
            if time.time() > heur_deadline:
                break
            if not best_mask[a]:
                continue
            affected = [k for k in range(K) if a in path_sets[k]]
            if not affected:
                best_mask[a] = False
                improved = True
                continue
            delta = -fixed[a]
            newinfo = {}
            ok = True
            for k in affected:
                p, c = dijkstra_k(orig[k], dest[k], k, best_mask, banned=a)
                if p is None:
                    ok = False
                    break
                delta += dem[k] * (c - pcost[k])
                newinfo[k] = (p, c)
                if delta >= 0.0:  # remaining affected costs can only increase
                    ok = False
                    break
            if ok and delta < -1e-9:
                best_mask[a] = False
                for k, (p, c) in newinfo.items():
                    paths[k] = p
                    path_sets[k] = set(p)
                    pcost[k] = c
                improved = True

    obj_h = total_cost(best_mask, pcost)
    if obj_h < best_obj - 1e-9:
        best_obj = obj_h
        best_sol = make_sol(best_obj, best_mask, path_list=paths)
        log_if_better(best_obj, best_sol)

    # ---------------- MIP (strong fixed-charge multicommodity formulation) ----------------
    modeled = [k for k in range(K) if orig[k] != dest[k]]
    cand = {}
    for k in modeled:
        ro = reachable(orig[k], out_adj)
        rd = reachable(dest[k], rev_adj)
        cand[k] = [i for i in range(A) if ro[arcs[i][0]] and rd[arcs[i][1]]]
    n_x = sum(len(cand[k]) for k in modeled)

    MAX_X = 2000000
    if n_x > MAX_X and modeled:
        # Restrict per-commodity candidate arcs to near-shortest ones.
        for margin in (1.5, 1.25, 1.1, 1.02, 1.0):
            cand2 = {}
            cnt = 0
            for k in modeled:
                d_o = dijkstra_all(orig[k], k, out_adj)
                d_d = dijkstra_all(dest[k], k, rev_adj)
                sp = d_o[dest[k]]
                if sp == INF:
                    cand2[k] = []
                    continue
                thr = sp * margin + 1e-9
                lst = []
                for i in range(A):
                    u, v = arcs[i]
                    du = d_o[u]
                    dv = d_d[v]
                    if du < INF and dv < INF and du + vc[i][k] + dv <= thr:
                        lst.append(i)
                cand2[k] = lst
                cnt += len(lst)
            cand = cand2
            n_x = cnt
            if cnt <= MAX_X:
                break

    # Ensure heuristic path arcs are candidates (validity of MIP start).
    for k in modeled:
        if paths[k]:
            s = set(cand[k])
            extra = False
            for i in paths[k]:
                if i not in s:
                    s.add(i)
                    extra = True
            if extra:
                cand[k] = sorted(s)
    n_x = sum(len(cand[k]) for k in modeled)

    run_mip = bool(modeled) and n_x <= 3000000
    remaining = time_limit - (time.time() - t0) - 0.5

    if run_mip and remaining > 1.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("fixed_charge_multicommodity")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1

            y_td = m.addVars(A, vtype=GRB.BINARY, name="y")
            yl = [y_td[i] for i in range(A)]
            idx_list = []
            for k in modeled:
                for i in cand[k]:
                    idx_list.append((k, i))
            x_td = m.addVars(idx_list, lb=0.0, ub=1.0, name="x")
            xl = [x_td[key] for key in idx_list]
            nxl = len(idx_list)

            coef = [dem[k] * vc[i][k] for (k, i) in idx_list]
            m.setObjective(gp.LinExpr(fixed, yl) + gp.LinExpr(coef, xl),
                           GRB.MINIMIZE)

            # Strong linking constraints x[k,i] <= y[i]
            m.addConstrs((xl[j] <= yl[idx_list[j][1]] for j in range(nxl)),
                         name="link")

            # Flow conservation: inflow - outflow = b
            for k in modeled:
                inflow = {}
                outflow = {}
                nodes = set()
                for i in cand[k]:
                    u, v = arcs[i]
                    var = x_td[(k, i)]
                    outflow.setdefault(u, []).append(var)
                    inflow.setdefault(v, []).append(var)
                    nodes.add(u)
                    nodes.add(v)
                okk = orig[k]
                dkk = dest[k]
                for node in nodes:
                    b = 0.0
                    if node == okk:
                        b = -1.0
                    elif node == dkk:
                        b = 1.0
                    inl = inflow.get(node)
                    outl = outflow.get(node)
                    expr = gp.quicksum(inl or []) - gp.quicksum(outl or [])
                    m.addConstr(expr == b)

            # MIP start from heuristic solution
            ystart = [1.0 if best_mask[i] else 0.0 for i in range(A)]
            m.setAttr("Start", yl, ystart)
            xstart = [1.0 if i in path_sets[k] else 0.0 for (k, i) in idx_list]
            m.setAttr("Start", xl, xstart)

            cb_state = {"last": best_obj}

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        objv = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                        if objv < cb_state["last"] - 1e-9:
                            cb_state["last"] = objv
                            yv = model.cbGetSolution(yl)
                            xv = model.cbGetSolution(xl)
                            om = [yv[i] > 0.5 for i in range(A)]
                            fl = {}
                            for j in range(nxl):
                                val = xv[j]
                                if val > 1e-7:
                                    kj, ij = idx_list[j]
                                    fl.setdefault(kj, {})[ij] = float(val)
                            sol = make_sol(objv, om, flow_list=fl)
                            log_if_better(objv, sol)
                    except Exception:
                        pass

            tl = time_limit - (time.time() - t0) - 0.5
            m.Params.TimeLimit = max(0.1, tl)
            m.optimize(callback)

            if m.SolCount > 0:
                objv = float(m.ObjVal)
                yv = m.getAttr("X", yl)
                om = [yv[i] > 0.5 for i in range(A)]
                xv = m.getAttr("X", xl)
                fl = {}
                for j in range(nxl):
                    val = xv[j]
                    if val > 1e-7:
                        kj, ij = idx_list[j]
                        fl.setdefault(kj, {})[ij] = float(val)
                # Polish: re-optimize routing exactly on the open arc set
                pf, pt = polish(om)
                if pf is not None and pt <= objv + 1e-7:
                    objv = pt
                    fl = pf
                if objv < best_obj - 1e-9:
                    best_obj = objv
                    best_mask = om
                    best_sol = make_sol(objv, om, flow_list=fl)
                    log_if_better(best_obj, best_sol)
        except Exception:
            pass

    # ---------------- Final polish and output ----------------
    pf, pt = polish(best_mask)
    if pf is not None and pt < best_obj - 1e-9:
        best_obj = pt
        best_sol = make_sol(best_obj, best_mask, flow_list=pf)
        log_if_better(best_obj, best_sol)

    out = {
        "objective_value": float(best_obj),
        "open_arcs": best_sol["open_arcs"],
        "routings": best_sol["routings"],
    }
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()