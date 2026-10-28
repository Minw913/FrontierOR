import argparse
import json
import time
import heapq
from collections import defaultdict


def fnum(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    time_limit = max(1, int(args.time_limit))
    deadline = t0 + float(time_limit)

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    params = inst.get("parameters", {})
    alpha = fnum(params.get("alpha", 1.0), 1.0)
    quota = fnum(params.get("quota", 0.0), 0.0)

    nodes = inst.get("nodes", [])
    edges_in = inst.get("edges", [])

    nodeinfo = {}
    sub_ids = []
    ns_ids = []
    for nd in nodes:
        nid = nd["id"]
        nodeinfo[nid] = nd
        if nd.get("type") == "substation":
            sub_ids.append(nid)
        else:
            ns_ids.append(nid)

    idx_of = {nid: i for i, nid in enumerate(ns_ids)}
    n = len(ns_ids)
    ROOT = n
    INF = float("inf")

    profit = [0.0] * n
    ncost = [0.0] * n
    nscen = [0.0] * n
    nweight = [0.0] * n
    is_turb = [False] * n
    for i, nid in enumerate(ns_ids):
        nd = nodeinfo[nid]
        if nd.get("type") == "potential_turbine":
            is_turb[i] = True
            profit[i] = fnum(nd.get("profit"))
            ncost[i] = fnum(nd.get("cost"))
            nscen[i] = fnum(nd.get("scenic_impact"))
            nweight[i] = alpha * ncost[i] + (1.0 - alpha) * nscen[i]
    turbines = [i for i in range(n) if is_turb[i]]

    # ---- collapse substations into ROOT; dedupe parallel edges (min weight) ----
    best_edge = {}
    sub_pair_edges = []
    for e in edges_in:
        a = e["from"]
        b = e["to"]
        c = fnum(e.get("cost"))
        s = fnum(e.get("scenic_impact"))
        w = alpha * c + (1.0 - alpha) * s
        ia = idx_of.get(a, ROOT)
        ib = idx_of.get(b, ROOT)
        if ia == ROOT and ib == ROOT:
            sub_pair_edges.append((a, b))
            continue
        if ia == ib:
            continue
        u, v = (ia, ib) if ia < ib else (ib, ia)
        cur = best_edge.get((u, v))
        if cur is None or w < cur[2]:
            best_edge[(u, v)] = (u, v, w, c, s, a, b)
    E = list(best_edge.values())  # (u, v, w, cost, scen, orig_a, orig_b)
    m_edges = len(E)
    pair_to_ei = {(E[ei][0], E[ei][1]): ei for ei in range(m_edges)}

    adj = [[] for _ in range(n + 1)]
    for ei in range(m_edges):
        u, v, w = E[ei][0], E[ei][1], E[ei][2]
        adj[u].append((v, w, ei))
        adj[v].append((u, w, ei))

    # ---- zero-cost substation backbone spanning tree ----
    uf = {s: s for s in sub_ids}

    def find(x):
        r = x
        while uf[r] != r:
            r = uf[r]
        while uf[x] != r:
            uf[x], x = r, uf[x]
        return r

    sub_edges_sel = []
    for a, b in sub_pair_edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[ra] = rb
            sub_edges_sel.append((a, b))
    if len(sub_ids) > 1:
        reps = []
        seen_r = set()
        for s in sub_ids:
            r = find(s)
            if r not in seen_r:
                seen_r.add(r)
                reps.append(s)
        for s in reps[1:]:
            sub_edges_sel.append((reps[0], s))

    def dijkstra(sources):
        d = [INF] * (n + 1)
        par = [-1] * (n + 1)
        pq = []
        for s0 in sources:
            if d[s0] > 0.0:
                d[s0] = 0.0
                pq.append((0.0, s0))
        heapq.heapify(pq)
        while pq:
            dist, u = heapq.heappop(pq)
            if dist > d[u] + 1e-12:
                continue
            for v, w, ei in adj[u]:
                ndist = dist + w
                if ndist < d[v] - 1e-12:
                    d[v] = ndist
                    par[v] = u
                    heapq.heappush(pq, (ndist, v))
        return d, par

    heur_budget = min(max(1.0, 0.2 * time_limit), 0.5 * time_limit)
    heur_deadline = t0 + heur_budget

    def add_path(v, par, in_T, sel_nodes, sel_edges, tot):
        while v != ROOT and not in_T[v]:
            in_T[v] = 1
            sel_nodes.add(v)
            if is_turb[v]:
                tot += profit[v]
            u = par[v]
            if u < 0:
                break
            key = (u, v) if u < v else (v, u)
            ei = pair_to_ei.get(key)
            if ei is not None:
                sel_edges.add(ei)
            v = u
        return tot

    def greedy_dynamic():
        in_T = bytearray(n + 1)
        in_T[ROOT] = 1
        tree_list = [ROOT]
        sel_nodes = set()
        sel_edges = set()
        tot = 0.0
        while tot < quota - 1e-9:
            over = time.time() > heur_deadline
            d, par = dijkstra(tree_list)
            added = False
            while True:
                best = -1
                best_ratio = -1.0
                for t in turbines:
                    if in_T[t] or d[t] == INF:
                        continue
                    c = d[t] + nweight[t]
                    r = profit[t] / (c if c > 1e-12 else 1e-12)
                    if r > best_ratio:
                        best_ratio = r
                        best = t
                if best < 0:
                    break
                newly = []
                v = best
                while v != ROOT and not in_T[v]:
                    in_T[v] = 1
                    sel_nodes.add(v)
                    newly.append(v)
                    if is_turb[v]:
                        tot += profit[v]
                    u = par[v]
                    if u < 0:
                        break
                    key = (u, v) if u < v else (v, u)
                    ei = pair_to_ei.get(key)
                    if ei is not None:
                        sel_edges.add(ei)
                    v = u
                tree_list.extend(newly)
                added = True
                if not over or tot >= quota - 1e-9:
                    break
            if not added or over:
                break
        return sel_nodes, sel_edges, tot

    def greedy_static():
        d, par = dijkstra([ROOT])
        cands = []
        for t in turbines:
            if d[t] < INF:
                c = d[t] + nweight[t]
                cands.append((profit[t] / (c if c > 1e-12 else 1e-12), t))
        cands.sort(reverse=True)
        in_T = bytearray(n + 1)
        in_T[ROOT] = 1
        sel_nodes = set()
        sel_edges = set()
        tot = 0.0
        for _, t in cands:
            if tot >= quota - 1e-9:
                break
            tot = add_path(t, par, in_T, sel_nodes, sel_edges, tot)
        return sel_nodes, sel_edges, tot

    def prim_mst(node_set_wr):
        visited = bytearray(n + 1)
        visited[ROOT] = 1
        cnt = 1
        total = len(node_set_wr)
        chosen = set()
        pq = []
        for v, w, ei in adj[ROOT]:
            if v in node_set_wr:
                pq.append((w, ei, v))
        heapq.heapify(pq)
        while pq and cnt < total:
            w, ei, v = heapq.heappop(pq)
            if visited[v]:
                continue
            visited[v] = 1
            cnt += 1
            chosen.add(ei)
            for vv, ww, eii in adj[v]:
                if not visited[vv] and vv in node_set_wr:
                    heapq.heappush(pq, (ww, eii, vv))
        if cnt < total:
            return None
        return chosen

    def prune(sel_nodes, sel_edges):
        node_set = set(sel_nodes)
        edge_set = set(sel_edges)
        adjE = defaultdict(list)
        for ei in edge_set:
            u, v = E[ei][0], E[ei][1]
            adjE[u].append((ei, v))
            adjE[v].append((ei, u))
        tot = 0.0
        for i in node_set:
            if is_turb[i]:
                tot += profit[i]
        stack = [u for u in node_set if len(adjE[u]) == 1]
        while stack:
            u = stack.pop()
            if u not in node_set or u == ROOT:
                continue
            if len(adjE[u]) != 1:
                continue
            if is_turb[u] and tot - profit[u] < quota - 1e-9:
                continue
            ei, v = adjE[u][0]
            edge_set.discard(ei)
            node_set.discard(u)
            adjE[u] = []
            if is_turb[u]:
                tot -= profit[u]
            if v != ROOT:
                adjE[v] = [(e2, o2) for (e2, o2) in adjE[v] if e2 != ei]
                if len(adjE[v]) == 1:
                    stack.append(v)
        return node_set, edge_set

    def polish(sel_nodes, sel_edges):
        wr = set(sel_nodes)
        wr.add(ROOT)
        pm = prim_mst(wr)
        if pm is not None:
            sel_edges = pm
        return prune(sel_nodes, sel_edges)

    def compute_flows(sel_nodes, sel_edges):
        adjE = [[] for _ in range(n + 1)]
        for ei in sel_edges:
            u, v = E[ei][0], E[ei][1]
            adjE[u].append((ei, v))
            adjE[v].append((ei, u))
        visited = [False] * (n + 1)
        visited[ROOT] = True
        order = [ROOT]
        par_node = [-1] * (n + 1)
        par_edge = [-1] * (n + 1)
        for u in order:
            for ei, v in adjE[u]:
                if not visited[v]:
                    visited[v] = True
                    par_node[v] = u
                    par_edge[v] = ei
                    order.append(v)
        size = [0] * (n + 1)
        for u in order:
            if u != ROOT:
                size[u] = 1
        for u in reversed(order):
            if u != ROOT:
                size[par_node[u]] += size[u]
        flows = {}
        for u in order:
            if u == ROOT:
                continue
            ei = par_edge[u]
            if ei < 0:
                continue
            a, b = E[ei][0], E[ei][1]
            if a == par_node[u]:
                flows[(ei, 0)] = float(size[u])
            else:
                flows[(ei, 1)] = float(size[u])
        return flows

    def evaluate(sel_turbs, sel_eids):
        tc = 0.0
        ts = 0.0
        for ei in sel_eids:
            tc += E[ei][3]
            ts += E[ei][4]
        for i in sel_turbs:
            tc += ncost[i]
            ts += nscen[i]
        return alpha * tc + (1.0 - alpha) * ts

    def build_solution(sel_turbs, sel_eids):
        obj = evaluate(sel_turbs, sel_eids)
        se = [{"from": int(E[ei][5]), "to": int(E[ei][6])} for ei in sorted(sel_eids)]
        for a, b in sub_edges_sel:
            se.append({"from": int(a), "to": int(b)})
        return obj, {
            "objective_value": obj,
            "selected_turbines": sorted(int(ns_ids[i]) for i in sel_turbs),
            "selected_edges": se,
        }

    def verify(sol):
        try:
            tp = 0.0
            for t in sol["selected_turbines"]:
                tp += fnum(nodeinfo[t].get("profit"))
            if tp < quota - 1e-6:
                return False
            g = defaultdict(list)
            for e in sol["selected_edges"]:
                a, b = e["from"], e["to"]
                g[a].append(b)
                g[b].append(a)
            need = set(sub_ids) | set(sol["selected_turbines"])
            if not need:
                return len(sol["selected_edges"]) == 0
            start = next(iter(need))
            seen = {start}
            st = [start]
            while st:
                u = st.pop()
                for v in g[u]:
                    if v not in seen:
                        seen.add(v)
                        st.append(v)
            if not need.issubset(seen):
                return False
            return len(sol["selected_edges"]) == len(seen) - 1
        except Exception:
            return False

    # ---- heuristic candidates ----
    best_obj = INF
    best_sol = None
    best_sn = set()
    best_se = set()
    fallback_sol = None
    for fn in (greedy_dynamic, greedy_static):
        try:
            sn, se, _ = fn()
            sn, se = polish(sn, se)
            sturbs = [i for i in sn if is_turb[i]]
            feas = sum(profit[i] for i in sturbs) >= quota - 1e-9
            obj, sol = build_solution(sturbs, se)
            if fallback_sol is None or (feas and obj < fnum(fallback_sol.get("objective_value"), INF)):
                fallback_sol = sol
            if feas and obj < best_obj - 1e-9:
                best_obj = obj
                best_sol = sol
                best_sn = set(sn)
                best_se = set(se)
        except Exception:
            pass

    if fallback_sol is None:
        fallback_sol = {
            "objective_value": 0.0,
            "selected_turbines": [],
            "selected_edges": [{"from": int(a), "to": int(b)} for a, b in sub_edges_sel],
        }

    if best_sol is not None and logger:
        logger.log_solution(best_obj, best_sol)

    # ---- MIP (single-commodity flow Steiner tree with quota) ----
    d0, _ = dijkstra([ROOT])
    reachable_profit = sum(profit[t] for t in turbines if d0[t] < INF)
    instance_feasible = reachable_profit >= quota - 1e-9

    remaining = deadline - time.time() - 0.75
    run_mip = (
        instance_feasible
        and remaining > 5.0
        and n > 0
        and m_edges > 0
        and best_obj > 1e-12
        and n <= 10000
        and m_edges <= 120000
    )

    if run_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("wind_farm")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0

            x = model.addVars(m_edges, vtype=GRB.BINARY, name="x")
            z = model.addVars(n, vtype=GRB.BINARY, name="z")
            M = float(max(1, n))
            fu = model.addVars(m_edges, lb=0.0, ub=M, name="fu")
            fv = model.addVars(m_edges, lb=0.0, ub=M, name="fv")

            model.setObjective(
                gp.quicksum(E[ei][2] * x[ei] for ei in range(m_edges))
                + gp.quicksum(nweight[i] * z[i] for i in turbines),
                GRB.MINIMIZE,
            )

            inc = [[] for _ in range(n)]
            for ei in range(m_edges):
                u, v = E[ei][0], E[ei][1]
                if u != ROOT:
                    inc[u].append(ei)
                if v != ROOT:
                    inc[v].append(ei)

            for i in range(n):
                le = gp.LinExpr()
                for ei in inc[i]:
                    if E[ei][0] == i:
                        le.add(fv[ei], 1.0)
                        le.add(fu[ei], -1.0)
                    else:
                        le.add(fu[ei], 1.0)
                        le.add(fv[ei], -1.0)
                model.addConstr(le == z[i])
            for ei in range(m_edges):
                model.addConstr(fu[ei] + fv[ei] <= M * x[ei])
                u, v = E[ei][0], E[ei][1]
                if u != ROOT:
                    model.addConstr(x[ei] <= z[u])
                if v != ROOT:
                    model.addConstr(x[ei] <= z[v])
            for i in range(n):
                le = gp.LinExpr()
                for ei in inc[i]:
                    le.add(x[ei], 1.0)
                if is_turb[i]:
                    model.addConstr(le >= z[i])
                else:
                    model.addConstr(le >= 2.0 * z[i])
            model.addConstr(gp.quicksum(profit[i] * z[i] for i in turbines) >= quota)
            model.addConstr(x.sum() == z.sum())

            if best_sol is not None and len(best_se) == len(best_sn):
                for i in range(n):
                    z[i].Start = 1.0 if i in best_sn else 0.0
                for ei in range(m_edges):
                    x[ei].Start = 1.0 if ei in best_se else 0.0
                flows = compute_flows(best_sn, best_se)
                for ei in range(m_edges):
                    fu[ei].Start = flows.get((ei, 0), 0.0)
                    fv[ei].Start = flows.get((ei, 1), 0.0)

            def _cb(mdl, where):
                nonlocal best_obj, best_sol
                if where == GRB.Callback.MIPSOL:
                    try:
                        xv = mdl.cbGetSolution(x)
                        zv = mdl.cbGetSolution(z)
                        se = [ei for ei in range(m_edges) if xv[ei] > 0.5]
                        stl = [i for i in turbines if zv[i] > 0.5]
                        obj, sol = build_solution(stl, se)
                        if obj < best_obj - 1e-9:
                            best_obj = obj
                            best_sol = sol
                            if logger:
                                logger.log_solution(obj, sol)
                    except Exception:
                        pass

            model.Params.TimeLimit = max(1.0, deadline - time.time() - 0.75)
            model.optimize(_cb)

            if model.SolCount > 0:
                try:
                    xv = model.getAttr("X", x)
                    zv = model.getAttr("Z", z)
                    se = [ei for ei in range(m_edges) if xv[ei] > 0.5]
                    stl = [i for i in turbines if zv[i] > 0.5]
                    obj, sol = build_solution(stl, se)
                    if obj < best_obj - 1e-9:
                        best_obj = obj
                        best_sol = sol
                        if logger:
                            logger.log_solution(obj, sol)
                except Exception:
                    pass
        except Exception:
            pass

    if best_sol is None:
        best_sol = fallback_sol
    else:
        try:
            if not verify(best_sol) and verify(fallback_sol):
                best_sol = fallback_sol
        except Exception:
            pass

    with open(args.solution_path, "w") as fh:
        json.dump(best_sol, fh)


if __name__ == "__main__":
    main()