import argparse
import json
import os
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = max(0, int(args.time_limit))

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    depot = inst["depot"]
    arcs = [(int(a[0]), int(a[1]), a[2]) for a in inst.get("arcs", [])]
    profs = [(int(p[0]), int(p[1]), p[2]) for p in inst.get("profitable_arcs", [])]
    vi_nodes = [int(v) for v in inst.get("VI_nodes", [])]
    strong_edges = [(int(e[0]), int(e[1])) for e in inst.get("strong_incompatibilities", [])]
    weak_edges = [(int(e[0]), int(e[1]), e[2]) for e in inst.get("weak_incompatibilities", [])]

    num_arcs = len(arcs)
    num_profs = len(profs)

    empty_sol = {"objective_value": 0.0, "served_arcs": [], "tour_arcs": []}
    best_obj = 0.0
    best_sol = empty_sol
    if logger:
        logger.log_solution(best_obj, best_sol)

    def write_solution(sol):
        out_dir = os.path.dirname(args.solution_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    if time_limit <= 0 or time.time() - start_time >= time_limit:
        write_solution(best_sol)
        return

    # Arc adjacency.
    in_arcs = defaultdict(list)
    out_arcs = defaultdict(list)
    nodes = {depot}
    pair_to_arcs = defaultdict(list)
    for a, (u, v, c) in enumerate(arcs):
        in_arcs[v].append(a)
        out_arcs[u].append(a)
        pair_to_arcs[(u, v)].append(a)
        nodes.add(u)
        nodes.add(v)

    out_profs = defaultdict(list)
    rep_arc = []
    for p, (u, v, r) in enumerate(profs):
        cand = pair_to_arcs.get((u, v))
        if cand:
            a = min(cand, key=lambda idx: (arcs[idx][2], idx))
        else:
            a = None
        rep_arc.append(a)
        out_profs[u].append(p)
        nodes.add(u)
        nodes.add(v)

    graph_nodes = set(vi_nodes)
    for i, j in strong_edges:
        graph_nodes.add(i)
        graph_nodes.add(j)
    for i, j, _ in weak_edges:
        graph_nodes.add(i)
        graph_nodes.add(j)
    nodes.update(graph_nodes)
    graph_nodes = sorted(graph_nodes)

    model = gp.Model("profitable_hauler")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1
    model.Params.OutputFlag = 0

    x_td = model.addVars(num_arcs, vtype=GRB.INTEGER, lb=0, name="x")
    y_td = model.addVars(num_profs, vtype=GRB.BINARY, name="y")
    z_td = model.addVars(graph_nodes, vtype=GRB.BINARY, name="z")
    w_td = model.addVars(len(weak_edges), vtype=GRB.BINARY, name="w")

    x_vars = [x_td[i] for i in range(num_arcs)]
    y_vars = [y_td[i] for i in range(num_profs)]
    w_vars = [w_td[i] for i in range(len(weak_edges))]

    model.setObjective(
        gp.quicksum(profs[p][2] * y_td[p] for p in range(num_profs))
        - gp.quicksum(arcs[a][2] * x_td[a] for a in range(num_arcs))
        - gp.quicksum(weak_edges[e][2] * w_td[e] for e in range(len(weak_edges))),
        GRB.MAXIMIZE,
    )

    # Serving requires traversal.
    for p in range(num_profs):
        a = rep_arc[p]
        if a is None:
            model.addConstr(y_td[p] == 0, name=f"serve_impossible_{p}")
        else:
            model.addConstr(y_td[p] <= x_td[a], name=f"serve_requires_traverse_{p}")

    # Active incompatibility-graph nodes.
    for i in graph_nodes:
        outs = out_profs.get(i, [])
        for p in outs:
            model.addConstr(y_td[p] <= z_td[i], name=f"activate_{i}_{p}")
        if outs:
            model.addConstr(z_td[i] <= gp.quicksum(y_td[p] for p in outs), name=f"active_def_{i}")
        else:
            model.addConstr(z_td[i] == 0, name=f"active_empty_{i}")

    for k, (i, j) in enumerate(strong_edges):
        model.addConstr(z_td[i] + z_td[j] <= 1, name=f"strong_{k}")

    for k, (i, j, pen) in enumerate(weak_edges):
        model.addConstr(z_td[i] + z_td[j] <= 1 + w_td[k], name=f"weak_{k}")

    # Directed flow balance.
    for n in nodes:
        model.addConstr(
            gp.quicksum(x_td[a] for a in in_arcs.get(n, ()))
            - gp.quicksum(x_td[a] for a in out_arcs.get(n, ()))
            == 0,
            name=f"flow_{n}",
        )

    def support_components(counts):
        adj = defaultdict(list)
        for a, c in enumerate(counts):
            if c > 0:
                u, v, _ = arcs[a]
                adj[u].append(v)
                adj[v].append(u)
        comp = {}
        comps = []
        for s in list(adj.keys()):
            if s in comp:
                continue
            cid = len(comps)
            comp[s] = cid
            stack = [s]
            members = []
            while stack:
                u = stack.pop()
                members.append(u)
                for v in adj[u]:
                    if v not in comp:
                        comp[v] = cid
                        stack.append(v)
            comps.append(members)
        return comp, comps

    def find_connectivity_cuts(counts, served):
        comp, comps = support_components(counts)
        depot_cid = comp.get(depot)
        served_by_comp = defaultdict(list)
        for p, flag in enumerate(served):
            if not flag:
                continue
            a = rep_arc[p]
            if a is None or counts[a] <= 0:
                continue
            u, v, _ = profs[p]
            cu = comp.get(u)
            cv = comp.get(v)
            if cu is None or cu != cv or cu == depot_cid:
                continue
            served_by_comp[cu].append(p)

        cuts = []
        for cid, plist in served_by_comp.items():
            S = set(comps[cid])
            out_idx = []
            seen = set()
            for u in S:
                for a in out_arcs.get(u, ()):
                    if arcs[a][1] not in S and a not in seen:
                        seen.add(a)
                        out_idx.append(a)
            out_key = tuple(sorted(out_idx))
            for p in plist:
                key = (p, out_key)
                if key not in model._cut_keys:
                    model._cut_keys.add(key)
                    cuts.append((out_idx, p))
        return cuts

    def build_solution(counts, served):
        counts = [int(c) for c in counts]
        served = [bool(s) for s in served]

        # A served profitable arc must be represented by at least one traversal.
        for p in range(num_profs):
            if served[p]:
                a = rep_arc[p]
                if a is None or counts[a] <= 0:
                    served[p] = False

        comp, comps = support_components(counts)
        depot_cid = comp.get(depot)

        # Any served profitable arc must lie in the depot component.  Other
        # traversal-only components are useless for a nonnegative-cost tour and
        # are removed to project the incumbent onto one connected closed tour.
        for p, flag in enumerate(served):
            if not flag:
                continue
            u, v, _ = profs[p]
            cu = comp.get(u)
            cv = comp.get(v)
            if cu is None or cu != cv or cu != depot_cid:
                return None

        for cid, members in enumerate(comps):
            if cid == depot_cid:
                continue
            S = set(members)
            for u in S:
                for a in out_arcs.get(u, ()):
                    if arcs[a][1] in S:
                        counts[a] = 0

        active = set()
        profit_sum = 0
        served_list = []
        for p, flag in enumerate(served):
            if not flag:
                continue
            u, v, r = profs[p]
            profit_sum += r
            if u in graph_nodes:
                active.add(u)
            a = rep_arc[p]
            served_list.append({"from": u, "to": v, "profit": r, "cost": arcs[a][2]})

        penalty_sum = 0
        graph_node_set = set(graph_nodes)
        for i, j, pen in weak_edges:
            if i in active and j in active and i in graph_node_set and j in graph_node_set:
                penalty_sum += pen

        travel_sum = 0
        tour_list = []
        for a, c in enumerate(counts):
            if c > 0:
                u, v, cost = arcs[a]
                travel_sum += cost * c
                tour_list.append({"from": u, "to": v, "count": int(c), "cost": cost})

        obj = float(profit_sum - travel_sum - penalty_sum)
        sol = {
            "objective_value": obj,
            "served_arcs": served_list,
            "tour_arcs": tour_list,
        }
        return obj, sol

    model._x_list = x_vars
    model._y_list = y_vars
    model._cut_keys = set()
    model._best_obj = best_obj
    model._best_sol = best_sol

    tol = 1e-7

    def callback(m, where):
        if where != GRB.Callback.MIPSOL:
            return
        if time.time() - start_time >= time_limit:
            m.terminate()
            return
        try:
            xv = m.cbGetSolution(m._x_list)
            yv = m.cbGetSolution(m._y_list)
        except gp.GurobiError:
            return

        counts = [int(v + 0.5) for v in xv]
        served = [v > 0.5 for v in yv]

        cuts = find_connectivity_cuts(counts, served)
        if cuts:
            for out_idx, p in cuts:
                if out_idx:
                    m.cbLazy(gp.quicksum(m._x_list[a] for a in out_idx) >= m._y_list[p])
                else:
                    m.cbLazy(m._y_list[p] <= 0)
            return

        built = build_solution(counts, served)
        if built is None:
            return
        obj, sol = built
        if obj > m._best_obj + tol:
            m._best_obj = obj
            m._best_sol = sol
            if logger:
                logger.log_solution(obj, sol)

    remaining = time_limit - (time.time() - start_time)
    if remaining <= 0:
        write_solution(best_sol)
        return
    model.Params.TimeLimit = remaining

    try:
        model.optimize(callback)
    except gp.GurobiError:
        pass

    best_obj = getattr(model, "_best_obj", best_obj)
    best_sol = getattr(model, "_best_sol", best_sol)

    # Project the final solver incumbent as well, in case the callback did not
    # get a chance to log it before termination.
    try:
        if model.SolCount > 0:
            xv = [v.X for v in x_vars]
            yv = [v.X for v in y_vars]
            counts = [int(v + 0.5) for v in xv]
            served = [v > 0.5 for v in yv]
            built = build_solution(counts, served)
            if built is not None:
                obj, sol = built
                if obj > best_obj + tol:
                    best_obj = obj
                    best_sol = sol
                    if logger:
                        logger.log_solution(obj, sol)
    except gp.GurobiError:
        pass

    write_solution(best_sol)


if __name__ == "__main__":
    main()