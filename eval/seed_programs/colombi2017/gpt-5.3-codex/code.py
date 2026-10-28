import argparse
import json
import math
import sys
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def tarjan_scc(num_nodes, edges):
    sys.setrecursionlimit(10**6)
    graph = [[] for _ in range(num_nodes)]
    for u, v in edges:
        graph[u].append(v)

    index = 0
    stack = []
    onstack = [False] * num_nodes
    idx = [-1] * num_nodes
    low = [0] * num_nodes
    comp_id = [-1] * num_nodes
    comps = []

    def strongconnect(v):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        for w in graph[v]:
            if idx[w] == -1:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif onstack[w]:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp_id[w] = len(comps)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in range(num_nodes):
        if idx[v] == -1:
            strongconnect(v)

    return comp_id, comps


def build_and_solve(instance, time_limit, logger=None):
    n = instance["num_nodes"]
    depot = instance["depot"]
    arcs = instance["arcs"]  # [i, j, cost]
    prof_arcs = instance["profitable_arcs"]  # [i, j, profit]
    vi_nodes = instance["VI_nodes"]
    strong_inc = instance["strong_incompatibilities"]  # [i, j]
    weak_inc = instance["weak_incompatibilities"]  # [i, j, penalty]

    m = len(arcs)
    pnum = len(prof_arcs)
    wnum = len(weak_inc)

    tail = [a[0] for a in arcs]
    head = [a[1] for a in arcs]
    acost = [a[2] for a in arcs]

    ptail = [p[0] for p in prof_arcs]
    phead = [p[1] for p in prof_arcs]
    pprofit = [p[2] for p in prof_arcs]

    # Map endpoint -> arc indices (to connect profitable service to traversal)
    endpoint_to_arc_ids = defaultdict(list)
    for a in range(m):
        endpoint_to_arc_ids[(tail[a], head[a])].append(a)

    # cost lookup for profitable arcs in output
    prof_arc_cost = []
    for k in range(pnum):
        ids = endpoint_to_arc_ids.get((ptail[k], phead[k]), [])
        if ids:
            prof_arc_cost.append(min(acost[a] for a in ids))
        else:
            prof_arc_cost.append(0.0)

    out_arcs_by_node = [[] for _ in range(n)]
    in_arcs_by_node = [[] for _ in range(n)]
    for a in range(m):
        out_arcs_by_node[tail[a]].append(a)
        in_arcs_by_node[head[a]].append(a)

    # profitable arcs by tail node
    prof_by_tail = defaultdict(list)
    for k in range(pnum):
        prof_by_tail[ptail[k]].append(k)

    model = gp.Model("ProfitableArcTourWithIncompatibilities")
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.Threads = 1
    model.Params.LazyConstraints = 1

    # Decision variables
    x = model.addVars(m, vtype=GRB.INTEGER, lb=0, name="x")        # traversals on network arcs
    y = model.addVars(pnum, vtype=GRB.BINARY, name="y")            # served profitable arcs
    z = model.addVars(len(vi_nodes), vtype=GRB.BINARY, name="z")   # active VI nodes
    w = model.addVars(wnum, vtype=GRB.BINARY, name="w")            # weak override paid

    vi_index = {node: idx for idx, node in enumerate(vi_nodes)}

    # Flow balance at each node
    for v in range(n):
        model.addConstr(
            gp.quicksum(x[a] for a in in_arcs_by_node[v]) ==
            gp.quicksum(x[a] for a in out_arcs_by_node[v]),
            name=f"flow_bal_{v}"
        )

    # Serve implies traverse (at least once on corresponding directed endpoint pair)
    for k in range(pnum):
        ids = endpoint_to_arc_ids.get((ptail[k], phead[k]), [])
        if ids:
            model.addConstr(y[k] <= gp.quicksum(x[a] for a in ids), name=f"serve_link_{k}")
        else:
            # Inconsistent data safeguard: impossible to serve if underlying arc missing
            model.addConstr(y[k] == 0, name=f"serve_impossible_{k}")

    # Activation of VI nodes
    for node in vi_nodes:
        zid = vi_index[node]
        leaving_prof = prof_by_tail.get(node, [])
        if leaving_prof:
            for k in leaving_prof:
                model.addConstr(y[k] <= z[zid], name=f"act_up_{node}_{k}")
            model.addConstr(z[zid] <= gp.quicksum(y[k] for k in leaving_prof), name=f"act_low_{node}")
        else:
            model.addConstr(z[zid] == 0, name=f"act_zero_{node}")

    # Strong incompatibilities
    for e, (i, j) in enumerate(strong_inc):
        if i in vi_index and j in vi_index:
            model.addConstr(z[vi_index[i]] + z[vi_index[j]] <= 1, name=f"strong_{e}")

    # Weak incompatibilities with override penalty
    weak_penalty = []
    for e, (i, j, pen) in enumerate(weak_inc):
        weak_penalty.append(pen)
        if i in vi_index and j in vi_index:
            model.addConstr(z[vi_index[i]] + z[vi_index[j]] <= 1 + w[e], name=f"weak_{e}")
        else:
            # if either endpoint absent from VI set, no need to pay penalty
            model.addConstr(w[e] == 0, name=f"weak_zero_{e}")

    # Objective: maximize profit - travel costs - weak penalties
    model.setObjective(
        gp.quicksum(pprofit[k] * y[k] for k in range(pnum))
        - gp.quicksum(acost[a] * x[a] for a in range(m))
        - gp.quicksum(weak_penalty[e] * w[e] for e in range(wnum)),
        GRB.MAXIMIZE
    )

    # store for callback
    model._x = x
    model._y = y
    model._best_logged = -float("inf")
    model._logger = logger
    model._n = n
    model._depot = depot
    model._tail = tail
    model._head = head
    model._ptail = ptail
    model._phead = phead
    model._pnum = pnum
    model._m = m

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            # incumbent logging
            if mdl._logger is not None:
                obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > mdl._best_logged + 1e-9:
                    mdl._best_logged = obj
                    mdl._logger.log(float(obj))

            # connectivity separation (lazy cuts)
            xsol = mdl.cbGetSolution(mdl._x)
            ysol = mdl.cbGetSolution(mdl._y)

            selected_edges = []
            for a in range(mdl._m):
                if xsol[a] > 1e-6:
                    selected_edges.append((mdl._tail[a], mdl._head[a]))

            if not selected_edges:
                return

            comp_id, comps = tarjan_scc(mdl._n, selected_edges)

            # group served profitable arcs by SCC if both ends inside same SCC
            served_prof_by_comp = defaultdict(list)
            for k in range(mdl._pnum):
                if ysol[k] > 0.5:
                    ci = comp_id[mdl._ptail[k]]
                    cj = comp_id[mdl._phead[k]]
                    if ci == cj:
                        served_prof_by_comp[ci].append(k)

            # For each SCC not containing depot: if no outgoing traversed flow and it serves profitable arcs, cut it
            for cid, nodes in enumerate(comps):
                if mdl._depot in nodes:
                    continue
                if cid not in served_prof_by_comp:
                    continue

                out_val = 0.0
                out_arc_ids = []
                node_set = set(nodes)
                for a in range(mdl._m):
                    if mdl._tail[a] in node_set and mdl._head[a] not in node_set:
                        out_arc_ids.append(a)
                        out_val += xsol[a]

                if out_val < 0.5:
                    lhs = gp.quicksum(mdl._x[a] for a in out_arc_ids)
                    # add one cut per violated served profitable arc in this SCC
                    for k in served_prof_by_comp[cid]:
                        mdl.cbLazy(lhs >= mdl._y[k])

    model.optimize(callback)

    # extract incumbent/best known feasible solution
    if model.SolCount == 0:
        # fallback (should not happen because all-zero is feasible)
        return {
            "objective_value": 0.0,
            "served_arcs": [],
            "tour_arcs": []
        }

    x_sol = [int(round(x[a].X)) for a in range(m)]
    y_sol = [1 if y[k].X > 0.5 else 0 for k in range(pnum)]
    w_sol = [1 if w[e].X > 0.5 else 0 for e in range(wnum)]

    served_arcs = []
    for k in range(pnum):
        if y_sol[k] == 1:
            served_arcs.append({
                "from": int(ptail[k]),
                "to": int(phead[k]),
                "profit": float(pprofit[k]),
                "cost": float(prof_arc_cost[k]),
            })

    tour_arcs = []
    for a in range(m):
        if x_sol[a] > 0:
            tour_arcs.append({
                "from": int(tail[a]),
                "to": int(head[a]),
                "count": int(x_sol[a]),
                "cost": float(acost[a]),
            })

    total_profit = sum(pprofit[k] for k in range(pnum) if y_sol[k] == 1)
    total_travel = sum(acost[a] * x_sol[a] for a in range(m))
    total_pen = sum(weak_penalty[e] * w_sol[e] for e in range(wnum))
    objective_value = float(total_profit - total_travel - total_pen)

    return {
        "objective_value": objective_value,
        "served_arcs": served_arcs,
        "tour_arcs": tour_arcs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    sol = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()