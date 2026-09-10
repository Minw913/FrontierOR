import argparse
import json
import heapq
import time
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    params = data["parameters"]
    alpha = float(params["alpha"])
    quota = float(params["quota"])

    nodes = data["nodes"]
    edges_raw = data["edges"]

    node_ids = [n["id"] for n in nodes]
    ntype = {n["id"]: n["type"] for n in nodes}
    node_profit = {}
    node_w = {}
    for n in nodes:
        nid = n["id"]
        if n["type"] == "potential_turbine":
            node_profit[nid] = float(n.get("profit", 0.0))
            node_w[nid] = alpha * float(n.get("cost", 0.0)) + (1 - alpha) * float(n.get("scenic_impact", 0.0))
        else:
            node_profit[nid] = 0.0
            node_w[nid] = 0.0

    # Filter self loops
    edges = []
    for e in edges_raw:
        if e["from"] == e["to"]:
            continue
        edges.append(e)

    E = len(edges)
    edge_w = [alpha * float(e.get("cost", 0.0)) + (1 - alpha) * float(e.get("scenic_impact", 0.0)) for e in edges]
    eu = [e["from"] for e in edges]
    ev = [e["to"] for e in edges]

    adj = defaultdict(list)
    for i in range(E):
        adj[eu[i]].append((ev[i], i))
        adj[ev[i]].append((eu[i], i))

    substations = [n["id"] for n in nodes if n["type"] == "substation"]
    turbines = [n["id"] for n in nodes if n["type"] == "potential_turbine"]
    N = len(nodes)

    def compute_obj(sel_turbines, sel_edge_idx):
        return sum(node_w[t] for t in sel_turbines) + sum(edge_w[i] for i in sel_edge_idx)

    def make_solution(sel_turbines, sel_edge_idx):
        obj = compute_obj(sel_turbines, sel_edge_idx)
        return obj, {
            "objective_value": obj,
            "selected_turbines": sorted(sel_turbines),
            "selected_edges": [{"from": eu[i], "to": ev[i]} for i in sorted(sel_edge_idx)],
        }

    # ---------------- Greedy heuristic ----------------
    def dijkstra(in_tree):
        dist = {s: 0.0 for s in in_tree}
        parent = {}
        pq = [(0.0, s) for s in in_tree]
        heapq.heapify(pq)
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for (v, ei) in adj[u]:
                if v in in_tree:
                    continue
                nd = d + edge_w[ei] + node_w[v]
                if nd < dist.get(v, float("inf")) - 1e-15:
                    dist[v] = nd
                    parent[v] = (u, ei)
                    heapq.heappush(pq, (nd, v))
        return dist, parent

    def attach(target, parent, in_tree, tree_edges):
        path_nodes = []
        path_edges = []
        v = target
        while v not in in_tree:
            path_nodes.append(v)
            u, ei = parent[v]
            path_edges.append(ei)
            v = u
        for n in path_nodes:
            in_tree.add(n)
        tree_edges.extend(path_edges)
        return path_nodes

    heuristic_sol = None
    heur_in_tree = None
    heur_edges = None
    if substations:
        in_tree = {substations[0]}
        tree_edges = []
        remaining = set(substations[1:])
        ok = True
        while remaining:
            dist, parent = dijkstra(in_tree)
            best = None
            bd = float("inf")
            for s in remaining:
                d = dist.get(s, float("inf"))
                if d < bd:
                    bd = d
                    best = s
            if best is None:
                ok = False
                break
            added = attach(best, parent, in_tree, tree_edges)
            for n in added:
                remaining.discard(n)
        if ok:
            cur_profit = sum(node_profit[n] for n in in_tree)
            feasible = True
            while cur_profit < quota - 1e-9:
                dist, parent = dijkstra(in_tree)
                need = quota - cur_profit
                best = None
                br = float("inf")
                for t in turbines:
                    if t in in_tree or node_profit[t] <= 0:
                        continue
                    d = dist.get(t)
                    if d is None:
                        continue
                    r = d / min(node_profit[t], need)
                    if r < br:
                        br = r
                        best = t
                if best is None:
                    feasible = False
                    break
                added = attach(best, parent, in_tree, tree_edges)
                cur_profit += sum(node_profit[n] for n in added)
            if feasible:
                sel_t = [n for n in in_tree if ntype[n] == "potential_turbine"]
                obj, sol = make_solution(sel_t, tree_edges)
                heuristic_sol = (obj, sol)
                heur_in_tree = set(in_tree)
                heur_edges = set(tree_edges)
                if logger:
                    logger.log_solution(obj, sol)

    # ---------------- MIP ----------------
    best_sol = heuristic_sol

    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("windfarm")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        remain = args.time_limit - (time.time() - t_start) - 2.0
        model.Params.TimeLimit = max(1.0, remain)

        y = {}
        for nid in node_ids:
            if ntype[nid] == "substation":
                y[nid] = model.addVar(lb=1.0, ub=1.0, vtype=GRB.BINARY, name=f"y_{nid}")
            else:
                y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")

        x = model.addVars(E, vtype=GRB.BINARY, name="x")
        f_uv = model.addVars(E, lb=0.0, ub=N - 1, name="fuv")
        f_vu = model.addVars(E, lb=0.0, ub=N - 1, name="fvu")

        for i in range(E):
            model.addConstr(x[i] <= y[eu[i]])
            model.addConstr(x[i] <= y[ev[i]])
            model.addConstr(f_uv[i] <= (N - 1) * x[i])
            model.addConstr(f_vu[i] <= (N - 1) * x[i])

        # tree: |edges| = |nodes| - 1
        model.addConstr(gp.quicksum(x[i] for i in range(E)) ==
                        gp.quicksum(y[nid] for nid in node_ids) - 1)

        root = substations[0] if substations else node_ids[0]
        if not substations:
            model.addConstr(y[root] == 1)

        inflow = defaultdict(list)
        outflow = defaultdict(list)
        for i in range(E):
            # flow u->v
            outflow[eu[i]].append(f_uv[i])
            inflow[ev[i]].append(f_uv[i])
            # flow v->u
            outflow[ev[i]].append(f_vu[i])
            inflow[eu[i]].append(f_vu[i])

        for nid in node_ids:
            if nid == root:
                continue
            model.addConstr(gp.quicksum(inflow[nid]) - gp.quicksum(outflow[nid]) == y[nid])

        # quota
        model.addConstr(gp.quicksum(node_profit[t] * y[t] for t in turbines) >= quota)

        model.setObjective(
            gp.quicksum(edge_w[i] * x[i] for i in range(E)) +
            gp.quicksum(node_w[t] * y[t] for t in turbines),
            GRB.MINIMIZE,
        )

        # warm start
        if heuristic_sol is not None:
            for nid in node_ids:
                y[nid].Start = 1.0 if nid in heur_in_tree else (1.0 if ntype[nid] == "substation" else 0.0)
            for i in range(E):
                x[i].Start = 1.0 if i in heur_edges else 0.0

        xlist = [x[i] for i in range(E)]
        ylist = [y[nid] for nid in node_ids]
        best_logged = [heuristic_sol[0] if heuristic_sol else float("inf")]

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < best_logged[0] - 1e-9:
                    xv = m.cbGetSolution(xlist)
                    yv = m.cbGetSolution(ylist)
                    sel_e = [i for i in range(E) if xv[i] > 0.5]
                    sel_t = [node_ids[j] for j in range(len(node_ids))
                             if yv[j] > 0.5 and ntype[node_ids[j]] == "potential_turbine"]
                    o2, sol = make_solution(sel_t, sel_e)
                    best_logged[0] = obj
                    if logger:
                        logger.log_solution(o2, sol)

        model.optimize(cb)

        if model.SolCount > 0:
            sel_e = [i for i in range(E) if x[i].X > 0.5]
            sel_t = [nid for nid in node_ids
                     if y[nid].X > 0.5 and ntype[nid] == "potential_turbine"]
            obj, sol = make_solution(sel_t, sel_e)
            if best_sol is None or obj < best_sol[0] - 1e-12:
                best_sol = (obj, sol)
                if logger and obj < best_logged[0] - 1e-9:
                    logger.log_solution(obj, sol)
            elif best_sol is not None and obj <= best_sol[0] + 1e-9:
                best_sol = (obj, sol)
    except Exception as ex:
        sys.stderr.write(f"MIP solve failed: {ex}\n")

    if best_sol is None:
        # fallback: connect substations only (may be infeasible w.r.t. quota)
        sel_e = []
        if substations:
            in_tree = {substations[0]}
            tree_edges = []
            remaining = set(substations[1:])
            while remaining:
                dist, parent = dijkstra(in_tree)
                best = None
                bd = float("inf")
                for s in remaining:
                    d = dist.get(s, float("inf"))
                    if d < bd:
                        bd = d
                        best = s
                if best is None:
                    break
                added = attach(best, parent, in_tree, tree_edges)
                for n in added:
                    remaining.discard(n)
            sel_e = tree_edges
            sel_t = [n for n in in_tree if ntype[n] == "potential_turbine"]
        else:
            sel_t = []
        obj, sol = make_solution(sel_t, sel_e)
        best_sol = (obj, sol)
        if logger:
            logger.log_solution(obj, sol)

    with open(args.solution_path, "w") as f:
        json.dump(best_sol[1], f, indent=2)


if __name__ == "__main__":
    main()