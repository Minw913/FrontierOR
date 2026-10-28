import argparse
import json
import math
import time
import heapq
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


class DSU:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}
        self.components = len(self.parent)

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True


def dijkstra_to_tree(source, selected_nodes, adj, edge_w, node_type, node_w):
    """Shortest path from source to any node in selected_nodes.
    Path cost includes edge weights + turbine node weights for entered unselected turbine nodes.
    """
    INF = float("inf")
    dist = {source: 0.0}
    prev_node = {}
    prev_edge = {}
    heap = [(0.0, source)]
    target = None

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, INF) + 1e-12:
            continue
        if u in selected_nodes and u != source:
            target = u
            break

        for v, eid in adj[u]:
            c = edge_w[eid]
            if node_type[v] == "potential_turbine" and v not in selected_nodes:
                c += node_w[v]
            nd = d + c
            if nd + 1e-12 < dist.get(v, INF):
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                heapq.heappush(heap, (nd, v))

    if target is None:
        return None

    path_edges_rev = []
    path_nodes_rev = [target]
    cur = target
    while cur != source:
        e = prev_edge[cur]
        path_edges_rev.append(e)
        cur = prev_node[cur]
        path_nodes_rev.append(cur)

    path_edges = list(reversed(path_edges_rev))
    path_nodes = list(reversed(path_nodes_rev))

    inc_cost = dist[target]
    if node_type[source] == "potential_turbine" and source not in selected_nodes:
        inc_cost += node_w[source]

    return {
        "target": target,
        "path_nodes": path_nodes,
        "path_edges": path_edges,
        "inc_cost": inc_cost,
    }


def build_heuristic_solution(
    node_ids,
    substations,
    turbines,
    profits,
    quota,
    adj,
    edge_w,
    node_type,
    node_w,
    edge_endpoints,
    heuristic_time_limit,
):
    start_t = time.time()

    if not substations:
        return None

    root = substations[0]
    selected_nodes = set([root])
    selected_edges = set()
    selected_turbines = set()
    current_profit = 0.0

    # Connect all substations first
    for s in substations[1:]:
        if time.time() - start_t > heuristic_time_limit:
            break
        sp = dijkstra_to_tree(s, selected_nodes, adj, edge_w, node_type, node_w)
        if sp is None:
            return None
        new_nodes = [n for n in sp["path_nodes"] if n not in selected_nodes]
        selected_nodes.update(sp["path_nodes"])
        selected_edges.update(sp["path_edges"])
        for n in new_nodes:
            if node_type[n] == "potential_turbine":
                selected_turbines.add(n)
                current_profit += profits[n]

    # Add turbines until quota met
    while current_profit + 1e-9 < quota:
        if time.time() - start_t > heuristic_time_limit:
            break

        best = None
        for t in turbines:
            if t in selected_nodes:
                continue
            sp = dijkstra_to_tree(t, selected_nodes, adj, edge_w, node_type, node_w)
            if sp is None:
                continue

            new_nodes = [n for n in sp["path_nodes"] if n not in selected_nodes]
            gain = sum(profits[n] for n in new_nodes if node_type[n] == "potential_turbine")
            if gain <= 1e-12:
                continue
            score = sp["inc_cost"] / gain

            cand = (score, sp, gain, new_nodes)
            if best is None or cand[0] < best[0]:
                best = cand

        if best is None:
            break

        _, sp, gain, new_nodes = best
        selected_nodes.update(sp["path_nodes"])
        selected_edges.update(sp["path_edges"])
        for n in new_nodes:
            if node_type[n] == "potential_turbine":
                selected_turbines.add(n)
                current_profit += profits[n]

    if current_profit + 1e-9 < quota:
        return None

    # Prune cycles -> MST on selected nodes using edges whose endpoints are both selected
    cand_edges = []
    for eid, (u, v) in edge_endpoints.items():
        if u in selected_nodes and v in selected_nodes:
            cand_edges.append(eid)
    cand_edges.sort(key=lambda e: edge_w[e])

    dsu = DSU(selected_nodes)
    tree_edges = set()
    for eid in cand_edges:
        u, v = edge_endpoints[eid]
        if dsu.union(u, v):
            tree_edges.add(eid)
            if len(tree_edges) == len(selected_nodes) - 1:
                break

    if len(tree_edges) != len(selected_nodes) - 1:
        return None

    # Recompute selected turbines/profit after pruning keeps same nodes by construction
    selected_turbines = {n for n in selected_nodes if node_type[n] == "potential_turbine"}
    current_profit = sum(profits[n] for n in selected_turbines)
    if current_profit + 1e-9 < quota:
        return None

    return {
        "selected_nodes": selected_nodes,
        "selected_edges": tree_edges,
        "selected_turbines": selected_turbines,
    }


def compute_objective(selected_turbines, selected_edges, node_w, edge_w):
    return float(sum(node_w[t] for t in selected_turbines) + sum(edge_w[e] for e in selected_edges))


def incumbent_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if obj + 1e-9 < model._best_logged:
            model._best_logged = obj
            if model._logger is not None:
                model._logger.log(float(obj))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    t0 = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    alpha = float(inst["parameters"]["alpha"])
    quota = float(inst["parameters"]["quota"])

    nodes_data = inst["nodes"]
    edges_data = inst["edges"]

    node_ids = [n["id"] for n in nodes_data]
    node_type = {n["id"]: n["type"] for n in nodes_data}
    node_cost = {n["id"]: float(n["cost"]) for n in nodes_data}
    node_profit = {n["id"]: float(n["profit"]) for n in nodes_data}
    node_scenic = {n["id"]: float(n["scenic_impact"]) for n in nodes_data}

    substations = [nid for nid in node_ids if node_type[nid] == "substation"]
    turbines = [nid for nid in node_ids if node_type[nid] == "potential_turbine"]
    steiners = [nid for nid in node_ids if node_type[nid] == "steiner"]

    edge_ids = [e["id"] for e in edges_data]
    edge_endpoints = {e["id"]: (e["from"], e["to"]) for e in edges_data}
    edge_cost = {e["id"]: float(e["cost"]) for e in edges_data}
    edge_scenic = {e["id"]: float(e["scenic_impact"]) for e in edges_data}

    # Weighted objective coefficients
    node_w = {nid: alpha * node_cost[nid] + (1.0 - alpha) * node_scenic[nid] for nid in node_ids}
    edge_w = {eid: alpha * edge_cost[eid] + (1.0 - alpha) * edge_scenic[eid] for eid in edge_ids}

    # Quick infeasibility check on quota
    max_profit = sum(node_profit[t] for t in turbines)
    if quota > max_profit + 1e-9:
        out = {
            "objective_value": 1e30,
            "selected_turbines": [],
            "selected_edges": [],
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    # Build adjacency
    adj = defaultdict(list)
    incident_edges = defaultdict(list)
    for eid in edge_ids:
        u, v = edge_endpoints[eid]
        adj[u].append((v, eid))
        adj[v].append((u, eid))
        incident_edges[u].append(eid)
        incident_edges[v].append(eid)

    # Heuristic start
    heuristic_sol = None
    heuristic_obj = float("inf")
    heuristic_budget = max(1.0, min(5.0, 0.2 * args.time_limit))
    heuristic_sol = build_heuristic_solution(
        node_ids=node_ids,
        substations=substations,
        turbines=turbines,
        profits=node_profit,
        quota=quota,
        adj=adj,
        edge_w=edge_w,
        node_type=node_type,
        node_w=node_w,
        edge_endpoints=edge_endpoints,
        heuristic_time_limit=heuristic_budget,
    )

    if heuristic_sol is not None:
        heuristic_obj = compute_objective(
            heuristic_sol["selected_turbines"], heuristic_sol["selected_edges"], node_w, edge_w
        )
        if logger:
            logger.log(float(heuristic_obj))

    elapsed = time.time() - t0
    remaining = max(1, int(args.time_limit - elapsed))

    # Build MIP
    model = gp.Model("wind_farm_tree")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = remaining
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    # Variables
    z = {}
    for nid in node_ids:
        if node_type[nid] == "substation":
            z[nid] = model.addVar(vtype=GRB.BINARY, lb=1.0, ub=1.0, name=f"z_{nid}")
        else:
            z[nid] = model.addVar(vtype=GRB.BINARY, name=f"z_{nid}")

    x = {eid: model.addVar(vtype=GRB.BINARY, name=f"x_{eid}") for eid in edge_ids}

    # Directed flow vars for connectivity
    # one flow var per directed arc induced by undirected edges
    f = {}
    M = len(node_ids)
    for eid in edge_ids:
        u, v = edge_endpoints[eid]
        f[(u, v, eid)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"f_{u}_{v}_{eid}")
        f[(v, u, eid)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"f_{v}_{u}_{eid}")

    model.update()

    # Root substation
    if not substations:
        # Invalid instance per description, but handle safely
        out = {"objective_value": 1e30, "selected_turbines": [], "selected_edges": []}
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)
        return
    root = substations[0]

    # Objective
    obj = gp.quicksum(edge_w[eid] * x[eid] for eid in edge_ids) + gp.quicksum(
        node_w[nid] * z[nid] for nid in turbines
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Edge-node linking
    for eid in edge_ids:
        u, v = edge_endpoints[eid]
        model.addConstr(x[eid] <= z[u], name=f"link_u_{eid}")
        model.addConstr(x[eid] <= z[v], name=f"link_v_{eid}")

    # Degree lower bound for selected non-root nodes
    for nid in node_ids:
        if nid == root:
            continue
        model.addConstr(
            gp.quicksum(x[eid] for eid in incident_edges[nid]) >= z[nid],
            name=f"deg_lb_{nid}",
        )

    # Tree edge count
    model.addConstr(
        gp.quicksum(x[eid] for eid in edge_ids) == gp.quicksum(z[nid] for nid in node_ids) - 1,
        name="tree_edge_count",
    )

    # Profit quota
    model.addConstr(
        gp.quicksum(node_profit[t] * z[t] for t in turbines) >= quota,
        name="quota",
    )

    # Flow capacity on arcs
    for eid in edge_ids:
        u, v = edge_endpoints[eid]
        model.addConstr(f[(u, v, eid)] <= M * x[eid], name=f"cap1_{eid}")
        model.addConstr(f[(v, u, eid)] <= M * x[eid], name=f"cap2_{eid}")

    # Flow conservation
    non_root_nodes = [nid for nid in node_ids if nid != root]

    for nid in non_root_nodes:
        inflow = []
        outflow = []
        for nbr, eid in adj[nid]:
            inflow.append(f[(nbr, nid, eid)])
            outflow.append(f[(nid, nbr, eid)])
        model.addConstr(gp.quicksum(inflow) - gp.quicksum(outflow) == z[nid], name=f"flow_{nid}")

    root_in = []
    root_out = []
    for nbr, eid in adj[root]:
        root_in.append(f[(nbr, root, eid)])
        root_out.append(f[(root, nbr, eid)])
    model.addConstr(
        gp.quicksum(root_out) - gp.quicksum(root_in) == gp.quicksum(z[nid] for nid in non_root_nodes),
        name="flow_root",
    )

    # MIP start from heuristic
    if heuristic_sol is not None:
        sel_nodes = heuristic_sol["selected_nodes"]
        sel_edges = heuristic_sol["selected_edges"]
        for nid in node_ids:
            z[nid].Start = 1.0 if nid in sel_nodes else 0.0
        for eid in edge_ids:
            x[eid].Start = 1.0 if eid in sel_edges else 0.0

    # Logging setup
    model._logger = logger
    model._best_logged = heuristic_obj if heuristic_sol is not None else float("inf")

    model.optimize(incumbent_callback)

    # Extract best available solution
    best_selected_turbines = []
    best_selected_edge_ids = set()
    best_obj = float("inf")

    if model.SolCount > 0:
        z_val = {nid: z[nid].X for nid in node_ids}
        x_val = {eid: x[eid].X for eid in edge_ids}

        sel_nodes = {nid for nid in node_ids if z_val[nid] > 0.5}
        sel_edges = {eid for eid in edge_ids if x_val[eid] > 0.5}
        sel_turbines = [t for t in turbines if z_val[t] > 0.5]

        obj_val = compute_objective(sel_turbines, sel_edges, node_w, edge_w)

        best_selected_turbines = sel_turbines
        best_selected_edge_ids = sel_edges
        best_obj = obj_val

    if heuristic_sol is not None and heuristic_obj + 1e-9 < best_obj:
        best_selected_turbines = sorted(list(heuristic_sol["selected_turbines"]))
        best_selected_edge_ids = set(heuristic_sol["selected_edges"])
        best_obj = heuristic_obj
    else:
        best_selected_turbines = sorted(best_selected_turbines)

    # If still no feasible solution
    if not math.isfinite(best_obj):
        out = {
            "objective_value": 1e30,
            "selected_turbines": [],
            "selected_edges": [],
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    # Build output edges list
    selected_edges_out = []
    for eid in sorted(best_selected_edge_ids):
        u, v = edge_endpoints[eid]
        selected_edges_out.append({"from": int(u), "to": int(v)})

    out = {
        "objective_value": float(best_obj),
        "selected_turbines": [int(t) for t in best_selected_turbines],
        "selected_edges": selected_edges_out,
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()