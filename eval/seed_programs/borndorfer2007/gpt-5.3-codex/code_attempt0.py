import argparse
import json
import math
import time
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_mappings(inst):
    nodes = inst["network"]["nodes"]
    edges = inst["network"]["edges"]

    node_ids = [n["id"] for n in nodes]
    node_set = set(node_ids)

    edge_id_to_idx = {}
    edge_idx_to_id = {}
    for i, e in enumerate(edges):
        eid = e.get("id", i)
        edge_id_to_idx[eid] = i
        edge_idx_to_id[i] = eid

    return node_ids, node_set, edge_id_to_idx, edge_idx_to_id


def normalize_mode_edge_indices(mode_edge_indices, edge_id_to_idx, num_edges):
    out = []
    for x in mode_edge_indices:
        if isinstance(x, int) and 0 <= x < num_edges:
            out.append(x)
        elif x in edge_id_to_idx:
            out.append(edge_id_to_idx[x])
    return sorted(set(out))


def reconstruct_path_from_pred(pred, s, t, h):
    # pred key: (h, node) -> (prev_node, edge_idx)
    nodes = [t]
    edges = []
    cur = t
    cur_h = h
    while cur_h > 0:
        key = (cur_h, cur)
        if key not in pred:
            return None, None
        prev, eidx = pred[key]
        edges.append(eidx)
        nodes.append(prev)
        cur = prev
        cur_h -= 1
    nodes.reverse()
    edges.reverse()
    if nodes[0] != s or nodes[-1] != t:
        return None, None
    return nodes, edges


def is_simple_node_path(nodes):
    return len(nodes) == len(set(nodes))


def bfs_min_hop_path(adj, s, t, max_hops):
    q = deque([s])
    parent = {s: (None, None)}
    depth = {s: 0}
    while q:
        u = q.popleft()
        if u == t:
            break
        if depth[u] >= max_hops:
            continue
        for v, eidx, _tt in adj[u]:
            if v not in parent:
                parent[v] = (u, eidx)
                depth[v] = depth[u] + 1
                q.append(v)
    if t not in parent:
        return None, None
    if depth[t] > max_hops:
        return None, None
    nodes = []
    edges = []
    cur = t
    while cur is not None:
        nodes.append(cur)
        p, e = parent[cur]
        if e is not None:
            edges.append(e)
        cur = p
    nodes.reverse()
    edges.reverse()
    return nodes, edges


def constrained_shortest_paths_from_source(adj, source, max_hops):
    # DP by exact number of hops
    # dist[h][node] = min travel time using exactly h hops
    inf = float("inf")
    dist = [defaultdict(lambda: inf) for _ in range(max_hops + 1)]
    pred = {}
    dist[0][source] = 0.0

    for h in range(1, max_hops + 1):
        for u, du in dist[h - 1].items():
            for v, eidx, tt in adj[u]:
                nd = du + tt
                if nd < dist[h][v]:
                    dist[h][v] = nd
                    pred[(h, v)] = (u, eidx)
    return dist, pred


def generate_candidate_lines(inst, edge_id_to_idx):
    gp_params = inst.get("global_parameters", {})
    L = int(gp_params.get("max_line_length_edges", 6))
    if L <= 0:
        L = 1

    edges = inst["network"]["edges"]
    num_edges = len(edges)

    candidates = []
    seen = set()  # dedup by mode + canonical edge sequence

    for mode in inst["modes"]:
        mode_name = mode["name"]
        terminals = list(dict.fromkeys(mode.get("terminals", [])))
        terminals = [t for t in terminals if t is not None]
        if len(terminals) < 2:
            continue

        mode_edge_indices = normalize_mode_edge_indices(
            mode.get("edge_indices", []), edge_id_to_idx, num_edges
        )
        if not mode_edge_indices:
            continue

        # Build undirected adjacency for mode
        adj = defaultdict(list)
        for eidx in mode_edge_indices:
            e = edges[eidx]
            u, v = e["endpoints"]
            tt = float(e.get("traveling_time_seconds", 0.0))
            adj[u].append((v, eidx, tt))
            adj[v].append((u, eidx, tt))

        terminal_set = set(terminals)
        terminals_sorted = sorted(terminals)

        # For each source terminal, add constrained shortest-time path + min-hop path
        for s in terminals_sorted:
            if s not in adj:
                continue
            dist, pred = constrained_shortest_paths_from_source(adj, s, L)
            for t in terminals_sorted:
                if t <= s:
                    continue
                # Time-optimal within hop limit
                best_h = None
                best_d = float("inf")
                for h in range(1, L + 1):
                    d = dist[h].get(t, float("inf"))
                    if d < best_d:
                        best_d = d
                        best_h = h
                if best_h is not None and best_d < float("inf"):
                    npath, epath = reconstruct_path_from_pred(pred, s, t, best_h)
                    if npath and is_simple_node_path(npath) and len(epath) <= L:
                        seq = tuple(epath)
                        rseq = tuple(reversed(epath))
                        key = (mode_name, min(seq, rseq))
                        if key not in seen:
                            seen.add(key)
                            candidates.append(
                                {
                                    "mode": mode_name,
                                    "nodes": npath,
                                    "edges_idx": epath,
                                    "terminals": (s, t),
                                }
                            )

                # Min-hop path (alternative structure)
                npath2, epath2 = bfs_min_hop_path(adj, s, t, L)
                if npath2 and is_simple_node_path(npath2) and len(epath2) <= L:
                    seq2 = tuple(epath2)
                    rseq2 = tuple(reversed(epath2))
                    key2 = (mode_name, min(seq2, rseq2))
                    if key2 not in seen:
                        seen.add(key2)
                        candidates.append(
                            {
                                "mode": mode_name,
                                "nodes": npath2,
                                "edges_idx": epath2,
                                "terminals": (s, t),
                            }
                        )

    # Attach costs/capacities
    mode_map = {m["name"]: m for m in inst["modes"]}
    gp_fixed = float(inst.get("global_parameters", {}).get("fixed_cost_per_line", 0.0))
    gp_op = float(inst.get("global_parameters", {}).get("operating_cost_per_edge", 0.0))

    edges = inst["network"]["edges"]

    for i, c in enumerate(candidates):
        mode = mode_map[c["mode"]]
        fixed = float(mode.get("fixed_cost_per_line", gp_fixed))
        veh_cap = float(mode.get("vehicle_capacity", 0))
        op_sum = 0.0
        mode_op_default = float(mode.get("operating_cost_per_edge", gp_op))
        for eidx in c["edges_idx"]:
            e = edges[eidx]
            op_sum += float(e.get("operating_cost", mode_op_default))
        c["line_index"] = i
        c["fixed_cost"] = fixed
        c["veh_cap"] = veh_cap
        c["op_cost"] = op_sum

    return candidates


def decompose_flow_to_paths(origin, destination, arc_flow, out_arcs_by_node, arc_to_uv, eps=1e-8):
    # arc_flow: dict arc_id -> positive flow
    residual = {a: f for a, f in arc_flow.items() if f > eps}
    paths = []

    while True:
        # BFS for any s->t path in residual graph
        q = deque([origin])
        parent_node = {origin: None}
        parent_arc = {}
        found = False
        while q and not found:
            u = q.popleft()
            for a in out_arcs_by_node.get(u, []):
                f = residual.get(a, 0.0)
                if f <= eps:
                    continue
                _uu, vv, _eidx = arc_to_uv[a]
                if vv not in parent_node:
                    parent_node[vv] = u
                    parent_arc[vv] = a
                    if vv == destination:
                        found = True
                        break
                    q.append(vv)

        if destination not in parent_node:
            break

        # Reconstruct path
        arcs = []
        cur = destination
        while cur != origin:
            a = parent_arc[cur]
            arcs.append(a)
            cur = parent_node[cur]
        arcs.reverse()

        bottleneck = min(residual[a] for a in arcs)
        if bottleneck <= eps:
            break

        paths.append((arcs, bottleneck))
        for a in arcs:
            residual[a] -= bottleneck
            if residual[a] <= eps:
                residual.pop(a, None)

    return paths


def solve_instance(inst, time_limit, logger=None):
    t0 = time.time()

    lam = float(inst["global_parameters"]["lambda"])
    F = float(inst["global_parameters"]["frequency_upper_bound_F"])

    node_ids, node_set, edge_id_to_idx, edge_idx_to_id = build_mappings(inst)
    edges = inst["network"]["edges"]
    num_edges = len(edges)

    # Build directed arcs from undirected edges: 2*i (u->v), 2*i+1 (v->u)
    arc_to_uv = {}
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    arc_time = {}
    for i, e in enumerate(edges):
        u, v = e["endpoints"]
        tt = float(e.get("traveling_time_seconds", 0.0))
        a_fwd = 2 * i
        a_bwd = 2 * i + 1
        arc_to_uv[a_fwd] = (u, v, i)
        arc_to_uv[a_bwd] = (v, u, i)
        out_arcs[u].append(a_fwd)
        in_arcs[v].append(a_fwd)
        out_arcs[v].append(a_bwd)
        in_arcs[u].append(a_bwd)
        arc_time[a_fwd] = tt
        arc_time[a_bwd] = tt
    num_arcs = 2 * num_edges

    # OD pairs
    od_list = []
    for row in inst.get("od_matrix", []):
        d = float(row["demand"])
        if d > 0:
            od_list.append((row["origin"], row["destination"], d))
    K = len(od_list)

    # Candidate lines
    candidates = generate_candidate_lines(inst, edge_id_to_idx)

    # If no candidates, create empty infeasible-like fallback later
    if not candidates:
        return {
            "objective_value": float("inf"),
            "active_lines": [],
            "active_passenger_paths": [],
        }

    Lc = len(candidates)

    # Edge -> lines incidence list
    lines_using_edge = [[] for _ in range(num_edges)]
    for l, c in enumerate(candidates):
        for eidx in c["edges_idx"]:
            lines_using_edge[eidx].append(l)

    # Build model
    m = gp.Model("multimodal_line_planning")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    remaining = max(1, int(time_limit - (time.time() - t0)))
    m.Params.TimeLimit = remaining
    m.Params.OutputFlag = 0

    y = m.addVars(Lc, vtype=GRB.BINARY, name="y")
    f = m.addVars(Lc, lb=0.0, ub=F, vtype=GRB.CONTINUOUS, name="freq")

    # Passenger arc flow vars x[k,a]
    x = {}
    for k in range(K):
        x[k] = m.addVars(num_arcs, lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{k}")

    # Link f and y
    for l in range(Lc):
        m.addConstr(f[l] <= F * y[l], name=f"link_{l}")

    # Flow conservation
    for k, (o, d, dem) in enumerate(od_list):
        for n in node_ids:
            rhs = dem if n == o else (-dem if n == d else 0.0)
            expr = gp.quicksum(x[k][a] for a in out_arcs.get(n, [])) - gp.quicksum(
                x[k][a] for a in in_arcs.get(n, [])
            )
            m.addConstr(expr == rhs, name=f"flow_k{k}_n{n}")

    # Edge frequency caps and arc passenger caps
    for eidx, e in enumerate(edges):
        cap_freq = float(e.get("edge_capacity", 0.0))
        lset = lines_using_edge[eidx]

        freq_expr = gp.quicksum(f[l] for l in lset)
        m.addConstr(freq_expr <= cap_freq, name=f"edge_freq_{eidx}")

        cap_pass_expr = gp.quicksum(candidates[l]["veh_cap"] * f[l] for l in lset)

        a1, a2 = 2 * eidx, 2 * eidx + 1
        m.addConstr(
            gp.quicksum(x[k][a1] for k in range(K)) <= cap_pass_expr,
            name=f"arc_cap_{a1}",
        )
        m.addConstr(
            gp.quicksum(x[k][a2] for k in range(K)) <= cap_pass_expr,
            name=f"arc_cap_{a2}",
        )

    # Objective
    line_cost = gp.quicksum(candidates[l]["fixed_cost"] * y[l] + candidates[l]["op_cost"] * f[l] for l in range(Lc))
    pass_time = gp.quicksum(arc_time[a] * x[k][a] for k in range(K) for a in range(num_arcs))

    m.setObjective(lam * line_cost + (1.0 - lam) * pass_time, GRB.MINIMIZE)

    best_inc = [float("inf")]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_inc[0]:
                best_inc[0] = obj
                logger.log(float(obj))

    m.optimize(cb)

    if m.SolCount == 0:
        return {
            "objective_value": float("inf"),
            "active_lines": [],
            "active_passenger_paths": [],
        }

    obj_val = float(m.ObjVal)

    # Extract active lines
    active_lines = []
    fvals = m.getAttr("X", f)
    for l, c in enumerate(candidates):
        freq = float(fvals[l])
        if freq > 1e-7:
            active_lines.append(
                {
                    "line_index": int(c["line_index"]),
                    "mode": c["mode"],
                    "nodes": [int(v) for v in c["nodes"]],
                    "edges": [int(edge_idx_to_id[eidx]) for eidx in c["edges_idx"]],
                    "frequency": freq,
                }
            )

    # Extract passenger paths by flow decomposition
    active_passenger_paths = []
    for k, (o, d, dem) in enumerate(od_list):
        # Build arc flow dict for this OD
        arc_flow = {}
        for a in range(num_arcs):
            val = float(x[k][a].X)
            if val > 1e-8:
                arc_flow[a] = val

        paths = decompose_flow_to_paths(o, d, arc_flow, out_arcs, arc_to_uv, eps=1e-9)

        # If tiny residual due numerical issues, ignore under tolerance
        sent = sum(fl for _arcs, fl in paths)
        missing = dem - sent
        if missing > 1e-5:
            # numerical fallback: push missing on a positive-flow-reachable path if possible
            # otherwise skip (should be rare)
            pass

        for arcs, fl in paths:
            if fl > 1e-8:
                active_passenger_paths.append(
                    {
                        "origin": int(o),
                        "destination": int(d),
                        "arcs": [int(a) for a in arcs],
                        "flow": float(fl),
                    }
                )

    sol = {
        "objective_value": obj_val,
        "active_lines": active_lines,
        "active_passenger_paths": active_passenger_paths,
    }
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = load_instance(args.instance_path)
    sol = solve_instance(inst, args.time_limit, logger=logger)

    if logger:
        logger.log_solution(float(sol["objective_value"]), sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()