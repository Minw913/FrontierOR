import argparse
import json
import time
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def bfs_reachable(start, adj):
    seen = set([start])
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def find_path_with_residual(origin, destination, demand, residual_caps, out_arcs, arcs):
    """
    BFS for any feasible path with residual capacity >= demand on each arc.
    Returns list of arc indices or None.
    """
    q = deque([origin])
    parent_node = {origin: None}
    parent_arc = {}

    while q:
        u = q.popleft()
        if u == destination:
            break
        for a_idx in out_arcs.get(u, []):
            if residual_caps[a_idx] < demand:
                continue
            v = arcs[a_idx]["to_node"]
            if v in parent_node:
                continue
            parent_node[v] = u
            parent_arc[v] = a_idx
            q.append(v)

    if destination not in parent_node:
        return None

    path = []
    cur = destination
    while cur != origin:
        a_idx = parent_arc[cur]
        path.append(a_idx)
        cur = parent_node[cur]
    path.reverse()
    return path


def extract_path_from_selected(selected_arc_idxs, origin, destination, arcs):
    """
    Extract one origin->destination path from selected arcs via BFS.
    Returns list of arc indices (empty if not found).
    """
    adj = defaultdict(list)
    for a_idx in selected_arc_idxs:
        u = arcs[a_idx]["from_node"]
        v = arcs[a_idx]["to_node"]
        adj[u].append((v, a_idx))

    q = deque([origin])
    parent_node = {origin: None}
    parent_arc = {}

    while q:
        u = q.popleft()
        if u == destination:
            break
        for v, a_idx in adj.get(u, []):
            if v in parent_node:
                continue
            parent_node[v] = u
            parent_arc[v] = a_idx
            q.append(v)

    if destination not in parent_node:
        return []

    path = []
    cur = destination
    while cur != origin:
        a_idx = parent_arc[cur]
        path.append(a_idx)
        cur = parent_node[cur]
    path.reverse()
    return path


class IncumbentCallback:
    def __init__(self, y_vars, penalties, logger):
        self.y_vars = y_vars
        self.penalties = penalties
        self.logger = logger
        self.best = float("inf")

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            val = 0.0
            for i, yv in enumerate(self.y_vars):
                if model.cbGetSolution(yv) > 0.5:
                    val += self.penalties[i]
            if val + 1e-9 < self.best:
                self.best = val
                self.logger.log(float(val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    network = inst["network"]
    comm_data = inst["commodities"]["commodity_list"]

    nodes = network["nodes"]
    arcs = network["arcs"]
    num_arcs = len(arcs)
    num_comms = len(comm_data)

    # Build adjacency for reachability and greedy BFS
    out_neighbors = defaultdict(list)
    rev_neighbors = defaultdict(list)
    out_arc_indices = defaultdict(list)
    for a_idx, a in enumerate(arcs):
        u = a["from_node"]
        v = a["to_node"]
        out_neighbors[u].append(v)
        rev_neighbors[v].append(u)
        out_arc_indices[u].append(a_idx)

    # Commodity penalties (reject costs)
    commodities = []
    for c in comm_data:
        penalty = c.get("artificial_arc_cost", c.get("revenue", 0))
        commodities.append({
            "commodity_id": c["commodity_id"],
            "origin": c["origin"],
            "destination": c["destination"],
            "demand": c["demand"],
            "penalty": penalty
        })

    # Reachability cache by origin / destination
    reach_from_origin = {}
    reach_to_dest = {}

    def get_reach_from(o):
        if o not in reach_from_origin:
            reach_from_origin[o] = bfs_reachable(o, out_neighbors)
        return reach_from_origin[o]

    def get_can_reach(d):
        if d not in reach_to_dest:
            reach_to_dest[d] = bfs_reachable(d, rev_neighbors)
        return reach_to_dest[d]

    # Allowed arcs per commodity (simple pruning)
    allowed_arcs_per_k = []
    relevant_nodes_per_k = []
    for k, c in enumerate(commodities):
        o = c["origin"]
        d = c["destination"]
        r1 = get_reach_from(o)
        r2 = get_can_reach(d)

        allowed = []
        rel_nodes = set([o, d])
        for a_idx, a in enumerate(arcs):
            if a["from_node"] in r1 and a["to_node"] in r2:
                allowed.append(a_idx)
                rel_nodes.add(a["from_node"])
                rel_nodes.add(a["to_node"])

        allowed_arcs_per_k.append(allowed)
        relevant_nodes_per_k.append(rel_nodes)

    # Build model
    model = gp.Model("unsplittable_mcf_reject")
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    elapsed = time.time() - t0
    remain = max(1, args.time_limit - int(elapsed))
    model.Params.TimeLimit = remain

    y_vars = []
    x_vars = {}  # (k, a_idx) -> var
    arc_to_kvars = defaultdict(list)

    # Variables
    for k, c in enumerate(commodities):
        y = model.addVar(vtype=GRB.BINARY, name=f"y_{c['commodity_id']}")
        y_vars.append(y)

        for a_idx in allowed_arcs_per_k[k]:
            xv = model.addVar(vtype=GRB.BINARY, name=f"x_{c['commodity_id']}_{a_idx}")
            x_vars[(k, a_idx)] = xv
            arc_to_kvars[a_idx].append((k, xv))

    model.update()

    # Flow + degree constraints
    for k, c in enumerate(commodities):
        o = c["origin"]
        d = c["destination"]
        y = y_vars[k]

        out_map = defaultdict(list)
        in_map = defaultdict(list)
        for a_idx in allowed_arcs_per_k[k]:
            xv = x_vars[(k, a_idx)]
            u = arcs[a_idx]["from_node"]
            v = arcs[a_idx]["to_node"]
            out_map[u].append(xv)
            in_map[v].append(xv)

        for n in relevant_nodes_per_k[k]:
            out_expr = gp.quicksum(out_map.get(n, []))
            in_expr = gp.quicksum(in_map.get(n, []))

            if n == o:
                model.addConstr(out_expr - in_expr == 1 - y, name=f"flow_o_{k}_{n}")
            elif n == d:
                model.addConstr(out_expr - in_expr == -1 + y, name=f"flow_d_{k}_{n}")
            else:
                model.addConstr(out_expr - in_expr == 0, name=f"flow_{k}_{n}")

            # Prevent branches/cycles and force zero flow when rejected
            model.addConstr(out_expr <= 1 - y, name=f"deg_out_{k}_{n}")
            model.addConstr(in_expr <= 1 - y, name=f"deg_in_{k}_{n}")

    # Capacity constraints
    for a_idx, a in enumerate(arcs):
        if a_idx not in arc_to_kvars:
            continue
        expr = gp.LinExpr()
        for k, xv in arc_to_kvars[a_idx]:
            expr += commodities[k]["demand"] * xv
        model.addConstr(expr <= a["capacity"], name=f"cap_{a_idx}")

    # Objective: lexicographic (primary reject penalty, secondary number of used arcs)
    penalties = [c["penalty"] for c in commodities]
    primary = gp.quicksum(penalties[k] * y_vars[k] for k in range(num_comms))
    secondary = gp.quicksum(x_vars[key] for key in x_vars.keys())
    big_m = len(x_vars) + 1
    model.setObjective(primary * big_m + secondary, GRB.MINIMIZE)

    # Greedy warm start
    residual = [a["capacity"] for a in arcs]
    sorted_k = sorted(range(num_comms), key=lambda i: penalties[i], reverse=True)

    greedy_reject = [1] * num_comms
    greedy_paths = {}

    for k in sorted_k:
        c = commodities[k]
        path = find_path_with_residual(
            c["origin"], c["destination"], c["demand"], residual, out_arc_indices, arcs
        )
        if path is not None:
            feasible = True
            for a_idx in path:
                if residual[a_idx] < c["demand"]:
                    feasible = False
                    break
            if feasible:
                greedy_reject[k] = 0
                greedy_paths[k] = path
                for a_idx in path:
                    residual[a_idx] -= c["demand"]

    for k in range(num_comms):
        y_vars[k].Start = float(greedy_reject[k])
        for a_idx in allowed_arcs_per_k[k]:
            x_vars[(k, a_idx)].Start = 0.0
        if greedy_reject[k] == 0:
            for a_idx in greedy_paths.get(k, []):
                if (k, a_idx) in x_vars:
                    x_vars[(k, a_idx)].Start = 1.0

    greedy_obj = float(sum(penalties[k] * greedy_reject[k] for k in range(num_comms)))
    if logger:
        logger.log(greedy_obj)

    # Optimize with incumbent logging
    if logger:
        cb = IncumbentCallback(y_vars, penalties, logger)
        model.optimize(cb)
    else:
        model.optimize()

    # Extract solution
    # Fallback (should rarely be needed): greedy
    has_sol = model.SolCount > 0

    if has_sol:
        y_sol = [1 if y_vars[k].X > 0.5 else 0 for k in range(num_comms)]
    else:
        y_sol = greedy_reject[:]

    objective_value = float(sum(penalties[k] * y_sol[k] for k in range(num_comms)))

    comm_out = []
    for k, c in enumerate(commodities):
        rejected = bool(y_sol[k] == 1)
        path_arcs = []

        if not rejected:
            if has_sol:
                selected = [a_idx for a_idx in allowed_arcs_per_k[k] if x_vars[(k, a_idx)].X > 0.5]
            else:
                selected = greedy_paths.get(k, [])

            path_idx = extract_path_from_selected(selected, c["origin"], c["destination"], arcs)

            # If path extraction failed, fall back to selected order empty-safe
            for a_idx in path_idx:
                a = arcs[a_idx]
                path_arcs.append({
                    "from": a["from_node"],
                    "to": a["to_node"],
                    "arc_id": a["arc_id"]
                })

        comm_out.append({
            "commodity_id": c["commodity_id"],
            "rejected": rejected,
            "path_arcs": path_arcs
        })

    sol = {
        "objective_value": objective_value,
        "commodities": comm_out
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()