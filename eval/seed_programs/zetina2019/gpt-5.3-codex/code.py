import argparse
import json
import math
import heapq
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def arc_key(i, j):
    return f"{i}_{j}"


def dijkstra(num_nodes, out_arcs, weights, source, target):
    """
    Shortest path on directed graph.
    out_arcs[u] = list of (v, a_idx)
    weights[a_idx] >= 0
    Returns list of arc indices for a source->target path, or None if unreachable.
    """
    dist = [math.inf] * num_nodes
    prev_node = [-1] * num_nodes
    prev_arc = [-1] * num_nodes
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == target:
            break
        for v, a in out_arcs[u]:
            nd = d + weights[a]
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_arc[v] = a
                heapq.heappush(pq, (nd, v))

    if math.isinf(dist[target]):
        return None

    path_arcs = []
    cur = target
    while cur != source:
        a = prev_arc[cur]
        if a < 0:
            return None
        path_arcs.append(a)
        cur = prev_node[cur]
    path_arcs.reverse()
    return path_arcs


def build_initial_solution(data, arcs, out_arcs, fixed_cost, var_cost, commodities):
    """
    Build a feasible warm start by routing each commodity on one shortest path
    (variable cost weighted by demand + small fixed-cost share).
    Returns:
      y0[a] in {0,1}
      x0[(k,a)] in {0,1}
    or (None, None) if any commodity has no path in full graph.
    """
    num_arcs = len(arcs)
    num_commodities = len(commodities)

    y0 = [0] * num_arcs
    x0 = {}

    # Small fixed-charge share to bias path selection toward fewer expensive arcs
    fixed_share = 1.0 / max(1, num_commodities)

    for k, com in enumerate(commodities):
        o = int(com["origin"])
        d = int(com["destination"])
        dem = float(com["demand"])

        weights = [dem * float(var_cost[a][k]) + fixed_share * float(fixed_cost[a]) for a in range(num_arcs)]
        path = dijkstra(data["num_nodes"], out_arcs, weights, o, d)
        if path is None:
            return None, None

        for a in path:
            y0[a] = 1
            x0[(k, a)] = 1.0

    return y0, x0


def solve_instance(instance, time_limit, logger=None):
    num_nodes = int(instance["num_nodes"])
    arcs_raw = instance["arcs"]
    commodities = instance["commodities"]
    num_commodities = int(instance["num_commodities"])

    arcs = [(int(i), int(j)) for i, j in arcs_raw]
    num_arcs = len(arcs)

    # Costs
    fixed_cost = [0.0] * num_arcs
    var_cost = [None] * num_arcs  # each is list length K

    for a, (i, j) in enumerate(arcs):
        k = arc_key(i, j)
        fixed_cost[a] = float(instance["fixed_costs"][k])
        vc = instance["variable_costs"][k]
        if len(vc) != num_commodities:
            raise ValueError(f"variable_costs[{k}] length {len(vc)} != num_commodities {num_commodities}")
        var_cost[a] = [float(v) for v in vc]

    # Adjacency for flow constraints
    out_by_node = [[] for _ in range(num_nodes)]
    in_by_node = [[] for _ in range(num_nodes)]
    out_arcs_for_sp = [[] for _ in range(num_nodes)]  # (to_node, arc_idx)
    for a, (i, j) in enumerate(arcs):
        out_by_node[i].append(a)
        in_by_node[j].append(a)
        out_arcs_for_sp[i].append((j, a))

    model = gp.Model("fixed_charge_multicommodity_flow")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)
    model.Params.MIPFocus = 1  # feasible solutions focus

    # Variables
    y = model.addVars(num_arcs, vtype=GRB.BINARY, name="y")
    x = model.addVars(num_commodities, num_arcs, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Linking constraints: no flow on closed arcs
    model.addConstrs((x[k, a] <= y[a] for k in range(num_commodities) for a in range(num_arcs)), name="link")

    # Flow conservation for each commodity and node
    for k, com in enumerate(commodities):
        origin = int(com["origin"])
        dest = int(com["destination"])
        for n in range(num_nodes):
            rhs = -1.0 if n == origin else (1.0 if n == dest else 0.0)
            model.addConstr(
                gp.quicksum(x[k, a] for a in in_by_node[n]) - gp.quicksum(x[k, a] for a in out_by_node[n]) == rhs,
                name=f"flow_k{k}_n{n}",
            )

    # Objective
    obj_fixed = gp.quicksum(fixed_cost[a] * y[a] for a in range(num_arcs))
    obj_var = gp.quicksum(
        float(commodities[k]["demand"]) * var_cost[a][k] * x[k, a]
        for k in range(num_commodities)
        for a in range(num_arcs)
    )
    model.setObjective(obj_fixed + obj_var, GRB.MINIMIZE)

    # Warm start
    y0, x0 = build_initial_solution(instance, arcs, out_arcs_for_sp, fixed_cost, var_cost, commodities)
    if y0 is not None:
        for a in range(num_arcs):
            y[a].Start = y0[a]
        for k in range(num_commodities):
            for a in range(num_arcs):
                x[k, a].Start = x0.get((k, a), 0.0)

    best_logged = {"obj": math.inf}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged["obj"]:
                best_logged["obj"] = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(cb)

    # Build output
    result = {
        "objective_value": 1e30,
        "open_arcs": {},
        "routings": {str(k): {} for k in range(num_commodities)},
    }

    if model.SolCount > 0:
        objv = float(model.ObjVal)
        result["objective_value"] = objv

        # Ensure final incumbent logged
        if logger and objv + 1e-9 < best_logged["obj"]:
            logger.log(objv)

        for a, (i, j) in enumerate(arcs):
            if y[a].X > 0.5:
                result["open_arcs"][arc_key(i, j)] = 1

        eps = 1e-9
        for k in range(num_commodities):
            rk = {}
            for a, (i, j) in enumerate(arcs):
                val = x[k, a].X
                if val > eps:
                    rk[arc_key(i, j)] = float(val)
            result["routings"][str(k)] = rk

    return result


def main():
    parser = argparse.ArgumentParser(description="Solve fixed-charge multi-commodity network design problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to JSONL log file (optional).")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_instance(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()