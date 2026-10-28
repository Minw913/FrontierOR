import argparse
import json
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def bfs_path_arc_indices(
    node_ids: List[int],
    arcs: List[Dict[str, int]],
    source: int,
    target: int,
) -> Optional[List[int]]:
    """Find any feasible directed s-t path (by BFS on nodes), return arc indices."""
    out_adj = defaultdict(list)
    for idx, a in enumerate(arcs):
        out_adj[a["from_node"]].append((a["to_node"], idx))

    prev_node = {nid: None for nid in node_ids}
    prev_arc = {nid: None for nid in node_ids}
    q = deque([source])
    seen = {source}

    found = False
    while q:
        u = q.popleft()
        if u == target:
            found = True
            break
        for v, aidx in out_adj[u]:
            if v not in seen:
                seen.add(v)
                prev_node[v] = u
                prev_arc[v] = aidx
                q.append(v)

    if not found:
        return None

    path_arcs = []
    cur = target
    while cur != source:
        aidx = prev_arc[cur]
        if aidx is None:
            return None
        path_arcs.append(aidx)
        cur = prev_node[cur]
    path_arcs.reverse()
    return path_arcs


def reconstruct_path_order(
    selected_idx: List[int],
    arcs: List[Dict[str, int]],
    source: int,
    target: int,
) -> List[int]:
    """Try to order selected arcs along s->t. Fallback to sorted indices."""
    sel_set = set(selected_idx)
    out_arc = {}
    in_count = defaultdict(int)
    out_count = defaultdict(int)

    for i in selected_idx:
        a = arcs[i]
        u, v = a["from_node"], a["to_node"]
        out_count[u] += 1
        in_count[v] += 1
        out_arc[u] = i

    ordered = []
    cur = source
    used = set()
    max_steps = len(selected_idx) + 5

    for _ in range(max_steps):
        if cur == target:
            break
        if cur not in out_arc:
            ordered = []
            break
        i = out_arc[cur]
        if i in used:
            ordered = []
            break
        used.add(i)
        ordered.append(i)
        cur = arcs[i]["to_node"]

    if len(ordered) == len(selected_idx):
        return ordered

    # Fallback
    return sorted(selected_idx)


def compute_objective(
    selected_idx: List[int],
    linear_costs: List[float],
    quadratic_costs: List[List[float]],
) -> float:
    sel = selected_idx
    obj = 0.0
    for i in sel:
        obj += float(linear_costs[i])
    for i in sel:
        qi = quadratic_costs[i]
        for j in sel:
            obj += float(qi[j])
    return obj


def solve(instance: Dict[str, Any], time_limit: int, logger: Optional[SolutionLogger]):
    nodes = instance["nodes"]
    arcs = instance["arcs"]
    linear_costs = instance["linear_costs"]
    quadratic_costs = instance["quadratic_costs"]
    source = instance["source_node"]
    target = instance["target_node"]

    node_ids = [n["id"] for n in nodes]
    n = len(node_ids)
    m = len(arcs)

    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for i, a in enumerate(arcs):
        outgoing[a["from_node"]].append(i)
        incoming[a["to_node"]].append(i)

    model = gp.Model("quadratic_shortest_path")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.NonConvex = 2
    model.Params.MIPFocus = 1

    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    p = model.addVars(node_ids, lb=0.0, ub=max(0, n - 1), vtype=GRB.CONTINUOUS, name="p")

    # Flow/degree constraints
    for v in node_ids:
        in_expr = gp.quicksum(x[i] for i in incoming[v])
        out_expr = gp.quicksum(x[i] for i in outgoing[v])

        if v == source:
            model.addConstr(out_expr == 1, name=f"src_out_{v}")
            model.addConstr(in_expr == 0, name=f"src_in_{v}")
        elif v == target:
            model.addConstr(in_expr == 1, name=f"tgt_in_{v}")
            model.addConstr(out_expr == 0, name=f"tgt_out_{v}")
        else:
            model.addConstr(in_expr == out_expr, name=f"flow_{v}")
            model.addConstr(in_expr <= 1, name=f"deg_in_{v}")
            model.addConstr(out_expr <= 1, name=f"deg_out_{v}")

    # MTZ-like ordering constraints to eliminate directed cycles
    model.addConstr(p[source] == 0, name="order_source")
    big_m = n
    for i, a in enumerate(arcs):
        u = a["from_node"]
        v = a["to_node"]
        model.addConstr(
            p[v] >= p[u] + 1 - big_m * (1 - x[i]),
            name=f"order_{i}",
        )

    # Objective: linear + ordered-pair quadratic sum
    obj = gp.QuadExpr()
    for i in range(m):
        ci = linear_costs[i]
        if ci != 0:
            obj += float(ci) * x[i]

    # Since Q is symmetric and ordered-pair sum is required:
    # sum_i q_ii x_i + 2*sum_{i<j} q_ij x_i x_j
    for i in range(m):
        qi = quadratic_costs[i]
        qii = qi[i]
        if qii != 0:
            obj += float(qii) * x[i]
        xi = x[i]
        for j in range(i + 1, m):
            qij = qi[j]
            if qij != 0:
                obj += 2.0 * float(qij) * xi * x[j]

    model.setObjective(obj, GRB.MINIMIZE)

    # Heuristic warm start: any BFS path
    warm_path = bfs_path_arc_indices(node_ids, arcs, source, target)
    if warm_path is not None:
        warm_set = set(warm_path)
        for i in range(m):
            x[i].Start = 1.0 if i in warm_set else 0.0

        # Initialize order along warm path
        cur = source
        order_val = {nid: 0.0 for nid in node_ids}
        step = 0
        for aidx in warm_path:
            nxt = arcs[aidx]["to_node"]
            step += 1
            order_val[nxt] = float(step)
            cur = nxt
        for nid in node_ids:
            p[nid].Start = order_val[nid]

    best_inc = {"obj": float("inf")}

    def cb(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < best_inc["obj"]:
                best_inc["obj"] = obj_val
                if logger:
                    logger.log(float(obj_val))

    model.optimize(cb)

    selected_idx = []
    objective_value = float("inf")

    if model.SolCount > 0:
        for i in range(m):
            if x[i].X > 0.5:
                selected_idx.append(i)

        # Recompute objective from selected set (robust to tolerances)
        objective_value = compute_objective(selected_idx, linear_costs, quadratic_costs)

        if logger and objective_value + 1e-9 < best_inc["obj"]:
            logger.log(float(objective_value))
    else:
        # No feasible solution found
        selected_idx = []
        objective_value = 1e30

    ordered_idx = reconstruct_path_order(selected_idx, arcs, source, target)

    solution_arcs = [
        {
            "id": int(arcs[i]["id"]),
            "from_node": int(arcs[i]["from_node"]),
            "to_node": int(arcs[i]["to_node"]),
        }
        for i in ordered_idx
    ]

    return {
        "objective_value": float(objective_value),
        "solution_arcs": solution_arcs,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve directed quadratic-cost s-t path problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log path.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()