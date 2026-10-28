import argparse
import json
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate(selected_ids, lin, Q):
    """Exact objective: sum of linear costs + sum over ordered pairs of quadratic costs."""
    total = 0.0
    for a in selected_ids:
        total += lin[a]
    for a in selected_ids:
        Qa = Q[a]
        for b in selected_ids:
            total += Qa[b]
    return total


def bfs_path(num_nodes, arcs, source, target):
    """Find any simple path (fewest arcs) from source to target. Returns list of arc ids or None."""
    adj = {}
    for a in arcs:
        adj.setdefault(a["from_node"], []).append((a["to_node"], a["id"]))
    prev = {source: None}
    q = deque([source])
    while q:
        u = q.popleft()
        if u == target:
            break
        for v, aid in adj.get(u, []):
            if v not in prev:
                prev[v] = (u, aid)
                q.append(v)
    if target not in prev:
        return None
    path = []
    cur = target
    while prev[cur] is not None:
        u, aid = prev[cur]
        path.append(aid)
        cur = u
    path.reverse()
    return path


def order_path_arcs(selected_ids, arcs_by_id, source, target):
    """Order the selected arcs along the path from source to target."""
    out_map = {}
    for aid in selected_ids:
        a = arcs_by_id[aid]
        out_map[a["from_node"]] = aid
    ordered = []
    cur = source
    visited = set()
    while cur != target and cur in out_map:
        aid = out_map[cur]
        if aid in visited:
            break
        visited.add(aid)
        ordered.append(aid)
        cur = arcs_by_id[aid]["to_node"]
    # append any leftovers (shouldn't happen for a valid simple path)
    for aid in selected_ids:
        if aid not in visited:
            ordered.append(aid)
    return ordered


def build_solution_dict(selected_ids, arcs_by_id, lin, Q, source, target):
    obj = evaluate(selected_ids, lin, Q)
    ordered = order_path_arcs(selected_ids, arcs_by_id, source, target)
    sol_arcs = [
        {
            "id": int(aid),
            "from_node": int(arcs_by_id[aid]["from_node"]),
            "to_node": int(arcs_by_id[aid]["to_node"]),
        }
        for aid in ordered
    ]
    return {"objective_value": float(obj), "solution_arcs": sol_arcs}, obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = load_instance(args.instance_path)
    num_nodes = inst["num_nodes"]
    num_arcs = inst["num_arcs"]
    source = inst["source_node"]
    target = inst["target_node"]
    arcs = inst["arcs"]
    lin = inst["linear_costs"]
    Q = inst["quadratic_costs"]

    arcs_by_id = {a["id"]: a for a in arcs}
    node_ids = [n["id"] for n in inst["nodes"]]

    best_solution = None
    best_obj = float("inf")

    # ---------- Initial heuristic: BFS shortest-hop path ----------
    init_path = bfs_path(num_nodes, arcs, source, target)
    if init_path is not None:
        sol_dict, obj = build_solution_dict(init_path, arcs_by_id, lin, Q, source, target)
        best_solution, best_obj = sol_dict, obj
        if logger:
            logger.log_solution(obj, sol_dict)

    # ---------- Gurobi model ----------
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("qspp", env=env)
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        elapsed = time.time() - start_time
        remaining = max(1.0, args.time_limit - elapsed - 2.0)
        model.Params.TimeLimit = remaining

        x = model.addVars(num_arcs, vtype=GRB.BINARY, name="x")
        u = model.addVars(node_ids, lb=0.0, ub=num_nodes, vtype=GRB.CONTINUOUS, name="u")

        out_arcs = {nid: [] for nid in node_ids}
        in_arcs = {nid: [] for nid in node_ids}
        for a in arcs:
            out_arcs[a["from_node"]].append(a["id"])
            in_arcs[a["to_node"]].append(a["id"])

        # Source / sink constraints (simple path)
        model.addConstr(gp.quicksum(x[i] for i in out_arcs[source]) == 1)
        model.addConstr(gp.quicksum(x[i] for i in in_arcs[source]) == 0)
        model.addConstr(gp.quicksum(x[i] for i in in_arcs[target]) == 1)
        model.addConstr(gp.quicksum(x[i] for i in out_arcs[target]) == 0)

        # Flow conservation + degree limits at intermediate nodes
        for nid in node_ids:
            if nid == source or nid == target:
                continue
            model.addConstr(
                gp.quicksum(x[i] for i in in_arcs[nid])
                == gp.quicksum(x[i] for i in out_arcs[nid])
            )
            model.addConstr(gp.quicksum(x[i] for i in out_arcs[nid]) <= 1)

        # MTZ subtour elimination (prevents disconnected cycles)
        N = num_nodes
        model.addConstr(u[source] == 0)
        for a in arcs:
            i, j = a["from_node"], a["to_node"]
            model.addConstr(u[j] >= u[i] + 1 - N * (1 - x[a["id"]]))

        # Objective: c^T x + x^T Q x  (ordered pairs; Q symmetric)
        obj_expr = gp.QuadExpr()
        for i in range(num_arcs):
            coef = lin[i] + Q[i][i]  # diagonal (x_i^2 = x_i)
            if coef != 0:
                obj_expr.add(x[i], coef)
        for i in range(num_arcs):
            Qi = Q[i]
            for j in range(i + 1, num_arcs):
                q = Qi[j]
                if q != 0:
                    obj_expr.add(x[i] * x[j], 2.0 * q)
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # MIP start from heuristic
        if init_path is not None:
            init_set = set(init_path)
            for i in range(num_arcs):
                x[i].Start = 1.0 if i in init_set else 0.0

        state = {"best_obj": best_obj, "best_solution": best_solution}

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                vals = m.cbGetSolution([x[i] for i in range(num_arcs)])
                sel = [i for i in range(num_arcs) if vals[i] > 0.5]
                sol_dict, obj = build_solution_dict(sel, arcs_by_id, lin, Q, source, target)
                if obj < state["best_obj"] - 1e-9:
                    state["best_obj"] = obj
                    state["best_solution"] = sol_dict
                    if logger:
                        logger.log_solution(obj, sol_dict)

        model.optimize(callback)

        # Extract final solution if available
        if model.SolCount > 0:
            sel = [i for i in range(num_arcs) if x[i].X > 0.5]
            sol_dict, obj = build_solution_dict(sel, arcs_by_id, lin, Q, source, target)
            if obj < state["best_obj"] - 1e-9:
                state["best_obj"] = obj
                state["best_solution"] = sol_dict
                if logger:
                    logger.log_solution(obj, sol_dict)

        if state["best_solution"] is not None:
            best_solution = state["best_solution"]
            best_obj = state["best_obj"]

    except Exception:
        # Fall back to heuristic solution if solver fails
        pass

    if best_solution is None:
        best_solution = {"objective_value": 0.0, "solution_arcs": []}

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()