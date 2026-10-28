import argparse
import json
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def find_path(origin, dest, selected_arcs, arc_info):
    """BFS to find a simple path from origin to dest using selected arc ids.
    Returns list of arc dicts {'from','to','arc_id'} or None."""
    if origin == dest:
        return []
    adj = {}
    for aid in selected_arcs:
        fn, tn = arc_info[aid][0], arc_info[aid][1]
        adj.setdefault(fn, []).append((tn, aid))
    # BFS
    parent = {origin: None}
    q = deque([origin])
    while q:
        u = q.popleft()
        if u == dest:
            break
        for (v, aid) in adj.get(u, []):
            if v not in parent:
                parent[v] = (u, aid)
                q.append(v)
    if dest not in parent:
        return None
    path = []
    node = dest
    while parent[node] is not None:
        u, aid = parent[node]
        path.append({"from": u, "to": node, "arc_id": aid})
        node = u
    path.reverse()
    return path


def build_solution(get_x, get_r, commodities, arc_info, arc_cost):
    """Build solution dict from value accessors. Returns (obj, sol_dict)."""
    entries = []
    total_cost = 0.0
    for c in commodities:
        k = c["commodity_id"]
        origin, dest = c["origin"], c["destination"]
        rej = get_r(k) > 0.5
        path = None
        if not rej:
            selected = [aid for aid in arc_info if get_x(k, aid) > 0.5]
            path = find_path(origin, dest, selected, arc_info)
            if path is None:
                rej = True
        if rej:
            total_cost += c["artificial_arc_cost"]
            entries.append({"commodity_id": k, "rejected": True, "path_arcs": []})
        else:
            for p in path:
                total_cost += c["demand"] * arc_cost[p["arc_id"]]
            entries.append({"commodity_id": k, "rejected": False, "path_arcs": path})
    return total_cost, {"objective_value": total_cost, "commodities": entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    arcs = data["network"]["arcs"]
    commodities = data["commodities"]["commodity_list"]

    arc_info = {}   # arc_id -> (from, to, capacity)
    arc_cost = {}   # arc_id -> per-unit cost
    out_arcs = {}   # node -> list arc_ids
    in_arcs = {}    # node -> list arc_ids
    for a in arcs:
        aid = a["arc_id"]
        arc_info[aid] = (a["from_node"], a["to_node"], a["capacity"])
        arc_cost[aid] = a.get("cost", 0)
        out_arcs.setdefault(a["from_node"], []).append(aid)
        in_arcs.setdefault(a["to_node"], []).append(aid)

    nodes = data["network"]["nodes"]

    # Initial trivial solution: reject everything
    best_obj, best_sol = build_solution(
        lambda k, a: 0.0, lambda k: 1.0, commodities, arc_info, arc_cost
    )
    if logger:
        logger.log_solution(best_obj, best_sol)

    # Build MIP
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("umcf", env=env)
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    K = [c["commodity_id"] for c in commodities]
    cmap = {c["commodity_id"]: c for c in commodities}
    A = list(arc_info.keys())

    # variables
    x = {}
    for k in K:
        dem = cmap[k]["demand"]
        for aid in A:
            # skip arcs whose capacity is below demand (can never carry it)
            if arc_info[aid][2] < dem:
                continue
            x[(k, aid)] = m.addVar(vtype=GRB.BINARY, name=f"x_{k}_{aid}")
    r = {k: m.addVar(vtype=GRB.BINARY, name=f"r_{k}") for k in K}

    # objective
    obj = gp.LinExpr()
    for k in K:
        c = cmap[k]
        obj += c["artificial_arc_cost"] * r[k]
        dem = c["demand"]
        for aid in A:
            if (k, aid) in x and arc_cost[aid] != 0:
                obj += dem * arc_cost[aid] * x[(k, aid)]
    m.setObjective(obj, GRB.MINIMIZE)

    # flow conservation
    for k in K:
        c = cmap[k]
        origin, dest = c["origin"], c["destination"]
        if origin == dest:
            # trivially routed; forbid using arcs, set r free (min cost -> 0)
            for aid in A:
                if (k, aid) in x:
                    x[(k, aid)].ub = 0
            continue
        for n in nodes:
            expr = gp.LinExpr()
            has_term = False
            for aid in out_arcs.get(n, []):
                if (k, aid) in x:
                    expr += x[(k, aid)]
                    has_term = True
            for aid in in_arcs.get(n, []):
                if (k, aid) in x:
                    expr -= x[(k, aid)]
                    has_term = True
            if n == origin:
                expr += r[k]
                m.addConstr(expr == 1)
            elif n == dest:
                expr -= r[k]
                m.addConstr(expr == -1)
            elif has_term:
                m.addConstr(expr == 0)

    # capacity constraints
    for aid in A:
        cap = arc_info[aid][2]
        expr = gp.LinExpr()
        has = False
        for k in K:
            if (k, aid) in x:
                expr += cmap[k]["demand"] * x[(k, aid)]
                has = True
        if has:
            m.addConstr(expr <= cap)

    # MIP start: reject all
    for k in K:
        r[k].Start = 1.0
    for key in x:
        x[key].Start = 0.0

    # time limit
    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    m.Params.TimeLimit = remaining

    state = {"best_obj": best_obj, "best_sol": best_sol}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xvals = model.cbGetSolution([x[key] for key in x])
                rvals = model.cbGetSolution([r[k] for k in K])
                xd = dict(zip(x.keys(), xvals))
                rd = dict(zip(K, rvals))
                obj_val, sol = build_solution(
                    lambda k, a: xd.get((k, a), 0.0),
                    lambda k: rd[k],
                    commodities, arc_info, arc_cost,
                )
                if obj_val < state["best_obj"] - 1e-9:
                    state["best_obj"] = obj_val
                    state["best_sol"] = sol
                    if logger:
                        logger.log_solution(obj_val, sol)
            except Exception:
                pass

    try:
        m.optimize(callback)
    except Exception:
        pass

    # extract final solution if available
    try:
        if m.SolCount > 0:
            xd = {key: x[key].X for key in x}
            rd = {k: r[k].X for k in K}
            obj_val, sol = build_solution(
                lambda k, a: xd.get((k, a), 0.0),
                lambda k: rd[k],
                commodities, arc_info, arc_cost,
            )
            if obj_val < state["best_obj"] - 1e-9:
                state["best_obj"] = obj_val
                state["best_sol"] = sol
                if logger:
                    logger.log_solution(obj_val, sol)
    except Exception:
        pass

    with open(args.solution_path, "w") as f:
        json.dump(state["best_sol"], f, indent=2)


if __name__ == "__main__":
    main()