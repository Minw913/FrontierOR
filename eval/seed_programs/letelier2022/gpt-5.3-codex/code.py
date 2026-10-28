import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_l_value(L_raw):
    if isinstance(L_raw, str):
        if L_raw.lower() == "infinity":
            return None
        return int(L_raw)
    if L_raw is None:
        return None
    return int(L_raw)


def compute_time_windows(all_item_ids, real_items, source_id, sink_id, arcs, T):
    # Longest-path style relaxations for lower/upper bounds in difference constraints:
    # p[v] >= p[u] + lag
    NEG_INF = -10**15

    dist_from_source = {i: NEG_INF for i in all_item_ids}
    dist_from_source[source_id] = 0

    for _ in range(len(all_item_ids) - 1):
        improved = False
        for a in arcs:
            u, v, l = a["from"], a["to"], int(a["lag"])
            if dist_from_source[u] > NEG_INF:
                cand = dist_from_source[u] + l
                if cand > dist_from_source[v]:
                    dist_from_source[v] = cand
                    improved = True
        if not improved:
            break

    dist_to_sink = {i: NEG_INF for i in all_item_ids}
    dist_to_sink[sink_id] = 0

    for _ in range(len(all_item_ids) - 1):
        improved = False
        for a in arcs:
            u, v, l = a["from"], a["to"], int(a["lag"])
            if dist_to_sink[v] > NEG_INF:
                cand = l + dist_to_sink[v]
                if cand > dist_to_sink[u]:
                    dist_to_sink[u] = cand
                    improved = True
        if not improved:
            break

    windows = {}
    for i in real_items:
        es = 1
        ls = T
        if dist_from_source[i] > NEG_INF:
            es = max(1, int(math.ceil(dist_from_source[i])))
        if dist_to_sink[i] > NEG_INF:
            ls = min(T, int(math.floor((T + 1) - dist_to_sink[i])))

        if es > ls:
            es, ls = 1, T
        windows[i] = (es, ls)
    return windows


def first_fit_decreasing(items, weight, W):
    bins = []  # each: {"load": int, "items": [id]}
    for i in sorted(items, key=lambda x: weight[x], reverse=True):
        w = weight[i]
        placed = False
        for b in bins:
            if b["load"] + w <= W:
                b["items"].append(i)
                b["load"] += w
                placed = True
                break
        if not placed:
            bins.append({"load": w, "items": [i]})
    return bins


def pack_period(items, weight, W, L_limit=None, time_limit=1.0):
    if not items:
        return {}, 0

    heuristic_bins = first_fit_decreasing(items, weight, W)
    heuristic_count = len(heuristic_bins)

    if time_limit <= 0.15 or len(items) <= 1:
        if L_limit is not None and heuristic_count > L_limit:
            return None, None
        assign = {}
        for b_idx, b in enumerate(heuristic_bins, start=1):
            for i in b["items"]:
                assign[i] = b_idx
        return assign, heuristic_count

    if L_limit is None:
        B = heuristic_count
    else:
        B = L_limit

    m = gp.Model("pack_period")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.05, time_limit)

    idx_items = list(items)
    I = range(len(idx_items))
    item_at = {k: idx_items[k] for k in I}
    Bset = range(1, B + 1)

    a = m.addVars(I, Bset, vtype=GRB.BINARY, name="a")
    u = m.addVars(Bset, vtype=GRB.BINARY, name="u")

    for k in I:
        m.addConstr(gp.quicksum(a[k, b] for b in Bset) == 1)

    for b in Bset:
        m.addConstr(
            gp.quicksum(weight[item_at[k]] * a[k, b] for k in I) <= W * u[b]
        )

    for b in range(1, B):
        m.addConstr(u[b] >= u[b + 1])

    m.setObjective(gp.quicksum(u[b] for b in Bset), GRB.MINIMIZE)

    # Warm start from heuristic if it fits in B
    if heuristic_count <= B:
        for b_idx, b in enumerate(heuristic_bins, start=1):
            for i in b["items"]:
                k = idx_items.index(i)
                a[k, b_idx].Start = 1
            u[b_idx].Start = 1

    m.optimize()

    if m.SolCount == 0:
        if heuristic_count <= B:
            assign = {}
            for b_idx, b in enumerate(heuristic_bins, start=1):
                for i in b["items"]:
                    assign[i] = b_idx
            return assign, heuristic_count
        return None, None

    assign = {}
    used = set()
    for k in I:
        for b in Bset:
            if a[k, b].X > 0.5:
                i = item_at[k]
                assign[i] = b
                used.add(b)
                break

    # Compact bin ids
    used_sorted = sorted(used)
    remap = {old: new for new, old in enumerate(used_sorted, start=1)}
    assign = {i: remap[b] for i, b in assign.items()}
    return assign, len(used_sorted)


def build_solution_json(assignments_map, weight):
    # assignments_map: item -> (bin, period)
    active = defaultdict(list)
    for i, (b, t) in assignments_map.items():
        active[(b, t)].append(i)

    assignments = []
    for i, (b, t) in sorted(assignments_map.items()):
        assignments.append(
            {"item_id": int(i), "bin": int(b), "period": int(t), "weight": int(weight[i])}
        )

    active_list = []
    for (b, t), items in sorted(active.items(), key=lambda x: (x[0][1], x[0][0])):
        tw = sum(weight[i] for i in items)
        active_list.append(
            {
                "bin": int(b),
                "period": int(t),
                "items": [int(i) for i in sorted(items)],
                "total_weight": int(tw),
            }
        )

    num_active = len(active_list)
    return {
        "objective_value": float(num_active),
        "assignments": assignments,
        "active_bin_periods": active_list,
        "num_active_bin_periods": int(num_active),
    }


def solve_exact_bin_time(
    real_items, weight, arcs, T, L, source_id, sink_id, windows, time_limit, logger, best_logged
):
    m = gp.Model("exact_bin_time")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.1, time_limit)
    m.Params.MIPFocus = 1

    B = L
    Bset = list(range(1, B + 1))

    x = {}
    period_expr = {}

    for i in real_items:
        es, ls = windows[i]
        terms = []
        for t in range(es, ls + 1):
            for b in Bset:
                x[i, t, b] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}_{b}")
                terms.append(t * x[i, t, b])
        period_expr[i] = gp.quicksum(terms)

    y = m.addVars(range(1, T + 1), Bset, vtype=GRB.BINARY, name="y")

    # Assignment
    for i in real_items:
        es, ls = windows[i]
        m.addConstr(
            gp.quicksum(x[i, t, b] for t in range(es, ls + 1) for b in Bset) == 1
        )

    # Capacity + activation
    for t in range(1, T + 1):
        for b in Bset:
            m.addConstr(
                gp.quicksum(
                    weight[i] * x[i, t, b]
                    for i in real_items
                    if windows[i][0] <= t <= windows[i][1]
                )
                <= 1.0 * m.getParamInfo("IntFeasTol")[2] + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0
            )
            m.remove(m.getConstrs()[-1])
            m.addConstr(
                gp.quicksum(
                    weight[i] * x[i, t, b]
                    for i in real_items
                    if windows[i][0] <= t <= windows[i][1]
                )
                <= W_GLOBAL * y[t, b]
            )

    # Symmetry (bin ordering per period)
    for t in range(1, T + 1):
        for b in range(1, B):
            m.addConstr(y[t, b] >= y[t, b + 1])

    # Precedence arcs
    fixed = {source_id: 0, sink_id: T + 1}

    for a in arcs:
        u, v, lag = a["from"], a["to"], int(a["lag"])

        lhs_u = fixed[u] if u in fixed else period_expr[u]
        lhs_v = fixed[v] if v in fixed else period_expr[v]
        m.addConstr(lhs_v - lhs_u >= lag)

    m.setObjective(gp.quicksum(y[t, b] for t in range(1, T + 1) for b in Bset), GRB.MINIMIZE)

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged["val"]:
                best_logged["val"] = float(obj)
                logger.log(float(obj))

    m.optimize(cb)

    if m.SolCount == 0:
        return None

    assign = {}
    for i in real_items:
        found = False
        es, ls = windows[i]
        for t in range(es, ls + 1):
            for b in Bset:
                var = x.get((i, t, b))
                if var is not None and var.X > 0.5:
                    assign[i] = (b, t)
                    found = True
                    break
            if found:
                break
        if not found:
            # fallback from period expression (rare numeric issue)
            t_val = int(round(period_expr[i].getValue()))
            t_val = max(1, min(T, t_val))
            assign[i] = (1, t_val)

    sol = build_solution_json(assign, weight)
    if logger is not None and sol["objective_value"] + 1e-9 < best_logged["val"]:
        best_logged["val"] = sol["objective_value"]
        logger.log(sol["objective_value"])
    return sol


def solve_aggregate_then_pack(
    real_items, weight, arcs, T, source_id, sink_id, windows, L, time_limit, logger, best_logged, start_time, total_time
):
    m = gp.Model("aggregate_period")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.1, time_limit)
    m.Params.MIPFocus = 1

    z = {}
    for i in real_items:
        es, ls = windows[i]
        for t in range(es, ls + 1):
            z[i, t] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}")

    ub_k = {}
    for t in range(1, T + 1):
        cnt = sum(1 for i in real_items if windows[i][0] <= t <= windows[i][1])
        ub_k[t] = cnt if L is None else min(L, cnt)

    k = m.addVars(range(1, T + 1), vtype=GRB.INTEGER, lb=0, name="k")
    for t in range(1, T + 1):
        m.addConstr(k[t] <= ub_k[t])

    for i in real_items:
        es, ls = windows[i]
        m.addConstr(gp.quicksum(z[i, t] for t in range(es, ls + 1)) == 1)

    # Aggregate capacity
    for t in range(1, T + 1):
        m.addConstr(
            gp.quicksum(
                weight[i] * z[i, t]
                for i in real_items
                if windows[i][0] <= t <= windows[i][1]
            )
            <= W_GLOBAL * k[t]
        )

    fixed = {source_id: 0, sink_id: T + 1}
    period_expr = {}
    for i in real_items:
        es, ls = windows[i]
        period_expr[i] = gp.quicksum(t * z[i, t] for t in range(es, ls + 1))

    for a in arcs:
        u, v, lag = a["from"], a["to"], int(a["lag"])
        lhs_u = fixed[u] if u in fixed else period_expr[u]
        lhs_v = fixed[v] if v in fixed else period_expr[v]
        m.addConstr(lhs_v - lhs_u >= lag)

    m.setObjective(gp.quicksum(k[t] for t in range(1, T + 1)), GRB.MINIMIZE)

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged["val"]:
                best_logged["val"] = float(obj)
                logger.log(float(obj))

    m.optimize(cb)
    if m.SolCount == 0:
        return None

    item_period = {}
    for i in real_items:
        es, ls = windows[i]
        chosen = None
        for t in range(es, ls + 1):
            if z[i, t].X > 0.5:
                chosen = t
                break
        if chosen is None:
            chosen = int(round(period_expr[i].getValue()))
            chosen = max(1, min(T, chosen))
        item_period[i] = chosen

    period_items = defaultdict(list)
    for i, t in item_period.items():
        period_items[t].append(i)

    assign = {}
    periods_nonempty = sorted([t for t in range(1, T + 1) if period_items[t]])
    for idx, t in enumerate(periods_nonempty):
        now = time.time()
        rem = max(0.0, total_time - (now - start_time))
        left = len(periods_nonempty) - idx
        per_t_budget = max(0.05, rem / max(1, left))

        p_assign, _ = pack_period(
            period_items[t], weight, W_GLOBAL, L_limit=L, time_limit=per_t_budget
        )
        if p_assign is None:
            # fallback heuristic (may fail only if finite-L and impossible)
            bins = first_fit_decreasing(period_items[t], weight, W_GLOBAL)
            if L is not None and len(bins) > L:
                return None
            p_assign = {}
            for b_idx, b in enumerate(bins, start=1):
                for i in b["items"]:
                    p_assign[i] = b_idx

        for i, b in p_assign.items():
            assign[i] = (b, t)

    sol = build_solution_json(assign, weight)
    if logger is not None and sol["objective_value"] + 1e-9 < best_logged["val"]:
        best_logged["val"] = sol["objective_value"]
        logger.log(sol["objective_value"])
    return sol


def solve_fallback_feasibility(real_items, arcs, T, source_id, sink_id, windows, time_limit):
    m = gp.Model("fallback_period_feas")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.05, time_limit)

    p = {}
    for i in real_items:
        es, ls = windows[i]
        p[i] = m.addVar(vtype=GRB.INTEGER, lb=es, ub=ls, name=f"p_{i}")

    fixed = {source_id: 0, sink_id: T + 1}
    for a in arcs:
        u, v, lag = a["from"], a["to"], int(a["lag"])
        lhs_u = fixed[u] if u in fixed else p[u]
        lhs_v = fixed[v] if v in fixed else p[v]
        m.addConstr(lhs_v - lhs_u >= lag)

    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return None
    return {i: int(round(p[i].X)) for i in real_items}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    start_time = time.time()
    total_time = int(args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_logged = {"val": float("inf")}

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    params = inst.get("parameters", {})
    T = int(params["T"])
    global W_GLOBAL
    W_GLOBAL = int(params["W"])
    L = parse_l_value(params.get("L", "infinity"))

    source_id = int(inst["source_item_id"])
    sink_id = int(inst["sink_item_id"])

    items = inst["items"]
    arcs = inst["arcs"]

    real_items = [int(it["item_id"]) for it in items if it.get("type") == "treatment"]
    if not real_items:
        # Fallback: exclude source/sink by id
        real_items = [
            int(it["item_id"])
            for it in items
            if int(it["item_id"]) not in {source_id, sink_id}
        ]

    weight = {}
    all_item_ids = []
    for it in items:
        iid = int(it["item_id"])
        all_item_ids.append(iid)
        if it.get("type") == "treatment":
            weight[iid] = int(it["weight"])

    # Time windows
    windows = compute_time_windows(
        all_item_ids=all_item_ids,
        real_items=real_items,
        source_id=source_id,
        sink_id=sink_id,
        arcs=arcs,
        T=T,
    )

    # Decide approach
    remaining = lambda: max(0.0, total_time - (time.time() - start_time))
    solution = None

    if L is not None:
        # Exact model with fixed max bins per period
        solution = solve_exact_bin_time(
            real_items=real_items,
            weight=weight,
            arcs=arcs,
            T=T,
            L=L,
            source_id=source_id,
            sink_id=sink_id,
            windows=windows,
            time_limit=remaining(),
            logger=logger,
            best_logged=best_logged,
        )
    else:
        # Unrestricted bins: optimize periods with aggregate bins, then exact/near-exact per-period packing
        solution = solve_aggregate_then_pack(
            real_items=real_items,
            weight=weight,
            arcs=arcs,
            T=T,
            source_id=source_id,
            sink_id=sink_id,
            windows=windows,
            L=None,
            time_limit=remaining(),
            logger=logger,
            best_logged=best_logged,
            start_time=start_time,
            total_time=total_time,
        )

    # Fallback if needed
    if solution is None and remaining() > 0.1:
        period_assign = solve_fallback_feasibility(
            real_items=real_items,
            arcs=arcs,
            T=T,
            source_id=source_id,
            sink_id=sink_id,
            windows=windows,
            time_limit=remaining(),
        )
        if period_assign is not None:
            period_items = defaultdict(list)
            for i, t in period_assign.items():
                period_items[t].append(i)

            assign = {}
            feasible = True
            for t in range(1, T + 1):
                if not period_items[t]:
                    continue
                p_assign, _ = pack_period(
                    period_items[t], weight, W_GLOBAL, L_limit=L, time_limit=max(0.05, remaining() / max(1, T))
                )
                if p_assign is None:
                    feasible = False
                    break
                for i, b in p_assign.items():
                    assign[i] = (b, t)

            if feasible:
                solution = build_solution_json(assign, weight)
                if logger and solution["objective_value"] + 1e-9 < best_logged["val"]:
                    best_logged["val"] = solution["objective_value"]
                    logger.log(solution["objective_value"])

    if solution is None:
        # As a last resort, emit empty feasible-unknown structure (should rarely happen on valid instances)
        solution = {
            "objective_value": float("inf"),
            "assignments": [],
            "active_bin_periods": [],
            "num_active_bin_periods": 0,
        }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()