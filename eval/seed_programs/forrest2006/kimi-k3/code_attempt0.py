import argparse
import json
import sys
import time

import gurobipy as gp
from gurobipy import GRB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the problem instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the solution JSON file will be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Optional path to a JSONL file for logging incumbents.")
    return parser.parse_args()


def greedy_initial_solution(S, O, K, slab_w, ord_w, ord_c,
                            order_slabs, slab_orders, deadline):
    """Fast constructive heuristic.

    Processes orders in decreasing weight. Each order is placed on an
    already-open compatible slab (best-fit) when possible; otherwise a new
    slab is opened when a greedy single-slab packing (respecting the color
    limit) yields a positive marginal objective contribution.
    """
    remaining = list(slab_w)
    colors_on_slab = [set() for _ in range(S)]
    is_open = [False] * S
    assignment = {}
    slab_sorted = [sorted(slab_orders[s], key=lambda o: -ord_w[o])
                   for s in range(S)]

    def try_open(o, cand_slabs):
        wo, co = ord_w[o], ord_c[o]
        best = None
        for s in cand_slabs:
            if time.time() > deadline and best is not None:
                break
            if is_open[s] or slab_w[s] < wo:
                continue
            cap = slab_w[s] - wo
            cols = {co}
            added_w = 0.0
            added = []
            for o2 in slab_sorted[s]:
                if o2 == o or o2 in assignment:
                    continue
                w2 = ord_w[o2]
                if w2 > cap:
                    continue
                c2 = ord_c[o2]
                if c2 not in cols:
                    if len(cols) >= K:
                        continue
                    cols.add(c2)
                cap -= w2
                added_w += w2
                added.append(o2)
            gain = 2.0 * (wo + added_w) - slab_w[s]
            if best is None or gain > best[0]:
                best = (gain, s, added, cols)
        if best is not None and best[0] > 1e-9:
            _, s, added, cols = best
            is_open[s] = True
            colors_on_slab[s] = set(cols)
            tot = ord_w[o]
            assignment[o] = s
            for o2 in added:
                assignment[o2] = s
                tot += ord_w[o2]
            remaining[s] = slab_w[s] - tot
            return True
        return False

    for idx, o in enumerate(sorted(range(O), key=lambda t: -ord_w[t])):
        if (idx & 63) == 0 and time.time() > deadline:
            break
        if o in assignment:
            continue
        cands = order_slabs[o]
        if not cands:
            continue
        wo, co = ord_w[o], ord_c[o]
        best_s = -1
        best_left = None
        for s in cands:
            if not is_open[s]:
                continue
            if remaining[s] < wo:
                continue
            if co not in colors_on_slab[s] and len(colors_on_slab[s]) >= K:
                continue
            left = remaining[s] - wo
            if best_left is None or left < best_left:
                best_left = left
                best_s = s
        if best_s >= 0:
            assignment[o] = best_s
            remaining[best_s] -= wo
            colors_on_slab[best_s].add(co)
        else:
            try_open(o, cands)
    return assignment


def solve(inst, time_limit, logger):
    t0 = time.time()
    S = int(inst["num_slabs"])
    O = int(inst["num_orders"])
    K = int(inst.get("max_colors_per_knapsack", 2))
    slab_w = [float(v) for v in inst["slab_weights"]]
    ord_w = [float(v) for v in inst["order_weights"]]
    ord_c = [int(v) for v in inst["order_colors"]]
    elig = inst["eligible_slabs_per_order"]

    # Eligibility structure (drop pairs where the order alone exceeds capacity).
    order_slabs = [[] for _ in range(O)]
    slab_orders = [[] for _ in range(S)]
    slab_color_set = [set() for _ in range(S)]
    for o in range(O):
        if o >= len(elig):
            break
        seen = set()
        for s in elig[o]:
            s = int(s)
            if s in seen or s < 0 or s >= S:
                continue
            seen.add(s)
            if ord_w[o] <= slab_w[s]:
                order_slabs[o].append(s)
                slab_orders[s].append(o)
                slab_color_set[s].add(ord_c[o])

    def build_solution(assignment):
        tot = 0.0
        used = set()
        for o, s in assignment.items():
            tot += ord_w[o]
            used.add(s)
        obj = 2.0 * tot - sum(slab_w[s] for s in used)
        sol = {
            "objective_value": float(obj),
            "assignments": [
                {"order": int(o), "slab": int(s)}
                for o, s in sorted(assignment.items())
            ],
            "slabs_used": [int(s) for s in sorted(used)],
        }
        return sol, obj

    # --- Constructive heuristic (also serves as warm start / fallback) ---
    greedy_deadline = t0 + max(0.5, min(0.2 * float(time_limit), 10.0))
    init_assignment = greedy_initial_solution(
        S, O, K, slab_w, ord_w, ord_c, order_slabs, slab_orders,
        greedy_deadline)
    best_sol, best_obj = build_solution(init_assignment)
    if logger is not None:
        logger.log_solution(best_obj, best_sol)

    if sum(len(v) for v in order_slabs) == 0:
        return best_sol

    # --- MIP model ---
    model = gp.Model("slab_order_allocation")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining_time = time_limit - (time.time() - t0)
    model.Params.TimeLimit = max(0.5, remaining_time - 0.5)

    x = {}
    for o in range(O):
        for s in order_slabs[o]:
            x[(o, s)] = model.addVar(vtype=GRB.BINARY, name=f"x_{o}_{s}")
    y = {}
    for s in range(S):
        if slab_orders[s]:
            y[s] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}")
    z = {}
    need_color = {}
    for s in range(S):
        cols = slab_color_set[s]
        need_color[s] = len(cols) > K
        if need_color[s]:
            for c in cols:
                z[(s, c)] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{c}")
    model.update()

    # Each order assigned to at most one slab.
    for o in range(O):
        if order_slabs[o]:
            model.addConstr(
                gp.quicksum(x[(o, s)] for s in order_slabs[o]) <= 1.0,
                name=f"once_{o}")

    # Capacity, opening and color-limit constraints.
    for s in range(S):
        if not slab_orders[s]:
            continue
        model.addConstr(
            gp.quicksum(ord_w[o] * x[(o, s)] for o in slab_orders[s])
            <= slab_w[s] * y[s],
            name=f"cap_{s}")
        if need_color[s]:
            orders_by_color = {}
            for o in slab_orders[s]:
                orders_by_color.setdefault(ord_c[o], []).append(o)
            for c, olist in orders_by_color.items():
                for o in olist:
                    model.addConstr(x[(o, s)] <= z[(s, c)],
                                    name=f"col_{o}_{s}")
            model.addConstr(
                gp.quicksum(z[(s, c)] for c in orders_by_color)
                <= float(K) * y[s],
                name=f"ncol_{s}")

    model.setObjective(
        2.0 * gp.quicksum(ord_w[o] * x[(o, s)] for (o, s) in x.keys())
        - gp.quicksum(slab_w[s] * y[s] for s in y.keys()),
        GRB.MAXIMIZE)

    # Warm start from the heuristic solution.
    for o, s in init_assignment.items():
        x[(o, s)].Start = 1.0
    used_slabs = set(init_assignment.values())
    colors_used = {}
    for o, s in init_assignment.items():
        colors_used.setdefault(s, set()).add(ord_c[o])
    for s in used_slabs:
        y[s].Start = 1.0
    for (s, c) in z.keys():
        z[(s, c)].Start = 1.0 if c in colors_used.get(s, ()) else 0.0

    x_keys = list(x.keys())
    x_vars = [x[k] for k in x_keys]
    state = {"best": best_obj}

    def callback(m, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            vals = m.cbGetSolution(x_vars)
        except gp.GurobiError:
            return
        assign = {}
        for (o, s), v in zip(x_keys, vals):
            if v > 0.5:
                assign[o] = s
        sol, objv = build_solution(assign)
        if objv > state["best"] + 1e-9:
            state["best"] = objv
            logger.log_solution(objv, sol)

    model.optimize(callback)

    # Extract the final incumbent.
    if model.SolCount > 0:
        try:
            vals = model.getAttr("X", x_vars)
            assign = {o: s for (o, s), v in zip(x_keys, vals) if v > 0.5}
            sol, objv = build_solution(assign)
            if objv > best_obj + 1e-9:
                best_sol, best_obj = sol, objv
                if logger is not None:
                    logger.log_solution(best_obj, best_sol)
        except gp.GurobiError:
            pass

    return best_sol


def main():
    args = parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    try:
        solution = solve(inst, args.time_limit, logger)
    except Exception as exc:  # guarantee a schema-valid output file
        print(f"solve failed: {exc}", file=sys.stderr)
        solution = {"objective_value": 0.0, "assignments": [], "slabs_used": []}

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)

    if logger is not None:
        close = getattr(logger, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()