import argparse
import json
import os
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def sanitize_instance(inst):
    num_slabs = int(inst["num_slabs"])
    num_orders = int(inst["num_orders"])
    num_colors = int(inst["num_colors"])
    max_colors = int(inst["max_colors_per_knapsack"])

    slab_weights = [float(w) for w in inst["slab_weights"]]
    order_weights = [float(w) for w in inst["order_weights"]]
    order_colors = [int(c) for c in inst["order_colors"]]

    raw_eligible = inst["eligible_slabs_per_order"]
    eligible_slabs_per_order = []
    for o in range(num_orders):
        slabs = raw_eligible[o] if o < len(raw_eligible) else []
        clean = sorted({int(s) for s in slabs if 0 <= int(s) < num_slabs})
        eligible_slabs_per_order.append(clean)

    return {
        "num_slabs": num_slabs,
        "num_orders": num_orders,
        "num_colors": num_colors,
        "max_colors": max_colors,
        "slab_weights": slab_weights,
        "order_weights": order_weights,
        "order_colors": order_colors,
        "eligible_slabs_per_order": eligible_slabs_per_order,
    }


def build_reverse_eligibility(num_slabs, eligible_slabs_per_order):
    eligible_orders_per_slab = [[] for _ in range(num_slabs)]
    for o, slabs in enumerate(eligible_slabs_per_order):
        for s in slabs:
            eligible_orders_per_slab[s].append(o)
    return eligible_orders_per_slab


def greedy_start(data, eligible_orders_per_slab):
    """
    Lightweight constructive heuristic to provide a feasible MIP start.
    """
    S = data["num_slabs"]
    slab_weights = data["slab_weights"]
    order_weights = data["order_weights"]
    order_colors = data["order_colors"]
    max_colors = data["max_colors"]

    unassigned = set(range(data["num_orders"]))
    assignment = {}  # order -> slab

    # Small slabs first often improves fill ratio quickly
    slab_order = sorted(range(S), key=lambda s: slab_weights[s])

    for s in slab_order:
        cap = slab_weights[s]
        cand = [o for o in eligible_orders_per_slab[s] if o in unassigned]
        if not cand:
            continue

        by_color = {}
        for o in cand:
            c = order_colors[o]
            by_color.setdefault(c, []).append(o)

        colors = list(by_color.keys())
        if not colors:
            continue

        # Limit pair enumeration to top colors by available weight
        color_score = {c: sum(order_weights[o] for o in by_color[c]) for c in colors}
        top_colors = [c for c, _ in sorted(color_score.items(), key=lambda kv: kv[1], reverse=True)[:8]]

        color_sets = [(c,) for c in top_colors]
        if max_colors >= 2:
            color_sets.extend(combinations(top_colors, 2))

        best_sel = []
        best_gain = 0.0

        for cs in color_sets:
            cs_set = set(cs)
            candidates = [o for o in cand if order_colors[o] in cs_set]
            candidates.sort(key=lambda o: order_weights[o], reverse=True)

            rem = cap
            sel = []
            packed = 0.0
            for o in candidates:
                w = order_weights[o]
                if w <= rem + 1e-9:
                    sel.append(o)
                    rem -= w
                    packed += w

            gain = 2.0 * packed - cap
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_sel = sel

        if best_sel:
            for o in best_sel:
                assignment[o] = s
                unassigned.remove(o)

    return assignment


def solve(instance, time_limit, logger=None):
    S = instance["num_slabs"]
    O = instance["num_orders"]
    slab_weights = instance["slab_weights"]
    order_weights = instance["order_weights"]
    order_colors = instance["order_colors"]
    eligible_slabs_per_order = instance["eligible_slabs_per_order"]
    max_colors = instance["max_colors"]

    eligible_orders_per_slab = build_reverse_eligibility(S, eligible_slabs_per_order)

    slab_colors = []
    for s in range(S):
        cols = sorted({order_colors[o] for o in eligible_orders_per_slab[s]})
        slab_colors.append(cols)

    model = gp.Model("steel_slab_order_assignment")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)
    model.Params.MIPFocus = 1

    # Variables
    x = {}
    for o in range(O):
        for s in eligible_slabs_per_order[o]:
            x[o, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{o}_{s}")

    y = model.addVars(S, vtype=GRB.BINARY, name="y")

    z = {}
    for s in range(S):
        for c in slab_colors[s]:
            z[s, c] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{c}")

    model.update()

    # Objective
    model.setObjective(
        gp.quicksum(2.0 * order_weights[o] * var for (o, s), var in x.items())
        - gp.quicksum(slab_weights[s] * y[s] for s in range(S)),
        GRB.MAXIMIZE,
    )

    # Each order assigned at most once
    for o in range(O):
        slabs = eligible_slabs_per_order[o]
        if slabs:
            model.addConstr(gp.quicksum(x[o, s] for s in slabs) <= 1, name=f"assign_once_{o}")

    # Capacity and slab activation semantics
    for s in range(S):
        ords = eligible_orders_per_slab[s]
        if ords:
            model.addConstr(
                gp.quicksum(order_weights[o] * x[o, s] for o in ords) <= slab_weights[s] * y[s],
                name=f"capacity_{s}",
            )
            model.addConstr(
                y[s] <= gp.quicksum(x[o, s] for o in ords),
                name=f"used_if_assigned_{s}",
            )
        else:
            model.addConstr(y[s] == 0, name=f"unused_no_orders_{s}")

    # Color-cardinality constraints
    for s in range(S):
        cols = slab_colors[s]
        if cols:
            model.addConstr(gp.quicksum(z[s, c] for c in cols) <= max_colors, name=f"max_colors_{s}")
            for c in cols:
                model.addConstr(z[s, c] <= y[s], name=f"color_only_if_used_{s}_{c}")

    for (o, s), var in x.items():
        c = order_colors[o]
        if (s, c) in z:
            model.addConstr(var <= z[s, c], name=f"link_color_{o}_{s}")
        else:
            model.addConstr(var == 0, name=f"invalid_color_{o}_{s}")

    # Greedy MIP start
    start_assignment = greedy_start(instance, eligible_orders_per_slab)
    if start_assignment:
        used_slabs = set()
        used_colors = set()
        for (o, s), var in x.items():
            if start_assignment.get(o, None) == s:
                var.Start = 1.0
                used_slabs.add(s)
                used_colors.add((s, order_colors[o]))
            else:
                var.Start = 0.0
        for s in range(S):
            y[s].Start = 1.0 if s in used_slabs else 0.0
        for (s, c), var in z.items():
            var.Start = 1.0 if (s, c) in used_colors else 0.0

    # Incumbent logging callback
    def cb(m, where):
        if logger and where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(float(obj_val))

    model.optimize(cb)

    # Extract best feasible solution
    assignments = []
    slabs_used = []
    objective_value = 0.0

    if model.SolCount > 0:
        for (o, s), var in x.items():
            if var.X > 0.5:
                assignments.append({"order": int(o), "slab": int(s)})

        slabs_used = [int(s) for s in range(S) if y[s].X > 0.5]

        assigned_weight = sum(order_weights[a["order"]] for a in assignments)
        used_weight = sum(slab_weights[s] for s in slabs_used)
        objective_value = float(2.0 * assigned_weight - used_weight)

        if logger:
            logger.log(objective_value)

    assignments.sort(key=lambda d: d["order"])
    slabs_used.sort()

    return {
        "objective_value": objective_value,
        "assignments": assignments,
        "slabs_used": slabs_used,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        raw_instance = json.load(f)
    instance = sanitize_instance(raw_instance)

    solution = solve(instance, args.time_limit, logger=logger)

    os.makedirs(os.path.dirname(args.solution_path), exist_ok=True) if os.path.dirname(args.solution_path) else None
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()