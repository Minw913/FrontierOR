import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_slabs = data["num_slabs"]
    num_orders = data["num_orders"]
    max_colors = data["max_colors_per_knapsack"]
    slab_w = data["slab_weights"]
    order_w = data["order_weights"]
    order_c = data["order_colors"]
    elig = data["eligible_slabs_per_order"]

    # Build eligible (order, slab) pairs; prune orders too heavy for a slab.
    pairs = []
    orders_per_slab = defaultdict(list)
    for o in range(num_orders):
        for s in elig[o]:
            if s < 0 or s >= num_slabs:
                continue
            if order_w[o] <= slab_w[s] + 1e-9:
                pairs.append((o, s))
                orders_per_slab[s].append(o)

    active_slabs = sorted(orders_per_slab.keys())

    # Colors present among eligible orders per slab
    colors_per_slab = {}
    for s in active_slabs:
        colors_per_slab[s] = sorted(set(order_c[o] for o in orders_per_slab[s]))

    m = gp.Model("steel_mill")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    m.Params.TimeLimit = remaining

    # Variables
    x = {}
    for (o, s) in pairs:
        x[o, s] = m.addVar(vtype=GRB.BINARY, name=f"x_{o}_{s}")
    y = {}
    for s in active_slabs:
        y[s] = m.addVar(vtype=GRB.BINARY, name=f"y_{s}")
    z = {}
    for s in active_slabs:
        # Only need color vars if the slab could exceed the color limit
        if len(colors_per_slab[s]) > max_colors:
            for c in colors_per_slab[s]:
                z[s, c] = m.addVar(vtype=GRB.BINARY, name=f"z_{s}_{c}")

    # Each order assigned at most once
    slabs_per_order = defaultdict(list)
    for (o, s) in pairs:
        slabs_per_order[o].append(s)
    for o, ss in slabs_per_order.items():
        m.addConstr(gp.quicksum(x[o, s] for s in ss) <= 1)

    # Capacity and linking to slab-in-service
    for s in active_slabs:
        m.addConstr(
            gp.quicksum(order_w[o] * x[o, s] for o in orders_per_slab[s])
            <= slab_w[s] * y[s]
        )

    # Color constraints
    for s in active_slabs:
        if len(colors_per_slab[s]) > max_colors:
            by_color = defaultdict(list)
            for o in orders_per_slab[s]:
                by_color[order_c[o]].append(o)
            for c, os_ in by_color.items():
                for o in os_:
                    m.addConstr(x[o, s] <= z[s, c])
            m.addConstr(
                gp.quicksum(z[s, c] for c in colors_per_slab[s]) <= max_colors * y[s]
            )

    # Objective: 2 * assigned order weight - used slab weight
    m.setObjective(
        2.0 * gp.quicksum(order_w[o] * x[o, s] for (o, s) in pairs)
        - gp.quicksum(slab_w[s] * y[s] for s in active_slabs),
        GRB.MAXIMIZE,
    )

    best = {"obj": None}

    def extract_solution_from_vals(get_val):
        assignments = []
        slabs_used_set = set()
        for (o, s), var in x.items():
            if get_val(var) > 0.5:
                assignments.append({"order": o, "slab": s})
                slabs_used_set.add(s)
        obj = 2.0 * sum(order_w[a["order"]] for a in assignments) - sum(
            slab_w[s] for s in slabs_used_set
        )
        return {
            "objective_value": obj,
            "assignments": assignments,
            "slabs_used": sorted(slabs_used_set),
        }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            sol_vals = model.cbGetSolution(list(x.values()))
            val_map = dict(zip(x.values(), sol_vals))
            sol = extract_solution_from_vals(lambda v: val_map[v])
            obj = sol["objective_value"]
            if best["obj"] is None or obj > best["obj"]:
                best["obj"] = obj
                if logger:
                    logger.log_solution(obj, sol)

    m.optimize(callback)

    # Final solution
    if m.SolCount > 0:
        sol = extract_solution_from_vals(lambda v: v.X)
    else:
        sol = {"objective_value": 0.0, "assignments": [], "slabs_used": []}

    if logger and (best["obj"] is None or sol["objective_value"] > best["obj"]):
        logger.log_solution(sol["objective_value"], sol)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()