import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def greedy_heuristic(data):
    """Simple greedy heuristic to generate an initial feasible solution."""
    num_slabs = data['num_slabs']
    num_orders = data['num_orders']
    max_colors = data['max_colors_per_knapsack']
    slab_weights = data['slab_weights']
    order_weights = data['order_weights']
    order_colors = data['order_colors']
    eligible_slabs = data['eligible_slabs_per_order']

    order_indices = sorted(range(num_orders), key=lambda o: order_weights[o], reverse=True)

    remaining_capacity = list(slab_weights)
    slab_colors = [set() for _ in range(num_slabs)]
    assignments = []
    slabs_used = set()

    for o in order_indices:
        best_slab = -1
        best_score = float('-inf')

        for s in eligible_slabs[o]:
            color = order_colors[o]
            if color in slab_colors[s]:
                can_assign = True
            elif len(slab_colors[s]) < max_colors:
                can_assign = True
            else:
                can_assign = False

            if can_assign and order_weights[o] <= remaining_capacity[s]:
                if s in slabs_used:
                    gain = 2.0 * order_weights[o]
                else:
                    gain = 2.0 * order_weights[o] - slab_weights[s]

                # Prefer slabs with less remaining capacity (best fit)
                score = gain * 1000 - remaining_capacity[s]
                if gain > 0 and score > best_score:
                    best_score = score
                    best_slab = s

        if best_slab >= 0:
            assignments.append({"order": o, "slab": best_slab})
            remaining_capacity[best_slab] -= order_weights[o]
            slab_colors[best_slab].add(order_colors[o])
            slabs_used.add(best_slab)

    obj_val = sum(2.0 * order_weights[a["order"]] for a in assignments) - sum(slab_weights[s] for s in slabs_used)

    return obj_val, assignments, sorted(list(slabs_used))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    num_slabs = data['num_slabs']
    num_orders = data['num_orders']
    num_colors = data['num_colors']
    max_colors = data['max_colors_per_knapsack']
    slab_weights = data['slab_weights']
    order_weights = data['order_weights']
    order_colors = data['order_colors']
    eligible_slabs = data['eligible_slabs_per_order']

    # Build eligible orders per slab
    eligible_orders_per_slab = [[] for _ in range(num_slabs)]
    for o in range(num_orders):
        for s in eligible_slabs[o]:
            eligible_orders_per_slab[s].append(o)

    # Colors per slab
    colors_per_slab = []
    for s in range(num_slabs):
        colors = set()
        for o in eligible_orders_per_slab[s]:
            colors.add(order_colors[o])
        colors_per_slab.append(sorted(colors))

    # Run greedy heuristic for initial solution and logging
    greedy_obj, greedy_assignments, greedy_slabs_used = greedy_heuristic(data)
    if logger:
        greedy_solution = {
            "objective_value": greedy_obj,
            "assignments": greedy_assignments,
            "slabs_used": greedy_slabs_used
        }
        logger.log_solution(greedy_obj, greedy_solution)

    # Build MIP
    model = gp.Model("slab_design")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = {}
    for o in range(num_orders):
        for s in eligible_slabs[o]:
            x[o, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{o}_{s}")

    y = {}
    for s in range(num_slabs):
        y[s] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}")

    z = {}
    for s in range(num_slabs):
        for c in colors_per_slab[s]:
            z[s, c] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{c}")

    model.update()

    # Objective: maximize 2*sum(assigned order weights) - sum(slab weights in service)
    obj = gp.LinExpr()
    for o in range(num_orders):
        for s in eligible_slabs[o]:
            obj += 2.0 * order_weights[o] * x[o, s]
    for s in range(num_slabs):
        obj -= slab_weights[s] * y[s]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint: each order assigned to at most one slab
    for o in range(num_orders):
        if len(eligible_slabs[o]) > 0:
            model.addConstr(
                gp.quicksum(x[o, s] for s in eligible_slabs[o]) <= 1,
                name=f"assign_{o}"
            )

    # Capacity constraint and service linkage
    for s in range(num_slabs):
        if eligible_orders_per_slab[s]:
            model.addConstr(
                gp.quicksum(order_weights[o] * x[o, s] for o in eligible_orders_per_slab[s])
                <= slab_weights[s] * y[s],
                name=f"cap_{s}"
            )
            # If slab is in service, at least one order must be assigned
            model.addConstr(
                y[s] <= gp.quicksum(x[o, s] for o in eligible_orders_per_slab[s]),
                name=f"service_{s}"
            )
        else:
            model.addConstr(y[s] == 0, name=f"no_orders_{s}")

    # Color limit per slab
    for s in range(num_slabs):
        if colors_per_slab[s]:
            model.addConstr(
                gp.quicksum(z[s, c] for c in colors_per_slab[s]) <= max_colors,
                name=f"color_limit_{s}"
            )

    # Color linkage: order assigned to slab implies its color is selected
    for o in range(num_orders):
        for s in eligible_slabs[o]:
            model.addConstr(x[o, s] <= z[s, order_colors[o]], name=f"clink_{o}_{s}")

    model.update()

    # Set MIP start from greedy heuristic
    try:
        for a in greedy_assignments:
            o, s = a["order"], a["slab"]
            if (o, s) in x:
                x[o, s].Start = 1.0
        for s in greedy_slabs_used:
            y[s].Start = 1.0
        for s in range(num_slabs):
            if s not in greedy_slabs_used:
                y[s].Start = 0.0
        for s in range(num_slabs):
            for c in colors_per_slab[s]:
                if s in greedy_slabs_used:
                    # Determine which colors are used on this slab in greedy solution
                    used_colors = set()
                    for a in greedy_assignments:
                        if a["slab"] == s:
                            used_colors.add(order_colors[a["order"]])
                    z[s, c].Start = 1.0 if c in used_colors else 0.0
                else:
                    z[s, c].Start = 0.0
        # Set unassigned orders
        assigned_orders = set(a["order"] for a in greedy_assignments)
        for o in range(num_orders):
            if o not in assigned_orders:
                for s in eligible_slabs[o]:
                    x[o, s].Start = 0.0
    except Exception:
        pass

    # Callback for logging incumbents
    best_logged_obj = greedy_obj

    def callback(model_cb, where):
        nonlocal best_logged_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_logged_obj + 1e-6:
                best_logged_obj = obj_val
                assignments = []
                slabs_used = set()
                for o in range(num_orders):
                    for s in eligible_slabs[o]:
                        val = model_cb.cbGetSolution(x[o, s])
                        if val > 0.5:
                            assignments.append({"order": o, "slab": s})
                            slabs_used.add(s)
                solution = {
                    "objective_value": obj_val,
                    "assignments": assignments,
                    "slabs_used": sorted(list(slabs_used))
                }
                if logger:
                    logger.log_solution(obj_val, solution)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        assignments = []
        slabs_used = set()
        for o in range(num_orders):
            for s in eligible_slabs[o]:
                if x[o, s].X > 0.5:
                    assignments.append({"order": o, "slab": s})
                    slabs_used.add(s)

        obj_val = model.ObjVal
        solution = {
            "objective_value": obj_val,
            "assignments": assignments,
            "slabs_used": sorted(list(slabs_used))
        }
    else:
        # Fallback to greedy solution
        solution = {
            "objective_value": greedy_obj,
            "assignments": greedy_assignments,
            "slabs_used": greedy_slabs_used
        }

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()