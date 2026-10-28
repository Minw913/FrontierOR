import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_slabs = data['num_slabs']
    num_orders = data['num_orders']
    num_colors = data['num_colors']
    max_colors_per_knapsack = data['max_colors_per_knapsack']
    slab_weights = data['slab_weights']
    order_weights = data['order_weights']
    order_colors = data['order_colors']
    eligible_slabs_per_order = data['eligible_slabs_per_order']

    # Build reverse eligibility and color sets
    eligible_orders_per_slab = [[] for _ in range(num_slabs)]
    colors_per_slab = [set() for _ in range(num_slabs)]
    for o in range(num_orders):
        for s in eligible_slabs_per_order[o]:
            eligible_orders_per_slab[s].append(o)
            colors_per_slab[s].add(order_colors[o])

    start_time = time.time()

    # Create Gurobi model
    model = gp.Model("Slab_Assignment")
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 0.01)  # Allow 1% gap for speed
    model.setParam(GRB.Param.LogToConsole, 0)

    # Variables
    assign = {}
    for o in range(num_orders):
        for s in eligible_slabs_per_order[o]:
            assign[o, s] = model.addVar(vtype=GRB.BINARY, name=f"assign_{o}_{s}")

    use_slab = [model.addVar(vtype=GRB.BINARY, name=f"use_{s}") for s in range(num_slabs)]

    color_used = {}
    for s in range(num_slabs):
        for c in colors_per_slab[s]:
            color_used[s, c] = model.addVar(vtype=GRB.BINARY, name=f"color_{s}_{c}")

    # Objective: maximize 2*sum(assigned weights) - sum(used slab weights)
    obj = gp.quicksum(2 * order_weights[o] * assign[o, s] for o, s in assign.keys()) - \
          gp.quicksum(slab_weights[s] * use_slab[s] for s in range(num_slabs))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    # Each order assigned at most once
    for o in range(num_orders):
        model.addConstr(gp.quicksum(assign[o, s] for s in eligible_slabs_per_order[o]) <= 1)

    # Weight capacity
    for s in range(num_slabs):
        model.addConstr(
            gp.quicksum(order_weights[o] * assign.get((o, s), 0) for o in eligible_orders_per_slab[s]) <=
            slab_weights[s] * use_slab[s]
        )

    # Color limits
    for s in range(num_slabs):
        possible_colors = list(colors_per_slab[s])
        if not possible_colors:
            model.addConstr(use_slab[s] == 0)
            continue
        model.addConstr(gp.quicksum(color_used[s, c] for c in possible_colors) <= max_colors_per_knapsack)

        for o_idx in eligible_orders_per_slab[s]:
            c = order_colors[o_idx]
            model.addConstr(assign[o_idx, s] <= color_used[s, c])

    # Link slab usage
    for s in range(num_slabs):
        model.addConstr(use_slab[s] >= gp.quicksum(assign[o, s] for o in eligible_orders_per_slab[s]) * 0.001)

    # Logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    best_obj = float('-inf')
    incumbent_assignments = []
    incumbent_slabs = []

    def callback(model, where):
        nonlocal best_obj, incumbent_assignments, incumbent_slabs
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                sol_assign = model.cbGetSolution(assign)
                sol_use = model.cbGetSolution(use_slab)

                assignments = []
                slabs_used = []
                for o_idx in range(num_orders):
                    for s_idx in eligible_slabs_per_order[o_idx]:
                        if sol_assign.get((o_idx, s_idx), 0) > 0.5:
                            assignments.append({"order": o_idx, "slab": s_idx})
                            if s_idx not in slabs_used:
                                slabs_used.append(s_idx)
                if logger:
                    logger.log(float(obj_val))
                incumbent_assignments = assignments
                incumbent_slabs = slabs_used

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        final_obj = model.ObjVal
        assignments = []
        slabs_used = []
        for o in range(num_orders):
            for s in eligible_slabs_per_order[o]:
                if assign[o, s].X > 0.5:
                    assignments.append({"order": int(o), "slab": int(s)})
                    if s not in slabs_used:
                        slabs_used.append(int(s))
    else:
        # No solution found
        final_obj = 0.0
        assignments = []
        slabs_used = []

    # Write output
    solution = {
        "objective_value": float(final_obj),
        "assignments": assignments,
        "slabs_used": slabs_used
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()