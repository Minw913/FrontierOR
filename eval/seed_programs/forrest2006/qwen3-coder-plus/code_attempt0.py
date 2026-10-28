import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_instance(instance_data, time_limit, log_path):
    num_slabs = instance_data['num_slabs']
    num_orders = instance_data['num_orders']
    num_colors = instance_data['num_colors']
    max_colors_per_knapsack = instance_data['max_colors_per_knapsack']
    slab_weights = instance_data['slab_weights']
    order_weights = instance_data['order_weights']
    order_colors = instance_data['order_colors']
    eligible_slabs_per_order = instance_data['eligible_slabs_per_order']

    # Create model
    model = gp.Model("steel_mill")

    # Set time limit
    model.setParam('TimeLimit', time_limit)

    # Create sets for colors per slab
    color_set_per_slab = [set() for _ in range(num_slabs)]
    orders_per_color_per_slab = [[[] for _ in range(num_colors)] for _ in range(num_slabs)]

    for order_idx in range(num_orders):
        for slab_idx in eligible_slabs_per_order[order_idx]:
            color = order_colors[order_idx]
            color_set_per_slab[slab_idx].add(color)
            orders_per_color_per_slab[slab_idx][color].append(order_idx)

    # Decision variables:
    # 1. x[i][j]: binary, order i assigned to slab j
    # 2. y[j]: binary, slab j in use
    # 3. z[c][j]: binary, color c used in slab j
    x = {}
    for order_idx in range(num_orders):
        for slab_idx in eligible_slabs_per_order[order_idx]:
            x[order_idx, slab_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{order_idx}_{slab_idx}")

    y = {}
    for slab_idx in range(num_slabs):
        y[slab_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{slab_idx}")

    z = {}
    for slab_idx in range(num_slabs):
        for color_idx in range(num_colors):
            if orders_per_color_per_slab[slab_idx][color_idx]:
                z[color_idx, slab_idx] = model.addVar(vtype=GRB.BINARY, name=f"z_{color_idx}_{slab_idx}")

    model.update()

    # Constraints:
    # 1. Each order assigned to at most one slab
    for order_idx in range(num_orders):
        model.addConstr(gp.quicksum(x[order_idx, slab_idx] 
                                   for slab_idx in eligible_slabs_per_order[order_idx]) <= 1,
                       name=f"assign_order_{order_idx}")

    # 2. If an order is assigned to a slab, that slab must be in use
    for order_idx in range(num_orders):
        for slab_idx in eligible_slabs_per_order[order_idx]:
            model.addConstr(x[order_idx, slab_idx] <= y[slab_idx],
                           name=f"slab_in_use_order_{order_idx}_slab_{slab_idx}")

    # 3. Capacity constraint - total weight on a slab must not exceed its capacity
    for slab_idx in range(num_slabs):
        expr = gp.quicksum(x[order_idx, slab_idx] * order_weights[order_idx]
                          for order_idx in range(num_orders)
                          if order_idx in {o for sublist in orders_per_color_per_slab[slab_idx] for o in sublist})
        model.addConstr(expr <= slab_weights[slab_idx] * y[slab_idx],
                       name=f"capacity_slab_{slab_idx}")

    # 4. Color usage constraints
    for slab_idx in range(num_slabs):
        for color_idx in range(num_colors):
            if orders_per_color_per_slab[slab_idx][color_idx]:
                model.addConstr(gp.quicksum(x[order_idx, slab_idx] 
                                           for order_idx in orders_per_color_per_slab[slab_idx][color_idx])
                               <= len(orders_per_color_per_slab[slab_idx][color_idx]) * z[color_idx, slab_idx],
                              name=f"color_usage_{color_idx}_slab_{slab_idx}")

    # 5. At most max_colors_per_knapsack distinct colors per slab
    for slab_idx in range(num_slabs):
        model.addConstr(gp.quicksum(z[color_idx, slab_idx] for color_idx in range(num_colors)
                                   if orders_per_color_per_slab[slab_idx][color_idx]) <= max_colors_per_knapsack * y[slab_idx],
                       name=f"max_colors_slab_{slab_idx}")

    # Objective: maximize sum of 2 * (total weight of assigned orders) - (total weight of used slabs)
    obj_expr = (gp.quicksum(2 * order_weights[order_idx] * 
                           gp.quicksum(x[order_idx, slab_idx] 
                                      for slab_idx in eligible_slabs_per_order[order_idx]))
               - gp.quicksum(slab_weights[slab_idx] * y[slab_idx]))

    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Set up callback for logging
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if log_path:
                logger.log(obj_val)

    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Optimize
    model.optimize(my_callback)

    # Extract solution
    assignments = []
    slabs_used = []

    for order_idx in range(num_orders):
        for slab_idx in eligible_slabs_per_order[order_idx]:
            if x[order_idx, slab_idx].x > 0.5:
                assignments.append({"order": order_idx, "slab": slab_idx})

    for slab_idx in range(num_slabs):
        if y[slab_idx].x > 0.5:
            slabs_used.append(slab_idx)

    # Calculate the final objective value based on current solution
    total_order_weight = sum(order_weights[assignment["order"]] for assignment in assignments)
    total_slab_weight = sum(slab_weights[slab_idx] for slab_idx in slabs_used)
    objective_value = 2 * total_order_weight - total_slab_weight

    solution = {
        "objective_value": objective_value,
        "assignments": assignments,
        "slabs_used": slabs_used
    }

    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, required=False,
                        help="Path to a JSONL file for logging intermediate solutions")

    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    solution = solve_instance(instance_data, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)