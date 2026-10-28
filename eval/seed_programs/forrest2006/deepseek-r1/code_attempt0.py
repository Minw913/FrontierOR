import argparse
import json
import gurobipy as gp
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    
    num_slabs = data['num_slabs']
    num_orders = data['num_orders']
    eligible_slabs_per_order = data['eligible_slabs_per_order']
    order_colors = data['order_colors']
    slab_weights = data['slab_weights']
    order_weights = data['order_weights']
    max_colors_per_knapsack = data['max_colors_per_knapsack']
    
    eligible_orders_per_slab = [[] for _ in range(num_slabs)]
    for order_idx in range(num_orders):
        for slab_idx in eligible_slabs_per_order[order_idx]:
            eligible_orders_per_slab[slab_idx].append(order_idx)
    
    colors_per_slab = []
    for j in range(num_slabs):
        colors_j = set()
        for order_idx in eligible_orders_per_slab[j]:
            colors_j.add(order_colors[order_idx])
        colors_per_slab.append(colors_j)
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    x = {}
    for i in range(num_orders):
        for j in eligible_slabs_per_order[i]:
            x[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}")
    
    y = model.addVars(num_slabs, vtype=gp.GRB.BINARY, name="y")
    
    z = {}
    for j in range(num_slabs):
        for c in colors_per_slab[j]:
            z[j, c] = model.addVar(vtype=gp.GRB.BINARY, name=f"z_{j}_{c}")
    
    for i in range(num_orders):
        if eligible_slabs_per_order[i]:
            model.addConstr(
                gp.quicksum(x[i, j] for j in eligible_slabs_per_order[i]) <= 1,
                name=f"order_{i}"
            )
    
    for j in range(num_slabs):
        if eligible_orders_per_slab[j]:
            total_weight = gp.quicksum(
                order_weights[i] * x[i, j] for i in eligible_orders_per_slab[j]
            )
            model.addConstr(
                total_weight <= slab_weights[j] * y[j],
                name=f"slab_weight_{j}"
            )
        else:
            model.addConstr(y[j] == 0, name=f"slab_{j}_empty")
    
    for j in range(num_slabs):
        for i in eligible_orders_per_slab[j]:
            c_i = order_colors[i]
            model.addConstr(
                x[i, j] <= z[j, c_i],
                name=f"color_{i}_{j}"
            )
    
    for j in range(num_slabs):
        if colors_per_slab[j]:
            model.addConstr(
                gp.quicksum(z[j, c] for c in colors_per_slab[j]) <= max_colors_per_knapsack,
                name=f"color_limit_{j}"
            )
    
    total_assigned = gp.quicksum(
        order_weights[i] * x[i, j] for (i, j) in x.keys()
    )
    total_slab_penalty = gp.quicksum(
        slab_weights[j] * y[j] for j in range(num_slabs)
    )
    model.setObjective(2 * total_assigned - total_slab_penalty, gp.GRB.MAXIMIZE)
    
    if logger:
        model._x = x
        model._y = y
        model._num_slabs = num_slabs
        model._logger = logger
        
        def gurobi_callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                x_vals = model.cbGetSolution(model._x)
                y_vals = model.cbGetSolution(model._y)
                assignments = []
                for (i, j) in model._x.keys():
                    if x_vals[i, j] > 0.5:
                        assignments.append({"order": i, "slab": j})
                slabs_used = [j for j in range(model._num_slabs) if y_vals[j] > 0.5]
                solution_dict = {
                    "assignments": assignments,
                    "slabs_used": slabs_used
                }
                model._logger.log_solution(obj_val, solution_dict)
        
        model.optimize(gurobi_callback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        assignments = []
        for (i, j) in x.keys():
            if x[i, j].X > 0.5:
                assignments.append({"order": i, "slab": j})
        slabs_used = [j for j in range(num_slabs) if y[j].X > 0.5]
        objective_value = model.ObjVal
    else:
        assignments = []
        slabs_used = []
        objective_value = 0.0
    
    solution_dict = {
        "objective_value": objective_value,
        "assignments": assignments,
        "slabs_used": slabs_used
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == "__main__":
    main()