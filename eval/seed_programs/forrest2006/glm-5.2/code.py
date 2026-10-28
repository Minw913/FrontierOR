import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_slabs = data["num_slabs"]
    num_orders = data["num_orders"]
    max_colors = data["max_colors_per_knapsack"]
    slab_weights = data["slab_weights"]
    order_weights = data["order_weights"]
    order_colors = data["order_colors"]
    eligible_slabs_per_order = data["eligible_slabs_per_order"]

    # Precompute eligible orders per slab
    eligible_orders_per_slab = [[] for _ in range(num_slabs)]
    for j, slabs in enumerate(eligible_slabs_per_order):
        for i in slabs:
            eligible_orders_per_slab[i].append(j)
    
    # Precompute eligible colors per slab
    eligible_colors_per_slab = [set() for _ in range(num_slabs)]
    for i in range(num_slabs):
        for j in eligible_orders_per_slab[i]:
            eligible_colors_per_slab[i].add(order_colors[j])

    # Initialize SolutionLogger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except ImportError:
        logger = None

    model = gp.Model("slab_assignment")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # Variables
    x = {}
    for i in range(num_slabs):
        for j in eligible_orders_per_slab[i]:
            x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    y = model.addVars(num_slabs, vtype=GRB.BINARY, name="y")

    z = {}
    for i in range(num_slabs):
        for c in eligible_colors_per_slab[i]:
            z[(i, c)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{c}")

    # Objective
    obj = gp.LinExpr()
    for i in range(num_slabs):
        for j in eligible_orders_per_slab[i]:
            obj += 2 * order_weights[j] * x[(i, j)]
        obj -= slab_weights[i] * y[i]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    # 1. Each order assigned to at most 1 slab
    for j in range(num_orders):
        model.addConstr(gp.quicksum(x[(i, j)] for i in eligible_slabs_per_order[j]) <= 1, name=f"assign_{j}")

    # 2. Slab capacity and activation
    for i in range(num_slabs):
        model.addConstr(gp.quicksum(order_weights[j] * x[(i, j)] for j in eligible_orders_per_slab[i]) <= slab_weights[i] * y[i], name=f"cap_{i}")
        for j in eligible_orders_per_slab[i]:
            model.addConstr(x[(i, j)] <= y[i], name=f"link_{i}_{j}")

    # 3. Color constraints
    for i in range(num_slabs):
        model.addConstr(gp.quicksum(z[(i, c)] for c in eligible_colors_per_slab[i]) <= max_colors, name=f"color_limit_{i}")
        for j in eligible_orders_per_slab[i]:
            c = order_colors[j]
            model.addConstr(x[(i, j)] <= z[(i, c)], name=f"color_link_{i}_{j}_{c}")

    x_list = list(x.values())
    x_keys = list(x.keys())
    
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val_cb = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            cb_assignments = []
            cb_slabs_used = []
            
            y_vals = model.cbGetSolution(y)
            for i in range(num_slabs):
                if y_vals[i] > 0.5:
                    cb_slabs_used.append(i)
                    
            if x_list:
                x_vals = model.cbGetSolution(x_list)
                for idx, (i, j) in enumerate(x_keys):
                    if x_vals[idx] > 0.5:
                        cb_assignments.append({"order": j, "slab": i})
                        
            sol_dict = {
                "objective_value": obj_val_cb,
                "assignments": cb_assignments,
                "slabs_used": cb_slabs_used
            }
            if logger:
                logger.log_solution(obj_val_cb, sol_dict)

    model.optimize(my_callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        assignments = []
        slabs_used = []
        for i in range(num_slabs):
            if y[i].X > 0.5:
                slabs_used.append(i)
        for (i, j) in x.keys():
            if x[(i, j)].X > 0.5:
                assignments.append({"order": j, "slab": i})
    else:
        obj_val = 0
        assignments = []
        slabs_used = []

    solution = {
        "objective_value": obj_val,
        "assignments": assignments,
        "slabs_used": slabs_used
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()