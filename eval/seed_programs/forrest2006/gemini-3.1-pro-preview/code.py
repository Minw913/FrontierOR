import argparse
import json
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve Steel Mill Setup Optimization Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default="", help="Path to log intermediate solutions")
    args = parser.parse_args()

    # Load problem instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_slabs = data["num_slabs"]
    num_orders = data["num_orders"]
    max_colors = data["max_colors_per_knapsack"]
    slab_weights = data["slab_weights"]
    order_weights = data["order_weights"]
    order_colors = data["order_colors"]

    # Pre-process eligibilities
    # Retain eligibilities strictly if the order isn't heavier than the slab
    V = {j: [] for j in range(num_slabs)}
    E = {i: [] for i in range(num_orders)}
    
    for i, slabs in enumerate(data["eligible_slabs_per_order"]):
        w_i = order_weights[i]
        for j in set(slabs): # Use set to avoid duplicates just in case
            if j < num_slabs and w_i <= slab_weights[j]:
                V[j].append(i)
                E[i].append(j)

    # Establish which colors are present among eligible orders for each slab
    C = {}
    for j in range(num_slabs):
        C[j] = set(order_colors[i] for i in V[j])

    # Initialize Gurobi model
    model = gp.Model("SteelMill")
    model.setParam("OutputFlag", 1)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)

    x = {} # x[i, j] = 1 if order i assigned to slab j
    y = {} # y[j] = 1 if slab j is used
    z = {} # z[j, k] = 1 if color k is assigned to slab j

    # Create variables
    for j in range(num_slabs):
        if not V[j]:
            continue
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
        for k in C[j]:
            z[j, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")
        for i in V[j]:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Objective: Maximize 2 * (assigned weights) - (opened slab weights)
    obj_expr = gp.LinExpr()
    for j in range(num_slabs):
        if not V[j]:
            continue
        obj_expr -= slab_weights[j] * y[j]
        for i in V[j]:
            obj_expr += 2.0 * order_weights[i] * x[i, j]
            
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Constraints
    
    # 1. Each order can be assigned at most once
    for i in range(num_orders):
        if E[i]:
            model.addConstr(gp.quicksum(x[i, j] for j in E[i]) <= 1, name=f"order_{i}")

    # 2. Capacity, color dependencies, and structural constraints per slab
    for j in range(num_slabs):
        if not V[j]:
            continue
            
        # Capacity Constraint
        weight_expr = gp.quicksum(order_weights[i] * x[i, j] for i in V[j])
        model.addConstr(weight_expr <= slab_weights[j] * y[j], name=f"cap_{j}")
        
        # Valid Inequality: Opening a slab is strictly beneficial if ordered weights offset the block's cost
        # Therefore, we enforce minimum fill conditions essentially halving unneeded combinations.
        model.addConstr(weight_expr >= (slab_weights[j] / 2.0) * y[j], name=f"min_val_{j}")

        # Limit distinct colors on slab
        actual_max_colors = min(max_colors, len(C[j]))
        model.addConstr(gp.quicksum(z[j, k] for k in C[j]) <= actual_max_colors * y[j], name=f"max_colors_{j}")

        # Consistency bounds linking variables structurally
        for k in C[j]:
            model.addConstr(z[j, k] <= y[j], name=f"z_y_{j}_{k}")
            
        for i in V[j]:
            model.addConstr(x[i, j] <= z[j, order_colors[i]], name=f"color_{i}_{j}")
            
    # Setup optional Logger callback
    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="maximize")

    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger is not None:
                    logger.log(obj_val)
            except Exception:
                pass

    # Solve the problem
    if logger is not None:
        model.optimize(log_callback)
    else:
        model.optimize()

    # Extract Solution
    assignments = []
    slabs_used = []
    real_obj = 0.0

    if model.SolCount > 0:
        for j in range(num_slabs):
            if j in y and y[j].X > 0.5:
                slabs_used.append(j)
                
        for j in range(num_slabs):
            if not V[j]:
                continue
            for i in V[j]:
                if x[i, j].X > 0.5:
                    assignments.append({
                        "order": i,
                        "slab": j
                    })

        # Recalculate precision objective
        for j in slabs_used:
            real_obj -= slab_weights[j]
        for asn in assignments:
            real_obj += 2.0 * order_weights[asn["order"]]

    # Construct and dump output solution 
    solution = {
        "objective_value": float(real_obj),
        "assignments": assignments,
        "slabs_used": slabs_used
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()