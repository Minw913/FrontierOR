import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    num_retailers = inst["num_retailers"]
    num_warehouses = inst["num_warehouses"]
    num_periods = inst["num_periods"]
    
    r_assign = inst["retailer_warehouse_assignment"]
    w_sets = {}
    for w_idx_str, r_list in inst["warehouse_retailer_sets"].items():
        w_sets[int(w_idx_str)] = r_list
        
    demands = inst["demands"]
    fixed_costs = inst["fixed_costs"]
    holding_costs = inst["holding_costs"]

    # Precompute remaining demands
    D_r = [[0] * num_periods for _ in range(num_retailers)]
    D_w = [[0] * num_periods for _ in range(num_warehouses)]
    D_p = [0] * num_periods

    for r in range(num_retailers):
        for t in range(num_periods - 1, -1, -1):
            D_r[r][t] = demands[r][t] + (D_r[r][t + 1] if t + 1 < num_periods else 0)

    for w in range(num_warehouses):
        for t in range(num_periods - 1, -1, -1):
            D_w[w][t] = sum(D_r[r][t] for r in w_sets[w])

    for t in range(num_periods - 1, -1, -1):
        D_p[t] = sum(D_w[w][t] for w in range(num_warehouses))

    model = gp.Model("supply_chain")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # Variables
    x_p = model.addVars(num_periods, vtype=GRB.CONTINUOUS, name="x_p")
    s_p = model.addVars(num_periods, vtype=GRB.CONTINUOUS, name="s_p")
    y_p = model.addVars(num_periods, vtype=GRB.BINARY, name="y_p")

    x_w = model.addVars(num_warehouses, num_periods, vtype=GRB.CONTINUOUS, name="x_w")
    s_w = model.addVars(num_warehouses, num_periods, vtype=GRB.CONTINUOUS, name="s_w")
    y_w = model.addVars(num_warehouses, num_periods, vtype=GRB.BINARY, name="y_w")

    x_r = model.addVars(num_retailers, num_periods, vtype=GRB.CONTINUOUS, name="x_r")
    s_r = model.addVars(num_retailers, num_periods, vtype=GRB.CONTINUOUS, name="s_r")
    y_r = model.addVars(num_retailers, num_periods, vtype=GRB.BINARY, name="y_r")

    # Constraints
    for t in range(num_periods):
        # Plant flow
        if t == 0:
            model.addConstr(x_p[t] == sum(x_w[w, t] for w in range(num_warehouses)) + s_p[t])
        else:
            model.addConstr(s_p[t - 1] + x_p[t] == sum(x_w[w, t] for w in range(num_warehouses)) + s_p[t])

        # Bounds for plant
        if D_p[t] > 0:
            model.addConstr(x_p[t] <= D_p[t] * y_p[t])
        else:
            model.addConstr(x_p[t] == 0)

    for w in range(num_warehouses):
        for t in range(num_periods):
            if t == 0:
                model.addConstr(x_w[w, t] == sum(x_r[r, t] for r in w_sets[w]) + s_w[w, t])
            else:
                model.addConstr(s_w[w, t - 1] + x_w[w, t] == sum(x_r[r, t] for r in w_sets[w]) + s_w[w, t])
            
            # Bounds for warehouse
            if D_w[w][t] > 0:
                model.addConstr(x_w[w, t] <= D_w[w][t] * y_w[w, t])
            else:
                model.addConstr(x_w[w, t] == 0)

    for r in range(num_retailers):
        for t in range(num_periods):
            if t == 0:
                model.addConstr(x_r[r, t] == demands[r][t] + s_r[r, t])
            else:
                model.addConstr(s_r[r, t - 1] + x_r[r, t] == demands[r][t] + s_r[r, t])

            # Bounds for retailer
            if D_r[r][t] > 0:
                model.addConstr(x_r[r, t] <= D_r[r][t] * y_r[r, t])
            else:
                model.addConstr(x_r[r, t] == 0)

    # Objective
    obj = gp.LinExpr()
    for t in range(num_periods):
        obj += fixed_costs["plant"][t] * y_p[t]
        obj += holding_costs["plant"] * s_p[t]
        for w in range(num_warehouses):
            obj += fixed_costs["warehouses"][w][t] * y_w[w, t]
            obj += holding_costs["warehouses"] * s_w[w, t]
        for r in range(num_retailers):
            obj += fixed_costs["retailers"][r][t] * y_r[r, t]
            obj += holding_costs["retailers"][r] * s_r[r, t]

    model.setObjective(obj, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = {
                "objective_value": obj_val,
                "x_plant": [model.cbGetSolution(x_p[t]) for t in range(num_periods)],
                "s_plant": [model.cbGetSolution(s_p[t]) for t in range(num_periods)],
                "y_plant": [int(round(model.cbGetSolution(y_p[t]))) for t in range(num_periods)],
                "x_warehouses": [[model.cbGetSolution(x_w[w, t]) for t in range(num_periods)] for w in range(num_warehouses)],
                "s_warehouses": [[model.cbGetSolution(s_w[w, t]) for t in range(num_periods)] for w in range(num_warehouses)],
                "y_warehouses": [[int(round(model.cbGetSolution(y_w[w, t]))) for t in range(num_periods)] for w in range(num_warehouses)],
                "x_retailers": [[model.cbGetSolution(x_r[r, t]) for t in range(num_periods)] for r in range(num_retailers)],
                "s_retailers": [[model.cbGetSolution(s_r[r, t]) for t in range(num_periods)] for r in range(num_retailers)],
                "y_retailers": [[int(round(model.cbGetSolution(y_r[r, t]))) for t in range(num_periods)] for r in range(num_retailers)],
            }
            if logger:
                logger.log_solution(obj_val, sol)

    model.optimize(callback=callback)

    # Extract final solution
    sol = {}
    if model.SolCount > 0:
        obj_val = model.ObjVal
        sol["objective_value"] = float(obj_val)
        sol["x_plant"] = [float(x_p[t].X) for t in range(num_periods)]
        sol["s_plant"] = [float(s_p[t].X) for t in range(num_periods)]
        sol["y_plant"] = [int(round(y_p[t].X)) for t in range(num_periods)]
        sol["x_warehouses"] = [[float(x_w[w, t].X) for t in range(num_periods)] for w in range(num_warehouses)]
        sol["s_warehouses"] = [[float(s_w[w, t].X) for t in range(num_periods)] for w in range(num_warehouses)]
        sol["y_warehouses"] = [[int(round(y_w[w, t].X)) for t in range(num_periods)] for w in range(num_warehouses)]
        sol["x_retailers"] = [[float(x_r[r, t].X) for t in range(num_periods)] for r in range(num_retailers)]
        sol["s_retailers"] = [[float(s_r[r, t].X) for t in range(num_periods)] for r in range(num_retailers)]
        sol["y_retailers"] = [[int(round(y_r[r, t].X)) for t in range(num_periods)] for r in range(num_retailers)]
    else:
        sol["objective_value"] = 0.0
        sol["x_plant"] = [0.0] * num_periods
        sol["s_plant"] = [0.0] * num_periods
        sol["y_plant"] = [0] * num_periods
        sol["x_warehouses"] = [[0.0] * num_periods for _ in range(num_warehouses)]
        sol["s_warehouses"] = [[0.0] * num_periods for _ in range(num_warehouses)]
        sol["y_warehouses"] = [[0] * num_periods for _ in range(num_warehouses)]
        sol["x_retailers"] = [[0.0] * num_periods for _ in range(num_retailers)]
        sol["s_retailers"] = [[0.0] * num_periods for _ in range(num_retailers)]
        sol["y_retailers"] = [[0] * num_periods for _ in range(num_retailers)]

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()