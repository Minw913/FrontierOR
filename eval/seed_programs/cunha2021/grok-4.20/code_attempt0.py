import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_retailers = data['num_retailers']
    num_warehouses = data['num_warehouses']
    num_periods = data['num_periods']
    retailer_warehouse_assignment = data['retailer_warehouse_assignment']
    demands = data['demands']
    fixed_costs = data['fixed_costs']
    holding_costs = data['holding_costs']

    # Precompute cumulative demands (remaining demand from t onward)
    # For retailers
    cum_demand_ret = [[[0 for _ in range(num_periods + 1)] for _ in range(num_retailers)]]
    cum_demand_ret = [[0] * (num_periods + 1) for _ in range(num_retailers)]
    for r in range(num_retailers):
        for t in range(num_periods - 1, -1, -1):
            cum_demand_ret[r][t] = cum_demand_ret[r][t + 1] + demands[r][t]

    # For warehouses
    cum_demand_wh = [[0] * (num_periods + 1) for _ in range(num_warehouses)]
    for w in range(num_warehouses):
        for t in range(num_periods - 1, -1, -1):
            total = 0
            for r in data['warehouse_retailer_sets'][str(w)]:
                total += demands[r][t]
            cum_demand_wh[w][t] = cum_demand_wh[w][t + 1] + total

    # For plant
    cum_demand_plant = [0] * (num_periods + 1)
    for t in range(num_periods - 1, -1, -1):
        total = 0
        for r in range(num_retailers):
            total += demands[r][t]
        cum_demand_plant[t] = cum_demand_plant[t + 1] + total

    # Create model
    model = gp.Model("SupplyChain")
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01  # Allow 1% gap for efficiency

    # Variables
    # Plant
    x_plant = model.addVars(num_periods, vtype=GRB.CONTINUOUS, lb=0, name="x_plant")
    s_plant = model.addVars(num_periods, vtype=GRB.CONTINUOUS, lb=0, name="s_plant")
    y_plant = model.addVars(num_periods, vtype=GRB.BINARY, name="y_plant")

    # Warehouses
    x_wh = model.addVars(num_warehouses, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="x_wh")
    s_wh = model.addVars(num_warehouses, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="s_wh")
    y_wh = model.addVars(num_warehouses, num_periods, vtype=GRB.BINARY, name="y_wh")

    # Retailers
    x_ret = model.addVars(num_retailers, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="x_ret")
    s_ret = model.addVars(num_retailers, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="s_ret")
    y_ret = model.addVars(num_retailers, num_periods, vtype=GRB.BINARY, name="y_ret")

    # Objective: fixed costs + holding costs
    obj = gp.quicksum(fixed_costs['plant'][t] * y_plant[t] for t in range(num_periods))
    obj += gp.quicksum(fixed_costs['warehouses'][w][t] * y_wh[w, t] for w in range(num_warehouses) for t in range(num_periods))
    obj += gp.quicksum(fixed_costs['retailers'][r][t] * y_ret[r, t] for r in range(num_retailers) for t in range(num_periods))
    obj += holding_costs['plant'] * gp.quicksum(s_plant[t] for t in range(num_periods))
    obj += holding_costs['warehouses'] * gp.quicksum(s_wh[w, t] for w in range(num_warehouses) for t in range(num_periods))
    for r in range(num_retailers):
        obj += holding_costs['retailers'][r] * gp.quicksum(s_ret[r, t] for t in range(num_periods))

    model.setObjective(obj, GRB.MINIMIZE)

    # Inventory balance and setup constraints
    M_plant = max(cum_demand_plant) + 1
    M_wh = [max(cum_demand_wh[w]) + 1 for w in range(num_warehouses)]
    M_ret = [[max(cum_demand_ret[r]) + 1 for _ in range(num_periods)] for r in range(num_retailers)]

    # Plant balance
    for t in range(num_periods):
        inflow = x_plant[t]
        outflow = gp.quicksum(x_wh[w, t] for w in range(num_warehouses))
        if t == 0:
            model.addConstr(s_plant[t] == inflow - outflow)
        else:
            model.addConstr(s_plant[t] == s_plant[t-1] + inflow - outflow)
        model.addConstr(x_plant[t] <= M_plant * y_plant[t])
        model.addConstr(x_plant[t] <= cum_demand_plant[t])

    # Warehouse balance
    for w in range(num_warehouses):
        for t in range(num_periods):
            inflow = x_wh[w, t]
            outflow = gp.quicksum(x_ret[r, t] for r in data['warehouse_retailer_sets'][str(w)])
            if t == 0:
                model.addConstr(s_wh[w, t] == inflow - outflow)
            else:
                model.addConstr(s_wh[w, t] == s_wh[w, t-1] + inflow - outflow)
            model.addConstr(x_wh[w, t] <= M_wh[w] * y_wh[w, t])
            model.addConstr(x_wh[w, t] <= cum_demand_wh[w][t])

    # Retailer balance
    for r in range(num_retailers):
        w = retailer_warehouse_assignment[r]
        for t in range(num_periods):
            inflow = x_ret[r, t]
            if t == 0:
                model.addConstr(s_ret[r, t] == inflow - demands[r][t])
            else:
                model.addConstr(s_ret[r, t] == s_ret[r, t-1] + inflow - demands[r][t])
            model.addConstr(x_ret[r, t] <= cum_demand_ret[r][t] * y_ret[r, t])
            model.addConstr(x_ret[r, t] <= cum_demand_ret[r][t])

    # Non-negative inventory at end (implicit by lb=0)

    # Callback for logging improved solutions
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    incumbent_value = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < incumbent_value[0] - 1e-6:
                incumbent_value[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        # Get solution values
        x_plant_sol = [x_plant[t].X for t in range(num_periods)]
        s_plant_sol = [s_plant[t].X for t in range(num_periods)]
        y_plant_sol = [int(y_plant[t].X + 0.5) for t in range(num_periods)]

        x_wh_sol = [[x_wh[w, t].X for t in range(num_periods)] for w in range(num_warehouses)]
        s_wh_sol = [[s_wh[w, t].X for t in range(num_periods)] for w in range(num_warehouses)]
        y_wh_sol = [[int(y_wh[w, t].X + 0.5) for t in range(num_periods)] for w in range(num_warehouses)]

        x_ret_sol = [[x_ret[r, t].X for t in range(num_periods)] for r in range(num_retailers)]
        s_ret_sol = [[s_ret[r, t].X for t in range(num_periods)] for r in range(num_retailers)]
        y_ret_sol = [[int(y_ret[r, t].X + 0.5) for t in range(num_periods)] for r in range(num_retailers)]

        objective_value = model.objVal
    else:
        # Infeasible - return zeros (should not happen for this problem)
        x_plant_sol = [0.0] * num_periods
        s_plant_sol = [0.0] * num_periods
        y_plant_sol = [0] * num_periods
        x_wh_sol = [[0.0] * num_periods for _ in range(num_warehouses)]
        s_wh_sol = [[0.0] * num_periods for _ in range(num_warehouses)]
        y_wh_sol = [[0] * num_periods for _ in range(num_warehouses)]
        x_ret_sol = [[0.0] * num_periods for _ in range(num_retailers)]
        s_ret_sol = [[0.0] * num_periods for _ in range(num_retailers)]
        y_ret_sol = [[0] * num_periods for _ in range(num_retailers)]
        objective_value = 0.0

    solution = {
        "objective_value": float(objective_value),
        "x_plant": [float(x) for x in x_plant_sol],
        "s_plant": [float(x) for x in s_plant_sol],
        "y_plant": y_plant_sol,
        "x_warehouses": [[float(x) for x in row] for row in x_wh_sol],
        "s_warehouses": [[float(x) for x in row] for row in s_wh_sol],
        "y_warehouses": y_wh_sol,
        "x_retailers": [[float(x) for x in row] for row in x_ret_sol],
        "s_retailers": [[float(x) for x in row] for row in s_ret_sol],
        "y_retailers": y_ret_sol
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()