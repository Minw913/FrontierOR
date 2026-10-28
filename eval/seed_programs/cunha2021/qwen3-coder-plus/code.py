import argparse
import json
import math
from itertools import product

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger
def solve_supply_chain(instance_path, solution_path, timelimit, log_path):
    # Load the instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_retailers = instance['num_retailers']
    num_warehouses = instance['num_warehouses']
    num_periods = instance['num_periods']

    # Assignment mappings (reverse of each other)
    retailer_to_warehouse = instance['retailer_warehouse_assignment']  # [retailer] -> warehouse
    warehouse_to_retailers = {w: [] for w in range(num_warehouses)}
    for r in range(num_retailers):
        w = retailer_to_warehouse[r]
        warehouse_to_retailers[w].append(r)
    
    demands = instance['demands']  # [retailer][period]
    fixed_cost_plant = instance['fixed_costs']['plant']  # [period]
    fixed_costs_warehouses = instance['fixed_costs']['warehouses']  # [warehouse][period]
    fixed_costs_retailers = instance['fixed_costs']['retailers']  # [retailer][period]
    holding_cost_plant = instance['holding_costs']['plant']
    holding_cost_warehouses = instance['holding_costs']['warehouses']
    holding_cost_retailers = instance['holding_costs']['retailers']  # [retailer]

    # Compute derived values
    # Demand per warehouse per period
    demand_per_warehouse = [[0 for t in range(num_periods)] for w in range(num_warehouses)]
    for r in range(num_retailers):
        for t in range(num_periods):
            w = retailer_to_warehouse[r]
            demand_per_warehouse[w][t] += demands[r][t]
    
    # Total remaining demand from period t onwards for each retailer/warehouse/facility
    total_remaining_demand_retailer = [[0 for t in range(num_periods)] for r in range(num_retailers)]
    total_remaining_demand_warehouse = [[0 for t in range(num_periods)] for w in range(num_warehouses)]
    total_remaining_demand_plant = [0] * num_periods

    # Calculate retailer demands
    for r in range(num_retailers):
        cumsum = 0
        for t in range(num_periods - 1, -1, -1):
            cumsum += demands[r][t]
            total_remaining_demand_retailer[r][t] = cumsum
    
    # Calculate warehouse demands
    for w in range(num_warehouses):
        cumsum = 0
        for t in range(num_periods - 1, -1, -1):
            cumsum += demand_per_warehouse[w][t]
            total_remaining_demand_warehouse[w][t] = cumsum

    # Calculate total plant demand
    cumsum = 0
    for t in range(num_periods - 1, -1, -1):
        cumsum += sum(demands[r][t] for r in range(num_retailers))
        total_remaining_demand_plant[t] = cumsum

    # Start building the model
    model = gp.Model("supply_chain")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', timelimit)
    model.setParam('LogFile', '')  # Disable logging for faster performance if needed, but usually good to keep it

    # Variables: x_ijk (quantity from i to j in k), s_ik (inventory at i in k), y_ik (setup binary at i in k)
    # x vars: production at plant, shipments plant->warehouse, shipments warehouse->retailer
    x_plant = model.addVars(num_periods, lb=0, name="x_plant")
    x_warehouses = model.addVars(num_warehouses, num_periods, lb=0, name="x_warehouses")
    x_retailers = model.addVars(num_retailers, num_periods, lb=0, name="x_retailers")

    # s vars: inventory at facilities
    s_plant = model.addVars(num_periods, lb=0, name="s_plant")
    s_warehouses = model.addVars(num_warehouses, num_periods, lb=0, name="s_warehouses")
    s_retailers = model.addVars(num_retailers, num_periods, lb=0, name="s_retailers")

    # y vars: binary setup variables
    y_plant = model.addVars(num_periods, vtype=GRB.BINARY, name="y_plant")
    y_warehouses = model.addVars(num_warehouses, num_periods, vtype=GRB.BINARY, name="y_warehouses")
    y_retailers = model.addVars(num_retailers, num_periods, vtype=GRB.BINARY, name="y_retailers")

    # Constraints
    # Plant inventory flow: init_inv + prod - sum(warehouse_shipments) = end_inv
    for t in range(num_periods):
        total_wh_inflow = sum(x_warehouses[w, t] for w in range(num_warehouses))
        prev_inv = s_plant[t-1] if t > 0 else 0
        model.addConstr(
            prev_inv + x_plant[t] - total_wh_inflow == s_plant[t],
            name=f"plant_flow_t{t}"
        )

    # Warehouse inventory flow: for each warehouse w
    for w in range(num_warehouses):
        for t in range(num_periods):
            total_ret_outflow = sum(x_retailers[r, t] for r in warehouse_to_retailers[w])
            prev_inv = s_warehouses[w, t-1] if t > 0 else 0
            model.addConstr(
                prev_inv + x_warehouses[w, t] - total_ret_outflow == s_warehouses[w, t],
                name=f"wh_flow_w{w}_t{t}"
            )

    # Retailer inventory flow: for each retailer r
    for r in range(num_retailers):
        for t in range(num_periods):
            prev_inv = s_retailers[r, t-1] if t > 0 else 0
            demand_r_t = demands[r][t]
            model.addConstr(
                prev_inv + x_retailers[r, t] - demand_r_t == s_retailers[r, t],
                name=f"ret_flow_r{r}_t{t}"
            )

    # Setup constraints: Only ship if facility is setup
    BigM = max(max(total_remaining_demand_plant), 
               max(max(dw) for dw in total_remaining_demand_warehouse),
               max(max(dr) for dr in total_remaining_demand_retailer))

    for t in range(num_periods):
        model.addConstr(x_plant[t] <= BigM * y_plant[t], name=f"setup_plant_t{t}")
        
    for w in range(num_warehouses):
        for t in range(num_periods):
            model.addConstr(x_warehouses[w, t] <= total_remaining_demand_warehouse[w][t] * y_warehouses[w, t], 
                            name=f"setup_wh_{w}_t{t}")

    for r in range(num_retailers):
        for t in range(num_periods):
            model.addConstr(x_retailers[r, t] <= total_remaining_demand_retailer[r][t] * y_retailers[r, t],
                            name=f"setup_ret_{r}_t{t}")


    # Objective function
    fixed_cost_expr = gp.quicksum(
        fixed_cost_plant[t] * y_plant[t] for t in range(num_periods)
    )
    fixed_cost_expr.add(gp.quicksum(
        fixed_costs_warehouses[w][t] * y_warehouses[w, t] for w in range(num_warehouses) for t in range(num_periods)
    ))
    fixed_cost_expr.add(gp.quicksum(
        fixed_costs_retailers[r][t] * y_retailers[r, t] for r in range(num_retailers) for t in range(num_periods)
    ))

    holding_cost_expr = gp.quicksum(
        holding_cost_plant * s_plant[t] for t in range(num_periods)
    )
    holding_cost_expr.add(gp.quicksum(
        holding_cost_warehouses * s_warehouses[w, t] for w in range(num_warehouses) for t in range(num_periods)
    ))
    holding_cost_expr.add(gp.quicksum(
        holding_cost_retailers[r] * s_retailers[r, t] for r in range(num_retailers) for t in range(num_periods)
    ))

    model.setObjective(fixed_cost_expr + holding_cost_expr, GRB.MINIMIZE)

    # Prepare logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def log_solution():
        if logger:
            obj_val = model.ObjVal if model.status == GRB.OPTIMAL else model.objVal
            logger.log(obj_val)

    model.optimize(callback=lambda m, where: log_solution() if where == GRB.Callback.MIPSOL else None)

    # Extract the solution regardless of status
    # Initialize solution dict
    solution = {
        "objective_value": round(model.ObjVal) if model.SolCount > 0 else float('inf'),
        "x_plant": [round(x_plant[t].X) for t in range(num_periods)],
        "s_plant": [round(s_plant[t].X) for t in range(num_periods)],
        "y_plant": [int(y_plant[t].X) for t in range(num_periods)],
        "x_warehouses": [
            [round(x_warehouses[w, t].X) for t in range(num_periods)]
            for w in range(num_warehouses)
        ],
        "s_warehouses": [
            [round(s_warehouses[w, t].X) for t in range(num_periods)]
            for w in range(num_warehouses)
        ],
        "y_warehouses": [
            [int(y_warehouses[w, t].X) for t in range(num_periods)]
            for w in range(num_warehouses)
        ],
        "x_retailers": [
            [round(x_retailers[r, t].X) for t in range(num_periods)]
            for r in range(num_retailers)
        ],
        "s_retailers": [
            [round(s_retailers[r, t].X) for t in range(num_periods)]
            for r in range(num_retailers)
        ],
        "y_retailers": [
            [int(y_retailers[r, t].X) for t in range(num_periods)]
            for r in range(num_retailers)
        ]
    }

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve_supply_chain(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()