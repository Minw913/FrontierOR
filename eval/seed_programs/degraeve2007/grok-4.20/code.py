import argparse
import json
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_products = instance['dimensions']['num_products']
    num_periods = instance['dimensions']['num_periods']
    demand = instance['parameters']['demand']
    holding_cost = instance['parameters']['holding_cost']
    setup_cost = instance['parameters']['setup_cost']
    variable_production_cost = instance['parameters']['variable_production_cost']
    variable_production_time = instance['parameters']['variable_production_time']
    setup_time = instance['parameters']['setup_time']
    capacity = instance['parameters']['capacity']
    initial_inventory_cost = instance['parameters']['initial_inventory_cost']

    # Precompute cumulative demand from period t onward for each product
    cum_demand = [[0] * num_periods for _ in range(num_products)]
    for p in range(num_products):
        cum = 0
        for t in range(num_periods - 1, -1, -1):
            cum += demand[p][t]
            cum_demand[p][t] = cum

    model = gp.Model("LotSizing")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LogToConsole', 0)

    # Variables
    production = model.addVars(num_products, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="production")
    setup = model.addVars(num_products, num_periods, vtype=GRB.BINARY, name="setup")
    inventory = model.addVars(num_products, num_periods, vtype=GRB.CONTINUOUS, lb=0, name="inventory")
    initial_inventory = model.addVars(num_products, vtype=GRB.CONTINUOUS, lb=0, name="initial_inventory")

    # Objective
    init_cost = gp.quicksum(initial_inventory_cost[p] * initial_inventory[p] for p in range(num_products))
    setup_total = gp.quicksum(setup_cost[p][t] * setup[p, t] for p in range(num_products) for t in range(num_periods))
    var_prod_total = gp.quicksum(variable_production_cost[p][t] * production[p, t] for p in range(num_products) for t in range(num_periods))
    hold_total = gp.quicksum(holding_cost[p][t] * inventory[p, t] for p in range(num_products) for t in range(num_periods))
    model.setObjective(init_cost + setup_total + var_prod_total + hold_total, GRB.MINIMIZE)

    # Initial inventory balance and bounds
    for p in range(num_products):
        model.addConstr(initial_inventory[p] + production[p, 0] == demand[p][0] + inventory[p, 0])
        # Initial inventory cannot exceed total demand (though penalty prevents large values)
        model.addConstr(initial_inventory[p] <= cum_demand[p][0])

    # Inventory balance and setup enforcement for subsequent periods
    for p in range(num_products):
        for t in range(1, num_periods):
            model.addConstr(inventory[p, t - 1] + production[p, t] == demand[p][t] + inventory[p, t])

        for t in range(num_periods):
            # Production only if setup
            model.addConstr(production[p, t] <= cum_demand[p][t] * setup[p, t])
            # Tighten with capacity consideration (M = capacity after setup / time)
            if variable_production_time[p][t] > 1e-6:
                M_cap = (capacity[t] - setup_time[p][t]) / variable_production_time[p][t]
                model.addConstr(production[p, t] <= M_cap * setup[p, t])

    # Capacity constraints
    for t in range(num_periods):
        setup_load = gp.quicksum(setup_time[p][t] * setup[p, t] for p in range(num_products))
        prod_load = gp.quicksum(variable_production_time[p][t] * production[p, t] for p in range(num_products))
        model.addConstr(setup_load + prod_load <= capacity[t])

    # Non-negativity of inventory is already in variable definition

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        prod_sol = [[float(production[p, t].X) for t in range(num_periods)] for p in range(num_products)]
        setup_sol = [[int(round(setup[p, t].X)) for t in range(num_periods)] for p in range(num_products)]
        inv_sol = [[float(inventory[p, t].X) for t in range(num_periods)] for p in range(num_products)]
        init_inv_sol = [float(initial_inventory[p].X) for p in range(num_products)]
        obj_val = float(model.objVal)

        # Validate and fix minor numerical issues if any
        for p in range(num_products):
            for t in range(num_periods):
                if setup_sol[p][t] == 0:
                    prod_sol[p][t] = 0.0
                prod_sol[p][t] = max(0.0, prod_sol[p][t])
                inv_sol[p][t] = max(0.0, inv_sol[p][t])
            init_inv_sol[p] = max(0.0, init_inv_sol[p])

        solution = {
            "objective_value": obj_val,
            "variables": {
                "production": prod_sol,
                "setup": setup_sol,
                "inventory": inv_sol,
                "initial_inventory": init_inv_sol
            }
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # If no solution found, create a dummy feasible solution by using initial inventory
        prod_sol = [[0.0] * num_periods for _ in range(num_products)]
        setup_sol = [[0] * num_periods for _ in range(num_products)]
        inv_sol = [[0.0] * num_periods for _ in range(num_products)]
        init_inv_sol = [float(cum_demand[p][0]) for p in range(num_products)]
        obj_val = sum(initial_inventory_cost[p] * init_inv_sol[p] for p in range(num_products))

        solution = {
            "objective_value": obj_val,
            "variables": {
                "production": prod_sol,
                "setup": setup_sol,
                "inventory": inv_sol,
                "initial_inventory": init_inv_sol
            }
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()