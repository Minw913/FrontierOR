import argparse
import json
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
        instance = json.load(f)

    num_facilities = instance['parameters']['num_facilities']
    num_customers = instance['parameters']['num_customers']
    num_periods = instance['parameters']['num_periods']
    seasonal_factors = instance['seasonal_factors']
    demands = instance['demands']
    holding_costs = instance['holding_costs']
    capacities = instance['capacities']
    transportation_costs = instance['transportation_costs']

    # Precompute customer demand per period
    demand = [[demands[c] * seasonal_factors[t] for t in range(num_periods)] for c in range(num_customers)]

    # Model
    model = gp.Model("LogisticsNetwork")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01
    model.Params.LogToConsole = 0

    # Variables
    assign = model.addVars(num_facilities, num_customers, vtype=GRB.BINARY, name="assign")
    inventory = model.addVars(num_facilities, num_periods, lb=0, name="inventory")
    production = model.addVars(num_facilities, num_periods, lb=0, name="production")

    # Objective: transportation + holding
    trans_cost = gp.quicksum(
        transportation_costs[f][c][t] * demand[c][t] * assign[f, c]
        for f in range(num_facilities)
        for c in range(num_customers)
        for t in range(num_periods)
    )
    hold_cost = gp.quicksum(
        holding_costs[f][t] * inventory[f, t]
        for f in range(num_facilities)
        for t in range(num_periods)
    )
    model.setObjective(trans_cost + hold_cost, GRB.MINIMIZE)

    # Assignment constraints
    for c in range(num_customers):
        model.addConstr(gp.quicksum(assign[f, c] for f in range(num_facilities)) == 1)

    # Inventory balance and capacity constraints
    for f in range(num_facilities):
        # Period 0
        model.addConstr(
            production[f, 0] == gp.quicksum(demand[c][0] * assign[f, c] for c in range(num_customers)) + inventory[f, 0]
        )
        model.addConstr(inventory[f, 0] >= 0)
        model.addConstr(production[f, 0] <= capacities[f][0])

        for t in range(1, num_periods):
            model.addConstr(
                production[f, t] + inventory[f, t-1] ==
                gp.quicksum(demand[c][t] * assign[f, c] for c in range(num_customers)) + inventory[f, t]
            )
            model.addConstr(production[f, t] <= capacities[f][t])
            model.addConstr(inventory[f, t] >= 0)

    # Callback for logging incumbents
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_obj = float('inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-6:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        assignment = {}
        for c in range(num_customers):
            for f in range(num_facilities):
                if assign[f, c].X > 0.5:
                    assignment[str(c)] = int(f)
                    break

        inventory_sol = {}
        for f in range(num_facilities):
            inventory_sol[str(f)] = [float(inventory[f, t].X) for t in range(num_periods)]

        objective_value = float(model.ObjVal)

        solution = {
            "objective_value": objective_value,
            "assignment": assignment,
            "inventory": inventory_sol
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible case - should not happen for this problem
        solution = {
            "objective_value": 0.0,
            "assignment": {str(c): 0 for c in range(num_customers)},
            "inventory": {str(f): [0.0] * num_periods for f in range(num_facilities)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()