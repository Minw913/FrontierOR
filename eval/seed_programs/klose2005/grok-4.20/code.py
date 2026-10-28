import json
import argparse
import time
from typing import Any, Dict
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_customers = data['num_customers']
    num_plants = data['num_plants']
    demands = data['customers']['demands']
    fixed_costs = data['plants']['fixed_costs']
    capacities = data['plants']['capacities']
    shipping_costs = data['shipping_costs']

    # Model
    model = gp.Model("facility_location")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0

    # Variables
    y = model.addVars(num_plants, vtype=GRB.BINARY, name="y")
    x = model.addVars(num_customers, num_plants, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Objective: shipping + fixed
    shipping_expr = gp.quicksum(shipping_costs[i][j] * x[i, j] for i in range(num_customers) for j in range(num_plants))
    fixed_expr = gp.quicksum(fixed_costs[j] * y[j] for j in range(num_plants))
    model.setObjective(shipping_expr + fixed_expr, GRB.MINIMIZE)

    # Constraints
    # Demand satisfaction
    for i in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_plants)) == 1.0, name=f"demand_{i}")

    # Capacity
    for j in range(num_plants):
        model.addConstr(gp.quicksum(demands[i] * x[i, j] for i in range(num_customers)) <= capacities[j] * y[j],
                        name=f"capacity_{j}")

    # Callback for logging incumbents
    best_obj = float('inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        open_plants = [j for j in range(num_plants) if y[j].X > 0.5]
        assignments = {}
        for i in range(num_customers):
            cust_key = f"customer_{i}"
            assignments[cust_key] = {}
            for j in range(num_plants):
                val = x[i, j].X
                if val > 1e-8:
                    assignments[cust_key][f"plant_{j}"] = round(float(val), 6)

        objective_value = float(model.objVal)

        # Final log of best found
        if logger and objective_value < best_obj + 1e-5:
            logger.log(objective_value)
    else:
        # Infeasible is impossible per problem statement
        raise RuntimeError("No feasible solution found.")

    # Write solution
    solution = {
        "objective_value": objective_value,
        "open_plants": [int(j) for j in open_plants],
        "assignments": assignments
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()