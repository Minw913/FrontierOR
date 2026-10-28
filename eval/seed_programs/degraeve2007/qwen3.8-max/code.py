import argparse
import json
import time
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

    # Read instance
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

    # Precompute cumulative demand from period t to end for each product
    cum_demand = [[0.0] * num_periods for _ in range(num_products)]
    for i in range(num_products):
        cum_demand[i][num_periods - 1] = demand[i][num_periods - 1]
        for t in range(num_periods - 2, -1, -1):
            cum_demand[i][t] = cum_demand[i][t + 1] + demand[i][t]

    # Precompute upper bound M[i][t] for production linking constraint
    M = [[0.0] * num_periods for _ in range(num_products)]
    for i in range(num_products):
        for t in range(num_periods):
            if variable_production_time[i][t] > 1e-10:
                cap_bound = (capacity[t] - setup_time[i][t]) / variable_production_time[i][t]
            else:
                cap_bound = float('inf')
            M[i][t] = max(0.0, min(cum_demand[i][t], cap_bound))

    # Build model
    model = gp.Model("CLSP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit - 1  # Leave buffer for solution extraction
    model.Params.OutputFlag = 0

    # Decision variables
    x = {}   # production quantity
    y = {}   # setup binary
    I = {}   # ending inventory
    I0 = {}  # initial inventory

    for i in range(num_products):
        I0[i] = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"I0_{i}")
        for t in range(num_periods):
            x[i, t] = model.addVar(lb=0.0, ub=M[i][t], vtype=GRB.CONTINUOUS, name=f"x_{i}_{t}")
            y[i, t] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"y_{i}_{t}")
            I[i, t] = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"I_{i}_{t}")

    model.update()

    # Objective function
    obj = gp.LinExpr()
    for i in range(num_products):
        obj += initial_inventory_cost[i] * I0[i]
        for t in range(num_periods):
            obj += setup_cost[i][t] * y[i, t]
            obj += variable_production_cost[i][t] * x[i, t]
            obj += holding_cost[i][t] * I[i, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Inventory balance constraints
    for i in range(num_products):
        # Period 0: initial inventory + production = demand + ending inventory
        model.addConstr(
            I0[i] + x[i, 0] == demand[i][0] + I[i, 0],
            name=f"balance_{i}_0"
        )
        # Periods 1 to T-1
        for t in range(1, num_periods):
            model.addConstr(
                I[i, t - 1] + x[i, t] == demand[i][t] + I[i, t],
                name=f"balance_{i}_{t}"
            )

    # Production linking constraints: x[i,t] <= M[i,t] * y[i,t]
    for i in range(num_products):
        for t in range(num_periods):
            if M[i][t] > 0:
                model.addConstr(
                    x[i, t] <= M[i][t] * y[i, t],
                    name=f"link_{i}_{t}"
                )
            else:
                # If M is 0, production must be 0
                model.addConstr(x[i, t] == 0, name=f"link_{i}_{t}")

    # Capacity constraints
    for t in range(num_periods):
        cap_expr = gp.LinExpr()
        for i in range(num_products):
            cap_expr += setup_time[i][t] * y[i, t]
            cap_expr += variable_production_time[i][t] * x[i, t]
        model.addConstr(cap_expr <= capacity[t], name=f"cap_{t}")

    # Add valid inequalities to strengthen the formulation
    # (l,S) style: for each product i and period t, production from t to end
    # must cover remaining demand minus what can be produced without setup
    for i in range(num_products):
        for t in range(num_periods):
            # sum_{s=t}^{T-1} x[i,s] >= cum_demand[i][t] - I[i,t-1]
            # This is implied by balance constraints, but let's add a stronger version:
            # x[i,t] + sum_{s=t+1}^{T-1} M[i,s]*y[i,s] >= cum_demand[i][t] - I[i,t-1]
            if t == 0:
                # I0[i] + sum_{s=0}^{T-1} x[i,s] >= cum_demand[i][0]
                # This is implied, skip
                pass
            else:
                lhs = gp.LinExpr()
                lhs += x[i, t]
                for s in range(t + 1, num_periods):
                    lhs += M[i][s] * y[i, s]
                # I[i,t-1] + x[i,t] + sum_{s>t} M[i,s]*y[i,s] >= cum_demand[i][t]
                lhs += I[i, t - 1]
                model.addConstr(lhs >= cum_demand[i][t], name=f"valid_{i}_{t}")

    model.update()

    # Callback to log incumbent solutions
    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                production = [[model.cbGetSolution(x[i, t]) for t in range(num_periods)]
                              for i in range(num_products)]
                setup = [[int(round(model.cbGetSolution(y[i, t]))) for t in range(num_periods)]
                         for i in range(num_products)]
                inventory = [[model.cbGetSolution(I[i, t]) for t in range(num_periods)]
                             for i in range(num_products)]
                init_inv = [model.cbGetSolution(I0[i]) for i in range(num_products)]

                solution_dict = {
                    "objective_value": obj_val,
                    "variables": {
                        "production": production,
                        "setup": setup,
                        "inventory": inventory,
                        "initial_inventory": init_inv
                    }
                }
                if logger:
                    logger.log_solution(obj_val, solution_dict)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        production = [[x[i, t].X for t in range(num_periods)] for i in range(num_products)]
        setup = [[int(round(y[i, t].X)) for t in range(num_periods)] for i in range(num_products)]
        inventory = [[I[i, t].X for t in range(num_periods)] for i in range(num_products)]
        init_inv = [I0[i].X for i in range(num_products)]

        solution_dict = {
            "objective_value": obj_val,
            "variables": {
                "production": production,
                "setup": setup,
                "inventory": inventory,
                "initial_inventory": init_inv
            }
        }

        if logger:
            logger.log_solution(obj_val, solution_dict)

    else:
        # Fallback: use initial inventory to satisfy all demand (feasible due to penalty cost)
        production = [[0.0] * num_periods for _ in range(num_products)]
        setup = [[0] * num_periods for _ in range(num_products)]
        inventory = [[0.0] * num_periods for _ in range(num_products)]
        init_inv = [0.0] * num_products

        # Set initial inventory to cover all demand
        for i in range(num_products):
            init_inv[i] = sum(demand[i])

        obj_val = sum(initial_inventory_cost[i] * init_inv[i] for i in range(num_products))

        solution_dict = {
            "objective_value": obj_val,
            "variables": {
                "production": production,
                "setup": setup,
                "inventory": inventory,
                "initial_inventory": init_inv
            }
        }

        if logger:
            logger.log_solution(obj_val, solution_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)


if __name__ == "__main__":
    main()