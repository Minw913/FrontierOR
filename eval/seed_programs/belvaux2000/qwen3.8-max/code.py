import argparse
import json
import sys
import time
from gurobipy import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    NI = data['parameters']['NI']
    NK = data['parameters']['NK']
    NT = data['parameters']['NT']

    setup_cost_f = data['costs']['setup_cost_f']
    startup_cost_g = data['costs']['startup_cost_g']
    holding_cost_h = data['costs']['holding_cost_h']
    backlogging_cost_e = data['costs']['backlogging_cost_e']

    capacity_C = data['machine_data']['capacity_C']
    startup_time_sigma = data['machine_data']['startup_time_sigma']

    demand = data['demand']
    initial_stock_s0 = data['initial_conditions']['initial_stock_s0']
    initial_backlog_r0 = data['initial_conditions']['initial_backlog_r0']

    # Build MIP model
    model = Model("production_planning")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = {}  # production
    s = {}  # inventory
    r = {}  # backlog
    y = {}  # setup
    z = {}  # startup

    for i in range(NI):
        for t in range(NT):
            s[i, t] = model.addVar(lb=0, name=f"s_{i}_{t}")
            r[i, t] = model.addVar(lb=0, name=f"r_{i}_{t}")

    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                x[i, k, t] = model.addVar(lb=0, ub=capacity_C[k], name=f"x_{i}_{k}_{t}")
                y[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{t}")
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")

    model.update()

    # Objective
    obj = LinExpr()
    for k in range(NK):
        for i in range(NI):
            for t in range(NT):
                obj += setup_cost_f * y[i, k, t]
                obj += startup_cost_g * z[i, k, t]
    for i in range(NI):
        for t in range(NT):
            obj += holding_cost_h[i] * s[i, t]
            obj += backlogging_cost_e[i] * r[i, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Flow balance constraints
    for i in range(NI):
        for t in range(NT):
            prev_s = initial_stock_s0[i] if t == 0 else s[i, t - 1]
            prev_r = initial_backlog_r0[i] if t == 0 else r[i, t - 1]
            model.addConstr(
                prev_s + quicksum(x[i, k, t] for k in range(NK)) - prev_r ==
                demand[i][t] + s[i, t] - r[i, t],
                name=f"flow_{i}_{t}"
            )

    # Each machine at most one item per period
    for k in range(NK):
        for t in range(NT):
            model.addConstr(
                quicksum(y[i, k, t] for i in range(NI)) <= 1,
                name=f"one_item_{k}_{t}"
            )

    # Production only if set up (linking constraint)
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                model.addConstr(
                    x[i, k, t] <= capacity_C[k] * y[i, k, t],
                    name=f"prod_setup_{i}_{k}_{t}"
                )

    # Capacity with startup loss
    for k in range(NK):
        for i in range(NI):
            for t in range(NT):
                model.addConstr(
                    x[i, k, t] + startup_time_sigma[k] * z[i, k, t] <= capacity_C[k],
                    name=f"cap_startup_{i}_{k}_{t}"
                )

    # Startup logic constraints
    for i in range(NI):
        for k in range(NK):
            # First period: no machine is set up before period 0
            model.addConstr(z[i, k, 0] == y[i, k, 0], name=f"startup_init_{i}_{k}")
            for t in range(1, NT):
                # z >= y[t] - y[t-1] (startup needed if switching to item)
                model.addConstr(
                    z[i, k, t] >= y[i, k, t] - y[i, k, t - 1],
                    name=f"startup_lb_{i}_{k}_{t}"
                )
                # z <= y[t] (can only startup if set up)
                model.addConstr(
                    z[i, k, t] <= y[i, k, t],
                    name=f"startup_ub1_{i}_{k}_{t}"
                )
                # z <= 1 - y[t-1] (no startup if same item was set up before)
                model.addConstr(
                    z[i, k, t] <= 1 - y[i, k, t - 1],
                    name=f"startup_ub2_{i}_{k}_{t}"
                )

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                # Extract solution from callback
                production = {}
                stock = {}
                backlog = {}
                setup_dict = {}
                startup_dict = {}

                for i in range(NI):
                    for t in range(NT):
                        stock[f"{i}_{t}"] = round(model.cbGetSolution(s[i, t]), 6)
                        backlog[f"{i}_{t}"] = round(model.cbGetSolution(r[i, t]), 6)

                for i in range(NI):
                    for k in range(NK):
                        for t in range(NT):
                            production[f"{i}_{k}_{t}"] = round(model.cbGetSolution(x[i, k, t]), 6)
                            setup_dict[f"{i}_{k}_{t}"] = int(round(model.cbGetSolution(y[i, k, t])))
                            startup_dict[f"{i}_{k}_{t}"] = int(round(model.cbGetSolution(z[i, k, t])))

                solution = {
                    "objective_value": obj_val,
                    "production": production,
                    "stock": stock,
                    "backlog": backlog,
                    "setup": setup_dict,
                    "startup": startup_dict
                }

                if logger:
                    logger.log_solution(obj_val, solution)
            except GurobiError:
                pass

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        production = {}
        stock = {}
        backlog = {}
        setup_dict = {}
        startup_dict = {}

        for i in range(NI):
            for t in range(NT):
                stock[f"{i}_{t}"] = round(s[i, t].X, 6)
                backlog[f"{i}_{t}"] = round(r[i, t].X, 6)

        for i in range(NI):
            for k in range(NK):
                for t in range(NT):
                    production[f"{i}_{k}_{t}"] = round(x[i, k, t].X, 6)
                    setup_dict[f"{i}_{k}_{t}"] = int(round(y[i, k, t].X))
                    startup_dict[f"{i}_{k}_{t}"] = int(round(z[i, k, t].X))

        objective_value = model.ObjVal

        solution = {
            "objective_value": objective_value,
            "production": production,
            "stock": stock,
            "backlog": backlog,
            "setup": setup_dict,
            "startup": startup_dict
        }

        # Log final solution if not already logged
        if logger:
            logger.log_solution(objective_value, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # Fallback: produce a trivial feasible solution (all backlog)
        production = {}
        stock = {}
        backlog = {}
        setup_dict = {}
        startup_dict = {}

        for i in range(NI):
            for t in range(NT):
                stock[f"{i}_{t}"] = 0.0
                # Compute cumulative backlog
                cum_demand = sum(demand[i][:t + 1])
                backlog[f"{i}_{t}"] = float(max(0, cum_demand - initial_stock_s0[i] + initial_backlog_r0[i]))

        for i in range(NI):
            for k in range(NK):
                for t in range(NT):
                    production[f"{i}_{k}_{t}"] = 0.0
                    setup_dict[f"{i}_{k}_{t}"] = 0
                    startup_dict[f"{i}_{k}_{t}"] = 0

        objective_value = sum(
            backlogging_cost_e[i] * backlog[f"{i}_{t}"]
            for i in range(NI) for t in range(NT)
        )

        solution = {
            "objective_value": objective_value,
            "production": production,
            "stock": stock,
            "backlog": backlog,
            "setup": setup_dict,
            "startup": startup_dict
        }

        if logger:
            logger.log_solution(objective_value, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    main()