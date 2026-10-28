import json
import argparse
import sys
from gurobipy import *


def main():
    parser = argparse.ArgumentParser(description="Two-echelon supply chain optimization")
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

    num_items = data['dimensions']['num_items']
    num_periods = data['dimensions']['num_periods']
    num_plants = data['dimensions']['num_plants']
    num_customers = data['dimensions']['num_customers']

    plant_capabilities = data['plant_capabilities']
    demand = data['demand']
    setup_cost = data['setup_cost']
    production_cost = data['production_cost']
    holding_cost = data['holding_cost']
    setup_time = data['setup_time']
    production_time = data['production_time']
    capacity = data['capacity']
    plant_opening_cost = data['plant_opening_cost']
    transportation_cost = data['transportation_cost']

    # Precompute total demand per item per period
    total_demand = [[0] * num_periods for _ in range(num_items)]
    for i in range(num_items):
        for c in range(num_customers):
            for t in range(num_periods):
                total_demand[i][t] += demand[i][c][t]

    # Total demand per item over horizon
    total_demand_item = [sum(total_demand[i]) for i in range(num_items)]

    # Precompute transportation cost for each (item, plant) pair
    trans_cost = [[0.0] * num_plants for _ in range(num_items)]
    for i in range(num_items):
        for p in range(num_plants):
            cost = 0
            for c in range(num_customers):
                for t in range(num_periods):
                    cost += transportation_cost[i][c][p][t] * demand[i][c][t]
            trans_cost[i][p] = cost

    # Build capable plants set for quick lookup
    capable_set = [set(plant_capabilities[i]) for i in range(num_items)]

    # Items capable at each plant
    plant_items = [[] for _ in range(num_plants)]
    for i in range(num_items):
        for p in plant_capabilities[i]:
            plant_items[p].append(i)

    # Create model
    model = Model("supply_chain")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # Decision variables
    Z = {}
    for p in range(num_plants):
        Z[p] = model.addVar(vtype=GRB.BINARY, name=f"Z_{p}")

    U = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            U[i, p] = model.addVar(vtype=GRB.BINARY, name=f"U_{i}_{p}")

    Y = {}
    X = {}
    I_var = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                Y[i, p, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{p}_{t}")
                X[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"X_{i}_{p}_{t}")
                I_var[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{p}_{t}")

    model.update()

    # Objective
    obj = LinExpr()

    # Plant opening costs
    for p in range(num_plants):
        obj += plant_opening_cost[p] * Z[p]

    # Setup, production, holding costs
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                obj += setup_cost[i][p][t] * Y[i, p, t]
                obj += production_cost[i][p][t] * X[i, p, t]
                obj += holding_cost[i][p][t] * I_var[i, p, t]

    # Transportation costs
    for i in range(num_items):
        for p in plant_capabilities[i]:
            obj += trans_cost[i][p] * U[i, p]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints

    # Each item assigned to exactly one capable plant
    for i in range(num_items):
        model.addConstr(quicksum(U[i, p] for p in plant_capabilities[i]) == 1, name=f"assign_{i}")

    # Assignment only to open plants
    for i in range(num_items):
        for p in plant_capabilities[i]:
            model.addConstr(U[i, p] <= Z[p], name=f"open_{i}_{p}")

    # Setup only if assigned
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                model.addConstr(Y[i, p, t] <= U[i, p], name=f"setup_assign_{i}_{p}_{t}")

    # Production only with setup (tight big-M: remaining demand from period t)
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                remaining = sum(total_demand[i][t2] for t2 in range(t, num_periods))
                if remaining > 0:
                    model.addConstr(X[i, p, t] <= remaining * Y[i, p, t], name=f"prod_setup_{i}_{p}_{t}")
                else:
                    X[i, p, t].setAttr("UB", 0)

    # Per-plant inventory balance
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                if t == 0:
                    model.addConstr(
                        X[i, p, 0] - I_var[i, p, 0] == total_demand[i][0] * U[i, p],
                        name=f"inv_bal_{i}_{p}_0")
                else:
                    model.addConstr(
                        I_var[i, p, t - 1] + X[i, p, t] - I_var[i, p, t] == total_demand[i][t] * U[i, p],
                        name=f"inv_bal_{i}_{p}_{t}")

    # Total production cannot exceed total demand (valid inequality)
    for i in range(num_items):
        for p in plant_capabilities[i]:
            model.addConstr(
                quicksum(X[i, p, t] for t in range(num_periods)) <= total_demand_item[i] * U[i, p],
                name=f"total_prod_{i}_{p}")

    # Capacity constraints
    for p in range(num_plants):
        for t in range(num_periods):
            if plant_items[p]:
                lhs = LinExpr()
                for i in plant_items[p]:
                    lhs += setup_time[i][p][t] * Y[i, p, t] + production_time[i][p][t] * X[i, p, t]
                model.addConstr(lhs <= capacity[p][t] * Z[p], name=f"cap_{p}_{t}")

    model.update()

    # Store data for callback
    model._Z = Z
    model._U = U
    model._Y = Y
    model._num_items = num_items
    model._num_plants = num_plants
    model._num_periods = num_periods
    model._plant_capabilities = plant_capabilities
    model._capable_set = capable_set
    model._logger = logger

    # Precompute key lists for faster solution extraction
    u_keys = []
    u_vars = []
    for i in range(num_items):
        for p in range(num_plants):
            u_keys.append(f"{i}_{p}")
            if p in capable_set[i]:
                u_vars.append(U[i, p])
            else:
                u_vars.append(None)

    y_keys = []
    y_vars = []
    for i in range(num_items):
        for p in range(num_plants):
            for t in range(num_periods):
                y_keys.append(f"{i}_{p}_{t}")
                if p in capable_set[i]:
                    y_vars.append(Y[i, p, t])
                else:
                    y_vars.append(None)

    model._u_keys = u_keys
    model._u_vars = u_vars
    model._y_keys = y_keys
    model._y_vars = y_vars

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            try:
                # Extract Z
                Z_dict = {}
                for p in range(model._num_plants):
                    Z_dict[str(p)] = int(round(model.cbGetSolution(model._Z[p])))

                # Extract U
                U_dict = {}
                active_u_vars = [v for v in model._u_vars if v is not None]
                u_vals = model.cbGetSolution(active_u_vars)
                idx = 0
                for k_idx, key in enumerate(model._u_keys):
                    if model._u_vars[k_idx] is not None:
                        U_dict[key] = int(round(u_vals[idx]))
                        idx += 1
                    else:
                        U_dict[key] = 0

                # Extract Y
                Y_dict = {}
                active_y_vars = [v for v in model._y_vars if v is not None]
                y_vals = model.cbGetSolution(active_y_vars)
                idx = 0
                for k_idx, key in enumerate(model._y_keys):
                    if model._y_vars[k_idx] is not None:
                        Y_dict[key] = int(round(y_vals[idx]))
                        idx += 1
                    else:
                        Y_dict[key] = 0

                solution = {
                    "objective_value": obj_val,
                    "Z": Z_dict,
                    "U": U_dict,
                    "Y": Y_dict
                }

                if model._logger:
                    model._logger.log_solution(obj_val, solution)
            except Exception:
                if model._logger:
                    model._logger.log(obj_val)

    model.setCallback(callback)

    # Optimize
    model.optimize()

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal

        Z_dict = {}
        for p in range(num_plants):
            Z_dict[str(p)] = int(round(Z[p].X))

        U_dict = {}
        for i in range(num_items):
            for p in range(num_plants):
                key = f"{i}_{p}"
                if p in capable_set[i]:
                    U_dict[key] = int(round(U[i, p].X))
                else:
                    U_dict[key] = 0

        Y_dict = {}
        for i in range(num_items):
            for p in range(num_plants):
                for t in range(num_periods):
                    key = f"{i}_{p}_{t}"
                    if p in capable_set[i]:
                        Y_dict[key] = int(round(Y[i, p, t].X))
                    else:
                        Y_dict[key] = 0

        solution = {
            "objective_value": obj_val,
            "Z": Z_dict,
            "U": U_dict,
            "Y": Y_dict
        }
    else:
        # No feasible solution found
        solution = {
            "objective_value": 1e20,
            "Z": {str(p): 0 for p in range(num_plants)},
            "U": {f"{i}_{p}": 0 for i in range(num_items) for p in range(num_plants)},
            "Y": {f"{i}_{p}_{t}": 0 for i in range(num_items) for p in range(num_plants) for t in range(num_periods)}
        }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Log final solution if not already logged via callback
    if logger and model.SolCount > 0:
        logger.log_solution(solution['objective_value'], solution)


if __name__ == "__main__":
    main()