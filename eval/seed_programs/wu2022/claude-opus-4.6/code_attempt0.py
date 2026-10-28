import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    num_items = data['dimensions']['num_items']
    num_periods = data['dimensions']['num_periods']
    num_plants = data['dimensions']['num_plants']
    num_customers = data['dimensions']['num_customers']

    plant_capabilities = data['plant_capabilities']
    customer_item_demands = data['customer_item_demands']
    demand = data['demand']
    setup_cost = data['setup_cost']
    production_cost = data['production_cost']
    holding_cost = data['holding_cost']
    setup_time = data['setup_time']
    production_time = data['production_time']
    capacity = data['capacity']
    plant_opening_cost = data['plant_opening_cost']
    transportation_cost = data['transportation_cost']

    # Precompute total demand for each item across all customers per period
    # demand[i][c][t]
    total_demand = []
    for i in range(num_items):
        td = [0.0] * num_periods
        for c in range(num_customers):
            for t in range(num_periods):
                td[t] += demand[i][c][t]
        total_demand.append(td)

    # Cumulative demand for each item through period t
    cum_demand = []
    for i in range(num_items):
        cd = [0.0] * num_periods
        cd[0] = total_demand[i][0]
        for t in range(1, num_periods):
            cd[t] = cd[t - 1] + total_demand[i][t]
        cum_demand.append(cd)

    # Total demand for each item over the entire horizon
    total_item_demand = [cum_demand[i][num_periods - 1] for i in range(num_items)]

    # Transportation cost for item i produced at plant p: sum over all customers and periods
    # For each item i, plant p: sum_c sum_t demand[i][c][t] * transportation_cost[i][c][p][t]
    trans_cost_item_plant = []
    for i in range(num_items):
        tc_plant = {}
        for p in plant_capabilities[i]:
            tc = 0.0
            for c in range(num_customers):
                for t in range(num_periods):
                    if demand[i][c][t] > 0:
                        tc += demand[i][c][t] * transportation_cost[i][c][p][t]
            tc_plant[p] = tc
        trans_cost_item_plant.append(tc_plant)

    start_time = time.time()

    model = gp.Model("two_echelon_supply_chain")
    model.setParam('TimeLimit', max(1, time_limit - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions

    # Decision variables
    # Z[p]: whether plant p is opened
    Z = model.addVars(num_plants, vtype=GRB.BINARY, name="Z")

    # U[i,p]: whether item i is assigned to plant p
    U = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            U[i, p] = model.addVar(vtype=GRB.BINARY, name=f"U_{i}_{p}")

    # Y[i,p,t]: whether setup is performed for item i at plant p in period t
    Y = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                Y[i, p, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{p}_{t}")

    # X[i,p,t]: production quantity of item i at plant p in period t
    X = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                ub = total_item_demand[i]  # can't produce more than total demand
                X[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub,
                                           name=f"X_{i}_{p}_{t}")

    # I[i,p,t]: inventory of item i at plant p at end of period t
    I = {}
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                I[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                           name=f"I_{i}_{p}_{t}")

    model.update()

    # Objective function
    obj = gp.LinExpr()

    # Plant opening costs
    for p in range(num_plants):
        obj += plant_opening_cost[p] * Z[p]

    # Setup costs, production costs, holding costs
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                obj += setup_cost[i][p][t] * Y[i, p, t]
                obj += production_cost[i][p][t] * X[i, p, t]
                obj += holding_cost[i][p][t] * I[i, p, t]

    # Transportation costs (linearized via U)
    for i in range(num_items):
        for p in plant_capabilities[i]:
            obj += trans_cost_item_plant[i][p] * U[i, p]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints

    # 1. Each item assigned to exactly one plant
    for i in range(num_items):
        model.addConstr(
            gp.quicksum(U[i, p] for p in plant_capabilities[i]) == 1,
            name=f"assign_{i}"
        )

    # 2. Item can only be assigned to an open plant
    for i in range(num_items):
        for p in plant_capabilities[i]:
            model.addConstr(U[i, p] <= Z[p], name=f"open_{i}_{p}")

    # 3. Setup only if item assigned to plant
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                model.addConstr(Y[i, p, t] <= U[i, p],
                                name=f"setup_assign_{i}_{p}_{t}")

    # 4. Production only if setup
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                M = total_item_demand[i]
                model.addConstr(X[i, p, t] <= M * Y[i, p, t],
                                name=f"prod_setup_{i}_{p}_{t}")

    # 5. Capacity constraints per plant per period
    for p in range(num_plants):
        items_at_p = [i for i in range(num_items) if p in plant_capabilities[i]]
        if not items_at_p:
            continue
        for t in range(num_periods):
            model.addConstr(
                gp.quicksum(
                    setup_time[i][p][t] * Y[i, p, t] + production_time[i][p][t] * X[i, p, t]
                    for i in items_at_p
                ) <= capacity[p][t] * Z[p],
                name=f"cap_{p}_{t}"
            )

    # 6. Inventory balance: I[i,p,t] = I[i,p,t-1] + X[i,p,t] - total_demand[i][t] * U[i,p]
    for i in range(num_items):
        for p in plant_capabilities[i]:
            for t in range(num_periods):
                if t == 0:
                    model.addConstr(
                        I[i, p, t] == X[i, p, t] - total_demand[i][t] * U[i, p],
                        name=f"inv_bal_{i}_{p}_{t}"
                    )
                else:
                    model.addConstr(
                        I[i, p, t] == I[i, p, t - 1] + X[i, p, t] - total_demand[i][t] * U[i, p],
                        name=f"inv_bal_{i}_{p}_{t}"
                    )

    # 7. Inventory at end of horizon should be zero (no excess) - optional but helps
    # Actually, inventory must be >= 0 (already set as lb), which combined with balance ensures
    # cumulative production >= cumulative demand when U[i,p]=1.
    # The inventory balance with I >= 0 and correct demand ensures no backlogging.

    # Callback for logging
    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal

        Z_sol = {}
        for p in range(num_plants):
            Z_sol[str(p)] = int(round(Z[p].X))

        U_sol = {}
        for i in range(num_items):
            for p in range(num_plants):
                key = f"{i}_{p}"
                if p in plant_capabilities[i]:
                    U_sol[key] = int(round(U[i, p].X))
                else:
                    U_sol[key] = 0

        Y_sol = {}
        for i in range(num_items):
            for p in range(num_plants):
                for t in range(num_periods):
                    key = f"{i}_{p}_{t}"
                    if p in plant_capabilities[i]:
                        Y_sol[key] = int(round(Y[i, p, t].X))
                    else:
                        Y_sol[key] = 0

        solution = {
            "objective_value": objective_value,
            "Z": Z_sol,
            "U": U_sol,
            "Y": Y_sol
        }
    else:
        # No feasible solution found - return empty/default
        Z_sol = {str(p): 0 for p in range(num_plants)}
        U_sol = {}
        for i in range(num_items):
            for p in range(num_plants):
                U_sol[f"{i}_{p}"] = 0
        Y_sol = {}
        for i in range(num_items):
            for p in range(num_plants):
                for t in range(num_periods):
                    Y_sol[f"{i}_{p}_{t}"] = 0

        solution = {
            "objective_value": float('inf'),
            "Z": Z_sol,
            "U": U_sol,
            "Y": Y_sol
        }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)