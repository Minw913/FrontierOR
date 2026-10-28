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

    # Extract dimensions
    I = data['dimensions']['num_items']
    T = data['dimensions']['num_periods']
    P = data['dimensions']['num_plants']
    C = data['dimensions']['num_customers']

    items = range(I)
    periods = range(T)
    plants = range(P)
    customers = range(C)

    # Precompute cumulative demand per item per period (for all customers)
    cum_demand = [[[0 for _ in periods] for _ in customers] for _ in items]
    for i in items:
        for c in customers:
            cum = 0
            for t in periods:
                cum += data['demand'][i][c][t]
                cum_demand[i][c][t] = cum

    total_demand = [[0] * T for _ in items]
    for i in items:
        for t in periods:
            total_demand[i][t] = sum(data['demand'][i][c][t] for c in customers)

    # Precompute total demand through each period per item
    cum_total_demand = [[0] * T for _ in items]
    for i in items:
        cum = 0
        for t in periods:
            cum += total_demand[i][t]
            cum_total_demand[i][t] = cum

    # Transportation cost: for each item, plant, period, the total cost to serve all demand in that period
    trans_cost = [[[0 for _ in periods] for _ in plants] for _ in items]
    for i in items:
        for p in plants:
            for t in periods:
                cost = 0
                for c in customers:
                    if data['demand'][i][c][t] > 0:
                        cost += data['demand'][i][c][t] * data['transportation_cost'][i][c][p][t]
                trans_cost[i][p][t] = cost

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Create Gurobi model
    model = gp.Model("MultiItemLotSizingSupplyChain")

    # Set time limit
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01  # Allow 1% gap for efficiency on larger instances
    model.Params.LogToConsole = 0

    # Variables
    # Z_p: 1 if plant p is opened
    Z = model.addVars(plants, vtype=GRB.BINARY, name="Z")

    # U_ip: 1 if item i is assigned to plant p
    U = {}
    for i in items:
        for p in data['plant_capabilities'][i]:  # only capable plants
            U[i, p] = model.addVar(vtype=GRB.BINARY, name=f"U_{i}_{p}")

    # Y_ipt: 1 if setup for item i at plant p in period t
    Y = model.addVars(items, plants, periods, vtype=GRB.BINARY, name="Y")

    # X_ipt: production quantity of item i at plant p in period t
    X = model.addVars(items, plants, periods, vtype=GRB.CONTINUOUS, lb=0, name="X")

    # I_ipt: inventory of item i at plant p at end of period t
    I = model.addVars(items, plants, periods, vtype=GRB.CONTINUOUS, lb=0, name="I")

    # Objective: plant opening + setup + production + holding + transportation
    plant_opening = gp.quicksum(data['plant_opening_cost'][p] * Z[p] for p in plants)

    setup_cost = gp.quicksum(
        data['setup_cost'][i][p][t] * Y[i, p, t]
        for i in items for p in plants for t in periods
    )

    prod_cost = gp.quicksum(
        data['production_cost'][i][p][t] * X[i, p, t]
        for i in items for p in plants for t in periods
    )

    hold_cost = gp.quicksum(
        data['holding_cost'][i][p][t] * I[i, p, t]
        for i in items for p in plants for t in periods
    )

    trans_total = gp.quicksum(
        trans_cost[i][p][t] * X[i, p, t]
        for i in items for p in plants for t in periods
    )

    model.setObjective(
        plant_opening + setup_cost + prod_cost + hold_cost + trans_total,
        GRB.MINIMIZE
    )

    # Constraints

    # 1. Each item assigned to exactly one capable plant
    for i in items:
        capable = data['plant_capabilities'][i]
        model.addConstr(
            gp.quicksum(U[i, p] for p in capable) == 1,
            name=f"assign_{i}"
        )

    # 2. Can only assign to open plant
    for i in items:
        for p in data['plant_capabilities'][i]:
            model.addConstr(
                U[i, p] <= Z[p],
                name=f"assign_open_{i}_{p}"
            )

    # 3. Setup only if assigned to that plant
    for i in items:
        for p in plants:
            for t in periods:
                if (i, p) in U:
                    model.addConstr(
                        Y[i, p, t] <= U[i, p],
                        name=f"setup_assign_{i}_{p}_{t}"
                    )
                else:
                    model.addConstr(
                        Y[i, p, t] == 0,
                        name=f"setup_impossible_{i}_{p}_{t}"
                    )

    # 4. Production only if setup
    for i in items:
        for p in plants:
            for t in periods:
                model.addConstr(
                    X[i, p, t] <= cum_total_demand[i][T-1] * Y[i, p, t],
                    name=f"prod_setup_{i}_{p}_{t}"
                )

    # 5. Capacity constraints (only at open plants)
    for p in plants:
        for t in periods:
            setup_time_expr = gp.quicksum(
                data['setup_time'][i][p][t] * Y[i, p, t]
                for i in items
            )
            prod_time_expr = gp.quicksum(
                data['production_time'][i][p][t] * X[i, p, t]
                for i in items
            )
            model.addConstr(
                setup_time_expr + prod_time_expr <= data['capacity'][p][t] * Z[p],
                name=f"capacity_{p}_{t}"
            )

    # 6. Inventory balance and demand satisfaction
    # For each item and plant, inventory flow must satisfy all demand if assigned
    M = 10**9  # Big-M
    for i in items:
        for p in plants:
            for t in periods:
                # Inventory balance
                if t == 0:
                    model.addConstr(
                        I[i, p, t] == X[i, p, t] - total_demand[i][t],
                        name=f"balance_{i}_{p}_{t}"
                    )
                else:
                    model.addConstr(
                        I[i, p, t] == I[i, p, t-1] + X[i, p, t] - total_demand[i][t],
                        name=f"balance_{i}_{p}_{t}"
                    )
                # Non-negative inventory (enforces no backlogging)
                model.addConstr(I[i, p, t] >= 0, name=f"inv_nonneg_{i}_{p}_{t}")

    # 7. If not assigned to plant, production and inventory must be zero
    for i in items:
        for p in plants:
            if (i, p) not in U:
                for t in periods:
                    model.addConstr(X[i, p, t] == 0, name=f"no_prod_{i}_{p}_{t}")
                    model.addConstr(I[i, p, t] == 0, name=f"no_inv_{i}_{p}_{t}")

    # 8. Linking U and production: if U=0 then all X=0 (redundant but helps)
    for i in items:
        for p in plants:
            if (i, p) in U:
                model.addConstr(
                    gp.quicksum(X[i, p, t] for t in periods) <= cum_total_demand[i][T-1] * U[i, p],
                    name=f"u_link_{i}_{p}"
                )

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SOLUTION_LIMIT]:
        obj_val = model.ObjVal

        Z_sol = {}
        for p in plants:
            Z_sol[str(p)] = 1 if Z[p].X > 0.5 else 0

        U_sol = {}
        for i in items:
            for p_idx in data['plant_capabilities'][i]:
                p = p_idx
                key = f"{i}_{p}"
                U_sol[key] = 1 if (i, p) in U and U[i, p].X > 0.5 else 0

        Y_sol = {}
        for i in items:
            for p in plants:
                for t in periods:
                    key = f"{i}_{p}_{t}"
                    Y_sol[key] = 1 if Y[i, p, t].X > 0.5 else 0

        solution = {
            "objective_value": float(obj_val),
            "Z": Z_sol,
            "U": U_sol,
            "Y": Y_sol
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible case - write empty with high cost
        solution = {
            "objective_value": 1e10,
            "Z": {str(p): 0 for p in plants},
            "U": {},
            "Y": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()