import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)
        
    num_items = data["dimensions"]["num_items"]
    num_periods = data["dimensions"]["num_periods"]
    num_plants = data["dimensions"]["num_plants"]
    num_customers = data["dimensions"]["num_customers"]

    plant_capabilities = data["plant_capabilities"]
    demand = data["demand"]
    setup_cost = data["setup_cost"]
    production_cost = data["production_cost"]
    holding_cost = data["holding_cost"]
    setup_time = data["setup_time"]
    production_time = data["production_time"]
    capacity = data["capacity"]
    plant_opening_cost = data["plant_opening_cost"]
    transportation_cost = data["transportation_cost"]
    
    # Pre-process plant capabilities into sets for faster lookups
    plant_capabilities_set = {i: set(plant_capabilities[i]) for i in range(num_items)}

    # Precompute aggregate demands spanning all customers
    D = [[0] * num_periods for _ in range(num_items)]
    for i in range(num_items):
        for j in range(num_customers):
            for t in range(num_periods):
                D[i][t] += demand[i][j][t]
                
    # Precompute total transportation costs parameterized over the horizon
    # Since shipments must exactly match customer demands instantly in the matching period
    tc_lookup = {}
    for i in range(num_items):
        for p in plant_capabilities_set[i]:
            val = 0
            for j in range(num_customers):
                for t in range(num_periods):
                    val += demand[i][j][t] * transportation_cost[i][j][p][t]
            tc_lookup[i, p] = val

    # Setup Gurobi model constraints
    model = gp.Model("TwoEchelonSupplyChain")
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Decision Variables
    Z = model.addVars(num_plants, vtype=GRB.BINARY, name="Z")
    U = {}
    Y = {}
    X = {}
    I_var = {}
    for i in range(num_items):
        for p in plant_capabilities_set[i]:
            U[i, p] = model.addVar(vtype=GRB.BINARY, name=f"U_{i}_{p}")
            for t in range(num_periods):
                Y[i, p, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{p}_{t}")
                X[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"X_{i}_{p}_{t}")
                I_var[i, p, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{i}_{p}_{t}")

    # 1. Assignment Constraints
    for i in range(num_items):
        model.addConstr(gp.quicksum(U[i, p] for p in plant_capabilities_set[i]) == 1, name=f"Assign_{i}")
        
    for i in range(num_items):
        for p in plant_capabilities_set[i]:
            model.addConstr(U[i, p] <= Z[p], name=f"Open_{i}_{p}")
            
            for t in range(num_periods):
                model.addConstr(Y[i, p, t] <= U[i, p], name=f"ValidSetup_{i}_{p}_{t}")
                
                # Logic linking continuous production to binary setups
                cap_pt = capacity[p][t]
                st = setup_time[i][p][t]
                pt = production_time[i][p][t]
                
                D_tail = sum(D[i][tau] for tau in range(t, num_periods))
                if st > cap_pt:
                    M = 0
                else:
                    if pt == 0:
                        M = D_tail
                    else:
                        M = min(D_tail, (cap_pt - st) / pt)
                if M < 0:
                    M = 0
                    
                model.addConstr(X[i, p, t] <= M * Y[i, p, t], name=f"ProdLimit_{i}_{p}_{t}")
                
                # Inventory Balance linking flows and ensuring fully met demand consecutively
                prev_I = I_var[i, p, t-1] if t > 0 else 0
                model.addConstr(I_var[i, p, t] == prev_I + X[i, p, t] - D[i][t] * U[i, p], name=f"Balance_{i}_{p}_{t}")

    # Valid Inequalities: Minimum Required Setup Check
    for i in range(num_items):
        total_d = sum(D[i])
        if total_d > 0:
            demanding_periods = [t for t in range(num_periods) if D[i][t] > 0]
            first_demand_period = demanding_periods[0]
            for p in plant_capabilities_set[i]:
                model.addConstr(gp.quicksum(Y[i, p, tau] for tau in range(first_demand_period + 1)) >= U[i, p])

    # 2. Capacity Constraints per Plant-Period Segment
    for p in range(num_plants):
        for t in range(num_periods):
            expr = gp.LinExpr()
            for i in range(num_items):
                if p in plant_capabilities_set[i]:
                    expr.add(setup_time[i][p][t] * Y[i, p, t] + production_time[i][p][t] * X[i, p, t])
            # Utilizing Z bounds naturally drops plant segments not selected by optimizer
            model.addConstr(expr <= capacity[p][t] * Z[p], name=f"Cap_{p}_{t}")

    # Objective Function
    obj = gp.LinExpr()
    obj += gp.quicksum(plant_opening_cost[p] * Z[p] for p in range(num_plants))
    obj += gp.quicksum(tc_lookup[i, p] * U[i, p] for i in range(num_items) for p in plant_capabilities_set[i])
    obj += gp.quicksum(
        setup_cost[i][p][t] * Y[i, p, t] +
        production_cost[i][p][t] * X[i, p, t] +
        holding_cost[i][p][t] * I_var[i, p, t]
        for i in range(num_items)
        for p in plant_capabilities_set[i]
        for t in range(num_periods)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Hook robust incumbent logging
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    def optimization_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)

    # Perform the optimization search routines yielding the finest solution bounded by `--time_limit`
    model.optimize(optimization_callback)

    # Write formatted structures matching standard schema
    solution = {
        "objective_value": None,
        "Z": {},
        "U": {},
        "Y": {}
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        for p in range(num_plants):
            solution["Z"][str(p)] = int(round(Z[p].X))
            
        for i in range(num_items):
            for p in range(num_plants):
                if p in plant_capabilities_set[i]:
                    solution["U"][f"{i}_{p}"] = int(round(U[i, p].X))
                    for t in range(num_periods):
                        solution["Y"][f"{i}_{p}_{t}"] = int(round(Y[i, p, t].X))
                else:
                    solution["U"][f"{i}_{p}"] = 0
                    for t in range(num_periods):
                        solution["Y"][f"{i}_{p}_{t}"] = 0

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()