import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def solve_production_problem(args):
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract parameters
    parameters = instance.get('parameters', {})
    NI = parameters.get('NI')
    NK = parameters.get('NK')
    NT = parameters.get('NT')

    # Extract costs
    costs = instance.get('costs', {})
    f_cost = costs.get('setup_cost_f')
    g_cost = costs.get('startup_cost_g')
    h_cost = costs.get('holding_cost_h')
    e_cost = costs.get('backlogging_cost_e')

    # Extract machine data
    machine_data = instance.get('machine_data', {})
    C = machine_data.get('capacity_C')
    sigma = machine_data.get('startup_time_sigma')

    # Extract demand
    d = instance.get('demand')

    # Extract initial conditions
    init_cond = instance.get('initial_conditions', {})
    s0 = init_cond.get('initial_stock_s0')
    r0 = init_cond.get('initial_backlog_r0')

    # Initialize Gurobi model
    model = gp.Model("Production")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Setup the solution logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Decision Variables
    # x: Production quantity of item i on machine k in period t
    x = model.addVars(NI, NK, NT, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    # s: Inventory level (stock) of item i at the end of period t
    s = model.addVars(NI, NT, vtype=GRB.CONTINUOUS, lb=0.0, name="s")
    # r: Backlog level of item i at the end of period t
    r = model.addVars(NI, NT, vtype=GRB.CONTINUOUS, lb=0.0, name="r")
    # y: 1 if machine k is set up to produce item i in period t, 0 otherwise
    y = model.addVars(NI, NK, NT, vtype=GRB.BINARY, name="y")
    # z: 1 if machine k starts up for item i in period t, 0 otherwise
    z = model.addVars(NI, NK, NT, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")

    # Constraints
    
    # 1. Inventory Balance Flow
    for i in range(NI):
        for t in range(NT):
            s_prev = s0[i] if t == 0 else s[i, t-1]
            r_prev = r0[i] if t == 0 else r[i, t-1]
            model.addConstr(
                s_prev + gp.quicksum(x[i, k, t] for k in range(NK)) - r_prev == d[i][t] + s[i, t] - r[i, t],
                name=f"inv_balance_{i}_{t}"
            )

    # 2. Maximum of one setup per machine per period
    for k in range(NK):
        for t in range(NT):
            model.addConstr(
                gp.quicksum(y[i, k, t] for i in range(NI)) <= 1,
                name=f"cap_y_{k}_{t}"
            )

    # 3. Capacity scaling and Startup conditions
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                y_prev = 0.0 if t == 0 else y[i, k, t-1]
                
                # Relate setup status explicitly to startup status (z = max(0, y_current - y_prev))
                model.addConstr(z[i, k, t] >= y[i, k, t] - y_prev, name=f"z_lb_{i}_{k}_{t}")
                model.addConstr(z[i, k, t] <= y[i, k, t], name=f"z_ub1_{i}_{k}_{t}")
                model.addConstr(z[i, k, t] <= 1 - y_prev, name=f"z_ub2_{i}_{k}_{t}")
                
                # Ensure production quantity alongside operational capacity and startup capacity loss
                model.addConstr(
                    x[i, k, t] + z[i, k, t] * sigma[k] <= C[k] * y[i, k, t],
                    name=f"cap_x_{i}_{k}_{t}"
                )

    # 4. Objective Function Calculation
    obj = (
        gp.quicksum(f_cost * y[i, k, t] + g_cost * z[i, k, t] 
                    for i in range(NI) for k in range(NK) for t in range(NT)) +
        gp.quicksum(h_cost[i] * s[i, t] + e_cost[i] * r[i, t] 
                    for i in range(NI) for t in range(NT))
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize and process results
    model.optimize(cb)

    if model.SolCount > 0:
        production_dict = {}
        stock_dict = {}
        backlog_dict = {}
        setup_dict = {}
        startup_dict = {}

        for i in range(NI):
            for k in range(NK):
                for t in range(NT):
                    key = f"{i}_{k}_{t}"
                    production_dict[key] = float(x[i, k, t].X)
                    setup_dict[key] = int(round(y[i, k, t].X))
                    startup_dict[key] = int(round(z[i, k, t].X))

        for i in range(NI):
            for t in range(NT):
                key = f"{i}_{t}"
                stock_dict[key] = float(s[i, t].X)
                backlog_dict[key] = float(r[i, t].X)

        solution = {
            "objective_value": float(model.ObjVal),
            "production": production_dict,
            "stock": stock_dict,
            "backlog": backlog_dict,
            "setup": setup_dict,
            "startup": startup_dict
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("Optimization successfully executed, but no feasible solution was found within the time limit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Item Multi-Machine Lotsizing with Backlogging and Setup Costs")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to input instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Location to save the final solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Time limit for optimization in seconds.")
    parser.add_argument('--log_path', type=str, required=False, help="Path prefix to track progressive bounds (if tracked).")
    args = parser.parse_args()
    
    solve_production_problem(args)