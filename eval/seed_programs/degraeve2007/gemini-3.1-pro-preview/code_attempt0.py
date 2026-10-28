import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Solve manufacturing lot-sizing problem.")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the output solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path to a JSONL file for logging intermediate solutions.")
    args = parser.parse_args()

    # Initialize SolutionLogger if requested
    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    dims = data['dimensions']
    num_products = dims['num_products']
    num_periods = dims['num_periods']

    params = data['parameters']
    demand = params['demand']
    holding_cost = params['holding_cost']
    setup_cost = params['setup_cost']
    vp_cost = params['variable_production_cost']
    vp_time = params['variable_production_time']
    setup_time = params['setup_time']
    capacity = params['capacity']
    ii_cost = params['initial_inventory_cost']

    # Initialize Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    
    model = gp.Model("LotSizing", env=env)

    # Apply execution constraints
    model.setParam('Threads', 1)
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)

    # Define decision variables
    X = [[model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"prod_{p}_{t}") for t in range(num_periods)] for p in range(num_products)]
    Y = [[model.addVar(vtype=GRB.BINARY, name=f"setup_{p}_{t}") for t in range(num_periods)] for p in range(num_products)]
    I = [[model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"inv_{p}_{t}") for t in range(num_periods)] for p in range(num_products)]
    I0 = [model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"init_inv_{p}") for p in range(num_products)]

    # Formulate constraints
    for p in range(num_products):
        for t in range(num_periods):
            # Demand satisfaction / Flow balance constraints
            if t == 0:
                model.addConstr(I0[p] + X[p][t] == demand[p][t] + I[p][t], name=f"flow_bal_{p}_{t}")
            else:
                model.addConstr(I[p][t-1] + X[p][t] == demand[p][t] + I[p][t], name=f"flow_bal_{p}_{t}")

            # Production bounded by minimum of relevant constraints and logical constraint linking to setup indicator
            rem_demand = sum(demand[p][k] for k in range(t, num_periods))
            v_time = vp_time[p][t]
            s_time = setup_time[p][t]
            cap_period = capacity[t]
            
            if v_time > 0:
                max_cap = max(0.0, (cap_period - s_time) / v_time)
            else:
                max_cap = float('inf')
            
            big_M = min(rem_demand, max_cap)
            model.addConstr(X[p][t] <= big_M * Y[p][t], name=f"prod_bound_{p}_{t}")

    for t in range(num_periods):
        # Overall period capacity threshold constraints
        model.addConstr(
            gp.quicksum(setup_time[p][t] * Y[p][t] + vp_time[p][t] * X[p][t] for p in range(num_products)) <= capacity[t],
            name=f"capacity_{t}"
        )

    # Formulate objective structure
    total_cost = gp.quicksum(ii_cost[p] * I0[p] for p in range(num_products)) + \
                 gp.quicksum(setup_cost[p][t] * Y[p][t] for p in range(num_products) for t in range(num_periods)) + \
                 gp.quicksum(vp_cost[p][t] * X[p][t] for p in range(num_products) for t in range(num_periods)) + \
                 gp.quicksum(holding_cost[p][t] * I[p][t] for p in range(num_products) for t in range(num_periods))

    model.setObjective(total_cost, GRB.MINIMIZE)

    # Solution logging integration
    best_obj = [float('inf')]

    def solver_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-5:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    # Optimize models
    model.optimize(solver_callback)

    # Extract final decision distributions securely
    if model.SolCount > 0:
        obj_val = model.ObjVal
        prod_res = [[X[p][t].X for t in range(num_periods)] for p in range(num_products)]
        setup_res = [[int(round(Y[p][t].X)) for t in range(num_periods)] for p in range(num_products)]
        inv_res = [[I[p][t].X for t in range(num_periods)] for p in range(num_products)]
        init_inv_res = [I0[p].X for p in range(num_products)]
    else:
        # Fallback to an absolute trivial valid state leveraging initial inventory penalty
        prod_res = [[0.0] * num_periods for _ in range(num_products)]
        setup_res = [[0] * num_periods for _ in range(num_products)]
        inv_res = [[0.0] * num_periods for _ in range(num_products)]
        init_inv_res = [float(sum(demand[p])) for p in range(num_products)]
        
        obj_val = 0.0
        for p in range(num_products):
            obj_val += ii_cost[p] * init_inv_res[p]
            curr_inv = init_inv_res[p]
            for t in range(num_periods):
                curr_inv -= demand[p][t]
                inv_res[p][t] = curr_inv
                obj_val += holding_cost[p][t] * curr_inv

    # Construct the finalized payload correlating uniformly to the defined solution schema
    solution = {
        "objective_value": obj_val,
        "variables": {
            "production": prod_res,
            "setup": setup_res,
            "inventory": inv_res,
            "initial_inventory": init_inv_res
        }
    }

    # Stream out artifacts
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()