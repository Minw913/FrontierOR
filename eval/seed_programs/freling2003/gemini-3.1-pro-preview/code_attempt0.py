import json
import argparse
import gurobipy as gp

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    # Load problem instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    params = data['parameters']
    num_facilities = params['num_facilities']
    num_customers = params['num_customers']
    num_periods = params['num_periods']

    seasonal_factors = data['seasonal_factors']
    demands = data['demands']
    holding_costs = data['holding_costs']
    capacities = data['capacities']
    trans_costs = data['transportation_costs']

    # Precalculate demands and cumulative demands per customer over time
    d = []
    cum_d = []
    for c in range(num_customers):
        d_c = []
        cd_c = []
        current_sum = 0.0
        for t in range(num_periods):
            val = demands[c] * seasonal_factors[t]
            d_c.append(val)
            current_sum += val
            cd_c.append(current_sum)
        d.append(d_c)
        cum_d.append(cd_c)

    # Precalculate cumulative capacities
    cum_C = []
    for f in range(num_facilities):
        cC_f = []
        current_sum = 0.0
        for t in range(num_periods):
            current_sum += capacities[f][t]
            cC_f.append(current_sum)
        cum_C.append(cC_f)

    # Precalculate aggregate supply cost for each assignment over the horizon
    supply_cost = {}
    for f in range(num_facilities):
        for c in range(num_customers):
            supply_cost[f, c] = sum(trans_costs[f][c][t] for t in range(num_periods))

    # Initialize Gurobi specific data dicts
    cap_dict = {}
    hc_dict = {}
    for f in range(num_facilities):
        for t in range(num_periods):
            cap_dict[f, t] = capacities[f][t]
            hc_dict[f, t] = holding_costs[f][t]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()

    model = gp.Model("Logistics", env=env)
    model.ModelSense = gp.GRB.MINIMIZE
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Decision Variables
    # Assignment: binary flags
    z = model.addVars(num_facilities, num_customers, vtype=gp.GRB.BINARY, obj=supply_cost, name="z")
    # Production: continuous variables
    P = model.addVars(num_facilities, num_periods, lb=0, ub=cap_dict, vtype=gp.GRB.CONTINUOUS, name="P")
    # Inventory: continuous variables
    I = model.addVars(num_facilities, num_periods, lb=0, obj=hc_dict, vtype=gp.GRB.CONTINUOUS, name="I")

    # Assignment Constraints: Each customer must be assigned to exactly 1 facility
    for c in range(num_customers):
        model.addConstr(z.sum('*', c) == 1, name=f"assign_{c}")

    # Flow Balance Constraints & Capacity Bounds
    for f in range(num_facilities):
        for t in range(num_periods):
            # Formulate Demand of assigned customers for facility f in period t  
            c_dict = {(f, c): d[c][t] for c in range(num_customers) if d[c][t] != 0}
            if c_dict:
                D_ft = z.prod(c_dict)
            else:
                D_ft = 0.0

            # Calculate inventory level
            if t == 0:
                model.addConstr(I[f, t] == P[f, t] - D_ft, name=f"flow_{f}_{t}")
            else:
                model.addConstr(I[f, t] == I[f, t-1] + P[f, t] - D_ft, name=f"flow_{f}_{t}")

            # Include valid inequality cut to aggressively enforce feasibility over continuous time assignments 
            c_cum_dict = {(f, c): cum_d[c][t] for c in range(num_customers) if cum_d[c][t] != 0}
            if c_cum_dict:
                left_expr = z.prod(c_cum_dict)
                model.addConstr(left_expr <= cum_C[f][t], name=f"cum_cap_{f}_{t}")

    # Setup conditional logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    def cb(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Solve the MIP
    model.optimize(cb)

    # Process and emit best found valid solution via correct Output Schema
    if model.SolCount > 0:
        ans_assignment = {}
        for c in range(num_customers):
            for f in range(num_facilities):
                if z[f, c].X > 0.5:
                    ans_assignment[str(c)] = f
                    break
        
        ans_inventory = {}
        for f in range(num_facilities):
            ans_inventory[str(f)] = [max(0.0, float(I[f, t].X)) for t in range(num_periods)]
            
        solution = {
            "objective_value": float(model.ObjVal),
            "assignment": ans_assignment,
            "inventory": ans_inventory
        }
        
        with open(args.solution_path, 'w') as fh:
            json.dump(solution, fh, indent=2)

if __name__ == "__main__":
    solve()