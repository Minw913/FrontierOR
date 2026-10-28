import argparse
import json
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Gurobi not found. Exiting.", file=sys.stderr)
    sys.exit(1)

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None
    # If the SolutionLogger is not available, proceed without logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    I = data['num_items']  # number of items
    T = data['num_periods']  # number of periods
    M = data['num_machines']  # number of machines

    machine_capabilities = data['machine_capabilities']
    capabilities_set = []
    for i in range(I):
        cap_set = set(m_val - 1 for m_val in machine_capabilities[i])  # convert to 0-indexed machine indices
        capabilities_set.append(cap_set)

    demand = data['demand']
    gross_remaining_demand = []
    for i in range(I):
        dem_list = demand[i]
        rd = [sum(dem_list[t:]) for t in range(T)]
        gross_remaining_demand.append(rd)

    initial_inv = data.get('initial_inventory', [0.0] * I)

    # Build dictionary keys for cost and time parameters (1-indexed)
    setup_time_dict = {}
    production_time_dict = {}
    setup_cost_dict = {}
    production_cost_dict = {}
    
    # Precomputed parameter dictionaries
    for i in range(I):
        cap_machines = capabilities_set[i]
        for m in cap_machines:
            key = f"item_{i+1}_machine_{m+1}"
            setup_time_dict[(i, m)] = data['setup_time'][key]
            production_time_dict[(i, m)] = data['production_time'][key]
            setup_cost_dict[(i, m)] = data['setup_cost'][key]
            production_cost_dict[(i, m)] = data['production_cost'][key]

    capacity = data['capacity']  # list of lists [machine][period], 0-indexed
    holding_cost = data['holding_cost']
    max_machines_per_item = data['max_machines_per_item_per_period']

    # Create model
    model = gp.Model()
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', False)
    model.setParam('Threads', 1)

    # Variables
    I_var = {}  # inventory for item i at end of period t: (i,t)
    for i in range(I):
        for t in range(T):
            I_var[(i, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"I_{i}_{t}")

    X = {}  # production quantity for item i on machine m in period t: (i,m,t)
    Y = {}  # setup indicator for item i on machine m in period t: (i,m,t)
    for i in range(I):
        cap_i = capabilities_set[i]
        for m in cap_i:
            for t in range(T):
                Y[(i, m, t)] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{m}_{t}")
                X[(i, m, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"X_{i}_{m}_{t}")

    # Flow conservation constraints
    for i in range(I):
        # Period 0
        cap_i = capabilities_set[i]
        expr = gp.quicksum(X[(i, m, 0)] for m in cap_i) - I_var[(i, 0)]
        expr += initial_inv[i]
        model.addConstr(expr == demand[i][0], name=f"flow_i{i}_t0")
        
        # Periods 1 to T-1
        for t in range(1, T):
            expr = gp.quicksum(X[(i, m, t)] for m in cap_i) + I_var[(i, t-1)] - I_var[(i, t)]
            model.addConstr(expr == demand[i][t], name=f"flow_i{i}_t{t}")

    # Capacity constraints
    for m in range(M):
        for t in range(T):
            expr = 0.0
            for i in range(I):
                if m in capabilities_set[i]:
                    st = setup_time_dict.get((i, m), 0)
                    pt = production_time_dict.get((i, m), 0)
                    expr += pt * X[(i, m, t)] + st * Y[(i, m, t)]
            model.addConstr(expr <= capacity[m][t], name=f"cap_m{m}_t{t}")

    # Linking constraints (setup and production bounds)
    for i in range(I):
        cap_i = capabilities_set[i]
        for m in cap_i:
            st = setup_time_dict.get((i, m))
            pt = production_time_dict.get((i, m))
            for t in range(T):
                cap_val = capacity[m][t]
                if cap_val < st:
                    M_val = 0.0
                else:
                    M_val = (cap_val - st) / pt
                total_demand = gross_remaining_demand[i][t]
                up_bound = min(M_val, total_demand)
                model.addConstr(X[(i, m, t)] <= up_bound * Y[(i, m, t)], name=f"link_{i}_{m}_{t}")

    # Maximum machines per item per period
    for i in range(I):
        cap_i = capabilities_set[i]
        max_mach = max_machines_per_item[i]
        for t in range(T):
            model.addConstr(gp.quicksum(Y[(i, m, t)] for m in cap_i) <= max_mach, name=f"max_mach_i{i}_{t}")

    # Objective function
    total_cost = 0.0
    
    # Setup costs
    for i in range(I):
        cap_i = capabilities_set[i]
        for m in cap_i:
            sc = setup_cost_dict.get((i, m))
            for t in range(T):
                total_cost += sc * Y[(i, m, t)]
    
    # Production costs
    for i in range(I):
        cap_i = capabilities_set[i]
        for m in cap_i:
            pc = production_cost_dict.get((i, m))
            for t in range(T):
                total_cost += pc * X[(i, m, t)]
    
    # Inventory costs
    for i in range(I):
        hc = holding_cost[i]
        for t in range(T):
            total_cost += hc * I_var[(i, t)]
    
    model.setObjective(total_cost, sense=GRB.MINIMIZE)

    # Callback for logging
    if logger:
        def callback_function(model, where):
            if where == GRB.Callback.MIPSOL:
                # Get the current best objective value
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback_function)
    else:
        model.optimize()

    # Prepare output
    output = {
        "objective_value": None,
        "production": {},
        "inventory": {},
        "setups": {}
    }

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        output["objective_value"] = model.ObjVal
        # Extract inventory variables
        for i in range(I):
            for t in range(T):
                key = f"I_item{i+1}_period{t+1}"
                inv_var = I_var.get((i,t))
                output["inventory"][key] = inv_var.X if inv_var is not None else 0.0
        # Extract production and setup variables
        for i in range(I):
            cap_i = capabilities_set[i]
            for m in cap_i:
                for t in range(T):
                    key_prod = f"X_item{i+1}_machine{m+1}_period{t+1}"
                    key_setup = f"Y_item{i+1}_machine{m+1}_period{t+1}"
                    prod_var = X.get((i, m, t))
                    setup_var = Y.get((i, m, t))
                    output["production"][key_prod] = prod_var.X if prod_var is not None else 0.0
                    output["setups"][key_setup] = int(round(setup_var.X)) if setup_var is not None else 0
    else:
        # If no solution was found, return zeros for all variables
        for i in range(I):
            for t in range(T):
                key = f"I_item{i+1}_period{t+1}"
                output["inventory"][key] = 0.0
        for i in range(I):
            cap_i = capabilities_set[i]
            for m in cap_i:
                for t in range(T):
                    key_prod = f"X_item{i+1}_machine{m+1}_period{t+1}"
                    key_setup = f"Y_item{i+1}_machine{m+1}_period{t+1}"
                    output["production"][key_prod] = 0.0
                    output["setups"][key_setup] = 0

    # Write the solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()