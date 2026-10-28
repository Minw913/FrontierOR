import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def clean_float(val):
    return 0.0 if abs(val) < 1e-8 else float(val)

def solve():
    parser = argparse.ArgumentParser(description="Capacitated Lot Sizing Problem Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=None, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_items']
    T = data['num_periods']
    demand = data['demand']
    setup_cost = data['setup_cost']
    holding_cost = data['holding_cost']
    variable_cost = data['variable_cost']
    setup_time = data['setup_time']
    variable_time = data['variable_time']
    capacity = data['capacity']
    ini_inv_cost = data['initial_inventory_cost']

    # Initialize Solution Logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Initialize Gurobi model setup
    model = gp.Model("LotSizing")
    if args.time_limit is not None:
        model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    # OutputFlag can be kept at default (1) or turned off depending on preference, default is fine.

    I = {}
    P = {}
    Y = {}

    # Define variables
    for i in range(N):
        # I[i, t] stores inventory available right at the start of period t.
        # So I[i, 0] is the initial inventory, and I[i, T] is the final final inventory.
        for t in range(T + 1):
            I[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"I_{i}_{t}")
        for t in range(T):
            P[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"P_{i}_{t}")
            Y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{t}")

    # Inventory balance constraints and final inventory closure
    for i in range(N):
        for t in range(T):
            model.addConstr(I[i, t] + P[i, t] == demand[i][t] + I[i, t+1], name=f"Bal_{i}_{t}")
        model.addConstr(I[i, T] == 0, name=f"FinalInv_{i}")

    # Capacity constraints for each period
    for t in range(T):
        model.addConstr(
            gp.quicksum(setup_time[i] * Y[i, t] + variable_time[i] * P[i, t] for i in range(N)) <= capacity[t],
            name=f"Cap_{t}"
        )

    # Calculate effective max capacity for bounding usage per item-period combination
    max_cap = {}
    for i in range(N):
        for t in range(T):
            if variable_time[i] > 0:
                mc = (capacity[t] - setup_time[i]) / variable_time[i]
                max_cap[i, t] = max(0.0, mc)
            else:
                max_cap[i, t] = 0.0 if setup_time[i] > capacity[t] else float('inf')

    # Setup forcing constraints and upper bounding based on problem limits
    for i in range(N):
        for t in range(T):
            rem_demand = sum(demand[i][k] for k in range(t, T))
            bound = min(rem_demand, max_cap[i, t])
            model.addConstr(P[i, t] <= bound * Y[i, t], name=f"Setup_{i}_{t}")

    # Strong Valid Inequalities: Standard lot-sizing Interval/Subset valid inequalities
    D = {}
    for i in range(N):
        for t in range(T):
            curr_sum = 0
            for k in range(t, T):
                curr_sum += demand[i][k]
                D[i, t, k] = curr_sum

    for i in range(N):
        for t in range(T):
            for k in range(t, T):
                if D[i, t, k] > 0:
                    lhs = I[i, t] + gp.quicksum(
                        min(max_cap[i, tau], D[i, tau, k]) * Y[i, tau]
                        for tau in range(t, k+1)
                    )
                    model.addConstr(lhs >= D[i, t, k], name=f"VI_{i}_{t}_{k}")

    # Objective function spanning inventory, variable production, setups, and start inventory
    obj = gp.quicksum(ini_inv_cost[i] * I[i, 0] for i in range(N)) + \
          gp.quicksum(setup_cost[i] * Y[i, t] for i in range(N) for t in range(T)) + \
          gp.quicksum(variable_cost[i] * P[i, t] for i in range(N) for t in range(T)) + \
          gp.quicksum(holding_cost[i] * I[i, t+1] for i in range(N) for t in range(T))
    
    model.setObjective(obj, GRB.MINIMIZE)

    # Logging intermediate solutions bounds setup
    best_obj = [float('inf')]
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-5:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    # Solve the model natively
    model.optimize(callback)

    if model.SolCount > 0:
        # Retrieve optimal solution and cleanup structures
        P_res = [[clean_float(P[i, t].X) for t in range(T)] for i in range(N)]
        Y_res = [[int(round(Y[i, t].X)) for t in range(T)] for i in range(N)]
        I_res = [[clean_float(I[i, t].X) for t in range(T + 1)] for i in range(N)]

        # Recalculating objective mathematically to ensure precision with the schema specifications
        final_obj_val = 0.0
        for i in range(N):
            final_obj_val += ini_inv_cost[i] * I_res[i][0]
            for t in range(T):
                final_obj_val += setup_cost[i] * Y_res[i][t]
                final_obj_val += variable_cost[i] * P_res[i][t]
                final_obj_val += holding_cost[i] * I_res[i][t+1]

        # Consolidate result schema wrapper
        solution = {
            "objective_value": final_obj_val,
            "production": P_res,
            "setup": Y_res,
            "inventory": I_res
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution found within the specified limits.")

if __name__ == "__main__":
    solve()