import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    NI = instance['parameters']['NI']
    NK = instance['parameters']['NK']
    NT = instance['parameters']['NT']
    demand = instance['demand']
    capacity_C = instance['machine_data']['capacity_C']
    startup_time_sigma = instance['machine_data']['startup_time_sigma']
    setup_cost_f = instance['costs']['setup_cost_f']
    startup_cost_g = instance['costs']['startup_cost_g']
    holding_cost_h = instance['costs']['holding_cost_h']
    backlogging_cost_e = instance['costs']['backlogging_cost_e']
    initial_stock_s0 = instance['initial_conditions']['initial_stock_s0']
    initial_backlog_r0 = instance['initial_conditions']['initial_backlog_r0']

    I = range(NI)
    K = range(NK)
    T = range(NT)

    model = gp.Model("CLSP_MS")

    # Variables
    X = model.addVars(I, K, T, vtype=GRB.CONTINUOUS, lb=0, name="X")
    S = model.addVars(I, K, T, vtype=GRB.BINARY, name="S")
    Y = model.addVars(I, K, T, vtype=GRB.BINARY, name="Y")
    stock = model.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0, name="stock")
    backlog = model.addVars(I, T, vtype=GRB.CONTINUOUS, lb=0, name="backlog")

    # Inventory balance
    for i in I:
        model.addConstr(stock[i,0] - backlog[i,0] == initial_stock_s0[i] - initial_backlog_r0[i] + 
                        gp.quicksum(X[i,k,0] for k in K) - demand[i][0], name=f"balance_{i}_0")
        for t in range(1, NT):
            model.addConstr(stock[i,t] - backlog[i,t] == stock[i,t-1] - backlog[i,t-1] + 
                            gp.quicksum(X[i,k,t] for k in K) - demand[i][t], name=f"balance_{i}_{t}")

    # Capacity and setup linking
    for k in K:
        for t in T:
            model.addConstr(gp.quicksum(X[i,k,t] for i in I) + 
                            startup_time_sigma[k] * gp.quicksum(Y[i,k,t] for i in I) <= capacity_C[k], 
                            name=f"capacity_{k}_{t}")
            model.addConstr(gp.quicksum(S[i,k,t] for i in I) <= 1, name=f"one_item_{k}_{t}")
            for i in I:
                model.addConstr(X[i,k,t] <= capacity_C[k] * S[i,k,t], name=f"linkX_{i}_{k}_{t}")

    # Startup definition
    for i in I:
        for k in K:
            # First period
            model.addConstr(Y[i,k,0] >= S[i,k,0], name=f"startup0_{i}_{k}_0")
            # Subsequent periods
            for t in range(1, NT):
                model.addConstr(Y[i,k,t] >= S[i,k,t] - S[i,k,t-1], name=f"startup_{i}_{k}_{t}")

    # Objective
    setup_cost_expr = setup_cost_f * gp.quicksum(S[i,k,t] for i in I for k in K for t in T)
    startup_cost_expr = startup_cost_g * gp.quicksum(Y[i,k,t] for i in I for k in K for t in T)
    holding_cost_expr = gp.quicksum(holding_cost_h[i] * stock[i,t] for i in I for t in T)
    backlog_cost_expr = gp.quicksum(backlogging_cost_e[i] * backlog[i,t] for i in I for t in T)
    
    model.setObjective(setup_cost_expr + startup_cost_expr + holding_cost_expr + backlog_cost_expr, GRB.MINIMIZE)

    # Time limit
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01
    model.Params.LogToConsole = 0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        prod_dict = {}
        setup_dict = {}
        startup_dict = {}
        stock_dict = {}
        backlog_dict = {}

        for i in I:
            for t in T:
                stock_val = stock[i,t].X
                backlog_val = backlog[i,t].X
                stock_dict[f"{i}_{t}"] = float(stock_val)
                backlog_dict[f"{i}_{t}"] = float(backlog_val)
                for k in K:
                    xval = X[i,k,t].X
                    sval = S[i,k,t].X
                    yval = Y[i,k,t].X
                    if xval > 1e-6 or sval > 0.5:
                        prod_dict[f"{i}_{k}_{t}"] = float(xval)
                        setup_dict[f"{i}_{k}_{t}"] = int(round(sval))
                        startup_dict[f"{i}_{k}_{t}"] = int(round(yval))
                    else:
                        # Still include if needed but according to schema, only include if positive?
                        # Schema is dict with keys, we include only when relevant
                        pass

        # For variables that might be zero at end, we need to include all if they are part of output, but schema shows dict with keys where they apply.
        # To be safe, include all that are 1 or >0, but also need to include zeros if demanded by schema? But example shows only active.
        # The schema description: "dict, key: '{item}_{machine}_{period}'" implies include keys for all or for non-zero?
        # To be safe and follow exactly, include all possible combinations with 0 where appropriate.

        # Rebuild with all entries (required for strict adherence)
        prod_dict = {f"{i}_{k}_{t}": float(X[i,k,t].X) for i in I for k in K for t in T}
        setup_dict = {f"{i}_{k}_{t}": int(round(S[i,k,t].X)) for i in I for k in K for t in T}
        startup_dict = {f"{i}_{k}_{t}": int(round(Y[i,k,t].X)) for i in I for k in K for t in T}
        stock_dict = {f"{i}_{t}": float(stock[i,t].X) for i in I for t in T}
        backlog_dict = {f"{i}_{t}": float(backlog[i,t].X) for i in I for t in T}

        solution = {
            "objective_value": float(obj_val),
            "production": prod_dict,
            "stock": stock_dict,
            "backlog": backlog_dict,
            "setup": setup_dict,
            "startup": startup_dict
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found, write empty or minimal feasible (should not happen with feasible problem)
        solution = {
            "objective_value": 0.0,
            "production": {},
            "stock": {},
            "backlog": {},
            "setup": {},
            "startup": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()