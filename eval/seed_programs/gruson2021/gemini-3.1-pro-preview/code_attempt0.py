import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def solve_problem(instance_path, solution_path, time_limit, log_path):
    # Parse instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
        
    num_periods = instance['num_periods']
    W = instance['num_warehouses']
    R = instance['num_retailers']
    S = instance['num_scenarios']
    probs = instance['scenario_probabilities']
    h_P = instance['holding_cost_plant']
    h_W = instance['holding_cost_warehouse']
    h_R = instance['holding_costs_retailers']
    
    sc_P = instance['setup_costs_plant']
    sc_W = instance['setup_costs_warehouses']
    sc_R = instance['setup_costs_retailers']
    
    demands = instance['demands']
    r2w = instance['retailer_to_warehouse']
    init_setups = instance.get('initial_setups_imposed', False)

    # Initialize Environment
    # Ensure execution is restricted to a single CPU core
    env = gp.Env()
    env.setParam("Threads", 1)
    
    model = gp.Model("SupplyChain", env=env)
    
    if time_limit is not None and time_limit > 0:
        model.setParam('TimeLimit', time_limit)
        
    # Initialize logger if a valid path is provided
    logger = None
    if log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(log_path, sense="minimize")
        except ImportError:
            pass
        
    def log_cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    # Model Variables
    y_P = model.addVars(num_periods, vtype=GRB.BINARY, name="y_P")
    y_W = model.addVars(W, num_periods, vtype=GRB.BINARY, name="y_W")
    y_R = model.addVars(R, num_periods, vtype=GRB.BINARY, name="y_R")
    
    # Apply initial setups constraint if imposed
    if init_setups:
        y_P[0].lb = 1
        for w in range(W):
            y_W[w, 0].lb = 1
        for i in range(R):
            y_R[i, 0].lb = 1

    # Disaggregated flow variables (multi-commodity facility location representation over demand fractions)
    p = {}
    w_var = {}
    r_var = {}
    ip = {}
    iw = {}
    ir = {}
    
    for xi in range(S):
        for i in range(R):
            for t in range(num_periods):
                D = demands[xi][i][t]
                if D > 0:
                    for tau in range(t + 1):
                        p[xi, i, t, tau] = model.addVar(lb=0, ub=1, name=f"p_{xi}_{i}_{t}_{tau}")
                        w_var[xi, i, t, tau] = model.addVar(lb=0, ub=1, name=f"w_{xi}_{i}_{t}_{tau}")
                        r_var[xi, i, t, tau] = model.addVar(lb=0, ub=1, name=f"r_{xi}_{i}_{t}_{tau}")
                        
                        # Inventory valid only to carry flow to next period (tau -> tau + 1)
                        if tau < t:
                            ip[xi, i, t, tau] = model.addVar(lb=0, ub=1, name=f"ip_{xi}_{i}_{t}_{tau}")
                            iw[xi, i, t, tau] = model.addVar(lb=0, ub=1, name=f"iw_{xi}_{i}_{t}_{tau}")
                            ir[xi, i, t, tau] = model.addVar(lb=0, ub=1, name=f"ir_{xi}_{i}_{t}_{tau}")

    # Objective - Setup costs component
    obj_setup = gp.quicksum(sc_P[tau] * y_P[tau] for tau in range(num_periods)) + \
                gp.quicksum(sc_W[w][tau] * y_W[w, tau] for w in range(W) for tau in range(num_periods)) + \
                gp.quicksum(sc_R[i][tau] * y_R[i, tau] for i in range(R) for tau in range(num_periods))

    # Objective - Expected inventory holding costs component
    obj_held = gp.LinExpr()
    
    for xi in range(S):
        prob = probs[xi]
        for i in range(R):
            w_idx = r2w[i]
            for t in range(num_periods):
                D = demands[xi][i][t]
                if D > 0:
                    for tau in range(t + 1):
                        # Strong setup forcing constraints
                        model.addConstr(p[xi, i, t, tau] <= y_P[tau])
                        model.addConstr(w_var[xi, i, t, tau] <= y_W[w_idx, tau])
                        model.addConstr(r_var[xi, i, t, tau] <= y_R[i, tau])
                        
                        # Flow conservation constraints
                        in_p = p[xi, i, t, tau] + (ip[xi, i, t, tau-1] if tau > 0 else 0)
                        out_p = w_var[xi, i, t, tau] + (ip[xi, i, t, tau] if tau < t else 0)
                        model.addConstr(in_p == out_p)
                        
                        in_w = w_var[xi, i, t, tau] + (iw[xi, i, t, tau-1] if tau > 0 else 0)
                        out_w = r_var[xi, i, t, tau] + (iw[xi, i, t, tau] if tau < t else 0)
                        model.addConstr(in_w == out_w)
                        
                        in_r = r_var[xi, i, t, tau] + (ir[xi, i, t, tau-1] if tau > 0 else 0)
                        # Demands structurally satisfy bounds natively, resolving entirely to consumption effectively at demand due period
                        out_r = (ir[xi, i, t, tau] if tau < t else 1.0)
                        model.addConstr(in_r == out_r)

                        # Accumulate objective terms for the carryover balances explicitly modeling exact holding times
                        if tau < t:
                            holding_cost_component = (h_P * ip[xi, i, t, tau]) + \
                                                     (h_W * iw[xi, i, t, tau]) + \
                                                     (h_R[i] * ir[xi, i, t, tau])
                            obj_held += prob * D * holding_cost_component

    model.setObjective(obj_setup + obj_held, GRB.MINIMIZE)
    
    # Solve
    model.optimize(log_cb)
    
    # Output Solution Tracking Verification and Formatting
    if model.SolCount > 0:
        setup_vars = {}
        for period in range(num_periods):
            # Facility 0: Plant
            setup_vars[f"y_0_{period}"] = int(round(y_P[period].X))
            # Facilities 1..W: Warehouses
            for w in range(W):
                setup_vars[f"y_{1+w}_{period}"] = int(round(y_W[w, period].X))
            # Facilities W+1..W+R: Retailers
            for i in range(R):
                setup_vars[f"y_{1+W+i}_{period}"] = int(round(y_R[i, period].X))
                
        prod_plant = {}
        deliv_wh = {}
        deliv_ret = {}
        inv_plant = {}
        inv_wh = {}
        inv_ret = {}
        
        for period in range(num_periods):
            for xi in range(S):
                p_val = 0.0
                w_val = [0.0] * W
                r_val = [0.0] * R
                ip_val = 0.0
                iw_val = [0.0] * W
                ir_val = [0.0] * R
                
                for i in range(R):
                    w_idx = r2w[i]
                    for t in range(period, num_periods):
                        D = demands[xi][i][t]
                        if D > 0:
                            p_val += D * p[xi, i, t, period].X
                            w_val[w_idx] += D * w_var[xi, i, t, period].X
                            r_val[i] += D * r_var[xi, i, t, period].X
                            
                            if period < t:
                                ip_val += D * ip[xi, i, t, period].X
                                iw_val[w_idx] += D * iw[xi, i, t, period].X
                                ir_val[i] += D * ir[xi, i, t, period].X
                                
                prod_plant[f"{period}_{xi}"] = round(p_val, 4)
                inv_plant[f"{period}_{xi}"] = round(ip_val, 4)
                
                for w in range(W):
                    deliv_wh[f"{w}_{period}_{xi}"] = round(w_val[w], 4)
                    inv_wh[f"{w}_{period}_{xi}"] = round(iw_val[w], 4)
                    
                for i in range(R):
                    deliv_ret[f"{i}_{period}_{xi}"] = round(r_val[i], 4)
                    inv_ret[f"{i}_{period}_{xi}"] = round(ir_val[i], 4)
                    
        result = {
            "objective_value": model.ObjVal,
            "setup_variables": setup_vars,
            "production_plant": prod_plant,
            "delivery_warehouse": deliv_wh,
            "delivery_retailer": deliv_ret,
            "inventory_plant": inv_plant,
            "inventory_warehouse": inv_wh,
            "inventory_retailer": inv_ret
        }
        
        with open(solution_path, 'w') as f:
            json.dump(result, f, indent=4)
    else:
        print("No feasible solution found within the specified time limit.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Three-Level Supply Chain LP/MIP Solver")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to input instance JSON")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to output solution JSON")
    parser.add_argument('--time_limit', type=int, default=0, help="Maximum run time in seconds")
    parser.add_argument('--log_path', type=str, default="", help="Path for JSONL incumbent solution logger")
    
    args = parser.parse_args()
    solve_problem(args.instance_path, args.solution_path, args.time_limit, args.log_path)