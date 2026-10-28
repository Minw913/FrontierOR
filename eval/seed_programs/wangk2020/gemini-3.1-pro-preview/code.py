import argparse
import json
import os
import sys
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default="")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Setup logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    # Extract top-level parameters
    T = data['num_time_periods']
    M = data['maximum_displacement_delta']
    airports = data['airports']
    flights = data['flights']
    scenarios = data['scenarios']

    # Retrieve rho, assuming a default of 0.5 if it is missing
    rho = data.get("rho")
    if rho is None:
        rho = data.get("weight_parameter_rho", 0.5)

    flights_data = {f['id']: f for f in flights}
    scenario_ids = [s['id'] for s in scenarios]

    # Pre-compute valid bounds to minimize variable definitions
    LB_d, UB_d = {}, {}
    LB_a, UB_a = {}, {}
    LB_ds, UB_ds = {}, {}
    LB_as, UB_as = {}, {}

    for f in flights:
        f_id = f['id']
        dep_req = f['dep_period']
        delta_sch = f['delta_sch']
        delta_min = f['delta_min']
        delta_max = f['delta_max']
        max_dep_delay = f['max_dep_delay']
        
        # Scheduled bounds
        LB_d[f_id] = max(0, dep_req - M)
        UB_d[f_id] = min(T - 1 - delta_sch, dep_req + M)
        
        LB_a[f_id] = LB_d[f_id] + delta_sch
        UB_a[f_id] = UB_d[f_id] + delta_sch
        
        for s in scenarios:
            s_id = s['id']
            # Operated bounds
            LB_ds[(f_id, s_id)] = LB_d[f_id]
            UB_ds[(f_id, s_id)] = min(T - 1 - delta_min, UB_d[f_id] + max_dep_delay)
            
            LB_as[(f_id, s_id)] = LB_ds[(f_id, s_id)] + delta_min
            UB_as[(f_id, s_id)] = min(T - 1, UB_ds[(f_id, s_id)] + delta_max)

    # Initialize Gurobi model
    model = gp.Model("AirTrafficManagement")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # ----------------------------
    # 1. Variables Definition
    # ----------------------------
    vars_w_dep, vars_w_arr = {}, {}
    for f_id in flights_data:
        for t in range(LB_d[f_id] + 1, UB_d[f_id] + 1):
            vars_w_dep[f_id, t] = model.addVar(vtype=GRB.BINARY)
        for t in range(LB_a[f_id] + 1, UB_a[f_id] + 1):
            vars_w_arr[f_id, t] = model.addVar(vtype=GRB.BINARY)

    vars_x_dep, vars_x_arr = {}, {}
    for f_id in flights_data:
        for s_id in scenario_ids:
            for t in range(LB_ds[(f_id, s_id)] + 1, UB_ds[(f_id, s_id)] + 1):
                vars_x_dep[f_id, s_id, t] = model.addVar(vtype=GRB.BINARY)
            for t in range(LB_as[(f_id, s_id)] + 1, UB_as[(f_id, s_id)] + 1):
                vars_x_arr[f_id, s_id, t] = model.addVar(vtype=GRB.BINARY)

    vars_D = {f_id: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS) for f_id in flights_data}
    vars_v_dep = {(f_id, s_id): model.addVar(lb=0.0, ub=flights_data[f_id]['max_dep_delay'], vtype=GRB.CONTINUOUS) 
                  for f_id in flights_data for s_id in scenario_ids}
    vars_v_arr = {(f_id, s_id): model.addVar(lb=0.0, ub=flights_data[f_id]['max_arr_delay'], vtype=GRB.CONTINUOUS) 
                  for f_id in flights_data for s_id in scenario_ids}

    # Helper getters that seamlessly transition over boundary bounds
    def get_w_dep(f, t):
        if t <= LB_d[f]: return 1.0
        if t > UB_d[f]: return 0.0
        return vars_w_dep[f, t]

    def get_w_arr(f, t):
        if t <= LB_a[f]: return 1.0
        if t > UB_a[f]: return 0.0
        return vars_w_arr[f, t]

    def get_x_dep(f, s, t):
        if t <= LB_ds[(f, s)]: return 1.0
        if t > UB_ds[(f, s)]: return 0.0
        return vars_x_dep[f, s, t]

    def get_x_arr(f, s, t):
        if t <= LB_as[(f, s)]: return 1.0
        if t > UB_as[(f, s)]: return 0.0
        return vars_x_arr[f, s, t]

    # Prebuild sequence sums corresponding to exact event times
    expr_w_dep_sum = {}
    expr_w_arr_sum = {}
    expr_x_dep_sum = {}
    expr_x_arr_sum = {}

    for f_id, fdata in flights_data.items():
        expr_w_dep_sum[f_id] = LB_d[f_id] + 1 + gp.quicksum(vars_w_dep[f_id, t] for t in range(LB_d[f_id] + 1, UB_d[f_id] + 1))
        expr_w_arr_sum[f_id] = LB_a[f_id] + 1 + gp.quicksum(vars_w_arr[f_id, t] for t in range(LB_a[f_id] + 1, UB_a[f_id] + 1))
        
        # En-route Duration
        model.addConstr(expr_w_arr_sum[f_id] - expr_w_dep_sum[f_id] == fdata['delta_sch'])
        
        # Displacement cost extraction
        df_var = expr_w_dep_sum[f_id] - 1
        model.addConstr(vars_D[f_id] >= df_var - fdata['dep_period'])
        model.addConstr(vars_D[f_id] >= fdata['dep_period'] - df_var)
        
        for s_id in scenario_ids:
            expr_x_dep_sum[f_id, s_id] = LB_ds[(f_id, s_id)] + 1 + gp.quicksum(vars_x_dep[f_id, s_id, t] for t in range(LB_ds[(f_id, s_id)] + 1, UB_ds[(f_id, s_id)] + 1))
            expr_x_arr_sum[f_id, s_id] = LB_as[(f_id, s_id)] + 1 + gp.quicksum(vars_x_arr[f_id, s_id, t] for t in range(LB_as[(f_id, s_id)] + 1, UB_as[(f_id, s_id)] + 1))
            
            # En-Route bounds in scenario
            model.addConstr(expr_x_arr_sum[f_id, s_id] - expr_x_dep_sum[f_id, s_id] >= fdata['delta_min'])
            model.addConstr(expr_x_arr_sum[f_id, s_id] - expr_x_dep_sum[f_id, s_id] <= fdata['delta_max'])
            
            # Delays mapping
            model.addConstr(vars_v_dep[f_id, s_id] == expr_x_dep_sum[f_id, s_id] - expr_w_dep_sum[f_id])
            model.addConstr(vars_v_arr[f_id, s_id] >= expr_x_arr_sum[f_id, s_id] - expr_w_arr_sum[f_id])

    # ----------------------------
    # 2. Add Monotonicity Constraints
    # ----------------------------
    for f_id in flights_data:
        for t in range(LB_d[f_id] + 2, UB_d[f_id] + 1):
            model.addConstr(vars_w_dep[f_id, t-1] >= vars_w_dep[f_id, t])
        for t in range(LB_a[f_id] + 2, UB_a[f_id] + 1):
            model.addConstr(vars_w_arr[f_id, t-1] >= vars_w_arr[f_id, t])
            
        for s_id in scenario_ids:
            for t in range(LB_ds[(f_id, s_id)] + 2, UB_ds[(f_id, s_id)] + 1):
                model.addConstr(vars_x_dep[f_id, s_id, t-1] >= vars_x_dep[f_id, s_id, t])
            for t in range(LB_as[(f_id, s_id)] + 2, UB_as[(f_id, s_id)] + 1):
                model.addConstr(vars_x_arr[f_id, s_id, t-1] >= vars_x_arr[f_id, s_id, t])

    # ----------------------------
    # 3. Connection Constraints
    # ----------------------------
    conn_sch = {}
    for list_key in ['aircraft_connections', 'passenger_connections']:
        for conn in data.get(list_key, []):
            k = (conn['flight_i'], conn['flight_j'])
            conn_sch[k] = max(conn_sch.get(k, 0), conn['min_connection_time'])

    conn_op = {}
    for conn in data.get('aircraft_connections', []):
        k = (conn['flight_i'], conn['flight_j'])
        conn_op[k] = max(conn_op.get(k, 0), conn['min_connection_time'])

    for (f1, f2), m in conn_sch.items():
        model.addConstr(expr_w_dep_sum[f2] - expr_w_arr_sum[f1] >= m)

    for (f1, f2), m in conn_op.items():
        for s_id in scenario_ids:
            model.addConstr(expr_x_dep_sum[f2, s_id] - expr_x_arr_sum[f1, s_id] >= m)

    # ----------------------------
    # 4. Capacity Envelope Constraints
    # ----------------------------
    dep_vars = {k: {s_id: {p: [] for p in range(T)} for s_id in scenario_ids} for k in airports}
    dep_coeffs = {k: {s_id: {p: [] for p in range(T)} for s_id in scenario_ids} for k in airports}
    dep_cons = {k: {s_id: {p: 0.0 for p in range(T)} for s_id in scenario_ids} for k in airports}
    
    arr_vars = {k: {s_id: {p: [] for p in range(T)} for s_id in scenario_ids} for k in airports}
    arr_coeffs = {k: {s_id: {p: [] for p in range(T)} for s_id in scenario_ids} for k in airports}
    arr_cons = {k: {s_id: {p: 0.0 for p in range(T)} for s_id in scenario_ids} for k in airports}

    for f_id, fdata in flights_data.items():
        apt_dep = fdata['dep_airport']
        apt_arr = fdata['arr_airport']
        for s_id in scenario_ids:
            for p in range(LB_ds[(f_id, s_id)], UB_ds[(f_id, s_id)] + 1):
                v1 = get_x_dep(f_id, s_id, p)
                v2 = get_x_dep(f_id, s_id, p + 1)
                
                if isinstance(v1, float): dep_cons[apt_dep][s_id][p] += v1
                else: 
                    dep_vars[apt_dep][s_id][p].append(v1)
                    dep_coeffs[apt_dep][s_id][p].append(1.0)
                    
                if isinstance(v2, float): dep_cons[apt_dep][s_id][p] -= v2
                else: 
                    dep_vars[apt_dep][s_id][p].append(v2)
                    dep_coeffs[apt_dep][s_id][p].append(-1.0)
                    
            for p in range(LB_as[(f_id, s_id)], UB_as[(f_id, s_id)] + 1):
                v1 = get_x_arr(f_id, s_id, p)
                v2 = get_x_arr(f_id, s_id, p + 1)
                
                if isinstance(v1, float): arr_cons[apt_arr][s_id][p] += v1
                else:
                    arr_vars[apt_arr][s_id][p].append(v1)
                    arr_coeffs[apt_arr][s_id][p].append(1.0)
                    
                if isinstance(v2, float): arr_cons[apt_arr][s_id][p] -= v2
                else:
                    arr_vars[apt_arr][s_id][p].append(v2)
                    arr_coeffs[apt_arr][s_id][p].append(-1.0)

    for s in scenarios:
        s_id = s['id']
        for p in range(T):
            for k in airports:
                cond = s['operating_conditions'][k][p]
                env = data.get('capacity_envelopes', {}).get(k, {}).get(cond, [])
                if not env: continue
                
                d_expr = gp.LinExpr(dep_coeffs[k][s_id][p], dep_vars[k][s_id][p]) + dep_cons[k][s_id][p]
                a_expr = gp.LinExpr(arr_coeffs[k][s_id][p], arr_vars[k][s_id][p]) + arr_cons[k][s_id][p]
                
                for seg in env:
                    model.addConstr(seg['a'] * d_expr + seg['b'] * a_expr <= seg['Q'])

    # ----------------------------
    # 5. Objective setting & Core Solving
    # ----------------------------
    disp_cost = gp.quicksum(vars_D[f_id] for f_id in flights_data)
    delay_cost = gp.quicksum(
        s['probability'] * (
            flights_data[f_id]['cost_dep_delay'] * vars_v_dep[f_id, s['id']] +
            flights_data[f_id]['cost_arr_delay'] * vars_v_arr[f_id, s['id']]
        ) for s in scenarios for f_id in flights_data
    )

    model.setObjective(rho * disp_cost + (1 - rho) * delay_cost, GRB.MINIMIZE)

    def cb(mod, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj = mod.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(cb)

    if model.SolCount == 0:
        sys.exit(0)  # No feasible solution found in given limits

    # ----------------------------
    # 6. Extract Result
    # ----------------------------
    output = {
        "objective_value": model.ObjVal,
        "schedule": {},
        "scenario_solutions": []
    }

    for f_id in flights_data:
        d_f = int(round(expr_w_dep_sum[f_id].getValue() - 1))
        a_f = int(round(expr_w_arr_sum[f_id].getValue() - 1))
        output["schedule"][str(f_id)] = {
            "scheduled_dep": d_f,
            "scheduled_arr": a_f
        }
        
    for s in scenarios:
        s_id = s['id']
        s_sol = {"scenario_id": s_id, "x_dep": {}, "x_arr": {}, "v_dep": {}, "v_arr": {}}
        
        for f_id in flights_data:
            d_f = output["schedule"][str(f_id)]["scheduled_dep"]
            a_f = output["schedule"][str(f_id)]["scheduled_arr"]
            
            d_f_s = int(round(expr_x_dep_sum[f_id, s_id].getValue() - 1))
            a_f_s = int(round(expr_x_arr_sum[f_id, s_id].getValue() - 1))
            
            # Ensuring precision safety on bounding
            v_dep = max(0, d_f_s - d_f)
            v_arr = max(0, a_f_s - a_f)
            
            s_sol["v_dep"][str(f_id)] = v_dep
            s_sol["v_arr"][str(f_id)] = v_arr
            
            for p in range(T):
                s_sol["x_dep"][f"{f_id}_{p}"] = 1 if p <= d_f_s else 0
                s_sol["x_arr"][f"{f_id}_{p}"] = 1 if p <= a_f_s else 0
                
        output["scenario_solutions"].append(s_sol)

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    solve()