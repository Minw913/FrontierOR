import json
import sys
import os
import gurobipy as gp
from gurobipy import GRB

def main():
    if len(sys.argv) != 3:
        print("Usage: python unit_commitment.py <instance_path> <solution_path>")
        sys.exit(1)
    
    instance_path = sys.argv[1]
    solution_path = sys.argv[2]
    
    with open(instance_path) as f:
        instance = json.load(f)
    
    T = instance['num_time_periods']
    generators = instance['generators']
    demand = instance['demand']
    reserve_requirement = instance['reserve_requirement']
    time_period_hours = instance['time_period_hours']
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    u = {}
    p = {}
    pb = {}
    v = {}
    w = {}
    L = {}
    L0 = {}
    D_val = {}
    cost_start = {}
    cost_prod = {}
    z = {}
    
    L_max = T + max(-gen['initial_status'] if gen['initial_status'] < 0 else 0 for gen in generators) + 1
    
    for gen_idx, gen in enumerate(generators):
        for t in range(T):
            u[gen_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{gen_idx}_{t}")
            p[gen_idx, t] = model.addVar(lb=0, ub=gen['p_max'], name=f"p_{gen_idx}_{t}")
            pb[gen_idx, t] = model.addVar(lb=0, ub=gen['p_max'], name=f"pb_{gen_idx}_{t}")
            v[gen_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"v_{gen_idx}_{t}")
            w[gen_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{gen_idx}_{t}")
            L[gen_idx, t] = model.addVar(lb=0, ub=L_max, name=f"L_{gen_idx}_{t}")
            L0[gen_idx, t] = model.addVar(lb=0, ub=L_max, name=f"L0_{gen_idx}_{t}")
            D_val[gen_idx, t] = model.addVar(lb=0, ub=L_max*time_period_hours, name=f"D_val_{gen_idx}_{t}")
            cost_start[gen_idx, t] = model.addVar(name=f"cost_start_{gen_idx}_{t}")
            cost_prod[gen_idx, t] = model.addVar(name=f"cost_prod_{gen_idx}_{t}")
            
            startup_tiers = gen['startup_cost']
            num_tiers = len(startup_tiers)
            for k in range(num_tiers):
                z[gen_idx, t, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{gen_idx}_{t}_{k}")
    
    for gen_idx, gen in enumerate(generators):
        initial_status = gen['initial_status']
        min_up_time = gen['min_up_time']
        min_down_time = gen['min_down_time']
        
        # Initial status constraints
        if initial_status > 0:
            remaining_up = max(0, min_up_time - initial_status)
            for t in range(remaining_up):
                model.addConstr(u[gen_idx, t] == 1, name=f"initial_up_{gen_idx}_{t}")
        elif initial_status < 0:
            remaining_down = max(0, min_down_time + initial_status)
            for t in range(remaining_down):
                model.addConstr(u[gen_idx, t] == 0, name=f"initial_down_{gen_idx}_{t}")
        
        # Consecutive off periods for t=0
        L_initial = -initial_status if initial_status < 0 else 0
        model.addConstr(L0[gen_idx, 0] <= L_max * (1 - u[gen_idx, 0]), name=f"L0_ub_0_{gen_idx}")
        model.addConstr(L0[gen_idx, 0] <= L_initial + 1, name=f"L0_ub2_0_{gen_idx}")
        model.addConstr(L0[gen_idx, 0] >= L_initial + 1 - L_max * u[gen_idx, 0], name=f"L0_lb_0_{gen_idx}")
        model.addConstr(L[gen_idx, 0] == L0[gen_idx, 0], name=f"L_def_0_{gen_idx}")
        
        # Startup cost downtime for t=0
        model.addConstr(D_val[gen_idx, 0] == L_initial * time_period_hours, name=f"D_val_0_{gen_idx}")
        
        for t in range(1, T):
            # Consecutive off periods
            model.addConstr(L0[gen_idx, t] <= L_max * (1 - u[gen_idx, t]), name=f"L0_ub_{gen_idx}_{t}")
            model.addConstr(L0[gen_idx, t] <= L[gen_idx, t-1] + 1, name=f"L0_ub2_{gen_idx}_{t}")
            model.addConstr(L0[gen_idx, t] >= L[gen_idx, t-1] + 1 - L_max * u[gen_idx, t], name=f"L0_lb_{gen_idx}_{t}")
            model.addConstr(L[gen_idx, t] == L0[gen_idx, t], name=f"L_def_{gen_idx}_{t}")
            
            # Startup cost downtime
            model.addConstr(D_val[gen_idx, t] == L[gen_idx, t-1] * time_period_hours, name=f"D_val_{gen_idx}_{t}")
        
        for t in range(T):
            # Power and reserve bounds
            model.addConstr(p[gen_idx, t] >= gen['p_min'] * u[gen_idx, t], name=f"p_min_{gen_idx}_{t}")
            model.addConstr(p[gen_idx, t] <= gen['p_max'] * u[gen_idx, t], name=f"p_max_{gen_idx}_{t}")
            model.addConstr(pb[gen_idx, t] >= p[gen_idx, t], name=f"pb_lb_{gen_idx}_{t}")
            model.addConstr(pb[gen_idx, t] <= gen['p_max'] * u[gen_idx, t], name=f"pb_ub_{gen_idx}_{t}")
            
            # Startup and shutdown variables
            if t == 0:
                if initial_status < 0:
                    model.addConstr(v[gen_idx, t] == u[gen_idx, t], name=f"v0_{gen_idx}")
                else:
                    model.addConstr(v[gen_idx, t] == 0, name=f"v0_{gen_idx}")
                model.addConstr(w[gen_idx, t] == 0, name=f"w0_{gen_idx}")
            else:
                model.addConstr(v[gen_idx, t] >= u[gen_idx, t] - u[gen_idx, t-1], name=f"v_lb_{gen_idx}_{t}")
                model.addConstr(v[gen_idx, t] <= u[gen_idx, t], name=f"v_ub1_{gen_idx}_{t}")
                model.addConstr(v[gen_idx, t] <= 1 - u[gen_idx, t-1], name=f"v_ub2_{gen_idx}_{t}")
                model.addConstr(w[gen_idx, t] >= u[gen_idx, t-1] - u[gen_idx, t], name=f"w_lb_{gen_idx}_{t}")
                model.addConstr(w[gen_idx, t] <= u[gen_idx, t-1], name=f"w_ub1_{gen_idx}_{t}")
                model.addConstr(w[gen_idx, t] <= 1 - u[gen_idx, t], name=f"w_ub2_{gen_idx}_{t}")
            
            # Ramping constraints
            if t >= 1:
                # Online ramp up/down
                model.addConstr(p[gen_idx, t] - p[gen_idx, t-1] <= gen['ramp_up_limit'] + gen['p_max'] * (1 - u[gen_idx, t-1]) + gen['p_max'] * (1 - u[gen_idx, t]), name=f"ramp_up_{gen_idx}_{t}")
                model.addConstr(p[gen_idx, t-1] - p[gen_idx, t] <= gen['ramp_down_limit'] + gen['p_max'] * (1 - u[gen_idx, t-1]) + gen['p_max'] * (1 - u[gen_idx, t]), name=f"ramp_down_{gen_idx}_{t}")
                # Startup ramp
                model.addConstr(p[gen_idx, t] <= gen['startup_ramp_limit'] + gen['p_max'] * (1 - v[gen_idx, t]), name=f"startup_ramp_{gen_idx}_{t}")
                # Shutdown ramp
                model.addConstr(p[gen_idx, t-1] <= gen['shutdown_ramp_limit'] + gen['p_max'] * (1 - w[gen_idx, t]), name=f"shutdown_ramp_{gen_idx}_{t}")
            
            # Minimum up/down time constraints
            if t < T - min_up_time + 1:
                model.addConstr(gp.quicksum(u[gen_idx, i] for i in range(t, t+min_up_time)) >= min_up_time * v[gen_idx, t], name=f"min_up_{gen_idx}_{t}")
            if t < T - min_down_time + 1:
                model.addConstr(gp.quicksum(1 - u[gen_idx, i] for i in range(t, t+min_down_time)) >= min_down_time * w[gen_idx, t], name=f"min_down_{gen_idx}_{t}")
            
            # Piecewise linear production cost
            segments = sorted(gen['production_cost_segments'], key=lambda seg: seg['output_mw_start'])
            x_vals = []
            y_vals = []
            current_x = segments[0]['output_mw_start']
            current_y = 0.0
            x_vals.append(current_x)
            y_vals.append(current_y)
            for seg in segments:
                x_start = seg['output_mw_start']
                x_end = seg['output_mw_end']
                slope = seg['marginal_cost_per_mwh']
                length = x_end - x_start
                current_y += length * slope
                x_vals.append(x_end)
                y_vals.append(current_y)
            model.addGenConstrPWL(p[gen_idx, t], cost_prod[gen_idx, t], x_vals, y_vals, name=f"prod_cost_{gen_idx}_{t}")
            
            # Startup cost tiers
            startup_tiers = sorted(gen['startup_cost'], key=lambda tier: tier['max_downtime_hours'] if tier['max_downtime_hours'] is not None else float('inf'))
            num_tiers = len(startup_tiers)
            model.addConstr(gp.quicksum(z[gen_idx, t, k] for k in range(num_tiers)) == v[gen_idx, t], name=f"startup_tier_sum_{gen_idx}_{t}")
            model.addConstr(cost_start[gen_idx, t] == gp.quicksum(startup_tiers[k]['cost'] * z[gen_idx, t, k] for k in range(num_tiers)), name=f"startup_cost_{gen_idx}_{t}")
            
            for k in range(num_tiers - 1):
                limit_k = startup_tiers[k]['max_downtime_hours']
                model.addConstr(D_val[gen_idx, t] <= limit_k + L_max * time_period_hours * (1 - z[gen_idx, t, k]), name=f"tier_upper_{gen_idx}_{t}_{k}")
            for k in range(1, num_tiers):
                limit_prev = startup_tiers[k-1]['max_downtime_hours']
                model.addConstr(D_val[gen_idx, t] >= limit_prev + 1 - L_max * time_period_hours * (1 - z[gen_idx, t, k]), name=f"tier_lower_{gen_idx}_{t}_{k}")
    
    # Demand and reserve constraints
    for t in range(T):
        model.addConstr(gp.quicksum(p[gen_idx, t] for gen_idx in range(len(generators))) == demand[t], name=f"demand_{t}")
        model.addConstr(gp.quicksum(pb[gen_idx, t] for gen_idx in range(len(generators))) >= demand[t] + reserve_requirement[t], name=f"reserve_{t}")
    
    # Objective
    total_cost = gp.quicksum(
        gen['no_load_cost'] * u[gen_idx, t] + cost_prod[gen_idx, t] + cost_start[gen_idx, t]
        for gen_idx, gen in enumerate(generators) for t in range(T)
    )
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Optimize
    def log_solution(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol_count = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
            with open(solution_path, 'w') as f:
                json.dump({"Status": "Optimal", "Objective": obj_val}, f)
    
    model.optimize(log_solution if solution_path else None)
    
    # Write final solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            "status": "Optimal" if model.status == GRB.OPTIMAL else "TimeLimit",
            "objective": model.ObjVal,
            "commitment": {},
            "production": {},
            "reserve": {}
        }
        for gen_idx in range(len(generators)):
            for t in range(T):
                solution['commitment'].setdefault(gen_idx, []).append(int(u[gen_idx, t].X))
                solution['production'].setdefault(gen_idx, []).append(p[gen_idx, t].X)
                solution['reserve'].setdefault(gen_idx, []).append(pb[gen_idx, t].X)
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        with open(solution_path, 'w') as f:
            json.dump({"status": "Infeasible" if model.status == GRB.INFEASIBLE else "Error", "objective": None}, f)

if __name__ == "__main__":
    main()