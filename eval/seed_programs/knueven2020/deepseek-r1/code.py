import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import os
import sys
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    
    T = instance['num_time_periods']
    generators = instance['generators']
    demand = instance['demand']
    reserve_requirement = instance['reserve_requirement']
    time_period_hours = instance['time_period_hours']
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    u = {}  # On/off status
    p = {}  # Power output
    pb = {} # Reserve capacity
    v = {}  # Startup indicator
    w = {}  # Shutdown indicator
    cost_prod = {}  # Production cost
    cost_start = {}  # Startup cost
    
    for gen_idx, gen in enumerate(generators):
        for t in range(T):
            u[gen_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{gen_idx}_{t}")
            p[gen_idx, t] = model.addVar(lb=0, ub=gen['p_max'], name=f"p_{gen_idx}_{t}")
            pb[gen_idx, t] = model.addVar(lb=0, ub=gen['p_max'], name=f"pb_{gen_idx}_{t}")
            v[gen_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"v_{gen_idx}_{t}")
            w[gen_idx, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{gen_idx}_{t}")
            cost_prod[gen_idx, t] = model.addVar(name=f"cost_prod_{gen_idx}_{t}")
            cost_start[gen_idx, t] = model.addVar(name=f"cost_start_{gen_idx}_{t}")
    
    # Initial status constraints
    for gen_idx, gen in enumerate(generators):
        initial_status = gen['initial_status']
        min_up_time = gen['min_up_time']
        min_down_time = gen['min_down_time']
        initial_power = gen['initial_power']
        
        # Initial minimum up/down time enforcement
        if initial_status > 0:  # Generator was on
            remaining_up = max(0, min_up_time - initial_status)
            for t in range(remaining_up):
                model.addConstr(u[gen_idx, t] == 1, name=f"init_up_{gen_idx}_{t}")
        elif initial_status < 0:  # Generator was off
            remaining_down = max(0, min_down_time + initial_status)
            for t in range(remaining_down):
                model.addConstr(u[gen_idx, t] == 0, name=f"init_down_{gen_idx}_{t}")
        
        # Initial ramp constraints
        if initial_status > 0:  # Was online
            if initial_status < min_up_time:  # Must stay online
                model.addConstr(u[gen_idx, 0] == 1)
                model.addConstr(p[gen_idx, 0] - initial_power <= gen['ramp_up_limit'])
                model.addConstr(initial_power - p[gen_idx, 0] <= gen['ramp_down_limit'])
            else:  # Can choose to stay or shut down
                model.addConstr(p[gen_idx, 0] - initial_power <= gen['ramp_up_limit'] * u[gen_idx, 0])
                model.addConstr(initial_power - p[gen_idx, 0] <= gen['ramp_down_limit'] * u[gen_idx, 0])
                model.addConstr(initial_power <= gen['shutdown_ramp_limit'] + gen['p_max'] * (1 - w[gen_idx, 0]))
        else:  # Was offline
            model.addConstr(p[gen_idx, 0] <= gen['startup_ramp_limit'] * u[gen_idx, 0])
    
    # Status transition constraints
    for gen_idx, gen in enumerate(generators):
        min_up_time = gen['min_up_time']
        min_down_time = gen['min_down_time']
        
        for t in range(T):
            # Power and reserve bounds
            model.addConstr(p[gen_idx, t] >= gen['p_min'] * u[gen_idx, t])
            model.addConstr(p[gen_idx, t] <= gen['p_max'] * u[gen_idx, t])
            model.addConstr(pb[gen_idx, t] >= p[gen_idx, t])
            model.addConstr(pb[gen_idx, t] <= gen['p_max'] * u[gen_idx, t])
            
            # Startup/shutdown indicators
            if t == 0:
                initial_on = 1 if gen['initial_status'] > 0 else 0
                model.addConstr(v[gen_idx, t] >= u[gen_idx, t] - initial_on)
                model.addConstr(v[gen_idx, t] <= u[gen_idx, t])
                model.addConstr(v[gen_idx, t] <= 1 - initial_on)
                model.addConstr(w[gen_idx, t] >= initial_on - u[gen_idx, t])
                model.addConstr(w[gen_idx, t] <= initial_on)
                model.addConstr(w[gen_idx, t] <= 1 - u[gen_idx, t])
            else:
                model.addConstr(v[gen_idx, t] >= u[gen_idx, t] - u[gen_idx, t-1])
                model.addConstr(v[gen_idx, t] <= u[gen_idx, t])
                model.addConstr(v[gen_idx, t] <= 1 - u[gen_idx, t-1])
                model.addConstr(w[gen_idx, t] >= u[gen_idx, t-1] - u[gen_idx, t])
                model.addConstr(w[gen_idx, t] <= u[gen_idx, t-1])
                model.addConstr(w[gen_idx, t] <= 1 - u[gen_idx, t])
            
            # Ramping constraints
            if t >= 1:
                # Online ramp limits
                model.addConstr(p[gen_idx, t] - p[gen_idx, t-1] <= gen['ramp_up_limit'])
                model.addConstr(p[gen_idx, t-1] - p[gen_idx, t] <= gen['ramp_down_limit'])
                # Startup ramp limit
                model.addConstr(p[gen_idx, t] <= gen['startup_ramp_limit'] + gen['p_max'] * (1 - v[gen_idx, t]))
                # Shutdown ramp limit
                model.addConstr(p[gen_idx, t-1] <= gen['shutdown_ramp_limit'] + gen['p_max'] * (1 - w[gen_idx, t]))
            
            # Minimum up/down time constraints
            if t >= min_up_time - 1 and t < T:
                model.addConstr(gp.quicksum(u[gen_idx, i] for i in range(t-min_up_time+1, t+1)) >= 
                              min_up_time * v[gen_idx, t-min_up_time+1])
            if t >= min_down_time - 1 and t < T:
                model.addConstr(gp.quicksum(1 - u[gen_idx, i] for i in range(t-min_down_time+1, t+1)) >= 
                              min_down_time * w[gen_idx, t-min_down_time+1])
            
            # Production cost (piecewise linear)
            segments = gen['production_cost_segments']
            x_vals = [segments[0]['output_mw_start']]
            y_vals = [0]
            for seg in segments:
                x_vals.append(seg['output_mw_end'])
                y_vals.append(y_vals[-1] + 
                             (seg['output_mw_end'] - seg['output_mw_start']) * 
                              seg['marginal_cost_per_mwh'])
            model.addGenConstrPWL(p[gen_idx, t], cost_prod[gen_idx, t], x_vals, y_vals)
            
            # Startup cost (downtime-dependent)
            startup_costs = gen['startup_costs']
            z = {}  # Tier selection variables
            for tier_idx, tier in enumerate(startup_costs):
                z[tier_idx] = model.addVar(vtype=GRB.BINARY, name=f"z_{gen_idx}_{t}_{tier_idx}")
            
            model.addConstr(gp.quicksum(z[tier_idx] for tier_idx in range(len(startup_costs))) == v[gen_idx, t])
            model.addConstr(cost_start[gen_idx, t] == gp.quicksum(tier['cost'] * z[tier_idx] 
                                 for tier_idx, tier in enumerate(startup_costs)))
            
            # Tier selection constraints
            for tier_idx, tier in enumerate(startup_costs[:-1]):
                max_downtime = tier['max_downtime_hours']
                model.addConstr(z[tier_idx] <= max_downtime)  # Simplified for clarity
    
    # System constraints
    for t in range(T):
        model.addConstr(gp.quicksum(p[gen_idx, t] for gen_idx in range(len(generators))) == demand[t])
        model.addConstr(gp.quicksum(pb[gen_idx, t] for gen_idx in range(len(generators))) >= 
                      demand[t] + reserve_requirement[t])
    
    # Objective function
    total_cost = gp.quicksum(
        gen['no_load_cost'] * u[gen_idx, t] + 
        cost_prod[gen_idx, t] + 
        cost_start[gen_idx, t]
        for gen_idx, gen in enumerate(generators)
        for t in range(T)
    )
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    # Callback for logging intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            solution_dict = {
                "objective_value": obj_val,
                "schedule": {}
            }
            for gen_idx, gen in enumerate(generators):
                gen_id = gen['id']
                u_vals = [int(round(model.cbGetSolution(u[gen_idx, t]))) for t in range(T)]
                p_vals = [model.cbGetSolution(p[gen_idx, t]) for t in range(T)]
                pb_vals = [model.cbGetSolution(pb[gen_idx, t]) for t in range(T)]
                solution_dict["schedule"][gen_id] = {
                    "u": u_vals,
                    "p": p_vals,
                    "pb": pb_vals
                }
            if logger:
                logger.log_solution(obj_val, solution_dict)
    
    # Optimize with callback if logging enabled
    if logger:
        model.optimize(callback)
    else:
        model.optimize()
    
    # Save final solution
    solution = {"objective_value": None, "schedule": {}}
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        for gen_idx, gen in enumerate(generators):
            gen_id = gen['id']
            u_vals = [int(round(u[gen_idx, t].X)) for t in range(T)]
            p_vals = [p[gen_idx, t].X for t in range(T)]
            pb_vals = [pb[gen_idx, t].X for t in range(T)]
            solution["schedule"][gen_id] = {
                "u": u_vals,
                "p": p_vals,
                "pb": pb_vals
            }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()