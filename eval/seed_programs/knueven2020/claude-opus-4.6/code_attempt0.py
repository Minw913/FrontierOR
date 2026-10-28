import argparse
import json
import time
import math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_uc(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_generators = data['num_generators']
    num_periods = data['num_time_periods']
    time_period_hours = data['time_period_hours']
    generators = data['generators']
    demand = data['demand']
    reserve_req = data['reserve_requirement']
    
    T = range(num_periods)
    G = range(num_generators)
    
    # Create model
    model = gp.Model("unit_commitment")
    model.setParam('TimeLimit', max(1, time_limit - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    
    # Decision variables
    u = {}  # on/off status
    v = {}  # startup indicator
    w = {}  # shutdown indicator
    p = {}  # power output above p_min
    pb = {}  # power available for reserve (above p_min)
    
    for g in G:
        gen = generators[g]
        for t in T:
            u[g, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{g}_{t}")
            v[g, t] = model.addVar(vtype=GRB.BINARY, name=f"v_{g}_{t}")
            w[g, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{g}_{t}")
            # p[g,t] represents power output above p_min
            p[g, t] = model.addVar(lb=0, ub=gen['p_max'] - gen['p_min'], name=f"p_{g}_{t}")
            pb[g, t] = model.addVar(lb=0, ub=gen['p_max'] - gen['p_min'], name=f"pb_{g}_{t}")
    
    # Startup cost variables - using downtime-dependent startup costs
    # For each generator, we have multiple startup cost tiers
    # delta[g, t, s] = 1 if generator g starts up at time t using tier s
    delta = {}
    for g in G:
        gen = generators[g]
        sc = gen['startup_costs']
        num_tiers = len(sc)
        for t in T:
            for s in range(num_tiers):
                delta[g, t, s] = model.addVar(vtype=GRB.BINARY, name=f"delta_{g}_{t}_{s}")
    
    # Piecewise linear production cost variables
    # For each generator, cost segments above p_min
    pc = {}  # power in each cost segment
    for g in G:
        gen = generators[g]
        segments = gen['production_cost_segments']
        for t in T:
            for s_idx in range(len(segments)):
                seg = segments[s_idx]
                seg_width = seg['output_mw_end'] - seg['output_mw_start']
                pc[g, t, s_idx] = model.addVar(lb=0, ub=seg_width, name=f"pc_{g}_{t}_{s_idx}")
    
    model.update()
    
    # Objective function
    obj = gp.LinExpr()
    
    for g in G:
        gen = generators[g]
        segments = gen['production_cost_segments']
        sc = gen['startup_costs']
        
        for t in T:
            # No-load cost
            obj += gen['no_load_cost'] * u[g, t]
            
            # Production cost (piecewise linear)
            for s_idx in range(len(segments)):
                seg = segments[s_idx]
                obj += seg['marginal_cost_per_mwh'] * time_period_hours * pc[g, t, s_idx]
            
            # Startup costs
            for s in range(len(sc)):
                obj += sc[s]['cost'] * delta[g, t, s]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Logical constraints: u, v, w relationship
    for g in G:
        gen = generators[g]
        init_status = gen['initial_status']
        u_prev = 1 if init_status > 0 else 0
        
        for t in T:
            if t == 0:
                model.addConstr(v[g, t] - w[g, t] == u[g, t] - u_prev, name=f"logic_{g}_{t}")
            else:
                model.addConstr(v[g, t] - w[g, t] == u[g, t] - u[g, t-1], name=f"logic_{g}_{t}")
            # Can't start up and shut down at the same time
            model.addConstr(v[g, t] + w[g, t] <= 1, name=f"no_simul_{g}_{t}")
    
    # 2. Power output linked to on/off status
    for g in G:
        gen = generators[g]
        for t in T:
            # p[g,t] <= (p_max - p_min) * u[g,t]
            model.addConstr(p[g, t] <= (gen['p_max'] - gen['p_min']) * u[g, t], name=f"pmax_{g}_{t}")
            # pb[g,t] >= p[g,t]
            model.addConstr(pb[g, t] >= p[g, t], name=f"pb_ge_p_{g}_{t}")
            # pb[g,t] <= (p_max - p_min) * u[g,t]
            model.addConstr(pb[g, t] <= (gen['p_max'] - gen['p_min']) * u[g, t], name=f"pbmax_{g}_{t}")
    
    # 3. Piecewise linear production cost: sum of segment outputs = p[g,t]
    for g in G:
        gen = generators[g]
        segments = gen['production_cost_segments']
        for t in T:
            model.addConstr(
                gp.quicksum(pc[g, t, s_idx] for s_idx in range(len(segments))) == p[g, t],
                name=f"seg_sum_{g}_{t}"
            )
    
    # 4. Demand balance
    for t in T:
        model.addConstr(
            gp.quicksum(generators[g]['p_min'] * u[g, t] + p[g, t] for g in G) == demand[t],
            name=f"demand_{t}"
        )
    
    # 5. Reserve requirement
    for t in T:
        model.addConstr(
            gp.quicksum(generators[g]['p_min'] * u[g, t] + pb[g, t] for g in G) >= demand[t] + reserve_req[t],
            name=f"reserve_{t}"
        )
    
    # 6. Ramp-up constraints
    for g in G:
        gen = generators[g]
        init_power = gen['initial_power']
        init_status = gen['initial_status']
        u_prev_val = 1 if init_status > 0 else 0
        p_min = gen['p_min']
        p_max = gen['p_max']
        ramp_up = gen['ramp_up_limit']
        ramp_down = gen['ramp_down_limit']
        startup_ramp = gen['startup_ramp_limit']
        shutdown_ramp = gen['shutdown_ramp_limit']
        
        for t in T:
            # Actual power = p_min * u[g,t] + p[g,t]
            # Previous actual power = p_min * u[g,t-1] + p[g,t-1]
            
            if t == 0:
                prev_power_above_min = init_power - p_min * u_prev_val if u_prev_val == 1 else 0
                # Ramp up: p_min*u[g,0] + p[g,0] - init_power <= ramp_up * u_prev + startup_ramp * v[g,0]
                # But we need to handle this more carefully
                # When staying on: (p_min + p[g,0]) - init_power <= ramp_up
                # When starting up: (p_min + p[g,0]) <= startup_ramp
                
                if u_prev_val == 1:
                    # p[g,0] + p_min * u[g,0] - init_power <= ramp_up * u[g,0] (if was on and stays on)
                    # But also need startup ramp if was off and turns on - not possible here since u_prev=1
                    # Actually, could shut down (w[g,0]=1) - but then p[g,0]=0, u[g,0]=0
                    # Ramp up: p_min*u[g,0] + p[g,0] <= init_power + ramp_up
                    # But we also need: if shutting down (u[g,0]=0), the previous period output <= shutdown_ramp
                    # That's a constraint on previous period which is before the horizon
                    model.addConstr(
                        p[g, t] <= prev_power_above_min + ramp_up - (ramp_up - (startup_ramp - p_min)) * v[g, t],
                        name=f"rampup_{g}_{t}"
                    ) if startup_ramp - p_min < ramp_up else model.addConstr(
                        p[g, t] <= prev_power_above_min + ramp_up,
                        name=f"rampup_{g}_{t}"
                    )
                else:
                    # Was off, can only start up
                    model.addConstr(
                        p[g, t] <= (startup_ramp - p_min) * v[g, t],
                        name=f"rampup_{g}_{t}"
                    )
                    # Also general ramp up using the formulation
            else:
                # General ramp up constraint:
                # p_min*u[g,t] + p[g,t] - p_min*u[g,t-1] - p[g,t-1] <= ramp_up * u[g,t-1] + startup_ramp * v[g,t]
                # => p[g,t] - p[g,t-1] <= ramp_up * u[g,t-1] + startup_ramp * v[g,t] - p_min*(u[g,t] - u[g,t-1])
                # => p[g,t] - p[g,t-1] <= ramp_up * u[g,t-1] + startup_ramp * v[g,t] - p_min*(v[g,t] - w[g,t])
                # => p[g,t] - p[g,t-1] <= ramp_up * u[g,t-1] + (startup_ramp - p_min) * v[g,t] + p_min * w[g,t]
                # Since w[g,t] means shutting down (u[g,t]=0), p[g,t]=0, so the constraint is automatically relaxed
                model.addConstr(
                    p[g, t] - p[g, t-1] <= ramp_up * u[g, t-1] + (startup_ramp - p_min) * v[g, t] + p_min * w[g, t],
                    name=f"rampup_{g}_{t}"
                )
        
        # 7. Ramp-down constraints
        for t in T:
            if t == 0:
                if u_prev_val == 1:
                    # Ramp down: init_power - (p_min*u[g,0] + p[g,0]) <= ramp_down * u[g,0] + shutdown_ramp * w[g,0]
                    # prev_power_above_min - p[g,0] <= ramp_down * u[g,0] + (shutdown_ramp - p_min) * w[g,0] + p_min * v[g,0]
                    # Since v[g,0] can't be 1 (was already on), it simplifies
                    model.addConstr(
                        prev_power_above_min - p[g, t] <= ramp_down * u[g, t] + (shutdown_ramp - p_min) * w[g, t],
                        name=f"rampdown_{g}_{t}"
                    )
                # If was off, no ramp down constraint needed for t=0
            else:
                # p[g,t-1] - p[g,t] <= ramp_down * u[g,t] + (shutdown_ramp - p_min) * w[g,t] + p_min * v[g,t]
                model.addConstr(
                    p[g, t-1] - p[g, t] <= ramp_down * u[g, t] + (shutdown_ramp - p_min) * w[g, t] + p_min * v[g, t],
                    name=f"rampdown_{g}_{t}"
                )
        
        # Also: reserve available must respect ramp limits
        for t in T:
            if t == 0:
                if u_prev_val == 1:
                    model.addConstr(
                        pb[g, t] <= prev_power_above_min + ramp_up - (ramp_up - (startup_ramp - p_min)) * v[g, t] if startup_ramp - p_min < ramp_up else
                        pb[g, t] <= prev_power_above_min + ramp_up,
                        name=f"pb_rampup_{g}_{t}"
                    )
                else:
                    model.addConstr(
                        pb[g, t] <= (startup_ramp - p_min) * v[g, t],
                        name=f"pb_rampup_{g}_{t}"
                    )
            else:
                model.addConstr(
                    pb[g, t] - p[g, t-1] <= ramp_up * u[g, t-1] + (startup_ramp - p_min) * v[g, t] + p_min * w[g, t],
                    name=f"pb_rampup_{g}_{t}"
                )
        
        # Shutdown ramp limit on pb: if generator shuts down next period
        for t in T:
            if t < num_periods - 1:
                model.addConstr(
                    pb[g, t] <= (p_max - p_min) * u[g, t] - (p_max - shutdown_ramp) * w[g, t+1],
                    name=f"pb_shutdown_{g}_{t}"
                )
                # Also actual power shutdown constraint
                model.addConstr(
                    p[g, t] <= (p_max - p_min) * u[g, t] - (p_max - shutdown_ramp) * w[g, t+1],
                    name=f"p_shutdown_{g}_{t}"
                )
    
    # 8. Minimum up-time constraints
    for g in G:
        gen = generators[g]
        min_up = gen['min_up_time']
        init_status = gen['initial_status']
        
        # Initial conditions: if generator was on for fewer periods than min_up
        if init_status > 0:
            remaining_up = max(0, min_up - init_status)
            for t in range(min(remaining_up, num_periods)):
                model.addConstr(u[g, t] == 1, name=f"init_minup_{g}_{t}")
        
        # For each startup at time t, must stay on for min_up periods
        for t in T:
            end = min(t + min_up, num_periods)
            if end > t:
                model.addConstr(
                    gp.quicksum(u[g, t2] for t2 in range(t, end)) >= min_up * v[g, t],
                    name=f"minup_{g}_{t}"
                )
    
    # 9. Minimum down-time constraints
    for g in G:
        gen = generators[g]
        min_down = gen['min_down_time']
        init_status = gen['initial_status']
        
        # Initial conditions: if generator was off for fewer periods than min_down
        if init_status < 0:
            remaining_down = max(0, min_down - (-init_status))
            for t in range(min(remaining_down, num_periods)):
                model.addConstr(u[g, t] == 0, name=f"init_mindown_{g}_{t}")
        
        # For each shutdown at time t, must stay off for min_down periods
        for t in T:
            end = min(t + min_down, num_periods)
            if end > t:
                model.addConstr(
                    gp.quicksum(1 - u[g, t2] for t2 in range(t, end)) >= min_down * w[g, t],
                    name=f"mindown_{g}_{t}"
                )
    
    # 10. Startup cost tier constraints
    for g in G:
        gen = generators[g]
        sc = gen['startup_costs']
        num_tiers = len(sc)
        init_status = gen['initial_status']
        
        for t in T:
            # Sum of delta[g,t,s] over all tiers = v[g,t]
            model.addConstr(
                gp.quicksum(delta[g, t, s] for s in range(num_tiers)) == v[g, t],
                name=f"startup_tier_sum_{g}_{t}"
            )
        
        # For each tier, determine the downtime windows
        # Tiers are ordered by increasing max_downtime_hours and increasing cost
        # Tier s applies if downtime is in [prev_max_downtime+1, max_downtime_hours_s] (in hours)
        # Convert to periods
        for t in T:
            for s in range(num_tiers):
                tier = sc[s]
                if s == 0:
                    # First tier: downtime from min_down_time to max_downtime_hours/time_period_hours
                    lower_dt_periods = gen['min_down_time']
                else:
                    prev_tier = sc[s-1]
                    lower_dt_periods = prev_tier['max_downtime_hours'] // time_period_hours + 1
                
                if tier['max_downtime_hours'] is not None:
                    upper_dt_periods = tier['max_downtime_hours'] // time_period_hours
                else:
                    upper_dt_periods = num_periods + 500  # effectively infinity
                
                # delta[g,t,s] can be 1 only if generator was off for between lower_dt_periods and upper_dt_periods
                # This means: sum of (1-u[g,t']) for t' in [t-upper_dt_periods, t-1] >= lower_dt_periods * delta[g,t,s]
                # And: there exists some on period within [t-upper_dt_periods-1, t-lower_dt_periods]
                # 
                # A simpler formulation: delta[g,t,s] <= sum of w[g,t'] for t' in relevant window
                # where the window is [t-upper_dt_periods, t-lower_dt_periods]
                
                # The window of shutdown times that would result in this tier
                # If generator shuts down at time t', it starts at time t
                # Downtime = t - t' periods
                # For tier s: lower_dt_periods <= t - t' <= upper_dt_periods
                # So t' in [t - upper_dt_periods, t - lower_dt_periods]
                
                w_sum = gp.LinExpr()
                for t_prime in range(max(0, t - upper_dt_periods), t - lower_dt_periods + 1):
                    w_sum += w[g, t_prime]
                
                # Also consider initial condition
                # If generator was initially off for |init_status| periods
                if init_status < 0:
                    off_periods_before = -init_status
                    # The generator was off for off_periods_before periods before period 0
                    # If it starts at time t, total downtime = off_periods_before + t periods
                    # But we also need to check if it was continuously off
                    # Actually, the downtime at startup t is: number of consecutive off periods before t
                    # If init was off, and stays off until t, downtime = off_periods_before + t
                    # For this to match tier s: lower_dt_periods <= off_periods_before + t <= upper_dt_periods
                    total_down = off_periods_before + t
                    if lower_dt_periods <= total_down <= upper_dt_periods:
                        # This is captured if w happened before the horizon
                        # We need to account for this - the initial shutdown happened before the horizon
                        # delta[g,t,s] can be 1 if generator was continuously off from start
                        # This means u[g, t'] = 0 for all t' < t
                        # We add a special term: if no w[g,t'] in the horizon triggers this,
                        # the initial off state should
                        pass  # Will handle below
                
                if t - lower_dt_periods + 1 > 0 or (init_status < 0 and lower_dt_periods <= (-init_status) + t <= upper_dt_periods):
                    # Check if initial state could contribute
                    has_initial = False
                    if init_status < 0:
                        total_down = (-init_status) + t
                        if lower_dt_periods <= total_down <= upper_dt_periods:
                            has_initial = True
                    
                    if has_initial:
                        # Add a binary that represents "was off since before horizon"
                        # This is 1 if u[g, t']=0 for all t' in [0, t-1]
                        # We can use: 1 - u[g, t'] for all t' < t, but that's not directly a variable
                        # Instead: delta[g,t,s] <= w_sum + (1 - sum(u[g,t'] for t' in [0,min(t,?)]))
                        # Actually, simpler: if no w happened in horizon before t, and initial was off,
                        # then we're still in initial off period
                        # The initial "shutdown" happened at time -(init_off_periods)
                        # We can add a "virtual" w contribution
                        w_sum += 1 - gp.quicksum(u[g, t2] for t2 in range(t)) if t > 0 else 1
                        # This is an overcount but let me think more carefully
                        # Actually if t > 0, we want: was the generator off for all of [0, t-1]?
                        # If so, and initial was off, the downtime = init_off + t, which matches this tier
                        # (1 - u[g,0]) * (1 - u[g,1]) * ... is nonlinear
                        # Better approach: just use the standard formulation
                
                # Standard Morales-España formulation for startup costs
                # delta[g,t,s] <= sum of w[g,t'] for t' in shutdown window
                # For the initial condition, we treat it as if w happened before the horizon
                
                # Let me redo this more carefully
                pass
        
    # Remove the tier constraints above and redo them properly
    # Actually, let me use a cleaner formulation
    
    # Remove all delta constraints and redo
    # First, remove the sum constraints we already added (they're fine, keep those)
    
    # For startup costs, use the Carrion-Arroyo formulation:
    # startup_cost[g,t] >= cost_s * (v[g,t] - sum(u[g,t'] for t' in appropriate window))
    
    # Let me use auxiliary startup cost variables instead
    sc_var = {}
    for g in G:
        gen = generators[g]
        for t in T:
            sc_var[g, t] = model.addVar(lb=0, name=f"sc_{g}_{t}")
    
    model.update()
    
    # Remove delta variables from objective and add sc_var instead
    # We need to rebuild the objective
    obj2 = gp.LinExpr()
    for g in G:
        gen = generators[g]
        segments = gen['production_cost_segments']
        for t in T:
            obj2 += gen['no_load_cost'] * u[g, t]
            for s_idx in range(len(segments)):
                seg = segments[s_idx]
                obj2 += seg['marginal_cost_per_mwh'] * time_period_hours * pc[g, t, s_idx]
            obj2 += sc_var[g, t]
    
    model.setObjective(obj2, GRB.MINIMIZE)
    
    # Startup cost constraints using Carrion-Arroyo / Morales-España approach
    for g in G:
        gen = generators[g]
        sc = gen['startup_costs']
        num_tiers = len(sc)
        init_status = gen['initial_status']
        min_down = gen['min_down_time']
        
        # Sort tiers by max_downtime (ascending), with None (cold) last
        # They should already be sorted in the data
        
        # Convert tier boundaries to periods
        tier_boundaries = []
        for s in range(num_tiers):
            if sc[s]['max_downtime_hours'] is not None:
                tier_boundaries.append(sc[s]['max_downtime_hours'] // time_period_hours)
            else:
                tier_boundaries.append(None)
        
        # The coldest (most expensive) startup cost
        cold_cost = sc[-1]['cost']
        
        for t in T:
            # Basic constraint: sc_var[g,t] >= cold_cost * v[g,t]
            model.addConstr(sc_var[g, t] >= cold_cost * v[g, t], name=f"sc_cold_{g}_{t}")
            
            # For cheaper tiers (hot, warm), add constraints that reduce cost if recently shut down
            for s in range(num_tiers - 1):
                tier_cost = sc[s]['cost']
                if s == 0:
                    # Downtime window: [min_down, tier_boundaries[s]] periods
                    lower = min_down
                else:
                    lower = tier_boundaries[s-1] + 1
                upper = tier_boundaries[s]
                
                # If generator was off for between lower and upper periods before starting at t,
                # the startup cost should be tier_cost instead of cold_cost
                # 
                # sc_var[g,t] >= tier_cost * v[g,t] - (cold_cost - tier_cost) * sum(u[g,t'] for t' in [t-upper, t-lower])
                # 
                # Actually, the standard formulation is:
                # sc_var[g,t] >= cost_s * (v[g,t] - sum_{t'=t-upper}^{t-lower-1??} u[g,t'])
                # No wait. Let me use the formulation from Morales-España et al.
                
                # Formulation: for each tier s with cost C_s and downtime range [D_s_lower, D_s_upper]:
                # sc_var[g,t] >= C_s * (v[g,t] - sum_{j=t-D_s_upper}^{t-D_s_lower} u[g,j])
                # where terms outside [0, T-1] use initial conditions
                
                # Actually the correct Carrion-Arroyo formulation is:
                # For each tier from cheapest to most expensive:
                # Let tier 0 be cheapest (hot) with max downtime D_0
                # sc_var[g,t] >= C_s * (v[g,t] - sum of u[g,j] for j in window)
                
                # Let me use a different, cleaner approach:
                # The startup cost depends on how long the generator has been off
                # We use: sc_var[g,t] >= sum over tiers s: (C_s - C_{s-1}) * (v[g,t] - sum u[g,j] for j in [t-D_s, t-1])
                # where C_{-1} = 0 and D_s is the upper bound of tier s in periods
                pass
        
        # Let me use the incremental startup cost formulation
        # Sort tiers: tier 0 (hot, cheapest), tier 1 (warm), ..., tier K (cold, most expensive)
        # Incremental costs: delta_cost[0] = sc[0]['cost']
        # delta_cost[s] = sc[s]['cost'] - sc[s-1]['cost'] for s >= 1
        
        for t in T:
            # sc_var[g,t] >= sum_{s=0}^{K} delta_cost[s] * (v[g,t] - sum_{j in window_s} u[g,j])
            # where window_s = [t - D_s, t - 1] (using initial u values for j < 0)
            # D_0 = tier_boundaries[0], D_1 = tier_boundaries[1], etc.
            
            # Actually this needs to be separate constraints for each tier
            for s in range(num_tiers):
                tier_cost = sc[s]['cost']
                if s == 0:
                    upper_dt = min_down  # minimum possible downtime
                else:
                    upper_dt = tier_boundaries[s-1]  # max downtime of previous tier
                
                # Current tier applies for downtime > upper_dt (of prev tier) and <= tier_boundaries[s]
                # The constraint is:
                # sc_var[g,t] >= tier_cost * (v[g,t] - sum u[g,j] for j in [t-upper_bound, t-min_down])
                
                # Let me just use the standard formulation from the literature:
                # For the s-th cheapest tier with cost C_s and applicable when downtime in [Ts_lower, Ts_upper]:
                
                if tier_boundaries[s] is not None:
                    dt_upper = tier_boundaries[s]
                else:
                    dt_upper = num_periods + 500
                
                # sc_var[g,t] >= C_s * (v[g,t] - sum_{j=t-dt_upper}^{t-min_down} u[g,j])
                # But this isn't quite right either.
                
                # Let me go back to basics. The Carrion & Arroyo (2006) formulation:
                # c_su[g,t] >= K_s * (y[g,t-Ts_upper+1] + y[g,t-Ts_upper+2] + ... + y[g,t-Ts_lower+1] - ... )
                # Actually it's simpler: for each startup type s with downtime range [Ts, Ts+1-1]:
                pass
        
    # OK, let me just use a simpler but correct formulation.
    # I'll use the formulation from Knueven et al. (2020) survey.
    
    # Remove previous sc constraints and redo
    # Actually, since I haven't finished adding them, let me just do it right.
    
    # For the Carrion-Arroyo startup cost formulation:
    # For each generator g, time t, and startup type s:
    # sc_var[g,t] >= C_s * (v[g,t] - sum_{i=max(0,t-D_s_upper)}^{t-D_s_lower} u[g,i])
    # where D_s_lower and D_s_upper are the downtime bounds for tier s
    # For initial periods, we use the initial status
    
    # But actually, we already added cold cost constraint. Let me remove all and redo.
    # Since we can't easily remove constraints in gurobi, let me just make sure we add the right ones.
    
    # Actually, I realize I should scrap this approach and rebuild more carefully.
    # Let me restart the model construction.
    
    # Since the above code is getting messy, let me just rebuild from scratch.
    
    model2 = gp.Model("uc_clean")
    model2.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 5))
    model2.setParam('Threads', 1)
    model2.setParam('MIPGap', 1e-4)
    model2.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Variables
    u = {}
    v = {}
    w = {}
    p_above = {}  # power above p_min
    pb_above = {}  # reserve available above p_min
    sc_cost = {}  # startup cost
    
    for g in G:
        gen = generators[g]
        for t in T:
            u[g, t] = model2.addVar(vtype=GRB.BINARY, name=f"u_{g}_{t}")
            v[g, t] = model2.addVar(vtype=GRB.BINARY, name=f"v_{g}_{t}")
            w[g, t] = model2.addVar(vtype=GRB.BINARY, name=f"w_{g}_{t}")
            p_above[g, t] = model2.addVar(lb=0, ub=gen['p_max'] - gen['p_min'], name=f"pa_{g}_{t}")
            pb_above[g, t] = model2.addVar(lb=0, ub=gen['p_max'] - gen['p_min'], name=f"pba_{g}_{t}")
            sc_cost[g, t] = model2.addVar(lb=0, name=f"sc_{g}_{t}")
    
    # Piecewise linear segments
    pc = {}
    for g in G:
        gen = generators[g]
        segments = gen['production_cost_segments']
        for t in T:
            for s_idx in range(len(segments)):
                seg = segments[s_idx]
                seg_width = seg['output_mw_end'] - seg['output_mw_start']
                pc[g, t, s_idx] = model2.addVar(lb=0, ub=seg_width, name=f"pc_{g}_{t}_{s_idx}")
    
    model2.update()
    
    # Objective
    obj = gp.LinExpr()
    for g in G:
        gen = generators[g]
        segments = gen['production_cost_segments']
        for t in T:
            obj += gen['no_load_cost'] * u[g, t]
            for s_idx in range(len(segments)):
                obj += segments[s_idx]['marginal_cost_per_mwh'] * time_period_hours * pc[g, t, s_idx]
            obj += sc_cost[g, t]
    model2.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Logical linking
    for g in G:
        gen = generators[g]
        init_on = 1 if gen['initial_status'] > 0 else 0
        for t in T:
            u_prev = u[g, t-1] if t > 0 else init_on
            model2.addConstr(v[g, t] - w[g, t] == u[g, t] - u_prev)
            model2.addConstr(v[g, t] + w[g, t] <= 1)
    
    # 2. Power bounds
    for g in G:
        gen = generators[g]
        for t in T:
            model2.addConstr(p_above[g, t] <= (gen['p_max'] - gen['p_min']) * u[g, t])
            model2.addConstr(pb_above[g, t] >= p_above[g, t])
            model2.addConstr(pb_above[g, t] <= (gen['p_max'] - gen['p_min']) * u[g, t])
    
    # 3. Segment sum = p_above
    for g in G:
        gen = generators[g]
        segments = gen['production_cost_segments']
        for t in T:
            model2.addConstr(
                gp.quicksum(pc[g, t, s] for s in range(len(segments))) == p_above[g, t]
            )
    
    # 4. Demand balance
    for t in T:
        model2.addConstr(
            gp.quicksum(generators[g]['p_min'] * u[g, t] + p_above[g, t] for g in G) == demand[t]
        )
    
    # 5. Reserve
    for t in T:
        model2.addConstr(
            gp.quicksum(generators[g]['p_min'] * u[g, t] + pb_above[g, t] for g in G) >= demand[t] + reserve_req[t]
        )
    
    # 6. Ramping constraints
    for g in G:
        gen = generators[g]
        p_min_g = gen['p_min']
        p_max_g = gen['p_max']
        ramp_up = gen['ramp_up_limit']
        ramp_down = gen['ramp_down_limit']
        su_ramp = gen['startup_ramp_limit']
        sd_ramp = gen['shutdown_ramp_limit']
        init_power = gen['initial_power']
        init_on = 1 if gen['initial_status'] > 0 else 0
        init_p_above = max(0, init_power - p_min_g) if init_on else 0
        
        for t in T:
            u_prev = u[g, t-1] if t > 0 else init_on
            p_prev = p_above[g, t-1] if t > 0 else init_p_above
            
            # Ramp up for power: p_above[g,t] - p_prev <= ramp_up * u_prev + (su_ramp - p_min_g) * v[g,t]
            # When starting up (v=1, u_prev=0): p_above <= su_ramp - p_min_g
            # When staying on (v=0, u_prev=1): p_above - p_prev <= ramp_up
            if isinstance(p_prev, (int, float)):
                model2.addConstr(
                    p_above[g, t] <= p_prev + ramp_up * u_prev + max(0, su_ramp - p_min_g) * v[g, t]
                )
            else:
                model2.addConstr(
                    p_above[g, t] - p_prev <= ramp_up * u_prev + max(0, su_ramp - p_min_g) * v[g, t]
                )
            
            # Ramp up for reserve (pb)
            if isinstance(p_prev, (int, float)):
                model2.addConstr(
                    pb_above[g, t] <= p_prev + ramp_up * u_prev + max(0, su_ramp - p_min_g) * v[g, t]
                )
            else:
                model2.addConstr(
                    pb_above[g, t] - p_prev <= ramp_up * u_prev + max(0, su_ramp - p_min_g) * v[g, t]
                )
            
            # Ramp down: p_prev - p_above[g,t] <= ramp_down * u[g,t] + (sd_ramp - p_min_g) * w[g,t]
            if isinstance(p_prev, (int, float)):
                model2.addConstr(
                    p_prev - p_above[g, t] <= ramp_down * u[g, t] + max(0, sd_ramp - p_min_g) * w[g, t]
                )
            else:
                model2.addConstr(
                    p_prev - p_above[g, t] <= ramp_down * u[g, t] + max(0, sd_ramp - p_min_g) * w[g, t]
                )
            
            # Shutdown ramp limit: if shutting down next period
            if t < num_periods - 1:
                model2.addConstr(
                    p_above[g, t] <= (p_max_g - p_min_g) * u[g, t] - max(0, p_max_g - sd_ramp) * w[g, t+1]
                )
                model2.addConstr(
                    pb_above[g, t] <= (p_max_g - p_min_g) * u[g, t] - max(0, p_max_g - sd_ramp) * w[g, t+1]
                )
    
    # 7. Minimum up-time
    for g in G:
        gen = generators[g]
        min_up = gen['min_up_time']
        init_status = gen['initial_status']
        
        if init_status > 0:
            remaining = max(0, min_up - init_status)
            for t in range(min(remaining, num_periods)):
                model2.addConstr(u[g, t] == 1)
        
        for t in T:
            window_end = min(t + min_up, num_periods)
            if window_end - t < min_up:
                # Near end of horizon - generator must stay on if started
                model2.addConstr(
                    gp.quicksum(u[g, t2] for t2 in range(t, window_end)) >= (window_end - t) * v[g, t]
                )
            else:
                model2.addConstr(
                    gp.quicksum(u[g, t2] for t2 in range(t, t + min_up)) >= min_up * v[g, t]
                )
    
    # 8. Minimum down-time
    for g in G:
        gen = generators[g]
        min_down = gen['min_down_time']
        init_status = gen['initial_status']
        
        if init_status < 0:
            remaining = max(0, min_down - (-init_status))
            for t in range(min(remaining, num_periods)):
                model2.addConstr(u[g, t] == 0)
        
        for t in T:
            window_end = min(t + min_down, num_periods)
            if window_end - t < min_down:
                model2.addConstr(
                    gp.quicksum(1 - u[g, t2] for t2 in range(t, window_end)) >= (window_end - t) * w[g, t]
                )
            else:
                model2.addConstr(
                    gp.quicksum(1 - u[g, t2] for t2 in range(t, t + min_down)) >= min_down * w[g, t]
                )
    
    # 9. Startup cost constraints
    for g in G:
        gen = generators[g]
        sc = gen['startup_costs']
        num_tiers = len(sc)
        init_status = gen['initial_status']
        min_down = gen['min_down_time']
        
        # Determine tier downtime boundaries in periods
        # Tiers are ordered from cheapest (hot) to most expensive (cold)
        tier_periods = []
        for s in range(num_tiers):
            if sc[s]['max_downtime_hours'] is not None:
                tier_periods.append(sc[s]['max_downtime_hours'] // time_period_hours)
            else:
                tier_periods.append(None)
        
        # For each time period and each tier, create a constraint
        # The idea: if the generator starts up at t and was off for D periods,
        # the startup cost should be C_s where tier s covers that D.
        
        # Using the approach from Knueven et al.:
        # For tier s (0-indexed, 0 = cheapest):
        #   - Tier 0: downtime in [min_down, tier_periods[0]]
        #   - Tier s>0: downtime in [tier_periods[s-1]+1, tier_periods[s]]
        
        # sc_cost[g,t] >= C_K * v[g,t]  (coldest tier, always valid)
        cold_cost = sc[-1]['cost']
        
        for t in T:
            model2.addConstr(sc_cost[g, t] >= cold_cost * v[g, t])
            
            # For cheaper tiers, we can reduce the cost if generator was recently on
            for s in range(num_tiers - 1):
                # If tier s applies (downtime <= tier_periods[s]),
                # then generator was on at some point in [t-tier_periods[s], t-min_down]
                # 
                # Constraint: sc_cost[g,t] >= C_s * v[g,t] + (C_s - C_{s+1}) * sum u[g,j] for j outside cheaper window
                # 
                # Actually, the standard formulation:
                # sc_cost[g,t] >= C_s * (v[g,t] - sum_{j=t-tier_periods[s]}^{t-min_down} u[g,j])
                # But we need to handle initial conditions
                
                D_upper = tier_periods[s]
                
                # The window of u values to check: j in [t - D_upper, t - min_down]
                # For j < 0, use initial condition
                u_sum = gp.LinExpr()
                for j in range(t - D_upper, t - min_down + 1):
                    if j >= 0:
                        u_sum += u[g, j]
                    elif j < 0:
                        # j is before the horizon
                        # Initial status: if init_status > 0, generator was on
                        # It was on for init_status periods before period 0
                        # So at time j (which is negative), it was on if j >= -init_status
                        if init_status > 0 and j >= -init_status:
                            u_sum += 1
                        elif init_status < 0:
                            # Was off; u = 0 for all j < 0
                            pass
                        # init_status > 0 and j < -init_status: we don't know, but conservatively 0
                
                model2.addConstr(sc_cost[g, t] >= sc[s]['cost'] * (v[g, t] - u_sum))
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    model2.optimize(callback)
    
    # Extract solution
    if model2.SolCount == 0:
        # No solution found - try to return something
        print("No feasible solution found!")
        result = {
            "objective_value": float('inf'),
            "schedule": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return
    
    schedule = {}
    total_cost = 0.0
    
    for g in G:
        gen = generators[g]
        gen_id = gen['id']
        u_vals = []
        p_vals = []
        pb_vals = []
        
        for t in T:
            u_val = round(u[g, t].X)
            pa_val = p_above[g, t].X if u_val == 1 else 0.0
            pba_val = pb_above[g, t].X if u_val == 1 else 0.0
            
            actual_p = gen['p_min'] * u_val + pa_val
            actual_pb = gen['p_min'] * u_val + pba_val
            
            u_vals.append(int(u_val))
            p_vals.append(round(actual_p, 6))
            pb_vals.append(round(actual_pb, 6))
        
        schedule[gen_id] = {
            "u": u_vals,
            "p": p_vals,
            "pb": pb_vals
        }
    
    result = {
        "objective_value": model2.ObjVal,
        "schedule": schedule
    }
    
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_uc(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()