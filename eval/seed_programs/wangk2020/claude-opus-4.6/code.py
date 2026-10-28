import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        data = json.load(f)

    T = data['num_time_periods']
    delta_max_disp = data['maximum_displacement_delta']
    airports = data['airports']
    flights = data['flights']
    aircraft_connections = data['aircraft_connections']
    passenger_connections = data['passenger_connections']
    capacity_envelopes = data['capacity_envelopes']
    scenarios = data['scenarios']
    rho = 0.5  # Default rho; check if provided
    if 'rho' in data:
        rho = data['rho']

    num_flights = len(flights)
    num_scenarios = len(scenarios)
    
    # Build flight lookup
    flight_by_id = {}
    for f_data in flights:
        flight_by_id[f_data['id']] = f_data

    start_time = time.time()

    model = gp.Model("ATFM")
    model.setParam('TimeLimit', max(1, time_limit - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)

    # Period indices: 1..T (we use 1-based indexing matching the problem)
    periods = list(range(1, T + 1))

    # ---- First-stage variables ----
    # For each flight f, scheduling window for departure:
    # scheduled departure can be in [dep_period - delta_max_disp, dep_period + delta_max_disp]
    # clipped to [1, T]
    
    # y_dep[f,t] = 1 if flight f is scheduled to depart no earlier than period t
    # y_arr[f,t] = 1 if flight f is scheduled to arrive no earlier than period t
    # These are non-increasing sequences that encode times.
    # scheduled_dep(f) = sum_t y_dep[f,t] (over t=1..T) ... wait, let me re-read.
    
    # Actually, the encoding: y[t]=1 if scheduled to depart no earlier than t.
    # If scheduled departure is at period p, then y[t]=1 for t<=p and y[t]=0 for t>p.
    # So sum of y over all t = p = scheduled departure period.
    # The sequence is non-increasing: y[t] >= y[t+1].
    
    # Scheduling window for departure of flight f:
    # [max(1, dep_period - delta_max_disp), min(T, dep_period + delta_max_disp)]
    
    y_dep = {}
    y_arr = {}
    
    for f_data in flights:
        fid = f_data['id']
        dep_req = f_data['dep_period']
        arr_req = f_data['arr_period']
        delta_sch = f_data['delta_sch']
        
        dep_lo = max(1, dep_req - delta_max_disp)
        dep_hi = min(T, dep_req + delta_max_disp)
        arr_lo = dep_lo + delta_sch
        arr_hi = dep_hi + delta_sch
        
        # y_dep[fid,t] for t in 1..T
        # For t <= dep_lo, y_dep must be 1 (since scheduled_dep >= dep_lo)
        # For t > dep_hi, y_dep must be 0
        for t in periods:
            if t <= dep_lo:
                y_dep[fid, t] = 1  # constant
            elif t > dep_hi:
                y_dep[fid, t] = 0  # constant
            else:
                y_dep[fid, t] = model.addVar(vtype=GRB.BINARY, name=f"yd_{fid}_{t}")
        
        for t in periods:
            if t <= arr_lo:
                y_arr[fid, t] = 1
            elif t > arr_hi:
                y_arr[fid, t] = 0
            else:
                y_arr[fid, t] = model.addVar(vtype=GRB.BINARY, name=f"ya_{fid}_{t}")

    model.update()

    # Non-increasing constraints for y_dep and y_arr
    for f_data in flights:
        fid = f_data['id']
        dep_req = f_data['dep_period']
        delta_sch = f_data['delta_sch']
        dep_lo = max(1, dep_req - delta_max_disp)
        dep_hi = min(T, dep_req + delta_max_disp)
        arr_lo = dep_lo + delta_sch
        arr_hi = dep_hi + delta_sch
        
        for t in range(1, T):
            # y_dep[fid,t] >= y_dep[fid,t+1]
            yt = y_dep[fid, t]
            yt1 = y_dep[fid, t + 1]
            if isinstance(yt, (int, float)) and isinstance(yt1, (int, float)):
                continue  # both constant, constraint automatically satisfied if data is correct
            elif isinstance(yt1, (int, float)):
                continue  # yt1 is constant 0 or yt is var >= 0, fine
            elif isinstance(yt, (int, float)):
                if yt == 1:
                    continue  # 1 >= yt1, always true for binary
                else:
                    model.addConstr(yt1 <= 0)
            else:
                model.addConstr(yt >= yt1, name=f"mono_yd_{fid}_{t}")
        
        for t in range(1, T):
            yt = y_arr[fid, t]
            yt1 = y_arr[fid, t + 1]
            if isinstance(yt, (int, float)) and isinstance(yt1, (int, float)):
                continue
            elif isinstance(yt1, (int, float)):
                continue
            elif isinstance(yt, (int, float)):
                if yt == 1:
                    continue
                else:
                    model.addConstr(yt1 <= 0)
            else:
                model.addConstr(yt >= yt1, name=f"mono_ya_{fid}_{t}")

    # En-route duration constraint for schedule:
    # sum_t y_arr[fid,t] - sum_t y_dep[fid,t] = delta_sch
    for f_data in flights:
        fid = f_data['id']
        delta_sch = f_data['delta_sch']
        
        sum_dep = gp.quicksum(y_dep[fid, t] if isinstance(y_dep[fid, t], gp.Var) else y_dep[fid, t] for t in periods)
        sum_arr = gp.quicksum(y_arr[fid, t] if isinstance(y_arr[fid, t], gp.Var) else y_arr[fid, t] for t in periods)
        
        model.addConstr(sum_arr - sum_dep == delta_sch, name=f"enroute_sch_{fid}")

    # Passenger connections (first set - schedule constraints):
    for conn in passenger_connections:
        fi = conn['flight_i']
        fj = conn['flight_j']
        min_ct = conn['min_connection_time']
        
        sum_dep_j = gp.quicksum(y_dep[fj, t] if isinstance(y_dep[fj, t], gp.Var) else y_dep[fj, t] for t in periods)
        sum_arr_i = gp.quicksum(y_arr[fi, t] if isinstance(y_arr[fi, t], gp.Var) else y_arr[fi, t] for t in periods)
        
        model.addConstr(sum_dep_j - sum_arr_i >= min_ct, name=f"pax_conn_{fi}_{fj}")

    # Aircraft connections also apply to schedule (first set includes both)
    for conn in aircraft_connections:
        fi = conn['flight_i']
        fj = conn['flight_j']
        min_ct = conn['min_connection_time']
        
        sum_dep_j = gp.quicksum(y_dep[fj, t] if isinstance(y_dep[fj, t], gp.Var) else y_dep[fj, t] for t in periods)
        sum_arr_i = gp.quicksum(y_arr[fi, t] if isinstance(y_arr[fi, t], gp.Var) else y_arr[fi, t] for t in periods)
        
        # This constraint might already be covered by passenger_connections if they overlap
        # But we add it for aircraft connections as part of the "first set"
        # Actually re-reading: "a first set of connected pairs covers both aircraft turnaround connections and passenger itinerary connections"
        # "a second set covers only aircraft turnaround connections"
        # passenger_connections = first set (schedule + operations)
        # aircraft_connections = second set (operations only)
        # So aircraft_connections don't need schedule constraints, only operational ones.
        # Let me re-read more carefully.
        
        # "For every connected flight pair in the first set of connections, the difference between 
        #  the sum of the second flight's scheduled-departure indicators and the sum of the first 
        #  flight's scheduled-arrival indicators over all time periods must be at least the minimum 
        #  connecting time for that pair."
        # The first set = passenger_connections (which covers both aircraft turnaround and passenger itinerary)
        # The second set = aircraft_connections (only aircraft turnaround, operational constraints only)
        
        # So we should NOT add schedule constraints for aircraft_connections.
        pass

    # ---- Second-stage variables ----
    x_dep = {}
    x_arr = {}
    v_dep = {}
    v_arr = {}

    for s_data in scenarios:
        sid = s_data['id']
        
        for f_data in flights:
            fid = f_data['id']
            dep_req = f_data['dep_period']
            delta_sch = f_data['delta_sch']
            max_dep_delay = f_data['max_dep_delay']
            max_arr_delay = f_data['max_arr_delay']
            delta_min = f_data['delta_min']
            delta_max = f_data['delta_max']
            
            dep_lo = max(1, dep_req - delta_max_disp)
            dep_hi = min(T, dep_req + delta_max_disp)
            
            # Operated departure can range from scheduled dep to scheduled dep + max_dep_delay
            # Scheduled dep ranges from dep_lo to dep_hi
            # So operated dep ranges from dep_lo to dep_hi + max_dep_delay
            op_dep_lo = dep_lo
            op_dep_hi = min(T, dep_hi + max_dep_delay)
            
            arr_lo_op = op_dep_lo + delta_min
            arr_hi_op = min(T, op_dep_hi + delta_max)
            
            for t in periods:
                if t <= op_dep_lo:
                    x_dep[sid, fid, t] = 1
                elif t > op_dep_hi:
                    x_dep[sid, fid, t] = 0
                else:
                    x_dep[sid, fid, t] = model.addVar(vtype=GRB.BINARY, name=f"xd_{sid}_{fid}_{t}")
            
            for t in periods:
                if t <= arr_lo_op:
                    x_arr[sid, fid, t] = 1
                elif t > arr_hi_op:
                    x_arr[sid, fid, t] = 0
                else:
                    x_arr[sid, fid, t] = model.addVar(vtype=GRB.BINARY, name=f"xa_{sid}_{fid}_{t}")
            
            v_dep[sid, fid] = model.addVar(lb=0, ub=max_dep_delay, vtype=GRB.INTEGER, name=f"vd_{sid}_{fid}")
            v_arr[sid, fid] = model.addVar(lb=0, ub=max_arr_delay, vtype=GRB.INTEGER, name=f"va_{sid}_{fid}")

    model.update()

    # Non-increasing constraints for x_dep, x_arr
    for s_data in scenarios:
        sid = s_data['id']
        for f_data in flights:
            fid = f_data['id']
            for t in range(1, T):
                xt = x_dep[sid, fid, t]
                xt1 = x_dep[sid, fid, t + 1]
                if isinstance(xt, (int, float)) and isinstance(xt1, (int, float)):
                    continue
                elif isinstance(xt1, (int, float)):
                    continue
                elif isinstance(xt, (int, float)):
                    if xt == 1:
                        continue
                    else:
                        model.addConstr(xt1 <= 0)
                else:
                    model.addConstr(xt >= xt1, name=f"mono_xd_{sid}_{fid}_{t}")
                
                xt = x_arr[sid, fid, t]
                xt1 = x_arr[sid, fid, t + 1]
                if isinstance(xt, (int, float)) and isinstance(xt1, (int, float)):
                    continue
                elif isinstance(xt1, (int, float)):
                    continue
                elif isinstance(xt, (int, float)):
                    if xt == 1:
                        continue
                    else:
                        model.addConstr(xt1 <= 0)
                else:
                    model.addConstr(xt >= xt1, name=f"mono_xa_{sid}_{fid}_{t}")

    # Departure delay: v_dep[sid,fid] = sum_t x_dep[sid,fid,t] - sum_t y_dep[fid,t]
    for s_data in scenarios:
        sid = s_data['id']
        for f_data in flights:
            fid = f_data['id']
            
            sum_xd = gp.quicksum(x_dep[sid, fid, t] if isinstance(x_dep[sid, fid, t], gp.Var) else x_dep[sid, fid, t] for t in periods)
            sum_yd = gp.quicksum(y_dep[fid, t] if isinstance(y_dep[fid, t], gp.Var) else y_dep[fid, t] for t in periods)
            
            model.addConstr(v_dep[sid, fid] == sum_xd - sum_yd, name=f"vdep_{sid}_{fid}")

    # Arrival delay: v_arr[sid,fid] >= sum_t x_arr[sid,fid,t] - sum_t y_arr[fid,t]
    for s_data in scenarios:
        sid = s_data['id']
        for f_data in flights:
            fid = f_data['id']
            
            sum_xa = gp.quicksum(x_arr[sid, fid, t] if isinstance(x_arr[sid, fid, t], gp.Var) else x_arr[sid, fid, t] for t in periods)
            sum_ya = gp.quicksum(y_arr[fid, t] if isinstance(y_arr[fid, t], gp.Var) else y_arr[fid, t] for t in periods)
            
            model.addConstr(v_arr[sid, fid] >= sum_xa - sum_ya, name=f"varr_{sid}_{fid}")

    # En-route duration constraints for operated flights:
    # delta_min <= sum_t x_arr - sum_t x_dep <= delta_max
    for s_data in scenarios:
        sid = s_data['id']
        for f_data in flights:
            fid = f_data['id']
            delta_min = f_data['delta_min']
            delta_max_val = f_data['delta_max']
            
            sum_xd = gp.quicksum(x_dep[sid, fid, t] if isinstance(x_dep[sid, fid, t], gp.Var) else x_dep[sid, fid, t] for t in periods)
            sum_xa = gp.quicksum(x_arr[sid, fid, t] if isinstance(x_arr[sid, fid, t], gp.Var) else x_arr[sid, fid, t] for t in periods)
            
            enroute = sum_xa - sum_xd
            model.addConstr(enroute >= delta_min, name=f"enr_min_{sid}_{fid}")
            model.addConstr(enroute <= delta_max_val, name=f"enr_max_{sid}_{fid}")

    # Aircraft connections (second set) - operational constraints only
    for conn in aircraft_connections:
        fi = conn['flight_i']
        fj = conn['flight_j']
        min_ct = conn['min_connection_time']
        
        for s_data in scenarios:
            sid = s_data['id']
            
            sum_xd_j = gp.quicksum(x_dep[sid, fj, t] if isinstance(x_dep[sid, fj, t], gp.Var) else x_dep[sid, fj, t] for t in periods)
            sum_xa_i = gp.quicksum(x_arr[sid, fi, t] if isinstance(x_arr[sid, fi, t], gp.Var) else x_arr[sid, fi, t] for t in periods)
            
            model.addConstr(sum_xd_j - sum_xa_i >= min_ct, name=f"ac_conn_{sid}_{fi}_{fj}")

    # Passenger connections - operational constraints (in addition to schedule constraints)
    # Wait - re-reading: "For every connected flight pair in the aircraft-only connection set, 
    # the difference between the sum of the second flight's operated-departure indicators and 
    # the sum of the first flight's operated-arrival indicators must be at least the minimum 
    # connecting time for that pair."
    # Only aircraft_connections have operational constraints explicitly mentioned.
    # But logically passenger connections should also be enforced operationally... 
    # Let me re-read the problem more carefully.
    
    # "For every connected flight pair in the aircraft-only connection set" - this is the second set
    # The first set has schedule constraints but the problem doesn't explicitly say it has operational constraints.
    # This seems like the first set (passenger_connections) only has schedule-level constraints,
    # while the second set (aircraft_connections) only has operational-level constraints.

    # Capacity constraints
    # Number of departures at airport a in period t under scenario s:
    # dep_count[s,a,t] = sum over flights departing from a of (x_dep[s,f,t] - x_dep[s,f,t+1])
    # For t=T, it's x_dep[s,f,T] (since x_dep[s,f,T+1] = 0)
    
    # Group flights by departure and arrival airports
    flights_dep_at = {a: [] for a in airports}
    flights_arr_at = {a: [] for a in airports}
    for f_data in flights:
        flights_dep_at[f_data['dep_airport']].append(f_data['id'])
        flights_arr_at[f_data['arr_airport']].append(f_data['id'])

    for s_data in scenarios:
        sid = s_data['id']
        oc = s_data['operating_conditions']
        
        for a in airports:
            for t_idx, t in enumerate(periods):
                # Determine weather condition
                cond = oc[a][t_idx]  # VMC or IMC
                envelope = capacity_envelopes[a][cond]
                
                if not envelope:
                    continue
                
                # Compute departures at airport a in period t
                dep_terms = []
                for fid in flights_dep_at[a]:
                    xt = x_dep[sid, fid, t]
                    if t < T:
                        xt1 = x_dep[sid, fid, t + 1]
                    else:
                        xt1 = 0
                    
                    if isinstance(xt, (int, float)) and isinstance(xt1, (int, float)):
                        val = xt - xt1
                        if val != 0:
                            dep_terms.append(val)
                    elif isinstance(xt1, (int, float)):
                        if xt1 == 0:
                            dep_terms.append(xt)
                        else:
                            dep_terms.append(xt - xt1)
                    elif isinstance(xt, (int, float)):
                        dep_terms.append(xt - xt1)
                    else:
                        dep_terms.append(xt - xt1)
                
                arr_terms = []
                for fid in flights_arr_at[a]:
                    xt = x_arr[sid, fid, t]
                    if t < T:
                        xt1 = x_arr[sid, fid, t + 1]
                    else:
                        xt1 = 0
                    
                    if isinstance(xt, (int, float)) and isinstance(xt1, (int, float)):
                        val = xt - xt1
                        if val != 0:
                            arr_terms.append(val)
                    elif isinstance(xt1, (int, float)):
                        if xt1 == 0:
                            arr_terms.append(xt)
                        else:
                            arr_terms.append(xt - xt1)
                    elif isinstance(xt, (int, float)):
                        arr_terms.append(xt - xt1)
                    else:
                        arr_terms.append(xt - xt1)
                
                for seg in envelope:
                    a_coeff = seg['a']
                    b_coeff = seg['b']
                    Q = seg['Q']
                    
                    lhs_terms = []
                    const = 0
                    
                    for dt in dep_terms:
                        if isinstance(dt, (int, float)):
                            const += a_coeff * dt
                        else:
                            lhs_terms.append(a_coeff * dt)
                    
                    for at_term in arr_terms:
                        if isinstance(at_term, (int, float)):
                            const += b_coeff * at_term
                        else:
                            lhs_terms.append(b_coeff * at_term)
                    
                    if lhs_terms:
                        model.addConstr(gp.quicksum(lhs_terms) + const <= Q, 
                                       name=f"cap_{sid}_{a}_{t}_{seg['a']}_{seg['b']}")
                    # If only constants, check feasibility (should be ok)

    # ---- Objective ----
    # Schedule displacement cost:
    # For each flight, displacement = |scheduled_dep - requested_dep|
    # scheduled_dep = sum_t y_dep[fid,t]
    # displacement_cost = sum over flights of |sum_t y_dep[fid,t] - dep_period|
    # 
    # Actually, let me re-read: "the displacement cost of a flight equals the absolute difference 
    # between its scheduled and requested departure periods, extracted from the indicator encoding 
    # as the sum over periods of the displacement-cost coefficient times the drop in 
    # scheduled-departure indicators at that period"
    # 
    # The "drop" at period t is y_dep[fid,t] - y_dep[fid,t+1] (which is 1 at the scheduled dep period, 0 elsewhere)
    # "displacement-cost coefficient" for period t = |t - dep_period|
    # So displacement_cost(f) = sum_t |t - dep_period| * (y_dep[fid,t] - y_dep[fid,t+1])
    # Since the drop is 1 at exactly one period (the scheduled departure), this equals |scheduled_dep - dep_period|
    
    disp_cost_expr = []
    for f_data in flights:
        fid = f_data['id']
        dep_req = f_data['dep_period']
        dep_lo = max(1, dep_req - delta_max_disp)
        dep_hi = min(T, dep_req + delta_max_disp)
        
        # The drop at period t: y_dep[fid,t] - y_dep[fid,t+1] for t in 1..T (y_dep[fid,T+1]=0)
        # Only non-zero in the window [dep_lo, dep_hi]
        for t in range(dep_lo, dep_hi + 1):
            coeff = abs(t - dep_req)
            if coeff == 0:
                continue
            
            yt = y_dep[fid, t]
            if t < T:
                yt1 = y_dep[fid, t + 1]
            else:
                yt1 = 0
            
            # drop = yt - yt1
            if isinstance(yt, (int, float)) and isinstance(yt1, (int, float)):
                disp_cost_expr.append(coeff * (yt - yt1))
            elif isinstance(yt1, (int, float)):
                disp_cost_expr.append(coeff * (yt - yt1))
            elif isinstance(yt, (int, float)):
                disp_cost_expr.append(coeff * (yt - yt1))
            else:
                disp_cost_expr.append(coeff * (yt - yt1))

    # Expected operational delay cost:
    delay_cost_expr = []
    for s_data in scenarios:
        sid = s_data['id']
        prob = s_data['probability']
        
        for f_data in flights:
            fid = f_data['id']
            cd = f_data['cost_dep_delay']
            ca = f_data['cost_arr_delay']
            
            delay_cost_expr.append(prob * cd * v_dep[sid, fid])
            delay_cost_expr.append(prob * ca * v_arr[sid, fid])

    objective = rho * gp.quicksum(disp_cost_expr) + (1 - rho) * gp.quicksum(delay_cost_expr)
    model.setObjective(objective, GRB.MINIMIZE)

    # Callback for logging
    best_obj = [float('inf')]
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0] - 1e-9:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)

    model.optimize(callback)

    # Extract solution
    if model.SolCount == 0:
        # No solution found, try to output something
        print("No feasible solution found")
        sol = {
            "objective_value": float('inf'),
            "schedule": {},
            "scenario_solutions": []
        }
        with open(solution_path, 'w') as f:
            json.dump(sol, f)
        return

    obj_val = model.ObjVal

    # Extract scheduled times
    schedule = {}
    for f_data in flights:
        fid = f_data['id']
        
        sch_dep = 0
        for t in periods:
            val = y_dep[fid, t]
            if isinstance(val, gp.Var):
                sch_dep += round(val.X)
            else:
                sch_dep += val
        
        sch_arr = 0
        for t in periods:
            val = y_arr[fid, t]
            if isinstance(val, gp.Var):
                sch_arr += round(val.X)
            else:
                sch_arr += val
        
        schedule[str(fid)] = {
            "scheduled_dep": int(sch_dep),
            "scheduled_arr": int(sch_arr)
        }

    # Extract scenario solutions
    scenario_solutions = []
    for s_data in scenarios:
        sid = s_data['id']
        
        xd_dict = {}
        xa_dict = {}
        vd_dict = {}
        va_dict = {}
        
        for f_data in flights:
            fid = f_data['id']
            
            for t in periods:
                val = x_dep[sid, fid, t]
                if isinstance(val, gp.Var):
                    xd_dict[f"{fid}_{t}"] = int(round(val.X))
                else:
                    xd_dict[f"{fid}_{t}"] = int(val)
                
                val = x_arr[sid, fid, t]
                if isinstance(val, gp.Var):
                    xa_dict[f"{fid}_{t}"] = int(round(val.X))
                else:
                    xa_dict[f"{fid}_{t}"] = int(val)
            
            vd_dict[str(fid)] = int(round(v_dep[sid, fid].X))
            va_dict[str(fid)] = int(round(v_arr[sid, fid].X))
        
        scenario_solutions.append({
            "scenario_id": sid,
            "x_dep": xd_dict,
            "x_arr": xa_dict,
            "v_dep": vd_dict,
            "v_arr": va_dict
        })

    sol = {
        "objective_value": obj_val,
        "schedule": schedule,
        "scenario_solutions": scenario_solutions
    }

    with open(solution_path, 'w') as f:
        json.dump(sol, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)