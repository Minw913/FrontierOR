import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_vessels = data['num_vessels']
    num_berths = data['num_berths']
    num_QCs = data['num_QCs']
    num_subblocks = data['num_subblocks']
    H = data['H']
    E = data['E']
    co = data['penalty_cost_coefficient_co']
    QC_cap = data['QC_capacity_per_timestep']
    D_L = data['loading_route_lengths_D_L']
    D_U = data['average_unloading_route_lengths_D_U']
    vessels = data['vessels']
    
    start_time = time.time()
    
    model = gp.Model("BACAP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_limit - 5))
    
    # Decision variables
    # x[v][b] = 1 if vessel v assigned to berth b
    x = {}
    for v in range(num_vessels):
        for b in range(num_berths):
            x[v, b] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{b}")
    
    # y[v][p] = 1 if vessel v uses profile p
    y = {}
    for v in range(num_vessels):
        for p in range(len(vessels[v]['qc_profiles'])):
            y[v, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}_{p}")
    
    # alpha[v] = start time for vessel v (integer)
    alpha = {}
    for v in range(num_vessels):
        ftw = vessels[v]['feasible_time_window']
        alpha[v] = model.addVar(vtype=GRB.INTEGER, lb=ftw[0], ub=ftw[1], name=f"alpha_{v}")
    
    # z[v][s] = 1 if subblock s assigned to vessel v
    z = {}
    for v in range(num_vessels):
        for s in range(num_subblocks):
            z[v, s] = model.addVar(vtype=GRB.BINARY, name=f"z_{v}_{s}")
    
    # beta[v] = end time for vessel v
    beta = {}
    for v in range(num_vessels):
        ftw = vessels[v]['feasible_time_window']
        beta[v] = model.addVar(vtype=GRB.INTEGER, lb=ftw[0], ub=ftw[1], name=f"beta_{v}")
    
    # Penalty variables
    early = {}
    late = {}
    for v in range(num_vessels):
        early[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"early_{v}")
        late[v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"late_{v}")
    
    # w[v][p][t] = 1 if vessel v uses profile p and starts at time t
    # This is needed to linearize the QC capacity constraints
    # To manage size, we create combined variable: for each vessel, start time
    # Actually, let's use a different approach for QC capacity
    
    # For QC capacity, we need to know for each time step t, how many QCs are used.
    # QC usage at time t = sum over vessels v: qc_usage_of_v_at_t
    # qc_usage_of_v_at_t depends on profile chosen and start time
    
    # Let's define: a[v][t] = 1 if vessel v starts at time t
    # Then combined with profile selection:
    # For vessel v with profile p starting at time t, at global time step t':
    #   if t <= t' <= t + handling_time_p - 1, usage = qc_per_timestep_p[t'-t]
    
    # This requires linearization. Let's use w[v,p,t] = y[v,p] * a[v,t]
    # But this creates too many variables for large instances.
    
    # Alternative: use indicator constraints or direct time-indexed formulation
    
    # Time-indexed formulation: 
    # Let's define gamma[v,t] = 1 if vessel v is being handled at time t AND starts at time t
    # Actually, let's use: s[v,t] = 1 if vessel v starts at time step t
    
    # Determine feasible start times for each vessel-profile combination
    vessel_start_times = {}
    for v in range(num_vessels):
        ftw = vessels[v]['feasible_time_window']
        profiles = vessels[v]['qc_profiles']
        starts = set()
        for p, prof in enumerate(profiles):
            ht = prof['handling_time']
            for t in range(ftw[0], ftw[1] - ht + 2):  # start from earliest, end such that beta <= ftw[1]
                starts.add(t)
        vessel_start_times[v] = sorted(starts)
    
    # s[v,t] = 1 if vessel v starts at time t
    s = {}
    for v in range(num_vessels):
        for t in vessel_start_times[v]:
            s[v, t] = model.addVar(vtype=GRB.BINARY, name=f"s_{v}_{t}")
    
    # w[v,p,t] = y[v,p] AND s[v,t] = 1 if vessel v uses profile p and starts at t
    w = {}
    for v in range(num_vessels):
        ftw = vessels[v]['feasible_time_window']
        profiles = vessels[v]['qc_profiles']
        for p, prof in enumerate(profiles):
            ht = prof['handling_time']
            for t in vessel_start_times[v]:
                if t + ht - 1 <= ftw[1]:
                    w[v, p, t] = model.addVar(vtype=GRB.BINARY, name=f"w_{v}_{p}_{t}")
    
    model.update()
    
    # Constraints
    
    # Each vessel assigned to exactly one berth
    for v in range(num_vessels):
        model.addConstr(gp.quicksum(x[v, b] for b in range(num_berths)) == 1, name=f"berth_assign_{v}")
    
    # Each vessel uses exactly one profile
    for v in range(num_vessels):
        model.addConstr(gp.quicksum(y[v, p] for p in range(len(vessels[v]['qc_profiles']))) == 1, name=f"profile_assign_{v}")
    
    # Each vessel starts at exactly one time
    for v in range(num_vessels):
        model.addConstr(gp.quicksum(s[v, t] for t in vessel_start_times[v]) == 1, name=f"start_assign_{v}")
    
    # Link alpha to s
    for v in range(num_vessels):
        model.addConstr(alpha[v] == gp.quicksum(t * s[v, t] for t in vessel_start_times[v]), name=f"alpha_link_{v}")
    
    # Link beta: beta[v] = alpha[v] + handling_time - 1
    # beta[v] = sum over (p,t) of w[v,p,t] * (t + ht_p - 1)
    for v in range(num_vessels):
        ftw = vessels[v]['feasible_time_window']
        profiles = vessels[v]['qc_profiles']
        expr = gp.LinExpr()
        for p, prof in enumerate(profiles):
            ht = prof['handling_time']
            for t in vessel_start_times[v]:
                if (v, p, t) in w:
                    expr += w[v, p, t] * (t + ht - 1)
        model.addConstr(beta[v] == expr, name=f"beta_link_{v}")
    
    # w[v,p,t] = y[v,p] * s[v,t] linearization
    # w <= y, w <= s, w >= y + s - 1
    # Also: sum of all w[v,p,t] over p,t = 1 (exactly one combination)
    for v in range(num_vessels):
        ftw = vessels[v]['feasible_time_window']
        profiles = vessels[v]['qc_profiles']
        w_sum = gp.LinExpr()
        for p, prof in enumerate(profiles):
            ht = prof['handling_time']
            for t in vessel_start_times[v]:
                if (v, p, t) in w:
                    model.addConstr(w[v, p, t] <= y[v, p], name=f"w_y_{v}_{p}_{t}")
                    model.addConstr(w[v, p, t] <= s[v, t], name=f"w_s_{v}_{p}_{t}")
                    model.addConstr(w[v, p, t] >= y[v, p] + s[v, t] - 1, name=f"w_ys_{v}_{p}_{t}")
                    w_sum += w[v, p, t]
        # For profiles that don't fit certain start times, s[v,t] should be 0 if only infeasible profiles
        # Actually, we need: for each v, sum of w[v,p,t] = 1
        model.addConstr(w_sum == 1, name=f"w_sum_{v}")
        
        # If a start time t doesn't work with profile p (beta > ftw[1]), w[v,p,t] doesn't exist
        # Need to ensure s[v,t] = 0 if no profile works at time t
        for t in vessel_start_times[v]:
            feasible_profiles = [p for p in range(len(profiles)) if (v, p, t) in w]
            if not feasible_profiles:
                model.addConstr(s[v, t] == 0, name=f"infeasible_start_{v}_{t}")
    
    # Subblock assignment: each vessel gets required number of subblocks
    for v in range(num_vessels):
        model.addConstr(gp.quicksum(z[v, s_idx] for s_idx in range(num_subblocks)) == vessels[v]['reserved_subblocks'], name=f"subblock_count_{v}")
    
    # Each subblock assigned to at most one vessel
    for s_idx in range(num_subblocks):
        model.addConstr(gp.quicksum(z[v, s_idx] for v in range(num_vessels)) <= 1, name=f"subblock_unique_{s_idx}")
    
    # Penalty constraints
    for v in range(num_vessels):
        etw = vessels[v]['expected_time_window']
        model.addConstr(early[v] >= etw[0] - alpha[v], name=f"early_{v}")
        model.addConstr(late[v] >= beta[v] - etw[1], name=f"late_{v}")
    
    # Berth non-overlap: if two vessels share a berth, they must not overlap in time
    # For vessels v1, v2 sharing berth b: either v1 finishes before v2 starts or vice versa
    # Use big-M or ordering variables
    
    # For each pair of vessels, for each berth:
    # If x[v1,b] = 1 and x[v2,b] = 1, then either beta[v1] < alpha[v2] or beta[v2] < alpha[v1]
    
    # Binary variable: order[v1,v2] = 1 means v1 before v2 when sharing a berth
    # We only need this for pairs where they could share a berth
    
    M_time = H + E + 10  # big M for time constraints
    
    order = {}
    for v1 in range(num_vessels):
        for v2 in range(v1 + 1, num_vessels):
            order[v1, v2] = model.addVar(vtype=GRB.BINARY, name=f"order_{v1}_{v2}")
    
    model.update()
    
    for v1 in range(num_vessels):
        for v2 in range(v1 + 1, num_vessels):
            for b in range(num_berths):
                # If both at berth b, enforce ordering
                # x[v1,b] + x[v2,b] - 1 <= indicator that they share berth b
                # If order[v1,v2]=1: beta[v1] < alpha[v2], i.e., beta[v1] <= alpha[v2] - 1
                # If order[v1,v2]=0: beta[v2] < alpha[v1], i.e., beta[v2] <= alpha[v1] - 1
                
                # beta[v1] - alpha[v2] + 1 <= M*(1 - order[v1,v2]) + M*(2 - x[v1,b] - x[v2,b])
                model.addConstr(
                    beta[v1] - alpha[v2] + 1 <= M_time * (1 - order[v1, v2]) + M_time * (2 - x[v1, b] - x[v2, b]),
                    name=f"nooverlap1_{v1}_{v2}_{b}"
                )
                model.addConstr(
                    beta[v2] - alpha[v1] + 1 <= M_time * order[v1, v2] + M_time * (2 - x[v1, b] - x[v2, b]),
                    name=f"nooverlap2_{v1}_{v2}_{b}"
                )
    
    # Berth span constraint: for each berth, the gap between earliest start and latest end <= H-1
    # rho[b] = min alpha[v] for v at berth b, varsigma[b] = max beta[v] for v at berth b
    # varsigma[b] - rho[b] <= H - 1
    
    # We need auxiliary variables for rho and varsigma per berth
    rho = {}
    varsigma = {}
    for b in range(num_berths):
        rho[b] = model.addVar(vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"rho_{b}")
        varsigma[b] = model.addVar(vtype=GRB.INTEGER, lb=1, ub=H + E, name=f"varsigma_{b}")
    
    model.update()
    
    for b in range(num_berths):
        for v in range(num_vessels):
            # If vessel v at berth b: rho[b] <= alpha[v], varsigma[b] >= beta[v]
            model.addConstr(rho[b] <= alpha[v] + M_time * (1 - x[v, b]), name=f"rho_lb_{b}_{v}")
            model.addConstr(varsigma[b] >= beta[v] - M_time * (1 - x[v, b]), name=f"varsigma_ub_{b}_{v}")
        
        # Span constraint
        model.addConstr(varsigma[b] - rho[b] <= H - 1, name=f"berth_span_{b}")
    
    # QC capacity constraints
    # For each time step t from E+1 to H: sum of QC usage at t <= QC_cap[t-1] (0-indexed)
    # For each time step t from 1 to E: QC usage at t + QC usage at t+H <= QC_cap[t-1]
    
    # QC usage at global time step t = sum over v,p,t_start of w[v,p,t_start] * qc_per_timestep_p[t - t_start]
    # where t_start <= t <= t_start + ht_p - 1
    
    # Pre-compute: for each global time step t, which (v,p,t_start) contribute
    qc_at_timestep = {}  # t -> list of (v,p,t_start, qc_amount)
    for v in range(num_vessels):
        profiles = vessels[v]['qc_profiles']
        for p, prof in enumerate(profiles):
            ht = prof['handling_time']
            qc_per = prof['qc_per_timestep']
            for t_start in vessel_start_times[v]:
                if (v, p, t_start) not in w:
                    continue
                for dt in range(ht):
                    t_global = t_start + dt
                    if t_global not in qc_at_timestep:
                        qc_at_timestep[t_global] = []
                    qc_at_timestep[t_global].append((v, p, t_start, qc_per[dt]))
    
    # Constraint for t from E+1 to H (1-indexed, so these are time steps E+1..H)
    for t in range(E + 1, H + 1):
        if t in qc_at_timestep:
            expr = gp.LinExpr()
            for (v, p, t_start, qc_amount) in qc_at_timestep[t]:
                expr += qc_amount * w[v, p, t_start]
            model.addConstr(expr <= QC_cap[t - 1], name=f"qc_cap_{t}")
    
    # Constraint for t from 1 to E: usage at t + usage at t+H <= QC_cap[t-1]
    for t in range(1, E + 1):
        expr = gp.LinExpr()
        if t in qc_at_timestep:
            for (v, p, t_start, qc_amount) in qc_at_timestep[t]:
                expr += qc_amount * w[v, p, t_start]
        t2 = t + H
        if t2 in qc_at_timestep:
            for (v, p, t_start, qc_amount) in qc_at_timestep[t2]:
                expr += qc_amount * w[v, p, t_start]
        if expr.size() > 0:
            model.addConstr(expr <= QC_cap[t - 1], name=f"qc_cap_wrap_{t}")
    
    # Objective
    # 1. Penalty cost
    penalty_cost = gp.LinExpr()
    for v in range(num_vessels):
        cv = vessels[v]['penalty_cost_coefficient']
        penalty_cost += cv * (early[v] + late[v])
    
    # 2. Loading transportation cost
    # For vessel v at berth b with subblocks S_v:
    # co * sum_{s in S_v} D_L[s][b] * loading_containers[v] / reserved_subblocks[v]
    # = co * loading_containers[v] / reserved_subblocks[v] * sum_{s in S_v} D_L[s][b]
    # Linearize: involves z[v,s] * x[v,b]
    
    # We need auxiliary: u[v,s,b] = z[v,s] * x[v,b]
    # But this could be huge. Let's think of alternatives.
    
    # Actually, the loading cost for vessel v = co * (loading_containers[v] / reserved_subblocks[v]) * sum_s sum_b z[v,s] * x[v,b] * D_L[s][b]
    # = co * (loading_containers[v] / reserved_subblocks[v]) * sum_s z[v,s] * sum_b x[v,b] * D_L[s][b]
    # Since vessel v is at exactly one berth, sum_b x[v,b] * D_L[s][b] is the distance from subblock s to the assigned berth
    
    # Let's define: for each vessel v, d_load[v] = sum_s sum_b z[v,s] * x[v,b] * D_L[s][b]
    # We can introduce: for each vessel v and berth b, let load_dist_at_berth[v,b] = x[v,b] * sum_s z[v,s] * D_L[s][b]
    # Hmm, still trilinear.
    
    # Alternative approach: introduce u[v,s,b] = z[v,s] * x[v,b]
    # Then d_load[v] = sum_s sum_b u[v,s,b] * D_L[s][b]
    # u[v,s,b] <= z[v,s], u[v,s,b] <= x[v,b], u[v,s,b] >= z[v,s] + x[v,b] - 1
    
    # This is potentially large. Let's check sizes:
    # num_vessels * num_subblocks * num_berths auxiliary binary variables
    # For small instances this is fine. For larger ones we may need to be smarter.
    
    # Alternative: since x[v,b] is binary and sum_b x[v,b] = 1,
    # we can define dist_v_s = sum_b x[v,b] * D_L[s][b] (continuous)
    # Then loading cost for v = co * (LC_v / RS_v) * sum_s z[v,s] * dist_v_s
    # But dist_v_s * z[v,s] is still bilinear.
    
    # Let's just use the u variables approach but only create them when needed.
    # Actually, we can decompose differently. 
    # 
    # Cost_load_v = co * (LC_v / RS_v) * sum_b x[v,b] * sum_s z[v,s] * D_L[s][b]
    # Define: load_sum[v,b] = sum_s z[v,s] * D_L[s][b] (continuous variable, since z[v,s] binary)
    # Then Cost_load_v = co * (LC_v / RS_v) * sum_b x[v,b] * load_sum[v,b]
    # Still bilinear: x[v,b] * load_sum[v,b]
    
    # Since we know the number of reserved subblocks is fixed and D_L values are known,
    # load_sum[v,b] has known bounds.
    
    # Let's use a different decomposition. We can introduce:
    # For each vessel v, berth b, subblock s: u[v,s,b] = z[v,s] * x[v,b]
    # But to reduce variables, note that the subblock assignment and berth assignment are 
    # separable in the objective if we can find a good decomposition.
    
    # Actually for moderate instance sizes, let's try direct u variables.
    # But if num_vessels=10, num_subblocks=20, num_berths=5, that's 1000 variables - fine.
    # If much larger, we'd need a different approach.
    
    # Let me check: can we separate subblock and berth decisions?
    # The loading cost = co * (LC_v/RS_v) * sum_s z[v,s] * D_L[s][berth_v]
    # where berth_v is the assigned berth.
    # This couples z and x. We need to linearize.
    
    # Use auxiliary continuous variables for the product
    loading_cost = gp.LinExpr()
    
    # For efficiency, let's create variables load_contrib[v,b] = x[v,b] * sum_s z[v,s]*D_L[s][b]
    # Use big-M linearization with load_contrib_var
    
    # Actually, let me use u[v,s,b] approach more carefully
    # u[v,s,b] binary = z[v,s] AND x[v,b]
    # loading_cost += co * (LC_v/RS_v) * D_L[s][b] * u[v,s,b]
    
    # For the u constraints, we only need:
    # u[v,s,b] >= z[v,s] + x[v,b] - 1  (this is enough with the minimization direction IF D_L > 0)
    # u[v,s,b] <= z[v,s]
    # u[v,s,b] <= x[v,b]
    # Actually in minimization with positive coefficients, the solver would want u=0.
    # So we need the lower bound: u >= z + x - 1.
    # But we also need u <= z and u <= x for correctness.
    
    # Alternatively, since we're minimizing and the coefficient is positive, 
    # the solver will try to make u small. The binding constraint is u >= z+x-1.
    # But u could be fractional in LP relaxation. Let's make u continuous [0,1] 
    # with the three constraints. That should work since z and x are binary.
    
    u = {}
    for v in range(num_vessels):
        lc = vessels[v]['loading_containers']
        rs = vessels[v]['reserved_subblocks']
        if lc == 0 or rs == 0:
            continue
        coeff_base = co * lc / rs
        for s_idx in range(num_subblocks):
            for b in range(num_berths):
                dl = D_L[s_idx][b]
                if dl == 0 and coeff_base == 0:
                    continue
                u[v, s_idx, b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"u_{v}_{s_idx}_{b}")
                loading_cost += coeff_base * dl * u[v, s_idx, b]
    
    model.update()
    
    for (v, s_idx, b) in u:
        model.addConstr(u[v, s_idx, b] >= z[v, s_idx] + x[v, b] - 1, name=f"u_lb_{v}_{s_idx}_{b}")
        model.addConstr(u[v, s_idx, b] <= z[v, s_idx], name=f"u_ub1_{v}_{s_idx}_{b}")
        model.addConstr(u[v, s_idx, b] <= x[v, b], name=f"u_ub2_{v}_{s_idx}_{b}")
    
    # 3. Unloading transportation cost
    unloading_cost = gp.LinExpr()
    for v in range(num_vessels):
        uc = vessels[v]['unloading_containers']
        if uc == 0:
            continue
        for b in range(num_berths):
            unloading_cost += co * D_U[b] * uc * x[v, b]
    
    # Total objective
    model.setObjective(penalty_cost + loading_cost + unloading_cost, GRB.MINIMIZE)
    
    # Store best solution found via callback
    best_solution = [None]
    best_obj = [float('inf')]
    
    def extract_solution():
        """Extract solution from model."""
        sol_vessels = []
        for v in range(num_vessels):
            # Find assigned berth
            assigned_berth = None
            for b in range(num_berths):
                if x[v, b].X > 0.5:
                    assigned_berth = b
                    break
            
            # Find chosen profile and start time
            chosen_profile = None
            chosen_start = None
            for p in range(len(vessels[v]['qc_profiles'])):
                for t in vessel_start_times[v]:
                    if (v, p, t) in w and w[v, p, t].X > 0.5:
                        chosen_profile = p
                        chosen_start = t
                        break
                if chosen_profile is not None:
                    break
            
            prof = vessels[v]['qc_profiles'][chosen_profile]
            ht = prof['handling_time']
            qc_per = prof['qc_per_timestep']
            alpha_v = chosen_start
            beta_v = chosen_start + ht - 1
            
            # Find assigned subblocks
            assigned_subblocks = []
            for s_idx in range(num_subblocks):
                if z[v, s_idx].X > 0.5:
                    assigned_subblocks.append(s_idx)
            
            sol_vessels.append({
                "vessel_id": vessels[v]['vessel_id'],
                "alpha": alpha_v,
                "beta": beta_v,
                "berth": assigned_berth,
                "profile": chosen_profile,
                "subblocks": assigned_subblocks,
                "handling_time": ht,
                "qc_per_timestep": qc_per
            })
        
        # Berth info
        sol_berths = []
        for b in range(num_berths):
            berth_vessels = [sv for sv in sol_vessels if sv['berth'] == b]
            if berth_vessels:
                rho_val = min(sv['alpha'] for sv in berth_vessels)
                varsigma_val = max(sv['beta'] for sv in berth_vessels)
            else:
                rho_val = 1
                varsigma_val = 1
            sol_berths.append({
                "berth": b,
                "rho": rho_val,
                "varsigma": varsigma_val
            })
        
        # QC usage
        qc_usage = {}
        for sv in sol_vessels:
            for dt in range(sv['handling_time']):
                t_global = sv['alpha'] + dt
                key = str(t_global)
                if key not in qc_usage:
                    qc_usage[key] = 0
                qc_usage[key] += sv['qc_per_timestep'][dt]
        # Remove zero entries
        qc_usage = {k: v for k, v in qc_usage.items() if v > 0}
        
        obj_val = model.ObjVal
        
        solution = {
            "objective_value": obj_val,
            "vessels": sol_vessels,
            "berths": sol_berths,
            "qc_usage": qc_usage
        }
        return solution
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    model.optimize(callback)
    
    if model.SolCount > 0:
        solution = extract_solution()
        best_solution[0] = solution
        if logger:
            logger.log_solution(solution['objective_value'], solution)
    else:
        # Return empty solution
        solution = {
            "objective_value": float('inf'),
            "vessels": [],
            "berths": [{"berth": b, "rho": 1, "varsigma": 1} for b in range(num_berths)],
            "qc_usage": {}
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == '__main__':
    main()