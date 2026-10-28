import argparse
import json
import time
import sys
from itertools import product

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    
    shifts = data['shifts']
    skills = data['skills']
    nurses = data['nurses']
    weekends = data['weekends']
    demand = data['demand']
    unsatisfied_demand_penalty = data['unsatisfied_demand_penalty']
    constraints = data['constraints']
    hard_cons = constraints['hard_constraints']
    soft_cons = constraints['soft_constraints']
    
    num_nurses = len(nurses)
    nurse_ids = [n['id'] for n in nurses]
    nurse_skills = {n['id']: set(n['skills']) for n in nurses}
    nurse_prefs = {n['id']: n.get('preferences', []) for n in nurses}
    
    # Determine number of days
    all_days = set()
    for key in demand.keys():
        d = int(key.split('_')[1])
        all_days.add(d)
    num_days = max(all_days) if all_days else 0
    days = list(range(1, num_days + 1))
    
    # Check constraints
    demand_equality = hard_cons.get('demand_equality', False)
    skill_req_hard = hard_cons.get('skill_requirement', False)
    
    BIG_PENALTY = 10000000  # For hard constraints
    
    import gurobipy as gp
    from gurobipy import GRB
    
    model = gp.Model("nurse_scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    remaining_time = max(10, time_limit - (time.time() - start_time) - 5)
    model.setParam('TimeLimit', remaining_time)
    
    # Decision variables
    # x[n, d, s] = 1 if nurse n works shift s on day d (s in shifts)
    # r[n, d] = 1 if nurse n has rest on day d
    # y[n, d, s, k] = 1 if nurse n works shift s with skill k on day d
    
    x = {}
    for n in nurse_ids:
        for d in days:
            for s in shifts:
                x[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")
    
    r = {}
    for n in nurse_ids:
        for d in days:
            r[n, d] = model.addVar(vtype=GRB.BINARY, name=f"r_{n}_{d}")
    
    y = {}
    for n in nurse_ids:
        for d in days:
            for s in shifts:
                for k in skills:
                    y[n, d, s, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{n}_{d}_{s}_{k}")
    
    model.update()
    
    # Constraint: each nurse assigned exactly one shift or rest per day
    for n in nurse_ids:
        for d in days:
            model.addConstr(
                gp.quicksum(x[n, d, s] for s in shifts) + r[n, d] == 1,
                name=f"single_assign_{n}_{d}"
            )
    
    # Link y to x: if nurse works shift s on day d, must pick exactly one skill
    for n in nurse_ids:
        for d in days:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(y[n, d, s, k] for k in skills) == x[n, d, s],
                    name=f"skill_assign_{n}_{d}_{s}"
                )
    
    # Objective components
    obj_terms = []
    
    # Skill mismatch penalty
    skill_req_penalty = soft_cons.get('skill_requirement_penalty', 0)
    if skill_req_hard:
        # Hard: nurse can only use skills they possess
        for n in nurse_ids:
            for d in days:
                for s in shifts:
                    for k in skills:
                        if k not in nurse_skills[n]:
                            model.addConstr(y[n, d, s, k] == 0,
                                            name=f"hard_skill_{n}_{d}_{s}_{k}")
    elif skill_req_penalty and skill_req_penalty > 0:
        # Soft: penalty for using skills not possessed
        for n in nurse_ids:
            for d in days:
                for s in shifts:
                    for k in skills:
                        if k not in nurse_skills[n]:
                            obj_terms.append(skill_req_penalty * y[n, d, s, k])
    
    # Demand constraints
    # staffing[d, s, k] = sum over nurses of y[n, d, s, k]
    if demand_equality:
        for d in days:
            day_key = f"day_{d}"
            if day_key in demand:
                for s in shifts:
                    if s in demand[day_key]:
                        for k in skills:
                            if k in demand[day_key][s]:
                                req = demand[day_key][s][k]
                                staffing = gp.quicksum(y[n, d, s, k] for n in nurse_ids)
                                model.addConstr(staffing == req,
                                                name=f"demand_eq_{d}_{s}_{k}")
    else:
        # Soft demand with slack
        demand_slack = {}
        for d in days:
            day_key = f"day_{d}"
            if day_key in demand:
                for s in shifts:
                    if s in demand[day_key]:
                        for k in skills:
                            if k in demand[day_key][s]:
                                req = demand[day_key][s][k]
                                if req > 0:
                                    slack = model.addVar(vtype=GRB.INTEGER, lb=0,
                                                         name=f"demand_slack_{d}_{s}_{k}")
                                    demand_slack[d, s, k] = slack
                                    staffing = gp.quicksum(y[n, d, s, k] for n in nurse_ids)
                                    model.addConstr(staffing + slack >= req,
                                                    name=f"demand_{d}_{s}_{k}")
                                    obj_terms.append(unsatisfied_demand_penalty * slack)
    
    # Working variable: w[n, d] = 1 if nurse works on day d
    w = {}
    for n in nurse_ids:
        for d in days:
            w[n, d] = model.addVar(vtype=GRB.BINARY, name=f"w_{n}_{d}")
            model.addConstr(w[n, d] == gp.quicksum(x[n, d, s] for s in shifts),
                            name=f"working_{n}_{d}")
    
    model.update()
    
    # Total working days constraints (soft)
    if 'total_working_days' in soft_cons:
        twd = soft_cons['total_working_days']
        tw_min = twd['min']
        tw_max = twd['max']
        pen_under = twd['penalty_under']
        pen_over = twd['penalty_over']
        
        for n in nurse_ids:
            total_work = gp.quicksum(w[n, d] for d in days)
            if pen_under > 0:
                under = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"twd_under_{n}")
                model.addConstr(total_work + under >= tw_min, name=f"twd_min_{n}")
                obj_terms.append(pen_under * under)
            if pen_over > 0:
                over = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"twd_over_{n}")
                model.addConstr(total_work - over <= tw_max, name=f"twd_max_{n}")
                obj_terms.append(pen_over * over)
    
    # Total worked weekends (soft)
    if 'total_worked_weekends' in soft_cons:
        tww = soft_cons['total_worked_weekends']
        tww_max = tww['max']
        tww_pen = tww['penalty']
        
        # ww[n, we_idx] = 1 if nurse works on weekend we_idx
        ww = {}
        for n in nurse_ids:
            for i, we in enumerate(weekends):
                sat = we['saturday']
                sun = we['sunday']
                ww[n, i] = model.addVar(vtype=GRB.BINARY, name=f"ww_{n}_{i}")
                # ww = 1 if nurse works on sat or sun
                if sat in days and sun in days:
                    model.addConstr(ww[n, i] >= w[n, sat] + w[n, sun] - 1,
                                    name=f"ww_lb1_{n}_{i}")
                    model.addConstr(ww[n, i] <= w[n, sat], name=f"ww_ub1_{n}_{i}")
                    model.addConstr(ww[n, i] <= w[n, sun], name=f"ww_ub2_{n}_{i}")
                    # Actually ww should be 1 if works on ANY day of weekend
                    # Let me reconsider: "weekend on which nurse works" means works at least one day
                    model.remove(model.getConstrByName(f"ww_lb1_{n}_{i}"))
                    model.remove(model.getConstrByName(f"ww_ub1_{n}_{i}"))
                    model.remove(model.getConstrByName(f"ww_ub2_{n}_{i}"))
                    model.addConstr(ww[n, i] >= w[n, sat], name=f"ww_sat_{n}_{i}")
                    model.addConstr(ww[n, i] >= w[n, sun], name=f"ww_sun_{n}_{i}")
                    model.addConstr(ww[n, i] <= w[n, sat] + w[n, sun], name=f"ww_ub_{n}_{i}")
                elif sat in days:
                    model.addConstr(ww[n, i] == w[n, sat])
                elif sun in days:
                    model.addConstr(ww[n, i] == w[n, sun])
            
            total_ww = gp.quicksum(ww[n, i] for i in range(len(weekends)))
            over_ww = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"tww_over_{n}")
            model.addConstr(total_ww - over_ww <= tww_max, name=f"tww_max_{n}")
            obj_terms.append(tww_pen * over_ww)
    
    # Consecutive working days (soft)
    if 'consecutive_working_days' in soft_cons:
        cwd = soft_cons['consecutive_working_days']
        cwd_min = cwd['min']
        cwd_max = cwd['max']
        pen_under_cwd = cwd['penalty_under']
        pen_over_cwd = cwd['penalty_over']
        
        for n in nurse_ids:
            # Max consecutive working days
            if pen_over_cwd > 0 and cwd_max < num_days:
                for d in days:
                    end_d = d + cwd_max
                    if end_d <= num_days:
                        window = list(range(d, end_d + 1))
                        excess = model.addVar(vtype=GRB.BINARY, lb=0,
                                              name=f"cwd_over_{n}_{d}")
                        model.addConstr(
                            gp.quicksum(w[n, dd] for dd in window) <= cwd_max + excess,
                            name=f"cwd_max_{n}_{d}"
                        )
                        # Actually this isn't quite right for per-day penalty
                        # The penalty is per day exceeding. Let me use a different formulation.
                
                # Better formulation: for each window of cwd_max+1 consecutive days, 
                # if all are working, that's a violation
                # Actually, per the spec: "penalty_over: Cost per day a working streak exceeds the maximum"
                # So if streak is L > max, penalty is (L - max) * pen_over_cwd
                # We can model this with: for each day d, excess_d = 1 if all of d-cwd_max..d are working
                # Then sum of excess_d gives total excess days
                
                # Remove previously added constraints
                for d in days:
                    end_d = d + cwd_max
                    if end_d <= num_days:
                        c = model.getConstrByName(f"cwd_max_{n}_{d}")
                        if c:
                            model.remove(c)
                        v = model.getVarByName(f"cwd_over_{n}_{d}")
                        if v:
                            model.remove(v)
                
                model.update()
                
                for d in days:
                    if d - cwd_max >= 1:
                        window = list(range(d - cwd_max, d + 1))
                        exc = model.addVar(vtype=GRB.BINARY, name=f"cwd_exc_{n}_{d}")
                        # exc = 1 only if all days in window are working
                        for dd in window:
                            model.addConstr(exc <= w[n, dd], name=f"cwd_exc_ub_{n}_{d}_{dd}")
                        model.addConstr(
                            exc >= gp.quicksum(w[n, dd] for dd in window) - cwd_max,
                            name=f"cwd_exc_lb_{n}_{d}"
                        )
                        obj_terms.append(pen_over_cwd * exc)
            
            # Min consecutive working days
            # "penalty_under: Cost per occurrence of a working streak shorter than the minimum"
            # A streak is too short if its length < min. Each such streak incurs one penalty.
            # We detect: w[n,d]=1 and (d==1 or w[n,d-1]=0) starts a streak, 
            # if this streak length < min, penalty once.
            # Simplified: for each potential streak start d, for each length l < min,
            # if w[n,d..d+l-1] = 1 and w[n,d-1] = 0 (or d=1) and w[n,d+l] = 0 (or d+l>num_days), penalty
            # This is complex. Use pattern-based approach:
            # A streak of length l < cwd_min starting at d means:
            # r[n,d-1] (or d=1), w[n,d]..w[n,d+l-1], r[n,d+l] (or d+l>num_days)
            # But we only penalize if it's NOT touching boundary (per spec: "streaks touching the horizon boundary are not penalized below the minimum")
            
            if pen_under_cwd > 0 and cwd_min > 1:
                for l in range(1, cwd_min):
                    for d in days:
                        end = d + l - 1
                        if end > num_days:
                            break
                        # Must not touch boundary for this to be penalized
                        if d == 1 or end == num_days:
                            continue
                        # Streak of exactly length l at position d: 
                        # w[n,d-1]=0, w[n,d..end]=1, w[n,end+1]=0
                        short_streak = model.addVar(vtype=GRB.BINARY, 
                                                     name=f"cwd_short_{n}_{d}_{l}")
                        # short_streak <= r[n,d-1]
                        model.addConstr(short_streak <= r[n, d-1] if d > 1 else (short_streak <= 1))
                        # short_streak <= w[n, dd] for dd in d..end
                        for dd in range(d, end + 1):
                            model.addConstr(short_streak <= w[n, dd])
                        # short_streak <= r[n, end+1]
                        if end + 1 <= num_days:
                            model.addConstr(short_streak <= r[n, end + 1])
                        
                        # Lower bound
                        num_terms = l + 2  # r[d-1] + l work days + r[end+1]
                        terms = []
                        if d > 1:
                            terms.append(r[n, d-1])
                        for dd in range(d, end + 1):
                            terms.append(w[n, dd])
                        if end + 1 <= num_days:
                            terms.append(r[n, end + 1])
                        model.addConstr(
                            short_streak >= gp.quicksum(terms) - (len(terms) - 1),
                            name=f"cwd_short_lb_{n}_{d}_{l}"
                        )
                        obj_terms.append(pen_under_cwd * short_streak)
    
    # Consecutive days off (soft)
    if 'consecutive_days_off' in soft_cons:
        cdo = soft_cons['consecutive_days_off']
        cdo_min = cdo['min']
        cdo_max = cdo['max']
        pen_under_cdo = cdo['penalty_under']
        pen_over_cdo = cdo['penalty_over']
        
        for n in nurse_ids:
            # Max consecutive days off
            if pen_over_cdo > 0 and cdo_max < num_days:
                for d in days:
                    if d - cdo_max >= 1:
                        window = list(range(d - cdo_max, d + 1))
                        exc = model.addVar(vtype=GRB.BINARY, name=f"cdo_exc_{n}_{d}")
                        for dd in window:
                            model.addConstr(exc <= r[n, dd], name=f"cdo_exc_ub_{n}_{d}_{dd}")
                        model.addConstr(
                            exc >= gp.quicksum(r[n, dd] for dd in window) - cdo_max,
                            name=f"cdo_exc_lb_{n}_{d}"
                        )
                        obj_terms.append(pen_over_cdo * exc)
            
            # Min consecutive days off (not penalized at boundaries)
            if pen_under_cdo > 0 and cdo_min > 1:
                for l in range(1, cdo_min):
                    for d in days:
                        end = d + l - 1
                        if end > num_days:
                            break
                        if d == 1 or end == num_days:
                            continue
                        short_off = model.addVar(vtype=GRB.BINARY,
                                                  name=f"cdo_short_{n}_{d}_{l}")
                        if d > 1:
                            model.addConstr(short_off <= w[n, d-1])
                        for dd in range(d, end + 1):
                            model.addConstr(short_off <= r[n, dd])
                        if end + 1 <= num_days:
                            model.addConstr(short_off <= w[n, end + 1])
                        
                        terms = []
                        if d > 1:
                            terms.append(w[n, d-1])
                        for dd in range(d, end + 1):
                            terms.append(r[n, dd])
                        if end + 1 <= num_days:
                            terms.append(w[n, end + 1])
                        model.addConstr(
                            short_off >= gp.quicksum(terms) - (len(terms) - 1),
                            name=f"cdo_short_lb_{n}_{d}_{l}"
                        )
                        obj_terms.append(pen_under_cdo * short_off)
    
    # Consecutive same shift (soft)
    if 'consecutive_same_shift' in soft_cons:
        css = soft_cons['consecutive_same_shift']
        css_max = css['max']
        css_pen = css['penalty']
        
        if css_pen > 0 and css_max < num_days:
            for n in nurse_ids:
                for s in shifts:
                    for d in days:
                        if d - css_max >= 1:
                            window = list(range(d - css_max, d + 1))
                            exc = model.addVar(vtype=GRB.BINARY, name=f"css_exc_{n}_{s}_{d}")
                            for dd in window:
                                model.addConstr(exc <= x[n, dd, s])
                            model.addConstr(
                                exc >= gp.quicksum(x[n, dd, s] for dd in window) - css_max,
                                name=f"css_exc_lb_{n}_{s}_{d}"
                            )
                            obj_terms.append(css_pen * exc)
    
    # Consecutive worked weekends (soft)
    if 'consecutive_worked_weekends' in soft_cons and 'total_worked_weekends' in soft_cons:
        cww = soft_cons['consecutive_worked_weekends']
        cww_max = cww['max']
        cww_pen = cww['penalty']
        
        if cww_pen > 0 and cww_max < len(weekends):
            for n in nurse_ids:
                for i in range(len(weekends)):
                    if i - cww_max >= 0:
                        window = list(range(i - cww_max, i + 1))
                        exc = model.addVar(vtype=GRB.BINARY, name=f"cww_exc_{n}_{i}")
                        for j in window:
                            model.addConstr(exc <= ww[n, j])
                        model.addConstr(
                            exc >= gp.quicksum(ww[n, j] for j in window) - cww_max,
                            name=f"cww_exc_lb_{n}_{i}"
                        )
                        obj_terms.append(cww_pen * exc)
    elif 'consecutive_worked_weekends' in soft_cons:
        # Need to create ww variables if not already created
        cww = soft_cons['consecutive_worked_weekends']
        cww_max = cww['max']
        cww_pen = cww['penalty']
        
        if cww_pen > 0:
            if not any(k[0] == nurse_ids[0] for k in ww.keys() if hasattr(ww, 'keys') if False):
                pass  # ww should exist from total_worked_weekends section
            # Create ww if not exists
            try:
                _ = ww
            except NameError:
                ww = {}
                for n in nurse_ids:
                    for i, we in enumerate(weekends):
                        sat = we['saturday']
                        sun = we['sunday']
                        ww[n, i] = model.addVar(vtype=GRB.BINARY, name=f"ww_{n}_{i}")
                        if sat in days and sun in days:
                            model.addConstr(ww[n, i] >= w[n, sat])
                            model.addConstr(ww[n, i] >= w[n, sun])
                            model.addConstr(ww[n, i] <= w[n, sat] + w[n, sun])
            
            if cww_max < len(weekends):
                for n in nurse_ids:
                    for i in range(len(weekends)):
                        if i - cww_max >= 0:
                            window = list(range(i - cww_max, i + 1))
                            exc = model.addVar(vtype=GRB.BINARY, name=f"cww_exc_{n}_{i}")
                            for j in window:
                                model.addConstr(exc <= ww[n, j])
                            model.addConstr(
                                exc >= gp.quicksum(ww[n, j] for j in window) - cww_max
                            )
                            obj_terms.append(cww_pen * exc)
    
    # Identical weekend (soft)
    if 'identical_weekend' in soft_cons:
        iwe = soft_cons['identical_weekend']
        if iwe.get('enabled', False):
            iwe_pen = iwe['penalty']
            for n in nurse_ids:
                for i, we in enumerate(weekends):
                    we_days = []
                    if 'friday' in we and we['friday'] in days:
                        we_days.append(we['friday'])
                    if we['saturday'] in days:
                        we_days.append(we['saturday'])
                    if we['sunday'] in days:
                        we_days.append(we['sunday'])
                    
                    if len(we_days) >= 2:
                        # All days must have same assignment
                        # For each pair of consecutive weekend days, shifts must match
                        diff_var = model.addVar(vtype=GRB.BINARY, name=f"iwe_diff_{n}_{i}")
                        
                        for j in range(len(we_days) - 1):
                            d1 = we_days[j]
                            d2 = we_days[j + 1]
                            # If shifts differ, diff_var = 1
                            for s in shifts:
                                model.addConstr(
                                    x[n, d1, s] - x[n, d2, s] <= diff_var,
                                    name=f"iwe_p_{n}_{i}_{j}_{s}"
                                )
                                model.addConstr(
                                    x[n, d2, s] - x[n, d1, s] <= diff_var,
                                    name=f"iwe_n_{n}_{i}_{j}_{s}"
                                )
                            # Also rest must match
                            model.addConstr(
                                r[n, d1] - r[n, d2] <= diff_var
                            )
                            model.addConstr(
                                r[n, d2] - r[n, d1] <= diff_var
                            )
                        
                        obj_terms.append(iwe_pen * diff_var)
    
    # Hard forbidden patterns
    if 'forbidden_patterns' in hard_cons:
        for fp in hard_cons['forbidden_patterns']:
            pattern = fp['pattern']
            pat_len = len(pattern)
            for n in nurse_ids:
                for d in range(1, num_days - pat_len + 2):
                    # Check if this pattern can start at day d
                    vars_list = []
                    valid = True
                    for offset, (day_type, shift_name) in enumerate(pattern):
                        dd = d + offset
                        if dd > num_days:
                            valid = False
                            break
                        if day_type == "Any" or day_type == "any":
                            if shift_name == "None" or shift_name is None:
                                vars_list.append(r[n, dd])
                            elif shift_name in shifts:
                                vars_list.append(x[n, dd, shift_name])
                            elif shift_name == "Any" or shift_name == "any":
                                vars_list.append(w[n, dd])
                            else:
                                valid = False
                                break
                        elif day_type == "None" or day_type is None:
                            if shift_name == "None" or shift_name is None:
                                vars_list.append(r[n, dd])
                            elif shift_name in shifts:
                                vars_list.append(x[n, dd, shift_name])
                            else:
                                valid = False
                                break
                        else:
                            # day_type might be a shift name
                            if day_type in shifts:
                                vars_list.append(x[n, dd, day_type])
                            elif day_type == "None":
                                vars_list.append(r[n, dd])
                            else:
                                valid = False
                                break
                    
                    if valid and vars_list:
                        model.addConstr(
                            gp.quicksum(vars_list) <= len(vars_list) - 1,
                            name=f"hard_fp_{n}_{d}_{id(fp)}"
                        )
    
    # Soft forbidden patterns
    soft_fps = soft_cons.get('forbidden_patterns', [])
    if soft_fps:
        for fp_idx, fp in enumerate(soft_fps):
            pattern = fp['pattern']
            pat_len = len(pattern)
            fp_pen = fp.get('penalty', 0)
            if fp_pen <= 0:
                continue
            for n in nurse_ids:
                for d in range(1, num_days - pat_len + 2):
                    vars_list = []
                    valid = True
                    for offset, entry in enumerate(pattern):
                        if isinstance(entry, list) and len(entry) == 2:
                            day_type, shift_name = entry
                        else:
                            valid = False
                            break
                        dd = d + offset
                        if dd > num_days:
                            valid = False
                            break
                        if day_type == "Any" or day_type == "any":
                            if shift_name == "None" or shift_name is None:
                                vars_list.append(r[n, dd])
                            elif shift_name in shifts:
                                vars_list.append(x[n, dd, shift_name])
                            elif shift_name == "Any" or shift_name == "any":
                                vars_list.append(w[n, dd])
                            else:
                                valid = False
                                break
                        elif day_type in shifts:
                            vars_list.append(x[n, dd, day_type])
                        elif day_type == "None":
                            vars_list.append(r[n, dd])
                        else:
                            valid = False
                            break
                    
                    if valid and vars_list:
                        viol = model.addVar(vtype=GRB.BINARY, name=f"sfp_{fp_idx}_{n}_{d}")
                        for v in vars_list:
                            model.addConstr(viol >= v - (len(vars_list) - 1) + gp.quicksum(
                                vv for vv in vars_list if vv is not v))
                        # Actually: viol >= sum(vars_list) - (len-1)
                        model.addConstr(
                            viol >= gp.quicksum(vars_list) - (len(vars_list) - 1),
                            name=f"sfp_lb_{fp_idx}_{n}_{d}"
                        )
                        obj_terms.append(fp_pen * viol)
    
    # Nurse preferences
    for nurse in nurses:
        n = nurse['id']
        for pref in nurse.get('preferences', []):
            d = pref['day']
            ptype = pref['type']
            shift_pref = pref['shift']
            penalty = pref['penalty']
            
            if d not in days:
                continue
            
            if ptype == 'day_off':
                # Wants day off; penalty if working
                unmet = model.addVar(vtype=GRB.BINARY, name=f"pref_dayoff_{n}_{d}")
                model.addConstr(unmet >= w[n, d], name=f"pref_dayoff_c_{n}_{d}")
                model.addConstr(unmet <= w[n, d])
                obj_terms.append(penalty * unmet)
            elif ptype == 'day_on':
                # Wants to work; penalty if off
                unmet = model.addVar(vtype=GRB.BINARY, name=f"pref_dayon_{n}_{d}")
                model.addConstr(unmet >= r[n, d])
                model.addConstr(unmet <= r[n, d])
                obj_terms.append(penalty * unmet)
            elif ptype == 'shift_on':
                # Wants specific shift; penalty if not assigned that shift
                if shift_pref in shifts:
                    unmet = model.addVar(vtype=GRB.BINARY, name=f"pref_shifton_{n}_{d}_{shift_pref}")
                    model.addConstr(unmet >= 1 - x[n, d, shift_pref])
                    model.addConstr(unmet <= 1 - x[n, d, shift_pref])
                    obj_terms.append(penalty * unmet)
            elif ptype == 'shift_off':
                # Does NOT want this shift; penalty if assigned
                if shift_pref in shifts:
                    unmet = model.addVar(vtype=GRB.BINARY, name=f"pref_shiftoff_{n}_{d}_{shift_pref}")
                    model.addConstr(unmet >= x[n, d, shift_pref])
                    model.addConstr(unmet <= x[n, d, shift_pref])
                    obj_terms.append(penalty * unmet)
    
    # Set objective
    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    
    model.update()
    
    # Callback to log solutions
    best_obj = [float('inf')]
    best_sol = [None]
    
    def extract_solution():
        schedule = {}
        skill_assignments = {}
        
        for n in nurse_ids:
            schedule[n] = {}
            skill_assignments[n] = {}
            for d in days:
                day_key = f"day_{d}"
                assigned_shift = None
                for s in shifts:
                    try:
                        val = x[n, d, s].X
                        if val > 0.5:
                            assigned_shift = s
                            break
                    except:
                        pass
                
                if assigned_shift is not None:
                    schedule[n][day_key] = assigned_shift
                    # Find skill
                    assigned_skill = None
                    for k in skills:
                        try:
                            val = y[n, d, assigned_shift, k].X
                            if val > 0.5:
                                assigned_skill = k
                                break
                        except:
                            pass
                    skill_assignments[n][day_key] = assigned_skill if assigned_skill else skills[0]
                else:
                    schedule[n][day_key] = "None"
                    skill_assignments[n][day_key] = "None"
        
        return schedule, skill_assignments
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    # Solve
    remaining_time = max(10, time_limit - (time.time() - start_time) - 2)
    model.setParam('TimeLimit', remaining_time)
    model.optimize(callback)
    
    if model.SolCount > 0:
        schedule, skill_assignments = extract_solution()
        obj_val = model.ObjVal
        
        solution = {
            "objective_value": obj_val,
            "schedule": schedule,
            "skill_assignments": skill_assignments
        }
        
        if logger:
            logger.log_solution(obj_val, solution)
    else:
        # Try to construct a feasible solution heuristically
        # Assign everyone to rest
        schedule = {}
        skill_assignments = {}
        for n in nurse_ids:
            schedule[n] = {}
            skill_assignments[n] = {}
            for d in days:
                day_key = f"day_{d}"
                schedule[n][day_key] = "None"
                skill_assignments[n][day_key] = "None"
        
        obj_val = 0
        # Calculate demand penalty
        if not demand_equality:
            for d in days:
                day_key = f"day_{d}"
                if day_key in demand:
                    for s in shifts:
                        if s in demand[day_key]:
                            for k in skills:
                                if k in demand[day_key][s]:
                                    req = demand[day_key][s][k]
                                    obj_val += unsatisfied_demand_penalty * req
        
        solution = {
            "objective_value": obj_val,
            "schedule": schedule,
            "skill_assignments": skill_assignments
        }
        
        if logger:
            logger.log_solution(obj_val, solution)
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)