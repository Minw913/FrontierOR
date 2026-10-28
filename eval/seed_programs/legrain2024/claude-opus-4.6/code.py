import argparse
import json
import time
import sys
import os


def solve_heuristic(data, time_limit_sec, logger, start_time):
    """Greedy/heuristic solver as fallback."""
    
    shifts = data['shifts']
    skills = data['skills']
    nurses = data['nurses']
    weekends = data['weekends']
    demand = data['demand']
    unsatisfied_demand_penalty = data['unsatisfied_demand_penalty']
    constraints = data['constraints']
    hard_cons = constraints['hard_constraints']
    soft_cons = constraints['soft_constraints']
    demand_equality = hard_cons.get('demand_equality', False)
    
    nurse_ids = [n['id'] for n in nurses]
    nurse_skills_map = {n['id']: set(n['skills']) for n in nurses}
    nurse_prefs_map = {n['id']: n.get('preferences', []) for n in nurses}
    
    all_days = set()
    for key in demand.keys():
        d = int(key.split('_')[1])
        all_days.add(d)
    for we in weekends:
        all_days.add(we['saturday'])
        all_days.add(we['sunday'])
        if 'friday' in we:
            all_days.add(we['friday'])
    for nurse in nurses:
        for pref in nurse.get('preferences', []):
            all_days.add(pref['day'])
    num_days = max(all_days) if all_days else 0
    days = list(range(1, num_days + 1))
    
    demand_struct = {}
    for d in days:
        day_key = f"day_{d}"
        if day_key in demand:
            for s in shifts:
                if s in demand[day_key]:
                    for k in skills:
                        if k in demand[day_key][s]:
                            req = demand[day_key][s][k]
                            if req > 0:
                                demand_struct[(d, s, k)] = req

    schedule = {}
    skill_assign = {}
    for n in nurse_ids:
        schedule[n] = {}
        skill_assign[n] = {}
        for d in days:
            schedule[n][d] = "None"
            skill_assign[n][d] = "None"
    
    skill_req_hard = hard_cons.get('skill_requirement', False)
    
    demand_items = sorted(demand_struct.items(), key=lambda x: -x[1])
    
    for (d, s, k), req in demand_items:
        assigned = 0
        candidates_with_skill = []
        candidates_without_skill = []
        for n in nurse_ids:
            if schedule[n][d] == "None":
                if k in nurse_skills_map[n]:
                    candidates_with_skill.append(n)
                else:
                    candidates_without_skill.append(n)
        
        for n in candidates_with_skill:
            if assigned >= req:
                break
            schedule[n][d] = s
            skill_assign[n][d] = k
            assigned += 1
        
        if not skill_req_hard and assigned < req:
            for n in candidates_without_skill:
                if assigned >= req:
                    break
                schedule[n][d] = s
                skill_assign[n][d] = k
                assigned += 1
    
    obj_val = compute_objective(data, schedule, skill_assign, days, nurse_ids, shifts, skills,
                                 nurse_skills_map, nurse_prefs_map, weekends, demand_struct,
                                 unsatisfied_demand_penalty, hard_cons, soft_cons, num_days)
    
    schedule_out = {}
    skill_out = {}
    for n in nurse_ids:
        schedule_out[n] = {}
        skill_out[n] = {}
        for d in days:
            day_key = f"day_{d}"
            schedule_out[n][day_key] = schedule[n][d]
            skill_out[n][day_key] = skill_assign[n][d]
    
    solution = {
        "objective_value": obj_val,
        "schedule": schedule_out,
        "skill_assignments": skill_out
    }
    
    if logger:
        logger.log_solution(obj_val, solution)
    
    return solution


def compute_objective(data, schedule, skill_assign, days, nurse_ids, shifts, skills,
                      nurse_skills_map, nurse_prefs_map, weekends, demand_struct,
                      unsatisfied_demand_penalty, hard_cons, soft_cons, num_days):
    total_cost = 0.0
    demand_equality = hard_cons.get('demand_equality', False)
    
    staffing = {}
    for n in nurse_ids:
        for d in days:
            s = schedule[n][d]
            k = skill_assign[n][d]
            if s != "None" and k != "None":
                key = (d, s, k)
                staffing[key] = staffing.get(key, 0) + 1
    
    if not demand_equality:
        for (d, s, k), req in demand_struct.items():
            actual = staffing.get((d, s, k), 0)
            if actual < req:
                total_cost += unsatisfied_demand_penalty * (req - actual)
    
    skill_req_hard = hard_cons.get('skill_requirement', False)
    skill_req_penalty = soft_cons.get('skill_requirement_penalty', 0)
    if not skill_req_hard and skill_req_penalty:
        for n in nurse_ids:
            for d in days:
                s = schedule[n][d]
                k = skill_assign[n][d]
                if s != "None" and k != "None":
                    if k not in nurse_skills_map[n]:
                        total_cost += skill_req_penalty
    
    for n in nurse_ids:
        for pref in nurse_prefs_map.get(n, []):
            d = pref['day']
            ptype = pref['type']
            shift_pref = pref.get('shift', 'None')
            penalty = pref['penalty']
            if d not in schedule[n] or penalty <= 0:
                continue
            
            s = schedule[n].get(d, "None")
            if ptype == 'day_off':
                if s != "None":
                    total_cost += penalty
            elif ptype == 'day_on':
                if s == "None":
                    total_cost += penalty
            elif ptype == 'shift_on':
                if s != shift_pref:
                    total_cost += penalty
            elif ptype == 'shift_off':
                if s == shift_pref:
                    total_cost += penalty
    
    if 'total_working_days' in soft_cons:
        twd = soft_cons['total_working_days']
        tw_min = twd['min']
        tw_max = twd['max']
        pen_under = twd['penalty_under']
        pen_over = twd['penalty_over']
        for n in nurse_ids:
            work_count = sum(1 for d in days if schedule[n][d] != "None")
            if work_count < tw_min:
                total_cost += pen_under * (tw_min - work_count)
            if work_count > tw_max:
                total_cost += pen_over * (work_count - tw_max)
    
    if 'total_worked_weekends' in soft_cons:
        tww = soft_cons['total_worked_weekends']
        tww_max = tww['max']
        tww_pen = tww['penalty']
        for n in nurse_ids:
            ww_count = 0
            for we in weekends:
                sat = we['saturday']
                sun = we['sunday']
                if (schedule[n].get(sat, "None") != "None") or \
                   (schedule[n].get(sun, "None") != "None"):
                    ww_count += 1
            if ww_count > tww_max:
                total_cost += tww_pen * (ww_count - tww_max)
    
    if 'consecutive_working_days' in soft_cons:
        cwd = soft_cons['consecutive_working_days']
        cwd_max = cwd['max']
        pen_over_cwd = cwd['penalty_over']
        cwd_min = cwd['min']
        pen_under_cwd = cwd['penalty_under']
        
        for n in nurse_ids:
            streak = 0
            streak_start = None
            for d in list(range(1, num_days + 1)) + [num_days + 1]:
                if d <= num_days and schedule[n].get(d, "None") != "None":
                    if streak == 0:
                        streak_start = d
                    streak += 1
                else:
                    if streak > 0:
                        if streak > cwd_max:
                            total_cost += pen_over_cwd * (streak - cwd_max)
                        if streak < cwd_min and streak_start > 1 and d <= num_days:
                            total_cost += pen_under_cwd
                    streak = 0
    
    if 'consecutive_days_off' in soft_cons:
        cdo = soft_cons['consecutive_days_off']
        cdo_max = cdo['max']
        pen_over_cdo = cdo['penalty_over']
        cdo_min = cdo['min']
        pen_under_cdo = cdo['penalty_under']
        
        for n in nurse_ids:
            streak = 0
            streak_start = None
            for d in list(range(1, num_days + 1)) + [num_days + 1]:
                if d <= num_days and schedule[n].get(d, "None") == "None":
                    if streak == 0:
                        streak_start = d
                    streak += 1
                else:
                    if streak > 0:
                        if streak > cdo_max:
                            total_cost += pen_over_cdo * (streak - cdo_max)
                        if streak < cdo_min and streak_start > 1 and d <= num_days:
                            total_cost += pen_under_cdo
                    streak = 0
    
    if 'consecutive_same_shift' in soft_cons:
        css = soft_cons['consecutive_same_shift']
        css_max = css['max']
        css_pen = css['penalty']
        
        for n in nurse_ids:
            for s in shifts:
                streak = 0
                for d in list(range(1, num_days + 1)) + [num_days + 1]:
                    if d <= num_days and schedule[n].get(d, "None") == s:
                        streak += 1
                    else:
                        if streak > css_max:
                            total_cost += css_pen * (streak - css_max)
                        streak = 0
    
    if 'consecutive_worked_weekends' in soft_cons:
        cww = soft_cons['consecutive_worked_weekends']
        cww_max = cww['max']
        cww_pen = cww['penalty']
        
        for n in nurse_ids:
            ww_list = []
            for we in weekends:
                sat = we['saturday']
                sun = we['sunday']
                worked = (schedule[n].get(sat, "None") != "None") or \
                         (schedule[n].get(sun, "None") != "None")
                ww_list.append(worked)
            
            streak = 0
            for worked in ww_list + [False]:
                if worked:
                    streak += 1
                else:
                    if streak > cww_max:
                        total_cost += cww_pen * (streak - cww_max)
                    streak = 0
    
    if 'identical_weekend' in soft_cons:
        iwe = soft_cons['identical_weekend']
        if iwe.get('enabled', False):
            iwe_pen = iwe['penalty']
            for n in nurse_ids:
                for we in weekends:
                    we_days_list = []
                    if 'friday' in we and we['friday'] in schedule[n]:
                        we_days_list.append(we['friday'])
                    if we['saturday'] in schedule[n]:
                        we_days_list.append(we['saturday'])
                    if we['sunday'] in schedule[n]:
                        we_days_list.append(we['sunday'])
                    
                    if len(we_days_list) >= 2:
                        shifts_on_we = [schedule[n].get(dd, "None") for dd in we_days_list]
                        if len(set(shifts_on_we)) > 1:
                            total_cost += iwe_pen
    
    return total_cost


def solve_with_gurobi(data, time_limit_sec, logger, start_time):
    """MIP solver using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    
    shifts = data['shifts']
    skills = data['skills']
    nurses = data['nurses']
    weekends = data['weekends']
    demand = data['demand']
    unsatisfied_demand_penalty = data['unsatisfied_demand_penalty']
    constraints = data['constraints']
    hard_cons = constraints['hard_constraints']
    soft_cons = constraints['soft_constraints']
    demand_equality = hard_cons.get('demand_equality', False)
    
    nurse_ids = [n['id'] for n in nurses]
    nurse_skills_map = {n['id']: set(n['skills']) for n in nurses}
    nurse_prefs_map = {n['id']: n.get('preferences', []) for n in nurses}
    
    all_days = set()
    for key in demand.keys():
        d = int(key.split('_')[1])
        all_days.add(d)
    for we in weekends:
        all_days.add(we['saturday'])
        all_days.add(we['sunday'])
        if 'friday' in we:
            all_days.add(we['friday'])
    for nurse in nurses:
        for pref in nurse.get('preferences', []):
            all_days.add(pref['day'])
    num_days = max(all_days) if all_days else 0
    days = list(range(1, num_days + 1))
    days_set = set(days)
    
    demand_struct = {}
    for d in days:
        day_key = f"day_{d}"
        if day_key in demand:
            for s in shifts:
                if s in demand[day_key]:
                    for k in skills:
                        if k in demand[day_key][s]:
                            req = demand[day_key][s][k]
                            if req > 0:
                                demand_struct[(d, s, k)] = req

    n_nurses = len(nurse_ids)
    n_shifts = len(shifts)
    n_skills = len(skills)
    n_days = len(days)
    
    print(f"Problem size: {n_nurses} nurses, {n_days} days, {n_shifts} shifts, {n_skills} skills", file=sys.stderr)

    skill_req_hard = hard_cons.get('skill_requirement', False)

    model = gp.Model("nurse_scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 1)
    model.setParam('LogToConsole', 1)
    remaining_time = max(10, time_limit_sec - (time.time() - start_time) - 5)
    model.setParam('TimeLimit', remaining_time)
    
    # Decision variables
    # x[n,d,s] = 1 if nurse n works shift s on day d
    x = {}
    for n in nurse_ids:
        for d in days:
            for s in shifts:
                x[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")
    
    # r[n,d] = 1 if nurse n rests on day d
    r = {}
    for n in nurse_ids:
        for d in days:
            r[n, d] = model.addVar(vtype=GRB.BINARY, name=f"r_{n}_{d}")
    
    # y[n,d,s,k] = 1 if nurse n works shift s with skill k on day d
    y = {}
    for n in nurse_ids:
        for d in days:
            for s in shifts:
                for k in skills:
                    if skill_req_hard and k not in nurse_skills_map[n]:
                        continue
                    y[n, d, s, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{n}_{d}_{s}_{k}")
    
    # w[n,d] = 1 if nurse n works on day d
    w = {}
    for n in nurse_ids:
        for d in days:
            w[n, d] = model.addVar(vtype=GRB.BINARY, name=f"w_{n}_{d}")
    
    model.update()
    
    # Single assignment per day
    for n in nurse_ids:
        for d in days:
            model.addConstr(
                gp.quicksum(x[n, d, s] for s in shifts) + r[n, d] == 1
            )
    
    # Link w to x
    for n in nurse_ids:
        for d in days:
            model.addConstr(w[n, d] == gp.quicksum(x[n, d, s] for s in shifts))
    
    # Skill assignment
    for n in nurse_ids:
        for d in days:
            for s in shifts:
                valid_skills = [k for k in skills if (n, d, s, k) in y]
                if valid_skills:
                    model.addConstr(
                        gp.quicksum(y[n, d, s, k] for k in valid_skills) == x[n, d, s]
                    )
                else:
                    if skill_req_hard:
                        model.addConstr(x[n, d, s] == 0)
    
    obj_terms = []
    
    # Skill mismatch penalty (soft)
    skill_req_penalty = soft_cons.get('skill_requirement_penalty', 0)
    if not skill_req_hard and skill_req_penalty and skill_req_penalty > 0:
        for n in nurse_ids:
            for d in days:
                for s in shifts:
                    for k in skills:
                        if k not in nurse_skills_map[n] and (n, d, s, k) in y:
                            obj_terms.append(skill_req_penalty * y[n, d, s, k])
    
    # Demand
    if demand_equality:
        for (d, s, k), req in demand_struct.items():
            staffing_expr = gp.quicksum(y[n, d, s, k] for n in nurse_ids if (n, d, s, k) in y)
            model.addConstr(staffing_expr == req)
    else:
        for (d, s, k), req in demand_struct.items():
            slack = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"dslack_{d}_{s}_{k}")
            staffing_expr = gp.quicksum(y[n, d, s, k] for n in nurse_ids if (n, d, s, k) in y)
            model.addConstr(staffing_expr + slack >= req)
            obj_terms.append(unsatisfied_demand_penalty * slack)
    
    # Weekend working variables
    ww = {}
    for n in nurse_ids:
        for i, we in enumerate(weekends):
            sat = we['saturday']
            sun = we['sunday']
            ww[n, i] = model.addVar(vtype=GRB.BINARY, name=f"ww_{n}_{i}")
            we_days_in_range = []
            if sat in days_set:
                we_days_in_range.append(sat)
            if sun in days_set:
                we_days_in_range.append(sun)
            
            if we_days_in_range:
                for dd in we_days_in_range:
                    model.addConstr(ww[n, i] >= w[n, dd])
                model.addConstr(ww[n, i] <= gp.quicksum(w[n, dd] for dd in we_days_in_range))
            else:
                model.addConstr(ww[n, i] == 0)
    
    model.update()
    
    # Total working days (soft)
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
                model.addConstr(total_work + under >= tw_min)
                obj_terms.append(pen_under * under)
            if pen_over > 0:
                over = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"twd_over_{n}")
                model.addConstr(total_work - over <= tw_max)
                obj_terms.append(pen_over * over)
    
    # Total worked weekends (soft)
    if 'total_worked_weekends' in soft_cons:
        tww_cons = soft_cons['total_worked_weekends']
        tww_max = tww_cons['max']
        tww_pen = tww_cons['penalty']
        
        for n in nurse_ids:
            total_ww = gp.quicksum(ww[n, i] for i in range(len(weekends)))
            over_ww = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"tww_over_{n}")
            model.addConstr(total_ww - over_ww <= tww_max)
            obj_terms.append(tww_pen * over_ww)
    
    # Consecutive working days (soft)
    if 'consecutive_working_days' in soft_cons:
        cwd = soft_cons['consecutive_working_days']
        cwd_min = cwd['min']
        cwd_max = cwd['max']
        pen_under_cwd = cwd['penalty_under']
        pen_over_cwd = cwd['penalty_over']
        
        for n in nurse_ids:
            # Over max
            if pen_over_cwd > 0 and cwd_max < num_days:
                window_size = cwd_max + 1
                for d_start in range(1, num_days - window_size + 2):
                    window = list(range(d_start, d_start + window_size))
                    exc = model.addVar(vtype=GRB.BINARY, name=f"cwd_exc_{n}_{d_start}")
                    model.addConstr(exc >= gp.quicksum(w[n, dd] for dd in window) - cwd_max)
                    model.addConstr(exc <= 1)
                    obj_terms.append(pen_over_cwd * exc)
            
            # Under min
            if pen_under_cwd > 0 and cwd_min > 1:
                for l in range(1, cwd_min):
                    for d_start in range(1, num_days + 1):
                        d_end = d_start + l - 1
                        if d_end > num_days:
                            break
                        if d_start == 1 or d_end == num_days:
                            continue
                        
                        short_streak = model.addVar(vtype=GRB.BINARY, name=f"cwd_short_{n}_{d_start}_{l}")
                        all_terms = [r[n, d_start - 1]] + [w[n, dd] for dd in range(d_start, d_end + 1)] + [r[n, d_end + 1]]
                        n_terms = len(all_terms)
                        for t in all_terms:
                            model.addConstr(short_streak <= t)
                        model.addConstr(short_streak >= gp.quicksum(all_terms) - (n_terms - 1))
                        obj_terms.append(pen_under_cwd * short_streak)
    
    # Consecutive days off (soft)
    if 'consecutive_days_off' in soft_cons:
        cdo = soft_cons['consecutive_days_off']
        cdo_min = cdo['min']
        cdo_max = cdo['max']
        pen_under_cdo = cdo['penalty_under']
        pen_over_cdo = cdo['penalty_over']
        
        for n in nurse_ids:
            if pen_over_cdo > 0 and cdo_max < num_days:
                window_size = cdo_max + 1
                for d_start in range(1, num_days - window_size + 2):
                    window = list(range(d_start, d_start + window_size))
                    exc = model.addVar(vtype=GRB.BINARY, name=f"cdo_exc_{n}_{d_start}")
                    model.addConstr(exc >= gp.quicksum(r[n, dd] for dd in window) - cdo_max)
                    model.addConstr(exc <= 1)
                    obj_terms.append(pen_over_cdo * exc)
            
            if pen_under_cdo > 0 and cdo_min > 1:
                for l in range(1, cdo_min):
                    for d_start in range(1, num_days + 1):
                        d_end = d_start + l - 1
                        if d_end > num_days:
                            break
                        if d_start == 1 or d_end == num_days:
                            continue
                        
                        short_off = model.addVar(vtype=GRB.BINARY, name=f"cdo_short_{n}_{d_start}_{l}")
                        all_terms = [w[n, d_start - 1]] + [r[n, dd] for dd in range(d_start, d_end + 1)] + [w[n, d_end + 1]]
                        for t in all_terms:
                            model.addConstr(short_off <= t)
                        model.addConstr(short_off >= gp.quicksum(all_terms) - (len(all_terms) - 1))
                        obj_terms.append(pen_under_cdo * short_off)
    
    # Consecutive same shift (soft)
    if 'consecutive_same_shift' in soft_cons:
        css = soft_cons['consecutive_same_shift']
        css_max = css['max']
        css_pen = css['penalty']
        
        if css_pen > 0 and css_max < num_days:
            for n in nurse_ids:
                for s in shifts:
                    window_size = css_max + 1
                    for d_start in range(1, num_days - window_size + 2):
                        window = list(range(d_start, d_start + window_size))
                        exc = model.addVar(vtype=GRB.BINARY, name=f"css_exc_{n}_{s}_{d_start}")
                        model.addConstr(exc >= gp.quicksum(x[n, dd, s] for dd in window) - css_max)
                        model.addConstr(exc <= 1)
                        obj_terms.append(css_pen * exc)
    
    # Consecutive worked weekends (soft)
    if 'consecutive_worked_weekends' in soft_cons:
        cww = soft_cons['consecutive_worked_weekends']
        cww_max = cww['max']
        cww_pen = cww['penalty']
        
        if cww_pen > 0 and cww_max < len(weekends):
            for n in nurse_ids:
                window_size = cww_max + 1
                for i_start in range(0, len(weekends) - window_size + 1):
                    window = list(range(i_start, i_start + window_size))
                    exc = model.addVar(vtype=GRB.BINARY, name=f"cww_exc_{n}_{i_start}")
                    model.addConstr(exc >= gp.quicksum(ww[n, j] for j in window) - cww_max)
                    model.addConstr(exc <= 1)
                    obj_terms.append(cww_pen * exc)
    
    # Identical weekend (soft)
    if 'identical_weekend' in soft_cons:
        iwe = soft_cons['identical_weekend']
        if iwe.get('enabled', False):
            iwe_pen = iwe['penalty']
            for n in nurse_ids:
                for i, we in enumerate(weekends):
                    we_days_list = []
                    if 'friday' in we and we['friday'] in days_set:
                        we_days_list.append(we['friday'])
                    if we['saturday'] in days_set:
                        we_days_list.append(we['saturday'])
                    if we['sunday'] in days_set:
                        we_days_list.append(we['sunday'])
                    
                    if len(we_days_list) >= 2:
                        diff_var = model.addVar(vtype=GRB.BINARY, name=f"iwe_diff_{n}_{i}")
                        for j in range(len(we_days_list) - 1):
                            d1 = we_days_list[j]
                            d2 = we_days_list[j + 1]
                            for s in shifts:
                                model.addConstr(x[n, d1, s] - x[n, d2, s] <= diff_var)
                                model.addConstr(x[n, d2, s] - x[n, d1, s] <= diff_var)
                            model.addConstr(r[n, d1] - r[n, d2] <= diff_var)
                            model.addConstr(r[n, d2] - r[n, d1] <= diff_var)
                        obj_terms.append(iwe_pen * diff_var)
    
    # Forbidden patterns helper
    def get_pattern_match_var(n, d, day_type, shift_name):
        if d < 1 or d > num_days:
            return None
        
        dt = day_type if day_type else ""
        sn = shift_name if shift_name else ""
        
        if sn in shifts:
            return x[n, d, sn]
        elif sn == "None" or sn == "" or sn is None:
            if dt in shifts:
                return x[n, d, dt]
            elif dt == "None" or dt == "":
                return r[n, d]
            elif dt.lower() == "any":
                return w[n, d]
            else:
                return None
        elif sn.lower() == "any":
            if dt == "None" or dt == "":
                return r[n, d]
            elif dt.lower() == "any":
                return None
            elif dt in shifts:
                return x[n, d, dt]
            else:
                return w[n, d]
        else:
            return None
    
    # Hard forbidden patterns
    if 'forbidden_patterns' in hard_cons:
        for fp_idx, fp in enumerate(hard_cons['forbidden_patterns']):
            pattern = fp['pattern']
            pat_len = len(pattern)
            for n in nurse_ids:
                for d in range(1, num_days - pat_len + 2):
                    vars_list = []
                    valid = True
                    for offset in range(pat_len):
                        entry = pattern[offset]
                        if isinstance(entry, list) and len(entry) == 2:
                            day_type, shift_name = entry
                        else:
                            valid = False
                            break
                        dd = d + offset
                        v = get_pattern_match_var(n, dd, day_type, shift_name)
                        if v is None:
                            continue
                        vars_list.append(v)
                    
                    if valid and vars_list:
                        model.addConstr(gp.quicksum(vars_list) <= len(vars_list) - 1)
    
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
                    for offset in range(pat_len):
                        entry = pattern[offset]
                        if isinstance(entry, list) and len(entry) == 2:
                            day_type, shift_name = entry
                        else:
                            valid = False
                            break
                        dd = d + offset
                        v = get_pattern_match_var(n, dd, day_type, shift_name)
                        if v is None:
                            continue
                        vars_list.append(v)
                    
                    if valid and vars_list:
                        viol = model.addVar(vtype=GRB.BINARY, name=f"sfp_{fp_idx}_{n}_{d}")
                        model.addConstr(viol >= gp.quicksum(vars_list) - (len(vars_list) - 1))
                        obj_terms.append(fp_pen * viol)
    
    # Nurse preferences
    for nurse in nurses:
        n = nurse['id']
        for pidx, pref in enumerate(nurse.get('preferences', [])):
            d = pref['day']
            ptype = pref['type']
            shift_pref = pref.get('shift', 'None')
            penalty = pref['penalty']
            
            if d < 1 or d > num_days or penalty <= 0:
                continue
            
            if ptype == 'day_off':
                obj_terms.append(penalty * w[n, d])
            elif ptype == 'day_on':
                obj_terms.append(penalty * r[n, d])
            elif ptype == 'shift_on':
                if shift_pref in shifts:
                    unmet = model.addVar(vtype=GRB.BINARY, name=f"pref_son_{n}_{d}_{pidx}")
                    model.addConstr(unmet >= 1 - x[n, d, shift_pref])
                    obj_terms.append(penalty * unmet)
            elif ptype == 'shift_off':
                if shift_pref in shifts:
                    obj_terms.append(penalty * x[n, d, shift_pref])
    
    if obj_terms:
        model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    else:
        model.setObjective(0, GRB.MINIMIZE)
    model.update()
    
    best_obj = [float('inf')]
    
    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                if logger:
                    logger.log(obj)
    
    remaining_time = max(5, time_limit_sec - (time.time() - start_time) - 2)
    model.setParam('TimeLimit', remaining_time)
    model.optimize(callback)
    
    if model.SolCount > 0:
        schedule_out = {}
        skill_out = {}
        for nn in nurse_ids:
            schedule_out[nn] = {}
            skill_out[nn] = {}
            for dd in days:
                day_key = f"day_{dd}"
                assigned_shift = None
                for ss in shifts:
                    try:
                        val = x[nn, dd, ss].X
                        if val > 0.5:
                            assigned_shift = ss
                            break
                    except:
                        pass
                
                if assigned_shift is not None:
                    schedule_out[nn][day_key] = assigned_shift
                    assigned_skill = None
                    for kk in skills:
                        if (nn, dd, assigned_shift, kk) in y:
                            try:
                                val = y[nn, dd, assigned_shift, kk].X
                                if val > 0.5:
                                    assigned_skill = kk
                                    break
                            except:
                                pass
                    skill_out[nn][day_key] = assigned_skill if assigned_skill else (skills[0] if skills else "None")
                else:
                    schedule_out[nn][day_key] = "None"
                    skill_out[nn][day_key] = "None"
        
        obj_val = model.ObjVal
        solution = {
            "objective_value": obj_val,
            "schedule": schedule_out,
            "skill_assignments": skill_out
        }
        
        if logger:
            logger.log_solution(obj_val, solution)
        
        model.dispose()
        return solution
    else:
        model.dispose()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    
    solution = None
    
    # Try Gurobi first
    try:
        solution = solve_with_gurobi(data, args.time_limit, logger, start_time)
    except Exception as e:
        print(f"Gurobi failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        solution = None
    
    # Fallback to heuristic
    if solution is None:
        print("Falling back to heuristic solver", file=sys.stderr)
        remaining = max(10, args.time_limit - (time.time() - start_time))
        solution = solve_heuristic(data, remaining, logger, start_time)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()