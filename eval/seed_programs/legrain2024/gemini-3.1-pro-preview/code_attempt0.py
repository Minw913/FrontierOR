import json
import argparse
import sys
import os
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def get_dow(d, weekends):
    sat_day = None
    for w in weekends:
        if 'saturday' in w:
            sat_day = w['saturday']
            break
    if sat_day is not None:
        dow_idx = (5 + (d - sat_day)) % 7
    else:
        dow_idx = (d - 1) % 7
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[dow_idx]

def get_vals(var_dict, model):
    if not var_dict:
        return {}
    keys = list(var_dict.keys())
    vars_list = list(var_dict.values())
    vals = model.cbGetSolution(vars_list)
    return dict(zip(keys, vals))

def main():
    parser = argparse.ArgumentParser(description="Nurse Scheduling Optimization")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    shifts = data.get('shifts', [])
    skills = data.get('skills', [])
    nurses = data.get('nurses', [])
    weekends = data.get('weekends', [])
    
    num_days = 0
    for d_str in data.get('demand', {}).keys():
        d = int(d_str.split('_')[1])
        if d > num_days: num_days = d
    for w in weekends:
        for k, v in w.items():
            if k != 'week' and v > num_days:
                num_days = v
    for n in nurses:
        for p in n.get('preferences', []):
            if p['day'] > num_days:
                num_days = p['day']
                
    days = list(range(1, num_days + 1))
    
    constraints = data.get('constraints', {})
    hard_constraints = constraints.get('hard_constraints', {})
    soft_constraints = constraints.get('soft_constraints', {})

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("NurseScheduling", env=env)
    
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    w = {}
    x = {}
    z = {}
    
    for n in nurses:
        nid = n['id']
        for d in days:
            w[nid, d] = model.addVar(vtype=GRB.BINARY, name=f"w_{nid}_{d}")
            for s in shifts:
                x[nid, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{nid}_{d}_{s}")
                for k in skills:
                    z[nid, d, s, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{nid}_{d}_{s}_{k}")
                    
    for n in nurses:
        nid = n['id']
        for d in days:
            model.addConstr(sum(x[nid, d, s] for s in shifts) == w[nid, d])
            for s in shifts:
                model.addConstr(sum(z[nid, d, s, k] for k in skills) == x[nid, d, s])
                
    obj_terms = []
    
    # Demand
    demand_eq = hard_constraints.get('demand_equality', False)
    unsat_pen = data.get('unsatisfied_demand_penalty', 0)
    
    for d_str, shifts_dict in data.get('demand', {}).items():
        d = int(d_str.split('_')[1])
        for s, skills_dict in shifts_dict.items():
            for k, req in skills_dict.items():
                staffing = sum(z[n['id'], d, s, k] for n in nurses)
                if demand_eq:
                    model.addConstr(staffing == req)
                else:
                    if unsat_pen > 0:
                        slack = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                        model.addConstr(staffing + slack >= req)
                        obj_terms.append(slack * unsat_pen)
                    else:
                        model.addConstr(staffing >= req)
                        
    # Skill requirement
    hard_skill = hard_constraints.get('skill_requirement', False)
    pen_skill = soft_constraints.get('skill_requirement_penalty', 0)
    
    for n in nurses:
        nid = n['id']
        n_skills = set(n.get('skills', []))
        for d in days:
            for s in shifts:
                for k in skills:
                    if k not in n_skills:
                        if hard_skill:
                            model.addConstr(z[nid, d, s, k] == 0)
                        elif pen_skill > 0:
                            obj_terms.append(z[nid, d, s, k] * pen_skill)

    # Preferences
    pref_enabled = soft_constraints.get('preferences_penalty', False)
    is_pref_enabled = pref_enabled if isinstance(pref_enabled, bool) else (pref_enabled > 0)
    
    if is_pref_enabled:
        for n in nurses:
            nid = n['id']
            for p in n.get('preferences', []):
                d = p['day']
                t = p['type']
                s_pref = p.get('shift', 'None')
                pen = p.get('penalty', 0)
                if pen > 0:
                    var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                    if t == 'day_off':
                        model.addConstr(var >= w[nid, d])
                    elif t == 'day_on':
                        model.addConstr(var >= 1 - w[nid, d])
                    elif t == 'shift_off':
                        if s_pref == 'None':
                            model.addConstr(var >= 1 - w[nid, d])
                        else:
                            model.addConstr(var >= x[nid, d, s_pref])
                    elif t == 'shift_on':
                        if s_pref == 'None':
                            model.addConstr(var >= w[nid, d])
                        else:
                            model.addConstr(var >= 1 - x[nid, d, s_pref])
                    obj_terms.append(var * pen)

    # Total Working Days
    twd = soft_constraints.get('total_working_days', {})
    if twd:
        min_days = twd.get('min', 0)
        max_days = twd.get('max', num_days)
        pen_under = twd.get('penalty_under', 0)
        pen_over = twd.get('penalty_over', 0)
        for n in nurses:
            nid = n['id']
            total_w = sum(w[nid, d] for d in days)
            if pen_under > 0:
                under = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                model.addConstr(under >= min_days - total_w)
                obj_terms.append(under * pen_under)
            if pen_over > 0:
                over = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                model.addConstr(over >= total_w - max_days)
                obj_terms.append(over * pen_over)

    # Worked Weekends
    tww = soft_constraints.get('total_worked_weekends', {})
    max_weekends = tww.get('max', len(weekends))
    pen_weekends = tww.get('penalty', 0)
    
    cww = soft_constraints.get('consecutive_worked_weekends', {})
    cw_max = cww.get('max', len(weekends))
    cw_pen = cww.get('penalty', 0)
    
    iw = soft_constraints.get('identical_weekend', {})
    iw_en = iw.get('enabled', False)
    iw_pen = iw.get('penalty', 0)

    for n in nurses:
        nid = n['id']
        worked_weekends_count = []
        for wknd in weekends:
            e_days = sorted([v for k, v in wknd.items() if k != 'week'])
            we = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
            for d in e_days:
                if d <= num_days:
                    model.addConstr(we >= w[nid, d])
            worked_weekends_count.append(we)
            
            if iw_en and iw_pen > 0:
                niw = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
                for i in range(len(e_days)-1):
                    d1, d2 = e_days[i], e_days[i+1]
                    if d1 <= num_days and d2 <= num_days:
                        model.addConstr(niw >= w[nid, d1] - w[nid, d2])
                        model.addConstr(niw >= w[nid, d2] - w[nid, d1])
                        for s in shifts:
                            model.addConstr(niw >= x[nid, d1, s] - x[nid, d2, s])
                            model.addConstr(niw >= x[nid, d2, s] - x[nid, d1, s])
                obj_terms.append(niw * iw_pen)
                
        if pen_weekends > 0 and len(worked_weekends_count) > 0:
            over_w = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
            model.addConstr(over_w >= sum(worked_weekends_count) - max_weekends)
            obj_terms.append(over_w * pen_weekends)
            
        if cw_pen > 0 and len(worked_weekends_count) > cw_max:
            for i in range(len(worked_weekends_count) - cw_max):
                over_cw = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                model.addConstr(over_cw >= sum(worked_weekends_count[i:i+cw_max+1]) - cw_max)
                obj_terms.append(over_cw * cw_pen)

    # Consecutive Working Days
    cwd = soft_constraints.get('consecutive_working_days', {})
    if cwd:
        min_cwd = cwd.get('min', 1)
        max_cwd = cwd.get('max', num_days)
        pen_cwd_under = cwd.get('penalty_under', 0)
        pen_cwd_over = cwd.get('penalty_over', 0)
        for n in nurses:
            nid = n['id']
            if pen_cwd_over > 0:
                for i in range(1, num_days - max_cwd + 1):
                    over_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                    model.addConstr(over_var >= sum(w[nid, d] for d in range(i, i + max_cwd + 1)) - max_cwd)
                    obj_terms.append(over_var * pen_cwd_over)
            if pen_cwd_under > 0 and min_cwd > 1:
                for L in range(1, min_cwd):
                    for i in range(2, num_days - L + 1):
                        j = i + L - 1
                        if j < num_days:
                            under_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                            model.addConstr(under_var >= (1 - w[nid, i-1]) + (1 - w[nid, j+1]) + sum(w[nid, d] for d in range(i, j+1)) - L - 1)
                            obj_terms.append(under_var * pen_cwd_under)

    # Consecutive Days Off
    cdo = soft_constraints.get('consecutive_days_off', {})
    if cdo:
        min_cdo = cdo.get('min', 1)
        max_cdo = cdo.get('max', num_days)
        pen_cdo_under = cdo.get('penalty_under', 0)
        pen_cdo_over = cdo.get('penalty_over', 0)
        for n in nurses:
            nid = n['id']
            if pen_cdo_over > 0:
                for i in range(1, num_days - max_cdo + 1):
                    over_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                    model.addConstr(over_var >= 1 - sum(w[nid, d] for d in range(i, i + max_cdo + 1)))
                    obj_terms.append(over_var * pen_cdo_over)
            if pen_cdo_under > 0 and min_cdo > 1:
                for L in range(1, min_cdo):
                    for i in range(2, num_days - L + 1):
                        j = i + L - 1
                        if j < num_days:
                            under_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                            model.addConstr(under_var >= w[nid, i-1] + w[nid, j+1] + sum(1 - w[nid, d] for d in range(i, j+1)) - L - 1)
                            obj_terms.append(under_var * pen_cdo_under)

    # Consecutive Same Shift
    css = soft_constraints.get('consecutive_same_shift', {})
    if css:
        max_css = css.get('max', num_days)
        pen_css = css.get('penalty', 0)
        if pen_css > 0:
            for n in nurses:
                nid = n['id']
                for s in shifts:
                    for i in range(1, num_days - max_css + 1):
                        over_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                        model.addConstr(over_var >= sum(x[nid, d, s] for d in range(i, i + max_css + 1)) - max_css)
                        obj_terms.append(over_var * pen_css)

    # Forbidden Patterns
    def apply_forbidden_pattern(n_id, start_d, pattern, is_hard, penalty=0):
        L = len(pattern)
        if start_d + L - 1 > num_days:
            return
        exprs = []
        for k in range(L):
            d = start_d + k
            day_type, s_pat = pattern[k]
            if day_type != "Any" and day_type != get_dow(d, weekends):
                return
            if s_pat == "None":
                exprs.append(1 - w[n_id, d])
            elif s_pat == "Any":
                exprs.append(w[n_id, d])
            elif s_pat in shifts:
                exprs.append(x[n_id, d, s_pat])
            else:
                return
        if is_hard:
            model.addConstr(sum(exprs) <= L - 1)
        else:
            if penalty > 0:
                fp_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
                model.addConstr(fp_var >= sum(exprs) - (L - 1))
                obj_terms.append(fp_var * penalty)

    for p_obj in hard_constraints.get('forbidden_patterns', []):
        pat = p_obj['pattern']
        for n in nurses:
            for st in days:
                apply_forbidden_pattern(n['id'], st, pat, is_hard=True)
                
    for p_obj in soft_constraints.get('forbidden_patterns', []):
        pat = p_obj['pattern']
        pen = p_obj.get('penalty', 0)
        for n in nurses:
            for st in days:
                apply_forbidden_pattern(n['id'], st, pat, is_hard=False, penalty=pen)

    if obj_terms:
        model.setObjective(sum(obj_terms), GRB.MINIMIZE)
    else:
        model.setObjective(0, GRB.MINIMIZE)

    model._w = w
    model._x = x
    model._z = z
    model._nurses = nurses
    model._num_days = num_days
    model._shifts = shifts
    model._skills = skills

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                w_vals = get_vals(model._w, model)
                x_vals = get_vals(model._x, model)
                z_vals = get_vals(model._z, model)
                
                schedule = {}
                skill_assignments = {}
                
                for n in model._nurses:
                    nid = n['id']
                    schedule[nid] = {}
                    skill_assignments[nid] = {}
                    for d in range(1, model._num_days + 1):
                        assigned_shift = 'None'
                        assigned_skill = 'None'
                        if w_vals.get((nid, d), 0) > 0.5:
                            for s in model._shifts:
                                if x_vals.get((nid, d, s), 0) > 0.5:
                                    assigned_shift = s
                                    for k in model._skills:
                                        if z_vals.get((nid, d, s, k), 0) > 0.5:
                                            assigned_skill = k
                                            break
                                    break
                        schedule[nid][f"day_{d}"] = assigned_shift
                        skill_assignments[nid][f"day_{d}"] = assigned_skill
                
                sol_dict = {
                    "objective_value": obj,
                    "schedule": schedule,
                    "skill_assignments": skill_assignments
                }
                if logger:
                    logger.log_solution(obj, sol_dict)
            except Exception as e:
                if logger:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj)

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        schedule = {}
        skill_assignments = {}
        for n in nurses:
            nid = n['id']
            schedule[nid] = {}
            skill_assignments[nid] = {}
            for d in days:
                assigned_shift = 'None'
                assigned_skill = 'None'
                if w[nid, d].X > 0.5:
                    for s in shifts:
                        if x[nid, d, s].X > 0.5:
                            assigned_shift = s
                            for k in skills:
                                if z[nid, d, s, k].X > 0.5:
                                    assigned_skill = k
                                    break
                            break
                schedule[nid][f"day_{d}"] = assigned_shift
                skill_assignments[nid][f"day_{d}"] = assigned_skill

        final_solution = {
            "objective_value": obj_val,
            "schedule": schedule,
            "skill_assignments": skill_assignments
        }
    else:
        final_solution = {
            "objective_value": float('inf'),
            "schedule": {},
            "skill_assignments": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()