import json
import argparse
import sys
import time as time_module

def main():
    parser = argparse.ArgumentParser(description='Nurse Scheduling Optimizer')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    # Parse instance
    shifts = inst['shifts']
    skills = inst['skills']
    nurses_data = inst['nurses']
    num_nurses = len(nurses_data)
    nurse_ids = [n['id'] for n in nurses_data]
    nurse_skill_sets = {n['id']: set(n.get('skills', [])) for n in nurses_data}
    nurse_preferences = {n['id']: n.get('preferences', []) for n in nurses_data}

    weekends = inst.get('weekends', [])
    num_weekends = len(weekends)
    weekend_definition = inst.get('weekend_definition', 'saturday_sunday')

    # Determine number of days
    num_days = 0
    demand_data = inst.get('demand', {})
    for key in demand_data:
        d = int(key.split('_')[1])
        if d > num_days:
            num_days = d
    for w in weekends:
        for k in ['saturday', 'sunday', 'friday']:
            if k in w and w[k] is not None and w[k] > num_days:
                num_days = w[k]

    if num_days == 0:
        num_days = 1

    days = list(range(1, num_days + 1))
    all_shifts = shifts + ['None']

    # Weekend day mapping
    weekend_days_list = []
    for w in weekends:
        wd = []
        if 'friday' in w and w['friday'] is not None:
            wd.append(w['friday'])
        if 'saturday' in w and w['saturday'] is not None:
            wd.append(w['saturday'])
        if 'sunday' in w and w['sunday'] is not None:
            wd.append(w['sunday'])
        weekend_days_list.append(wd)

    # All weekend days set
    all_weekend_days = set()
    for wd in weekend_days_list:
        all_weekend_days.update(wd)

    # Constraints
    constraints = inst.get('constraints', {})
    hard = constraints.get('hard_constraints', {})
    soft = constraints.get('soft_constraints', {})

    skill_req_hard = hard.get('skill_requirement', True)
    demand_equality = hard.get('demand_equality', False)
    hard_forbidden = hard.get('forbidden_patterns', [])

    unsat_demand_penalty = inst.get('unsatisfied_demand_penalty', 0)
    skill_penalty = soft.get('skill_requirement_penalty', 0)

    # Build Gurobi model
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model('nurse_scheduling')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, args.time_limit - 5))
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}
    for n in nurse_ids:
        for d in days:
            for s in all_shifts:
                x[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f'x_{n}_{d}_{s}')

    # Skill assignment variables
    y = {}
    for n in nurse_ids:
        n_skills = nurse_skill_sets[n]
        for d in days:
            for s in shifts:
                for k in skills:
                    if skill_req_hard and k not in n_skills:
                        continue
                    y[n, d, s, k] = model.addVar(vtype=GRB.BINARY, name=f'y_{n}_{d}_{s}_{k}')

    model.update()

    # Objective expression
    obj = gp.LinExpr()

    # Constraint: each nurse gets exactly one shift per day
    for n in nurse_ids:
        for d in days:
            model.addConstr(gp.quicksum(x[n, d, s] for s in all_shifts) == 1, name=f'assign_{n}_{d}')

    # Constraint: if working a shift, exactly one skill
    for n in nurse_ids:
        for d in days:
            for s in shifts:
                valid_ks = [k for k in skills if (n, d, s, k) in y]
                if valid_ks:
                    model.addConstr(
                        gp.quicksum(y[n, d, s, k] for k in valid_ks) == x[n, d, s],
                        name=f'skill_link_{n}_{d}_{s}'
                    )
                else:
                    model.addConstr(x[n, d, s] == 0, name=f'no_skill_{n}_{d}_{s}')

    # Demand constraints
    slack_vars = {}
    if not demand_equality:
        for d in days:
            day_key = f'day_{d}'
            if day_key in demand_data:
                for s in shifts:
                    if s in demand_data[day_key]:
                        for k in skills:
                            if k in demand_data[day_key][s] and demand_data[day_key][s][k] > 0:
                                slack_vars[d, s, k] = model.addVar(
                                    vtype=GRB.CONTINUOUS, lb=0,
                                    name=f'slack_{d}_{s}_{k}'
                                )

    model.update()

    for d in days:
        day_key = f'day_{d}'
        if day_key in demand_data:
            for s in shifts:
                if s in demand_data[day_key]:
                    for k in skills:
                        if k in demand_data[day_key][s]:
                            dem = demand_data[day_key][s][k]
                            if dem > 0:
                                staffing = gp.quicksum(
                                    y[n, d, s, k] for n in nurse_ids if (n, d, s, k) in y
                                )
                                if demand_equality:
                                    model.addConstr(staffing == dem, name=f'demand_eq_{d}_{s}_{k}')
                                else:
                                    sv = slack_vars.get((d, s, k))
                                    if sv is not None:
                                        model.addConstr(staffing + sv >= dem, name=f'demand_{d}_{s}_{k}')
                                        obj += unsat_demand_penalty * sv
                                    else:
                                        model.addConstr(staffing >= dem, name=f'demand_{d}_{s}_{k}')

    # Working indicator: w[n,d] = 1 - x[n,d,'None']
    # We'll use expressions directly

    # === SOFT CONSTRAINT: Total working days ===
    twd = soft.get('total_working_days', None)
    if twd:
        twd_min = twd.get('min', 0)
        twd_max = twd.get('max', num_days)
        pen_under = twd.get('penalty_under', 0)
        pen_over = twd.get('penalty_over', 0)
        if pen_under > 0 or pen_over > 0:
            for n in nurse_ids:
                work_days_n = gp.quicksum(1 - x[n, d, 'None'] for d in days)
                if pen_under > 0 and twd_min > 0:
                    under_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'twd_under_{n}')
                    model.addConstr(under_var >= twd_min - work_days_n)
                    obj += pen_under * under_var
                if pen_over > 0 and twd_max < num_days:
                    over_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'twd_over_{n}')
                    model.addConstr(over_var >= work_days_n - twd_max)
                    obj += pen_over * over_var

    # === SOFT CONSTRAINT: Total worked weekends ===
    tww = soft.get('total_worked_weekends', None)
    if tww and num_weekends > 0:
        tww_max = tww.get('max', num_weekends)
        tww_pen = tww.get('penalty', 0)
        if tww_pen > 0:
            ww = {}
            for n in nurse_ids:
                for wi in range(num_weekends):
                    ww[n, wi] = model.addVar(vtype=GRB.BINARY, name=f'ww_{n}_{wi}')
                    for d in weekend_days_list[wi]:
                        if d <= num_days:
                            model.addConstr(ww[n, wi] >= 1 - x[n, d, 'None'], name=f'ww_lb_{n}_{wi}_{d}')
                total_ww = gp.quicksum(ww[n, wi] for wi in range(num_weekends))
                over_ww = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'ww_over_{n}')
                model.addConstr(over_ww >= total_ww - tww_max)
                obj += tww_pen * over_ww

    # === SOFT CONSTRAINT: Consecutive working days ===
    cwd = soft.get('consecutive_working_days', None)
    if cwd:
        cwd_min = cwd.get('min', 0)
        cwd_max = cwd.get('max', num_days)
        pen_under = cwd.get('penalty_under', 0)
        pen_over = cwd.get('penalty_over', 0)

        if pen_over > 0 and cwd_max < num_days:
            for n in nurse_ids:
                for d in range(cwd_max + 1, num_days + 1):
                    window_sum = gp.quicksum(1 - x[n, dd, 'None'] for dd in range(d - cwd_max, d + 1))
                    over_v = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'cwd_over_{n}_{d}')
                    model.addConstr(over_v >= window_sum - cwd_max)
                    obj += pen_over * over_v

        if pen_under > 0 and cwd_min > 1:
            for n in nurse_ids:
                for d in range(2, num_days - cwd_min + 2):
                    if d + cwd_min - 1 <= num_days:
                        short_v = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'cwd_under_{n}_{d}')
                        w_d = 1 - x[n, d, 'None']
                        w_prev = 1 - x[n, d - 1, 'None'] if d > 1 else 0
                        w_end = 1 - x[n, d + cwd_min - 1, 'None']
                        model.addConstr(short_v >= w_d - w_prev - w_end)
                        obj += pen_under * short_v

    # === SOFT CONSTRAINT: Consecutive days off ===
    cdo = soft.get('consecutive_days_off', None)
    if cdo:
        cdo_min = cdo.get('min', 0)
        cdo_max = cdo.get('max', num_days)
        pen_under = cdo.get('penalty_under', 0)
        pen_over = cdo.get('penalty_over', 0)

        if pen_over > 0 and cdo_max < num_days:
            for n in nurse_ids:
                for d in range(cdo_max + 1, num_days + 1):
                    window_sum = gp.quicksum(x[n, dd, 'None'] for dd in range(d - cdo_max, d + 1))
                    over_v = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'cdo_over_{n}_{d}')
                    model.addConstr(over_v >= window_sum - cdo_max)
                    obj += pen_over * over_v

        if pen_under > 0 and cdo_min > 1:
            for n in nurse_ids:
                for d in range(2, num_days - cdo_min + 2):
                    if d + cdo_min - 1 <= num_days:
                        short_v = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'cdo_under_{n}_{d}')
                        r_d = x[n, d, 'None']
                        r_prev = x[n, d - 1, 'None'] if d > 1 else 0
                        r_end = x[n, d + cdo_min - 1, 'None']
                        model.addConstr(short_v >= r_d - r_prev - r_end)
                        obj += pen_under * short_v

    # === SOFT CONSTRAINT: Consecutive same shift ===
    css = soft.get('consecutive_same_shift', None)
    if css:
        css_max = css.get('max', num_days)
        css_pen = css.get('penalty', 0)
        if css_pen > 0 and css_max < num_days:
            for n in nurse_ids:
                for s in shifts:
                    for d in range(css_max + 1, num_days + 1):
                        window_sum = gp.quicksum(x[n, dd, s] for dd in range(d - css_max, d + 1))
                        over_v = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'css_over_{n}_{s}_{d}')
                        model.addConstr(over_v >= window_sum - css_max)
                        obj += css_pen * over_v

    # === SOFT CONSTRAINT: Consecutive worked weekends ===
    cww = soft.get('consecutive_worked_weekends', None)
    if cww and num_weekends > 0:
        cww_max = cww.get('max', num_weekends)
        cww_pen = cww.get('penalty', 0)
        if cww_pen > 0 and cww_max < num_weekends:
            # ww variables should already exist from total_worked_weekends
            # If not, create them
            ww_exists = tww is not None and tww.get('penalty', 0) > 0
            if not ww_exists:
                ww = {}
                for n in nurse_ids:
                    for wi in range(num_weekends):
                        ww[n, wi] = model.addVar(vtype=GRB.BINARY, name=f'ww2_{n}_{wi}')
                        for d in weekend_days_list[wi]:
                            if d <= num_days:
                                model.addConstr(ww[n, wi] >= 1 - x[n, d, 'None'])
                model.update()

            for n in nurse_ids:
                for j in range(cww_max + 1, num_weekends + 1):
                    window_sum = gp.quicksum(ww[n, wi] for wi in range(j - cww_max, j + 1))
                    over_v = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'cww_over_{n}_{j}')
                    model.addConstr(over_v >= window_sum - cww_max)
                    obj += cww_pen * over_v

    # === SOFT CONSTRAINT: Identical weekend ===
    iw = soft.get('identical_weekend', None)
    if iw and iw.get('enabled', False) and num_weekends > 0:
        iw_pen = iw.get('penalty', 0)
        if iw_pen > 0:
            for n in nurse_ids:
                for wi in range(num_weekends):
                    wd = weekend_days_list[wi]
                    if len(wd) < 2:
                        continue
                    viol = model.addVar(vtype=GRB.BINARY, name=f'iw_viol_{n}_{wi}')
                    for i in range(len(wd)):
                        for j in range(i + 1, len(wd)):
                            d1, d2 = wd[i], wd[j]
                            if d1 <= num_days and d2 <= num_days:
                                for s in all_shifts:
                                    model.addConstr(x[n, d1, s] - x[n, d2, s] <= viol)
                                    model.addConstr(x[n, d2, s] - x[n, d1, s] <= viol)
                    obj += iw_pen * viol

    # === HARD CONSTRAINT: Forbidden patterns ===
    for fp in hard_forbidden:
        pattern = fp.get('pattern', [])
        if not pattern:
            continue
        pat_shifts = [p[1] if isinstance(p, list) and len(p) >= 2 else p for p in pattern]
        pat_len = len(pat_shifts)
        for n in nurse_ids:
            for start in range(1, num_days - pat_len + 2):
                # The pattern must NOT appear starting at 'start'
                # sum of indicators that each day matches <= pat_len - 1
                match_expr = gp.LinExpr()
                valid = True
                for i, ps in enumerate(pat_shifts):
                    d = start + i
                    if d > num_days:
                        valid = False
                        break
                    if ps == 'None':
                        match_expr += x[n, d, 'None']
                    elif ps in shifts:
                        match_expr += x[n, d, ps]
                    else:
                        valid = False
                        break
                if valid:
                    model.addConstr(match_expr <= pat_len - 1, name=f'fp_{n}_{start}')

    # === SOFT CONSTRAINT: Soft forbidden patterns ===
    soft_fp = soft.get('forbidden_patterns', [])
    if soft_fp:
        for fp in soft_fp:
            pattern = fp.get('pattern', [])
            fp_pen = fp.get('penalty', 0)
            if not pattern or fp_pen <= 0:
                continue
            pat_shifts = [p[1] if isinstance(p, list) and len(p) >= 2 else p for p in pattern]
            pat_len = len(pat_shifts)
            for n in nurse_ids:
                for start in range(1, num_days - pat_len + 2):
                    match_vars = []
                    valid = True
                    for i, ps in enumerate(pat_shifts):
                        d = start + i
                        if d > num_days:
                            valid = False
                            break
                        if ps == 'None':
                            match_vars.append(x[n, d, 'None'])
                        elif ps in shifts:
                            match_vars.append(x[n, d, ps])
                        else:
                            valid = False
                            break
                    if valid:
                        occ = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'sfp_{n}_{start}')
                        model.addConstr(occ >= gp.quicksum(match_vars) - pat_len + 1)
                        obj += fp_pen * occ

    # === PREFERENCES ===
    prefs_enabled = soft.get('preferences_penalty', True)
    if prefs_enabled:
        for n in nurse_ids:
            for pref in nurse_preferences[n]:
                d = pref.get('day', 0)
                if d < 1 or d > num_days:
                    continue
                ptype = pref.get('type', '')
                pshift = pref.get('shift', 'None')
                pen = pref.get('penalty', 0)
                if pen <= 0:
                    continue

                viol = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'pref_{n}_{d}_{ptype}')
                if ptype == 'day_off':
                    # Penalty if nurse works on day d
                    model.addConstr(viol >= 1 - x[n, d, 'None'])
                elif ptype == 'day_on':
                    # Penalty if nurse is off on day d
                    model.addConstr(viol >= x[n, d, 'None'])
                elif ptype == 'shift_on':
                    # Penalty if nurse doesn't work the specified shift
                    if pshift in all_shifts:
                        model.addConstr(viol >= 1 - x[n, d, pshift])
                    else:
                        model.addConstr(viol >= 1)
                elif ptype == 'shift_off':
                    # Penalty if nurse works the specified shift
                    if pshift in all_shifts:
                        model.addConstr(viol >= x[n, d, pshift])
                    else:
                        continue
                obj += pen * viol

    # === SKILL MISMATCH PENALTY (soft skill requirement) ===
    if not skill_req_hard and skill_penalty > 0:
        for n in nurse_ids:
            n_skills = nurse_skill_sets[n]
            for d in days:
                for s in shifts:
                    for k in skills:
                        if k not in n_skills and (n, d, s, k) in y:
                            obj += skill_penalty * y[n, d, s, k]

    # Set objective
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()

    # Callback for logging incumbents
    best_obj = [float('inf')]

    def my_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            cb_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if cb_obj < best_obj[0]:
                best_obj[0] = cb_obj
                if logger:
                    try:
                        sol_dict = extract_solution(cb_model, x, y, nurse_ids, days, shifts, all_shifts)
                        logger.log_solution(cb_obj, sol_dict)
                    except:
                        logger.log(cb_obj)

    def extract_solution(mdl, x_vars, y_vars, n_ids, ds, shfts, all_shfts):
        schedule = {}
        skill_assignments = {}
        for n in n_ids:
            schedule[n] = {}
            skill_assignments[n] = {}
            for d in ds:
                assigned_shift = 'None'
                for s in all_shfts:
                    if x_vars[n, d, s].X > 0.5:
                        assigned_shift = s
                        break
                schedule[n][f'day_{d}'] = assigned_shift

                if assigned_shift != 'None' and assigned_shift in shfts:
                    assigned_skill = 'None'
                    best_val = -1
                    for k in skills:
                        if (n, d, assigned_shift, k) in y_vars:
                            val = y_vars[n, d, assigned_shift, k].X
                            if val > best_val:
                                best_val = val
                                assigned_skill = k
                    skill_assignments[n][f'day_{d}'] = assigned_skill
                else:
                    skill_assignments[n][f'day_{d}'] = 'None'

        obj_val = mdl.ObjVal
        return {
            "objective_value": obj_val,
            "schedule": schedule,
            "skill_assignments": skill_assignments
        }

    # Optimize
    model.optimize(my_callback)

    # Extract final solution
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.INTERRUPTED:
        if model.SolCount > 0:
            solution = extract_solution(model, x, y, nurse_ids, days, shifts, all_shifts)
            obj_val = model.ObjVal
            solution['objective_value'] = obj_val
        else:
            # Fallback: assign everyone to rest
            solution = build_fallback(nurse_ids, days, num_days)
            obj_val = solution['objective_value']
    else:
        solution = build_fallback(nurse_ids, days, num_days)
        obj_val = solution['objective_value']

    if logger:
        logger.log_solution(solution['objective_value'], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def build_fallback(nurse_ids, days, num_days):
    schedule = {}
    skill_assignments = {}
    for n in nurse_ids:
        schedule[n] = {}
        skill_assignments[n] = {}
        for d in days:
            schedule[n][f'day_{d}'] = 'None'
            skill_assignments[n][f'day_{d}'] = 'None'
    return {
        "objective_value": 0.0,
        "schedule": schedule,
        "skill_assignments": skill_assignments
    }


if __name__ == '__main__':
    main()