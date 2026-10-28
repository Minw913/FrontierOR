import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    shifts = instance['shifts']
    skills = instance['skills']
    nurses = instance['nurses']
    nurse_ids = [n['id'] for n in nurses]
    weekends = instance['weekends']
    demand = instance['demand']
    unsatisfied_demand_penalty = instance['unsatisfied_demand_penalty']
    constraints = instance['constraints']
    hard = constraints.get('hard_constraints', {})
    soft = constraints.get('soft_constraints', {})

    # Preprocess
    num_days = max(int(k.split('_')[1]) for k in demand.keys())
    days = list(range(1, num_days + 1))
    work_shifts = shifts
    all_shifts = work_shifts + ['None']
    skill_dict = {n['id']: set(n['skills']) for n in nurses}
    preferences = {n['id']: n['preferences'] for n in nurses}

    demand_dict = {}
    for d_str, shift_dict in demand.items():
        d = int(d_str.split('_')[1])
        for s, skill_d in shift_dict.items():
            for sk, val in skill_d.items():
                demand_dict[(d, s, sk)] = val

    model = gp.Model("NurseScheduling")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)

    # Decision variables
    x = {(n, d, s): model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")
         for n in nurse_ids for d in days for s in all_shifts}

    y = {(n, d, s, sk): model.addVar(vtype=GRB.BINARY, name=f"y_{n}_{d}_{s}_{sk}")
         for n in nurse_ids for d in days for s in work_shifts for sk in skills}

    slack = {(d, s, sk): model.addVar(vtype=GRB.INTEGER, lb=0, name=f"slack_{d}_{s}_{sk}")
             for d, s, sk in demand_dict.keys()}

    # Objective accumulator
    obj = 0.0

    # 1. Exactly one shift per day (hard)
    if hard.get('single_assignment_per_day', True):
        for n in nurse_ids:
            for d in days:
                model.addConstr(gp.quicksum(x[(n, d, s)] for s in all_shifts) == 1)

    # 2. Link y to x
    for n in nurse_ids:
        for d in days:
            for s in work_shifts:
                model.addConstr(gp.quicksum(y[(n, d, s, sk)] for sk in skills) == x[(n, d, s)])

    # 3. Demand
    demand_equality = hard.get('demand_equality', False)
    for (d, s, sk), req in demand_dict.items():
        staffing = gp.quicksum(y[(n, d, s, sk)] for n in nurse_ids)
        if demand_equality:
            model.addConstr(staffing == req)
        else:
            model.addConstr(staffing + slack[(d, s, sk)] >= req)
            obj += unsatisfied_demand_penalty * slack[(d, s, sk)]

    # 4. Skill requirement
    skill_hard = hard.get('skill_requirement', True)
    skill_penalty = soft.get('skill_requirement_penalty', 0)
    for n in nurse_ids:
        n_skills = skill_dict[n]
        for d in days:
            for s in work_shifts:
                for sk in skills:
                    if sk not in n_skills:
                        if skill_hard:
                            model.addConstr(y[(n, d, s, sk)] == 0)
                        else:
                            obj += skill_penalty * y[(n, d, s, sk)]

    # 5. Total working days (soft)
    tw = soft.get('total_working_days', {})
    if tw:
        min_work = tw.get('min', 0)
        max_work = tw.get('max', len(days))
        p_under = tw.get('penalty_under', 0)
        p_over = tw.get('penalty_over', 0)
        for n in nurse_ids:
            work_days = gp.quicksum(x[(n, d, s)] for d in days for s in work_shifts)
            under = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"under_work_{n}")
            over = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"over_work_{n}")
            model.addConstr(work_days + under >= min_work)
            model.addConstr(work_days <= max_work + over)
            obj += p_under * under + p_over * over

    # 6. Total worked weekends (soft) - FIXED
    tw_wk = soft.get('total_worked_weekends', {})
    if tw_wk and weekends:
        max_wk = tw_wk.get('max', 0)
        p_wk = tw_wk.get('penalty', 0)
        for n in nurse_ids:
            weekend_vars = []
            for w in weekends:
                sat = w['saturday']
                sun = w['sunday']
                works_sat = gp.quicksum(x[(n, sat, s)] for s in work_shifts)
                works_sun = gp.quicksum(x[(n, sun, s)] for s in work_shifts)
                works_weekend = model.addVar(vtype=GRB.BINARY, name=f"wk_{n}_{w['week']}")
                model.addConstr(works_weekend >= works_sat)
                model.addConstr(works_weekend >= works_sun)
                weekend_vars.append(works_weekend)
            over_wk = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"over_wk_{n}")
            model.addConstr(gp.quicksum(weekend_vars) <= max_wk + over_wk)
            obj += p_wk * over_wk

    # 7. Preferences (soft)
    pref_weight = soft.get('preferences_penalty', 0)
    if pref_weight > 0:
        for n in nurse_ids:
            for p in preferences.get(n, []):
                d = p['day']
                ptype = p['type']
                shift = p.get('shift')
                pen = p['penalty']
                if ptype == 'day_off':
                    obj += pen * (1 - x[(n, d, 'None')])   # penalty if works
                elif ptype == 'day_on':
                    obj += pen * x[(n, d, 'None')]         # penalty if off
                elif ptype == 'shift_on' and shift and shift != 'None':
                    obj += pen * (1 - x[(n, d, shift)])
                elif ptype == 'shift_off' and shift and shift != 'None':
                    obj += pen * x[(n, d, shift)]

    # Final objective
    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for incumbent logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj_val)

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and model.SolCount > 0:
        schedule = {}
        skill_assignments = {}
        for n in nurse_ids:
            schedule[n] = {}
            skill_assignments[n] = {}
            for d in days:
                day_key = f"day_{d}"
                assigned_shift = None
                for s in all_shifts:
                    if x[(n, d, s)].X > 0.5:
                        assigned_shift = None if s == 'None' else s
                        break
                schedule[n][day_key] = assigned_shift

                if assigned_shift is None:
                    skill_assignments[n][day_key] = None
                else:
                    assigned_skill = None
                    for s in work_shifts:
                        for sk in skills:
                            if y[(n, d, s, sk)].X > 0.5:
                                assigned_skill = sk
                                break
                        if assigned_skill is not None:
                            break
                    skill_assignments[n][day_key] = assigned_skill

        objective_value = float(model.objVal)

        solution = {
            "objective_value": objective_value,
            "schedule": schedule,
            "skill_assignments": skill_assignments
        }

        if logger:
            logger.log_solution(objective_value, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or no solution
        fallback = {
            "objective_value": 1e9,
            "schedule": {},
            "skill_assignments": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(fallback, f, indent=2)


if __name__ == "__main__":
    main()