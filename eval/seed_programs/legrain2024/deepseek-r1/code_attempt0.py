import json
import sys
import gurobipy as gp
from gurobipy import GRB

def main(instance_path, output_path, time_limit):
    with open(instance_path) as f:
        instance = json.load(f)
    
    nurses = instance['nurses']
    work_shifts = instance['shifts']
    skills = set()
    for nurse in nurses:
        skills.update(nurse['skills'])
    skills = list(skills)
    weekends = instance['weekends']
    constraints = instance['constraints']
    unsatisfied_demand_penalty = instance['unsatisfied_demand_penalty']
    
    num_days = len(instance['demand'])
    days = range(1, num_days + 1)
    shifts_all = work_shifts + ['None']
    
    demand_dict = {}
    for d in days:
        day_key = f'day_{d}'
        if day_key in instance['demand']:
            for s in work_shifts:
                if s in instance['demand'][day_key]:
                    for k in skills:
                        if k in instance['demand'][day_key][s]:
                            demand_dict[(d, s, k)] = instance['demand'][day_key][s][k]
                        else:
                            demand_dict[(d, s, k)] = 0
                else:
                    for k in skills:
                        demand_dict[(d, s, k)] = 0
        else:
            for s in work_shifts:
                for k in skills:
                    demand_dict[(d, s, k)] = 0
    
    nurse_skills = {nurse['id']: set(nurse['skills']) for nurse in nurses}
    nurse_ids = [nurse['id'] for nurse in nurses]
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    x = {}
    for n in nurse_ids:
        for d in days:
            for s in shifts_all:
                x[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f'x_{n}_{d}_{s}')
    
    w = {}
    for n in nurse_ids:
        for d in days:
            for s in work_shifts:
                for k in skills:
                    w[n, d, s, k] = model.addVar(vtype=GRB.BINARY, name=f'w_{n}_{d}_{s}_{k}')
    
    for n in nurse_ids:
        for d in days:
            model.addConstr(gp.quicksum(x[n, d, s] for s in shifts_all) == 1)
            for s in work_shifts:
                model.addConstr(x[n, d, s] == gp.quicksum(w[n, d, s, k] for k in skills))
    
    if constraints['hard_constraints']['demand_equality']:
        for (d, s, k), dem in demand_dict.items():
            model.addConstr(gp.quicksum(w[n, d, s, k] for n in nurse_ids) == dem)
    else:
        slack_minus = {}
        for (d, s, k), dem in demand_dict.items():
            slack_minus[d, s, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'slack_minus_{d}_{s}_{k}')
            model.addConstr(gp.quicksum(w[n, d, s, k] for n in nurse_ids) + slack_minus[d, s, k] >= dem)
    
    if constraints['hard_constraints']['skill_requirement']:
        for n in nurse_ids:
            for d in days:
                for s in work_shifts:
                    for k in skills:
                        if k not in nurse_skills[n]:
                            model.addConstr(w[n, d, s, k] == 0)
    
    if constraints['soft_constraints'].get('preferences_penalty', True):
        for nurse in nurses:
            n = nurse['id']
            for p in nurse['preferences']:
                d = p['day']
                penalty_val = p['penalty']
                if p['type'] == 'day_off':
                    model.addConstr(penalty_val * (1 - x[n, d, 'None']) >= 0)
                elif p['type'] == 'day_on':
                    model.addConstr(penalty_val * x[n, d, 'None'] >= 0)
                elif p['type'] == 'shift_on':
                    s0 = p['shift']
                    model.addConstr(penalty_val * (1 - x[n, d, s0]) >= 0)
                elif p['type'] == 'shift_off':
                    s0 = p['shift']
                    model.addConstr(penalty_val * x[n, d, s0] >= 0)
    
    total_work_days = {}
    slack_under = {}
    slack_over = {}
    for n in nurse_ids:
        total_work_days[n] = gp.quicksum(1 - x[n, d, 'None'] for d in days)
        slack_under[n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'slack_under_{n}')
        slack_over[n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'slack_over_{n}')
        min_days = constraints['soft_constraints']['total_working_days']['min']
        max_days = constraints['soft_constraints']['total_working_days']['max']
        model.addConstr(total_work_days[n] + slack_under[n] >= min_days)
        model.addConstr(total_work_days[n] - slack_over[n] <= max_days)
    
    works_weekend = {}
    for n in nurse_ids:
        for w in weekends:
            sat = w['saturday']
            sun = w['sunday']
            works_weekend[n, w['week']] = model.addVar(vtype=GRB.BINARY, name=f'works_weekend_{n}_{w["week"]}')
            model.addConstr(works_weekend[n, w['week']] >= 1 - x[n, sat, 'None'])
            model.addConstr(works_weekend[n, w['week']] >= 1 - x[n, sun, 'None'])
    
    total_weekends_worked = {}
    slack_over_weekends = {}
    for n in nurse_ids:
        total_weekends_worked[n] = gp.quicksum(works_weekend[n, w['week']] for w in weekends)
        slack_over_weekends[n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'slack_over_weekends_{n}')
        max_weekends = constraints['soft_constraints']['total_worked_weekends']['max']
        model.addConstr(total_weekends_worked[n] - slack_over_weekends[n] <= max_weekends)
    
    excess = {}
    max_consecutive_working = constraints['soft_constraints']['consecutive_working_days']['max']
    for n in nurse_ids:
        for d in range(max_consecutive_working + 1, num_days + 1):
            excess[n, d] = model.addVar(vtype=GRB.BINARY, name=f'excess_{n}_{d}')
            model.addConstr(excess[n, d] <= 1 - x[n, i, 'None'] for i in range(d - max_consecutive_working, d + 1))
            model.addConstr(excess[n, d] >= gp.quicksum(1 - x[n, i, 'None'] for i in range(d - max_consecutive_working, d + 1)) - max_consecutive_working)
    
    excess_off = {}
    max_consecutive_off = constraints['soft_constraints']['consecutive_days_off']['max']
    for n in nurse_ids:
        for d in range(max_consecutive_off + 1, num_days + 1):
            excess_off[n, d] = model.addVar(vtype=GRB.BINARY, name=f'excess_off_{n}_{d}')
            model.addConstr(excess_off[n, d] <= x[n, i, 'None'] for i in range(d - max_consecutive_off, d + 1))
            model.addConstr(excess_off[n, d] >= gp.quicksum(x[n, i, 'None'] for i in range(d - max_consecutive_off, d + 1)) - max_consecutive_off)
    
    excess_same_shift = {}
    max_same_shift = constraints['soft_constraints']['consecutive_same_shift']['max']
    for n in nurse_ids:
        for d in range(max_same_shift + 1, num_days + 1):
            for s in work_shifts:
                excess_same_shift[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f'excess_same_shift_{n}_{d}_{s}')
                model.addConstr(excess_same_shift[n, d, s] <= x[n, i, s] for i in range(d - max_same_shift, d + 1))
                model.addConstr(excess_same_shift[n, d, s] >= gp.quicksum(x[n, i, s] for i in range(d - max_same_shift, d + 1)) - max_same_shift)
    
    streak_length = {}
    for n in nurse_ids:
        for i, w in enumerate(weekends):
            week = w['week']
            streak_length[n, week] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(weekends), name=f'streak_length_{n}_{week}')
            if i == 0:
                model.addConstr(streak_length[n, week] <= len(weekends) * works_weekend[n, week])
                model.addConstr(streak_length[n, week] >= works_weekend[n, week])
            else:
                prev_week = weekends[i-1]['week']
                model.addConstr(streak_length[n, week] <= len(weekends) * works_weekend[n, week])
                model.addConstr(streak_length[n, week] >= streak_length[n, prev_week] + works_weekend[n, week] - len(weekends) * (1 - works_weekend[n, week]))
                model.addConstr(streak_length[n, week] <= streak_length[n, prev_week] + 1)
    
    identical_weekend_penalty = {}
    for n in nurse_ids:
        for w in weekends:
            sat = w['saturday']
            sun = w['sunday']
            identical_weekend_penalty[n, w['week']] = model.addVar(vtype=GRB.BINARY, name=f'identical_weekend_penalty_{n}_{w["week"]}')
            for s in shifts_all:
                z = model.addVar(vtype=GRB.BINARY, name=f'z_{n}_{w["week"]}_{s}')
                model.addConstr(z <= x[n, sat, s])
                model.addConstr(z <= x[n, sun, s])
                model.addConstr(z >= x[n, sat, s] + x[n, sun, s] - 1)
            model.addConstr(identical_weekend_penalty[n, w['week']] == 1 - gp.quicksum(z for s in shifts_all))
    
    obj = 0
    if not constraints['hard_constraints']['demand_equality']:
        obj += unsatisfied_demand_penalty * gp.quicksum(slack_minus.values())
    
    if not constraints['hard_constraints']['skill_requirement']:
        penalty_skill = constraints['soft_constraints']['skill_requirement_penalty']
        for n in nurse_ids:
            for d in days:
                for s in work_shifts:
                    for k in skills:
                        if k not in nurse_skills[n]:
                            obj += penalty_skill * w[n, d, s, k]
    
    penalty_under_days = constraints['soft_constraints']['total_working_days']['penalty_under']
    penalty_over_days = constraints['soft_constraints']['total_working_days']['penalty_over']
    obj += penalty_under_days * gp.quicksum(slack_under.values()) + penalty_over_days * gp.quicksum(slack_over.values())
    
    penalty_weekends = constraints['soft_constraints']['total_worked_weekends']['penalty']
    obj += penalty_weekends * gp.quicksum(slack_over_weekends.values())
    
    penalty_consecutive_working = constraints['soft_constraints']['consecutive_working_days']['penalty_over']
    obj += penalty_consecutive_working * gp.quicksum(excess.values())
    
    penalty_consecutive_off = constraints['soft_constraints']['consecutive_days_off']['penalty_over']
    obj += penalty_consecutive_off * gp.quicksum(excess_off.values())
    
    penalty_same_shift = constraints['soft_constraints']['consecutive_same_shift']['penalty']
    obj += penalty_same_shift * gp.quicksum(excess_same_shift.values())
    
    penalty_consecutive_weekends = constraints['soft_constraints']['consecutive_worked_weekends']['penalty']
    for n in nurse_ids:
        for w in weekends:
            week = w['week']
            excess_weekend = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'excess_weekend_{n}_{week}')
            model.addConstr(excess_weekend >= streak_length[n, week] - constraints['soft_constraints']['consecutive_worked_weekends']['max'])
            obj += penalty_consecutive_weekends * excess_weekend
    
    if constraints['soft_constraints']['identical_weekend']['enabled']:
        penalty_identical = constraints['soft_constraints']['identical_weekend']['penalty']
        obj += penalty_identical * gp.quicksum(identical_weekend_penalty.values())
    
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    schedule = {n: {d: None for d in days} for n in nurse_ids}
    skill_assignments = {n: {d: None for d in days} for n in nurse_ids}
    
    for n in nurse_ids:
        for d in days:
            for s in shifts_all:
                if x[n, d, s].X > 0.5:
                    schedule[n][d] = s
                    if s != 'None':
                        for k in skills:
                            if w[n, d, s, k].X > 0.5:
                                skill_assignments[n][d] = k
                    else:
                        skill_assignments[n][d] = 'None'
    
    output = {
        "schedule": schedule,
        "skill_assignments": skill_assignments
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    instance_path = sys.argv[1]
    output_path = sys.argv[2]
    time_limit = float(sys.argv[3])
    main(instance_path, output_path, time_limit)