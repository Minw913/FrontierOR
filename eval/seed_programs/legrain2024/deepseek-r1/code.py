import argparse
import json
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def main(instance_path, solution_path, time_limit, log_path=None):
    with open(instance_path) as f:
        instance = json.load(f)
    
    nurses = instance['nurses']
    work_shifts = instance['shifts']
    skills = instance['skills']
    weekends = instance['weekends']
    constraints = instance['constraints']
    unsatisfied_demand_penalty = instance.get('unsatisfied_demand_penalty', 0)
    
    num_days = len(instance['demand'])
    days = list(range(1, num_days + 1))
    shifts_all = work_shifts + ['None']
    
    demand_dict = {}
    for d in days:
        day_key = f'day_{d}'
        day_demand = instance['demand'].get(day_key, {})
        for s in work_shifts:
            shift_demand = day_demand.get(s, {})
            for k in skills:
                demand_dict[(d, s, k)] = shift_demand.get(k, 0)
    
    nurse_skills = {nurse['id']: set(nurse['skills']) for nurse in nurses}
    nurse_ids = [nurse['id'] for nurse in nurses]
    
    logger = None
    if log_path:
        logger = SolutionLogger(log_path, sense="minimize")
    
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
    
    slack_minus = {}
    if constraints['hard_constraints'].get('demand_equality', False):
        for (d, s, k), dem in demand_dict.items():
            model.addConstr(gp.quicksum(w[n, d, s, k] for n in nurse_ids) == dem)
    else:
        for (d, s, k), dem in demand_dict.items():
            slack_minus[d, s, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'slack_minus_{d}_{s}_{k}')
            model.addConstr(gp.quicksum(w[n, d, s, k] for n in nurse_ids) + slack_minus[d, s, k] >= dem)
    
    if constraints['hard_constraints'].get('skill_requirement', False):
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
        for wd in weekends:
            week_num = wd['week']
            sat = wd['saturday']
            sun = wd['sunday']
            works_weekend[n, week_num] = model.addVar(vtype=GRB.BINARY, name=f'works_weekend_{n}_{week_num}')
            model.addConstr(works_weekend[n, week_num] >= 1 - x[n, sat, 'None'])
            model.addConstr(works_weekend[n, week_num] >= 1 - x[n, sun, 'None'])
            model.addConstr(works_weekend[n, week_num] <= (1 - x[n, sat, 'None']) + (1 - x[n, sun, 'None']))
    
    total_weekends_worked = {}
    slack_over_weekends = {}
    for n in nurse_ids:
        total_weekends_worked[n] = gp.quicksum(works_weekend[n, wd['week']] for wd in weekends)
        slack_over_weekends[n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'slack_over_weekends_{n}')
        max_weekends = constraints['soft_constraints']['total_worked_weekends']['max']
        model.addConstr(total_weekends_worked[n] - slack_over_weekends[n] <= max_weekends)
    
    excess_work = {}
    max_consecutive_working = constraints['soft_constraints']['consecutive_working_days']['max']
    penalty_consecutive_working = constraints['soft_constraints']['consecutive_working_days']['penalty_over']
    for n in nurse_ids:
        for d in range(max_consecutive_working + 1, num_days + 1):
            excess_work[n, d] = model.addVar(vtype=GRB.BINARY, name=f'excess_work_{n}_{d}')
            model.addConstr(excess_work[n, d] <= gp.quicksum(1 - x[n, i, 'None'] for i in range(d - max_consecutive_working, d + 1)))
            model.addConstr(excess_work[n, d] >= gp.quicksum(1 - x[n, i, 'None'] for i in range(d - max_consecutive_working, d + 1)) - max_consecutive_working)
    
    excess_off = {}
    max_consecutive_off = constraints['soft_constraints']['consecutive_days_off']['max']
    penalty_consecutive_off = constraints['soft_constraints']['consecutive_days_off']['penalty_over']
    for n in nurse_ids:
        for d in range(max_consecutive_off + 1, num_days + 1):
            excess_off[n, d] = model.addVar(vtype=GRB.BINARY, name=f'excess_off_{n}_{d}')
            model.addConstr(excess_off[n, d] <= gp.quicksum(x[n, i, 'None'] for i in range(d - max_consecutive_off, d + 1)))
            model.addConstr(excess_off[n, d] >= gp.quicksum(x[n, i, 'None'] for i in range(d - max_consecutive_off, d + 1)) - max_consecutive_off)
    
    excess_same_shift = {}
    max_same_shift = constraints['soft_constraints']['consecutive_same_shift']['max']
    penalty_same_shift = constraints['soft_constraints']['consecutive_same_shift']['penalty']
    for n in nurse_ids:
        for d in range(max_same_shift + 1, num_days + 1):
            for s in work_shifts:
                excess_same_shift[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f'excess_same_shift_{n}_{d}_{s}')
                model.addConstr(excess_same_shift[n, d, s] <= gp.quicksum(x[n, i, s] for i in range(d - max_same_shift, d + 1)))
                model.addConstr(excess_same_shift[n, d, s] >= gp.quicksum(x[n, i, s] for i in range(d - max_same_shift, d + 1)) - max_same_shift)
    
    streak_length = {}
    for n in nurse_ids:
        for i, wd in enumerate(weekends):
            week_num = wd['week']
            streak_length[n, week_num] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(weekends), name=f'streak_length_{n}_{week_num}')
            if i == 0:
                model.addConstr(streak_length[n, week_num] == works_weekend[n, week_num])
            else:
                prev_week = weekends[i-1]['week']
                model.addConstr(streak_length[n, week_num] >= works_weekend[n, week_num])
                model.addConstr(streak_length[n, week_num] <= streak_length[n, prev_week] + works_weekend[n, week_num])
                model.addConstr(streak_length[n, week_num] >= streak_length[n, prev_week] + works_weekend[n, week_num] - (1 - works_weekend[n, week_num]) * len(weekends))
    
    in_excess = {}
    max_consecutive_weekends = constraints['soft_constraints']['consecutive_worked_weekends']['max']
    penalty_consecutive_weekends = constraints['soft_constraints']['consecutive_worked_weekends']['penalty']
    M = len(weekends) + 1
    for n in nurse_ids:
        for wd in weekends:
            week_num = wd['week']
            in_excess[n, week_num] = model.addVar(vtype=GRB.BINARY, name=f'in_excess_{n}_{week_num}')
            model.addConstr(streak_length[n, week_num] <= max_consecutive_weekends + in_excess[n, week_num] * M)
            model.addConstr(in_excess[n, week_num] >= (streak_length[n, week_num] - max_consecutive_weekends) / M)
    
    identical_weekend_penalty = {}
    if constraints['soft_constraints']['identical_weekend']['enabled']:
        penalty_identical = constraints['soft_constraints']['identical_weekend']['penalty']
        for n in nurse_ids:
            for wd in weekends:
                week_num = wd['week']
                sat = wd['saturday']
                sun = wd['sunday']
                identical_weekend_penalty[n, week_num] = model.addVar(vtype=GRB.BINARY, name=f'identical_weekend_penalty_{n}_{week_num}')
                for s in shifts_all:
                    z = model.addVar(vtype=GRB.BINARY, name=f'z_{n}_{week_num}_{s}')
                    model.addConstr(z <= x[n, sat, s])
                    model.addConstr(z <= x[n, sun, s])
                    model.addConstr(z >= x[n, sat, s] + x[n, sun, s] - 1)
                model.addConstr(identical_weekend_penalty[n, week_num] == 1 - gp.quicksum(z for s in shifts_all))
    
    obj = 0
    if not constraints['hard_constraints'].get('demand_equality', False):
        obj += unsatisfied_demand_penalty * gp.quicksum(slack_minus.values())
    
    if not constraints['hard_constraints'].get('skill_requirement', False):
        penalty_skill = constraints['soft_constraints'].get('skill_requirement_penalty', 0)
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
    
    obj += penalty_consecutive_working * gp.quicksum(excess_work.values())
    obj += penalty_consecutive_off * gp.quicksum(excess_off.values())
    obj += penalty_same_shift * gp.quicksum(excess_same_shift.values())
    obj += penalty_consecutive_weekends * gp.quicksum(in_excess.values())
    
    if constraints['soft_constraints']['identical_weekend']['enabled']:
        penalty_identical = constraints['soft_constraints']['identical_weekend']['penalty']
        obj += penalty_identical * gp.quicksum(identical_weekend_penalty.values())
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    def callback_wrapper(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                logger.log(model.cbGet(GRB.Callback.MIPSOL_OBJ))
    
    if logger:
        model.optimize(callback_wrapper)
    else:
        model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        objective_value = model.ObjVal
        schedule = {n: {} for n in nurse_ids}
        skill_assignments = {n: {} for n in nurse_ids}
        
        for n in nurse_ids:
            for d in days:
                for s in shifts_all:
                    if x[n, d, s].X > 0.5:
                        schedule[n][f'day_{d}'] = s
                        if s != 'None':
                            for k in skills:
                                if w[n, d, s, k].X > 0.5:
                                    skill_assignments[n][f'day_{d}'] = k
                        else:
                            skill_assignments[n][f'day_{d}'] = 'None'
    else:
        objective_value = 1e20
        schedule = {n: {f'day_{d}': 'None' for d in days} for n in nurse_ids}
        skill_assignments = {n: {f'day_{d}': 'None' for d in days} for n in nurse_ids}
    
    solution_dict = {
        "objective_value": objective_value,
        "schedule": schedule,
        "skill_assignments": skill_assignments
    }
    
    if logger and model.SolCount > 0:
        logger.log_solution(objective_value, solution_dict)
    
    with open(solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)