import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    shifts = instance['shifts']
    skills = instance['skills']
    nurses = instance['nurses']
    nurse_ids = [nurse['id'] for nurse in nurses]
    demand = instance['demand']
    constraints = instance['constraints']
    hard_constraints = constraints['hard_constraints']
    soft_constraints = constraints.get('soft_constraints', {})
    unsatisfied_demand_penalty = instance.get('unsatisfied_demand_penalty', 0)
    weekend_definition = instance['weekend_definition']
    weekends = instance['weekends']

    days = list(demand.keys())
    num_days = len(days)
    num_nurses = len(nurses)
    num_shifts = len(shifts)
    num_skills = len(skills)
    num_weekends = len(weekends)

    model = Model('NurseScheduling')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    schedule = model.addVars(num_nurses, num_days, num_shifts, vtype=GRB.BINARY, name='schedule')
    skill_assignments = model.addVars(num_vnurses, num_days, num_skills, vtype=GRB.BINARY, name='skill_assignments')
    unsatisfied_demand = model.addVars(num_days, num_shifts, num_skills, vtype=GRB.INTEGER, name='unsatisfied_demand', lb=0)

    # Constraints
    if hard_constraints.get('single_assignment_per_day', False):
        model.addConstrs((quicksum(schedule[n, d, s] for s in range(num_shifts)) == 1 for n in range(num_nurses) for d in range(num_days)), name='single_assignment_per_day')

    for d, day in enumerate(days):
        for s, shift in enumerate(shifts):
            for sk, skill in enumerate(skills):
                demand_value = demand.get(day, {}).get(shift, {}).get(skill, 0)
                if hard_constraints.get('demand_equality', False):
                    model.addConstr(quicksum(schedule[n, d, s] * skill_assignments[n, d, sk] for n in range(num_nurses) if skill in nurses[n].get('skills', [])) == demand_value, name=f'demand_equality_{day}_{shift}_{skill}')
                else:
                    model.addConstr(quicksum(schedule[n, d, s] * skill_assignments[n, d, sk] for n in range(num_nurses) if skill in nurses[n].get('skills', [])) + unsatisfied_demand[d, s, sk] >= demand_value, name=f'demand_satisfaction_{day}_{shift}_{skill}')

    if hard_constraints.get('skill_requirement', False):
        model.addConstrs((skill_assignments[n, d, sk] <= quicksum(schedule[n, d, s] for s in range(num_shifts)) for n in range(num_nurses) for d in range(num_days) for sk in range(num_skills)), name='skill_requirement_hard')

    for pattern in hard_constraints.get('forbidden_patterns', []):
        pattern_length = len(pattern['pattern'])
        for n in range(num_nurses):
            for d in range(num_days - pattern_length + 1):
                expr = quicksum(schedule[n, d + i, shifts.index(pattern['pattern'][i][1])] if pattern['pattern'][i][1] in shifts else 0 for i in range(pattern_length))
                model.addConstr(expr <= pattern_length - 1, name=f'forbidden_pattern_{n}_{d}_{pattern["description"]}')

    objective = 0
    objective += unsatisfied_demand_penalty * quicksum(unsatisfied_demand[d, s, sk] for d in range(num_days) for s in range(num_shifts) for sk in range(num_skills))

    if 'total_working_days' in soft_constraints:
        total_working_days_min = soft_constraints['total_working_days']['min']
        total_working_days_max = soft_constraints['total_working_days']['max']
        penalty_under = soft_constraints['total_working_days']['penalty_under']
        penalty_over = soft_constraints['total_working_days']['penalty_over']
        total_working_days = model.addVars(num_nurses, vtype=GRB.INTEGER, name='total_working_days')
        model.addConstrs((total_working_days[n] == quicksum(schedule[n, d, s] for d in range(num_days) for s in range(num_shifts) if shifts[s] != 'None') for n in range(num_nurses)), name='total_working_days_def')
        working_days_under = model.addVars(num_nurses, vtype=GRB.INTEGER, lb=0, name='working_days_under')
        working_days_over = model.addVars(num_nurses, vtype=GRB.INTEGER, lb=0, name='working_days_over')
        model.addConstrs((working_days_under[n] >= total_working_days_min - total_working_days[n] for n in range(num_nurses)), name='working_days_under_def')
        model.addConstrs((working_days_over[n] >= total_working_days[n] - total_working_days_max for n in range(num_nurses)), name='working_days_over_def')
        objective += penalty_under * quicksum(working_days_under[n] for n in range(num_nurses))
        objective += penalty_over * quicksum(working_days_over[n] for n in range(num_nurses))

    if 'total_worked_weekends' in soft_constraints:
        total_worked_weekends_max = soft_constraints['total_worked_weekends']['max']
        penalty = soft_constraints['total_worked_weekends']['penalty']
        total_worked_weekends = model.addVars(num_nurses, vtype=GRB.INTEGER, name='total_worked_weekends')
        model.addConstrs((total_worked_weekends[n] == quicksum(1 for w, weekend in enumerate(weekends) if quicksum(schedule[n, weekend['saturday'] - 1, s] + schedule[n, weekend['sunday'] - 1, s] for s in range(num_shifts) if shifts[s] != 'None') > 0) for n in range(num_nurses)), name='total_worked_weekends_def')
        worked_weekends_over = model.addVars(num_nurses, vtype=GRB.INTEGER, lb=0, name='worked_weekends_over')
        model.addConstrs((worked_weekends_over[n] >= total_worked_weekends[n] - total_worked_weekends_max for n in range(num_nurses)), name='worked_weekends_over_def')
        objective += penalty * quicksum(worked_weekends_over[n] for n in range(num_nurses))

    if 'consecutive_working_days' in soft_constraints:
        consecutive_working_days_min = soft_constraints['consecutive_working_days']['min']
        consecutive_working_days_max = soft_constraints['consecutive_working_days']['max']
        penalty_under = soft_constraints['consecutive_working_days']['penalty_under']
        penalty_over = soft_constraints['consecutive_working_days']['penalty_over']
        for n in range(num_nurses):
            consecutive_working_streaks = model.addVars(num_days, vtype=GRB.INTEGER, name=f'consecutive_working_streaks_{n}')
            for d in range(num_days):
                if d == 0:
                    model.addConstr(consecutive_working_streaks[d] == quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None'), name=f'consecutive_working_streaks_init_{n}_{d}')
                else:
                    model.addConstr(consecutive_working_streaks[d] >= quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None') * (consecutive_working_streaks[d-1] + 1), name=f'consecutive_working_streaks_update_{n}_{d}')
                    model.addConstr(consecutive_working_streaks[d] <= quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None') * (num_days + 1), name=f'consecutive_working_streaks_upper_{n}_{d}')
            for d in range(1, num_days):
                working_streak_under = model.addVar(vtype=GRB.BINARY, name=f'working_streak_under_{n}_{d}')
                model.addConstr(working_streak_under >= (consecutive_working_streaks[d-1] < consecutive_working_days_min) - (1 - quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None')), name=f'working_streak_under_def_{n}_{d}')
                objective += penalty_under * working_streak_under
                working_streak_over = model.addVar(vtype=GRB.INTEGER, lb=0, name=f'working_streak_over_{n}_{d}')
                model.addConstr(working_streak_over >= consecutive_working_streaks[d-1] - consecutive_working_days_max, name=f'working_streak_over_def_{n}_{d}')
                model.addConstr(working_streak_over <= (num_days + 1) * quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None'), name=f'working_streak_over_upper_{n}_{d}')
                objective += penalty_over * working_streak_over

    if 'consecutive_days_off' in soft_constraints:
        consecutive_days_off_min = soft_constraints['consecutive_days_off']['min']
        consecutive_days_off_max = soft_constraints['consecutive_days_off']['max']
        penalty_under = soft_constraints['consecutive_days_off']['penalty_under']
        penalty_over = soft_constraints['consecutive_days_off']['penalty_over']
        for n in range(num_nurses):
            consecutive_days_off_streaks = model.addVars(num_days, vtype=GRB.INTEGER, name=f'consecutive_days_off_streaks_{n}')
            for d in range(num_days):
                if d == 0:
                    model.addConstr(consecutive_days_off_streaks[d] == 1 - quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None'), name=f'consecutive_days_off_streaks_init_{n}_{d}')
                else:
                    model.addConstr(consecutive_days_off_streaks[d] >= (1 - quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None')) * (consecutive_days_off_streaks[d-1] + 1), name=f'consecutive_days_off_streaks_update_{n}_{d}')
                    model.addConstr(consecutive_days_off_streaks[d] <= (1 - quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None')) * (num_days + 1), name=f'consecutive_days_off_streaks_upper_{n}_{d}')
            for d in range(1, num_days):
                days_off_streak_under = model.addVar(vtype=GRB.BINARY, name=f'days_off_streak_under_{n}_{d}')
                model.addConstr(days_off_streak_under >= (consecutive_days_off_streaks[d-1] < consecutive_days_off_min) - (1 - quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] == 'None')), name=f'days_off_streak_under_def_{n}_{d}')
                objective += penalty_under * days_off_streak_under
                days_off_streak_over = model.addVar(vtype=GRB.INTEGER, lb=0, name=f'days_off_streak_over_{n}_{d}')
                model.addConstr(days_off_streak_over >= consecutive_days_off_streaks[d-1] - consecutive_days_off_max, name=f'days_off_streak_over_def_{n}_{d}')
                model.addConstr(days_off_streak_over <= (num_days + 1) * (1 - quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None')), name=f'days_off_streak_over_upper_{n}_{d}')
                objective += penalty_over * days_off_streak_over

    if 'consecutive_same_shift' in soft_constraints:
        consecutive_same_shift_max = soft_constraints['consecutive_same_shift']['max']
        penalty = soft_constraints['consecutive_same_shift']['penalty']
        for n in range(num_nurses):
            for s, shift in enumerate(shifts):
                if shift != 'None':
                    consecutive_same_shift_streaks = model.addVars(num_days, vtype=GRB.INTEGER, name=f'consecutive_same_shift_streaks_{n}_{s}')
                    for d in range(num_days):
                        if d == 0:
                            model.addConstr(consecutive_same_shift_streaks[d] == schedule[n, d, s], name=f'consecutive_same_shift_streaks_init_{n}_{s}_{d}')
                        else:
                            model.addConstr(consecutive_same_shift_streaks[d] >= schedule[n, d, s] * (consecutive_same_shift_streaks[d-1] + 1), name=f'consecutive_same_shift_streaks_update_{n}_{s}_{d}')
                            model.addConstr(consecutive_same_shift_streaks[d] <= schedule[n, d, s] * (num_days + 1), name=f'consecutive_same_shift_streaks_upper_{n}_{s}_{d}')
                    for d in range(1, num_days):
                        same_shift_streak_over = model.addVar(vtype=GRB.INTEGER, lb=0, name=f'same_shift_streak_over_{n}_{s}_{d}')
                        model.addConstr(same_shift_streak_over >= consecutive_same_shift_streaks[d-1] - consecutive_same_shift_max, name=f'same_shift_streak_over_def_{n}_{s}_{d}')
                        model.addConstr(same_shift_streak_over <= (num_days + 1) * schedule[n, d, s], name=f'same_shift_streak_over_upper_{n}_{s}_{d}')
                        objective += penalty * same_shift_streak_over

    if 'consecutive_worked_weekends' in soft_constraints:
        consecutive_worked_weekends_max = soft_constraints['consecutive_worked_weekends']['max']
        penalty = soft_constraints['consecutive_worked_weekends']['penalty']
        for n in range(num_nurses):
            consecutive_worked_weekends_streaks = model.addVars(num_weekends, vtype=GRB.INTEGER, name=f'consecutive_worked_weekends_streaks_{n}')
            for w, weekend in enumerate(weekends):
                worked_weekend = 1 if quicksum(schedule[n, weekend['saturday'] - 1, s] + schedule[n, weekend['sunday'] - 1, s] for s in range(num_shifts) if shifts[s] != 'None') > 0 else 0
                if w == 0:
                    model.addConstr(consecutive_worked_weekends_streaks[w] == worked_weekend, name=f'consecutive_worked_weekends_streaks_init_{n}_{w}')
                else:
                    model.addConstr(consecutive_worked_weekends_streaks[w] >= worked_weekend * (consecutive_worked_weekends_streaks[w-1] + 1), name=f'consecutive_worked_weekends_streaks_update_{n}_{w}')
                    model.addConstr(consecutive_worked_weekends_streaks[w] <= worked_weekend * (num_weekends + 1), name=f'consecutive_worked_weekends_streaks_upper_{n}_{w}')
            for w in range(1, num_weekends):
                worked_weekends_streak_over = model.addVar(vtype=GRB.INTEGER, lb=0, name=f'worked_weekends_streak_over_{n}_{w}')
                model.addConstr(worked_weekends_streak_over >= consecutive_worked_weekends_streaks[w-1] - consecutive_worked_weekends_max, name=f'worked_weekends_streak_over_def_{n}_{w}')
                model.addConstr(worked_weekends_streak_over <= (num_weekends + 1) * worked_weekend, name=f'worked_weekends_streak_over_upper_{n}_{w}')
                objective += penalty * worked_weekends_streak_over

    if 'identical_weekend' in soft_constraints and soft_constraints['identical_weekend']['enabled']:
        penalty = soft_constraints['identical_weekend']['penalty']
        for n in range(num_nurses):
            for w, weekend in enumerate(weekends):
                saturday_shift = model.addVar(vtype=GRB.INTEGER, name=f'saturday_shift_{n}_{weekend["saturday"]}')
                sunday_shift = model.addVar(vtype=GRB.INTEGER, name=f'sunday_shift_{n}_{weekend["sunday"]}')
                model.addConstr(saturday_shift == quicksum(s * schedule[n, weekend['saturday'] - 1, s] for s in range(num_shifts)), name=f'saturday_shift_def_{n}_{weekend["saturday"]}')
                model.addConstr(sunday_shift == quicksum(s * schedule[n, weekend['sunday'] - 1, s] for s in range(num_shifts)), name=f'sunday_shift_def_{n}_{weekend["sunday"]}')
                identical_weekend_violation = model.addVar(vtype=GRB.BINARY, name=f'identical_weekend_violation_{n}_{weekend["saturday"]}')
                model.addConstr(identical_weekend_violation >= (saturday_shift != sunday_shift), name=f'identical_weekend_violation_def_{n}_{weekend["saturday"]}')
                objective += penalty * identical_weekend_violation

    if 'skill_requirement_penalty' in soft_constraints:
        skill_requirement_penalty = soft_constraints['skill_requirement_penalty']
        skill_mismatch = model.addVars(num_nurses, num_days, num_skills, vtype=GRB.BINARY, name='skill_mismatch')
        model.addConstrs((skill_mismatch[n, d, sk] >= skill_assignments[n, d, sk] - (1 if skills[sk] in nurses[n].get('skills', []) else 0) for n in range(num_nurses) for d in range(num_days) for sk in range(num_skills)), name='skill_mismatch_def')
        model.addConstrs((skill_mismatch[n, d, sk] <= skill_assignments[n, d, sk] for n in range(num_nurses) for d in range(num_days) for sk in range(num_skills)), name='skill_mismatch_upper')
        model.addConstrs((skill_mismatch[n, d, sk] >= 0 for n in range(num_nurses) for d in range(num_days) for sk in range(num_skills)), name='skill_mismatch_lower')
        objective += skill_requirement_penalty * quicksum(skill_mismatch[n, d, sk] for n in range(num_nurses) for d in range(num_days) for sk in range(num_skills))

    if 'preferences_penalty' in soft_constraints and soft_constraints['preferences_penalty']:
        preferences_penalty = soft_constraints['preferences_penalty']
        for n, nurse in enumerate(nurses):
            for preference in nurse.get('preferences', []):
                d = preference['day'] - 1
                if preference['type'] == 'day_off':
                    objective += preference.get('penalty', 0) * quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None')
                elif preference['type'] == 'day_on':
                    objective += preference.get('penalty', 0) * (1 - quicksum(schedule[n, d, s] for s in range(num_shifts) if shifts[s] != 'None'))
                elif preference['type'] == 'shift_on':
                    if preference['shift'] in shifts:
                        s = shifts.index(preference['shift'])
                        objective += preference.get('penalty', 0) * (1 - schedule[n, d, s])
                elif preference['type'] == 'shift_off':
                    if preference['shift'] in shifts:
                        s = shifts.index(preference['shift'])
                        objective += preference.get('penalty', 0) * schedule[n, d, s]

    model.setObjective(objective, GRB.MINIMIZE)
    return model, schedule, skill_assignments, nurse_ids, days, shifts, skills

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    if 'None' not in instance['shifts']:
        instance['shifts'].append('None')
    shifts = instance['shifts']
    for nurse in instance['nurses']:
        for preference in nurse['preferences']:
            if preference['shift'] == 'None':
                preference['shift'] = None
    model, schedule, skill_assignments, nurse_ids, days, shifts, skills = solve(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.SolCount > 0:
        solution = {
            'objective_value': model.ObjVal,
            'schedule': {},
            'skill_assignments': {}
        }
        for n, nurse_id in enumerate(nurse_ids):
            solution['schedule'][nurse_id] = {}
            solution['skill_assignments'][nurse_id] = {}
            for d, day in enumerate(days):
                shift_assigned = None
                skill_assigned = None
                for s, shift in enumerate(shifts):
                    if schedule[n, d, s].x > 0.5:
                        shift_assigned = shift
                        break
                if shift_assigned is not None and shift_assigned != 'None':
                    for sk, skill in enumerate(skills):
                        if skill_assignments[n, d, sk].x > 0.5:
                            skill_assigned = skill
                            break
                solution['schedule'][nurse_id][day] = shift_assigned if shift_assigned is not None else 'None'
                solution['skill_assignments'][nurse_id][day] = skill_assigned if shift_assigned != 'None' else None

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

        if logger:
            logger.log_solution(model.ObjVal, solution)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()