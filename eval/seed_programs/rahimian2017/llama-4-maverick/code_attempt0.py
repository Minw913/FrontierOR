import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_nurses = instance['num_nurses']
    num_days = instance['num_days']
    num_shift_types = instance['num_shift_types']
    num_weekends = instance['num_weekends']
    shift_types = instance['shift_types']
    forbidden_shift_rotations = instance['forbidden_shift_rotations']
    nurses = instance['nurses']
    shift_on_requests = instance['shift_on_requests']
    shift_off_requests = instance['shift_off_requests']
    coverage_requirements = instance['coverage_requirements']

    model = Model('nurse_scheduling')
    model.modelSense = GRB.MINIMIZE

    # Variables
    assignment = model.addVars(num_nurses, num_days, num_shift_types, vtype=GRB.BINARY, name='assignment')
    coverage_under = model.addVars(num_days, num_shift_types, vtype=GRB.INTEGER, name='coverage_under', lb=0)
    coverage_over = model.addVars(num_days, num_shift_types, vtype=GRB.INTEGER, name='coverage_over', lb=0)

    # Objective function
    objective = quicksum(req['penalty'] * (1 - assignment[nurse_id, day, shift_type]) 
                         for nurse_id, day, shift_type, req in 
                         [(req['nurse_id'], req['day'], req['shift_type'], req) for req in shift_on_requests]) + \
                quicksum(req['penalty'] * assignment[nurse_id, day, shift_type] 
                         for nurse_id, day, shift_type, req in 
                         [(req['nurse_id'], req['day'], req['shift_type'], req) for req in shift_off_requests]) + \
                quicksum(coverage_under[day, shift_type] * req['under_weight'] + 
                         coverage_over[day, shift_type] * req['over_weight'] 
                         for day, shift_type, req in 
                         [(req['day'], req['shift_type'], req) for req in coverage_requirements])
    model.setObjective(objective)

    # Constraints
    for nurse_id in range(num_nurses):
        nurse = nurses[nurse_id]
        for day in range(num_days):
            # Each nurse works at most one shift per day
            model.addConstr(quicksum(assignment[nurse_id, day, shift_type] for shift_type in range(num_shift_types)) <= 1, 
                            name=f'nurse_{nurse_id}_day_{day}_one_shift')
            # Nurse cannot work on requested days off
            if day in nurse['day_off_requests']:
                model.addConstr(quicksum(assignment[nurse_id, day, shift_type] for shift_type in range(num_shift_types)) == 0, 
                                name=f'nurse_{nurse_id}_day_{day}_off_request')

        for shift_type in range(num_shift_types):
            # Maximum shifts per type
            model.addConstr(quicksum(assignment[nurse_id, day, shift_type] for day in range(num_days)) <= nurse['max_shifts_per_type'][str(shift_type)], 
                            name=f'nurse_{nurse_id}_shift_type_{shift_type}_max')

        # Total working time
        total_minutes = quicksum(assignment[nurse_id, day, shift_type] * shift_types[shift_type]['length_minutes'] 
                                 for day in range(num_days) for shift_type in range(num_shift_types))
        model.addConstr(total_minutes >= nurse['min_total_minutes'], name=f'nurse_{nurse_id}_min_total_minutes')
        model.addConstr(total_minutes <= nurse['max_total_minutes'], name=f'nurse_{nurse_id}_max_total_minutes')

        # Consecutive shifts and days off constraints
        for day in range(num_days - nurse['max_consecutive_shifts']):
            model.addConstr(quicksum(assignment[nurse_id, d, shift_type] for d in range(day, day + nurse['max_consecutive_shifts'] + 1) for shift_type in range(num_shift_types)) <= nurse['max_consecutive_shifts'], 
                            name=f'nurse_{nurse_id}_day_{day}_max_consecutive_shifts')
        for day in range(num_days - nurse['min_consecutive_days_off']):
            model.addConstr(quicksum(1 - quicksum(assignment[nurse_id, d, shift_type] for shift_type in range(num_shift_types)) for d in range(day, day + nurse['min_consecutive_days_off'] + 1)) >= 1, 
                            name=f'nurse_{nurse_id}_day_{day}_min_consecutive_days_off')

        # Forbidden shift rotations
        for day in range(num_days - 1):
            for forbidden in forbidden_shift_rotations:
                model.addConstr(assignment[nurse_id, day, forbidden['shift_type']] + assignment[nurse_id, day + 1, forbidden['cannot_be_followed_by']] <= 1, 
                                name=f'nurse_{nurse_id}_day_{day}_forbidden_rotation_{forbidden["shift_type"]}_{forbidden["cannot_be_followed_by"]}')

        # Maximum weekends
        weekend_work = quicksum(1 for weekend in range(num_weekends) if any(assignment[nurse_id, day, shift_type] for day in [weekend * 2 + 5, weekend * 2 + 6] for shift_type in range(num_shift_types)))
        model.addConstr(weekend_work <= nurse['max_weekends'], name=f'nurse_{nurse_id}_max_weekends')

    for day, shift_type, req in [(req['day'], req['shift_type'], req) for req in coverage_requirements]:
        actual_nurses = quicksum(assignment[nurse_id, day, shift_type] for nurse_id in range(num_nurses))
        model.addConstr(coverage_under[day, shift_type] >= req['preferred'] - actual_nurses, name=f'coverage_under_day_{day}_shift_type_{shift_type}')
        model.addConstr(coverage_over[day, shift_type] >= actual_nurses - req['preferred'], name=f'coverage_over_day_{day}_shift_type_{shift_type}')

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f'Optimal objective value: {model.objVal}')
    else:
        print(f'Best objective value found: {model.objVal}')

    return model

def extract_solution(model, instance):
    num_nurses = instance['num_nurses']
    num_days = instance['num_days']
    num_shift_types = instance['num_shift_types']
    coverage_requirements = instance['coverage_requirements']

    schedule = {}
    for nurse_id in range(num_nurses):
        nurse_schedule = {}
        for day in range(num_days):
            for shift_type in range(num_shift_types):
                if model.getVarByName(f'assignment[{nurse_id},{day},{shift_type}]').x > 0.5:
                    nurse_schedule[day] = shift_type
                    break
            else:
                nurse_schedule[day] = None
        schedule[nurse_id] = nurse_schedule

    coverage_under = {}
    coverage_over = {}
    for day, shift_type, req in [(req['day'], req['shift_type'], req) for req in coverage_requirements]:
        key = f'{day}_{shift_type}'
        coverage_under[key] = int(model.getVarByName(f'coverage_under[{day},{shift_type}]').x)
        coverage_over[key] = int(model.getVarByName(f'coverage_over[{day},{shift_type}]').x)

    solution = {
        'objective_value': model.objVal,
        'solution': {
            'schedule': schedule,
            'coverage_under': coverage_under,
            'coverage_over': coverage_over
        }
    }
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    model = Model('nurse_scheduling')
    model.params.TimeLimit = args.time_limit
    model.params.LogToConsole = 0

    solve(instance)

    # Callback to log intermediate solutions
    def log_solution(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(log_solution)

    solution = extract_solution(model, instance)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()