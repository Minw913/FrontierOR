import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def model_instance(instance):
    num_nurses = instance['num_nurses']
    num_weeks = instance['num_weeks']
    num_days = instance['num_days']
    num_shifts = instance['num_shifts']
    num_skills = instance['num_skills']
    skills = instance['skills']
    shifts = instance['shifts']
    forbidden_shift_successions = instance['forbidden_shift_successions']
    penalty_weights = instance['penalty_weights']
    nurses = instance['nurses']
    demand = instance['demand']

    # Create model
    model = gp.Model('nurse_scheduling')
    model.modelSense = GRB.MINIMIZE

    # Decision variables
    x = model.addVars(num_nurses, num_days, num_shifts, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_nurses, num_days, num_skills, vtype=GRB.BINARY, name='y')
    z = model.addVars(num_nurses, num_days, vtype=GRB.BINARY, name='z')

    # Constraints
    # 1. Demand coverage
    for d in range(num_days):
        for s, shift in enumerate(shifts):
            for k, skill in enumerate(skills):
                min_demand = demand[d][shift['name']][skill]['minimum']
                model.addConstr(gp.quicksum(y[n, d, k] for n in range(num_nurses) if skill in nurses[n]['skills']) >= min_demand)

    # 2. Skill and shift assignment
    for n in range(num_nurses):
        for d in range(num_days):
            model.addConstr(gp.quicksum(x[n, d, s] for s in range(num_shifts)) <= 1)
            for s in range(num_shifts):
                for k, skill in enumerate(skills):
                    model.addConstr(y[n, d, k] <= x[n, d, s])
                    model.addConstr(y[n, d, k] <= (1 if skill in nurses[n]['skills'] else 0))

    # 3. Forbidden shift successions
    for n in range(num_nurses):
        for d in range(num_days - 1):
            for s1, shift1 in enumerate(shifts):
                for s2, shift2 in enumerate(shifts):
                    if [shift1['name'], shift2['name']] in forbidden_shift_successions:
                        model.addConstr(x[n, d, s1] + x[n, d + 1, s2] <= 1)

    # 4. z variable (worked days)
    for n in range(num_nurses):
        for d in range(num_days):
            model.addConstr(z[n, d] == gp.quicksum(x[n, d, s] for s in range(num_shifts)))

    # Objective function components
    insufficient_staffing = gp.quicksum(
        penalty_weights['c_S1'] * max(0, demand[d][shift['name']][skill]['optimal'] - gp.quicksum(y[n, d, k] for n in range(num_nurses) if skill in nurses[n]['skills']))
        for d in range(num_days)
        for s, shift in enumerate(shifts)
        for k, skill in enumerate(skills)
    )

    consecutive_worked_days_penalty = gp.quicksum(
        penalty_weights['c_S2a'] * (gp.quicksum(
            z[n, d + i] for i in range(contract['CD_plus'] + 1)
        ) - contract['CD_plus']) for n, nurse in enumerate(nurses) for d in range(num_days - contract['CD_plus'] - 1) if contract['CD_plus'] < num_days - d - 1)
    consecutive_worked_days_penalty += gp.quicksum(
        penalty_weights['c_S2a'] * (contract['CD_minus'] - gp.quicksum(
            z[n, d + i] for i in range(contract['CD_minus'])
        )) for n, nurse in enumerate(nurses) for d in range(num_days - contract['CD_minus'] + 1) if d + contract['CD_minus'] <= num_days)

    consecutive_same_shift_penalty = gp.quicksum(
        penalty_weights['c_S2b'] * (gp.quicksum(
            x[n, d + i, s] for i in range(shift['CS_plus'] + 1)
        ) - shift['CS_plus']) for n in range(num_nurses) for d in range(num_days - shift['CS_plus'] - 1) for s, shift in enumerate(shifts) if shift['CS_plus'] < num_days - d - 1)
    consecutive_same_shift_penalty += gp.quicksum(
        penalty_weights['c_S2b'] * (shift['CS_minus'] - gp.quicksum(
            x[n, d + i, s] for i in range(shift['CS_minus'])
        )) for n in range(num_nurses) for d in range(num_days - shift['CS_minus'] + 1) for s, shift in enumerate(shifts) if d + shift['CS_minus'] <= num_days)

    consecutive_days_off_penalty = gp.quicksum(
        penalty_weights['c_S3'] * (contract['CR_plus'] - (1 - gp.quicksum(
            z[n, d + i] for i in range(contract['CR_plus'] + 1)
        ))) for n, nurse in enumerate(nurses) for d in range(num_days - contract['CR_plus'] - 1) if contract['CR_plus'] < num_days - d - 1)
    consecutive_days_off_penalty += gp.quicksum(
        penalty_weights['c_S3'] * ((1 - gp.quicksum(
            z[n, d + i] for i in range(contract['CR_minus'])
        )) - contract['CR_minus']) for n, nurse in enumerate(nurses) for d in range(num_days - contract['CR_minus'] + 1) if d + contract['CR_minus'] <= num_days)

    preference_violations = gp.quicksum(
        penalty_weights['c_S4'] * x[n, pref['day'], s]
        for n, nurse in enumerate(nurses)
        for pref in nurse['preferences']
        for s, shift in enumerate(shifts)
        if shift['name'] == pref['shift']
    )

    incomplete_weekend = gp.quicksum(
        penalty_weights['c_S5'] * (z[n, d] + z[n, d + 1] - 1)
        for n in range(num_nurses)
        for d in range(0, num_days - 1, 7)
        if d + 1 < num_days
    )

    total_assignment_deviation = gp.quicksum(
        penalty_weights['c_S6'] * (gp.quicksum(z[n, d] for d in range(num_days)) - contract['L_plus'])
        for n, nurse in enumerate(nurses)
        if gp.quicksum(z[n, d] for d in range(num_days)) > contract['L_plus']
    )
    total_assignment_deviation += gp.quicksum(
        penalty_weights['c_S6'] * (contract['L_minus'] - gp.quicksum(z[n, d] for d in range(num_days)))
        for n, nurse in enumerate(nurses)
        if gp.quicksum(z[n, d] for d in range(num_days)) < contract['L_minus']
    )

    worked_weekend_excess = gp.quicksum(
        penalty_weights['c_S7'] * (gp.quicksum(
            (z[n, d] or z[n, d + 1]) for d in range(0, num_days - 1, 7)
        ) - contract['WE_plus'])
        for n, nurse in enumerate(nurses)
        if gp.quicksum((z[n, d] or z[n, d + 1]) for d in range(0, num_days - 1, 7)) > contract['WE_plus']
    )

    model.setObjective(
        insufficient_staffing +
        consecutive_worked_days_penalty +
        consecutive_same_shift_penalty +
        consecutive_days_off_penalty +
        preference_violations +
        incomplete_weekend +
        total_assignment_deviation +
        worked_weekend_excess
    )

    return model, x, y, z

def solve_model(model, time_limit, logger):
    model.Params.TimeLimit = time_limit
    model.optimize()
    if logger:
        for sol in range(model.SolCount):
            model.Params.SolutionNumber = sol
            objective_value = model.getObjective().getValue()
            logger.log(objective_value)

def construct_solution(instance, x, y, z):
    nurses = instance['nurses']
    num_days = instance['num_days']
    num_shifts = instance['num_shifts']
    num_skills = instance['num_skills']
    skills = instance['skills']
    shifts = instance['shifts']

    solution = {'schedule': []}
    for n, nurse in enumerate(nurses):
        nurse_schedule = {'nurse_id': nurse['id'], 'nurse_name': nurse['name'], 'assignments': []}
        for d in range(num_days):
            for s in range(num_shifts):
                if x[n, d, s].x > 0.5:
                    for k in range(num_skills):
                        if y[n, d, k].x > 0.5:
                            nurse_schedule['assignments'].append({
                                'day': d,
                                'shift': shifts[s]['name'],
                                'skill': skills[k]
                            })
        solution['schedule'].append(nurse_schedule)
    solution['objective_value'] = sum(
        x[n, d, s].x * 1 for n in range(len(nurses)) for d in range(num_days) for s in range(num_shifts)
    )  # This is a placeholder; actual objective value should be computed based on the model's objective function
    return solution

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    model, x, y, z = model_instance(instance)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solve_model(model, args.time_limit, logger)
    solution = construct_solution(instance, x, y, z)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()