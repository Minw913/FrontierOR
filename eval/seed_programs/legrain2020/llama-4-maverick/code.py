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
    num_shifts = len(instance['shifts'])
    num_skills = instance['num_skills']
    skills = instance['skills']
    shifts = instance['shifts']
    shift_names = [shift['name'] for shift in shifts]
    forbidden_shift_successions = instance.get('forbidden_shift_successions', [])
    penalty_weights = instance['penalty_weights']
    nurses = instance['nurses']
    demand = instance['demand']

    model = gp.Model('nurse_scheduling')
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    x = model.addVars(num_nurses, num_days, num_shifts, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_nurses, num_days, num_skills, vtype=GRB.BINARY, name='y')
    z = model.addVars(num_nurses, num_days, vtype=GRB.BINARY, name='z')

    # Demand coverage
    shortage_vars = {}
    for d in range(num_days):
        for s, shift_name in enumerate(shift_names):
            if shift_name in demand[d]:
                for skill_name in demand[d][shift_name]:
                    skill_index = skills.index(skill_name)
                    min_demand = demand[d][shift_name][skill_name]['minimum']
                    optimal_demand = demand[d][shift_name][skill_name]['optimal']
                    shortage = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"shortage_{d}_{shift_name}_{skill_name}")
                    shortage_vars[(d, shift_name, skill_name)] = shortage
                    model.addConstr(gp.quicksum(y[n, d, skill_index] for n in range(num_nurses) if skill_name in nurses[n]['skills']) >= min_demand)
                    model.addConstr(shortage >= optimal_demand - gp.quicksum(y[n, d, skill_index] for n in range(num_nurses) if skill_name in nurses[n]['skills']))

    # Skill and shift assignment
    for n in range(num_nurses):
        for d in range(num_days):
            model.addConstr(gp.quicksum(x[n, d, s] for s in range(num_shifts)) <= 1)
            for s, shift_name in enumerate(shift_names):
                for k, skill_name in enumerate(skills):
                    if skill_name in nurses[n]['skills']:
                        model.addConstr(y[n, d, k] <= x[n, d, s])
                    else:
                        model.addConstr(y[n, d, k] == 0)

    # Forbidden shift successions
    for n in range(num_nurses):
        for d in range(num_days - 1):
            for s1, shift1_name in enumerate(shift_names):
                for s2, shift2_name in enumerate(shift_names):
                    if [shift1_name, shift2_name] in forbidden_shift_successions:
                        model.addConstr(x[n, d, s1] + x[n, d + 1, s2] <= 1)

    # z variable (worked days)
    for n in range(num_nurses):
        for d in range(num_days):
            model.addConstr(z[n, d] == gp.quicksum(x[n, d, s] for s in range(num_shifts)))

    # Objective function components
    insufficient_staffing = gp.quicksum(
        penalty_weights['c_S1'] * shortage_vars.get((d, shift_name, skill_name), 0)
        for d in range(num_days)
        for shift_name in demand[d]
        for skill_name in demand[d][shift_name]
    )

    consecutive_worked_days_penalty = 0
    consecutive_same_shift_penalty = 0
    consecutive_days_off_penalty = 0
    for n in range(num_nurses):
        for d in range(num_days):
            if d + nurses[n]['contract']['CD_plus'] + 1 <= num_days:
                consecutive_worked_days_penalty += penalty_weights['c_S2a'] * gp.max_(0, gp.quicksum(z[n, d + i] for i in range(nurses[n]['contract']['CD_plus'] + 1)) - nurses[n]['contract']['CD_plus'])
            if d + nurses[n]['contract']['CD_minus'] <= num_days:
                consecutive_worked_days_penalty += penalty_weights['c_S2a'] * gp.max_(0, nurses[n]['contract']['CD_minus'] - gp.quicksum(z[n, d + i] for i in range(nurses[n]['contract']['CD_minus'])))
            for s in range(num_shifts):
                if d + shifts[s]['CS_plus'] + 1 <= num_days:
                    consecutive_same_shift_penalty += penalty_weights['c_S2b'] * gp.max_(0, gp.quicksum(x[n, d + i, s] for i in range(shifts[s]['CS_plus'] + 1)) - shifts[s]['CS_plus'])
                if d + shifts[s]['CS_minus'] <= num_days:
                    consecutive_same_shift_penalty += penalty_weights['c_S2b'] * gp.max_(0, shifts[s]['CS_minus'] - gp.quicksum(x[n, d + i, s] for i in range(shifts[s]['CS_minus'])))
            if d + nurses[n]['contract']['CR_plus'] + 1 <= num_days:
                consecutive_days_off_penalty += penalty_weights['c_S3'] * gp.max_(0, gp.quicksum(1 - z[n, d + i] for i in range(nurses[n]['contract']['CR_plus'] + 1)) - nurses[n]['contract']['CR_plus'])
            if d + nurses[n]['contract']['CR_minus'] <= num_days:
                consecutive_days_off_penalty += penalty_weights['c_S3'] * gp.max_(0, nurses[n]['contract']['CR_minus'] - gp.quicksum(1 - z[n, d + i] for i in range(nurses[n]['contract']['CR_minus'])))

    preference_violations = gp.quicksum(
        penalty_weights['c_S4'] * x[n, d, shift_names.index(pref['shift'])]
        for n, nurse in enumerate(nurses)
        for pref in nurse.get('preferences', [])
        if pref['shift'] in shift_names
        for d in [pref['day']]
    )

    incomplete_weekend = gp.quicksum(
        penalty_weights['c_S5'] * gp.abs_(z[n, d] - z[n, d + 1])
        for n in range(num_nurses)
        for d in range(0, num_days - 1, 7)
        if d + 1 < num_days
    )

    total_assignment_deviation = gp.quicksum(
        penalty_weights['c_S6'] * gp.max_(0, gp.quicksum(z[n, d] for d in range(num_days)) - nurses[n]['contract']['L_plus'])
        for n in range(num_nurses)
    )
    total_assignment_deviation += gp.quicksum(
        penalty_weights['c_S6'] * gp.max_(0, nurses[n]['contract']['L_minus'] - gp.quicksum(z[n, d] for d in range(num_days)))
        for n in range(num_nurses)
    )

    worked_weekend_excess = gp.quicksum(
        penalty_weights['c_S7'] * gp.max_(0, gp.quicksum(gp.max_(z[n, d], z[n, d + 1]) for d in range(0, num_days - 1, 7)) - nurses[n]['contract']['WE_plus'])
        for n in range(num_nurses)
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
    shifts = instance['shifts']
    shift_names = [shift['name'] for shift in shifts]
    skills = instance['skills']

    solution = {'schedule': []}
    for n, nurse in enumerate(nurses):
        nurse_schedule = {'nurse_id': nurse['id'], 'nurse_name': nurse['name'], 'assignments': []}
        for d in range(num_days):
            for s, shift_name in enumerate(shift_names):
                if x[n, d, s].x > 0.5:
                    for k, skill in enumerate(skills):
                        if y[n, d, k].x > 0.5:
                            nurse_schedule['assignments'].append({
                                'day': d,
                                'shift': shift_name,
                                'skill': skill
                            })
        solution['schedule'].append(nurse_schedule)
    try:
        solution['objective_value'] = float(x.model.ObjVal)
    except:
        solution['objective_value'] = float('inf')
    return solution

def main():
    global model
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