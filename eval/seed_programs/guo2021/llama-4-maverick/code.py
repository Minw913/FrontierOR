import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance, time_limit, log_path=None):
    num_patients = instance['num_patients']
    num_hospitals = instance['num_hospitals']
    num_days = instance['num_days']
    num_ors_per_hospital = instance['num_ors_per_hospital']
    num_scenarios = instance['num_scenarios']
    kappa_1 = instance['global_parameters']['kappa_1']
    kappa_2 = instance['global_parameters']['kappa_2']
    kappa_3 = instance['global_parameters']['kappa_3']
    kappa_4 = instance['global_parameters']['kappa_4']

    patients = instance['patients']
    hospitals = instance['hospitals']
    scenarios = instance['scenarios']

    mandatory_patients = [p['patient_id'] for p in patients if p['is_mandatory']]
    non_mandatory_patients = [p['patient_id'] for p in patients if not p['is_mandatory']]

    model = Model('surgery_scheduling')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Variables
    u = model.addVars([(h['hospital_id'], d['day_id']) for h in hospitals for d in h['days']], vtype=GRB.BINARY, name='u')
    y = model.addVars([(h['hospital_id'], d['day_id'], or_index) for h in hospitals for d in h['days'] for or_index in range(num_ors_per_hospital)], vtype=GRB.BINARY, name='y')
    x = model.addVars([(h['hospital_id'], d['day_id'], p['patient_id'], or_index) for h in hospitals for d in h['days'] for p in patients for or_index in range(num_ors_per_hospital)], vtype=GRB.BINARY, name='x')
    w = model.addVars(non_mandatory_patients, vtype=GRB.BINARY, name='w')
    z = model.addVars([(s['scenario_id'], p['patient_id'], h['hospital_id'], d['day_id'], or_index) for s in scenarios for p in patients for h in hospitals for d in h['days'] for or_index in range(num_ors_per_hospital)], vtype=GRB.BINARY, name='z')

    # Objective
    suite_opening_cost = quicksum(h['days'][d['day_id']]['G_hd'] * u[h['hospital_id'], d['day_id']] for h in hospitals for d in h['days'])
    or_opening_cost = quicksum(h['days'][d['day_id']]['F_hd'] * y[h['hospital_id'], d['day_id'], or_index] for h in hospitals for d in h['days'] for or_index in range(num_ors_per_hospital))
    scheduling_benefit = quicksum(next((p['c_sched_per_day'][str(d['day_label'])] for d_label, d_value in p['c_sched_per_day'].items() if int(d_label) == d['day_label']), 0) * x[h['hospital_id'], d['day_id'], p['patient_id'], or_index] for h in hospitals for d in h['days'] for p in patients for or_index in range(num_ors_per_hospital))
    postponement_penalty = quicksum(p['c_unsched'] * w[p['patient_id']] for p in patients if not p['is_mandatory'])
    cancellation_cost = (1/num_scenarios) * quicksum(quicksum(p['c_cancel'] * (x[h['hospital_id'], d['day_id'], p['patient_id'], or_index] - z[s['scenario_id'], p['patient_id'], h['hospital_id'], d['day_id'], or_index]) for h in hospitals for d in h['days'] for or_index in range(num_ors_per_hospital)) for s in scenarios for p in patients)

    obj = suite_opening_cost + or_opening_cost - kappa_1 * scheduling_benefit + kappa_2 * postponement_penalty + cancellation_cost
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    for p in patients:
        patient_id = p['patient_id']
        if p['is_mandatory']:
            model.addConstr(quicksum(x[h['hospital_id'], d['day_id'], patient_id, or_index] for h in hospitals for d in h['days'] for or_index in range(num_ors_per_hospital)) == 1)
        else:
            model.addConstr(quicksum(x[h['hospital_id'], d['day_id'], patient_id, or_index] for h in hospitals for d in h['days'] for or_index in range(num_ors_per_hospital)) + w[patient_id] == 1)

    for h in hospitals:
        for d in h['days']:
            for or_index in range(num_ors_per_hospital):
                model.addConstr(y[h['hospital_id'], d['day_id'], or_index] <= u[h['hospital_id'], d['day_id']])

    for h in hospitals:
        for d in h['days']:
            for or_index in range(num_ors_per_hospital):
                model.addConstr(quicksum(x[h['hospital_id'], d['day_id'], p['patient_id'], or_index] for p in patients) <= y[h['hospital_id'], d['day_id'], or_index])

    for s in scenarios:
        for h in hospitals:
            for d in h['days']:
                for or_index in range(num_ors_per_hospital):
                    model.addConstr(quicksum(s['surgery_durations_minutes'][p['patient_id']] * z[s['scenario_id'], p['patient_id'], h['hospital_id'], d['day_id'], or_index] for p in patients) <= d['B_hd'] * y[h['hospital_id'], d['day_id'], or_index])

    for s in scenarios:
        for p in patients:
            for h in hospitals:
                for d in h['days']:
                    for or_index in range(num_ors_per_hospital):
                        model.addConstr(z[s['scenario_id'], p['patient_id'], h['hospital_id'], d['day_id'], or_index] <= x[h['hospital_id'], d['day_id'], p['patient_id'], or_index])

    model.optimize()

    logger = SolutionLogger(log_path, sense='minimize') if log_path else None
    if model.SolCount > 0:
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            if logger:
                logger.log(model.PoolObjVal)

    if model.SolCount > 0:
        model.Params.SolutionNumber = 0
        solution = {
            'objective_value': model.ObjVal,
            'u': {(h['hospital_id'], d['day_id']): int(u[h['hospital_id'], d['day_id']].X) for h in hospitals for d in h['days']},
            'y': {(h['hospital_id'], d['day_id'], or_index): int(y[h['hospital_id'], d['day_id'], or_index].X) for h in hospitals for d in h['days'] for or_index in range(num_ors_per_hospital)},
            'x': {(h['hospital_id'], d['day_id'], p['patient_id'], or_index): int(x[h['hospital_id'], d['day_id'], p['patient_id'], or_index].X) for h in hospitals for d in h['days'] for p in patients for or_index in range(num_ors_per_hospital) if x[h['hospital_id'], d['day_id'], p['patient_id'], or_index].X > 0.5},
            'w': {p: int(w[p].X) for p in non_mandatory_patients}
        }
    else:
        solution = {
            'objective_value': float('inf'),
            'u': {},
            'y': {},
            'x': {},
            'w': {}
        }

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()