import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_instance(instance):
    P = instance['num_patients_P']
    L = instance['provider_service_hours_L_minutes']
    patients = instance['patients']
    cost_structures = instance['cost_structures']
    cost_structure_name = list(cost_structures.keys())[0]
    c_w = cost_structures[cost_structure_name]['c_w']
    c_g = cost_structures[cost_structure_name]['c_g']
    c_o = cost_structures[cost_structure_name]['c_o']

    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Variables
    x = model.addVars(P, P, vtype=GRB.BINARY, name='x')
    t = model.addVars(P, vtype=GRB.CONTINUOUS, name='t')
    model.addConstrs((x.sum(p, '*') == 1 for p in range(P)), name='assign_patient')
    model.addConstrs((x.sum('*', i) == 1 for i in range(P)), name='assign_position')
    model.addConstr(t[0] == 0, name='start_time_zero')
    model.addConstrs((t[i] >= t[i-1] for i in range(1, P)), name='non_decreasing_start_time')
    model.addConstrs((t[i] <= L for i in range(P)), name='start_time_within_session')

    # Calculate expected costs
    expected_durations = [patients[p]['mean_duration_adequate_prep'] * patients[p]['mean_prep_adequacy'] + 
                          patients[p]['mean_duration_inadequate_prep'] * (1 - patients[p]['mean_prep_adequacy']) for p in range(P)]
    expected_arrival_deviations = [patients[p]['mean_arrival_time_deviation'] for p in range(P)]

    # Auxiliary variables and constraints for expected waiting, idle, and overtime costs
    # This part is simplified and might need adjustments based on the exact DRO formulation required
    waiting = model.addVars(P, vtype=GRB.CONTINUOUS, name='waiting')
    idle = model.addVars(P-1, vtype=GRB.CONTINUOUS, name='idle')
    overtime = model.addVar(vtype=GRB.CONTINUOUS, name='overtime')

    model.addConstrs((waiting[i] >= gp.quicksum(x[i, p] * expected_arrival_deviations[p] for p in range(P)) for i in range(P)), name='waiting_time')
    model.addConstrs((idle[i-1] >= t[i] - t[i-1] - gp.quicksum(x[i-1, p] * expected_durations[p] for p in range(P)) for i in range(1, P)), name='idle_time')
    model.addConstr(overtime >= gp.quicksum(x[P-1, p] * expected_durations[p] for p in range(P)) + t[P-1] - L, name='overtime')

    # Objective: Minimize worst-case expected total operational cost
    objective = gp.quicksum(c_w * waiting[i] for i in range(P)) + gp.quicksum(c_g * idle[i] for i in range(P-1)) + c_o * overtime
    model.setObjective(objective, GRB.MINIMIZE)

    return model, x, t, cost_structure_name, c_w, c_g, c_o, patients

def extract_solution(model, x, t, cost_structure_name, c_w, c_g, c_o, patients):
    P = len(patients)
    assignment = {}
    schedule = {}
    patient_start_times = {}

    for p in range(P):
        for i in range(P):
            if x[i, p].X > 0.5:
                assignment[str(patients[p]['patient_index'])] = i + 1
                patient_start_times[str(patients[p]['patient_index'])] = t[i].X

    for i in range(P):
        schedule[str(i+1)] = t[i].X

    solution = {
        'objective_value': model.ObjVal,
        'cost_structure': cost_structure_name,
        'c_w': c_w,
        'c_g': c_g,
        'c_o': c_o,
        'assignment': assignment,
        'schedule': schedule,
        'patient_start_times': patient_start_times,
        'model_variables': {}  # Populate this if necessary
    }

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    model, x, t, cost_structure_name, c_w, c_g, c_o, patients = solve_instance(instance)

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = extract_solution(model, x, t, cost_structure_name, c_w, c_g, c_o, patients)
        if logger:
            logger.log_solution(model.ObjVal, solution)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()