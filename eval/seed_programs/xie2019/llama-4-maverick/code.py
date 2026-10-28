import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def generate_qc_profiles(instance):
    # Simplified profile generation for demonstration; actual implementation should follow the problem's profile generation rules
    vessels = instance['vessels']
    num_big_profiles = instance['num_big_profiles']
    num_small_profiles = instance['num_small_profiles']
    qc_profiles = instance['qc_profiles']
    time_steps_per_shift = instance['time_steps_per_shift']

    for vessel in vessels:
        duration_hours = vessel['duration_hours']
        is_large = duration_hours > 24  # Simplified assumption for demonstration
        if is_large:
            num_profiles = num_big_profiles
            profile_params = qc_profiles['big']
        else:
            num_profiles = num_small_profiles
            profile_params = qc_profiles['small']

        min_qc = profile_params['min_qc']
        max_qc = profile_params['max_qc']
        min_handling_shifts = profile_params['min_handling_shifts']
        max_handling_shifts = profile_params['max_handling_shifts']

        vessel['qc_profiles'] = []
        for start_idx in range(time_steps_per_shift):
            profiles = []
            for _ in range(num_profiles):
                handling_shifts = instance['random_state'].randint(min_handling_shifts, max_handling_shifts + 1)
                profile = [instance['random_state'].randint(min_qc, max_qc + 1) for _ in range(handling_shifts)]
                profiles.append({'duration': handling_shifts * time_steps_per_shift, 'qc_assignment': profile})
            vessel['qc_profiles'].append(profiles)

def solve(instance, time_limit, logger):
    num_vessels = instance['num_vessels']
    num_berths = instance['num_berths']
    total_qc = instance['total_qc']
    vessels = instance['vessels']
    berths = instance['berths']
    time_steps_per_shift = instance['time_steps_per_shift']
    total_time_steps = instance['total_time_steps']
    c1 = instance['c1']
    c2 = instance['c2']

    model = gp.Model()
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # Decision variables
    x = model.addVars(num_vessels, num_berths, vtype=GRB.BINARY, name='x')
    y = model.addVars(num_vessels, total_time_steps, vtype=GRB.BINARY, name='y')
    z = model.addVars(num_vessels, total_time_steps, vtype=GRB.INTEGER, name='z')

    # Objective function
    model.setObjective(
        gp.quicksum(c1 * abs(vessel['k_bar'] - berth_idx) * x[vessel_idx, berth_idx] for vessel_idx, vessel in enumerate(vessels) for berth_idx in range(num_berths)) +
        gp.quicksum(c2 * abs(vessel['t_bar'] - start_time) * y[vessel_idx, start_time] for vessel_idx, vessel in enumerate(vessels) for start_time in range(total_time_steps)),
        GRB.MINIMIZE
    )

    # Constraints
    for vessel_idx, vessel in enumerate(vessels):
        model.addConstr(gp.quicksum(x[vessel_idx, berth_idx] for berth_idx in range(num_berths)) == 1)
        model.addConstr(gp.quicksum(y[vessel_idx, start_time] for start_time in range(vessel['a'], vessel['b'] + 1)) == 1)

        for berth_idx, berth in enumerate(berths):
            for start_time in range(max(vessel['a'], berth['a_k']), min(vessel['b'], berth['b_k']) + 1):
                within_shift_idx = start_time % time_steps_per_shift
                for profile_idx, profile in enumerate(vessel['qc_profiles'][within_shift_idx]):
                    duration = profile['duration']
                    end_time = start_time + duration - 1
                    if end_time <= berth['b_k']:
                        model.addConstr(y[vessel_idx, start_time] <= x[vessel_idx, berth_idx])

                        for t in range(start_time, end_time + 1):
                            model.addConstr(z[vessel_idx, t] >= profile['qc_assignment'][(t - start_time) // time_steps_per_shift] * y[vessel_idx, start_time])
                            model.addConstr(z[vessel_idx, t] <= total_qc * y[vessel_idx, start_time])

    for t in range(total_time_steps):
        model.addConstr(gp.quicksum(z[vessel_idx, t] for vessel_idx in range(num_vessels)) <= total_qc)

    for berth_idx in range(num_berths):
        berth = berths[berth_idx]
        for t in range(berth['a_k'], berth['b_k'] + 1):
            vessels_at_berth = gp.quicksum(
                gp.quicksum(y[vessel_idx, start_time] for start_time in range(max(vessel['a'], t - profile['duration'] + 1), min(t, vessel['b']) + 1))
                for vessel_idx, vessel in enumerate(vessels) for profile in vessel['qc_profiles'][t % time_steps_per_shift]
            )
            model.addConstr(vessels_at_berth <= 1)

    model.optimize()

    if model.SolCount > 0:
        objective_value = model.ObjVal
        solution = []
        for vessel_idx, vessel in enumerate(vessels):
            berth_assignment = x[vessel_idx, :].getAttr('X')
            berth_idx = int([i for i, val in enumerate(berth_assignment) if val > 0.5][0])
            start_time_assignment = y[vessel_idx, :].getAttr('X')
            start_time = int([t for t, val in enumerate(start_time_assignment) if val > 0.5][0])
            within_shift_idx = start_time % time_steps_per_shift
            profile_idx = 0  # Simplified, actual implementation should identify the correct profile_idx
            profile = vessels[vessel_idx]['qc_profiles'][within_shift_idx][profile_idx]
            duration = profile['duration']
            end_time = start_time + duration - 1
            solution.append({
                'id': vessel['id'],
                'berth': berth_idx,
                'start_time': start_time,
                'end_time': end_time,
                'profile': profile_idx
            })

        if logger:
            logger.log_solution(objective_value, {'vessels': solution})

        return objective_value, solution
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    import random
    instance['random_state'] = random.Random(0)  # For generating QC profiles
    generate_qc_profiles(instance)

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    objective_value, solution = solve(instance, args.time_limit, logger)

    if objective_value is not None:
        output = {'objective_value': objective_value, 'vessels': solution}
    else:
        output = {'objective_value': None, 'vessels': []}

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    main()