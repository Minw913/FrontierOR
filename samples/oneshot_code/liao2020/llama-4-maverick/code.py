import argparse
import json
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def calculate_objective(schedule, instance):
    P = instance['num_patients_P']
    L = instance['provider_service_hours_L_minutes']
    patients = instance['patients']
    cost_structures = list(instance['cost_structures'].values())[0]
    c_w = cost_structures['c_w']
    c_g = cost_structures['c_g']
    c_o = cost_structures['c_o']

    total_cost = 0
    actual_start_time = 0
    for i in range(P):
        p = patients[i]
        mean_duration_adequate = p['mean_duration_adequate_prep']
        mean_duration_inadequate = p['mean_duration_inadequate_prep']
        prob_adequate = p['mean_prep_adequacy']
        mean_duration = mean_duration_adequate * prob_adequate + mean_duration_inadequate * (1 - prob_adequate)
        mean_arrival_deviation = p['mean_arrival_time_deviation']

        arrival_time = schedule[i] + mean_arrival_deviation
        actual_start_time = max(actual_start_time, arrival_time)
        waiting_time = max(0, actual_start_time - schedule[i])
        total_cost += c_w * waiting_time

        if i < P - 1:
            idle_time = max(0, schedule[i+1] - (actual_start_time + mean_duration))
            total_cost += c_g * idle_time

        actual_start_time += mean_duration

    overtime = max(0, actual_start_time - L)
    total_cost += c_o * overtime

    return total_cost

def solve_instance(instance, time_limit, logger):
    P = instance['num_patients_P']
    L = instance['provider_service_hours_L_minutes']

    # Initial guess for schedule
    initial_schedule = [i * L / (P - 1) if P > 1 else 0 for i in range(P)]

    # Define bounds for schedule
    bounds = [(0, L) for _ in range(P)]
    bounds[0] = (0, 0)  # First appointment starts at 0

    def objective(schedule):
        return calculate_objective(schedule, instance)

    best_objective = float('inf')
    best_schedule = None

    # Simple iterative improvement
    current_schedule = initial_schedule
    for _ in range(1000):  # Arbitrary number of iterations
        new_schedule = current_schedule[:]
        for i in range(1, P):
            new_schedule[i] = min(L, max(0, new_schedule[i] + (i % 2 * 2 - 1) * 0.1))
        new_objective = objective(new_schedule)
        if new_objective < best_objective:
            best_objective = new_objective
            best_schedule = new_schedule
            if logger:
                logger.log(best_objective)
        current_schedule = new_schedule

    solution = {}
    solution['objective_value'] = best_objective

    schedule_sol = {}
    patient_start_times = {}
    assignment_sol = {}
    for i in range(P):
        schedule_sol[i + 1] = best_schedule[i]
        patient_start_times[i + 1] = best_schedule[i]
        assignment_sol[instance['patients'][i]['patient_index']] = i + 1

    solution['assignment'] = {str(k): v for k, v in assignment_sol.items()}
    solution['schedule'] = schedule_sol
    solution['patient_start_times'] = patient_start_times

    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()