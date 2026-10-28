import argparse
import json
import math
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vessels = instance['num_vessels']
    num_berths = instance['num_berths']
    time_horizon_hours = instance['time_horizon_hours']
    total_time_steps = instance['total_time_steps']
    c1 = instance['c1']
    c2 = instance['c2']
    vessels = instance['vessels']
    berths = instance['berths']

    for vessel in vessels:
        duration_hours = vessel['duration_hours']
        if time_horizon_hours == 0:
            duration_time_steps = 0
        else:
            duration_time_steps = math.ceil(duration_hours * total_time_steps / time_horizon_hours)
        vessel['duration_time_steps'] = duration_time_steps

    vessels_by_k_bar = [[] for _ in range(num_berths)]
    for vessel in vessels:
        k_bar = vessel['k_bar']
        if 0 <= k_bar < num_berths:
            vessels_by_k_bar[k_bar].append(vessel)

    for k in range(num_berths):
        vessels_by_k_bar[k] = sorted(vessels_by_k_bar[k], key=lambda x: x['a'])

    assignment = {}
    for k in range(num_berths):
        current_time = berths[k]['a_k']
        for vessel in vessels_by_k_bar[k]:
            start_time = max(current_time, vessel['a'])
            end_time = start_time + vessel['duration_time_steps'] - 1
            assignment[vessel['id']] = {
                'berth': k,
                'start_time': start_time,
                'end_time': end_time
            }
            current_time = end_time + 1

    for vessel in vessels:
        if vessel['id'] not in assignment:
            start_time = max(0, vessel['a'])
            end_time = start_time + vessel['duration_time_steps'] - 1
            assignment[vessel['id']] = {
                'berth': 0,
                'start_time': start_time,
                'end_time': end_time
            }

    total_cost = 0
    solution_vessels = []
    for vessel in vessels:
        sol = assignment[vessel['id']]
        berth_dev = abs(sol['berth'] - vessel['k_bar'])
        time_dev = abs(sol['start_time'] - vessel['t_bar'])
        cost = c1 * berth_dev + c2 * time_dev
        total_cost += cost
        solution_vessels.append({
            'id': vessel['id'],
            'berth': sol['berth'],
            'start_time': sol['start_time'],
            'end_time': sol['end_time'],
            'profile': 0
        })

    solution = {
        "objective_value": total_cost,
        "vessels": solution_vessels
    }

    if logger:
        logger.log_solution(total_cost, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()