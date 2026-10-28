import json
import argparse
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, log_path):
    n = instance['n']
    m = instance['m']
    jobs = instance['jobs']

    # Sort jobs by weight-to-processing-time ratio
    jobs.sort(key=lambda x: x['w'] / x['p'], reverse=True)

    # Assign jobs to machines in order of their weight-to-processing-time ratio
    machines = [{'jobs': [], 'load': 0} for _ in range(m)]
    for j, job in enumerate(jobs):
        min_machine = min(machines, key=lambda x: x['load'])
        min_machine['jobs'].append({'j_orig': job['j'], 'p': job['p'], 'w': job['w'], 'C_j': min_machine['load'] + job['p']})
        min_machine['load'] += job['p']

    # Calculate objective value
    objective_value = sum(sum(job['w'] * job['C_j'] for job in machine['jobs']) for machine in machines)

    # Format machines
    machines = [{'schedule_cost': sum(job['w'] * job['C_j'] for job in machine['jobs']), 'jobs': machine['jobs']} for machine in machines]

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    if logger:
        logger.log(objective_value)

    return objective_value, machines

def write_solution(solution_path, objective_value, machines):
    solution = {
        'objective_value': objective_value,
        'machines': machines
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, machines = solve_instance(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, machines)

if __name__ == '__main__':
    main()