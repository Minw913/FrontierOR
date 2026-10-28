import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    batch_capacity = instance['batch_capacity']
    jobs = instance['jobs']

    model = Model('Batch Processing')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Job attributes
    job_release_date = {job['job_id']: job['release_date'] for job in jobs}
    job_processing_time = {job['job_id']: job['processing_time'] for job in jobs}
    job_size = {job['job_id']: job['size'] for job in jobs}

    # Decision variables
    x = model.addVars(num_jobs, num_machines, num_jobs, vtype=GRB.BINARY, name='x')
    start_time = model.addVars(num_machines, num_jobs, name='start_time')
    processing_time = model.addVars(num_machines, num_jobs, name='processing_time')
    completion_time = model.addVars(num_jobs, name='completion_time')

    # Constraints
    for j in range(num_jobs):
        job_id = jobs[j]['job_id']
        model.addConstr(quicksum(x[j, m, b] for m in range(num_machines) for b in range(num_jobs)) == 1, name=f'job_assignment_{job_id}')

    for m in range(num_machines):
        for b in range(num_jobs - 1):
            model.addConstr(start_time[m, b + 1] >= start_time[m, b] + processing_time[m, b], name=f'batch_sequence_{m}_{b}')

    for m in range(num_machines):
        for b in range(num_jobs):
            model.addConstr(quicksum(job_size[jobs[j]['job_id']] * x[j, m, b] for j in range(num_jobs)) <= batch_capacity, name=f'batch_capacity_{m}_{b}')
            model.addConstr(processing_time[m, b] >= quicksum(job_processing_time[jobs[j]['job_id']] * x[j, m, b] for j in range(num_jobs)), name=f'batch_processing_time_{m}_{b}')
            model.addConstr(start_time[m, b] >= quicksum(job_release_date[jobs[j]['job_id']] * x[j, m, b] for j in range(num_jobs)), name=f'batch_start_time_{m}_{b}')

    for j in range(num_jobs):
        job_id = jobs[j]['job_id']
        model.addConstr(completion_time[j] == quicksum(start_time[m, b] + processing_time[m, b] for m in range(num_machines) for b in range(num_jobs) if x[j, m, b].x == 1), name=f'completion_time_{job_id}')

    # Objective function
    model.setObjective(quicksum(completion_time[j] for j in range(num_jobs)), GRB.MINIMIZE)

    # Solve the model
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        # Extract solution
        batches = []
        job_assignments = {}
        for m in range(num_machines):
            for b in range(num_jobs):
                if any(x[j, m, b].x > 0.5 for j in range(num_jobs)):
                    batch_jobs = [jobs[j]['job_id'] for j in range(num_jobs) if x[j, m, b].x > 0.5]
                    batch = {
                        'batch_id': b + 1,
                        'machine': m + 1,
                        'start_time': start_time[m, b].x,
                        'processing_time': processing_time[m, b].x,
                        'jobs': batch_jobs
                    }
                    batches.append(batch)
                    for job_id in batch_jobs:
                        job_assignments[str(job_id)] = {
                            'batch': b + 1,
                            'machine': m + 1,
                            'completion_time': start_time[m, b].x + processing_time[m, b].x
                        }

        solution = {
            'objective_value': objective_value,
            'batches': batches,
            'job_assignments': job_assignments
        }
        return solution
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, type=str)
    parser.add_argument('--solution_path', required=True, type=str)
    parser.add_argument('--time_limit', required=True, type=int)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)
    write_solution(args.solution_path, solution)

if __name__ == '__main__':
    main()