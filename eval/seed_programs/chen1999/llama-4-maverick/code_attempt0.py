import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit):
    problem_type = instance['problem_type']
    machine_type = instance['machine_type']
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    if problem_type == 'weighted_tardy_jobs':
        due_dates = instance['jobs']['due_dates']

    model = gp.Model()
    model.setParam('TimeLimit', time_limit)

    if problem_type == 'weighted_completion_time':
        schedule, completion_times = solve_weighted_completion_time(instance, model)
        objective_value = sum(weights[job] * completion_times[job] for job in range(num_jobs))
        output = {
            'objective_value': objective_value,
            'schedule': {f'machine_{m}': [job for job in schedule[m]] for m in range(num_machines) if schedule[m]},
            'completion_times': {f'job_{job}': completion_times[job] for job in range(num_jobs)}
        }
    else:
        schedule, tardy_jobs, on_time_jobs, objective_value = solve_weighted_tardy_jobs(instance, model)
        output = {
            'objective_value': objective_value,
            'schedule': {f'machine_{m}': [job for job in schedule[m]] for m in range(num_machines) if schedule[m]},
            'tardy_jobs': [int(job) for job in tardy_jobs],
            'on_time_jobs': [int(job) for job in on_time_jobs]
        }
    return output

def solve_weighted_completion_time(instance, model):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']

    job_vars = model.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name='job_assignment')
    seq_vars = model.addVars(num_jobs, num_jobs, num_machines, vtype=GRB.BINARY, name='job_sequencing')
    completion_time_vars = model.addVars(num_jobs, name='completion_times')

    model.addConstrs((job_vars.sum(job, '*') == 1 for job in range(num_jobs)), name='assign_job_to_one_machine')
    model.addConstrs((seq_vars[job1, job2, machine] + seq_vars[job2, job1, machine] <= 1 for job1 in range(num_jobs) for job2 in range(num_jobs) if job1 != job2 for machine in range(num_machines)), name='sequencing_consistency')
    model.addConstrs((seq_vars[job1, job2, machine] >= job_vars[job1, machine] + job_vars[job2, machine] - 1 for job1 in range(num_jobs) for job2 in range(num_jobs) if job1 != job2 for machine in range(num_machines)), name='sequencing_dependency')
    model.addConstrs((completion_time_vars[job1] >= completion_time_vars[job2] + processing_times[job1][machine] - (1 - seq_vars[job2, job1, machine]) * (sum(processing_times[job][machine] for job in range(num_jobs)) + 1) for job1 in range(num_jobs) for job2 in range(num_jobs) if job1 != job2 for machine in range(num_machines)), name='completion_time_constraint1')
    model.addConstrs((completion_time_vars[job] >= processing_times[job][machine] - (1 - job_vars[job, machine]) * (sum(processing_times[j][machine] for j in range(num_jobs)) + 1) for job in range(num_jobs) for machine in range(num_machines)), name='completion_time_constraint2')

    model.setObjective(sum(weights[job] * completion_time_vars[job] for job in range(num_jobs)), GRB.MINIMIZE)

    model.optimize()

    schedule = [[] for _ in range(num_machines)]
    completion_times = {}
    for job in range(num_jobs):
        for machine in range(num_machines):
            if job_vars[job, machine].x > 0.5:
                schedule[machine].append(job)
                completion_times[job] = completion_time_vars[job].x
    return schedule, completion_times

def solve_weighted_tardy_jobs(instance, model):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    due_dates = instance['jobs']['due_dates']

    job_vars = model.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name='job_assignment')
    seq_vars = model.addVars(num_jobs, num_jobs, num_machines, vtype=GRB.BINARY, name='job_sequencing')
    tardy_vars = model.addVars(num_jobs, vtype=GRB.BINARY, name='tardy_jobs')
    completion_time_vars = model.addVars(num_jobs, name='completion_times')

    model.addConstrs((job_vars.sum(job, '*') == 1 - tardy_vars[job] for job in range(num_jobs)), name='assign_job_to_one_machine_or_tardy')
    model.addConstrs((seq_vars[job1, job2, machine] + seq_vars[job2, job1, machine] <= 1 for job1 in range(num_jobs) for job2 in range(num_jobs) if job1 != job2 for machine in range(num_machines)), name='sequencing_consistency')
    model.addConstrs((seq_vars[job1, job2, machine] >= job_vars[job1, machine] + job_vars[job2, machine] - 1 for job1 in range(num_jobs) for job2 in range(num_jobs) if job1 != job2 for machine in range(num_machines)), name='sequencing_dependency')
    model.addConstrs((completion_time_vars[job1] >= completion_time_vars[job2] + processing_times[job1][machine] - (1 - seq_vars[job2, job1, machine]) * (sum(processing_times[job][machine] for job in range(num_jobs)) + 1) for job1 in range(num_jobs) for job2 in range(num_jobs) if job1 != job2 for machine in range(num_machines)), name='completion_time_constraint1')
    model.addConstrs((completion_time_vars[job] >= processing_times[job][machine] - (1 - job_vars[job, machine]) * (sum(processing_times[j][machine] for j in range(num_jobs)) + 1) for job in range(num_jobs) for machine in range(num_machines)), name='completion_time_constraint2')
    model.addConstrs((completion_time_vars[job] <= due_dates[job] + tardy_vars[job] * (sum(processing_times[j][machine] for j in range(num_jobs) for machine in range(num_machines)) + 1) for job in range(num_jobs)), name='due_date_constraint')

    model.setObjective(sum(weights[job] * tardy_vars[job] for job in range(num_jobs)), GRB.MINIMIZE)

    model.optimize()

    schedule = [[] for _ in range(num_machines)]
    tardy_jobs = set()
    on_time_jobs = set()
    for job in range(num_jobs):
        if tardy_vars[job].x > 0.5:
            tardy_jobs.add(job)
        else:
            on_time_jobs.add(job)
            for machine in range(num_machines):
                if job_vars[job, machine].x > 0.5:
                    schedule[machine].append(job)
    objective_value = sum(weights[job] for job in tardy_jobs)
    return schedule, tardy_jobs, on_time_jobs, objective_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    output = solve_instance(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=4)

    if logger:
        logger.log(output['objective_value'])

if __name__ == '__main__':
    main()