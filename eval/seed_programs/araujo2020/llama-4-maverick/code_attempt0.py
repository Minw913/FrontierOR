import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def calculate_critical_path(instance):
    # Simplified critical path calculation for demonstration
    # Actual implementation should consider job durations and precedence
    project_critical_paths = {}
    for project in instance['projects']:
        project_id = project['project_id']
        # Assuming artificial sink job's duration is 0 and it's the last job
        project_critical_paths[project_id] = 0  # Simplified
    return project_critical_paths

def calculate_earliest_start(instance, project_critical_paths):
    job_earliest_start = {}
    for job in instance['jobs']:
        job_id = job['job_id']
        project_id = job['project_id']
        release_date = next(p['release_date'] for p in instance['projects'] if p['project_id'] == project_id)
        # Simplified calculation, actual should consider predecessors and durations
        job_earliest_start[job_id] = release_date  # Simplified
    return job_earliest_start

def calculate_latest_start(instance, project_critical_paths, job_earliest_start):
    job_latest_start = {}
    for job in instance['jobs']:
        job_id = job['job_id']
        project_id = job['project_id']
        critical_path = project_critical_paths[project_id]
        # Simplified, actual should consider job duration and successors
        job_latest_start[job_id] = job_earliest_start[job_id] + critical_path  # Simplified
    return job_latest_start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    project_critical_paths = calculate_critical_path(instance)
    job_earliest_start = calculate_earliest_start(instance, project_critical_paths)
    job_latest_start = calculate_latest_start(instance, project_critical_paths, job_earliest_start)

    model = Model()
    model.Params.TimeLimit = args.time_limit

    job_vars = {}
    for job in instance['jobs']:
        job_id = job['job_id']
        for mode in job['modes']:
            mode_id = mode['mode_id']
            earliest_start = job_earliest_start[job_id]
            latest_start = job_latest_start[job_id]
            for start_time in range(earliest_start, latest_start + 1):
                var_name = f'x_{job_id}_{mode_id}_{start_time}'
                job_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=var_name)

    # Objective: Minimize total project delay
    project_delays = []
    for project in instance['projects']:
        project_id = project['project_id']
        artificial_sink_job_id = project['artificial_sink_job_id']
        critical_path = project_critical_paths[project_id]
        release_date = project['release_date']
        delay_var = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(delay_var >= quicksum([job_vars[f'x_{artificial_sink_job_id}_{mode["mode_id"]}_{start_time}'] * start_time for mode in next(job['modes'] for job in instance['jobs'] if job['job_id'] == artificial_sink_job_id) for start_time in range(job_earliest_start[artificial_sink_job_id], job_latest_start[artificial_sink_job_id] + 1)]) - release_date - critical_path)
        project_delays.append(delay_var)

    model.setObjective(quicksum(project_delays) + 0.001 * model.addVar(vtype=GRB.CONTINUOUS, name='makespan'), GRB.MINIMIZE)

    # Constraints
    # 1. Each job is assigned exactly one mode and start time
    for job in instance['jobs']:
        job_id = job['job_id']
        model.addConstr(quicksum([job_vars[f'x_{job_id}_{mode["mode_id"]}_{start_time}'] for mode in job['modes'] for start_time in range(job_earliest_start[job_id], job_latest_start[job_id] + 1)]) == 1)

    # 2. Precedence constraints
    for job in instance['jobs']:
        job_id = job['job_id']
        for successor_id in job['successors']:
            for mode in job['modes']:
                mode_id = mode['mode_id']
                duration = mode['duration']
                for start_time in range(job_earliest_start[job_id], job_latest_start[job_id] + 1):
                    var_name = f'x_{job_id}_{mode_id}_{start_time}'
                    model.addConstr(quicksum([job_vars[f'x_{successor_id}_{m["mode_id"]}_{s}'] * s for m in next(j['modes'] for j in instance['jobs'] if j['job_id'] == successor_id) for s in range(job_earliest_start[successor_id], job_latest_start[successor_id] + 1) if s < start_time + duration]) <= 1 - job_vars[var_name])

    # 3. Renewable resource constraints
    renewable_resources = instance['resources']['renewable']
    for t in range(max([project['release_date'] for project in instance['projects']]) + max([project_critical_paths[project['project_id']] for project in instance['projects']])):
        for resource in renewable_resources:
            resource_id = resource['resource_id']
            capacity = resource['capacity']
            model.addConstr(quicksum([job_vars[f'x_{job["job_id"]}_{mode["mode_id"]}_{start_time}'] * mode['renewable_consumption'][resource_id] for job in instance['jobs'] for mode in job['modes'] for start_time in range(max(0, t - mode['duration'] + 1), min(t + 1, job_latest_start[job['job_id']] + 1)) if job_earliest_start[job['job_id']] <= start_time <= job_latest_start[job['job_id']]]) <= capacity)

    # 4. Non-renewable resource constraints
    non_renewable_resources = instance['resources']['nonrenewable']
    for resource in non_renewable_resources:
        resource_id = resource['resource_id']
        capacity = resource['capacity']
        model.addConstr(quicksum([job_vars[f'x_{job["job_id"]}_{mode["mode_id"]}_{start_time}'] * mode['nonrenewable_consumption'][resource_id] for job in instance['jobs'] for mode in job['modes'] for start_time in range(job_earliest_start[job['job_id']], job_latest_start[job['job_id']] + 1)]) <= capacity)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {'schedule': []}
        for job in instance['jobs']:
            job_id = job['job_id']
            for mode in job['modes']:
                mode_id = mode['mode_id']
                for start_time in range(job_earliest_start[job_id], job_latest_start[job_id] + 1):
                    var_name = f'x_{job_id}_{mode_id}_{start_time}'
                    if job_vars[var_name].x > 0.5:
                        solution['schedule'].append({'job_id': job_id, 'mode_id': mode_id, 'start_time': start_time})

        makespan = max([max([start_time + next(job['modes'] for job in instance['jobs'] if job['job_id'] == project['artificial_sink_job_id'])['duration'] for project in instance['projects']]) for start_time in [s['start_time'] for s in solution['schedule'] if s['job_id'] in [project['artificial_sink_job_id'] for project in instance['projects']]]])
        solution['makespan'] = makespan
        solution['objective_value'] = model.objVal

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

        if logger:
            logger.log(model.objVal)

if __name__ == '__main__':
    main()