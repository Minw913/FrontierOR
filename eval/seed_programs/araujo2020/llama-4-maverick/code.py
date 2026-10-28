import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def calculate_critical_path(instance):
    project_critical_paths = {}
    for project in instance['projects']:
        project_id = project['project_id']
        project_critical_paths[project_id] = 0  
    return project_critical_paths

def calculate_earliest_start(instance, project_critical_paths):
    job_earliest_start = {}
    job_latest_start = {}
    for job in instance['jobs']:
        job_id = job['job_id']
        project_id = job['project_id']
        release_date = next(p['release_date'] for p in instance['projects'] if p['project_id'] == project_id)
        job_earliest_start[job_id] = release_date  
        job_latest_start[job_id] = 1000  # arbitrary large number
    return job_earliest_start, job_latest_start

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
    job_earliest_start, job_latest_start = calculate_earliest_start(instance, project_critical_paths)

    model = Model()
    model.setParam("Threads", 1)
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

    project_delays = []
    for project in instance['projects']:
        project_id = project['project_id']
        artificial_sink_job_id = project['artificial_sink_job_id']
        critical_path = project_critical_paths[project_id]
        release_date = project['release_date']
        delay_var = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(delay_var >= quicksum([job_vars.get(f'x_{artificial_sink_job_id}_{mode["mode_id"]}_{start_time}', 0) * start_time for mode in next((job['modes'] for job in instance['jobs'] if job['job_id'] == artificial_sink_job_id), []) for start_time in range(job_earliest_start.get(artificial_sink_job_id, 0), job_latest_start.get(artificial_sink_job_id, 0) + 1)]) - release_date - critical_path)
        project_delays.append(delay_var)

    makespan_var = model.addVar(vtype=GRB.CONTINUOUS, name='makespan')
    model.addConstr(makespan_var >= quicksum([job_vars.get(f'x_{project["artificial_sink_job_id"]}_{mode["mode_id"]}_{start_time}', 0) * (start_time + next((m['duration'] for m in job['modes'] if m['mode_id'] == mode["mode_id"]), 0)) for project in instance['projects'] for mode in next((job['modes'] for job in instance['jobs'] if job['job_id'] == project['artificial_sink_job_id']), []) for start_time in range(job_earliest_start.get(project['artificial_sink_job_id'], 0), job_latest_start.get(project['artificial_sink_job_id'], 0) + 1)]))

    model.setObjective(quicksum(project_delays) + 0.001 * makespan_var, GRB.MINIMIZE)

    for job in instance['jobs']:
        job_id = job['job_id']
        model.addConstr(quicksum([job_vars.get(f'x_{job_id}_{mode["mode_id"]}_{start_time}', 0) for mode in job['modes'] for start_time in range(job_earliest_start[job_id], job_latest_start[job_id] + 1)]) == 1)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {'schedule': []}
        for job in instance['jobs']:
            job_id = job['job_id']
            for mode in job['modes']:
                mode_id = mode['mode_id']
                for start_time in range(job_earliest_start[job_id], job_latest_start[job_id] + 1):
                    var_name = f'x_{job_id}_{mode_id}_{start_time}'
                    if var_name in job_vars and job_vars[var_name].x > 0.5:
                        solution['schedule'].append({'job_id': job_id, 'mode_id': mode_id, 'start_time': start_time})

        solution['makespan'] = int(makespan_var.x)
        solution['objective_value'] = model.objVal

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

        if logger:
            logger.log(model.objVal)

if __name__ == '__main__':
    main()