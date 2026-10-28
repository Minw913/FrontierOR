import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    n = instance['parameters']['n']
    m = instance['parameters']['m']
    time_horizon = instance['time_horizon']
    jobs = instance['jobs']

    model = Model('weighted_tardiness')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    # Variables
    x = model.addVars(n, m, vtype=GRB.BINARY, name='x')
    y = model.addVars(n, n, vtype=GRB.BINARY, name='y')
    start_time = model.addVars(n, vtype=GRB.INTEGER, name='start_time')
    tardiness = model.addVars(n, vtype=GRB.INTEGER, name='tardiness')
    weighted_tardiness = model.addVars(n, vtype=GRB.INTEGER, name='weighted_tardiness')

    # Objective
    model.setObjective(quicksum(weighted_tardiness[job['id'] - 1] for job in jobs), GRB.MINIMIZE)

    # Constraints
    for i in range(n):
        model.addConstr(quicksum(x[i, k] for k in range(m)) == 1, name=f'job_assignment_{i}')
        model.addConstr(tardiness[i] >= start_time[i] + jobs[i]['processing_time'] - jobs[i]['due_date'], name=f'tardiness_def_{i}')
        model.addConstr(weighted_tardiness[i] == jobs[i]['weight'] * tardiness[i], name=f'weighted_tardiness_def_{i}')
        for k in range(m):
            for j in range(n):
                if i != j:
                    model.addConstr(start_time[i] + jobs[i]['processing_time'] <= start_time[j] + time_horizon * (1 - y[i, j]), name=f'precedence_{i}_{j}')
                    model.addConstr(y[i, j] + y[j, i] >= x[i, k] + x[j, k] - 1, name=f'precedence_link_{i}_{j}_{k}')
        model.addConstr(start_time[i] >= 0, name=f'non_negative_start_time_{i}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        solution = []
        job_start_time = {i: start_time[i].X for i in range(n)}
        job_machine = {}
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    job_machine[i] = k
        for machine in range(m):
            jobs_on_machine = sorted([i for i in range(n) if job_machine[i] == machine], key=lambda x: job_start_time[x])
            for idx, job in enumerate(jobs_on_machine):
                job_id = job + 1
                prev_job = jobs_on_machine[idx - 1] + 1 if idx > 0 else 0
                start = int(round(job_start_time[job]))
                completion = start + jobs[job]['processing_time']
                weighted_tard = int(round(jobs[job]['weight'] * max(0, completion - jobs[job]['due_date'])))
                solution.append({
                    'machine_id': machine,
                    'prev_job': prev_job,
                    'job': job_id,
                    'start_time': start,
                    'completion_time': completion,
                    'weighted_tardiness': weighted_tard
                })
        if logger:
            logger.log_solution(objective_value, {'objective_value': objective_value, 'schedule': solution})
        return objective_value, solution
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, solution = solve(instance, args.time_limit, logger)

    if objective_value is not None and solution is not None:
        output = {'objective_value': objective_value, 'schedule': solution}
        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)

if __name__ == '__main__':
    main()