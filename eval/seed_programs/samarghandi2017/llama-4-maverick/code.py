import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    processing_times = instance['processing_times']
    due_dates = instance['due_dates']

    model = Model('no_wait_flow_shop')
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Decision variables
    x = model.addVars(n_jobs, n_jobs, vtype=GRB.BINARY, name='x')
    start = model.addVars(n_jobs, n_machines, name='start')

    # Objective: minimize makespan
    makespan = model.addVar(name='makespan')
    model.setObjective(makespan, GRB.MINIMIZE)

    # Constraints
    for i in range(n_jobs):
        for k in range(n_machines - 1):
            model.addConstr(start[i, k + 1] == start[i, k] + processing_times[k][i], name=f'no_wait_{i}_{k}')

    for i in range(n_jobs):
        model.addConstr(start[i, n_machines - 1] + processing_times[n_machines - 1][i] <= due_dates[i], name=f'due_date_{i}')

    for i in range(n_jobs):
        for j in range(n_jobs):
            if i != j:
                M = 1e6
                for k in range(n_machines):
                    model.addConstr(start[j, k] >= start[i, k] + processing_times[k][i] - M * (1 - x[i, j]), name=f'precedence_{i}_{j}_{k}')
                    model.addConstr(start[i, k] >= start[j, k] + processing_times[k][j] - M * x[i, j], name=f'precedence_{j}_{i}_{k}')

    for i in range(n_jobs):
        model.addConstr(quicksum(x[i, j] for j in range(n_jobs) if i != j) <= 1, name=f'at_most_one_successor_{i}')
        model.addConstr(quicksum(x[j, i] for j in range(n_jobs) if i != j) <= 1, name=f'at_most_one_predecessor_{i}')

    model.addConstr(quicksum(x[i, j] for i in range(n_jobs) for j in range(n_jobs) if i != j) == n_jobs - 1, name='single_chain')

    for i in range(n_jobs):
        model.addConstr(makespan >= start[i, n_machines - 1] + processing_times[n_machines - 1][i], name=f'makespan_def_{i}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        sequence = []
        job = None
        for i in range(n_jobs):
            if sum(x[i, j].X for j in range(n_jobs) if i != j) == 1 and sum(x[j, i].X for j in range(n_jobs) if i != j) == 0:
                job = i
                sequence.append(job + 1)
                break
        while len(sequence) < n_jobs:
            for j in range(n_jobs):
                if x[job, j].X > 0.5:
                    job = j
                    sequence.append(job + 1)
                    break

        if model.getAttr('SolCount') > 0:
            objective_value = makespan.X if makespan.X > 0 else max(start[i, n_machines - 1].X + processing_times[n_machines - 1][i] for i in range(n_jobs))
        else:
            objective_value = float('inf')

        if logger and model.getAttr('SolCount') > 0:
            logger.log(objective_value)

        return objective_value, sequence
    else:
        raise Exception(f"Model status: {model.status}")

def write_solution(solution_path, objective_value, sequence):
    solution = {
        'objective_value': objective_value,
        'sequence': sequence
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    try:
        objective_value, sequence = solve_instance(instance, args.time_limit, logger)
        write_solution(args.solution_path, objective_value, sequence)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()