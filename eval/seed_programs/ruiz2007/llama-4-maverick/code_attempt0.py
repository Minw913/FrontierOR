import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    n_jobs = instance['n_jobs']
    n_machines = instance['n_machines']
    processing_times = instance['processing_times']
    no_idle_flags = instance.get('no_idle_flags', [False] * n_machines)

    model = Model('flowshop')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Job permutation variables
    x = model.addVars(n_jobs, n_jobs, vtype=GRB.BINARY, name='x')

    # Completion time variables
    c = model.addVars(n_jobs, n_machines, name='c')

    # Objective: minimize makespan
    makespan = model.addVar(name='makespan')
    model.addConstr(makespan == c[n_jobs-1, n_machines-1], name='makespan_def')
    model.setObjective(makespan, GRB.MINIMIZE)

    # Constraints
    for j in range(n_jobs):
        model.addConstr(quicksum(x[j, k] for k in range(n_jobs)) == 1, name=f'job_assign_{j}')
        model.addConstr(quicksum(x[k, j] for k in range(n_jobs)) == 1, name=f'position_assign_{j}')

    for k in range(n_jobs):
        for m in range(n_machines):
            job_idx = sum(j * x[j, k] for j in range(n_jobs))
            if m == 0:
                model.addConstr(c[k, m] >= processing_times[job_idx][m], name=f'c_{k}_{m}')
            else:
                model.addConstr(c[k, m] >= c[k, m-1] + processing_times[job_idx][m], name=f'c_{k}_{m}_precedence')
            if k > 0:
                model.addConstr(c[k, m] >= c[k-1, m] + processing_times[job_idx][m], name=f'c_{k}_{m}_seq')
                if no_idle_flags[m]:
                    model.addConstr(c[k, m] == c[k-1, m] + processing_times[job_idx][m], name=f'no_idle_{k}_{m}')

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        sequence = [int(sum(j * x[j, k].X for j in range(n_jobs))) for k in range(n_jobs)]
        completion_times = {f'machine_{m}': [c[k, m].X for k in range(n_jobs)] for m in range(n_machines)}
        solution = {
            'objective_value': makespan.X,
            'sequence': sequence,
            'completion_times': completion_times
        }
        if logger:
            logger.log(makespan.X)
        return solution
    else:
        raise Exception(f'Gurobi model failed to find a solution within the time limit. Status: {model.status}')

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None
    solution = solve(instance, args.time_limit, logger)
    write_solution(args.solution_path, solution)

if __name__ == '__main__':
    main()