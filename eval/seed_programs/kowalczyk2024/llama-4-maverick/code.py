import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import sys

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    return instance_data

def solve_instance(instance_data, time_limit, logger):
    n = instance_data['n']
    m = instance_data['m']
    jobs = instance_data['jobs']
    jobs_dict = {job['id']: job for job in jobs}

    model = Model('parallel_machines_scheduling')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  

    job_ids = [job['id'] for job in jobs]
    x = model.addVars(job_ids, range(m), vtype=GRB.BINARY, name='x')  
    C = model.addVars(job_ids, name='C')  
    T = model.addVars(job_ids, name='T')  

    model.setObjective(quicksum(job['w'] * T[job['id']] for job in jobs), GRB.MINIMIZE)

    for job in jobs:
        j = job['id']
        model.addConstr(quicksum(x[j, k] for k in range(m)) == 1, f'assign_{j}')
        model.addConstr(T[j] >= C[j] - job['d'], f'tardiness_{j}')
        model.addConstr(C[j] >= job['p'], f'completion_{j}')

    for k in range(m):
        model.addConstrs((C[j] >= C[i] + job['p'] - instance_data['T'] * (1 - x[i, k] + x[j, k]) 
                          for i in job_ids for j in job_ids if i != j), name=f'precedence_{k}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        schedule = {}
        for k in range(m):
            machine_time = 0
            job_list = sorted([j for j in job_ids if x[j, k].x > 0.5], key=lambda j: C[j].x)
            for j in job_list:
                schedule[str(j)] = machine_time
                machine_time += jobs_dict[j]['p']

        solution = {'objective_value': objective_value, 'schedule': schedule}
        return solution
    else:
        print(f"Model status: {model.status}", file=sys.stderr)
        return None

def write_solution(solution_path, solution):
    if solution is not None:
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No solution to write.", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Solve parallel machines scheduling problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance_data = read_instance(args.instance_path)
    solution = solve_instance(instance_data, args.time_limit, logger)
    write_solution(args.solution_path, solution)

if __name__ == '__main__':
    main()