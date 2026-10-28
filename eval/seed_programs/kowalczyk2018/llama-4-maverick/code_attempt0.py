import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, log_path):
    n = instance['n']
    m = instance['m']
    jobs = instance['jobs']

    # Sort jobs by weight-to-processing-time ratio in descending order
    jobs.sort(key=lambda x: x['w'] / x['p'], reverse=True)

    model = gp.Model('parallel_machines')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False

    # Variables: x[j][k] = 1 if job j is assigned to machine k, 0 otherwise
    x = [[model.addVar(vtype=GRB.BINARY, name=f'x_{j}_{k}') for k in range(m)] for j in range(n)]

    # Objective: minimize total weighted completion time
    model.setObjective(gp.quicksum(jobs[j]['w'] * gp.quicksum(x[i][k] * (sum(jobs[l]['p'] for l in range(i+1)) + jobs[i]['p']/2) for k in range(m)) for i, j in enumerate(jobs)), GRB.MINIMIZE)

    # Constraints: each job is assigned to exactly one machine
    for j in range(n):
        model.addConstr(gp.quicksum(x[j][k] for k in range(m)) == 1, name=f'job_assignment_{j}')

    # Solve the model
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        machines = []
        objective_value = model.objVal
        x_sol = [[x[j][k].x for k in range(m)] for j in range(n)]

        for k in range(m):
            machine_jobs = []
            completion_time = 0
            for j in range(n):
                if x_sol[j][k] > 0.5:
                    completion_time += jobs[j]['p']
                    machine_jobs.append({
                        'j_orig': jobs[j]['j'],
                        'p': jobs[j]['p'],
                        'w': jobs[j]['w'],
                        'C_j': completion_time
                    })
            schedule_cost = sum(job['w'] * job['C_j'] for job in machine_jobs)
            machines.append({'schedule_cost': schedule_cost, 'jobs': machine_jobs})

        return objective_value, machines
    else:
        raise Exception(f"Model status: {model.status}")

    return None, None

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