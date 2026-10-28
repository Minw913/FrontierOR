import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance_data = json.load(f)
    return instance_data

def solve_instance(instance_data, time_limit, logger):
    n = instance_data['n']
    m = instance_data['m']
    jobs = instance_data['jobs']

    model = Model('parallel_machines_scheduling')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Variables
    x = model.addVars(n, m, vtype=GRB.BINARY, name='x')  # Job assignment
    y = model.addVars(n, n, vtype=GRB.BINARY, name='y')  # Job order on machines
    C = model.addVars(n, name='C')  # Completion times
    T = model.addVars(n, name='T')  # Tardiness
    S = model.addVars(n, name='S')  # Start times

    # Objective: Minimize total weighted tardiness
    model.setObjective(quicksum(job['w'] * T[job['id']] for job in jobs), GRB.MINIMIZE)

    # Constraints
    for job in jobs:
        j = job['id']
        # Each job is assigned to exactly one machine
        model.addConstr(quicksum(x[j, k] for k in range(m)) == 1, f'assign_{j}')

        # Tardiness calculation
        model.addConstr(T[j] >= C[j] - job['d'], f'tardiness_{j}')

        # Start and completion time calculation based on assignment and order
        for k in range(m):
            model.addConstr(C[j] >= S[j] + job['p'], f'completion_{j}')
            model.addConstr(S[j] <= job['p'] * (1 - x[j, k]) + quicksum(jobs[i]['p'] * y[i, j] for i in jobs if i['id'] != j), f'start_upper_{j}_{k}')
            model.addConstr(S[j] >= -job['p'] * (1 - x[j, k]) + quicksum(jobs[i]['p'] * y[i, j] for i in jobs if i['id'] != j), f'start_lower_{j}_{k}')

    # Ensure that for any two jobs on the same machine, one precedes the other
    for i in jobs:
        for j in jobs:
            if i['id'] != j['id']:
                for k in range(m):
                    model.addConstr(y[i['id'], j['id']] + y[j['id'], i['id']] <= x[i['id'], k] + x[j['id'], k] - 1, f'order_{i["id"]}_{j["id"]}_{k}')
                    model.addConstr(y[i['id'], j['id']] + y[j['id'], i['id']] >= x[i['id'], k] + x[j['id'], k] - 2, f'order_lower_{i["id"]}_{j["id"]}_{k}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        schedule = {job['id']: S[job['id']].x for job in jobs}
        solution = {'objective_value': objective_value, 'schedule': schedule}
        return solution
    else:
        raise Exception(f"Model status: {model.status}")

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

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