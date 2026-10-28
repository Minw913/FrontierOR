import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance, time_limit, logger):
    n = instance['n']
    m = instance['m']
    num_scenarios = instance['num_scenarios']
    facility_capacity_limits = instance['facility_capacity_limits']
    capacity_requirements = instance['capacity_requirements']
    release_times = instance['release_times']
    scenarios = instance['scenarios']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 1  
    model.Params.LogToConsole = 0

    assignment = model.addVars(n, m, vtype=GRB.BINARY, name='assignment')
    model.addConstrs((assignment.sum(j, '*') == 1 for j in range(n)), name='assign_job')

    makespan_vars = []
    for s, scenario in enumerate(scenarios):
        start = model.addVars(n, vtype=GRB.INTEGER, name=f'start_{s}')
        makespan = model.addVar(vtype=GRB.INTEGER, name=f'makespan_{s}')
        makespan_vars.append(makespan)

        model.addConstrs((start[j] >= release_times[j] for j in range(n)), name=f'release_time_{s}')

        processing_times = scenario['processing_times']
        M = max(release_times) + max(max(row[j] for row in processing_times) for j in range(n))
        for j in range(n):
            for f in range(m):
                model.addConstr(start[j] + processing_times[f][j] <= makespan + (1 - assignment[j, f]) * M, name=f'makespan_def_{s}_{j}_{f}')

        # Add resource capacity constraints
        for f in range(m):
            for t in range(max(release_times) + max(max(row) for row in processing_times)):
                jobs = [j for j in range(n) if capacity_requirements[f][j] > 0]
                model.addConstr(gp.quicksum(assignment[j, f] * capacity_requirements[f][j] for j in jobs if (start[j].x >= t - processing_times[f][j] and start[j].x <= t)) <= facility_capacity_limits[f], name=f'capacity_{s}_{f}_{t}')

    expected_makespan = gp.quicksum(scenario['probability'] * makespan_vars[s] for s, scenario in enumerate(scenarios))
    model.setObjective(expected_makespan, GRB.MINIMIZE)

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        if logger:
            logger.log(objective_value)

        solution = {
            'objective_value': objective_value,
            'assignment': {str(j): [f for f in range(m) if assignment[j, f].x >= 0.5][0] for j in range(n)},
            'schedule': {}
        }

        for s, scenario in enumerate(scenarios):
            scenario_solution = {}
            processing_times = scenario['processing_times']
            start_vars = [model.getVarByName(f'start_{s}[{j}]').x for j in range(n)]
            for j in range(n):
                f = [f for f in range(m) if assignment[j, f].x >= 0.5][0]
                start_time = start_vars[j]
                finish_time = start_time + processing_times[f][j]
                scenario_solution[str(j)] = {
                    'facility': f,
                    'start': int(start_time),
                    'finish': int(finish_time)
                }
            solution['schedule'][str(s)] = scenario_solution

        return solution
    else:
        return {
            'objective_value': float('inf'),
            'assignment': {str(j): 0 for j in range(n)},
            'schedule': {}
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()