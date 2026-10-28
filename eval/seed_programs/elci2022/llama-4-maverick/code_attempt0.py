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

    # Create a Gurobi model
    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 1  # Ensure single-threaded execution

    # First-stage decision: Assign jobs to facilities
    assignment = model.addVars(n, m, vtype=GRB.BINARY, name='assignment')

    # Constraint: Each job is assigned to exactly one facility
    model.addConstrs((assignment.sum(j, '*') == 1 for j in range(n)), name='assign_job')

    # Second-stage decision: Schedule jobs on facilities for each scenario
    for s, scenario in enumerate(scenarios):
        start = model.addVars(n, vtype=GRB.INTEGER, name=f'start_{s}')
        makespan = model.addVar(vtype=GRB.INTEGER, name=f'makespan_{s}')

        # Constraint: Job start time is no earlier than its release time
        model.addConstrs((start[j] >= release_times[j] for j in range(n)), name=f'release_time_{s}')

        # Constraint: Makespan is the maximum completion time among all jobs
        processing_times = scenario['processing_times']
        model.addConstrs((start[j] + processing_times[f][j] <= makespan for j in range(n) for f in range(m)), name=f'makespan_def_{s}')
        model.addConstrs((start[j] + processing_times[f][j] <= makespan for j in range(n) for f in range(m) if assignment[j, f] == 1), name=f'makespan_def_alt_{s}')

        # Constraint: Resource capacity constraint for each facility at every point in time
        for f in range(m):
            jobs_on_facility = [j for j in range(n) if assignment[j, f].x >= 0.5]  # Use x attribute after optimization
            for t in range(max(release_times[j] for j in jobs_on_facility) + max(processing_times[f][j] for j in jobs_on_facility) + 1):
                overlapping_jobs = [j for j in jobs_on_facility if start[j].x <= t and start[j].x + processing_times[f][j] > t]
                model.addConstr(gp.quicksum(capacity_requirements[f][j] * assignment[j, f] for j in overlapping_jobs) <= facility_capacity_limits[f], name=f'capacity_{s}_{f}_{t}')

    # Objective: Minimize the expected makespan across all scenarios
    expected_makespan = gp.quicksum(scenario['probability'] * model.getVarByName(f'makespan_{s}') for s, scenario in enumerate(scenarios))
    model.setObjective(expected_makespan, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        if logger:
            logger.log(objective_value)

        # Extract and format the solution
        solution = {
            'objective_value': objective_value,
            'assignment': {j: f for j in range(n) for f in range(m) if assignment[j, f].x >= 0.5},
            'schedule': {}
        }

        for s, scenario in enumerate(scenarios):
            scenario_solution = {}
            processing_times = scenario['processing_times']
            for j in range(n):
                f = solution['assignment'][j]
                start_time = model.getVarByName(f'start_{s}').x
                finish_time = start_time + processing_times[f][j]
                scenario_solution[j] = {
                    'facility': f,
                    'start': start_time,
                    'finish': finish_time
                }
            solution['schedule'][s] = scenario_solution

        return solution
    else:
        raise Exception(f"Model optimization failed with status {model.status}")

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