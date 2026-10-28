import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    num_tasks = instance['num_tasks']
    cycle_time = instance['cycle_time']
    num_stations = instance['num_stations']
    task_processing_times = instance['task_processing_times']
    precedence_arcs = instance['precedence_arcs']
    earliest_station = instance['earliest_station']
    latest_station = instance['latest_station']

    model = Model('assembly_line_balancing')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Decision variables
    task_assignments = model.addVars(num_tasks, num_stations, vtype=GRB.BINARY, name='task_assignments')
    worst_case_idle_time = model.addVar(vtype=GRB.CONTINUOUS, name='worst_case_idle_time', lb=0)

    # Objective: minimize worst_case_idle_time
    model.setObjective(worst_case_idle_time, GRB.MINIMIZE)

    # Constraints
    # 1. Each task is assigned to exactly one station
    model.addConstrs((quicksum(task_assignments[t, s] for s in range(num_stations)) == 1 for t in range(num_tasks)), name='assign_once')

    # 2. Respect precedence relationships
    for pred, succ in precedence_arcs:
        model.addConstrs((quicksum(s * task_assignments[pred, s] for s in range(num_stations)) <= quicksum(s * task_assignments[succ, s] for s in range(num_stations)) for s in range(num_stations)), name='precedence')

    # 3. Total processing time at each station does not exceed cycle time
    model.addConstrs((quicksum(task_processing_times[str(t)] * task_assignments[t, s] for t in range(num_tasks)) <= cycle_time - worst_case_idle_time for s in range(num_stations)), name='cycle_time')

    # 4. Tasks are assigned within their feasible station range
    for t in range(num_tasks):
        for s in range(num_stations):
            if s < int(earliest_station[str(t)]) - 1 or s > int(latest_station[str(t)]) - 1:
                model.addConstr(task_assignments[t, s] == 0, name=f'feasible_range_{t}_{s}')

    # Lower bound on worst_case_idle_time based on total slack
    total_slack = num_stations * cycle_time - instance['total_processing_time']
    model.addConstr(worst_case_idle_time >= total_slack / cycle_time, name='worst_case_idle_time_lb')

    # Optimize
    model.optimize()

    # Log and return solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        task_assignments_sol = {t: s + 1 for t, s in enumerate([s for s in range(num_stations) if task_assignments[t, s].X > 0.5])}
        objective_value = worst_case_idle_time.X
        if logger:
            logger.log(objective_value)
        return objective_value, task_assignments_sol
    else:
        raise Exception(f"Gurobi model status: {model.status}")

def write_solution(solution_path, objective_value, task_assignments):
    solution = {
        'objective_value': objective_value,
        'task_assignments': task_assignments
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    objective_value, task_assignments = solve_instance(instance, args.time_limit, logger)
    write_solution(args.solution_path, objective_value, task_assignments)

if __name__ == '__main__':
    main()