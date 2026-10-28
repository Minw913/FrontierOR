import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, log_path=None):
    num_tasks = instance['num_tasks']
    num_stations = instance['num_stations']
    processing_times = instance['processing_times']
    precedence_relations = instance['precedence_relations']
    forward_setup_times = instance['forward_setup_times']
    backward_setup_times = instance['backward_setup_times']

    model = Model('assembly_line')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False  

    task_assignments = model.addVars(num_tasks, num_stations, vtype=GRB.BINARY, name='task_assignments')
    task_order = model.addVars(num_tasks, num_tasks, num_stations, vtype=GRB.BINARY, name='task_order')
    cycle_time = model.addVar(name='cycle_time')

    model.setObjective(cycle_time, GRB.MINIMIZE)

    for i in range(num_tasks):
        model.addConstr(quicksum(task_assignments[i, k] for k in range(num_stations)) == 1, f'task_assignment_{i}')

    for i, j in precedence_relations:
        i, j = i - 1, j - 1  
        model.addConstr(quicksum(k * task_assignments[j, k] for k in range(num_stations)) - quicksum(k * task_assignments[i, k] for k in range(num_stations)) >= 0, f'precedence_station_{i}_{j}')
        for k in range(num_stations):
            model.addConstr(task_assignments[i, k] + task_assignments[j, k] - task_order[i, j, k] <= 1, f'precedence_order_{i}_{j}_{k}')

    for k in range(num_stations):
        station_tasks = [i for i in range(num_tasks)]
        model.addConstr(cycle_time >= quicksum(processing_times[i] * task_assignments[i, k] for i in range(num_tasks)) +
                        quicksum(forward_setup_times[i][j] * task_order[i, j, k] for i in range(num_tasks) for j in range(num_tasks)) +
                        quicksum(backward_setup_times[i][j] * task_assignments[i, k] * task_assignments[j, k] for i in range(num_tasks) for j in range(num_tasks)), f'cycle_time_station_{k}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        assignment = {}
        station_sequences = {}
        for k in range(num_stations):
            station_tasks = sorted([i for i in range(num_tasks) if task_assignments[i, k].X > 0.5], key=lambda i: sum([task_order[i, j, k].X for j in range(num_tasks)]))
            if station_tasks:
                station_sequences[k+1] = [i+1 for i in station_tasks]
                for i in station_tasks:
                    assignment[i+1] = k+1

        solution = {
            'objective_value': model.objVal,
            'assignment': assignment,
            'station_sequences': station_sequences,
            'cycle_time': model.objVal
        }
        return solution
    else:
        raise Exception(f"Model status: {model.status}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_instance(instance, args.time_limit, args.log_path)
    if logger:
        logger.log(solution['objective_value'])

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()