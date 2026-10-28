import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve_instance(instance):
    n = instance['n']
    num_activities = instance['num_activities']
    num_resources = instance['num_resources']
    d_bar = instance['d_bar']
    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    model = Model()
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0

    # Activity start times
    start_times = model.addVars(num_activities, vtype=GRB.INTEGER, name='start_times')

    # Objective: minimize project duration
    model.setObjective(start_times[num_activities - 1], GRB.MINIMIZE)

    # Constraints
    for constraint in temporal_constraints:
        i, j, delta = constraint['i'], constraint['j'], constraint['delta']
        model.addConstr(start_times[j] - start_times[i] >= delta, f'temporal_constraint_{i}_{j}')

    for resource_id, resource in resources.items():
        capacity = resource['capacity']
        periods = set(resource['periods'])  
        consumption = []
        for i in range(num_activities):
            if str(i) in activities:
                activity = activities[str(i)]
                processing_time = activity.get('processing_time', 0)
                demand = activity.get('resource_demands', {}).get(resource_id, 0)
                overlap = sum(1 for t in periods if start_times[i] < t <= start_times[i] + processing_time)
                consumption.append(demand * overlap)
            else:
                consumption.append(0)
        model.addConstr(quicksum(consumption) <= capacity, f'resource_constraint_{resource_id}')

    # Project start activity starts at time 0
    model.addConstr(start_times[0] == 0, 'project_start')

    # Non-negativity constraints for start times
    for i in range(1, num_activities):
        model.addConstr(start_times[i] >= 0, f'non_negative_start_{i}')

    return model, start_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, start_times = solve_instance(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.optimize()

    if model.SolCount > 0:
        solution = {
            'objective_value': int(model.ObjVal),
            'start_times': {i: int(start_times[i].X) for i in range(instance['num_activities'])}
        }
        if logger:
            logger.log(solution['objective_value'])
    else:
        solution = {
            'objective_value': None,
            'start_times': None
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()