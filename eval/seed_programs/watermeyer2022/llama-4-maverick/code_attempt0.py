import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    num_activities = instance['num_activities_total']
    activities = instance['activities']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']
    max_project_duration = instance['max_project_duration']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable console logging

    # Decision variables: start times for activities
    start_times = model.addVars(num_activities, vtype=GRB.INTEGER, name='start_time')

    # Objective: minimize the start time of the project-end activity
    model.setObjective(start_times[instance['num_real_activities'] + 1], GRB.MINIMIZE)

    # Constraint: project-start activity starts at time 0
    model.addConstr(start_times[0] == 0, name='project_start')

    # Constraints: temporal relationships between activities
    for constraint in temporal_constraints:
        from_activity = constraint['from_activity']
        to_activity = constraint['to_activity']
        time_lag = constraint['time_lag']
        model.addConstr(start_times[to_activity] - start_times[from_activity] >= time_lag, 
                        name=f'temporal_{from_activity}_{to_activity}')

    # Constraints: partially renewable resource capacities
    for resource in resources:
        resource_id = resource['resource_id']
        capacity = resource['capacity']
        demands = resource['demands']
        Pi_k = resource['Pi_k']
        consumption = gp.LinExpr()
        for activity in activities:
            activity_id = activity['id']
            processing_time = activity['processing_time']
            demand = demands.get(str(activity_id), 0)
            # Calculate the number of periods in Pi_k during which the activity is in execution
            execution_periods = gp.LinExpr()
            for t in Pi_k:
                execution_periods += gp.quicksum([1 for s in range(max(0, t-processing_time+1), min(t+1, start_times[activity_id]+processing_time)) if s == start_times[activity_id]])
            consumption += demand * execution_periods
        model.addConstr(consumption <= capacity, name=f'resource_{resource_id}')

    # Correct implementation of resource constraints
    for resource in resources:
        resource_id = resource['resource_id']
        capacity = resource['capacity']
        demands = resource['demands']
        Pi_k = resource['Pi_k']
        resource_usage = model.addVars(Pi_k, vtype=GRB.INTEGER, name=f'resource_usage_{resource_id}')
        model.addConstrs((resource_usage[t] == 0 for t in range(max_project_duration+1) if t not in Pi_k), name=f'resource_usage_zero_{resource_id}')
        for activity in activities:
            activity_id = activity['id']
            processing_time = activity['processing_time']
            demand = demands.get(str(activity_id), 0)
            if demand > 0:
                for t in Pi_k:
                    model.addConstr(resource_usage[t] >= gp.quicksum([demand for s in range(max(0, t-processing_time+1), min(t+1, start_times[activity_id]+1)) if s == start_times[activity_id]]), name=f'resource_usage_{resource_id}_{activity_id}_{t}')
        model.addConstr(gp.quicksum(resource_usage[t] for t in Pi_k) <= capacity, name=f'resource_capacity_{resource_id}')

    # Optimize the model
    model.optimize()
    if model.SolCount > 0:
        schedule = {activity: start_times[activity].X for activity in range(num_activities)}
        objective_value = schedule[instance['num_real_activities'] + 1]
        if logger:
            logger.log(objective_value)
        return objective_value, schedule
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    instance = read_instance(args.instance_path)
    objective_value, schedule = solve_instance(instance, args.time_limit, logger)

    if objective_value is not None and schedule is not None:
        solution = {
            'objective_value': objective_value,
            'schedule': {str(activity): int(start_time) for activity, start_time in schedule.items()}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    main()