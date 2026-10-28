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
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  

    start_times = model.addVars(num_activities, vtype=GRB.INTEGER, name='start_time')
    model.setObjective(start_times[instance['num_real_activities'] + 1], GRB.MINIMIZE)

    model.addConstr(start_times[0] == 0, name='project_start')

    for constraint in temporal_constraints:
        from_activity = constraint['from_activity']
        to_activity = constraint['to_activity']
        time_lag = constraint['time_lag']
        model.addConstr(start_times[to_activity] - start_times[from_activity] >= time_lag, 
                        name=f'temporal_{from_activity}_{to_activity}')

    for resource in resources:
        resource_id = resource['resource_id']
        capacity = resource['capacity']
        demands = resource['demands']
        Pi_k = resource['Pi_k']
        
        total_consumption = gp.LinExpr()
        for activity in activities:
            activity_id = activity['id']
            processing_time = activity['processing_time']
            demand = demands.get(str(activity_id), 0)
            if demand > 0:
                execution_indicator = model.addVars([t for t in range(max_project_duration+1)], vtype=GRB.BINARY, name=f'activity_{activity_id}_exec_indicator')
                for t in range(max_project_duration+1):
                    model.addConstr(execution_indicator[t] >= 1 - (start_times[activity_id] + processing_time - 1 - t)/max_project_duration, name=f'exec_indicator_ub_{activity_id}_{t}')
                    model.addConstr(execution_indicator[t] >= (t - start_times[activity_id])/max_project_duration, name=f'exec_indicator_lb_{activity_id}_{t}')
                    model.addConstr(execution_indicator[t] <= 1 + (start_times[activity_id] - t)/max_project_duration, name=f'exec_indicator_ub2_{activity_id}_{t}')
                    model.addConstr(execution_indicator[t] <= 1 - (start_times[activity_id] + processing_time - 1 - t)/max_project_duration, name=f'exec_indicator_lb2_{activity_id}_{t}')
                total_consumption += demand * gp.quicksum(execution_indicator[t] for t in Pi_k if t <= max_project_duration)
        
        model.addConstr(total_consumption <= capacity, name=f'resource_{resource_id}')

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