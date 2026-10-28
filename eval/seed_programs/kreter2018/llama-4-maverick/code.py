import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, logger):
    num_real_activities = instance['num_real_activities']
    num_total_activities = instance['num_total_activities']
    num_resource_types = instance['num_resource_types']
    max_project_duration = instance['max_project_duration']
    activities = instance['activities']
    precedence_relations = instance['precedence_relations']
    resource_costs = instance['resource_costs']

    model = gp.Model("ResourceProvisioning")
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  

    start_times = model.addVars(num_total_activities, vtype=GRB.INTEGER, name="start_times")
    resource_amounts = model.addVars(num_resource_types, vtype=GRB.INTEGER, name="resource_amounts", lb=0)

    model.setObjective(gp.quicksum(resource_costs[k] * resource_amounts[k] for k in range(num_resource_types)), GRB.MINIMIZE)

    model.addConstr(start_times[0] == 0, "project_start")
    model.addConstr(start_times[num_total_activities - 1] <= max_project_duration - activities[num_total_activities - 1]['duration'], "project_deadline")

    for pred, succ in precedence_relations:
        model.addConstr(start_times[succ] >= start_times[pred] + activities[pred]['duration'], f"precedence_{pred}_{succ}")

    activity_in_progress = model.addVars(num_total_activities, max_project_duration, vtype=GRB.BINARY, name="activity_in_progress")

    for i in range(num_total_activities):
        for t in range(max_project_duration):
            if t >= start_times[i].LB and t < start_times[i].UB + activities[i]['duration']:
                model.addConstr(activity_in_progress[i, t] >= 1 - (t < start_times[i]) - (t >= start_times[i] + activities[i]['duration']), f"activity_in_progress_{i}_{t}_lower")
                model.addConstr(activity_in_progress[i, t] <= 1 - (t < start_times[i]), f"activity_in_progress_{i}_{t}_upper1")
                model.addConstr(activity_in_progress[i, t] <= 1 - (t >= start_times[i] + activities[i]['duration']), f"activity_in_progress_{i}_{t}_upper2")

    for k in range(num_resource_types):
        for t in range(max_project_duration):
            model.addConstr(gp.quicksum(activities[i]['resource_requirements'][k] * activity_in_progress[i, t] for i in range(num_total_activities) if activities[i]['duration'] > 0) <= resource_amounts[k], f"resource_{k}_at_{t}")

    model.Params.TimeLimit = time_limit

    model.optimize()

    if model.SolCount > 0:
        objective_value = model.getObjective().getValue()
        start_times_solution = {i: start_times[i].X for i in range(num_total_activities)}
        resource_amounts_solution = {k: resource_amounts[k].X for k in range(num_resource_types)}

        if logger:
            logger.log(objective_value)

        return {
            "objective_value": objective_value,
            "start_times": {f"{i}": int(start_times_solution[i]) for i in range(num_total_activities)},
            "resource_amounts": {f"{k}": int(resource_amounts_solution[k]) for k in range(num_resource_types)},
        }
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Solve Resource Provisioning Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve_instance(instance, args.time_limit, logger)

    if solution is not None:
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()