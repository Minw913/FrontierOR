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

    # Create a Gurobi model
    model = gp.Model("ResourceProvisioning")
    model.Params.LogToConsole = 0  # Suppress Gurobi console output

    # Decision variables: Start times for activities
    start_times = model.addVars(num_total_activities, vtype=GRB.INTEGER, name="start_times")

    # Decision variables: Resource provision levels
    resource_amounts = model.addVars(num_resource_types, vtype=GRB.INTEGER, name="resource_amounts", lb=0)

    # Objective: Minimize total resource provision cost
    model.setObjective(gp.quicksum(resource_costs[k] * resource_amounts[k] for k in range(num_resource_types)), GRB.MINIMIZE)

    # Constraint: Project start activity starts at time 0
    model.addConstr(start_times[0] == 0, "project_start")

    # Constraint: Project end activity starts within the deadline
    model.addConstr(start_times[num_total_activities - 1] <= max_project_duration - activities[num_total_activities - 1]['duration'], "project_deadline")

    # Constraints: Precedence relations
    for pred, succ in precedence_relations:
        model.addConstr(start_times[succ] >= start_times[pred] + activities[pred]['duration'], f"precedence_{pred}_{succ}")

    # Constraints: Resource availability
    for t in range(max_project_duration):
        resource_usage = gp.quicksum(
            activities[i]['resource_requirements'][k] 
            for i in range(num_total_activities) 
            for k in range(num_resource_types) 
            if activities[i]['duration'] > 0  # Only consider real activities
            and start_times[i] <= t 
            and start_times[i] + activities[i]['duration'] > t
        )
        for k in range(num_resource_types):
            model.addConstr(
                gp.quicksum(
                    activities[i]['resource_requirements'][k] 
                    for i in range(num_total_activities) 
                    if activities[i]['duration'] > 0  # Only consider real activities
                    and start_times[i] <= t 
                    and start_times[i] + activities[i]['duration'] > t
                ) <= resource_amounts[k], 
                f"resource_{k}_at_{t}"
            )

    # Set time limit
    model.Params.TimeLimit = time_limit

    # Optimize the model
    model.optimize()

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.getObjective().getValue()
        start_times_solution = {i: start_times[i].X for i in range(num_total_activities)}
        resource_amounts_solution = {k: resource_amounts[k].X for k in range(num_resource_types)}

        # Log intermediate solution if logger is provided
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