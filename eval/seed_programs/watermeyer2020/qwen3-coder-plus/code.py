import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_resource_constrained_project_scheduling(instance_data, time_limit):
    n = instance_data["n"]
    num_activities = instance_data["num_activities"]
    num_resources = instance_data["num_resources"]
    d_bar = instance_data["d_bar"]
    
    activities = instance_data["activities"]
    temporal_constraints = instance_data["temporal_constraints"]
    resources = instance_data["resources"]

    # Calculate total duration for big-M
    max_duration = sum(activities[i]['processing_time'] for i in range(num_activities))
    T = d_bar  # Maximum project duration

    # Create model
    model = gp.Model("ResourceConstrainedProjectScheduling")
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)  # Single core constraint

    # Decision variables: start time for each activity
    start_times = {}
    for i in range(num_activities):
        start_times[i] = model.addVar(vtype=GRB.INTEGER, name=f"S_{i}", lb=0, ub=T)

    # Define end time variables
    end_times = {}
    for i in range(num_activities):
        processing_time = activities[i]["processing_time"]
        if processing_time > 0:
            end_times[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"E_{i}")
            model.addConstr(end_times[i] == start_times[i] + processing_time)
        else:
            # If processing time is 0, we don't need an extra variable
            pass

    # Objective: minimize start time of project-end activity (activity num_activities - 1)
    project_end_activity = num_activities - 1
    model.setObjective(start_times[project_end_activity], GRB.MINIMIZE)

    # Add temporal constraints
    for tc in temporal_constraints:
        i = tc['i']
        j = tc['j']
        delta = tc['delta']
        model.addConstr(start_times[j] - start_times[i] >= delta, f"temporal_{i}_{j}")

    # Resource constraints for partially renewable resources
    for res_id in range(num_resources):
        res_info = resources[res_id]
        capacity = res_info['capacity']
        periods = set(res_info['periods'])  # Convert to set for fast lookups
        
        # Check capacity across all relevant activities
        total_resource_usage = gp.LinExpr()
        
        for act_id in range(num_activities):
            if act_id not in activities or res_id not in activities[act_id]["resource_demands"]:
                continue  # Skip if resource demand not specified
            demand = activities[act_id]["resource_demands"][res_id]
            if demand == 0:
                continue  # Skip if no demand
            
            processing_time = activities[act_id]["processing_time"]
            
            # Calculate resource consumption based on which periods activity overlaps with resource availability
            if processing_time == 0:
                continue  # Skip activities with 0 processing time
                
            # Compute which periods the activity would occupy based on its start time and duration
            # This requires binary expansion to capture overlap correctly
            # For each activity, we'll compute number of periods in its execution that also exist in resource periods
            
            # Binary variables to represent if activity i is active in period p
            # We'll add variable that represents exact period overlap
            resource_consumption_for_act = gp.LinExpr()
            
            # Count how many periods in the resource availability schedule the activity overlaps with
            for start_period in range(T + 1):  # Try every potential start moment
                # Binary variable indicating when act is scheduled at certain time  
                if_var = model.addVar(vtype=GRB.BINARY, name=f"use_if_start_{act_id}_{start_period}")
                
                # Indicator constraints to determine resource consumption
                model.addGenConstrIndicator(if_var, 1, start_times[act_id] == start_period)

                # Now calculate resource consumption for this start time
                exec_periods = []  # Range of periods the activity executes in when scheduled at start_period
                for t in range(processing_time):
                    exec_period = start_period + t + 1  # Activity executes from strict after start to including finish
                    if exec_period <= d_bar and exec_period in periods:
                        exec_periods.append(exec_period)
                
                # Constraint: the total consumption contribution for this start time assignment
                contrib = len(exec_periods) * demand
                total_resource_usage += contrib * if_var

        # Final resource capacity constraint
        model.addConstr(total_resource_usage <= capacity, f"resource_capacity_{res_id}")

    # Ensure project start (activity 0) begins at 0
    model.addConstr(start_times[0] == 0, name="project_start")

    # Optimize
    model.optimize()

    # Extract and return the solution
    if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
        raise Exception(f"No solution found, status = {model.status}")

    solution = {}
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        solution["start_times"] = {}
        solution["objective_value"] = None
    else:
        solution["start_times"] = {str(i): int(start_times[i].x) for i in range(num_activities)}
        solution["objective_value"] = int(model.objVal)

    return solution


def calculate_resource_usage(activity_id, start_time, processing_time, resource_periods, resource_demand):
    """
    Calculates resource usage for an activity with specific start time over resource availability periods.
    :param activity_id: ID of the activity
    :param start_time: When the activity starts
    :param processing_time: How long it takes
    :param resource_periods: List of resource available periods
    :param resource_demand: Amount the activity demands per period it's active
    :return: Tuple of (total resource usage, list of periods when active during resource availability)
    """
    resource_set = set(resource_periods)
    total_use = 0
    
    # The activity runs in the exclusive-inclusive interval (start_time, start_time + processing_time]
    exec_periods = set()  # Periods when activity is active
    
    for period_offset in range(1, processing_time + 1):
        exec_period = start_time + period_offset
        if exec_period in resource_set:
            total_use += resource_demand
            exec_periods.add(exec_period)
    
    return total_use, exec_periods


def create_integer_model(instance_data, time_limit):
    n = instance_data["n"]
    num_activities = instance_data["num_activities"]
    num_resources = instance_data["num_resources"]
    d_bar = instance_data["d_bar"]

    activities = instance_data["activities"]
    temporal_constraints = instance_data["temporal_constraints"]
    resources = instance_data["resources"]

    # Start model
    model = gp.Model("ProjectSchedulingExact")
    model.setParam(GRB.Param.OutputFlag, 0)  # Suppress console log
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.Threads, 1)

    # Decision variables: start time for each activity [0, d_bar]
    start_times = model.addVars(num_activities, vtype=GRB.INTEGER, name="start", lb=0, ub=d_bar)
    
    # Objective - minimize start of last activity
    model.setObjective(start_times[num_activities-1], GRB.MINIMIZE)

    # Start time of project must be 0
    model.addConstr(start_times[0] == 0, name="project_start_zero")

    # Temporal constraints
    for c in temporal_constraints:
        i = c["i"]
        j = c["j"]
        delta = c["delta"]
        model.addConstr(start_times[j] - start_times[i] >= delta, name=f"temp_{i}_{j}_{delta}")

    # Resource constraints
    for r_id in range(num_resources):
        capacity = resources[r_id]["capacity"]
        resource_periods = set(resources[r_id]["periods"])
        
        # Sum over all activities using this resource
        resource_usage_expr = gp.LinExpr()
        
        for act_id, data in enumerate(activities.values()):
            if r_id not in data.get("resource_demands", {}):
                continue
                
            demand = data["resource_demands"][r_id]
            if demand == 0:
                continue
            
            processing_time = data["processing_time"]
            if processing_time == 0:
                continue
            
            # Calculate the number of resource periods this activity will consume
            # when it starts at different possible start times
            # To model this exactly, we use a piecewise approach with binary indicators
            total_resource_usage_across_time = 0
            usage_contributions = gp.LinExpr()

            for s in range(d_bar + 1):
                # Binary variable for if activity starts at time s
                start_at_s = model.addVar(vtype=GRB.BINARY, name=f"start_{act_id}_at_{s}")

                # Link binary var to actual start time value
                model.addGenConstrIndicator(start_at_s, 1, start_times[act_id] == s)

                # Calculate resource consumption for this potential start
                consumption_count = 0
                for t_offset in range(processing_time):
                    t = s + 1 + t_offset  # Activity is active at exact time interval
                    if t <= d_bar and t in resource_periods:
                        consumption_count += 1
                
                if consumption_count > 0:
                    usage_contributions.addTerms(consumption_count * demand, start_at_s)
            
            resource_usage_expr.add(usage_contributions)

        model.addConstr(resource_usage_expr <= capacity, name=f"resource_{r_id}")

    return model, start_times


def improved_solve(instance_json, time_limit):
    """
    An improved solver using a more direct approach
    """
    n = instance_json["n"]
    num_activities = instance_json["num_activities"]
    num_resources = instance_json["num_resources"]
    d_bar = instance_json["d_bar"]

    activities = instance_json["activities"]
    temporal_constraints = instance_json["temporal_constraints"]
    resources = instance_json["resources"]

    # Create model
    model = gp.Model("PRCPSP")
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.Threads, 1)

    # Variables - start time for each activity [0, dbar]
    S = model.addVars(num_activities, lb=0, ub=d_bar, vtype=GRB.INTEGER, name="start")

    # Objectve
    model.setObjective(S[num_activities-1], GRB.MINIMIZE)

    # Constraints
    model.addConstr(S[0] == 0, name="start_at_zero")
    
    # Temporal constraints
    for tc in temporal_constraints:
        i, j, delta = tc['i'], tc['j'], tc['delta']
        model.addConstr(S[j] - S[i] >= delta, f"temp_{i}_{j}")

    # Partially Renewable Resource constraints
    # For each time slice and each resource we need to count resource allocation
    for r_idx in range(num_resources):
        capacity = resources[str(r_idx)]["capacity"]
        per = set(resources[str(r_idx)]["periods"])  # Available periods for res r
        
        # Total used amount over all activities
        # The resource usage depends on which periods the activity executes
        usage_sum = 0
        
        # For each activity, its resource usage is determined by overlap between
        # execution periods and available periods of this resource
        for a_idx, act_props in activities.items():
            a_idx = int(a_idx)
            processing_time = act_props["processing_time"]
            if str(r_idx) not in act_props["resource_demands"]:
                continue  # This activity doesn'd require this resource
                
            per_period_demand = act_props["resource_demands"][str(r_idx)]
            if per_period_demand == 0:
                continue
                
            # For each possible start time, we calculate potential usage
            possible_usages = []
            
            for s in range(d_bar + 1):
                usage_at_s = 0
                # Activity occupies periods (s+1) to (s+processing_time) inclusive
                for t in range(s + 1, min(s + 1 + processing_time, d_bar + 1)):
                    if t in per:
                        usage_at_s += per_period_demand
                        
                # This usage happens only if the activity really starts at time s
                # We handle this with indicator constraint later as above solution failed on large instances
                binary_use = model.addVar(vtype=GRB.BINARY, name=f"x_{a_idx}_{s}")
                model.addGenConstrIndicator(binary_use, 1, S[a_idx] == s)
                
                possible_usages.append(usage_at_s * binary_use)
                
            usage_sum += gp.quicksum(possible_usages)
            
        model.addConstr(usage_sum <= capacity, f"resc_cap_{r_idx}")

    # Solve
    model.optimize()
    
    # Format result
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            solution = {
                "start_times": {str(i): int(S[i].X) for i in range(num_activities)},
                "objective_value": int(model.ObjVal)
            }
            return solution
        else:
            # No feasible solution found even though not infeasible state
            return {
                "start_times": {},
                "objective_value": None
            }
    else:
        return {
            "start_times": {},
            "objective_value": None
        }


def main():
    parser = argparse.ArgumentParser(description='Solve Resource-Constrained Project Scheduling Problem.')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the problem instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum time in seconds to run the solver')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to the log JSONL file')
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Solve using improved version to avoid complex formulations
    solution = improved_solve(instance_data, args.time_limit)
    
    if solution['objective_value'] is not None and logger:
        logger.log(solution['objective_value'])

    # Save solutions
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()